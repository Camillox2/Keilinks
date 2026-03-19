"""
Servidor da Keilinks v8 — Consciencia + Memoria por Usuario
Camadas: Normalizar -> Reflexao -> Busca Semantica -> Modelo Neural -> Web -> Fallback
Melhorias v8:
  1. Historico de conversa no prompt
  2. Personalidade injetada no prompt
  3. Memoria ativa no prompt
  4. Consciencia temporal
  5. Memoria por usuario
  6. Auto-avaliacao pos-resposta (3 tentativas)
"""

import torch
import sys
import os
import json
import threading
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from modelo.keilinks import Keilinks, MODELOS
from dados.tokenizador import Tokenizador
from dados.retrieval import Retrieval
from dados.knowledge import Knowledge
from dados.database import (
    inicializar_banco, migrar_json_para_mysql,
    conversa_salvar, conversa_historico,
    knowledge_total, knowledge_por_fonte,
    crawler_log_recentes, memoria_get, memoria_set, memoria_todos,
    usuario_criar, usuario_login, usuario_por_token,
    chat_criar, chat_listar, chat_mensagens, chat_deletar, chat_atualizar_titulo,
)
from busca.web import pesquisar, precisa_buscar
from cerebro.crawler import CrawlerBackground
from cerebro.memoria import Memoria
from cerebro.reflexao import Reflexao
from cerebro.normalizador import normalizar
from cerebro.consciencia import Consciencia

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
modelos_carregados = {}
tokenizador = None

# ─── Sistemas do Cerebro ──────────────────────────────────────────────────
retrieval    = Retrieval()
knowledge    = Knowledge()
memoria      = Memoria(os.path.join(BASE_DIR, 'dados', 'memoria.json'))
reflexao     = Reflexao()
consciencia  = Consciencia(os.path.join(BASE_DIR, 'dados'))
crawler      = CrawlerBackground(intervalo_minutos=5)

INTERFACE_DIR   = os.path.join(BASE_DIR, 'interface')
DADOS_DIR       = os.path.join(BASE_DIR, 'dados')
CHECKPOINTS     = os.path.join(BASE_DIR, 'checkpoints')
APRENDIZADO     = os.path.join(DADOS_DIR, 'aprendizado.txt')

contador_conversas = 0
RETREINAR_A_CADA = 50

# ─── Personalidade resumida (injetada no prompt) ────────────────────────
PERSONALIDADE_RESUMO = ""
_sobre_path = os.path.join(DADOS_DIR, 'sobre_keilinks.txt')
if os.path.exists(_sobre_path):
    with open(_sobre_path, 'r', encoding='utf-8') as _f:
        _linhas = _f.readlines()
    _partes = []
    for _l in _linhas:
        _l = _l.strip()
        if _l.startswith('#') or not _l:
            continue
        if any(kw in _l.lower() for kw in ['nome:', 'criador:', 'sou ', 'direta', 'humor', 'informal', 'girias', 'nao uso', 'honesta']):
            _partes.append(_l)
        if len(_partes) >= 8:
            break
    PERSONALIDADE_RESUMO = ' | '.join(_partes)


# ─── Carregamento ─────────────────────────────────────────────────────────

def carregar_modelo(tipo):
    global tokenizador
    caminhos = {
        'flash':  os.path.join(CHECKPOINTS, 'keilinks_flash.pt'),
        'padrao': os.path.join(CHECKPOINTS, 'keilinks_final.pt'),
        'ultra':  os.path.join(CHECKPOINTS, 'keilinks_ultra.pt'),
    }
    caminho = caminhos.get(tipo)
    if not caminho or not os.path.exists(caminho):
        return False
    if tokenizador is None:
        vocab_path = os.path.join(DADOS_DIR, 'vocab.json')
        if not os.path.exists(vocab_path):
            return False
        tokenizador = Tokenizador(vocab_path)
    ckpt = torch.load(caminho, map_location=device, weights_only=False)
    modelo = Keilinks(ckpt['config']).to(device)
    modelo.load_state_dict(ckpt['modelo'])
    modelo.eval()
    modelos_carregados[tipo] = modelo
    print(f"  [{tipo.upper()}] ok")
    return True


def inicializar():
    print(f"\n{'='*50}")
    print(f"  Keilinks v8 — Consciencia + Memoria")
    print(f"  Device: {device.upper()}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")

    inicializar_banco()

    try:
        if knowledge_total() == 0:
            migrar_json_para_mysql(BASE_DIR)
    except Exception as e:
        print(f"[Migracao] {e}")

    for tipo in ['flash', 'padrao', 'ultra']:
        carregar_modelo(tipo)

    retrieval.carregar(
        os.path.join(DADOS_DIR, 'conversas.txt'),
        os.path.join(DADOS_DIR, 'aprendizado.txt'),
    )

    knowledge.iniciar_embeddings()

    disponiveis = list(modelos_carregados.keys())
    total_k = knowledge_total()
    fontes = knowledge_por_fonte()
    print(f"\n  Modelos: {disponiveis if disponiveis else 'nenhum'}")
    print(f"  Knowledge: {total_k} fatos {fontes}")
    print(f"  Retrieval: {len(retrieval.pares)} pares (semantico)")
    if PERSONALIDADE_RESUMO:
        print(f"  Personalidade: {len(PERSONALIDADE_RESUMO)} chars")
    print(f"\n  http://localhost:5000\n")

    # crawler.iniciar()


# ─── Utilidades ───────────────────────────────────────────────────────────

def texto_eh_lixo(texto):
    if not texto or len(texto) < 3:
        return True
    if len(set(texto)) < len(texto) * 0.15:
        return True
    palavras = texto.split()
    if len(palavras) < 2 and len(texto) > 20:
        return True
    curtas = sum(1 for p in palavras if len(p) <= 2)
    if len(palavras) > 3 and curtas / len(palavras) > 0.7:
        return True
    if len(palavras) <= 3 and len(texto) < 25:
        reais = [p for p in palavras if len(p) >= 3]
        if len(reais) < 2:
            return True
    if len(palavras) >= 3:
        unicas = set(p.lower() for p in palavras)
        if len(unicas) < max(len(palavras) * 0.6, 2):
            return True
    if any(t in texto for t in ['<vitor>', '<keilinks>', '<fim>', '<pad>']):
        return True
    return False


def resposta_eh_relevante(pergunta, resposta):
    import re
    def _palavras(txt):
        return set(w for w in re.findall(r'[a-z\u00e0-\u00ff]+', txt.lower()) if len(w) >= 3)

    p_palavras = _palavras(pergunta)
    r_palavras = _palavras(resposta)

    if not p_palavras or not r_palavras:
        return True

    overlap = p_palavras & r_palavras
    if overlap:
        return True

    TEMAS = {
        'python': {'programacao', 'codigo', 'linguagem', 'script', 'biblioteca'},
        'javascript': {'programacao', 'codigo', 'web', 'frontend', 'react', 'node'},
        'keilinks': {'inteligencia', 'modelo', 'neural', 'vitor', 'criou', 'construiu'},
        'vitor': {'criador', 'programador', 'desenvolvedor', 'keilinks', 'fez'},
        'inteligente': {'inteligencia', 'cerebro', 'aprender', 'conhecimento', 'modelo'},
        'triste': {'sentimento', 'emocao', 'ajudar', 'conversar'},
        'musica': {'artista', 'cantor', 'banda', 'album', 'som'},
        'futebol': {'time', 'jogo', 'gol', 'campeonato', 'vasco'},
    }

    for tema, relacionados in TEMAS.items():
        if tema in p_palavras and r_palavras & relacionados:
            return True

    if len(r_palavras) > 15:
        return True

    return False


def _montar_contexto_temporal():
    """Retorna contexto temporal: hora, dia da semana, periodo"""
    agora = datetime.now()
    dias = ['segunda', 'terca', 'quarta', 'quinta', 'sexta', 'sabado', 'domingo']
    dia_semana = dias[agora.weekday()]
    hora = agora.hour
    if 5 <= hora < 12:
        periodo = 'manha'
    elif 12 <= hora < 18:
        periodo = 'tarde'
    elif 18 <= hora < 22:
        periodo = 'noite'
    else:
        periodo = 'madrugada'
    return f"{dia_semana}, {agora.strftime('%H:%M')}, {periodo}"


def gerar_com_modelo(mensagem, tipo, temperatura=0.8, max_tokens=200,
                     historico=None, contexto_memoria='', contexto_semantico=''):
    """Gera resposta com modelo neural + historico + personalidade + memoria + tempo"""
    modelo = None
    tipo_usado = None
    for t in [tipo, 'flash', 'padrao', 'ultra']:
        if t in modelos_carregados:
            modelo = modelos_carregados[t]
            tipo_usado = t
            break
    if not modelo:
        return None

    temp_ajustada = min(temperatura, 0.5) if tipo_usado == 'flash' else min(temperatura, 0.6)

    cfg = modelo.config if hasattr(modelo, 'config') else {}
    ctx_max = cfg.get('contexto_max', 512)

    # Historico de conversa
    prompt_historico = ''
    if historico:
        for h_p, h_r in historico:
            prompt_historico += f'<vitor>{h_p}<fim><keilinks>{h_r}<fim>'

    # Contexto extra (personalidade + memoria + tempo + semantico)
    partes_ctx = []
    if PERSONALIDADE_RESUMO:
        partes_ctx.append(PERSONALIDADE_RESUMO[:200])
    if contexto_memoria:
        partes_ctx.append(contexto_memoria[:150])
    partes_ctx.append(f"agora: {_montar_contexto_temporal()}")
    if contexto_semantico:
        partes_ctx.append(f"info: {contexto_semantico[:150]}")
    ctx_texto = ' | '.join(partes_ctx)

    msg_com_ctx = f"{mensagem} ({ctx_texto})" if ctx_texto else mensagem
    prompt = f'{prompt_historico}<vitor>{msg_com_ctx}<fim><keilinks>'

    tokens_prompt = tokenizador.encode(prompt)
    margem = max_tokens + 20
    if len(tokens_prompt) > ctx_max - margem:
        tokens_prompt = tokens_prompt[-(ctx_max - margem):]

    # Auto-avaliacao: 3 tentativas com temp decrescente
    temps = [temp_ajustada, temp_ajustada * 0.6, 0.2]
    for temp in temps:
        tokens_input = torch.tensor([tokens_prompt], dtype=torch.long).to(device)
        with torch.no_grad():
            saida = modelo.gerar(tokens_input, max_tokens=max_tokens, temperatura=temp)
        texto = tokenizador.decode(saida[0].tolist())
        if '<keilinks>' in texto:
            resp = texto.split('<keilinks>')[-1]
            if '<fim>' in resp:
                resp = resp.split('<fim>')[0]
            resp = resp.strip()
        else:
            resp = texto.strip()

        if resp and not texto_eh_lixo(resp) and resposta_eh_relevante(mensagem, resp):
            if reflexao.validar_resposta(resp):
                return resp

    return None


def salvar_conversa_txt(pergunta, resposta):
    os.makedirs(DADOS_DIR, exist_ok=True)
    with open(APRENDIZADO, 'a', encoding='utf-8') as arq:
        arq.write(f"<vitor>{pergunta}<fim><keilinks>{resposta}<fim>\n")


def retreinar_background(tipo):
    try:
        import subprocess
        subprocess.Popen([sys.executable, 'treino/retreinar.py', '--modelo', tipo], cwd=BASE_DIR)
    except Exception as e:
        print(f"[Re-treino] Erro: {e}")


# ─── Rotas ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(INTERFACE_DIR, 'index.html')

@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(INTERFACE_DIR, filename)

@app.errorhandler(404)
def fallback_to_index(e):
    if request.method == 'GET' and not request.path.startswith('/api/'):
        return send_from_directory(INTERFACE_DIR, 'index.html')
    return e


@app.route('/api/status')
def status():
    modelos_info = {}
    for nome in ['flash', 'padrao', 'ultra']:
        cfg = MODELOS[nome]
        modelos_info[nome] = {
            'disponivel': nome in modelos_carregados,
            'params': f"{cfg['num_camadas']} camadas, dim {cfg['dim']}",
        }
    total_k = knowledge_total()
    fontes = knowledge_por_fonte()
    return jsonify({
        'online':     len(modelos_carregados) > 0,
        'device':     device,
        'gpu':        torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'modelos':    modelos_info,
        'knowledge':  total_k,
        'knowledge_fontes': fontes,
        'retrieval':  len(retrieval.pares),
        'humor':      memoria.dados.get('humor_atual', 'neutro'),
        'consciencia': consciencia.resumo(),
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    global contador_conversas

    dados = request.get_json(force=True) or {}
    mensagem    = dados.get('mensagem', '').strip()
    tipo        = dados.get('modelo', 'flash')
    temperatura = float(dados.get('temperatura', 0.8))
    max_tokens  = int(dados.get('max_tokens', 200))
    chat_id     = dados.get('chat_id')
    token       = dados.get('token')
    web_enabled = dados.get('web_enabled', True)

    user = None
    if token:
        user = usuario_por_token(token)

    if not mensagem:
        return jsonify({'erro': 'Mensagem vazia'}), 400

    mensagem_original = mensagem
    mensagem_normalizada = normalizar(mensagem)

    user_id = user['id'] if user else None

    analise = reflexao.analisar(mensagem_normalizada)
    pensamento = []
    if mensagem_normalizada != mensagem_original.lower().strip():
        pensamento.append(f"Entendi: {mensagem_normalizada}")
    pensamento.append(f"Tipo: {analise['tipo']}")

    # ─── HISTORICO DO CHAT ───────────────────────────────────────────
    historico = []
    if chat_id and user_id:
        try:
            msgs_anteriores = chat_mensagens(chat_id, user_id)
            if msgs_anteriores:
                for msg_ant in msgs_anteriores[-5:]:
                    p = msg_ant.get('pergunta', '')
                    r = msg_ant.get('resposta', '')
                    if p and r:
                        historico.append((p, r))
                if historico:
                    pensamento.append(f"Historico: {len(historico)} turnos")
        except Exception:
            pass

    # ─── MEMORIA DO USUARIO ──────────────────────────────────────────
    contexto_memoria = memoria.gerar_contexto(user_id=user_id)
    if contexto_memoria:
        pensamento.append(f"Memoria: {contexto_memoria[:60]}...")

    # ─── CONSCIENCIA (antes de responder) ────────────────────────────
    info_consciencia = consciencia.antes_de_responder(
        mensagem_normalizada, analise['tipo']
    )
    if info_consciencia['feedback']:
        pensamento.append(f"Feedback: {info_consciencia['feedback']}")
    if info_consciencia['contexto_emocional']:
        pensamento.append(f"Emocao: {info_consciencia['contexto_emocional']}")

    resposta = None
    fonte = 'desconhecido'
    usou_web = False
    contexto_web = ''
    contexto_semantico = ''

    # ─── CAMADA 1: Busca Semantica ───────────────────────────────────
    pensamento.append("Busca semantica...")
    resp_ret, score = retrieval.buscar(mensagem_normalizada)

    if resp_ret and score >= 0.50:
        resposta = resp_ret
        fonte = 'retrieval'
        pensamento.append(f"Match semantico direto (score {score:.2f})")
    elif resp_ret and score >= 0.20:
        contexto_semantico = resp_ret
        pensamento.append(f"Contexto semantico (score {score:.2f})")

    if not resposta:
        resp_know = knowledge.buscar(mensagem_normalizada)
        if resp_know:
            if not contexto_semantico:
                if not modelos_carregados:
                    resposta = resp_know
                    fonte = 'knowledge'
                    pensamento.append("Encontrei no knowledge")
                else:
                    contexto_semantico = resp_know
                    pensamento.append("Knowledge como contexto")
            else:
                contexto_semantico += '\n' + resp_know

    # ─── CAMADA 2: Modelo Neural ─────────────────────────────────────
    if not resposta and modelos_carregados:
        pensamento.append("Gerando com modelo neural...")
        resp_modelo = gerar_com_modelo(
            mensagem_normalizada, tipo, temperatura, max_tokens,
            historico=historico,
            contexto_memoria=contexto_memoria,
            contexto_semantico=contexto_semantico[:200] if contexto_semantico else '',
        )
        if resp_modelo:
            resposta = resp_modelo
            fonte = 'modelo'
            pensamento.append("Modelo gerou resposta valida")
        else:
            pensamento.append("Modelo gerou lixo, descartado")
            if resp_ret and score >= 0.25:
                resposta = resp_ret
                fonte = 'retrieval'
                pensamento.append(f"Usando retrieval como fallback (score {score:.2f})")

    # ─── CAMADA 3: Web ───────────────────────────────────────────────
    if not resposta and web_enabled and (analise['precisa_web'] or analise['tipo'] == 'pergunta_factual'):
        pensamento.append("Pesquisando na web...")
        resultado = pesquisar(analise['topico_extraido'] or mensagem_normalizada)
        if resultado:
            linhas = resultado.replace('[Fonte: Wikipedia]\n', '').replace('[Fonte: Web]\n', '')
            resposta = linhas.strip()
            contexto_web = resultado
            usou_web = True
            fonte = 'web'
            knowledge.adicionar(mensagem, resposta, fonte='web')
            pensamento.append("Encontrei na web e salvei")

    # ─── CAMADA 4: Web forcada ───────────────────────────────────────
    if not resposta and web_enabled:
        pensamento.append("Buscando na web (ultimo recurso)...")
        resultado = pesquisar(mensagem_normalizada)
        if resultado:
            linhas = resultado.replace('[Fonte: Wikipedia]\n', '').replace('[Fonte: Web]\n', '')
            resposta = linhas.strip()
            usou_web = True
            fonte = 'web'
            knowledge.adicionar(mensagem, resposta, fonte='web')

    # ─── CAMADA 5: Fallback ──────────────────────────────────────────
    if not resposta:
        resposta = "Ainda nao sei sobre isso. Mas vou pesquisar e aprender. Tenta perguntar de outro jeito."
        fonte = 'fallback'
        topico = analise.get('topico_extraido', '')
        if topico:
            threading.Thread(
                target=crawler.crawl_agora,
                args=([topico],),
                daemon=True
            ).start()
            pensamento.append(f"Agendei pesquisa sobre '{topico}'")

    # ─── Pos-processamento ───────────────────────────────────────────

    # Score de confianca e prefixo de incerteza
    score_sem = score if resp_ret else 0.0
    confianca = consciencia.score_confianca(analise['tipo'], fonte, score_sem)
    pensamento.append(f"Confianca: {confianca}%")

    # Adiciona prefixo natural de incerteza quando confianca e baixa
    if fonte == 'modelo' and confianca < 70:
        prefixo = consciencia.gerar_prefixo_incerteza(confianca)
        if prefixo and not resposta.lower().startswith(prefixo.split()[0]):
            resposta = prefixo + resposta

    # Consciencia pos-resposta (diario, emocao, feedback)
    sucesso = fonte not in ('fallback',)
    consciencia.depois_de_responder(
        mensagem_original, resposta, analise['tipo'], fonte, sucesso
    )

    memoria.atualizar(mensagem_original, resposta, user_id=user_id)

    if fonte in ['retrieval', 'knowledge', 'web']:
        salvar_conversa_txt(mensagem_original, resposta)
        retrieval.adicionar(mensagem_original, resposta)
        # Auto-treino: salva no conversas.txt pra proximo treino aprender
        if fonte == 'web':
            try:
                conversas_path = os.path.join(DADOS_DIR, 'conversas.txt')
                with open(conversas_path, 'a', encoding='utf-8') as f:
                    f.write(f"<vitor>{mensagem_original}<fim><keilinks>{resposta}<fim>\n")
            except Exception:
                pass

    uid = user['id'] if user else None
    conversa_salvar(mensagem_original, resposta, fonte, chat_id=chat_id, usuario_id=uid)
    contador_conversas += 1

    if chat_id and user:
        try:
            msgs = chat_mensagens(chat_id, user['id'])
            if msgs and len(msgs) == 1:
                titulo = mensagem_original[:80]
                chat_atualizar_titulo(chat_id, titulo)
        except Exception:
            pass

    if contador_conversas % RETREINAR_A_CADA == 0:
        threading.Thread(target=retreinar_background, args=(tipo,), daemon=True).start()

    return jsonify({
        'resposta':    resposta,
        'modelo':      tipo if fonte == 'modelo' else None,
        'usou_web':    usou_web,
        'fonte':       fonte,
        'confianca':   confianca,
        'pensamento':  pensamento,
        'fonte_web':   contexto_web[:200] if usou_web else None,
    })


@app.route('/api/ensinar', methods=['POST'])
def ensinar():
    dados = request.get_json(force=True) or {}
    pergunta = dados.get('pergunta', '').strip()
    resposta = dados.get('resposta', '').strip()
    if not pergunta or not resposta:
        return jsonify({'erro': 'pergunta e resposta obrigatorias'}), 400
    salvar_conversa_txt(pergunta, resposta)
    knowledge.adicionar(pergunta, resposta, fonte='ensino')
    retrieval.adicionar(pergunta, resposta)
    return jsonify({'ok': True, 'total': knowledge.total()})


@app.route('/api/crawl', methods=['POST'])
def crawl_manual():
    dados = request.get_json(force=True) or {}
    topicos = dados.get('topicos', None)
    novos = crawler.crawl_agora(topicos)
    return jsonify({'novos_fatos': novos, 'total': knowledge.total()})


# ─── Autenticacao helper ─────────────────────────────────────────────────

def autenticar():
    token = None
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        token = auth[7:]
    if not token:
        dados = request.get_json(silent=True) or {}
        token = dados.get('token')
    if not token:
        return None
    return usuario_por_token(token)


# ─── Auth endpoints ──────────────────────────────────────────────────────

@app.route('/api/registrar', methods=['POST'])
def registrar():
    dados = request.get_json(force=True) or {}
    username = dados.get('username', '').strip().lower()
    senha = dados.get('senha', '').strip()
    nome = dados.get('nome', '').strip() or username
    if not username or not senha:
        return jsonify({'erro': 'Username e senha obrigatorios'}), 400
    if len(username) < 3:
        return jsonify({'erro': 'Username precisa ter pelo menos 3 caracteres'}), 400
    if len(senha) < 4:
        return jsonify({'erro': 'Senha precisa ter pelo menos 4 caracteres'}), 400
    user = usuario_criar(username, senha, nome)
    if not user:
        return jsonify({'erro': 'Username ja existe'}), 409
    return jsonify(user)


@app.route('/api/login', methods=['POST'])
def login():
    dados = request.get_json(force=True) or {}
    username = dados.get('username', '').strip().lower()
    senha = dados.get('senha', '').strip()
    if not username or not senha:
        return jsonify({'erro': 'Username e senha obrigatorios'}), 400
    user = usuario_login(username, senha)
    if not user:
        return jsonify({'erro': 'Username ou senha incorretos'}), 401
    return jsonify(user)


# ─── Chats endpoints ─────────────────────────────────────────────────────

@app.route('/api/chats', methods=['GET'])
def listar_chats():
    user = autenticar()
    if not user:
        return jsonify({'erro': 'Nao autenticado'}), 401
    return jsonify(chat_listar(user['id']))


@app.route('/api/chats', methods=['POST'])
def criar_chat():
    user = autenticar()
    if not user:
        return jsonify({'erro': 'Nao autenticado'}), 401
    dados = request.get_json(force=True) or {}
    titulo = dados.get('titulo', 'Nova conversa')
    chat = chat_criar(user['id'], titulo)
    return jsonify(chat)


@app.route('/api/chats/<int:chat_id>', methods=['GET'])
def ver_chat(chat_id):
    user = autenticar()
    if not user:
        return jsonify({'erro': 'Nao autenticado'}), 401
    msgs = chat_mensagens(chat_id, user['id'])
    if msgs is None:
        return jsonify({'erro': 'Chat nao encontrado'}), 404
    return jsonify(msgs)


@app.route('/api/chats/<int:chat_id>', methods=['DELETE'])
def deletar_chat(chat_id):
    user = autenticar()
    if not user:
        return jsonify({'erro': 'Nao autenticado'}), 401
    if chat_deletar(chat_id, user['id']):
        return jsonify({'ok': True})
    return jsonify({'erro': 'Chat nao encontrado'}), 404


@app.route('/api/historico')
def historico():
    try:
        return jsonify(conversa_historico(50))
    except Exception:
        return jsonify([])


@app.route('/api/memoria')
def ver_memoria():
    return jsonify({
        'global': memoria.dados,
        'usuarios': len(memoria._usuarios),
    })


@app.route('/api/consciencia')
def ver_consciencia():
    """Estado de consciencia da Keilinks: emocao, confianca, diario"""
    resumo = consciencia.resumo()
    # Adiciona ultimas entradas do diario
    diario = []
    if os.path.exists(consciencia.diario_path):
        try:
            with open(consciencia.diario_path, 'r', encoding='utf-8') as f:
                linhas = f.readlines()
            diario = [l.strip() for l in linhas[-10:] if l.strip()]
        except Exception:
            pass
    resumo['diario_recente'] = diario
    return jsonify(resumo)


@app.route('/api/crawler-log')
def ver_crawler_log():
    try:
        return jsonify(crawler_log_recentes(20))
    except Exception:
        return jsonify([])


if __name__ == '__main__':
    inicializar()
    app.run(host='0.0.0.0', port=5000, debug=False)
