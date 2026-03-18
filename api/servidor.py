"""
Servidor da Keilinks v6 — Cerebro Completo + MySQL + Fuzzy
Camadas: Normalizar -> Reflexao -> Retrieval -> Knowledge -> Web -> Modelo -> Fallback
Auto-aprendizado + Crawler multi-fonte em background + MySQL + Traducao EN>>PT
"""

import torch
import sys
import os
import json
import threading
from datetime import datetime

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

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
modelos_carregados = {}
tokenizador = None

# ─── Sistemas do Cerebro ──────────────────────────────────────────────────
retrieval  = Retrieval()
knowledge  = Knowledge()
memoria    = Memoria(os.path.join(BASE_DIR, 'dados', 'memoria.json'))
reflexao   = Reflexao()
crawler    = CrawlerBackground(intervalo_minutos=5)

INTERFACE_DIR   = os.path.join(BASE_DIR, 'interface')
DADOS_DIR       = os.path.join(BASE_DIR, 'dados')
CHECKPOINTS     = os.path.join(BASE_DIR, 'checkpoints')
APRENDIZADO     = os.path.join(DADOS_DIR, 'aprendizado.txt')

contador_conversas = 0
RETREINAR_A_CADA = 50


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
    print(f"  Keilinks v6 — MySQL + Multi-Crawler + Fuzzy")
    print(f"  Device: {device.upper()}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")

    # Inicializa MySQL
    inicializar_banco()

    # Migra JSONs existentes (so faz algo se tiver dados e tabelas vazias)
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

    disponiveis = list(modelos_carregados.keys())
    total_k = knowledge_total()
    fontes = knowledge_por_fonte()
    print(f"\n  Modelos: {disponiveis if disponiveis else 'nenhum'}")
    print(f"  Knowledge: {total_k} fatos {fontes}")
    print(f"  Retrieval: {len(retrieval.pares)} pares")
    print(f"\n  http://localhost:5000\n")

    # Inicia crawler em background (a cada 5 min)
    crawler.iniciar()


# ─── Utilidades ───────────────────────────────────────────────────────────

def texto_eh_lixo(texto):
    if not texto or len(texto) < 3:
        return True
    # Muita repeticao de caracteres
    if len(set(texto)) < len(texto) * 0.15:
        return True
    palavras = texto.split()
    # Palavra unica gigante (sem espacos)
    if len(palavras) < 2 and len(texto) > 20:
        return True
    # Muitas palavras curtinhas sem sentido
    curtas = sum(1 for p in palavras if len(p) <= 2)
    if len(palavras) > 3 and curtas / len(palavras) > 0.7:
        return True
    # Frases sem sentido: poucas palavras e nao parece resposta real
    if len(palavras) <= 3 and len(texto) < 25:
        # Precisa ter pelo menos 2 palavras com 3+ letras
        reais = [p for p in palavras if len(p) >= 3]
        if len(reais) < 2:
            return True
    # Mesma palavra repetida demais
    if len(palavras) >= 3:
        unicas = set(p.lower() for p in palavras)
        if len(unicas) < max(len(palavras) * 0.6, 2):
            return True
    # Contem tokens especiais que vazaram
    if any(t in texto for t in ['<vitor>', '<keilinks>', '<fim>', '<pad>']):
        return True
    return False


def gerar_com_modelo(mensagem, tipo, temperatura=0.8, max_tokens=200):
    modelo = None
    for t in [tipo, 'flash', 'padrao', 'ultra']:
        if t in modelos_carregados:
            modelo = modelos_carregados[t]
            break
    if not modelo:
        return None

    prompt = f"<vitor>{mensagem}<fim><keilinks>"
    tokens = torch.tensor([tokenizador.encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        saida = modelo.gerar(tokens, max_tokens=max_tokens, temperatura=temperatura)
    texto = tokenizador.decode(saida[0].tolist())
    if '<keilinks>' in texto:
        resp = texto.split('<keilinks>')[-1]
        if '<fim>' in resp:
            resp = resp.split('<fim>')[0]
        resp = resp.strip()
    else:
        resp = texto.strip()
    return None if texto_eh_lixo(resp) else resp


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

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(INTERFACE_DIR, filename)


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

    # Autenticacao opcional — pega usuario se tiver token
    user = None
    if token:
        user = usuario_por_token(token)

    if not mensagem:
        return jsonify({'erro': 'Mensagem vazia'}), 400

    # ─── NORMALIZACAO (corrige typos, expande abreviacoes) ────────────
    mensagem_original = mensagem
    mensagem_normalizada = normalizar(mensagem)

    # ─── REFLEXAO ─────────────────────────────────────────────────────
    analise = reflexao.analisar(mensagem_normalizada)
    pensamento = []
    if mensagem_normalizada != mensagem_original.lower().strip():
        pensamento.append(f"Entendi: {mensagem_normalizada}")
    pensamento.append(f"Tipo: {analise['tipo']}")

    resposta = None
    fonte = 'desconhecido'
    usou_web = False
    contexto_web = ''

    # ─── CAMADA 1: Retrieval ──────────────────────────────────────────
    pensamento.append("Buscando no dataset...")
    resp_ret, score = retrieval.buscar(mensagem_normalizada)
    if resp_ret and score >= 0.20:
        resposta = resp_ret
        fonte = 'retrieval'
        pensamento.append(f"Encontrei (score {score:.2f})")

    # ─── CAMADA 2: Knowledge (MySQL) ─────────────────────────────────
    if not resposta:
        pensamento.append("Buscando no knowledge...")
        resp_know = knowledge.buscar(mensagem_normalizada)
        if resp_know:
            resposta = resp_know
            fonte = 'knowledge'
            pensamento.append("Encontrei no knowledge")

    # ─── CAMADA 3: Web ───────────────────────────────────────────────
    if not resposta and (analise['precisa_web'] or analise['tipo'] == 'pergunta_factual'):
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

    # ─── CAMADA 4: Modelo neural ─────────────────────────────────────
    if not resposta and modelos_carregados:
        pensamento.append("Gerando com modelo...")
        resp_modelo = gerar_com_modelo(mensagem_normalizada, tipo, temperatura, max_tokens)
        if resp_modelo and reflexao.validar_resposta(resp_modelo):
            resposta = resp_modelo
            fonte = 'modelo'
            pensamento.append("Modelo gerou resposta valida")
        else:
            pensamento.append("Modelo gerou lixo, descartado")

    # ─── CAMADA 5: Web forcada ───────────────────────────────────────
    if not resposta:
        pensamento.append("Buscando na web (ultimo recurso)...")
        resultado = pesquisar(mensagem_normalizada)
        if resultado:
            linhas = resultado.replace('[Fonte: Wikipedia]\n', '').replace('[Fonte: Web]\n', '')
            resposta = linhas.strip()
            usou_web = True
            fonte = 'web'
            knowledge.adicionar(mensagem, resposta, fonte='web')

    # ─── CAMADA 6: Fallback ──────────────────────────────────────────
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

    memoria.atualizar(mensagem_original, resposta)

    # So salva pra treino se a resposta faz sentido pro contexto
    # Nao salva respostas do modelo (podem ser lixo) nem fallback
    if fonte in ['retrieval', 'knowledge', 'web']:
        salvar_conversa_txt(mensagem_original, resposta)
        retrieval.adicionar(mensagem_original, resposta)

    # Salva no MySQL (guarda a mensagem original do usuario)
    uid = user['id'] if user else None
    conversa_salvar(mensagem_original, resposta, fonte, chat_id=chat_id, usuario_id=uid)
    contador_conversas += 1

    # Auto-gera titulo do chat na primeira mensagem
    if chat_id and user:
        try:
            msgs = chat_mensagens(chat_id, user['id'])
            if msgs and len(msgs) == 1:
                titulo = mensagem_original[:80]
                chat_atualizar_titulo(chat_id, titulo)
        except Exception:
            pass

    # Re-treino periodico
    if contador_conversas % RETREINAR_A_CADA == 0:
        threading.Thread(target=retreinar_background, args=(tipo,), daemon=True).start()

    return jsonify({
        'resposta':    resposta,
        'modelo':      tipo if fonte == 'modelo' else None,
        'usou_web':    usou_web,
        'fonte':       fonte,
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
    """Forca o crawler a buscar agora"""
    dados = request.get_json(force=True) or {}
    topicos = dados.get('topicos', None)
    novos = crawler.crawl_agora(topicos)
    return jsonify({'novos_fatos': novos, 'total': knowledge.total()})


# ─── Autenticacao helper ─────────────────────────────────────────────────

def autenticar():
    """Extrai usuario do token no header ou body. Retorna dict do usuario ou None."""
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
    return jsonify(memoria.dados)


@app.route('/api/crawler-log')
def ver_crawler_log():
    """Retorna logs recentes do crawler"""
    try:
        return jsonify(crawler_log_recentes(20))
    except Exception:
        return jsonify([])


if __name__ == '__main__':
    inicializar()
    app.run(host='0.0.0.0', port=5000, debug=False)
