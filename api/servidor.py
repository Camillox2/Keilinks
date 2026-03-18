"""
Servidor da Keilinks v5 — Cerebro Completo + MySQL
Camadas: Reflexao -> Retrieval -> Knowledge -> Web -> Modelo -> Fallback
Auto-aprendizado + Crawler multi-fonte em background + MySQL
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
)
from busca.web import pesquisar, precisa_buscar
from cerebro.crawler import CrawlerBackground
from cerebro.memoria import Memoria
from cerebro.reflexao import Reflexao

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
    print(f"  Keilinks v5 — MySQL + Multi-Crawler")
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
    if len(set(texto)) < len(texto) * 0.15:
        return True
    palavras = texto.split()
    if len(palavras) < 2 and len(texto) > 20:
        return True
    curtas = sum(1 for p in palavras if len(p) <= 2)
    if len(palavras) > 3 and curtas / len(palavras) > 0.7:
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

    if not mensagem:
        return jsonify({'erro': 'Mensagem vazia'}), 400

    # ─── REFLEXAO ─────────────────────────────────────────────────────
    analise = reflexao.analisar(mensagem)
    pensamento = []
    pensamento.append(f"Tipo: {analise['tipo']}")

    resposta = None
    fonte = 'desconhecido'
    usou_web = False
    contexto_web = ''

    # ─── CAMADA 1: Retrieval ──────────────────────────────────────────
    pensamento.append("Buscando no dataset...")
    resp_ret, score = retrieval.buscar(mensagem)
    if resp_ret and score >= 0.20:
        resposta = resp_ret
        fonte = 'retrieval'
        pensamento.append(f"Encontrei (score {score:.2f})")

    # ─── CAMADA 2: Knowledge (MySQL) ─────────────────────────────────
    if not resposta:
        pensamento.append("Buscando no knowledge...")
        resp_know = knowledge.buscar(mensagem)
        if resp_know:
            resposta = resp_know
            fonte = 'knowledge'
            pensamento.append("Encontrei no knowledge")

    # ─── CAMADA 3: Web ───────────────────────────────────────────────
    if not resposta and (analise['precisa_web'] or analise['tipo'] == 'pergunta_factual'):
        pensamento.append("Pesquisando na web...")
        resultado = pesquisar(analise['topico_extraido'] or mensagem)
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
        resp_modelo = gerar_com_modelo(mensagem, tipo, temperatura, max_tokens)
        if resp_modelo and reflexao.validar_resposta(resp_modelo):
            resposta = resp_modelo
            fonte = 'modelo'
            pensamento.append("Modelo gerou resposta valida")
        else:
            pensamento.append("Modelo gerou lixo, descartado")

    # ─── CAMADA 5: Web forcada ───────────────────────────────────────
    if not resposta:
        pensamento.append("Buscando na web (ultimo recurso)...")
        resultado = pesquisar(mensagem)
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

    memoria.atualizar(mensagem, resposta)

    if fonte not in ['fallback']:
        salvar_conversa_txt(mensagem, resposta)
        retrieval.adicionar(mensagem, resposta)

    # Salva no MySQL
    conversa_salvar(mensagem, resposta, fonte)
    contador_conversas += 1

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
