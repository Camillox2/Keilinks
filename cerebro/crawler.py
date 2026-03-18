"""
Crawler Multi-Fonte da Keilinks v5
Busca conhecimento em background de 4+ fontes:
  - Wikipedia PT
  - StackOverflow (API publica)
  - Reddit (JSON publico)
  - Dev.to (API publica)
Intervalo: 5 minutos. Salva no MySQL.
"""

import os
import sys
import time
import threading
import re
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dados.database import (
    knowledge_adicionar, knowledge_existe, knowledge_total,
    crawler_log_salvar, conversa_historico
)

HEADERS = {'User-Agent': 'Keilinks/2.0 (IA pessoal; github.com)'}

# Topicos base do Vitor
TOPICOS_BASE = [
    'Python (linguagem de programacao)', 'PyTorch', 'React (JavaScript)',
    'Node.js', 'Inteligencia artificial', 'Rede neural artificial',
    'Transformer (modelo de linguagem)', 'CUDA', 'GPU',
    'Machine learning', 'Deep learning', 'JavaScript',
    'Club de Regatas Vasco da Gama', 'Campeonato Brasileiro de Futebol',
    'Curitiba', 'Engenharia de software', 'Sigmund Freud',
    'Psicologia', 'Pizza',
]

# Tags para StackOverflow
STACK_TAGS = ['python', 'pytorch', 'react', 'javascript', 'node.js', 'cuda', 'flask', 'mysql']

# Subs do Reddit
REDDIT_SUBS = ['Python', 'MachineLearning', 'learnprogramming', 'webdev', 'programming']

# Tags do Dev.to
DEVTO_TAGS = ['python', 'machinelearning', 'javascript', 'webdev', 'react']


def _log(fonte: str, msg: str):
    """Log formatado com timestamp"""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [Crawler] {fonte}: {msg}")


def _categorizar(texto: str) -> str:
    """Tenta categorizar o conteudo"""
    t = texto.lower()
    if any(w in t for w in ['python', 'javascript', 'react', 'node', 'flask', 'codigo', 'programar', 'bug', 'api']):
        return 'programacao'
    if any(w in t for w in ['neural', 'machine learning', 'deep learning', 'pytorch', 'cuda', 'gpu', 'transformer', 'ia']):
        return 'tech'
    if any(w in t for w in ['vasco', 'futebol', 'gol', 'campeonato', 'copa']):
        return 'futebol'
    if any(w in t for w in ['atomo', 'fisica', 'quimica', 'biologia', 'ciencia']):
        return 'ciencia'
    return 'geral'


# ─── FONTES ──────────────────────────────────────────────────────────────

def buscar_wikipedia(titulo: str) -> dict | None:
    """Busca resumo na Wikipedia PT"""
    try:
        search_url = "https://pt.wikipedia.org/w/api.php"
        params = {
            'action': 'query', 'list': 'search',
            'srsearch': titulo, 'srlimit': 1, 'format': 'json'
        }
        resp = requests.get(search_url, params=params, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return None

        resultados = resp.json().get('query', {}).get('search', [])
        if not resultados:
            return None

        titulo_real = resultados[0]['title']
        summary_url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(titulo_real)}"
        resp2 = requests.get(summary_url, headers=HEADERS, timeout=8)
        if resp2.status_code != 200:
            return None

        data = resp2.json()
        resumo = data.get('extract', '')
        if len(resumo) < 50:
            return None

        return {
            'titulo': titulo_real,
            'resumo': resumo[:800],
            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
        }
    except Exception:
        return None


def buscar_stackoverflow(tag: str, limite: int = 3) -> list[dict]:
    """Busca perguntas populares no StackOverflow por tag"""
    try:
        url = "https://api.stackexchange.com/2.3/questions"
        params = {
            'order': 'desc', 'sort': 'votes',
            'tagged': tag, 'site': 'stackoverflow',
            'filter': 'withbody', 'pagesize': limite,
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []

        items = resp.json().get('items', [])
        resultados = []
        for item in items:
            titulo = item.get('title', '')
            # Pega resposta aceita se existir
            body = item.get('body', '')
            # Limpa HTML
            texto = re.sub(r'<[^>]+>', '', body)[:600]
            if titulo and texto:
                resultados.append({
                    'titulo': titulo,
                    'texto': texto,
                    'url': item.get('link', ''),
                    'score': item.get('score', 0),
                })
        return resultados
    except Exception:
        return []


def buscar_reddit(sub: str, limite: int = 5) -> list[dict]:
    """Busca posts quentes de um subreddit"""
    try:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limite}"
        headers = {'User-Agent': 'Keilinks/2.0 (IA pessoal de estudo)'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []

        posts = resp.json().get('data', {}).get('children', [])
        resultados = []
        for post in posts:
            d = post.get('data', {})
            titulo = d.get('title', '')
            texto = d.get('selftext', '')[:500]
            if d.get('stickied'):
                continue
            if titulo and len(titulo) > 10:
                resultados.append({
                    'titulo': titulo,
                    'texto': texto if texto else titulo,
                    'url': f"https://reddit.com{d.get('permalink', '')}",
                    'score': d.get('score', 0),
                })
        return resultados
    except Exception:
        return []


def buscar_devto(tag: str, limite: int = 5) -> list[dict]:
    """Busca artigos populares no Dev.to"""
    try:
        url = f"https://dev.to/api/articles?tag={tag}&top=7&per_page={limite}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []

        artigos = resp.json()
        resultados = []
        for art in artigos:
            titulo = art.get('title', '')
            descricao = art.get('description', '')
            if titulo:
                resultados.append({
                    'titulo': titulo,
                    'texto': descricao[:400] if descricao else titulo,
                    'url': art.get('url', ''),
                    'score': art.get('positive_reactions_count', 0),
                })
        return resultados
    except Exception:
        return []


# ─── EXTRAI TOPICOS DE CONVERSAS ─────────────────────────────────────────

def extrair_topicos_de_conversas() -> list[str]:
    """Extrai topicos das ultimas conversas no MySQL"""
    try:
        recentes = conversa_historico(20)
    except Exception:
        return []

    topicos = set()
    patterns = [
        r'(?:o que (?:e|sao|significa)|quem (?:e|foi|sao))\s+(.+)',
        r'(?:me (?:fala|conta|explica) (?:sobre|do|da|de))\s+(.+)',
        r'(?:como funciona)\s+(.+)',
    ]
    for conv in recentes:
        msg = conv.get('pergunta', '').lower()
        for pat in patterns:
            match = re.search(pat, msg)
            if match:
                topico = match.group(1).strip().rstrip('?.,!')
                if len(topico) > 2:
                    topicos.add(topico)

    return list(topicos)[:10]


# ─── CRAWL PRINCIPAL ─────────────────────────────────────────────────────

def crawl_wikipedia(topicos: list[str]) -> int:
    """Crawl Wikipedia para os topicos dados"""
    novos = 0
    for topico in topicos:
        if knowledge_existe(topico):
            continue

        resultado = buscar_wikipedia(topico)
        if not resultado:
            continue

        cat = _categorizar(resultado['resumo'])
        knowledge_adicionar(
            pergunta=f"o que e {resultado['titulo'].lower()}",
            resposta=resultado['resumo'],
            fonte='wikipedia', categoria=cat,
            url=resultado['url'],
        )
        # Variacao
        knowledge_adicionar(
            pergunta=f"me fala sobre {resultado['titulo'].lower()}",
            resposta=resultado['resumo'],
            fonte='wikipedia', categoria=cat,
            url=resultado['url'],
        )
        novos += 1
        time.sleep(0.5)

    crawler_log_salvar('wikipedia', f"{len(topicos)} topicos", True, novos)
    if novos > 0:
        _log('Wikipedia', f"+{novos} fatos novos")
    return novos


def crawl_stackoverflow() -> int:
    """Crawl StackOverflow por tags"""
    novos = 0
    topicos_encontrados = []
    for tag in STACK_TAGS:
        resultados = buscar_stackoverflow(tag, limite=2)
        for r in resultados:
            if knowledge_existe(r['titulo']):
                continue
            knowledge_adicionar(
                pergunta=r['titulo'],
                resposta=r['texto'],
                fonte='stackoverflow', categoria='programacao',
                url=r['url'], relevancia=r.get('score', 0),
            )
            novos += 1
            topicos_encontrados.append(tag)
        time.sleep(0.3)

    crawler_log_salvar('stackoverflow', ', '.join(set(topicos_encontrados)), True, novos)
    if novos > 0:
        _log('StackOverflow', f"+{novos} fatos ({', '.join(set(topicos_encontrados))})")
    return novos


def crawl_reddit() -> int:
    """Crawl Reddit dos subs configurados"""
    novos = 0
    subs_com_resultado = []
    for sub in REDDIT_SUBS:
        resultados = buscar_reddit(sub, limite=3)
        for r in resultados:
            if knowledge_existe(r['titulo']):
                continue
            cat = _categorizar(r['titulo'] + ' ' + r.get('texto', ''))
            knowledge_adicionar(
                pergunta=r['titulo'],
                resposta=r['texto'],
                fonte='reddit', categoria=cat,
                url=r['url'], relevancia=r.get('score', 0),
            )
            novos += 1
            subs_com_resultado.append(f"r/{sub}")
        time.sleep(0.5)

    crawler_log_salvar('reddit', ', '.join(set(subs_com_resultado)), True, novos)
    if novos > 0:
        _log('Reddit', f"+{novos} fatos ({', '.join(set(subs_com_resultado))})")
    return novos


def crawl_devto() -> int:
    """Crawl Dev.to por tags"""
    novos = 0
    tags_usadas = []
    for tag in DEVTO_TAGS:
        resultados = buscar_devto(tag, limite=3)
        for r in resultados:
            if knowledge_existe(r['titulo']):
                continue
            knowledge_adicionar(
                pergunta=r['titulo'],
                resposta=r['texto'],
                fonte='devto', categoria='tech',
                url=r['url'], relevancia=r.get('score', 0),
            )
            novos += 1
            tags_usadas.append(tag)
        time.sleep(0.3)

    crawler_log_salvar('devto', ', '.join(set(tags_usadas)), True, novos)
    if novos > 0:
        _log('Dev.to', f"+{novos} fatos ({', '.join(set(tags_usadas))})")
    return novos


def crawl_completo(topicos: list[str] = None) -> int:
    """Executa um ciclo completo de crawling em todas as fontes"""
    total_novos = 0

    if topicos is None:
        topicos = TOPICOS_BASE.copy()
        topicos.extend(extrair_topicos_de_conversas())

    # Wikipedia
    try:
        total_novos += crawl_wikipedia(topicos)
    except Exception as e:
        _log('Wikipedia', f"Erro: {e}")
        crawler_log_salvar('wikipedia', str(e)[:200], False, 0)

    # StackOverflow
    try:
        total_novos += crawl_stackoverflow()
    except Exception as e:
        _log('StackOverflow', f"Erro: {e}")
        crawler_log_salvar('stackoverflow', str(e)[:200], False, 0)

    # Reddit
    try:
        total_novos += crawl_reddit()
    except Exception as e:
        _log('Reddit', f"Erro: {e}")
        crawler_log_salvar('reddit', str(e)[:200], False, 0)

    # Dev.to
    try:
        total_novos += crawl_devto()
    except Exception as e:
        _log('Dev.to', f"Erro: {e}")
        crawler_log_salvar('devto', str(e)[:200], False, 0)

    total = knowledge_total()
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [Crawler] Ciclo completo: +{total_novos} fatos | Total: {total}")

    return total_novos


class CrawlerBackground:
    """Roda o crawler periodicamente em background"""

    def __init__(self, intervalo_minutos=5):
        self.intervalo = intervalo_minutos * 60
        self.rodando = False
        self.thread = None

    def iniciar(self):
        if self.rodando:
            return
        self.rodando = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[Crawler] Rodando em background a cada {self.intervalo // 60} min")

    def parar(self):
        self.rodando = False

    def _loop(self):
        time.sleep(10)  # Espera servidor iniciar
        while self.rodando:
            try:
                crawl_completo()
            except Exception as e:
                _log('ERRO', str(e))
            time.sleep(self.intervalo)

    def crawl_agora(self, topicos: list[str] = None) -> int:
        """Forca um crawl imediato"""
        return crawl_completo(topicos)


if __name__ == '__main__':
    print("Executando crawler manualmente...")
    novos = crawl_completo()
    print(f"Concluido: {novos} fatos novos | Total: {knowledge_total()}")
