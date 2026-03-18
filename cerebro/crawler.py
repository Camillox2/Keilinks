"""
Crawler Multi-Fonte da Keilinks v6
Fontes: Wikipedia, StackOverflow, Reddit, Dev.to, Hacker News, Google News
Intervalo: 5 minutos. Salva no MySQL. Traduz EN>>PT automaticamente.
Crawl em paralelo com ThreadPoolExecutor.
"""

import os
import sys
import time
import threading
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dados.database import (
    knowledge_adicionar, knowledge_existe, knowledge_total,
    crawler_log_salvar, conversa_historico
)
from cerebro.tradutor import traduzir_se_ingles
from cerebro.livros import crawl_livros_completo

HEADERS = {'User-Agent': 'Keilinks/2.0 (IA pessoal; github.com)'}

# Topicos base
TOPICOS_BASE = [
    # Tecnologia
    'Python (linguagem de programacao)', 'PyTorch', 'React (JavaScript)',
    'Node.js', 'Inteligencia artificial', 'Rede neural artificial',
    'Transformer (modelo de linguagem)', 'CUDA', 'GPU',
    'Machine learning', 'Deep learning', 'JavaScript',
    # Futebol
    'Club de Regatas Vasco da Gama', 'Campeonato Brasileiro de Futebol',
    # Geral
    'Curitiba', 'Engenharia de software', 'Pizza',
    # Conversacao e autoajuda
    'Comunicacao interpessoal', 'Inteligencia emocional',
    'Empatia', 'Escuta ativa', 'Assertividade',
    'Autoajuda', 'Desenvolvimento pessoal',
    'Psicologia positiva', 'Resiliencia (psicologia)',
    'Motivacao', 'Autoestima', 'Ansiedade',
    # Livros famosos de autoajuda/conversa
    'Como Fazer Amigos e Influenciar Pessoas',
    'O Poder do Habito', 'Inteligencia Emocional (livro)',
    'Pai Rico Pai Pobre', 'O Poder do Agora',
    'Sapiens (livro)', 'Mindset', 'Habitos Atomicos',
    # Psicologia
    'Sigmund Freud', 'Carl Jung', 'Psicologia',
    'Psicanalise', 'Terapia cognitivo-comportamental',
    'Linguagem corporal', 'Programacao neurolinguistica',
]

STACK_TAGS = ['python', 'pytorch', 'react', 'javascript', 'node.js', 'cuda', 'flask', 'mysql']
REDDIT_SUBS = ['Python', 'MachineLearning', 'learnprogramming', 'webdev', 'programming']
DEVTO_TAGS = ['python', 'machinelearning', 'javascript', 'webdev', 'react']

# Topicos para Google News (PT-BR)
NEWS_TOPICOS = ['tecnologia', 'inteligencia artificial', 'programacao', 'vasco futebol', 'ciencia']


def _log(fonte: str, msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [Crawler] {fonte}: {msg}")


def _categorizar(texto: str) -> str:
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


def _traduzir_e_salvar(pergunta: str, resposta: str, fonte: str, categoria: str = None,
                       url: str = None, relevancia: int = 0):
    """Traduz se ingles e salva no MySQL"""
    resp_final, idioma = traduzir_se_ingles(resposta)
    perg_final, _ = traduzir_se_ingles(pergunta)
    cat = categoria or _categorizar(resp_final)
    knowledge_adicionar(perg_final, resp_final, fonte, cat, url, relevancia)


# ─── FONTES ──────────────────────────────────────────────────────────────

def buscar_wikipedia(titulo: str) -> dict | None:
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
            body = item.get('body', '')
            texto = re.sub(r'<[^>]+>', '', body)[:600]
            if titulo and texto:
                resultados.append({
                    'titulo': titulo, 'texto': texto,
                    'url': item.get('link', ''), 'score': item.get('score', 0),
                })
        return resultados
    except Exception:
        return []


def buscar_reddit(sub: str, limite: int = 5) -> list[dict]:
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
                    'titulo': titulo, 'texto': texto if texto else titulo,
                    'url': f"https://reddit.com{d.get('permalink', '')}",
                    'score': d.get('score', 0),
                })
        return resultados
    except Exception:
        return []


def buscar_devto(tag: str, limite: int = 5) -> list[dict]:
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
                    'titulo': titulo, 'texto': descricao[:400] if descricao else titulo,
                    'url': art.get('url', ''), 'score': art.get('positive_reactions_count', 0),
                })
        return resultados
    except Exception:
        return []


def buscar_hackernews(limite: int = 10) -> list[dict]:
    """Busca top stories do Hacker News"""
    try:
        resp = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return []
        ids = resp.json()[:limite]
        resultados = []
        for item_id in ids:
            try:
                item_resp = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json",
                    headers=HEADERS, timeout=5
                )
                if item_resp.status_code != 200:
                    continue
                item = item_resp.json()
                titulo = item.get('title', '')
                url = item.get('url', '')
                texto = item.get('text', '')
                if not texto:
                    texto = titulo
                # Limpa HTML se tiver
                texto = re.sub(r'<[^>]+>', '', texto)[:500]
                if titulo:
                    resultados.append({
                        'titulo': titulo, 'texto': texto,
                        'url': url, 'score': item.get('score', 0),
                    })
            except Exception:
                continue
            time.sleep(0.1)
        return resultados
    except Exception:
        return []


def buscar_google_news(topico: str, limite: int = 5) -> list[dict]:
    """Busca noticias via Google News RSS (PT-BR, gratis, sem key)"""
    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(topico)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)
        resultados = []
        for item in root.findall('.//item')[:limite]:
            titulo = item.findtext('title', '')
            descricao = item.findtext('description', '')
            link = item.findtext('link', '')
            pub_date = item.findtext('pubDate', '')
            # Limpa HTML da descricao
            descricao = re.sub(r'<[^>]+>', '', descricao)[:400] if descricao else titulo
            if titulo:
                resultados.append({
                    'titulo': titulo,
                    'texto': descricao,
                    'url': link,
                    'score': 0,
                    'data': pub_date,
                })
        return resultados
    except Exception:
        return []


# ─── EXTRAI TOPICOS ──────────────────────────────────────────────────────

def extrair_topicos_de_conversas() -> list[str]:
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


# ─── CRAWL POR FONTE ─────────────────────────────────────────────────────

def crawl_wikipedia(topicos: list[str]) -> int:
    novos = 0
    for topico in topicos:
        if knowledge_existe(topico):
            continue
        resultado = buscar_wikipedia(topico)
        if not resultado:
            continue
        cat = _categorizar(resultado['resumo'])
        knowledge_adicionar(
            f"o que e {resultado['titulo'].lower()}", resultado['resumo'],
            'wikipedia', cat, resultado['url'],
        )
        novos += 1
        time.sleep(0.5)
    crawler_log_salvar('wikipedia', f"{len(topicos)} topicos", True, novos)
    if novos > 0:
        _log('Wikipedia', f"+{novos} fatos novos")
    return novos


def crawl_stackoverflow() -> int:
    novos = 0
    tags_usadas = []
    for tag in STACK_TAGS:
        resultados = buscar_stackoverflow(tag, limite=2)
        for r in resultados:
            if knowledge_existe(r['titulo']):
                continue
            _traduzir_e_salvar(
                r['titulo'], r['texto'], 'stackoverflow', 'programacao',
                r['url'], r.get('score', 0),
            )
            novos += 1
            tags_usadas.append(tag)
        time.sleep(0.3)
    crawler_log_salvar('stackoverflow', ', '.join(set(tags_usadas)), True, novos)
    if novos > 0:
        _log('StackOverflow', f"+{novos} fatos ({', '.join(set(tags_usadas))})")
    return novos


def crawl_reddit() -> int:
    novos = 0
    subs_usados = []
    for sub in REDDIT_SUBS:
        resultados = buscar_reddit(sub, limite=3)
        for r in resultados:
            if knowledge_existe(r['titulo']):
                continue
            _traduzir_e_salvar(
                r['titulo'], r['texto'], 'reddit', None,
                r['url'], r.get('score', 0),
            )
            novos += 1
            subs_usados.append(f"r/{sub}")
        time.sleep(0.5)
    crawler_log_salvar('reddit', ', '.join(set(subs_usados)), True, novos)
    if novos > 0:
        _log('Reddit', f"+{novos} fatos ({', '.join(set(subs_usados))})")
    return novos


def crawl_devto() -> int:
    novos = 0
    tags_usadas = []
    for tag in DEVTO_TAGS:
        resultados = buscar_devto(tag, limite=3)
        for r in resultados:
            if knowledge_existe(r['titulo']):
                continue
            _traduzir_e_salvar(
                r['titulo'], r['texto'], 'devto', 'tech',
                r['url'], r.get('score', 0),
            )
            novos += 1
            tags_usadas.append(tag)
        time.sleep(0.3)
    crawler_log_salvar('devto', ', '.join(set(tags_usadas)), True, novos)
    if novos > 0:
        _log('Dev.to', f"+{novos} fatos ({', '.join(set(tags_usadas))})")
    return novos


def crawl_hackernews() -> int:
    novos = 0
    resultados = buscar_hackernews(limite=8)
    titulos = []
    for r in resultados:
        if knowledge_existe(r['titulo']):
            continue
        _traduzir_e_salvar(
            r['titulo'], r['texto'], 'hackernews', 'tech',
            r['url'], r.get('score', 0),
        )
        novos += 1
        titulos.append(r['titulo'][:30])
    crawler_log_salvar('hackernews', f"{len(resultados)} stories", True, novos)
    if novos > 0:
        _log('HackerNews', f"+{novos} fatos")
    return novos


def crawl_noticias() -> int:
    novos = 0
    topicos_usados = []
    for topico in NEWS_TOPICOS:
        resultados = buscar_google_news(topico, limite=3)
        for r in resultados:
            if knowledge_existe(r['titulo']):
                continue
            cat = _categorizar(r['titulo'] + ' ' + r.get('texto', ''))
            knowledge_adicionar(
                r['titulo'], r['texto'], 'web', cat, r['url'],
            )
            novos += 1
            topicos_usados.append(topico)
        time.sleep(0.3)
    crawler_log_salvar('noticias', ', '.join(set(topicos_usados)), True, novos)
    if novos > 0:
        _log('Noticias', f"+{novos} fatos ({', '.join(set(topicos_usados))})")
    return novos


# ─── CRAWL COMPLETO (PARALELO) ───────────────────────────────────────────

def crawl_completo(topicos: list[str] = None) -> int:
    total_novos = 0

    if topicos is None:
        topicos = TOPICOS_BASE.copy()
        topicos.extend(extrair_topicos_de_conversas())

    # Executa fontes em paralelo com ThreadPoolExecutor
    fontes = {
        'Wikipedia': lambda: crawl_wikipedia(topicos),
        'StackOverflow': crawl_stackoverflow,
        'Reddit': crawl_reddit,
        'Dev.to': crawl_devto,
        'HackerNews': crawl_hackernews,
        'Noticias': crawl_noticias,
        'Livros': crawl_livros_completo,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for nome, func in fontes.items():
            futures[executor.submit(func)] = nome

        for future in as_completed(futures):
            nome = futures[future]
            try:
                novos = future.result()
                total_novos += novos
            except Exception as e:
                _log(nome, f"Erro: {e}")
                crawler_log_salvar(nome.lower(), str(e)[:200], False, 0)

    total = knowledge_total()
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [Crawler] Ciclo completo: +{total_novos} fatos | Total: {total}")

    return total_novos


class CrawlerBackground:
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
        print(f"[Crawler] Rodando em background a cada {self.intervalo // 60} min (6 fontes, paralelo)")

    def parar(self):
        self.rodando = False

    def _loop(self):
        time.sleep(10)
        while self.rodando:
            try:
                crawl_completo()
            except Exception as e:
                _log('ERRO', str(e))
            time.sleep(self.intervalo)

    def crawl_agora(self, topicos: list[str] = None) -> int:
        return crawl_completo(topicos)


if __name__ == '__main__':
    print("Executando crawler manualmente (6 fontes, paralelo)...")
    novos = crawl_completo()
    print(f"Concluido: {novos} fatos novos | Total: {knowledge_total()}")
