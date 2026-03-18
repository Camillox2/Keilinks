"""
Crawler de Livros da Keilinks
Busca livros gratuitos sobre conversacao, psicologia e autoajuda.
Fontes:
  - Project Gutenberg (livros inteiros em texto, dominio publico)
  - Open Library (resumos e descricoes de livros modernos)
  - Google Books (previews/snippets)
Salva conteudo no MySQL como knowledge para treinar o modelo.
"""

import os
import sys
import re
import time
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dados.database import knowledge_adicionar, knowledge_existe, knowledge_total

HEADERS = {'User-Agent': 'Keilinks/2.0 (IA pessoal educacional)'}


# ─── Livros para buscar ──────────────────────────────────────────────────

# Termos de busca para Project Gutenberg (dominio publico)
GUTENBERG_BUSCAS = [
    'psychology',
    'conversation',
    'self-help',
    'human nature',
    'social',
    'communication',
    'philosophy of mind',
    'emotions',
    'education',
    'rhetoric',
]

# Termos para Open Library (livros modernos - pega resumos)
OPENLIB_BUSCAS = [
    'como conversar com pessoas',
    'inteligencia emocional',
    'comunicacao interpessoal',
    'autoajuda conversacao',
    'psicologia comportamental',
    'linguagem corporal',
    'escuta ativa',
    'empatia comunicacao',
    'habilidades sociais',
    'desenvolvimento pessoal',
    'como fazer amigos',
    'arte de conversar',
    'persuasao comunicacao',
    'assertividade',
]

# Termos para Google Books (snippets em PT)
GBOOKS_BUSCAS = [
    'como conversar melhor',
    'tecnicas de conversacao',
    'inteligencia emocional',
    'comunicacao nao violenta',
    'como ser mais social',
    'psicologia da comunicacao',
    'autoajuda relacionamentos',
    'como ouvir pessoas',
    'empatia no dia a dia',
    'habilidades interpessoais',
]


def _log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [Livros] {msg}")


def _limpar_texto(texto: str) -> str:
    """Limpa texto de livro removendo formatacao excessiva"""
    # Remove linhas em branco excessivas
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    # Remove cabecalhos do Gutenberg
    texto = re.sub(r'\*\*\*.*?\*\*\*', '', texto)
    return texto.strip()


def _dividir_em_trechos(texto: str, tamanho: int = 500) -> list[str]:
    """Divide texto longo em trechos de ~500 chars por paragrafo"""
    paragrafos = texto.split('\n\n')
    trechos = []
    trecho_atual = ''

    for p in paragrafos:
        p = p.strip()
        if not p or len(p) < 30:
            continue
        if len(trecho_atual) + len(p) > tamanho:
            if trecho_atual:
                trechos.append(trecho_atual.strip())
            trecho_atual = p
        else:
            trecho_atual += '\n\n' + p if trecho_atual else p

    if trecho_atual and len(trecho_atual) >= 50:
        trechos.append(trecho_atual.strip())

    return trechos


# ─── PROJECT GUTENBERG ───────────────────────────────────────────────────

def buscar_gutenberg(termo: str, limite: int = 3) -> list[dict]:
    """Busca livros no Project Gutenberg via Gutendex API"""
    try:
        url = "https://gutendex.com/books/"
        params = {
            'search': termo,
            'languages': 'pt,en',
            'mime_type': 'text/plain',
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        livros = resp.json().get('results', [])[:limite]
        resultados = []

        for livro in livros:
            titulo = livro.get('title', '')
            autores = ', '.join(a.get('name', '') for a in livro.get('authors', []))

            # Pega URL do texto plano
            formats = livro.get('formats', {})
            txt_url = None
            for fmt, url in formats.items():
                if 'text/plain' in fmt and '.txt' in url:
                    txt_url = url
                    break

            if not txt_url:
                # Tenta UTF-8
                for fmt, url in formats.items():
                    if 'text/plain' in fmt:
                        txt_url = url
                        break

            if titulo and txt_url:
                resultados.append({
                    'titulo': titulo,
                    'autor': autores,
                    'url': txt_url,
                    'id': livro.get('id', 0),
                })

        return resultados
    except Exception as e:
        _log(f"Erro Gutenberg busca: {e}")
        return []


def baixar_texto_gutenberg(url: str, max_chars: int = 50000) -> str:
    """Baixa texto de um livro do Gutenberg"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return ''
        texto = resp.text[:max_chars]
        return _limpar_texto(texto)
    except Exception:
        return ''


def crawl_gutenberg() -> int:
    """Crawla livros do Project Gutenberg e salva trechos no MySQL"""
    novos = 0

    for termo in GUTENBERG_BUSCAS:
        livros = buscar_gutenberg(termo, limite=2)

        for livro in livros:
            # Checa se ja processou esse livro
            chave = f"livro: {livro['titulo']}"
            if knowledge_existe(chave):
                continue

            _log(f"Baixando: {livro['titulo'][:50]}... ({livro['autor']})")
            texto = baixar_texto_gutenberg(livro['url'])

            if len(texto) < 200:
                continue

            # Divide em trechos e salva cada um
            trechos = _dividir_em_trechos(texto, tamanho=600)

            # Salva metadado do livro
            knowledge_adicionar(
                chave,
                f"Livro: {livro['titulo']} por {livro['autor']}. Conteudo sobre {termo}.",
                'web', 'geral', livro['url'],
            )

            # Salva ate 20 trechos por livro (os mais relevantes)
            salvos = 0
            for trecho in trechos[:20]:
                if len(trecho) < 50:
                    continue
                pergunta = f"trecho do livro {livro['titulo'][:60]}"
                if not knowledge_existe(pergunta):
                    knowledge_adicionar(
                        pergunta, trecho,
                        'web', 'geral', livro['url'],
                    )
                    salvos += 1

            novos += salvos
            _log(f"  +{salvos} trechos salvos de '{livro['titulo'][:40]}'")
            time.sleep(1)

        time.sleep(0.5)

    return novos


# ─── OPEN LIBRARY ────────────────────────────────────────────────────────

def buscar_open_library(termo: str, limite: int = 5) -> list[dict]:
    """Busca livros na Open Library e pega descricoes"""
    try:
        url = "https://openlibrary.org/search.json"
        params = {'q': termo, 'limit': limite, 'language': 'por'}
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        docs = resp.json().get('docs', [])
        resultados = []

        for doc in docs:
            titulo = doc.get('title', '')
            autor = ', '.join(doc.get('author_name', [])[:2])
            descricao = ''

            # Tenta pegar descricao detalhada
            key = doc.get('key', '')
            if key:
                try:
                    detail_resp = requests.get(
                        f"https://openlibrary.org{key}.json",
                        headers=HEADERS, timeout=8
                    )
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()
                        desc = detail.get('description', '')
                        if isinstance(desc, dict):
                            desc = desc.get('value', '')
                        descricao = desc[:800] if desc else ''
                except Exception:
                    pass

            # Fallback: primeira frase do titulo
            if not descricao:
                first_sentence = doc.get('first_sentence', [''])
                if isinstance(first_sentence, list) and first_sentence:
                    descricao = first_sentence[0] if isinstance(first_sentence[0], str) else ''

            if titulo and descricao and len(descricao) > 30:
                resultados.append({
                    'titulo': titulo,
                    'autor': autor,
                    'descricao': descricao,
                })

        return resultados
    except Exception as e:
        _log(f"Erro Open Library: {e}")
        return []


def crawl_open_library() -> int:
    """Crawla resumos de livros da Open Library"""
    novos = 0

    for termo in OPENLIB_BUSCAS:
        livros = buscar_open_library(termo, limite=3)

        for livro in livros:
            chave = f"sobre o livro {livro['titulo'].lower()}"
            if knowledge_existe(chave):
                continue

            texto = f"{livro['descricao']}"
            if livro['autor']:
                texto = f"Livro de {livro['autor']}. {texto}"

            knowledge_adicionar(
                chave, texto,
                'web', 'geral',
            )
            novos += 1

        time.sleep(0.5)

    if novos > 0:
        _log(f"Open Library: +{novos} resumos de livros")
    return novos


# ─── GOOGLE BOOKS ────────────────────────────────────────────────────────

def buscar_google_books(termo: str, limite: int = 5) -> list[dict]:
    """Busca livros no Google Books API (gratis, sem key)"""
    try:
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            'q': termo,
            'langRestrict': 'pt',
            'maxResults': limite,
            'printType': 'books',
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []

        items = resp.json().get('items', [])
        resultados = []

        for item in items:
            info = item.get('volumeInfo', {})
            titulo = info.get('title', '')
            autores = ', '.join(info.get('authors', [])[:2])
            descricao = info.get('description', '')

            if titulo and descricao and len(descricao) > 50:
                resultados.append({
                    'titulo': titulo,
                    'autor': autores,
                    'descricao': descricao[:800],
                    'url': info.get('infoLink', ''),
                })

        return resultados
    except Exception as e:
        _log(f"Erro Google Books: {e}")
        return []


def crawl_google_books() -> int:
    """Crawla descricoes de livros do Google Books em PT"""
    novos = 0

    for termo in GBOOKS_BUSCAS:
        livros = buscar_google_books(termo, limite=3)

        for livro in livros:
            chave = f"livro {livro['titulo'].lower()[:80]}"
            if knowledge_existe(chave):
                continue

            texto = livro['descricao']
            if livro['autor']:
                texto = f"Livro de {livro['autor']}. {texto}"

            knowledge_adicionar(
                chave, texto,
                'web', 'geral', livro.get('url'),
            )
            novos += 1

        time.sleep(0.5)

    if novos > 0:
        _log(f"Google Books: +{novos} descricoes de livros")
    return novos


# ─── CRAWL COMPLETO DE LIVROS ────────────────────────────────────────────

def crawl_livros_completo() -> int:
    """Executa crawl completo de todas as fontes de livros"""
    total = 0

    _log("Iniciando busca de livros...")

    # Google Books (mais rapido, descricoes em PT)
    try:
        total += crawl_google_books()
    except Exception as e:
        _log(f"Erro Google Books: {e}")

    # Open Library (resumos detalhados)
    try:
        total += crawl_open_library()
    except Exception as e:
        _log(f"Erro Open Library: {e}")

    # Project Gutenberg (livros inteiros - mais lento)
    try:
        total += crawl_gutenberg()
    except Exception as e:
        _log(f"Erro Gutenberg: {e}")

    _log(f"Livros concluido: +{total} novos conteudos | Total knowledge: {knowledge_total()}")
    return total


if __name__ == '__main__':
    print("Buscando livros sobre conversacao, autoajuda e psicologia...")
    print("Fontes: Google Books + Open Library + Project Gutenberg\n")
    novos = crawl_livros_completo()
    print(f"\nConcluido: {novos} novos conteudos salvos no MySQL")
    print(f"Total knowledge: {knowledge_total()}")
