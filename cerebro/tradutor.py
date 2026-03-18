"""
Tradutor EN>>PT da Keilinks
Usa MyMemory API (gratis) para traduzir conteudo em ingles para portugues.
Cache local para nao traduzir a mesma coisa duas vezes.
"""

import requests
import hashlib
import json
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, 'dados', 'cache_traducoes.json')

_cache = {}


def _carregar_cache():
    global _cache
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                _cache = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            _cache = {}


def _salvar_cache():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    # Limita cache a 500 entradas
    if len(_cache) > 500:
        items = sorted(_cache.items(), key=lambda x: x[1].get('ts', ''), reverse=True)
        _cache.clear()
        _cache.update(dict(items[:500]))
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(_cache, f, ensure_ascii=False, indent=2)


def _hash_texto(texto: str) -> str:
    return hashlib.md5(texto.encode('utf-8')).hexdigest()[:12]


def _detectar_idioma(texto: str) -> str:
    """Detecta se o texto e ingles ou portugues (heuristica simples)"""
    palavras_en = {'the', 'is', 'are', 'was', 'were', 'have', 'has', 'been',
                   'will', 'would', 'could', 'should', 'can', 'may', 'might',
                   'this', 'that', 'these', 'those', 'with', 'from', 'for',
                   'and', 'but', 'not', 'you', 'all', 'they', 'their',
                   'what', 'which', 'who', 'when', 'where', 'how', 'why',
                   'each', 'every', 'both', 'few', 'more', 'most', 'other',
                   'some', 'such', 'than', 'too', 'very', 'just', 'about'}
    palavras_pt = {'que', 'nao', 'uma', 'com', 'para', 'por', 'mais', 'como',
                   'mas', 'foi', 'bem', 'sem', 'nos', 'ele', 'ela', 'isso',
                   'esta', 'esse', 'essa', 'tem', 'ser', 'ter', 'seu', 'sua',
                   'dos', 'das', 'nos', 'nas', 'pelo', 'pela', 'entre',
                   'sobre', 'depois', 'antes', 'muito', 'tambem', 'quando',
                   'onde', 'quem', 'qual', 'ainda', 'outro', 'outra'}

    words = set(re.findall(r'[a-z]+', texto.lower()))
    en_count = len(words & palavras_en)
    pt_count = len(words & palavras_pt)

    if en_count > pt_count and en_count >= 1:
        return 'en'
    if en_count >= 3:
        return 'en'
    return 'pt'


def traduzir(texto: str, de: str = 'en', para: str = 'pt-br') -> str:
    """
    Traduz texto usando MyMemory API.
    Retorna texto traduzido ou original se falhar.
    """
    if not texto or len(texto.strip()) < 5:
        return texto

    # Checa cache
    chave = _hash_texto(texto)
    if chave in _cache:
        return _cache[chave].get('traducao', texto)

    try:
        # MyMemory API - gratis, sem key
        # Limita a 500 chars por request (limite da API gratis)
        texto_cortado = texto[:500]
        url = "https://api.mymemory.translated.net/get"
        params = {
            'q': texto_cortado,
            'langpair': f'{de}|{para}',
        }
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return texto

        data = resp.json()
        traducao = data.get('responseData', {}).get('translatedText', '')

        if not traducao or traducao.upper() == texto_cortado.upper():
            return texto

        # Salva no cache
        _cache[chave] = {
            'original': texto_cortado,
            'traducao': traducao,
            'ts': datetime.now().isoformat(),
        }
        _salvar_cache()

        return traducao

    except Exception:
        return texto


def traduzir_se_ingles(texto: str) -> tuple[str, str]:
    """
    Detecta idioma e traduz se for ingles.
    Retorna (texto_final, idioma_original)
    """
    idioma = _detectar_idioma(texto)
    if idioma == 'en':
        traduzido = traduzir(texto)
        return traduzido, 'en'
    return texto, 'pt'


# Carrega cache ao importar
_carregar_cache()


if __name__ == '__main__':
    testes = [
        "Python is a high-level programming language",
        "What is machine learning and how does it work",
        "PyTorch e uma biblioteca de deep learning",
        "The transformer architecture uses self-attention mechanisms",
    ]
    for t in testes:
        resultado, idioma = traduzir_se_ingles(t)
        print(f"[{idioma}] {t[:50]}...")
        print(f"  >> {resultado[:80]}...")
        print()
