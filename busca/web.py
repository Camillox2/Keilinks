"""
Módulo de busca web da Keilinks
Quando ela não sabe responder, busca na internet e traz o resultado
como contexto extra para a resposta.
"""

import requests
from bs4 import BeautifulSoup
import re


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def buscar_duckduckgo(query: str, max_resultados: int = 3) -> list[dict]:
    """Busca no DuckDuckGo e retorna os primeiros resultados"""
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}"
        resp = requests.get(url, headers=HEADERS, timeout=6)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        resultados = []

        # Extrai snippets da página lite do DuckDuckGo
        for tr in soup.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) >= 2:
                link_tag = tds[0].find('a')
                snippet = tds[1].get_text(strip=True) if len(tds) > 1 else ''
                if link_tag and snippet:
                    resultados.append({
                        'titulo': link_tag.get_text(strip=True),
                        'link':   link_tag.get('href', ''),
                        'texto':  snippet
                    })
                    if len(resultados) >= max_resultados:
                        break

        return resultados
    except Exception as e:
        return []


def buscar_wikipedia(query: str) -> str:
    """
    Busca na Wikipedia em português usando o endpoint de busca.
    Primeiro encontra o artigo mais relevante, depois pega o resumo.
    """
    try:
        # Passo 1: busca pelo título mais relevante
        search_url = "https://pt.wikipedia.org/w/api.php"
        params = {
            'action': 'query', 'list': 'search',
            'srsearch': query, 'srlimit': 1, 'format': 'json'
        }
        resp = requests.get(search_url, params=params, headers=HEADERS, timeout=6)
        if resp.status_code != 200:
            return ''

        resultados = resp.json().get('query', {}).get('search', [])
        if not resultados:
            return ''

        titulo = resultados[0]['title']

        # Passo 2: pega o resumo do artigo encontrado
        summary_url = "https://pt.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(titulo)
        resp2 = requests.get(summary_url, headers=HEADERS, timeout=6)
        if resp2.status_code == 200:
            return resp2.json().get('extract', '')[:600]
    except Exception:
        pass
    return ''


def _limpar(texto: str) -> str:
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto[:400]


def pesquisar(pergunta: str) -> str:
    """
    Função principal chamada pela Keilinks.
    Tenta Wikipedia primeiro (mais confiável), depois DuckDuckGo.
    Retorna um resumo do que encontrou ou string vazia se nada.
    """
    # Tenta Wikipedia
    wiki = buscar_wikipedia(pergunta)
    if wiki and len(wiki) > 80:
        return f"[Fonte: Wikipedia]\n{_limpar(wiki)}"

    # Tenta DuckDuckGo
    resultados = buscar_duckduckgo(pergunta)
    if resultados:
        partes = []
        for r in resultados[:2]:
            if r['texto']:
                partes.append(f"• {r['titulo']}: {_limpar(r['texto'])}")
        if partes:
            return "[Fonte: Web]\n" + "\n".join(partes)

    return ""


def precisa_buscar(mensagem: str) -> bool:
    """
    Detecta se a pergunta provavelmente precisa de busca na web.
    Palavras-chave que indicam perguntas factuais ou atuais.
    """
    gatilhos = [
        'quem é', 'o que é', 'o que foi', 'quando foi', 'quando é',
        'onde fica', 'onde é', 'como funciona', 'qual é a capital',
        'quantos', 'quanto custa', 'preço', 'notícia', 'hoje',
        'agora', 'atual', 'recente', 'último', 'última',
        'jogo', 'resultado', 'placar', 'clima', 'temperatura',
        'quem ganhou', 'quem venceu', 'campeonato', 'copa',
        'filme', 'série', 'lançamento', 'estreia',
    ]
    msg = mensagem.lower()
    return any(g in msg for g in gatilhos)
