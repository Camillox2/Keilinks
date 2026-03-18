"""
Normalizador de texto da Keilinks
Corrige erros de digitacao, expande abreviacoes, normaliza antes de buscar.
"""

import re
import unicodedata
from difflib import get_close_matches


# Abreviacoes comuns em PT-BR
ABREVIACOES = {
    'oq': 'o que',
    'oqe': 'o que',
    'oque': 'o que',
    'oq e': 'o que e',
    'vc': 'voce',
    'vcs': 'voces',
    'tb': 'tambem',
    'tbm': 'tambem',
    'pq': 'por que',
    'prq': 'por que',
    'cmg': 'comigo',
    'ctg': 'contigo',
    'msg': 'mensagem',
    'blz': 'beleza',
    'flw': 'falou',
    'vlw': 'valeu',
    'obg': 'obrigado',
    'pfv': 'por favor',
    'pfvr': 'por favor',
    'qnd': 'quando',
    'qdo': 'quando',
    'td': 'tudo',
    'tds': 'todos',
    'mt': 'muito',
    'mto': 'muito',
    'msm': 'mesmo',
    'ngm': 'ninguem',
    'alg': 'alguem',
    'algm': 'alguem',
    'dps': 'depois',
    'hj': 'hoje',
    'hr': 'hora',
    'hrs': 'horas',
    'min': 'minutos',
    'seg': 'segundos',
    'qts': 'quantos',
    'qnts': 'quantos',
    'nd': 'nada',
    'dnv': 'de novo',
    'n': 'nao',
    'to': 'estou',
    'ta': 'esta',
    'eh': 'e',
    'aki': 'aqui',
    'agr': 'agora',
    'bjs': 'beijos',
    'abs': 'abracos',
    'tmj': 'estamos juntos',
    'slk': 'se liga',
    'mlk': 'moleque',
    'mds': 'meu deus',
    'krl': 'caramba',
    'pdc': 'pode crer',
    'fds': 'fim de semana',
    'sdds': 'saudades',
    'cmr': 'comere',
    'bora': 'vamos',
    'trm': 'treinar',
}

# Dicionario de termos tech conhecidos (para correcao de typos)
TERMOS_CONHECIDOS = [
    'python', 'javascript', 'typescript', 'react', 'node', 'nodejs',
    'pytorch', 'tensorflow', 'flask', 'django', 'fastapi',
    'html', 'css', 'sql', 'mysql', 'postgresql', 'mongodb',
    'docker', 'kubernetes', 'linux', 'windows', 'macos',
    'git', 'github', 'gitlab', 'vscode', 'vim',
    'api', 'rest', 'graphql', 'json', 'xml',
    'machine', 'learning', 'deep', 'neural', 'rede',
    'inteligencia', 'artificial', 'modelo', 'treino', 'treinamento',
    'cuda', 'gpu', 'cpu', 'ram', 'vram', 'memoria',
    'transformer', 'attention', 'embedding', 'tokenizador',
    'vasco', 'futebol', 'campeonato', 'brasileiro', 'copa',
    'curitiba', 'brasil', 'parana',
    'programacao', 'programa', 'codigo', 'funcao', 'classe',
    'variavel', 'loop', 'array', 'lista', 'dicionario',
    'servidor', 'banco', 'dados', 'tabela', 'coluna',
    'frontend', 'backend', 'fullstack', 'devops',
    'keilinks', 'vitor', 'keila',
]


def remover_acentos(texto: str) -> str:
    """Remove acentos: programação → programacao"""
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def expandir_abreviacoes(texto: str) -> str:
    """Expande abreviacoes: 'oq e python' → 'o que e python'"""
    palavras = texto.split()
    resultado = []
    i = 0
    while i < len(palavras):
        # Tenta combinar 2 palavras primeiro
        if i + 1 < len(palavras):
            par = f"{palavras[i]} {palavras[i+1]}"
            if par in ABREVIACOES:
                resultado.append(ABREVIACOES[par])
                i += 2
                continue

        palavra = palavras[i]
        if palavra in ABREVIACOES:
            resultado.append(ABREVIACOES[palavra])
        else:
            resultado.append(palavra)
        i += 1

    return ' '.join(resultado)


def corrigir_typos(texto: str) -> str:
    """Corrige erros de digitacao usando difflib"""
    palavras = texto.split()
    resultado = []
    for palavra in palavras:
        if len(palavra) < 3:
            resultado.append(palavra)
            continue
        # Se a palavra ja esta no dicionario, nao mexe
        if palavra.lower() in TERMOS_CONHECIDOS:
            resultado.append(palavra)
            continue
        # Tenta achar match proximo
        matches = get_close_matches(palavra.lower(), TERMOS_CONHECIDOS, n=1, cutoff=0.7)
        if matches:
            resultado.append(matches[0])
        else:
            resultado.append(palavra)
    return ' '.join(resultado)


def levenshtein(s1: str, s2: str) -> int:
    """Distancia de Levenshtein entre duas strings"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def palavras_similares(p1: str, p2: str, max_dist: int = 2) -> bool:
    """Verifica se duas palavras sao similares (tolerancia a typos)"""
    if p1 == p2:
        return True
    if abs(len(p1) - len(p2)) > max_dist:
        return False
    return levenshtein(p1, p2) <= max_dist


def normalizar(texto: str) -> str:
    """
    Pipeline completo de normalizacao:
    1. Lowercase
    2. Remove acentos
    3. Expande abreviacoes
    4. Corrige typos
    """
    texto = texto.lower().strip()
    texto = remover_acentos(texto)
    texto = expandir_abreviacoes(texto)
    texto = corrigir_typos(texto)
    return texto


if __name__ == '__main__':
    testes = [
        "oq e pyton",
        "vc sabe javasript",
        "me fala sobre pytorh",
        "oque e react",
        "oq e inteligencia artificial",
        "pq o vasco perdeu",
        "como funciona o tensoflow",
    ]
    for t in testes:
        print(f"  {t:40s} >> {normalizar(t)}")
