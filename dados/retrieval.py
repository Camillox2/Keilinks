"""
Retrieval v2: busca por palavras (muito melhor que n-grams de caracteres)
"sou seu criador" agora casa com "quem te criou"
"""

import os
import re


def _palavras(texto: str) -> set:
    """Extrai palavras normalizadas"""
    return set(re.findall(r'[a-záàâãéèêíìîóòôõúùûç]+', texto.lower()))


def _similaridade(a: str, b: str) -> float:
    """Similaridade Jaccard por palavras"""
    pa, pb = _palavras(a), _palavras(b)
    if not pa or not pb:
        return 0.0
    # Palavras em comum / total de palavras únicas
    intersecao = pa & pb
    uniao = pa | pb
    jaccard = len(intersecao) / len(uniao)

    # Bonus: se uma contém a outra quase toda
    cobertura_a = len(intersecao) / len(pa) if pa else 0
    cobertura_b = len(intersecao) / len(pb) if pb else 0
    bonus = max(cobertura_a, cobertura_b) * 0.3

    return min(jaccard + bonus, 1.0)


class Retrieval:
    def __init__(self):
        self.pares = []

    def carregar(self, *arquivos):
        self.pares = []
        for arq in arquivos:
            if not os.path.exists(arq):
                continue
            with open(arq, 'r', encoding='utf-8') as f:
                texto = f.read()
            for bloco in texto.split('<vitor>'):
                if '<fim><keilinks>' not in bloco:
                    continue
                partes = bloco.split('<fim><keilinks>')
                pergunta = partes[0].strip()
                resposta = partes[1].split('<fim>')[0].strip() if len(partes) > 1 else ''
                if pergunta and resposta:
                    self.pares.append((pergunta, resposta))
        print(f"[Retrieval] {len(self.pares)} pares carregados")

    def buscar(self, pergunta: str, limiar=0.20) -> tuple[str | None, float]:
        """Retorna (resposta, score). Score 0-1."""
        if not self.pares:
            return None, 0.0
        melhor_sim = 0.0
        melhor_resp = None
        for p, r in self.pares:
            sim = _similaridade(pergunta, p)
            if sim > melhor_sim:
                melhor_sim = sim
                melhor_resp = r
        if melhor_sim >= limiar:
            return melhor_resp, melhor_sim
        return None, melhor_sim

    def adicionar(self, pergunta: str, resposta: str):
        self.pares.append((pergunta, resposta))
