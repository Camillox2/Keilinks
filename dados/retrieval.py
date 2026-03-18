"""
Retrieval v3: busca por palavras com fuzzy matching
Tolera erros de digitacao: "pyton" casa com "python"
"""

import os
import re
from cerebro.normalizador import levenshtein


def _palavras(texto: str) -> set:
    """Extrai palavras normalizadas (com acentos)"""
    return set(re.findall(r'[a-z\u00e0-\u00ff]+', texto.lower()))


def _similaridade_fuzzy(a: str, b: str) -> float:
    """Similaridade Jaccard com fuzzy matching por Levenshtein"""
    pa, pb = _palavras(a), _palavras(b)
    if not pa or not pb:
        return 0.0

    # Match fuzzy: conta palavras que sao iguais OU parecidas (dist <= 2)
    matches = 0
    for wa in pa:
        if len(wa) <= 2:
            # Palavras curtas: so match exato
            if wa in pb:
                matches += 1
            continue
        for wb in pb:
            if wa == wb:
                matches += 1
                break
            if len(wb) >= 3 and abs(len(wa) - len(wb)) <= 2:
                if levenshtein(wa, wb) <= 2:
                    matches += 1
                    break

    uniao = len(pa | pb)
    jaccard = matches / uniao if uniao > 0 else 0

    # Bonus de cobertura
    cobertura_a = matches / len(pa) if pa else 0
    cobertura_b = matches / len(pb) if pb else 0
    bonus = max(cobertura_a, cobertura_b) * 0.3

    return min(jaccard + bonus, 1.0)


class Retrieval:
    def __init__(self):
        self.pares = []

    def carregar(self, *arquivos):
        self.pares = []
        visto = set()
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
                    # Deduplicacao: evita pares repetidos
                    chave = pergunta.lower().strip()
                    if chave not in visto:
                        visto.add(chave)
                        self.pares.append((pergunta, resposta))
        print(f"[Retrieval] {len(self.pares)} pares carregados (deduplicados)")

    def buscar(self, pergunta: str, limiar=0.20) -> tuple[str | None, float]:
        """Retorna (resposta, score). Score 0-1. Usa fuzzy matching."""
        if not self.pares:
            return None, 0.0
        melhor_sim = 0.0
        melhor_resp = None
        for p, r in self.pares:
            sim = _similaridade_fuzzy(pergunta, p)
            if sim > melhor_sim:
                melhor_sim = sim
                melhor_resp = r
        if melhor_sim >= limiar:
            return melhor_resp, melhor_sim
        return None, melhor_sim

    def adicionar(self, pergunta: str, resposta: str):
        # Evita duplicatas ao adicionar
        chave = pergunta.lower().strip()
        for p, _ in self.pares:
            if p.lower().strip() == chave:
                return
        self.pares.append((pergunta, resposta))
