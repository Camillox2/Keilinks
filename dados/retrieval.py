"""
Retrieval v4: busca semantica + fuzzy fallback
Usa sentence-transformers para entender significado
Fallback para Levenshtein se embeddings nao disponivel
"""

import os
import re
from cerebro.normalizador import levenshtein


def _palavras(texto: str) -> set:
    """Extrai palavras normalizadas"""
    return set(re.findall(r'[a-z\u00e0-\u00ff]+', texto.lower()))


def _similaridade_fuzzy(a: str, b: str) -> float:
    """Similaridade Jaccard com fuzzy matching (fallback)"""
    pa, pb = _palavras(a), _palavras(b)
    if not pa or not pb:
        return 0.0

    matches = 0
    for wa in pa:
        if len(wa) <= 2:
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
    cobertura_a = matches / len(pa) if pa else 0
    cobertura_b = matches / len(pb) if pb else 0
    bonus = max(cobertura_a, cobertura_b) * 0.3
    return min(jaccard + bonus, 1.0)


class Retrieval:
    def __init__(self):
        self.pares = []
        self._indice_semantico = None
        self._embeddings_ok = False

    def _iniciar_embeddings(self):
        """Inicializa busca semantica (lazy)"""
        if self._indice_semantico is not None:
            return

        try:
            from cerebro.embeddings import IndiceSemantico

            cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'retrieval_cache.npy')
            self._indice_semantico = IndiceSemantico(cache_path=cache)

            perguntas = [p for p, _ in self.pares]
            respostas = [r for _, r in self.pares]
            self._indice_semantico.construir(perguntas, respostas)
            self._embeddings_ok = True
        except Exception as e:
            print(f"[Retrieval] Embeddings indisponivel: {e}")
            self._embeddings_ok = False

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
                    chave = pergunta.lower().strip()
                    if chave not in visto:
                        visto.add(chave)
                        self.pares.append((pergunta, resposta))
        print(f"[Retrieval] {len(self.pares)} pares carregados (deduplicados)")

        # Reinicializa embeddings com novos dados
        self._indice_semantico = None
        self._embeddings_ok = False
        self._iniciar_embeddings()

    def buscar(self, pergunta: str, limiar=0.20) -> tuple[str | None, float]:
        """Busca semantica com fallback fuzzy. Retorna (resposta, score)."""
        if not self.pares:
            return None, 0.0

        # Tenta busca semantica primeiro
        if self._embeddings_ok and self._indice_semantico:
            try:
                resultados = self._indice_semantico.buscar(pergunta, top_k=1)
                if resultados:
                    texto, resposta, score = resultados[0]
                    if score >= limiar:
                        return resposta, score
            except Exception:
                pass

        # Fallback: busca fuzzy
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
        chave = pergunta.lower().strip()
        for p, _ in self.pares:
            if p.lower().strip() == chave:
                return
        self.pares.append((pergunta, resposta))

        # Atualiza indice semantico
        if self._embeddings_ok and self._indice_semantico:
            try:
                self._indice_semantico.adicionar(pergunta, resposta)
            except Exception:
                pass
