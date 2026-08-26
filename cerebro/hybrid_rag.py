"""Fachada legada de RAG sem disputar VRAM com a geração.

O RAG de produção está em :mod:`keilinks_v5.rag`: SQLite FTS5, proveniência,
isolamento por tenant e embeddings opcionalmente em CPU. Estas classes mantêm
apenas a API usada por scripts V4 antigos e não carregam modelos remotos nem
movem embeddings para a GPU automaticamente.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

WORD_RE = re.compile(r"[\wÀ-ÿ]{2,}", re.UNICODE)
STOPWORDS_PT = {
    "a",
    "o",
    "as",
    "os",
    "um",
    "uma",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "que",
    "se",
    "eu",
    "você",
    "voce",
    "ele",
    "ela",
    "eles",
    "elas",
    "meu",
    "minha",
    "seu",
    "sua",
    "nosso",
    "nossa",
    "e",
    "ou",
    "mas",
    "como",
    "quando",
    "onde",
    "porque",
    "qual",
    "quais",
    "quem",
}


def tokenize_pt(text: str) -> list[str]:
    return [token for token in WORD_RE.findall(text.casefold()) if token not in STOPWORDS_PT]


class BM25Retriever:
    """BM25 pequeno e determinístico para compatibilidade e testes."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.average_length = 1.0
        self.idf: dict[str, float] = {}

    def index(self, docs: list[str]) -> None:
        self.docs = list(docs)
        self.doc_tokens = [tokenize_pt(doc) for doc in self.docs]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))
        total = len(self.docs)
        self.idf = {
            token: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if top_k < 1:
            return []
        scores = [0.0] * len(self.docs)
        for token in tokenize_pt(query):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, tokens in enumerate(self.doc_tokens):
                frequency = tokens.count(token)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[index] / self.average_length
                )
                scores[index] += idf * frequency * (self.k1 + 1) / denominator
        return [
            (index, score)
            for index, score in sorted(enumerate(scores), key=lambda row: (-row[1], row[0]))[:top_k]
            if score > 0
        ]


class HybridRAG:
    """Compatibilidade lexical. Migre para ``LocalKnowledgeStore`` na API V6."""

    def __init__(self, cache_path: str | None = None, **_: Any) -> None:
        self.cache_path = cache_path
        self.documents: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.bm25 = BM25Retriever()

    def add_documents(self, docs: list[str], metas: list[dict[str, Any]] | None = None) -> None:
        self.documents.extend(str(doc).strip() for doc in docs if str(doc).strip())
        if metas:
            self.metadata.extend(metas[: len(docs)])
        while len(self.metadata) < len(self.documents):
            self.metadata.append({})
        self.bm25.index(self.documents)

    def search(self, query: str, top_k: int = 5, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "text": self.documents[index],
                "score": round(score, 8),
                "metadata": self.metadata[index],
            }
            for index, score in self.bm25.search(query, top_k=top_k)
        ]
