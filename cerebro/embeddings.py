import torch
import torch.nn.functional as F
import numpy as np
import os
from sentence_transformers import SentenceTransformer


_modelo_compartilhado = None


def _get_modelo():
    global _modelo_compartilhado
    if _modelo_compartilhado is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _modelo_compartilhado = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ).to(device)
    return _modelo_compartilhado


class GeradorEmbeddings:
    def __init__(self, modelo_nome='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.modelo = SentenceTransformer(modelo_nome).to(self.device)

    def gerar_vetor(self, texto):
        vetor = self.modelo.encode(texto, convert_to_tensor=True, device=self.device)
        return vetor.cpu().tolist()


class IndiceSemantico:
    """Índice de busca semântica por cosine similarity com cache em disco."""

    def __init__(self, cache_path=None):
        self.cache_path = cache_path
        self.modelo = _get_modelo()
        self.device = next(self.modelo.parameters()).device
        self.perguntas = []
        self.respostas = []
        self.embeddings = None  # tensor (N, dim)

        if cache_path and os.path.exists(cache_path):
            try:
                dados = np.load(cache_path, allow_pickle=True).item()
                self.perguntas = dados['perguntas']
                self.respostas = dados['respostas']
                self.embeddings = torch.tensor(dados['embeddings'], device=self.device)
            except Exception:
                pass

    def _salvar_cache(self):
        if self.cache_path and self.embeddings is not None:
            dados = {
                'perguntas': self.perguntas,
                'respostas': self.respostas,
                'embeddings': self.embeddings.cpu().numpy(),
            }
            np.save(self.cache_path, dados)

    def construir(self, perguntas, respostas):
        """Constrói índice a partir de listas de perguntas e respostas."""
        self.perguntas = list(perguntas)
        self.respostas = list(respostas)
        if not perguntas:
            self.embeddings = None
            return
        self.embeddings = self.modelo.encode(
            self.perguntas, convert_to_tensor=True, device=self.device,
            show_progress_bar=False, batch_size=64
        )
        self._salvar_cache()

    def adicionar(self, pergunta, resposta):
        """Adiciona um par ao índice."""
        self.perguntas.append(pergunta)
        self.respostas.append(resposta)
        novo_emb = self.modelo.encode(
            [pergunta], convert_to_tensor=True, device=self.device,
            show_progress_bar=False
        )
        if self.embeddings is None:
            self.embeddings = novo_emb
        else:
            self.embeddings = torch.cat([self.embeddings, novo_emb], dim=0)
        self._salvar_cache()

    def buscar(self, pergunta, top_k=3):
        """Busca os top_k mais similares. Retorna [(pergunta, resposta, score), ...]"""
        if self.embeddings is None or len(self.perguntas) == 0:
            return []
        query_emb = self.modelo.encode(
            [pergunta], convert_to_tensor=True, device=self.device,
            show_progress_bar=False
        )
        sims = F.cosine_similarity(query_emb, self.embeddings)
        k = min(top_k, len(self.perguntas))
        valores, indices = torch.topk(sims, k)
        resultados = []
        for score, idx in zip(valores.tolist(), indices.tolist()):
            resultados.append((self.perguntas[idx], self.respostas[idx], score))
        return resultados