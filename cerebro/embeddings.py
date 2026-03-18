"""
Embeddings semanticos da Keilinks
Busca por significado ao inves de palavras exatas
Usa sentence-transformers com paraphrase-multilingual-MiniLM-L12-v2 (~80MB)
"""

import os
import numpy as np

_modelo = None
_device = None


def _carregar_modelo():
    """Carrega modelo de embeddings (lazy load)"""
    global _modelo, _device
    if _modelo is not None:
        return _modelo

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=_device)
        print(f"[Embeddings] paraphrase-multilingual-MiniLM-L12-v2 carregado ({_device})")
        return _modelo
    except ImportError:
        print("[Embeddings] sentence-transformers nao instalado. pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"[Embeddings] Erro ao carregar modelo: {e}")
        return None


def gerar_embeddings(textos, batch_size=64):
    """Gera embeddings para lista de textos"""
    modelo = _carregar_modelo()
    if modelo is None:
        return None
    return modelo.encode(textos, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)


def similaridade_cosseno(emb_query, emb_base):
    """Calcula similaridade de cosseno entre query e base de embeddings"""
    # emb_query: (dim,) ou (1, dim)
    # emb_base: (n, dim)
    if emb_query.ndim == 1:
        emb_query = emb_query.reshape(1, -1)

    # Normaliza (sentence-transformers ja normaliza, mas garante)
    norma_q = np.linalg.norm(emb_query, axis=1, keepdims=True)
    norma_b = np.linalg.norm(emb_base, axis=1, keepdims=True)

    if np.any(norma_q == 0) or np.any(norma_b == 0):
        return np.zeros(emb_base.shape[0])

    emb_query = emb_query / norma_q
    emb_base = emb_base / norma_b

    return (emb_query @ emb_base.T).flatten()


class IndiceSemantico:
    """Indice de busca semantica com cache em disco"""

    def __init__(self, cache_path=None):
        self.textos = []
        self.respostas = []
        self.embeddings = None
        self.cache_path = cache_path
        self._cache_emb_path = cache_path.replace('.npy', '_emb.npy') if cache_path else None
        self._cache_meta_path = cache_path.replace('.npy', '_meta.npy') if cache_path else None

    def _tentar_cache(self, n_textos):
        """Tenta carregar cache se numero de textos bate"""
        if not self._cache_emb_path or not os.path.exists(self._cache_emb_path):
            return False
        try:
            embs = np.load(self._cache_emb_path)
            if embs.shape[0] == n_textos:
                self.embeddings = embs
                print(f"  [Embeddings] Cache carregado ({n_textos} vetores)")
                return True
        except Exception:
            pass
        return False

    def _salvar_cache(self):
        """Salva embeddings no disco"""
        if self._cache_emb_path and self.embeddings is not None:
            try:
                os.makedirs(os.path.dirname(self._cache_emb_path), exist_ok=True)
                np.save(self._cache_emb_path, self.embeddings)
            except Exception:
                pass

    def construir(self, textos, respostas=None):
        """Constroi indice a partir de textos (e opcionalmente respostas)"""
        self.textos = list(textos)
        self.respostas = list(respostas) if respostas else []

        if not self.textos:
            self.embeddings = np.array([])
            return

        # Tenta cache
        if self._tentar_cache(len(self.textos)):
            return

        # Gera embeddings
        print(f"  [Embeddings] Gerando embeddings para {len(self.textos)} textos...")
        self.embeddings = gerar_embeddings(self.textos)
        if self.embeddings is not None:
            self._salvar_cache()
            print(f"  [Embeddings] OK ({self.embeddings.shape})")
        else:
            print("  [Embeddings] Falhou — usando fallback")

    def buscar(self, query, top_k=3):
        """Busca textos mais similares. Retorna [(texto, resposta, score), ...]"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        emb_q = gerar_embeddings([query])
        if emb_q is None:
            return []

        scores = similaridade_cosseno(emb_q[0], self.embeddings)
        indices = np.argsort(scores)[::-1][:top_k]

        resultados = []
        for idx in indices:
            idx = int(idx)
            resposta = self.respostas[idx] if idx < len(self.respostas) else None
            resultados.append((self.textos[idx], resposta, float(scores[idx])))

        return resultados

    def adicionar(self, texto, resposta=None):
        """Adiciona um texto ao indice (sem recalcular tudo)"""
        self.textos.append(texto)
        if resposta is not None:
            self.respostas.append(resposta)

        emb = gerar_embeddings([texto])
        if emb is not None and self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, emb])
            self._salvar_cache()
        elif emb is not None:
            self.embeddings = emb
            self._salvar_cache()
