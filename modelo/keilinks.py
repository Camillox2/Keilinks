"""
Keilinks — IA pessoal do Vitor
Modelos disponíveis:
  - Flash : principal, conversacional (~250M params, 32K vocab, 2048 ctx)
  - Pro   : equilibrado (~275M params, 32K vocab, 2048 ctx)
  - Ultra : mais profundo (~300M params, 32K vocab, 2048 ctx)

KV-Cache: geração 2-3x mais rápida (não recomputa tokens anteriores)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import math


class AtencaoMultiCabeca(nn.Module):
    def __init__(self, dim, num_cabecas, dropout=0.1):
        super().__init__()
        assert dim % num_cabecas == 0
        self.num_cabecas = num_cabecas
        self.dim_cabeca = dim // num_cabecas
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj_saida = nn.Linear(dim, dim, bias=False)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = [
            t.view(B, T, self.num_cabecas, self.dim_cabeca).transpose(1, 2)
            for t in qkv
        ]

        # KV-Cache: concatena com chaves/valores anteriores
        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, K], dim=2)
            V = torch.cat([V_prev, V], dim=2)

        novo_cache = (K, V)

        x = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(kv_cache is None)  # causal so no prefill, no decode ja tem mascara pelo cache
        )
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_saida(x), novo_cache


class FeedForward(nn.Module):
    """SwiGLU — mais moderna que ReLU, usada no LLaMA"""
    def __init__(self, dim, dim_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, dim_ff, bias=False)
        self.w2 = nn.Linear(dim_ff, dim, bias=False)
        self.w3 = nn.Linear(dim, dim_ff, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norma = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norma)


class BlocoTransformer(nn.Module):
    def __init__(self, dim, num_cabecas, dim_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.atencao = AtencaoMultiCabeca(dim, num_cabecas, dropout)
        self.ff = FeedForward(dim, dim_ff, dropout)

    def forward(self, x, kv_cache=None):
        residual = x
        x_norm = self.norm1(x)
        attn_out, novo_cache = self.atencao(x_norm, kv_cache=kv_cache)
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x, novo_cache


class Keilinks(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.usar_grad_checkpoint = False

        self.embedding_token = nn.Embedding(config['vocab_size'], config['dim'])
        self.embedding_posicao = nn.Embedding(config['contexto_max'], config['dim'])
        self.drop_entrada = nn.Dropout(config['dropout'])

        self.blocos = nn.ModuleList([
            BlocoTransformer(config['dim'], config['num_cabecas'], config['dim_ff'], config['dropout'])
            for _ in range(config['num_camadas'])
        ])

        self.norm_final = RMSNorm(config['dim'])
        self.cabeca_saida = nn.Linear(config['dim'], config['vocab_size'], bias=False)
        self.cabeca_saida.weight = self.embedding_token.weight  # weight tying

        self.apply(self._init_pesos)

        n = self._n_params()
        nome = config.get('nome', 'Keilinks')
        print(f"{nome} — {n/1e6:.1f}M params | {config['num_camadas']} camadas | dim {config['dim']}")

    def _init_pesos(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def _n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.drop_entrada(self.embedding_token(tokens) + self.embedding_posicao(pos))
        for bloco in self.blocos:
            if self.usar_grad_checkpoint and self.training:
                x, _ = torch_checkpoint(bloco, x, use_reentrant=False)
            else:
                x, _ = bloco(x)
        logits = self.cabeca_saida(self.norm_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def gerar(self, tokens, max_tokens=300, temperatura=0.8, top_p=0.9):
        """Gera tokens com KV-Cache (2-3x mais rapido)"""
        self.eval()
        ctx_max = self.config['contexto_max']
        eos_id = self.config.get('eos_id', 1)
        num_camadas = len(self.blocos)

        # Prefill: processa todos os tokens de uma vez, guarda cache
        tokens_in = tokens[:, -ctx_max:]
        B, T = tokens_in.shape
        pos = torch.arange(T, device=tokens_in.device).unsqueeze(0)
        x = self.drop_entrada(self.embedding_token(tokens_in) + self.embedding_posicao(pos))

        caches = [None] * num_camadas
        for i, bloco in enumerate(self.blocos):
            x, caches[i] = bloco(x)

        logits = self.cabeca_saida(self.norm_final(x))
        logits = logits[:, -1, :] / max(temperatura, 1e-8)

        # Amostra primeiro token
        proximo = self._amostrar(logits, top_p)
        if proximo.item() == eos_id:
            return tokens
        tokens = torch.cat([tokens, proximo], dim=1)
        pos_atual = T

        # Decode: um token por vez usando cache
        for _ in range(max_tokens - 1):
            if pos_atual >= ctx_max:
                # Contexto estourou — reseta cache e faz prefill de novo
                return self._gerar_sem_cache(tokens, max_tokens - (pos_atual - T), temperatura, top_p)

            pos = torch.tensor([[pos_atual]], device=tokens.device)
            x = self.embedding_token(proximo) + self.embedding_posicao(pos)

            for i, bloco in enumerate(self.blocos):
                x, caches[i] = bloco(x, kv_cache=caches[i])

            logits = self.cabeca_saida(self.norm_final(x))
            logits = logits[:, -1, :] / max(temperatura, 1e-8)

            proximo = self._amostrar(logits, top_p)
            if proximo.item() == eos_id:
                break
            tokens = torch.cat([tokens, proximo], dim=1)
            pos_atual += 1

        return tokens

    def _amostrar(self, logits, top_p):
        """Top-p sampling"""
        probs = F.softmax(logits, dim=-1)
        probs_ord, idx_ord = torch.sort(probs, descending=True)
        cum = torch.cumsum(probs_ord, dim=-1)
        probs_ord[cum - probs_ord > top_p] = 0.0
        probs_ord /= probs_ord.sum()
        return idx_ord.gather(-1, torch.multinomial(probs_ord, 1))

    def _gerar_sem_cache(self, tokens, max_tokens, temperatura, top_p):
        """Fallback sem cache quando contexto estoura"""
        eos_id = self.config.get('eos_id', 1)
        for _ in range(max_tokens):
            ctx = tokens[:, -self.config['contexto_max']:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperatura, 1e-8)
            proximo = self._amostrar(logits, top_p)
            if proximo.item() == eos_id:
                break
            tokens = torch.cat([tokens, proximo], dim=1)
        return tokens


# ─── Configuracoes dos modelos ─────────────────────────────────────────────

CONFIG_FLASH = {
    'nome':         'Keilinks Flash v3',
    'vocab_size':   32000,
    'dim':          896,
    'num_cabecas':  16,
    'num_camadas':  23,
    'dim_ff':       2384,
    'contexto_max': 2048,
    'dropout':      0.05,
}

CONFIG_KEILINKS = {
    'nome':         'Keilinks Pro',
    'vocab_size':   32000,
    'dim':          1152,
    'num_cabecas':  18,
    'num_camadas':  15,
    'dim_ff':       3060,
    'contexto_max': 2048,
    'dropout':      0.05,
}

CONFIG_ULTRA = {
    'nome':         'Keilinks Ultra v2',
    'vocab_size':   32000,
    'dim':          1008,
    'num_cabecas':  18,
    'num_camadas':  22,
    'dim_ff':       2682,
    'contexto_max': 2048,
    'dropout':      0.05,
}

MODELOS = {
    'flash': CONFIG_FLASH,
    'padrao': CONFIG_KEILINKS,
    'ultra': CONFIG_ULTRA,
}
