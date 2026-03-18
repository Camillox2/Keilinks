"""
Keilinks — IA pessoal do Vitor
Modelos disponíveis:
  - Flash : rápido, leve, respostas instantâneas (~8M params)
  - v2    : equilibrado, padrão (~31M params)
  - Ultra : mais profundo, melhor qualidade (~85M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = [
            t.view(B, T, self.num_cabecas, self.dim_cabeca).transpose(1, 2)
            for t in qkv
        ]
        x = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_saida(x)


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

    def forward(self, x):
        x = x + self.atencao(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Keilinks(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

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
            x = bloco(x)
        logits = self.cabeca_saida(self.norm_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def gerar(self, tokens, max_tokens=300, temperatura=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_tokens):
            ctx = tokens[:, -self.config['contexto_max']:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperatura, 1e-8)
            probs = F.softmax(logits, dim=-1)
            probs_ord, idx_ord = torch.sort(probs, descending=True)
            cum = torch.cumsum(probs_ord, dim=-1)
            probs_ord[cum - probs_ord > top_p] = 0.0
            probs_ord /= probs_ord.sum()
            proximo = idx_ord.gather(-1, torch.multinomial(probs_ord, 1))
            if proximo.item() == self.config.get('eos_id', 1):
                break
            tokens = torch.cat([tokens, proximo], dim=1)
        return tokens


# ─── Configurações dos modelos ─────────────────────────────────────────────

CONFIG_FLASH = {
    'nome':         'Keilinks Flash',
    'vocab_size':   8000,
    'dim':          384,
    'num_cabecas':  6,
    'num_camadas':  6,
    'dim_ff':       1024,
    'contexto_max': 512,
    'dropout':      0.05,
}

CONFIG_KEILINKS = {
    'nome':         'Keilinks',
    'vocab_size':   8000,
    'dim':          640,
    'num_cabecas':  10,
    'num_camadas':  10,
    'dim_ff':       1920,
    'contexto_max': 512,
    'dropout':      0.1,
}

CONFIG_ULTRA = {
    'nome':         'Keilinks Ultra',
    'vocab_size':   8000,
    'dim':          896,
    'num_cabecas':  14,
    'num_camadas':  14,
    'dim_ff':       2688,
    'contexto_max': 1024,
    'dropout':      0.1,
}

MODELOS = {
    'flash': CONFIG_FLASH,
    'padrao': CONFIG_KEILINKS,
    'ultra': CONFIG_ULTRA,
}
