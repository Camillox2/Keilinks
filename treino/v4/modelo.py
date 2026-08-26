"""Transformer decoder Keilinks V4.

Mudanças principais: Grouped Query Attention, RoPE, RMSNorm, SwiGLU,
weight tying, SDPA/Flash Attention automático, checkpointing seletivo e
loss com ignore_index=-100 para SFT somente na resposta.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from treino.v4.config import ModelConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float()).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("head_dim deve ser par para RoPE")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin = sin[None, None, :, :].to(device=x.device, dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    return (x[:, :, None, :, :]
            .expand(batch, kv_heads, repeats, seq_len, head_dim)
            .reshape(batch, kv_heads * repeats, seq_len, head_dim))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.dropout = config.dropout
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", 0.0)

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Aplica QK-Norm para estabilidade antes de RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            old_k, old_v = cache
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
        new_cache = (k, v)

        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        if self.attn_logit_softcapping > 0.0:
            # SDPA não permite aplicar tanh nos logits antes do softmax.
            # Caminho explícito é usado apenas quando o experimento V5 o pede.
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) * scale
            cap = self.attn_logit_softcapping
            scores = cap * torch.tanh(scores / cap)
            if cache is None:
                causal_mask = torch.ones(
                    (seq_len, k_rep.size(-2)),
                    dtype=torch.bool,
                    device=scores.device,
                ).triu(diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            probabilities = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            if self.training and self.dropout:
                probabilities = F.dropout(probabilities, p=self.dropout)
            out = torch.matmul(probabilities, v_rep)
        else:
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=cache is None,
            )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.ff_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ff_dim, bias=False)
        self.down_proj = nn.Linear(config.ff_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, cache=cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class KeilinksV4(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.checkpoint_mode = "none"
        self.checkpoint_every = 2
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        head_dim = config.dim // config.n_heads
        cos, sin = build_rope_cache(head_dim, config.context_length, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.o_proj.weight.mul_(scale)
                block.ffn.down_proj.weight.mul_(scale)

    def set_gradient_checkpointing(self, mode: str = "selective", every: int = 2) -> None:
        if mode not in {"none", "selective", "full"}:
            raise ValueError("mode deve ser none, selective ou full")
        self.checkpoint_mode = mode
        self.checkpoint_every = max(1, every)

    def _should_checkpoint(self, layer_idx: int) -> bool:
        if not self.training or self.checkpoint_mode == "none":
            return False
        return self.checkpoint_mode == "full" or layer_idx % self.checkpoint_every == 0

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"Sequência {seq_len} excede contexto {self.config.context_length}")
        x = self.dropout(self.token_embedding(input_ids))
        cos, sin = self.rope_cos[:seq_len], self.rope_sin[:seq_len]
        for idx, block in enumerate(self.blocks):
            if self._should_checkpoint(idx):
                def custom_forward(hidden: torch.Tensor, block: DecoderBlock = block) -> torch.Tensor:
                    return block(hidden, cos, sin, cache=None)[0]
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x, _ = block(x, cos, sin, cache=None)
        logits = self.lm_head(self.final_norm(x))
        if getattr(self.config, "final_logit_softcapping", 0.0) > 0.0:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 0.8, top_p: float = 0.9,
                 eos_id: Optional[int] = None, repetition_penalty: float = 1.1) -> torch.Tensor:
        self.eval()
        tokens = input_ids[:, -self.config.context_length:]
        caches: List[Optional[KVCache]] = [None] * len(self.blocks)
        x = self.token_embedding(tokens)
        seq_len = tokens.size(1)
        cos, sin = self.rope_cos[:seq_len], self.rope_sin[:seq_len]
        for idx, block in enumerate(self.blocks):
            x, caches[idx] = block(x, cos, sin, cache=None)
        logits = self.lm_head(self.final_norm(x))[:, -1, :]
        if getattr(self.config, "final_logit_softcapping", 0.0) > 0.0:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        for _ in range(max_new_tokens):
            next_token = self._sample(logits, tokens, temperature, top_p, repetition_penalty)
            if eos_id is not None and torch.all(next_token == eos_id):
                break
            tokens = torch.cat((tokens, next_token), dim=1)
            if tokens.size(1) >= self.config.context_length:
                break
            pos = tokens.size(1) - 1
            x = self.token_embedding(next_token)
            cos, sin = self.rope_cos[pos:pos+1], self.rope_sin[pos:pos+1]
            for idx, block in enumerate(self.blocks):
                x, caches[idx] = block(x, cos, sin, cache=caches[idx])
            logits = self.lm_head(self.final_norm(x))[:, -1, :]
        return tokens

    @staticmethod
    def _sample(logits: torch.Tensor, generated: torch.Tensor, temperature: float,
                top_p: float, repetition_penalty: float) -> torch.Tensor:
        logits = logits.float() / max(temperature, 1e-5)
        if repetition_penalty != 1.0:
            for batch_idx in range(logits.size(0)):
                seen = generated[batch_idx, -128:].unique()
                values = logits[batch_idx, seen]
                logits[batch_idx, seen] = torch.where(
                    values > 0, values / repetition_penalty, values * repetition_penalty)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def checkpoint_payload(self, step: int, **extra: object) -> dict:
        return {"version": 4, "step": step, "model": self.state_dict(),
                "config": asdict(self.config), **extra}
