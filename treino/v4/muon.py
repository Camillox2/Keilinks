"""Otimizador Muon (MomentUm Orthogonalized by Newton-Schulz) e construtor híbrido.

Referência: Keller Jordan et al. (2024).
Otimiza tensores 2D de pesos ocultos via Newton-Schulz. Ganhos de convergência
dependem do modelo e dos dados; este arquivo permanece experimental até haver
benchmark reproduzível do Keilinks. Parâmetros 1D, embeddings e LM head são
delegados para AdamW (8-bit / Fused).
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Calcula a raiz zero / ortogonalização de G via iteração de Newton-Schulz de ordem 5."""
    assert len(G.shape) == 2, f"Esperado tensor 2D, recebido {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if (G.is_cuda and torch.cuda.is_bf16_supported()) else G.float()
    norm = X.norm() + eps
    X = X / norm
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Otimizador Muon para matrizes 2D de pesos em Transformers."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(f"Muon suporta apenas matrizes 2D, recebido tensor de shape {g.shape}")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update_mat = g.add(buf, alpha=momentum)
                else:
                    update_mat = buf

                u = zeropower_via_newtonschulz5(update_mat, steps=ns_steps)
                # Escala geométrica baseada na proporção de aspectos da matriz
                aspect = max(1.0, float(g.size(0) / g.size(1))) ** 0.5
                u.mul_(aspect)

                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(u, alpha=-lr)

        return loss


class HybridOptimizer:
    """Wrapper que coordena Muon (pesos 2D) e AdamW (1D, embeddings, norm) conjuntamente."""

    def __init__(self, muon_opt: Muon, adam_opt: Optimizer) -> None:
        self.muon_opt = muon_opt
        self.adam_opt = adam_opt
        self.param_groups = self.muon_opt.param_groups + self.adam_opt.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adam_opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        self.muon_opt.step(closure=closure)
        self.adam_opt.step(closure=closure)

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict(),
            "adam": self.adam_opt.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "muon" in state_dict and "adam" in state_dict:
            self.muon_opt.load_state_dict(state_dict["muon"])
            self.adam_opt.load_state_dict(state_dict["adam"])
        else:
            # Compatibilidade com checkpoint de otimizador monolítico
            try:
                self.adam_opt.load_state_dict(state_dict)
            except Exception:
                pass


def build_muon_hybrid_optimizer(
    model: torch.nn.Module,
    lr_muon: float = 0.02,
    lr_adam: float = 3e-4,
    weight_decay: float = 0.05,
    device_type: str = "cuda",
) -> HybridOptimizer:
    """Separa parâmetros em 2D (Muon) vs 1D/Embedding (AdamW 8-bit ou Fused AdamW)."""
    muon_params: List[torch.nn.Parameter] = []
    adam_params: List[torch.nn.Parameter] = []

    # O embedding e lm_head não devem ser processados por Muon
    special_names = {"token_embedding", "lm_head", "embedding_token", "cabeca_saida"}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_special = any(s in name for s in special_names)
        if p.ndim == 2 and not is_special:
            muon_params.append(p)
        else:
            adam_params.append(p)

    muon_opt = Muon(muon_params, lr=lr_muon, weight_decay=weight_decay)

    # Tenta usar AdamW 8-bit para economizar VRAM
    adam_opt = None
    try:
        import bitsandbytes as bnb
        adam_opt = bnb.optim.AdamW8bit(adam_params, lr=lr_adam, betas=(0.9, 0.95), weight_decay=weight_decay)
    except Exception:
        try:
            adam_opt = torch.optim.AdamW(
                adam_params, lr=lr_adam, betas=(0.9, 0.95), weight_decay=weight_decay, fused=device_type == "cuda"
            )
        except TypeError:
            adam_opt = torch.optim.AdamW(adam_params, lr=lr_adam, betas=(0.9, 0.95), weight_decay=weight_decay)

    return HybridOptimizer(muon_opt, adam_opt)
