"""Configurações da Keilinks V4.

Os perfis maiores existem para evolução do projeto, mas o perfil recomendado para
pré-treino completo numa RTX 5050 Laptop de 8 GB é o ``core_380m``. Os perfis
acima de 500M exigem checkpointing agressivo, batch 1 e possivelmente offload.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int = 32_000
    dim: int = 1_152
    n_layers: int = 24
    n_heads: int = 18
    n_kv_heads: int = 6
    ff_dim: int = 3_072
    context_length: int = 2_048
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5

    def validate(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim deve ser divisível por n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads deve ser divisível por n_kv_heads")
        if self.ff_dim <= self.dim:
            raise ValueError("ff_dim deve ser maior que dim")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    profile: str
    micro_batch_size: int
    grad_accum_steps: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    save_interval: int = 1_000
    eval_batches: int = 20
    compile_mode: str = "reduce-overhead"
    checkpoint_mode: str = "selective"
    checkpoint_every: int = 2
    optimizer: str = "adamw_fused"
    precision: str = "bf16"
    num_workers: int = 2
    prefetch_factor: int = 3
    seed: int = 42


MODEL_PROFILES: Dict[str, ModelConfig] = {
    "core_380m": ModelConfig(
        name="Keilinks Core V4 380M", dim=1_152, n_layers=24,
        n_heads=18, n_kv_heads=6, ff_dim=3_072,
    ),
    "core_500m": ModelConfig(
        name="Keilinks Core V4 500M", dim=1_280, n_layers=26,
        n_heads=20, n_kv_heads=5, ff_dim=3_584,
    ),
    "core_800m": ModelConfig(
        name="Keilinks Core V4 800M Experimental", dim=1_536, n_layers=30,
        n_heads=24, n_kv_heads=6, ff_dim=4_096,
    ),
    "core_1b": ModelConfig(
        name="Keilinks Core V4 1B Experimental", dim=1_792, n_layers=30,
        n_heads=28, n_kv_heads=7, ff_dim=4_864,
    ),
}


TRAIN_PROFILES: Dict[str, TrainConfig] = {
    "rtx5050_380m": TrainConfig(
        profile="rtx5050_380m", micro_batch_size=1, grad_accum_steps=16,
        max_steps=160_000, learning_rate=3e-4, min_learning_rate=3e-5,
        warmup_steps=2_000, checkpoint_mode="selective", checkpoint_every=2,
    ),
    "rtx5050_500m": TrainConfig(
        profile="rtx5050_500m", micro_batch_size=1, grad_accum_steps=24,
        max_steps=180_000, learning_rate=2.5e-4, min_learning_rate=2.5e-5,
        warmup_steps=2_500, checkpoint_mode="full", checkpoint_every=1,
    ),
    "rtx5050_800m": TrainConfig(
        profile="rtx5050_800m", micro_batch_size=1, grad_accum_steps=32,
        max_steps=220_000, learning_rate=2e-4, min_learning_rate=2e-5,
        warmup_steps=3_000, checkpoint_mode="full", checkpoint_every=1,
        optimizer="adamw_8bit",
    ),
}


def get_model_config(name: str) -> ModelConfig:
    try:
        config = MODEL_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"Modelo desconhecido: {name}. Opções: {list(MODEL_PROFILES)}") from exc
    config.validate()
    return config


def get_train_config(name: str) -> TrainConfig:
    try:
        return TRAIN_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"Perfil desconhecido: {name}. Opções: {list(TRAIN_PROFILES)}") from exc
