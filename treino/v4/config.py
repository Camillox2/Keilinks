"""Configurações da Keilinks V4.

O perfil recomendado para pré-treino completo numa RTX 5050 Laptop de 8 GB
é o ``core_380m``. Perfis acima de 500M exigem benchmark local, checkpointing
agressivo e possivelmente offload.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


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
    # Defaults V4 preservam compatibilidade com checkpoints já existentes.
    # Experimentos V5 devem usar um perfil explícito abaixo.
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    use_qk_norm: bool = False
    attn_logit_softcapping: float = 0.0
    final_logit_softcapping: float = 0.0

    def validate(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim deve ser divisível por n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads deve ser divisível por n_kv_heads")
        if self.ff_dim <= self.dim:
            raise ValueError("ff_dim deve ser maior que dim")
        if self.context_length < 128:
            raise ValueError("context_length muito pequeno")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta deve ser positivo")
        if self.attn_logit_softcapping < 0 or self.final_logit_softcapping < 0:
            raise ValueError("soft-capping não pode ser negativo")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    profile: str
    phase: str
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
    optimizer: str = "adamw_8bit"
    precision: str = "bf16"
    num_workers: int = 2
    prefetch_factor: int = 3
    seed: int = 42


MODEL_PROFILES: dict[str, ModelConfig] = {
    "core_380m": ModelConfig(
        name="Keilinks Core V4 380M", dim=1_152, n_layers=24,
        n_heads=18, n_kv_heads=6, ff_dim=3_072,
    ),
    "core_500m": ModelConfig(
        name="Keilinks Core V4 500M", dim=1_280, n_layers=26,
        n_heads=20, n_kv_heads=5, ff_dim=3_584,
    ),
    "core_380m_v5_experimental": ModelConfig(
        name="Keilinks Core V5 Experimental 380M",
        dim=1_152,
        n_layers=24,
        n_heads=18,
        n_kv_heads=6,
        ff_dim=3_072,
        rope_theta=500_000.0,
        norm_eps=1e-6,
        use_qk_norm=True,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    ),
    "core_380m_modern_2k": ModelConfig(
        # Compatibilidade para checkpoints antigos e benchmarks A/B. O perfil
        # operacional abaixo passou a usar 8k após medição na RTX 5050.
        name="Keilinks Core V4 Modern 380M (2K legado)",
        dim=1_152,
        n_layers=24,
        n_heads=18,
        n_kv_heads=6,
        ff_dim=3_072,
        rope_theta=500_000.0,
        norm_eps=1e-6,
        use_qk_norm=True,
        attn_logit_softcapping=0.0,
        final_logit_softcapping=0.0,
    ),
    "core_380m_modern": ModelConfig(
        # Perfil operacional da RTX 5050. O benchmark local confirmou 8k
        # estáveis com checkpointing completo e torch.compile. Soft-caps
        # seguem desligados pois materializam QK e perdem a atenção fundida.
        name="Keilinks Core V4 Modern 380M (8K)",
        dim=1_152,
        n_layers=24,
        n_heads=18,
        n_kv_heads=6,
        ff_dim=3_072,
        context_length=8_192,
        rope_theta=500_000.0,
        norm_eps=1e-6,
        use_qk_norm=True,
        attn_logit_softcapping=0.0,
        final_logit_softcapping=0.0,
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


TRAIN_PROFILES: dict[str, TrainConfig] = {
    "rtx5050_380m_2k": TrainConfig(
        profile="rtx5050_380m_2k", phase="pretrain",
        micro_batch_size=1, grad_accum_steps=16,
        max_steps=160_000, learning_rate=3e-4, min_learning_rate=3e-5,
        warmup_steps=2_000, checkpoint_mode="selective", checkpoint_every=2,
        optimizer="adamw_8bit",
    ),
    "rtx5050_380m": TrainConfig(
        profile="rtx5050_380m", phase="pretrain",
        # 1 x 8.192 x 4 preserva os 32.768 tokens por atualização do perfil
        # 2k, enquanto o checkpoint completo manteve pico abaixo de 7 GiB.
        micro_batch_size=1, grad_accum_steps=4,
        max_steps=160_000, learning_rate=3e-4, min_learning_rate=3e-5,
        warmup_steps=2_000, checkpoint_mode="full", checkpoint_every=1,
        # ``reduce-overhead`` usa CUDA Graphs e falhou com checkpointing total
        # + quatro microbatches no PyTorch 2.11 local. ``max-autotune`` sem
        # graphs ficou minutos compilando nesta GPU; o modo padrão é estável,
        # usa Inductor e não adiciona esse custo de cold-start.
        compile_mode="default",
        optimizer="adamw_8bit",
    ),
    "rtx5050_500m": TrainConfig(
        profile="rtx5050_500m", phase="pretrain",
        micro_batch_size=1, grad_accum_steps=24,
        max_steps=180_000, learning_rate=2.5e-4, min_learning_rate=2.5e-5,
        warmup_steps=2_500, checkpoint_mode="full", checkpoint_every=1,
        optimizer="adamw_8bit",
    ),
    "rtx5050_800m": TrainConfig(
        profile="rtx5050_800m", phase="pretrain",
        micro_batch_size=1, grad_accum_steps=32,
        max_steps=220_000, learning_rate=2e-4, min_learning_rate=2e-5,
        warmup_steps=3_000, checkpoint_mode="full", checkpoint_every=1,
        optimizer="adamw_8bit",
    ),
    "rtx5050_sft_380m_2k": TrainConfig(
        profile="rtx5050_sft_380m_2k", phase="sft",
        micro_batch_size=1, grad_accum_steps=16,
        max_steps=10_000, learning_rate=5e-5, min_learning_rate=5e-6,
        warmup_steps=200, weight_decay=0.05,
        eval_interval=100, save_interval=500,
        checkpoint_mode="selective", checkpoint_every=2,
        optimizer="adamw_8bit",
    ),
    "rtx5050_sft_380m": TrainConfig(
        # Mantém 32.768 tokens por atualização no Core operacional de 8k.
        profile="rtx5050_sft_380m", phase="sft",
        micro_batch_size=1, grad_accum_steps=4,
        max_steps=10_000, learning_rate=5e-5, min_learning_rate=5e-6,
        warmup_steps=200, weight_decay=0.05,
        eval_interval=50, save_interval=100,
        checkpoint_mode="full", checkpoint_every=1,
        compile_mode="default",
        optimizer="adamw_8bit",
    ),
    "rtx5050_sft_500m": TrainConfig(
        profile="rtx5050_sft_500m", phase="sft",
        micro_batch_size=1, grad_accum_steps=24,
        max_steps=12_000, learning_rate=4e-5, min_learning_rate=4e-6,
        warmup_steps=250, weight_decay=0.05,
        eval_interval=100, save_interval=500,
        checkpoint_mode="full", checkpoint_every=1,
        optimizer="adamw_8bit",
    ),
}


def get_model_config(name: str) -> ModelConfig:
    try:
        config = MODEL_PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"Modelo desconhecido: {name}. Opções: {list(MODEL_PROFILES)}"
        ) from exc
    config.validate()
    return config


def get_train_config(name: str) -> TrainConfig:
    try:
        return TRAIN_PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"Perfil desconhecido: {name}. Opções: {list(TRAIN_PROFILES)}"
        ) from exc
