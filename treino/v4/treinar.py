"""Supervised fine-tuning assistant-only da Keilinks V4.

Use ``--init-checkpoint`` para iniciar a partir do melhor checkpoint de
pré-treino. ``--resume`` é reservado para retomar a própria fase SFT.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from treino.v4.config import ModelConfig, TrainConfig, get_model_config, get_train_config
from treino.v4.dataset import PackedBinaryDataset
from treino.v4.modelo import KeilinksV4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mark_compiled_microbatch(compiled: bool) -> None:
    """Marca a fronteira de microbatch para CUDA Graphs do ``torch.compile``."""

    if not compiled:
        return
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(marker):
        marker()


def cosine_lr(step: int, total: int, warmup: int,
              max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def make_loader(dataset: PackedBinaryDataset, config: TrainConfig,
                shuffle: bool) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": config.micro_batch_size,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": config.num_workers,
        "drop_last": True,
    }
    if shuffle:
        kwargs["sampler"] = RandomSampler(
            dataset,
            replacement=True,
            num_samples=max(
                len(dataset), config.max_steps * config.grad_accum_steps
            ),
        )
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def steps_for_epochs(dataset_blocks: int, config: TrainConfig, epochs: float) -> int:
    """Converte épocas reais do SFT em passos, sem repetir corpus curto demais."""

    if dataset_blocks < config.micro_batch_size:
        raise ValueError("Dataset SFT menor que o batch físico")
    if epochs <= 0:
        raise ValueError("Épocas de SFT devem ser positivas")
    batches_per_epoch = dataset_blocks // config.micro_batch_size
    updates_per_epoch = batches_per_epoch / config.grad_accum_steps
    return max(1, math.ceil(updates_per_epoch * epochs))


def infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def build_optimizer(model: torch.nn.Module, config: TrainConfig,
                    device: torch.device):
    if config.optimizer in {"muon_hybrid", "muon"}:
        from treino.v4.muon import build_muon_hybrid_optimizer
        return build_muon_hybrid_optimizer(
            model,
            lr_muon=getattr(config, "lr_muon", 0.02),
            lr_adam=config.learning_rate,
            weight_decay=config.weight_decay,
            device_type=device.type,
        )
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                model.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=config.weight_decay,
            )
        except Exception as exc:
            print(
                f"[aviso] AdamW 8-bit indisponível ({exc}); usando AdamW comum. "
                "Em uma GPU de 8 GB isso pode causar falta de memória."
            )
    options = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": config.weight_decay,
    }
    try:
        return torch.optim.AdamW(
            model.parameters(), fused=device.type == "cuda", **options
        )
    except TypeError:
        return torch.optim.AdamW(model.parameters(), **options)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = (
        torch.bfloat16
        if precision == "bf16" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.autocast("cuda", dtype=dtype)


def atomic_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_v4_checkpoint(path: Path, device: torch.device | None = None) -> dict:
    # Carregar primeiro na CPU evita copiar pesos e estados do otimizador juntos
    # para uma GPU de 8 GB. O modelo é transferido depois de forma controlada.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or "model" not in checkpoint
        or "config" not in checkpoint
    ):
        raise ValueError(f"Checkpoint V4 inválido: {path}")
    return checkpoint


def checkpoint_model_config(checkpoint: dict) -> ModelConfig:
    config = ModelConfig(**checkpoint["config"])
    config.validate()
    return config


def validate_architecture(requested: ModelConfig, loaded: ModelConfig,
                          source: Path) -> None:
    fields = (
        "vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
        "ff_dim", "context_length",
    )
    differences = [
        field for field in fields
        if getattr(requested, field) != getattr(loaded, field)
    ]
    if differences:
        details = ", ".join(
            f"{field}={getattr(loaded, field)} "
            f"(esperado {getattr(requested, field)})"
            for field in differences
        )
        raise ValueError(f"Arquitetura incompatível em {source}: {details}")


@torch.no_grad()
def evaluate(model: KeilinksV4, loader: DataLoader, device: torch.device,
             precision: str, batches: int) -> float:
    model.eval()
    losses = []
    iterator = iter(loader)
    for _ in range(batches):
        try:
            input_ids, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            input_ids, labels = next(iterator)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, precision):
            _, loss = model(input_ids, labels)
        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    if not losses:
        raise RuntimeError("Validação não produziu loss finito")
    return sum(losses) / len(losses)


def save_checkpoint(model: KeilinksV4, optimizer, step: int, best_val: float,
                    model_name: str, train_config: TrainConfig,
                    output_dir: Path, filename: str,
                    include_optimizer: bool = True) -> None:
    extra = {
        "best_validation_loss": best_val,
        "model_profile": model_name,
        "train_config": asdict(train_config),
        "phase": "sft",
        "torch_version": torch.__version__,
    }
    if include_optimizer:
        extra["optimizer"] = optimizer.state_dict()
    atomic_save(
        model.checkpoint_payload(step, **extra),
        output_dir / filename,
    )


def export_inference_checkpoint(source: Path, target: Path) -> None:
    checkpoint = load_v4_checkpoint(source)
    payload = {
        "version": checkpoint.get("version", 4),
        "step": checkpoint.get("step", 0),
        "model": checkpoint["model"],
        "config": checkpoint["config"],
        "phase": "sft",
        "model_profile": checkpoint.get("model_profile"),
        "train_config": checkpoint.get("train_config"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "source_checkpoint": source.name,
        "inference_only": True,
    }
    atomic_save(payload, target)


def resolve_bootstrap(args: argparse.Namespace, requested: ModelConfig,
                      device: torch.device, output_dir: Path):
    resume_path = Path(args.resume) if args.resume else output_dir / "latest.pt"
    if resume_path.exists():
        checkpoint = load_v4_checkpoint(resume_path, device)
        loaded = checkpoint_model_config(checkpoint)
        validate_architecture(requested, loaded, resume_path)
        if checkpoint.get("phase") not in (None, "sft"):
            raise ValueError(f"{resume_path} não pertence à fase SFT")
        return loaded, checkpoint, resume_path, True

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Pré-treino não encontrado: {init_path}")
        checkpoint = load_v4_checkpoint(init_path, device)
        loaded = checkpoint_model_config(checkpoint)
        validate_architecture(requested, loaded, init_path)
        if checkpoint.get("phase") not in (None, "pretrain"):
            raise ValueError(f"{init_path} não pertence à fase de pré-treino")
        return loaded, checkpoint, init_path, False

    if args.allow_random_init:
        print("[aviso] SFT iniciado aleatoriamente por solicitação explícita")
        return requested, None, None, False

    raise ValueError(
        "Informe --init-checkpoint com o checkpoint de pré-treino. "
        "Use --allow-random-init apenas para testes deliberados."
    )


def train(args: argparse.Namespace) -> None:
    requested_config = get_model_config(args.model)
    train_config = get_train_config(args.profile)
    if train_config.optimizer in {"muon_hybrid", "muon"} and not args.experimental_muon:
        raise ValueError(
            "Muon é experimental no Keilinks. Use --experimental-muon somente após "
            "um benchmark de baseline e com checkpoint separado."
        )
    if train_config.phase != "sft":
        raise ValueError(
            f"Perfil {args.profile} é de {train_config.phase}, não de SFT"
        )
    if args.steps is not None:
        train_config = TrainConfig(**{
            **asdict(train_config), "max_steps": args.steps,
        })
    if args.workers is not None:
        train_config = TrainConfig(**{
            **asdict(train_config), "num_workers": args.workers,
        })

    seed_everything(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output)
    model_config, bootstrap, bootstrap_path, is_resume = resolve_bootstrap(
        args, requested_config, device, output_dir
    )

    train_data = PackedBinaryDataset(args.data, "train")
    validation_data = PackedBinaryDataset(args.data, "validation")
    if train_data.context_length != model_config.context_length:
        raise ValueError(
            f"Dataset usa contexto {train_data.context_length}; "
            f"modelo usa {model_config.context_length}"
        )
    if args.steps is None:
        epochs = 3.0 if args.epochs is None else args.epochs
        derived_steps = steps_for_epochs(len(train_data), train_config, epochs)
        train_config = TrainConfig(**{
            **asdict(train_config), "max_steps": derived_steps,
        })
        print(
            f"SFT: {epochs:g} épocas em {len(train_data)} blocos -> "
            f"{derived_steps} passos de otimização"
        )
    safe_warmup = min(train_config.warmup_steps, max(1, train_config.max_steps // 10))
    if safe_warmup != train_config.warmup_steps:
        train_config = TrainConfig(**{
            **asdict(train_config), "warmup_steps": safe_warmup,
        })
        print(f"SFT: warmup ajustado para {safe_warmup} passos")

    train_loader = make_loader(train_data, train_config, shuffle=True)
    validation_loader = make_loader(validation_data, train_config, shuffle=False)
    batches = infinite_batches(train_loader)

    model = KeilinksV4(model_config)
    if bootstrap is not None:
        model.load_state_dict(bootstrap["model"], strict=True)
        print(
            ("Retomando SFT" if is_resume else "Carregando pré-treino")
            + f": {bootstrap_path}"
        )
    model.to(device)
    model.set_gradient_checkpointing(
        train_config.checkpoint_mode, train_config.checkpoint_every
    )

    optimizer = build_optimizer(model, train_config, device)
    start_step = 0
    best_val = math.inf
    if is_resume and bootstrap is not None:
        if "optimizer" in bootstrap:
            optimizer.load_state_dict(bootstrap["optimizer"])
        start_step = int(bootstrap.get("step", 0)) + 1
        best_val = float(bootstrap.get("best_validation_loss", math.inf))
    del bootstrap

    executable = model
    compiled = False
    if not args.no_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            executable = torch.compile(
                model, mode=train_config.compile_mode, fullgraph=False
            )
            compiled = True
            print(f"torch.compile ativo: {train_config.compile_mode}")
        except Exception as exc:
            print(f"[aviso] torch.compile falhou: {exc}")

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tokens_window = 0
    loss_window = 0.0
    micro_window = 0
    last_log = time.perf_counter()

    print(
        f"{model_config.name} | {model.parameter_count()/1e6:.1f}M | "
        f"{device} | SFT assistant-only"
    )

    for step in range(start_step, train_config.max_steps):
        lr = cosine_lr(
            step, train_config.max_steps, train_config.warmup_steps,
            train_config.learning_rate, train_config.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(train_config.grad_accum_steps):
            input_ids, labels = next(batches)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target_tokens = int((labels != -100).sum().item())
            if target_tokens == 0:
                raise RuntimeError(
                    "Batch sem tokens do assistant; reconstrua o dataset"
                )
            mark_compiled_microbatch(compiled)
            with autocast_context(device, train_config.precision):
                _, loss = executable(input_ids, labels, False)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"Loss inválido no passo {step}: {loss}")
                if compiled:
                    loss = loss.clone()
                scaled_loss = loss / train_config.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            tokens_window += target_tokens
            loss_window += float(loss.item())
            micro_window += 1

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(f"Gradiente inválido no passo {step}: {grad_norm}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0:
            now = time.perf_counter()
            tok_s = tokens_window / max(now - last_log, 1e-6)
            avg_loss = loss_window / max(micro_window, 1)
            vram = (
                torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda" else 0.0
            )
            print(
                f"[{step:>7}] loss {avg_loss:.4f} | "
                f"{tok_s:,.0f} target tok/s | VRAM {vram:.2f}G"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "step": step,
                    "train_loss": avg_loss,
                    "lr": lr,
                    "target_tokens_per_second": tok_s,
                    "grad_norm": float(grad_norm),
                    "vram_gb": vram,
                    "timestamp": time.time(),
                }) + "\n")
            tokens_window = 0
            loss_window = 0.0
            micro_window = 0
            last_log = now

        if step > 0 and step % train_config.eval_interval == 0:
            val_loss = evaluate(
                model, validation_loader, device,
                train_config.precision, train_config.eval_batches,
            )
            print(
                f"val_loss={val_loss:.4f} | "
                f"ppl={math.exp(min(val_loss, 20)):.2f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model, optimizer, step, best_val,
                    args.model, train_config, output_dir, "best.pt",
                    include_optimizer=False,
                )

        if step > 0 and step % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, step, best_val,
                args.model, train_config, output_dir, "latest.pt",
                include_optimizer=True,
            )

    save_checkpoint(
        model, optimizer, train_config.max_steps - 1, best_val,
        args.model, train_config, output_dir, "final.pt",
        include_optimizer=True,
    )
    selected = (
        output_dir / "best.pt"
        if (output_dir / "best.pt").exists()
        else output_dir / "final.pt"
    )
    export_inference_checkpoint(selected, output_dir / "keilinks_v4.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT assistant-only da Keilinks V4"
    )
    parser.add_argument("--model", default="core_380m_modern")
    parser.add_argument("--profile", default="rtx5050_sft_380m")
    parser.add_argument("--data", default="dados/v4/packed_sft_8k")
    parser.add_argument("--output", default="checkpoints/v4-sft")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--epochs",
        type=float,
        help="Épocas de SFT; padrão seguro: 3. Ignorado quando --steps é informado.",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument(
        "--experimental-muon",
        action="store_true",
        help="Autoriza o otimizador Muon experimental quando o perfil o selecionar.",
    )
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs deve ser positivo")
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "CUDA sem memória: feche o Ollama, use core_380m ou checkpoint full."
        ) from exc
