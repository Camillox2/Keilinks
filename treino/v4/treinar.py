"""Treinamento otimizado da Keilinks V4 para GPUs de 8 GB."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from treino.v4.config import TrainConfig, get_model_config, get_train_config
from treino.v4.dataset import PackedBinaryDataset
from treino.v4.modelo import KeilinksV4


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total, warmup, max_lr, min_lr):
    if step < warmup: return max_lr * (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_loader(dataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    kwargs = dict(dataset=dataset, batch_size=config.micro_batch_size,
                  pin_memory=torch.cuda.is_available(), num_workers=config.num_workers,
                  drop_last=True)
    if shuffle:
        kwargs["sampler"] = RandomSampler(dataset, replacement=True,
                                           num_samples=max(len(dataset), config.max_steps))
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def infinite_batches(loader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True: yield from loader


def build_optimizer(model, config: TrainConfig, device):
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate,
                                       betas=(0.9, 0.95), weight_decay=config.weight_decay)
        except Exception as exc:
            print(f"[aviso] AdamW 8-bit indisponível ({exc}); usando AdamW fused.")
    try:
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.95), weight_decay=config.weight_decay,
                                 fused=device.type == "cuda")
    except TypeError:
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.95), weight_decay=config.weight_decay)


def autocast_context(device, precision):
    if device.type != "cuda": return nullcontext()
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", dtype=torch.float16)


@torch.no_grad()
def evaluate(model, loader, device, precision, batches):
    model.eval(); losses = []; iterator = iter(loader)
    for _ in range(batches):
        try: input_ids, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader); input_ids, labels = next(iterator)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, precision): _, loss = model(input_ids, labels)
        if loss is not None: losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def atomic_save(payload, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temp); os.replace(temp, target)


def save_checkpoint(model, optimizer, step, best_val, model_name, train_config, output_dir, filename):
    payload = model.checkpoint_payload(step, optimizer=optimizer.state_dict(),
        best_validation_loss=best_val, model_profile=model_name,
        train_config=asdict(train_config), torch_version=torch.__version__)
    atomic_save(payload, output_dir / filename)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0)) + 1, float(ckpt.get("best_validation_loss", math.inf))


def train(args):
    model_config, train_config = get_model_config(args.model), get_train_config(args.profile)
    if args.steps: train_config = TrainConfig(**{**asdict(train_config), "max_steps": args.steps})
    if args.workers is not None: train_config = TrainConfig(**{**asdict(train_config), "num_workers": args.workers})
    seed_everything(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    train_dataset = PackedBinaryDataset(args.data, "train")
    val_dataset = PackedBinaryDataset(args.data, "validation")
    if train_dataset.context_length != model_config.context_length:
        raise ValueError("Reconstrua o dataset com o mesmo contexto do modelo")
    train_loader = make_loader(train_dataset, train_config, True)
    val_loader = make_loader(val_dataset, train_config, False)
    batches = infinite_batches(train_loader)
    model = KeilinksV4(model_config).to(device)
    model.set_gradient_checkpointing(train_config.checkpoint_mode, train_config.checkpoint_every)
    optimizer = build_optimizer(model, train_config, device)
    output_dir = Path(args.output); start_step = 0; best_val = math.inf
    resume_path = Path(args.resume) if args.resume else output_dir / "latest.pt"
    if resume_path.exists(): start_step, best_val = load_checkpoint(resume_path, model, optimizer, device)
    model_exec = model
    if not args.no_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model_exec = torch.compile(model, mode=train_config.compile_mode, fullgraph=False)
            print(f"torch.compile ativo: {train_config.compile_mode}")
        except Exception as exc: print(f"[aviso] torch.compile falhou: {exc}")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not torch.cuda.is_bf16_supported()))
    model.train(); optimizer.zero_grad(set_to_none=True); output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    tokens_since_log = 0; last_log_time = time.perf_counter(); running_loss = 0.0; running_micro = 0
    print(f"{model_config.name} | {model.parameter_count()/1e6:.1f}M parâmetros | {device}")
    for step in range(start_step, train_config.max_steps):
        lr = cosine_lr(step, train_config.max_steps, train_config.warmup_steps,
                       train_config.learning_rate, train_config.min_learning_rate)
        for group in optimizer.param_groups: group["lr"] = lr
        for _ in range(train_config.grad_accum_steps):
            input_ids, labels = next(batches)
            input_ids = input_ids.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            with autocast_context(device, train_config.precision):
                _, loss = model_exec(input_ids, labels)
                if loss is None: raise RuntimeError("Loss não calculado")
                scaled_loss = loss / train_config.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.item()); running_micro += 1
            tokens_since_log += int((labels != -100).sum().item())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if step % 20 == 0:
            now = time.perf_counter(); elapsed = max(now - last_log_time, 1e-6)
            tok_s = tokens_since_log / elapsed; avg_loss = running_loss / max(running_micro, 1)
            vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(f"[{step:>7}] loss {avg_loss:.4f} | {tok_s:,.0f} target tok/s | VRAM {vram:.2f}G")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "train_loss": avg_loss, "lr": lr,
                    "target_tokens_per_second": tok_s, "grad_norm": float(grad_norm),
                    "vram_gb": vram, "timestamp": time.time()}) + "\n")
            tokens_since_log = 0; running_loss = 0.0; running_micro = 0; last_log_time = now
        if step > 0 and step % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, train_config.precision, train_config.eval_batches)
            print(f"avaliação: val_loss={val_loss:.4f} | ppl={math.exp(min(val_loss,20)):.2f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step, best_val, args.model, train_config, output_dir, "best.pt")
        if step > 0 and step % train_config.save_interval == 0:
            save_checkpoint(model, optimizer, step, best_val, args.model, train_config, output_dir, "latest.pt")
    save_checkpoint(model, optimizer, train_config.max_steps-1, best_val,
                    args.model, train_config, output_dir, "final.pt")
    shutil.copy2(output_dir / ("best.pt" if (output_dir / "best.pt").exists() else "final.pt"),
                 output_dir / "keilinks_v4.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="core_380m")
    parser.add_argument("--profile", default="rtx5050_380m")
    parser.add_argument("--data", default="dados/v4/packed")
    parser.add_argument("--output", default="checkpoints/v4")
    parser.add_argument("--resume"); parser.add_argument("--steps", type=int)
    parser.add_argument("--workers", type=int); parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try: train(parse_args())
    except torch.OutOfMemoryError as exc:
        raise SystemExit("CUDA sem memória: use core_380m, feche o Ollama ou ative checkpoint full") from exc
