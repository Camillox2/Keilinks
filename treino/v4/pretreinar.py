"""Pré-treino causal da Keilinks V4 em texto português."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch

from dados.tokenizador import Tokenizador
from treino.v4.config import TrainConfig, get_model_config, get_train_config
from treino.v4.modelo import KeilinksV4
from treino.v4.treinar import atomic_save, autocast_context, build_optimizer, cosine_lr, seed_everything


def iter_documents(path: Path) -> Iterator[str]:
    buffer = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip(): buffer.append(line.rstrip())
            elif buffer:
                yield "\n".join(buffer).strip(); buffer = []
    if buffer: yield "\n".join(buffer).strip()


def split_document(text: str, validation_ratio: float) -> str:
    digest = hashlib.blake2b(text[:2000].encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64 - 1)
    return "validation" if value < validation_ratio else "train"


def tokenize_to_binary(input_path: Path, tokenizer: Tokenizador, output_dir: Path,
                       validation_ratio: float = 0.005) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "documents": 0, "rejected": 0}
    eos_id = tokenizer.vocab.get("<fim>", 3)
    with (output_dir / "train_tokens.bin").open("wb") as train_handle, \
         (output_dir / "validation_tokens.bin").open("wb") as val_handle:
        for document in iter_documents(input_path):
            if len(document) < 100:
                counts["rejected"] += 1; continue
            split = split_document(document, validation_ratio)
            tokens = tokenizer.encode(document)
            if len(tokens) < 16:
                counts["rejected"] += 1; continue
            array = np.asarray(tokens + [eos_id], dtype=np.int32)
            (val_handle if split == "validation" else train_handle).write(array.tobytes())
            counts[split] += len(array); counts["documents"] += 1
            if counts["documents"] % 10000 == 0:
                print(f"{counts['documents']:,} docs | {(counts['train']+counts['validation'])/1e9:.2f}B tokens")
    metadata = {"format": "keilinks-pretrain-v4", "dtype": "int32",
                "vocab_size": tokenizer.tam_vocab, "validation_ratio": validation_ratio, **counts}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class TokenMemmap:
    def __init__(self, path: Path) -> None:
        self.length = path.stat().st_size // np.dtype(np.int32).itemsize
        self.data = np.memmap(path, dtype=np.int32, mode="r", shape=(self.length,))

    def sample(self, rng: np.random.Generator, batch_size: int,
               context: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.length <= context + 1: raise ValueError("Corpus menor que o contexto")
        starts = rng.integers(0, self.length-context-1, size=batch_size, dtype=np.int64)
        x = np.stack([self.data[s:s+context] for s in starts])
        y = np.stack([self.data[s+1:s+context+1] for s in starts])
        return torch.from_numpy(x.astype(np.int64, copy=False)), torch.from_numpy(y.astype(np.int64, copy=False))


@torch.no_grad()
def evaluate(model, corpus, rng, device, config, context, batches=20) -> float:
    model.eval(); losses = []
    for _ in range(batches):
        x, y = corpus.sample(rng, config.micro_batch_size, context)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_context(device, config.precision): _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train(); return sum(losses) / len(losses)


def pretrain(args: argparse.Namespace) -> None:
    model_config, train_config = get_model_config(args.model), get_train_config(args.profile)
    if args.steps: train_config = TrainConfig(**{**asdict(train_config), "max_steps": args.steps})
    tokenizer = Tokenizador(args.vocab); binary_dir = Path(args.binary_dir)
    if args.rebuild_binary or not (binary_dir / "metadata.json").exists():
        print(json.dumps(tokenize_to_binary(Path(args.input), tokenizer, binary_dir,
                                            args.validation_ratio), indent=2))
    if args.prepare_only: return
    train_corpus = TokenMemmap(binary_dir / "train_tokens.bin")
    val_corpus = TokenMemmap(binary_dir / "validation_tokens.bin")
    seed_everything(train_config.seed)
    rng, eval_rng = np.random.default_rng(train_config.seed), np.random.default_rng(train_config.seed+1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model = KeilinksV4(model_config).to(device)
    model.set_gradient_checkpointing(train_config.checkpoint_mode, train_config.checkpoint_every)
    optimizer = build_optimizer(model, train_config, device)
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    start_step, best_val = 0, math.inf
    resume = Path(args.resume) if args.resume else output_dir / "pretrain_latest.pt"
    if resume.exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = int(ckpt.get("step", 0))+1, float(ckpt.get("best_validation_loss", math.inf))
    model_exec = model
    if not args.no_compile and hasattr(torch, "compile") and device.type == "cuda":
        try: model_exec = torch.compile(model, mode=train_config.compile_mode, fullgraph=False)
        except Exception as exc: print(f"[aviso] compile falhou: {exc}")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda" and not torch.cuda.is_bf16_supported()))
    model.train(); optimizer.zero_grad(set_to_none=True)
    context = model_config.context_length; last_time = time.perf_counter(); tokens_window = 0
    log_path = output_dir / "pretrain_log.jsonl"
    print(f"{model_config.name} | {model.parameter_count()/1e6:.1f}M | corpus {train_corpus.length/1e9:.2f}B tokens")
    for step in range(start_step, train_config.max_steps):
        lr = cosine_lr(step, train_config.max_steps, train_config.warmup_steps,
                       train_config.learning_rate, train_config.min_learning_rate)
        for group in optimizer.param_groups: group["lr"] = lr
        loss_sum = 0.0
        for _ in range(train_config.grad_accum_steps):
            x, y = train_corpus.sample(rng, train_config.micro_batch_size, context)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_context(device, train_config.precision):
                _, loss = model_exec(x, y); scaled = loss / train_config.grad_accum_steps
            scaler.scale(scaled).backward(); loss_sum += float(loss.item()); tokens_window += x.numel()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if step % 20 == 0:
            now = time.perf_counter(); tok_s = tokens_window / max(now-last_time, 1e-6)
            vram = torch.cuda.max_memory_allocated()/1e9 if device.type=="cuda" else 0
            train_loss = loss_sum / train_config.grad_accum_steps
            print(f"[{step:>7}] loss {train_loss:.4f} | {tok_s:,.0f} tok/s | VRAM {vram:.2f}G")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step":step,"loss":train_loss,"lr":lr,
                    "tokens_per_second":tok_s,"grad_norm":float(grad_norm),"vram_gb":vram})+"\n")
            tokens_window=0; last_time=now
        if step>0 and step%train_config.eval_interval==0:
            val_loss = evaluate(model, val_corpus, eval_rng, device, train_config, context)
            print(f"val_loss {val_loss:.4f} | ppl {math.exp(min(val_loss,20)):.2f}")
            if val_loss < best_val:
                best_val = val_loss
                atomic_save(model.checkpoint_payload(step,best_validation_loss=best_val,
                    model_profile=args.model,train_config=asdict(train_config)), output_dir/"pretrain_best.pt")
        if step>0 and step%train_config.save_interval==0:
            atomic_save(model.checkpoint_payload(step,optimizer=optimizer.state_dict(),
                best_validation_loss=best_val,model_profile=args.model,
                train_config=asdict(train_config)), output_dir/"pretrain_latest.pt")


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="core_380m")
    p.add_argument("--profile",default="rtx5050_380m"); p.add_argument("--input",default="dados/v4/pretrain/pretrain_pt.txt")
    p.add_argument("--vocab",default="dados/vocab.json"); p.add_argument("--binary-dir",default="dados/v4/pretrain_binary")
    p.add_argument("--output",default="checkpoints/v4"); p.add_argument("--validation-ratio",type=float,default=.005)
    p.add_argument("--steps",type=int); p.add_argument("--resume"); p.add_argument("--rebuild-binary",action="store_true")
    p.add_argument("--prepare-only",action="store_true"); p.add_argument("--no-compile",action="store_true"); return p.parse_args()


if __name__ == "__main__": pretrain(parse_args())
