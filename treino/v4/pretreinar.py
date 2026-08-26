"""Pré-treino causal da Keilinks V4 em texto português.

O corpus é separado por documento antes da tokenização. Binários registram o
hash do vocabulário para impedir treino silencioso com IDs incompatíveis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from treino.v4.config import ModelConfig, TrainConfig, get_model_config, get_train_config
from treino.v4.modelo import KeilinksV4
from treino.v4.tokenizador import TokenizadorV4
from treino.v4.treinar import (
    atomic_save,
    autocast_context,
    build_optimizer,
    cosine_lr,
    seed_everything,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_documents(path: Path) -> Iterator[str]:
    buffer = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                buffer.append(line.rstrip())
            elif buffer:
                document = "\n".join(buffer).strip()
                if document:
                    yield document
                buffer = []
    if buffer:
        document = "\n".join(buffer).strip()
        if document:
            yield document


def split_document(text: str, validation_ratio: float) -> str:
    digest = hashlib.blake2b(
        text[:4000].encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / float(2**64 - 1)
    return "validation" if value < validation_ratio else "train"


def tokenize_to_binary(
    input_path: Path,
    tokenizer: TokenizadorV4,
    vocab_path: Path,
    output_dir: Path,
    validation_ratio: float = 0.005,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_final = output_dir / "train_tokens.bin"
    validation_final = output_dir / "validation_tokens.bin"
    train_temp = train_final.with_suffix(".bin.tmp")
    validation_temp = validation_final.with_suffix(".bin.tmp")
    metadata_temp = output_dir / "metadata.json.tmp"
    counts = {
        "train": 0,
        "validation": 0,
        "documents": 0,
        "rejected": 0,
    }
    eos_id = tokenizer.vocab["<fim>"]
    try:
        with (
            train_temp.open("wb") as train_handle,
            validation_temp.open("wb") as validation_handle,
        ):
            for document in iter_documents(input_path):
                if len(document) < 100:
                    counts["rejected"] += 1
                    continue
                split = split_document(document, validation_ratio)
                tokens = tokenizer.encode(document)
                if len(tokens) < 16:
                    counts["rejected"] += 1
                    continue
                array = np.asarray(tokens + [eos_id], dtype=np.int32)
                target = (
                    validation_handle if split == "validation" else train_handle
                )
                target.write(array.tobytes())
                counts[split] += len(array)
                counts["documents"] += 1
                if counts["documents"] % 10_000 == 0:
                    total = counts["train"] + counts["validation"]
                    print(
                        f"{counts['documents']:,} docs | "
                        f"{total/1e9:.3f}B tokens"
                    )
        if counts["train"] < 10_000 or counts["validation"] < 2_048:
            raise ValueError(
                "Corpus ou validação muito pequeno. "
                "Aumente os dados antes do pré-treino."
            )
        metadata = {
            "format": "keilinks-pretrain-v4",
            "dtype": "int32",
            "source_path": str(input_path),
            "source_size": input_path.stat().st_size,
            "source_sha256": sha256_file(input_path),
            "vocab_path": str(vocab_path),
            "vocab_sha256": sha256_file(vocab_path),
            "vocab_size": tokenizer.tam_vocab,
            "validation_ratio": validation_ratio,
            **counts,
        }
        metadata_temp.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        os.replace(train_temp, train_final)
        os.replace(validation_temp, validation_final)
        os.replace(metadata_temp, output_dir / "metadata.json")
        return metadata
    except Exception:
        for path in (train_temp, validation_temp, metadata_temp):
            path.unlink(missing_ok=True)
        raise


def validate_binary_metadata(
    metadata_path: Path,
    vocab_path: Path,
    tokenizer: TokenizadorV4,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "keilinks-pretrain-v4":
        raise ValueError("Formato dos binários não é V4")
    if metadata.get("vocab_sha256") != sha256_file(vocab_path):
        raise ValueError(
            "Vocabulário mudou. Execute novamente com --rebuild-binary"
        )
    if int(metadata.get("vocab_size", -1)) != tokenizer.tam_vocab:
        raise ValueError(
            "Tamanho do vocabulário incompatível com os binários"
        )
    return metadata


class TokenMemmap:
    def __init__(self, path: Path) -> None:
        if (
            not path.exists()
            or path.stat().st_size % np.dtype(np.int32).itemsize
        ):
            raise ValueError(f"Binário inválido: {path}")
        self.length = path.stat().st_size // np.dtype(np.int32).itemsize
        self.data = np.memmap(
            path, dtype=np.int32, mode="r", shape=(self.length,)
        )

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        context: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.length <= context + 1:
            raise ValueError("Corpus menor que o contexto")
        starts = rng.integers(
            0,
            self.length - context - 1,
            size=batch_size,
            dtype=np.int64,
        )
        x = np.stack(
            [self.data[start:start + context] for start in starts]
        )
        y = np.stack(
            [self.data[start + 1:start + context + 1] for start in starts]
        )
        return (
            torch.from_numpy(x.astype(np.int64, copy=False)),
            torch.from_numpy(y.astype(np.int64, copy=False)),
        )


@torch.no_grad()
def evaluate(
    model: KeilinksV4,
    corpus: TokenMemmap,
    rng: np.random.Generator,
    device: torch.device,
    config: TrainConfig,
    context: int,
    batches: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = corpus.sample(
            rng, config.micro_batch_size, context
        )
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, config.precision):
            _, loss = model(x, y)
        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    if not losses:
        raise RuntimeError("Validação não produziu loss finito")
    return sum(losses) / len(losses)


def checkpoint_config(checkpoint: dict) -> ModelConfig:
    config = ModelConfig(**checkpoint["config"])
    config.validate()
    return config


def validate_resume(
    requested: ModelConfig,
    loaded: ModelConfig,
    checkpoint_path: Path,
) -> None:
    fields = (
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "ff_dim",
        "context_length",
    )
    different = [
        field
        for field in fields
        if getattr(requested, field) != getattr(loaded, field)
    ]
    if different:
        raise ValueError(
            f"Checkpoint {checkpoint_path} incompatível nos campos: "
            f"{different}"
        )


def mark_compiled_microbatch(compiled: bool) -> None:
    """Delimita microbatches para CUDA Graphs durante acumulação de gradiente.

    ``reduce-overhead`` habilita CUDA Graphs. Com checkpointing completo, a
    heurística automática pode confundir recomputação do backward com uma nova
    invocação e sobrescrever saídas ainda vivas. A API oficial marca a fronteira
    de cada microbatch sem afetar execução eager ou modos sem CUDA Graphs.
    """

    if not compiled:
        return
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(marker):
        marker()


def pretrain(args: argparse.Namespace) -> None:
    train_config = get_train_config(args.profile)
    if train_config.phase != "pretrain":
        raise ValueError(
            f"Perfil {args.profile} é de {train_config.phase}, "
            "não de pré-treino"
        )
    if args.steps is not None:
        train_config = TrainConfig(**{
            **asdict(train_config),
            "max_steps": args.steps,
        })

    vocab_path = Path(args.vocab)
    tokenizer = TokenizadorV4(vocab_path)
    model_config = replace(
        get_model_config(args.model),
        vocab_size=tokenizer.tam_vocab,
    )
    model_config.validate()

    binary_dir = Path(args.binary_dir)
    metadata_path = binary_dir / "metadata.json"
    if args.rebuild_binary or not metadata_path.exists():
        metadata = tokenize_to_binary(
            Path(args.input),
            tokenizer,
            vocab_path,
            binary_dir,
            args.validation_ratio,
        )
    else:
        metadata = validate_binary_metadata(
            metadata_path, vocab_path, tokenizer
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    train_corpus = TokenMemmap(binary_dir / "train_tokens.bin")
    validation_corpus = TokenMemmap(
        binary_dir / "validation_tokens.bin"
    )
    seed_everything(train_config.seed)
    rng = np.random.default_rng(train_config.seed)
    eval_rng = np.random.default_rng(train_config.seed + 1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pause_file = (
        Path(args.pause_file)
        if args.pause_file
        else output_dir / "PAUSE_REQUESTED"
    )
    resume_path = (
        Path(args.resume)
        if args.resume
        else output_dir / "pretrain_latest.pt"
    )
    checkpoint = None
    start_step = 0
    best_val = math.inf
    if resume_path.exists():
        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("phase") not in (None, "pretrain"):
            raise ValueError(
                f"{resume_path} não é checkpoint de pré-treino"
            )
        loaded_config = checkpoint_config(checkpoint)
        validate_resume(model_config, loaded_config, resume_path)
        model_config = loaded_config

    model = KeilinksV4(model_config)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.set_gradient_checkpointing(
        train_config.checkpoint_mode,
        train_config.checkpoint_every,
    )

    optimizer = build_optimizer(model, train_config, device)
    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0)) + 1
        best_val = float(
            checkpoint.get("best_validation_loss", math.inf)
        )
        print(f"Retomando pré-treino do passo {start_step}")
    del checkpoint

    executable = model
    compiled = False
    if (
        not args.no_compile
        and hasattr(torch, "compile")
        and device.type == "cuda"
    ):
        try:
            executable = torch.compile(
                model,
                mode=train_config.compile_mode,
                fullgraph=False,
            )
            compiled = True
            print(f"torch.compile ativo: {train_config.compile_mode}")
        except Exception as exc:
            print(f"[aviso] torch.compile falhou: {exc}")

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    context = model_config.context_length
    log_path = output_dir / "pretrain_log.jsonl"
    tokens_window = 0
    loss_window = 0.0
    micro_window = 0
    last_log = time.perf_counter()
    print(
        f"{model_config.name} | "
        f"{model.parameter_count()/1e6:.1f}M | "
        f"corpus {train_corpus.length/1e9:.3f}B tokens | {device}"
    )

    def save(
        name: str,
        step: int,
        include_optimizer: bool,
    ) -> None:
        extra = {
            "best_validation_loss": best_val,
            "model_profile": args.model,
            "train_config": asdict(train_config),
            "phase": "pretrain",
            "corpus_metadata": metadata,
            "torch_version": torch.__version__,
        }
        if include_optimizer:
            extra["optimizer"] = optimizer.state_dict()
        atomic_save(
            model.checkpoint_payload(step, **extra),
            output_dir / name,
        )

    for step in range(start_step, train_config.max_steps):
        lr = cosine_lr(
            step,
            train_config.max_steps,
            train_config.warmup_steps,
            train_config.learning_rate,
            train_config.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(train_config.grad_accum_steps):
            x, y = train_corpus.sample(
                rng,
                train_config.micro_batch_size,
                context,
            )
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mark_compiled_microbatch(compiled)
            with autocast_context(
                device, train_config.precision
            ):
                _, loss = executable(x, y, False)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Loss inválido no passo {step}: {loss}"
                    )
                if compiled:
                    # Desacopla a saída do grafo CUDA antes do backward. Isso
                    # é recomendado pelo PyTorch quando a saída poderia ser
                    # sobrescrita pela recomputação do checkpoint.
                    loss = loss.clone()
                scaled_loss = (
                    loss / train_config.grad_accum_steps
                )
            scaler.scale(scaled_loss).backward()
            tokens_window += x.numel()
            loss_window += float(loss.item())
            micro_window += 1

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(
                f"Gradiente inválido no passo {step}: {grad_norm}"
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0:
            now = time.perf_counter()
            tok_s = tokens_window / max(
                now - last_log, 1e-6
            )
            avg_loss = (
                loss_window / max(micro_window, 1)
            )
            vram = (
                torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"[{step:>7}] loss {avg_loss:.4f} | "
                f"{tok_s:,.0f} tok/s | VRAM {vram:.2f}G"
            )
            with log_path.open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps({
                    "step": step,
                    "loss": avg_loss,
                    "lr": lr,
                    "tokens_per_second": tok_s,
                    "grad_norm": float(grad_norm),
                    "vram_gb": vram,
                    "timestamp": time.time(),
                }) + "\n")
            tokens_window = 0
            loss_window = 0.0
            micro_window = 0
            last_log = now

        if (
            step > 0
            and step % train_config.eval_interval == 0
        ):
            val_loss = evaluate(
                model,
                validation_corpus,
                eval_rng,
                device,
                train_config,
                context,
                train_config.eval_batches,
            )
            print(
                f"val_loss={val_loss:.4f} | "
                f"ppl={math.exp(min(val_loss, 20)):.2f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                save(
                    "pretrain_best.pt",
                    step,
                    include_optimizer=False,
                )

        if (
            step > 0
            and step % train_config.save_interval == 0
        ):
            save(
                "pretrain_latest.pt",
                step,
                include_optimizer=True,
            )

        if pause_file.exists():
            save(
                "pretrain_paused.pt",
                step,
                include_optimizer=True,
            )
            print(
                f"Pausa solicitada em {pause_file}; checkpoint seguro salvo "
                f"no passo {step}. Remova o arquivo para retomar.",
                flush=True,
            )
            return

    save(
        "pretrain_final.pt",
        train_config.max_steps - 1,
        include_optimizer=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pré-treino causal Keilinks V4"
    )
    parser.add_argument("--model", default="core_380m_modern")
    parser.add_argument("--profile", default="rtx5050_380m")
    parser.add_argument(
        "--input",
        default="dados/v4/pretrain/pretrain_pt.txt",
    )
    parser.add_argument(
        "--vocab", default="dados/v4/pretrain/tokenizer.json"
    )
    parser.add_argument(
        "--binary-dir",
        default="dados/v4/pretrain_binary",
    )
    parser.add_argument(
        "--output", default="checkpoints/v4-pretrain"
    )
    parser.add_argument(
        "--validation-ratio", type=float, default=0.005
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument(
        "--rebuild-binary", action="store_true"
    )
    parser.add_argument(
        "--prepare-only", action="store_true"
    )
    parser.add_argument(
        "--no-compile", action="store_true"
    )
    parser.add_argument(
        "--pause-file",
        help=(
            "Arquivo sentinela para pausa segura; por padrão usa "
            "<output>/PAUSE_REQUESTED."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        pretrain(parse_args())
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "CUDA sem memória: feche o Ollama, reduza "
            "contexto/perfil ou use checkpoint full."
        ) from exc
