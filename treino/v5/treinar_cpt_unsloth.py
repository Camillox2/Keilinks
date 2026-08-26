"""Continued pretraining QLoRA do Keilinks em corpus PT-BR auditado.

Este estágio adapta linguagem e domínio; ele não substitui o SFT de instrução
nem promove automaticamente um checkpoint. Use sempre um modelo *Base*, não o
adaptador Instruct em produção, e rode SFT/DPO somente depois deste candidato
passar pelas avaliações.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keilinks_v5.cpt import require_approved_cpt_manifest
from treino.v5.treinar_unsloth import TARGET_MODULES, _hardware_manifest

DEFAULT_BASE_MODEL = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenize_cpt_batch(
    batch: dict[str, list[str]], tokenizer: Any, max_length: int
) -> dict[str, list[list[int]]]:
    eos_token = tokenizer.eos_token
    if not eos_token:
        raise ValueError("tokenizer do modelo Base não possui eos_token")
    texts = [text.rstrip() + eos_token for text in batch["text"]]
    return tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=False)


def _training_args(training_args_cls: Any, args: argparse.Namespace, bf16: bool) -> Any:
    common = {
        "output_dir": str(Path(args.output)),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "warmup_steps": max(1, round(args.max_steps * 0.05)),
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": max(1, min(args.eval_steps, args.max_steps)),
        "save_strategy": "steps",
        "save_steps": max(1, min(args.save_steps, args.max_steps)),
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "seed": args.seed,
        "bf16": bf16,
        "fp16": not bf16,
        "report_to": "none",
        "tf32": True,
    }
    try:
        return training_args_cls(**common)
    except TypeError:
        # Transformers anteriores usavam o nome longo deste argumento.
        common["evaluation_strategy"] = common.pop("eval_strategy")
        return training_args_cls(**common)


def train(args: argparse.Namespace) -> Path:
    data_manifest_path = Path(args.data_manifest)
    try:
        data_manifest = require_approved_cpt_manifest(data_manifest_path)
    except ValueError as exc:
        raise SystemExit(f"CPT bloqueado: {exc}") from None
    train_path = Path(args.train_data)
    validation_path = Path(args.validation_data)
    if not train_path.exists() or not validation_path.exists():
        raise SystemExit("Dados CPT ausentes; rode treino.v5.preparar_cpt prepare primeiro.")
    try:
        # Unsloth precisa vir antes de transformers para aplicar seus patches.
        import torch
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Dependências de treino ausentes. Execute scripts/setup_unsloth.ps1 no Python 3.13."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("Treino Unsloth requer CUDA; nenhuma GPU NVIDIA foi encontrada.")
    if args.save_steps != args.eval_steps:
        raise SystemExit(
            "--save-steps deve ser igual a --eval-steps para preservar o melhor checkpoint"
        )
    if not 128 <= args.max_seq_length <= 4096:
        raise SystemExit("--max-seq-length deve ficar entre 128 e 4096 na RTX de 8 GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if not tokenizer.eos_token:
        raise RuntimeError("o tokenizer do modelo Base não possui eos_token")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=True,
        loftq_config=None,
    )
    model.config.use_cache = False

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    dataset = dataset.map(
        _tokenize_cpt_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length},
        remove_columns=dataset["train"].column_names,
        desc="Tokenizando corpus CPT",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = _training_args(TrainingArguments, args, torch.cuda.is_bf16_supported())
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": collator,
    }
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    metrics = trainer.evaluate()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    output_manifest = {
        "schema_version": 1,
        "purpose": "continued_pretraining_qlora",
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": args.model,
        "output_adapter": str(output_dir),
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": _sha256_file(data_manifest_path),
        "data_status": data_manifest["status"],
        "train_data": str(train_path),
        "validation_data": str(validation_path),
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "gradient_accumulation": args.gradient_accumulation,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "target_modules": TARGET_MODULES,
            "rslora": True,
        },
        "hardware": _hardware_manifest(torch),
        "metrics": metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "promotion": "manual_only; execute SFT/evaluation before serving",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continued pretraining QLoRA Unsloth da Keilinks")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-manifest", default="keilinks_data/cpt/pilot/manifest.json")
    parser.add_argument("--train-data", default="keilinks_data/cpt/pilot/train.jsonl")
    parser.add_argument("--validation-data", default="keilinks_data/cpt/pilot/validation.jsonl")
    parser.add_argument("--output", default="checkpoints/keilinks-qwen3-4b-base-cpt-pilot")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    trained_path = train(parse_args())
    print(f"Adaptador CPT salvo em: {trained_path}")
