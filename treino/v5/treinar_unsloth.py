"""Fine-tuning QLoRA do Keilinks com Unsloth em GPUs de 8 GB.

O treino é offline e usa apenas JSONL previamente preparado/manifestado.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _hardware_manifest(torch: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        data["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_mb": round(properties.total_memory / 1024**2),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "cuda_runtime": torch.version.cuda,
        }
    return data


CHATML_ASSISTANT_HEADER = "<|im_start|>assistant\n"
CHATML_END = "<|im_end|>"
CHATML_PAD = "<|endoftext|>"


def _assistant_spans(rendered: str) -> list[tuple[int, int]]:
    """Retorna os trechos de respostas no formato ChatML do tokenizer Qwen."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        header_at = rendered.find(CHATML_ASSISTANT_HEADER, cursor)
        if header_at < 0:
            break
        body_start = header_at + len(CHATML_ASSISTANT_HEADER)
        end_at = rendered.find(CHATML_END, body_start)
        if end_at < 0:
            raise ValueError("chat template Qwen gerou uma resposta sem <|im_end|>")
        spans.append((body_start, end_at + len(CHATML_END)))
        cursor = end_at + len(CHATML_END)
    if not spans:
        raise ValueError("chat template não encontrou nenhum turno assistant")
    return spans


def _tokenize_with_assistant_labels(
    example: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """Mascara todos os tokens que não pertencem a turnos da assistente.

    O chat template atual de Qwen não expõe `{% generation %}`, portanto o
    `assistant_only_loss` automático do TRL não consegue criar a máscara. A
    máscara é construída pelos offsets dos trechos assistant no ChatML já
    renderizado, incluindo apenas o corpo e o terminador de cada resposta.
    """
    messages = example["messages"]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = list(tokenized["input_ids"])
    offsets = tokenized["offset_mapping"]
    spans = _assistant_spans(rendered)
    labels = [
        token_id if any(offset_start < end and offset_end > start for start, end in spans) else -100
        for token_id, (offset_start, offset_end) in zip(full_ids, offsets, strict=True)
    ]

    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    if not any(token != -100 for token in labels):
        raise ValueError("exemplo não possui tokens de resposta da assistente após truncamento")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def train(args: argparse.Namespace) -> Path:
    try:
        # Unsloth precisa vir antes de transformers/TRL para aplicar seus patches.
        import torch
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Dependências de treino ausentes. Execute scripts/setup_unsloth.ps1 no Python 3.13."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("Treino Unsloth requer CUDA; nenhuma GPU NVIDIA foi encontrada.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_path = Path(args.train_data)
    validation_path = Path(args.validation_data)
    if not train_path.exists() or not validation_path.exists():
        raise SystemExit("Dados preparados ausentes. Rode: python -m treino.v5.preparar_sft")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_steps != args.eval_steps:
        raise SystemExit(
            "--save-steps deve ser igual a --eval-steps para promover o melhor checkpoint"
        )
    bf16 = torch.cuda.is_bf16_supported()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # O checkpoint Qwen usa o terminador ChatML real. Alguns caminhos de
    # compatibilidade do TRL/Transformers ainda tentam usar o placeholder
    # "<EOS_TOKEN>", que não pertence ao vocabulário e interrompe o treino.
    # Fixamos ambos os papéis no token existente antes de construir o trainer.
    if tokenizer.convert_tokens_to_ids(CHATML_END) is None:
        raise RuntimeError(f"O tokenizer do modelo não contém o token {CHATML_END!r}")
    if tokenizer.convert_tokens_to_ids(CHATML_PAD) is None:
        raise RuntimeError(f"O tokenizer do modelo não contém o token {CHATML_PAD!r}")
    tokenizer.eos_token = CHATML_END
    tokenizer.pad_token = CHATML_PAD
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
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
        _tokenize_with_assistant_labels,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length},
        remove_columns=dataset["train"].column_names,
        desc="Mascareando loss fora das respostas da Keilinks",
    )
    config_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=max(1, round(args.max_steps * 0.05)),
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=max(1, min(args.eval_steps, args.max_steps)),
        save_strategy="steps",
        save_steps=max(1, min(args.save_steps, args.max_steps)),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=args.seed,
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        tf32=True,
        max_length=args.max_seq_length,
        packing=False,
        # Os labels já são mascarados acima, pois o template Qwen não fornece
        # máscara nativa de geração para o TRL atual.
        assistant_only_loss=False,
        eos_token=CHATML_END,
        pad_token=CHATML_PAD,
    )
    training_args = SFTConfig(**config_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    metrics = trainer.evaluate()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": args.model,
        "output_adapter": str(output_dir),
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
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino QLoRA Unsloth da Keilinks V5")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default="keilinks_data/training/v2/train.jsonl")
    parser.add_argument(
        "--validation-data",
        default="keilinks_data/training/v2/validation.jsonl",
    )
    parser.add_argument("--output", default="checkpoints/keilinks-qwen3-4b-lora")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    trained_path = train(parse_args())
    print(f"Adapter salvo em: {trained_path}")
