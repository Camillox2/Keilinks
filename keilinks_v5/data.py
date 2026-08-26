"""Preparação versionada de dados conversacionais para SFT/QLoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant"}
FORBIDDEN_TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CPF_RE = re.compile(r"\b\d{3}[.\s-]?\d{3}[.\s-]?\d{3}[-\s]?\d{2}\b")
PHONE_RE = re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[-\s]?\d{4}\b")
TOKEN_RE = re.compile(r"\b(?:sk|hf|ghp)[_-][A-Za-z0-9_-]{16,}\b", flags=re.IGNORECASE)
WORD_RE = re.compile(r"[\wÀ-ÿ]{2,}", flags=re.UNICODE)


@dataclass(frozen=True)
class PreparedDataset:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    train_examples: int
    validation_examples: int


def redact_sensitive_text(value: str) -> str:
    """Redação conservadora de padrões de PII e tokens antes de treinar."""
    value = EMAIL_RE.sub("[EMAIL_REMOVIDO]", value)
    value = IP_RE.sub("[IP_REMOVIDO]", value)
    value = CPF_RE.sub("[CPF_REMOVIDO]", value)
    value = PHONE_RE.sub("[TELEFONE_REMOVIDO]", value)
    return TOKEN_RE.sub("[TOKEN_REMOVIDO]", value)


def _validated_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        raise ValueError("messages deve ter ao menos duas mensagens")
    messages: list[dict[str, str]] = []
    previous_role = ""
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError("cada mensagem deve ser objeto")
        role = str(raw.get("role", "")).strip().lower()
        content = str(raw.get("content", "")).strip()
        if role not in ALLOWED_ROLES or not content:
            raise ValueError("papel ou conteúdo de mensagem inválido")
        if any(marker in content for marker in FORBIDDEN_TEMPLATE_MARKERS):
            raise ValueError("conteúdo contém marcador reservado do template")
        if role == previous_role and role != "system":
            raise ValueError("papéis consecutivos inválidos")
        messages.append({"role": role, "content": redact_sensitive_text(content)})
        previous_role = role
    if messages[-1]["role"] != "assistant":
        raise ValueError("a conversa deve terminar em assistant")
    return messages


def iter_conversations(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                messages = _validated_messages(raw.get("messages"))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            record_id = str(raw.get("id") or f"{path.stem}-{line_number}")
            group_id = str(raw.get("group_id") or record_id)
            yield {
                "id": record_id,
                "group_id": group_id,
                "source": str(raw.get("source", "unknown")),
                "license": str(raw.get("license", "unknown")),
                "messages": messages,
            }


def _is_validation(group_id: str, validation_percent: int) -> bool:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return digest[0] % 100 < validation_percent


def _normalize_for_holdout(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return set(WORD_RE.findall(normalized))


def _holdout_prompts(paths: Iterable[Path]) -> list[tuple[Path, str, set[str]]]:
    prompts: list[tuple[Path, str, set[str]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    prompt = str(record["prompt"]).strip()
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"holdout inválido em {path}:{line_number}: {exc}") from exc
                tokens = _normalize_for_holdout(prompt)
                if tokens:
                    prompts.append((path, prompt, tokens))
    return prompts


def _assert_no_holdout_overlap(
    records: Iterable[dict[str, Any]], holdouts: list[tuple[Path, str, set[str]]]
) -> None:
    """Bloqueia vazamento exato ou quase-exato de prompts de avaliação congelada."""
    for record in records:
        for message in record["messages"]:
            if message["role"] != "user":
                continue
            candidate = message["content"]
            candidate_tokens = _normalize_for_holdout(candidate)
            if len(candidate_tokens) < 4:
                continue
            for holdout_path, holdout_prompt, holdout_tokens in holdouts:
                overlap = len(candidate_tokens & holdout_tokens)
                similarity = overlap / len(candidate_tokens | holdout_tokens)
                if similarity >= 0.60:
                    raise ValueError(
                        "possível vazamento de avaliação: "
                        f"registro {record['id']!r} se sobrepõe ao holdout {holdout_path} "
                        f"({similarity:.2f}): {holdout_prompt!r}"
                    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def prepare_sft_dataset(
    input_paths: list[Path],
    output_dir: Path,
    validation_percent: int = 10,
    holdout_paths: list[Path] | None = None,
) -> PreparedDataset:
    if not 1 <= validation_percent < 50:
        raise ValueError("validation_percent deve estar entre 1 e 49")

    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for input_path in input_paths:
        for record in iter_conversations(input_path):
            if record["id"] in seen_ids:
                raise ValueError(f"id duplicado: {record['id']}")
            seen_ids.add(record["id"])
            records.append(record)
            source_counts[record["source"]] += 1
            licenses[record["license"]] += 1
    if not records:
        raise ValueError("nenhuma conversa válida encontrada")
    holdout_paths = holdout_paths or []
    if holdout_paths:
        _assert_no_holdout_overlap(records, _holdout_prompts(holdout_paths))

    train_records = [
        record for record in records if not _is_validation(record["group_id"], validation_percent)
    ]
    validation_records = [
        record for record in records if _is_validation(record["group_id"], validation_percent)
    ]
    if not train_records or not validation_records:
        raise ValueError("split vazio; adicione mais conversas ou ajuste validation_percent")

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "manifest.json"
    train_count = _write_jsonl(train_path, train_records)
    validation_count = _write_jsonl(validation_path, validation_records)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": [str(path) for path in input_paths],
        "holdout_paths": [str(path) for path in holdout_paths],
        "validation_percent": validation_percent,
        "train_examples": train_count,
        "validation_examples": validation_count,
        "sources": dict(source_counts),
        "licenses": dict(licenses),
        "redaction": "email, IP, CPF, telefone e tokens de padrões conhecidos",
        "training_policy": "offline_only; no user conversations without explicit consent",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return PreparedDataset(
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        train_examples=train_count,
        validation_examples=validation_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara conversas aprovadas para SFT Keilinks V5")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="JSONL de conversas; pode ser repetido",
    )
    parser.add_argument("--output-dir", default="keilinks_data/training/v1")
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument(
        "--holdout",
        action="append",
        default=[],
        help="JSONL congelado com campo prompt; dados que coincidam são bloqueados",
    )
    args = parser.parse_args()
    inputs = [Path(path) for path in args.input] or [Path("dados/v4/conversas_curadas_v4.jsonl")]
    default_holdout = Path("dados/v4/eval/keilinks_eval_v4.jsonl")
    holdouts = [Path(path) for path in args.holdout]
    if not holdouts and default_holdout.exists():
        holdouts = [default_holdout]
    result = prepare_sft_dataset(
        inputs,
        Path(args.output_dir),
        args.validation_percent,
        holdouts,
    )
    print(
        json.dumps(
            {
                "train": str(result.train_path),
                "validation": str(result.validation_path),
                "manifest": str(result.manifest_path),
                "train_examples": result.train_examples,
                "validation_examples": result.validation_examples,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
