"""Valida arquivos JSONL conversacionais antes de packing ou treinamento."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

ALLOWED_ROLES = {"system", "user", "assistant"}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
MARKERS = ("<sistema>", "<vitor>", "<keilinks>", "<fim>")


def iter_records(paths: Iterable[Path]):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON inválido em {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Registro não é objeto em {path}:{line_number}"
                    )
                yield path, line_number, record


def validate_record(path: Path, line_number: int, record: dict) -> tuple[int, str]:
    record_id = str(record.get("id") or "").strip()
    if not record_id:
        raise ValueError(f"ID ausente em {path}:{line_number}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Messages inválido em {path}:{line_number}")

    expected = "user"
    user_turns = 0
    assistant_turns = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Mensagem {index} inválida em {path}:{line_number}"
            )
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role not in ALLOWED_ROLES or not content:
            raise ValueError(
                f"Papel/conteúdo inválido em {path}:{line_number}, mensagem {index}"
            )
        if any(marker in content for marker in MARKERS):
            raise ValueError(
                f"Marcador especial vazou em {path}:{line_number}, mensagem {index}"
            )
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError(
                f"Possível segredo em {path}:{line_number}, mensagem {index}"
            )
        if role == "system":
            if index != 0:
                raise ValueError(
                    f"System fora do início em {path}:{line_number}"
                )
            continue
        if role != expected:
            raise ValueError(
                f"Turnos não alternados em {path}:{line_number}: "
                f"esperado {expected}, recebido {role}"
            )
        if role == "user":
            user_turns += 1
            expected = "assistant"
        else:
            assistant_turns += 1
            expected = "user"

    if messages[-1].get("role") != "assistant":
        raise ValueError(f"Conversa não termina no assistant em {path}:{line_number}")
    if user_turns != assistant_turns:
        raise ValueError(f"Turnos desequilibrados em {path}:{line_number}")
    return user_turns, str(record.get("category") or "general")


def run(paths: list[Path], minimum_multiturn_ratio: float) -> dict:
    seen_ids = set()
    total = 0
    multiturn = 0
    categories = Counter()
    for path, line_number, record in iter_records(paths):
        record_id = str(record.get("id") or "").strip()
        if record_id in seen_ids:
            raise ValueError(f"ID duplicado: {record_id}")
        seen_ids.add(record_id)
        user_turns, category = validate_record(path, line_number, record)
        total += 1
        multiturn += int(user_turns >= 2)
        categories[category] += 1

    if total == 0:
        raise ValueError("Nenhuma conversa encontrada")
    ratio = multiturn / total
    if ratio < minimum_multiturn_ratio:
        raise ValueError(
            f"Apenas {ratio:.1%} das conversas são multi-turno; "
            f"mínimo exigido {minimum_multiturn_ratio:.1%}"
        )
    report = {
        "files": [str(path) for path in paths],
        "total": total,
        "multiturn": multiturn,
        "multiturn_ratio": round(ratio, 6),
        "categories": dict(categories),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida JSONL de conversas")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--min-multiturn-ratio", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 <= args.min_multiturn_ratio <= 1:
        parser.error("--min-multiturn-ratio deve ficar entre 0 e 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(
        [Path(value) for value in arguments.input],
        arguments.min_multiturn_ratio,
    )
