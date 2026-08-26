"""Converte feedback explicitamente aprovado em pares DPO auditáveis.

Este módulo não treina nada. Ele só cria dados quando uma pessoa revisora
alterou o registro consentido para ``status=approved``. Dessa forma uma nota
automática, uma correção maliciosa ou uma resposta de outro modelo nunca entra
diretamente na política da Keilinks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keilinks_v5.data import FORBIDDEN_TEMPLATE_MARKERS, redact_sensitive_text


def _clean(value: Any, *, field: str, max_chars: int) -> str:
    text = redact_sensitive_text(str(value or "").strip())
    if not text:
        raise ValueError(f"{field} vazio")
    if len(text) > max_chars:
        raise ValueError(f"{field} excede {max_chars} caracteres")
    if any(marker in text for marker in FORBIDDEN_TEMPLATE_MARKERS):
        raise ValueError(f"{field} contém marcador reservado do template")
    return text


def prepare_preferences(input_path: Path, output_path: Path) -> dict[str, object]:
    """Gera pares ``prompt/chosen/rejected`` apenas de feedback aprovado."""
    selected: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped["invalid_json"] += 1
                continue
            if raw.get("status") != "approved":
                skipped["not_human_approved"] += 1
                continue
            if raw.get("consent_to_training") is not True:
                skipped["no_training_consent"] += 1
                continue
            if raw.get("rating") != "down":
                skipped["not_a_correction"] += 1
                continue
            try:
                prompt = _clean(raw.get("prompt"), field="prompt", max_chars=20_000)
                chosen = _clean(raw.get("correction"), field="correction", max_chars=20_000)
                rejected = _clean(raw.get("response"), field="response", max_chars=20_000)
            except ValueError as exc:
                skipped[str(exc)] += 1
                continue
            if chosen == rejected:
                skipped["correction_equals_response"] += 1
                continue
            selected.append(
                {
                    "id": f"feedback-{line_number}",
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "consented_human_reviewed_feedback",
                    "license": "proprietary-keilinks",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "approved_pairs": len(selected),
        "skipped": dict(sorted(skipped.items())),
        "policy": (
            "human_review_required; explicit_consent_required; offline_only; "
            "run frozen evaluation before promotion"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara pares DPO de feedback humano aprovado")
    parser.add_argument("--input", default="keilinks_data/feedback/consented_feedback.jsonl")
    parser.add_argument("--output", default="keilinks_data/preferences/approved_dpo.jsonl")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"feedback não encontrado: {input_path}")
    manifest = prepare_preferences(input_path, Path(args.output))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
