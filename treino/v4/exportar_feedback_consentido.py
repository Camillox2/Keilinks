"""Exporta somente feedback humano aprovado para um treino futuro da Keilinks.

O runtime nunca chama este módulo. A pessoa precisa ter dado consentimento,
um revisor precisa aprovar cada item e o consentimento ainda precisa estar
ativo no instante da exportação. Isso impede que um clique ou uma conversa
ruim altere pesos do modelo em produção.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dados.database import feedback_treino_aprovados, feedback_treino_revisar

ROOT = Path(__file__).resolve().parents[2]


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_approved_feedback(output_dir: Path) -> dict[str, Any]:
    """Cria artefatos separados para SFT e pares de preferência (DPO/ORPO)."""

    sft_records: list[dict[str, Any]] = []
    preference_records: list[dict[str, Any]] = []
    skipped_negative_without_correction = 0
    for feedback in feedback_treino_aprovados():
        prompt = str(feedback["prompt"])
        response = str(feedback["response"])
        correction = str(feedback.get("correction") or "")
        common = {
            "source": "keilinks_user_feedback_reviewed_v1",
            "license": "user-explicit-consent-reviewed",
            "source_id": f"feedback:{feedback['id']}",
            "synthetic": False,
            "category": "reviewed_user_feedback",
            "feedback_id": feedback["id"],
            "quality_score": feedback["quality_score"],
            "redacted": feedback["redacted"],
        }
        if feedback["rating"] == "up":
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            record = {
                **common,
                "group_id": f"feedback:{feedback['id']}",
                "messages": messages,
            }
            record["id"] = _stable_id("feedback_sft", record)
            sft_records.append(record)
            continue

        if not correction:
            skipped_negative_without_correction += 1
            continue
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": correction},
        ]
        sft_record = {
            **common,
            "group_id": f"feedback:{feedback['id']}",
            "messages": messages,
        }
        sft_record["id"] = _stable_id("feedback_sft", sft_record)
        sft_records.append(sft_record)
        preference_record = {
            "id": _stable_id(
                "feedback_preference",
                {"prompt": prompt, "chosen": correction, "rejected": response},
            ),
            "source": common["source"],
            "license": common["license"],
            "feedback_id": feedback["id"],
            "prompt": prompt,
            "chosen": correction,
            "rejected": response,
            "quality_score": feedback["quality_score"],
            "redacted": feedback["redacted"],
        }
        preference_records.append(preference_record)

    output_dir = output_dir.resolve()
    sft_path = output_dir / "approved_sft.jsonl"
    preferences_path = output_dir / "approved_preferences.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(sft_path, sft_records)
    _write_jsonl(preferences_path, preference_records)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "policy": (
            "explicit_consent_at_capture_and_export; human_review_required; "
            "offline_training_only; pii_redacted_before_queue"
        ),
        "approved_sft_examples": len(sft_records),
        "approved_preference_pairs": len(preference_records),
        "skipped_negative_without_correction": skipped_negative_without_correction,
        "sft_path": str(sft_path),
        "preferences_path": str(preferences_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aprova explicitamente e exporta feedback consentido da Keilinks"
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "dados" / "v4" / "feedback_reviewed"),
        help="Pasta de saída; não substitui o SFT base.",
    )
    parser.add_argument(
        "--approve-id",
        action="append",
        type=int,
        default=[],
        help="ID de feedback a aprovar; pode ser repetido.",
    )
    parser.add_argument(
        "--reject-id",
        action="append",
        type=int,
        default=[],
        help="ID de feedback a rejeitar; pode ser repetido.",
    )
    parser.add_argument("--note", default="", help="Nota curta da revisão humana.")
    args = parser.parse_args()

    for feedback_id in args.approve_id:
        if feedback_treino_revisar(feedback_id, approved=True, note=args.note) is None:
            raise SystemExit(f"Feedback {feedback_id} não encontrado")
    for feedback_id in args.reject_id:
        if feedback_treino_revisar(feedback_id, approved=False, note=args.note) is None:
            raise SystemExit(f"Feedback {feedback_id} não encontrado")

    print(json.dumps(export_approved_feedback(Path(args.output_dir)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
