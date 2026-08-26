"""Preparação auditável de corpus para continued pretraining (CPT/DAPT).

O módulo deliberadamente não baixa dados, não aceita termos e não aprova um
corpus. Ele recebe somente coletas já manifestadas, elimina registros sem
proveniência verificável ou com padrões sensíveis e cria um artefato que ainda
precisa de aprovação humana antes de o treinador poder usá-lo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keilinks_v5.data import FORBIDDEN_TEMPLATE_MARKERS, WORD_RE, redact_sensitive_text

REQUIRED_RECORD_FIELDS = (
    "text",
    "source_key",
    "dataset_id",
    "license",
    "source_url",
    "content_sha256",
)
CONTROL_MARKERS = (*FORBIDDEN_TEMPLATE_MARKERS, "<tool_call>", "</tool_call>")


@dataclass(frozen=True)
class PreparedCPTDataset:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    train_documents: int
    validation_documents: int
    rejected_documents: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_validation(content_hash: str, validation_percent: int) -> bool:
    return int(content_hash[:2], 16) % 100 < validation_percent


def _manifest_for_input(input_path: Path) -> Path:
    return input_path.with_suffix(".manifest.json")


def _read_collection_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifesto de coleta inválido: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"manifesto de coleta deve ser um objeto: {path}")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"manifesto de coleta não contém source: {path}")
    acknowledgement = str(source.get("acknowledgement", "")).strip()
    accepted_terms = {str(term).strip() for term in manifest.get("accepted_terms", [])}
    if not acknowledgement or acknowledgement not in accepted_terms:
        raise ValueError(
            "manifesto de coleta não prova o aceite necessário para "
            f"{source.get('key', '?')}: {path}"
        )
    return manifest


def _validate_record(raw: Any) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not_object"
    missing = [field for field in REQUIRED_RECORD_FIELDS if not str(raw.get(field, "")).strip()]
    if missing:
        return None, f"missing_{missing[0]}"
    raw_text = raw["text"]
    if not isinstance(raw_text, str):
        return None, "text_not_string"
    text = unicodedata.normalize("NFC", raw_text).strip()
    if len(text) < 160:
        return None, "too_short"
    if any(marker in text for marker in CONTROL_MARKERS):
        return None, "reserved_marker"
    if redact_sensitive_text(text) != text:
        return None, "sensitive_pattern"
    content_hash = str(raw["content_sha256"]).strip().lower()
    if _sha256_bytes(text.encode("utf-8")) != content_hash:
        return None, "content_hash_mismatch"
    tokens = WORD_RE.findall(text.casefold())
    if len(tokens) < 40:
        return None, "too_few_tokens"
    if len(set(tokens)) / len(tokens) < 0.20:
        return None, "low_token_diversity"
    return (
        {
            "id": content_hash,
            "group_id": content_hash,
            "text": text,
            "source_key": str(raw["source_key"]).strip(),
            "dataset_id": str(raw["dataset_id"]).strip(),
            "license": str(raw["license"]).strip(),
            "source_url": str(raw["source_url"]).strip(),
            "content_sha256": content_hash,
        },
        None,
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def prepare_cpt_dataset(
    input_paths: list[Path],
    output_dir: Path,
    *,
    manifest_paths: list[Path] | None = None,
    validation_percent: int = 5,
) -> PreparedCPTDataset:
    """Cria splits CPT somente de coletas que comprovem aceite de termos.

    A saída recebe ``status=requires_manual_approval``. Isso torna a curadoria
    humana uma condição técnica para que ``treinar_cpt_unsloth`` possa rodar.
    """
    if not input_paths:
        raise ValueError("informe ao menos um corpus de entrada")
    if not 1 <= validation_percent < 50:
        raise ValueError("validation_percent deve estar entre 1 e 49")
    if manifest_paths is not None and len(manifest_paths) != len(input_paths):
        raise ValueError("--source-manifest deve aparecer uma vez para cada --input")

    resolved_manifests = manifest_paths or [_manifest_for_input(path) for path in input_paths]
    collection_manifests = [_read_collection_manifest(path) for path in resolved_manifests]
    accepted_sources: dict[str, set[tuple[str, str, str]]] = {}
    for collection_manifest in collection_manifests:
        source = collection_manifest["source"]
        source_key = str(source["key"]).strip()
        accepted_sources.setdefault(source_key, set()).add(
            (
                str(source["dataset_id"]).strip(),
                str(source["license"]).strip(),
                str(source["source_url"]).strip(),
            )
        )

    accepted: list[dict[str, str]] = []
    rejections: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    source_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {input_path}:{line_number}") from exc
                record, reason = _validate_record(raw)
                if reason:
                    rejections[reason] += 1
                    continue
                assert record is not None
                expected_source = (
                    record["dataset_id"],
                    record["license"],
                    record["source_url"],
                )
                if expected_source not in accepted_sources.get(record["source_key"], set()):
                    rejections["source_not_in_accepted_manifest"] += 1
                    continue
                if record["content_sha256"] in seen_hashes:
                    rejections["duplicate_content"] += 1
                    continue
                seen_hashes.add(record["content_sha256"])
                source_counts[record["source_key"]] += 1
                license_counts[record["license"]] += 1
                accepted.append(record)

    if not accepted:
        raise ValueError("nenhum documento passou pelos gates de CPT")
    train_records = [
        record
        for record in accepted
        if not _is_validation(record["content_sha256"], validation_percent)
    ]
    validation_records = [
        record
        for record in accepted
        if _is_validation(record["content_sha256"], validation_percent)
    ]
    if not train_records or not validation_records:
        raise ValueError("split CPT vazio; forneça mais documentos ou ajuste validation_percent")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "manifest.json"
    for path in (train_path, validation_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"saída já existe: {path}; escolha outro diretório")
    train_count = _write_jsonl(train_path, train_records)
    validation_count = _write_jsonl(validation_path, validation_records)
    manifest = {
        "schema_version": 1,
        "status": "requires_manual_approval",
        "purpose": "continued_pretraining",
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": [str(path) for path in input_paths],
        "collection_manifests": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in resolved_manifests
        ],
        "validation_percent": validation_percent,
        "train_documents": train_count,
        "validation_documents": validation_count,
        "rejected_documents": sum(rejections.values()),
        "rejection_reasons": dict(sorted(rejections.items())),
        "sources": dict(sorted(source_counts.items())),
        "licenses": dict(sorted(license_counts.items())),
        "gates_applied": [
            "explicit_collection_terms",
            "provenance_fields",
            "content_sha256",
            "exact_deduplication",
            "sensitive_pattern_exclusion",
            "reserved_marker_exclusion",
            "minimum_length_and_token_diversity",
            "deterministic_validation_split",
        ],
        "approval_required": [
            "amostra humana de qualidade e relevância",
            "revisão de licenças para o uso pretendido",
            "decontaminação contra avaliações congeladas",
            "confirmação explícita pelo responsável pelo modelo",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return PreparedCPTDataset(
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        train_documents=train_count,
        validation_documents=validation_count,
        rejected_documents=sum(rejections.values()),
    )


def approve_cpt_manifest(manifest_path: Path, approval_note: str) -> dict[str, Any]:
    """Registra uma aprovação humana explícita para um corpus já preparado."""
    note = approval_note.strip()
    if len(note) < 12:
        raise ValueError("a nota de aprovação deve explicar a revisão humana")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifesto CPT inválido: {manifest_path}") from exc
    if manifest.get("status") != "requires_manual_approval":
        raise ValueError("somente manifestos pendentes de revisão podem ser aprovados")
    manifest["status"] = "approved_for_training"
    manifest["approved_at"] = datetime.now(UTC).isoformat()
    manifest["approval_note"] = note
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def require_approved_cpt_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifesto CPT inválido: {manifest_path}") from exc
    if manifest.get("purpose") != "continued_pretraining":
        raise ValueError("manifesto não pertence a continued pretraining")
    if manifest.get("status") != "approved_for_training":
        raise ValueError(
            "corpus ainda não foi aprovado; revise e rode o comando approve com confirmação manual"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara e aprova corpus CPT Keilinks")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="Aplica gates e cria split candidato")
    prepare.add_argument("--input", action="append", required=True)
    prepare.add_argument("--source-manifest", action="append")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--validation-percent", type=int, default=5)
    approve = subcommands.add_parser(
        "approve", help="Registra revisão humana de um corpus candidato"
    )
    approve.add_argument("--manifest", required=True)
    approve.add_argument("--approval-note", required=True)
    approve.add_argument("--confirmed-manual-review", action="store_true")
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_cpt_dataset(
            [Path(path) for path in args.input],
            Path(args.output_dir),
            manifest_paths=[Path(path) for path in args.source_manifest]
            if args.source_manifest
            else None,
            validation_percent=args.validation_percent,
        )
        print(
            json.dumps(
                {
                    "train": str(result.train_path),
                    "validation": str(result.validation_path),
                    "manifest": str(result.manifest_path),
                    "train_documents": result.train_documents,
                    "validation_documents": result.validation_documents,
                    "rejected_documents": result.rejected_documents,
                },
                ensure_ascii=False,
            )
        )
        return
    if not args.confirmed_manual_review:
        raise SystemExit("approve exige --confirmed-manual-review")
    approved = approve_cpt_manifest(Path(args.manifest), args.approval_note)
    print(json.dumps({"status": approved["status"], "manifest": args.manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
