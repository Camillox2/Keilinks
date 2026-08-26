"""Coleta conversas PT-BR rastreáveis para SFT do Keilinks 380M.

Fontes humanas e de feedback ficam separadas de instruções sintéticas. Isso
permite que o mix posterior limite dados sintéticos, em vez de deixar o maior
arquivo decidir a personalidade do assistente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keilinks_v5.data import FORBIDDEN_TEMPLATE_MARKERS, redact_sensitive_text
from treino.v4.coletar_corpus import SqliteDeduper
from treino.v4.dataset import canonical_conversation, normalize_text, validate_messages
from treino.v4.preparar_dados import good_oasst_message, oasst_conversation

ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = ROOT / "dados" / "v4" / "sft"


@dataclass(frozen=True)
class ConversationSource:
    key: str
    dataset_id: str
    config: str | None
    license: str
    source_url: str
    acknowledgement: str
    synthetic: bool
    note: str


SOURCES: dict[str, ConversationSource] = {
    "oasst1_pt": ConversationSource(
        key="oasst1_pt",
        dataset_id="OpenAssistant/oasst1",
        config=None,
        license="Apache-2.0",
        source_url="https://huggingface.co/datasets/OpenAssistant/oasst1",
        acknowledgement="oasst1_apache",
        synthetic=False,
        note=(
            "Primeira versão das conversas OpenAssistant; usa apenas cadeias "
            "PT revisadas, não removidas e não sintéticas."
        ),
    ),
    "oasst2_pt": ConversationSource(
        key="oasst2_pt",
        dataset_id="OpenAssistant/oasst2",
        config=None,
        license="Apache-2.0",
        source_url="https://huggingface.co/datasets/OpenAssistant/oasst2",
        acknowledgement="oasst2_apache",
        synthetic=False,
        note=(
            "Conversas avaliadas pela comunidade; usa apenas cadeias PT "
            "revisadas e não removidas."
        ),
    ),
    "aya_pt": ConversationSource(
        key="aya_pt",
        dataset_id="botbotrobotics/aya_dataset_pt",
        config=None,
        license="Apache-2.0",
        source_url="https://huggingface.co/datasets/botbotrobotics/aya_dataset_pt",
        acknowledgement="aya_apache",
        synthetic=False,
        note="Pequeno conjunto de anotações originais em português; preserva instrução e resposta.",
    ),
    "tucano_sft": ConversationSource(
        key="tucano_sft",
        dataset_id="TucanoBR/Tucano-SFT",
        config=None,
        license="MIT + Apache-2.0 (componentes declarados no dataset card)",
        source_url="https://huggingface.co/datasets/TucanoBR/Tucano-SFT",
        acknowledgement="tucano_sft_components",
        synthetic=True,
        note="Conversas geradas por modelos ajustados; entram com teto de proporção no mix.",
    ),
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def dataset_revision(dataset_id: str) -> str:
    try:
        from huggingface_hub import HfApi

        return str(HfApi().dataset_info(dataset_id, timeout=20).sha or "")
    except Exception:
        return "unavailable"


def clean_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            return []
        content = normalize_text(redact_sensitive_text(str(message.get("content") or "")))
        if (
            not content
            or "\ufffd" in content
            or any(marker in content for marker in FORBIDDEN_TEMPLATE_MARKERS)
        ):
            return []
        cleaned.append({"role": role, "content": content})
    return cleaned if validate_messages(cleaned) else []


def make_record(
    source: ConversationSource,
    messages: list[dict[str, str]],
    *,
    source_id: str,
    group_id: str,
    category: str,
    revision: str,
) -> dict[str, object]:
    canonical = canonical_conversation(messages)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "id": digest,
        "group_id": group_id or digest,
        "source": source.key,
        "source_id": source_id,
        "dataset_id": source.dataset_id,
        "dataset_config": source.config,
        "dataset_revision": revision,
        "license": source.license,
        "source_url": source.source_url,
        "synthetic": source.synthetic,
        "category": category,
        "content_sha256": digest,
        "messages": messages,
    }


def collect_oasst2(
    source: ConversationSource, limit: int, revision: str
) -> Iterable[dict[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset(source.dataset_id)
    items = [dict(item) for split in ("train", "validation") for item in dataset[split]]
    by_id = {str(item.get("message_id")): item for item in items if item.get("message_id")}

    def chain_is_human(item: dict[str, Any]) -> bool:
        current: dict[str, Any] | None = item
        seen_ids: set[str] = set()
        while current is not None:
            message_id = str(current.get("message_id") or "")
            if not message_id or message_id in seen_ids or bool(current.get("synthetic", False)):
                return False
            seen_ids.add(message_id)
            parent_id = str(current.get("parent_id") or "")
            current = by_id.get(parent_id) if parent_id else None
        return True

    written = 0
    for item in items:
        if item.get("role") != "assistant" or bool(item.get("synthetic", False)):
            continue
        if not good_oasst_message(item) or not chain_is_human(item):
            continue
        chain = oasst_conversation(item, by_id)
        if not chain:
            continue
        messages = clean_messages(chain)
        if not messages:
            continue
        yield make_record(
            source,
            messages,
            source_id=str(item.get("message_id") or ""),
            group_id=str(item.get("message_tree_id") or item.get("message_id") or ""),
            category="human_feedback_dialogue",
            revision=revision,
        )
        written += 1
        if written >= limit:
            break


def collect_aya(
    source: ConversationSource, limit: int, revision: str
) -> Iterable[dict[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset(source.dataset_id, split="train", streaming=True)
    written = 0
    for index, item in enumerate(dataset):
        if str(item.get("annotation_type") or "") != "original-annotations":
            continue
        messages = clean_messages(
            [
                {"role": "user", "content": str(item.get("inputs") or "")},
                {"role": "assistant", "content": str(item.get("targets") or "")},
            ]
        )
        if not messages:
            continue
        source_id = str(item.get("__index_level_0__") or index)
        yield make_record(
            source,
            messages,
            source_id=source_id,
            group_id=source_id,
            category="human_instruction",
            revision=revision,
        )
        written += 1
        if written >= limit:
            break


def collect_tucano(
    source: ConversationSource, limit: int, revision: str
) -> Iterable[dict[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset(source.dataset_id, split="train", streaming=True)
    written = 0
    for index, item in enumerate(dataset):
        messages = clean_messages(list(item.get("conversations") or []))
        if not messages:
            continue
        source_id = str(index)
        yield make_record(
            source,
            messages,
            source_id=source_id,
            group_id=source_id,
            category="synthetic_instruction",
            revision=revision,
        )
        written += 1
        if written >= limit:
            break


COLLECTORS = {
    "oasst1_pt": collect_oasst2,
    "oasst2_pt": collect_oasst2,
    "aya_pt": collect_aya,
    "tucano_sft": collect_tucano,
}


def parse_limits(values: list[str], sources: list[str]) -> dict[str, int]:
    defaults = {
        "oasst1_pt": 100_000,
        "oasst2_pt": 100_000,
        "aya_pt": 10_000,
        "tucano_sft": 100_000,
    }
    result = {key: defaults[key] for key in sources}
    for value in values:
        try:
            key, raw_limit = value.split("=", 1)
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError("Use --limit fonte=numero, por exemplo tucano_sft=100000") from exc
        if key not in result or limit < 1:
            raise ValueError(f"Limite inválido: {value}")
        result[key] = limit
    return result


def collect_one(
    source: ConversationSource, output_dir: Path, *, limit: int, accepted_terms: set[str]
) -> dict[str, object]:
    if source.acknowledgement not in accepted_terms:
        raise ValueError(
            f"{source.key} exige --accept-terms {source.acknowledgement}; leia {source.source_url}"
        )
    output = output_dir / f"{source.key}.jsonl"
    if output.exists():
        raise FileExistsError(f"Saída já existe: {output}")
    deduper = SqliteDeduper(output_dir / "conversation_dedup.sqlite3")
    revision = dataset_revision(source.dataset_id)
    rejected: Counter[str] = Counter()
    written = 0
    started = now()
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            for record in COLLECTORS[source.key](source, limit, revision):
                digest = str(record["content_sha256"])
                if not deduper.add(digest):
                    rejected["exact_duplicate"] += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                if written % 2_000 == 0:
                    print(f"{source.key}: {written:,} conversas")
    finally:
        deduper.close()
    manifest = {
        "schema_version": 1,
        "kind": "keilinks_v4_sft_source",
        "status": "complete",
        "source": asdict(source),
        "dataset_revision": revision,
        "output": str(output),
        "started_at": started,
        "finished_at": now(),
        "limit": limit,
        "records_written": written,
        "accepted_terms": sorted(accepted_terms),
        "gates": [
            "source-specific language/annotation filtering",
            "PII pattern redaction",
            "reserved template marker exclusion",
            "conversation role validation",
            "sqlite exact conversation deduplication",
        ],
        "rejections": dict(sorted(rejected.items())),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coleta conversas PT-BR para SFT V4")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source", action="append", choices=sorted(SOURCES))
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", action="append", default=[])
    parser.add_argument("--accept-terms", action="append", default=[])
    args = parser.parse_args()
    if not args.list and not args.source:
        parser.error("Informe ao menos um --source ou use --list")
    return args


def main() -> None:
    args = parse_args()
    if args.list:
        print(
            json.dumps(
                {key: asdict(value) for key, value in SOURCES.items()}, ensure_ascii=False, indent=2
            )
        )
        return
    assert args.source is not None
    run_id = f"public-conversations-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    output_dir = Path(args.output_dir) if args.output_dir else SFT_ROOT / run_id
    if output_dir.exists():
        raise SystemExit(f"Diretório já existe: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    limits = parse_limits(args.limit, args.source)
    terms = set(args.accept_terms)
    manifests: list[dict[str, object]] = []
    try:
        for source_key in args.source:
            manifests.append(
                collect_one(
                    SOURCES[source_key], output_dir, limit=limits[source_key], accepted_terms=terms
                )
            )
    except Exception as exc:
        (output_dir / "run.manifest.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise
    run_manifest = {
        "schema_version": 1,
        "kind": "keilinks_v4_sft_run",
        "status": "complete",
        "sources": [item["source"]["key"] for item in manifests],
        "records": sum(int(item["records_written"]) for item in manifests),
        "source_manifests": [f"{item['source']['key']}.manifest.json" for item in manifests],
    }
    (output_dir / "run.manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
