"""Coleta pequena, versionada e consciente de licença de texto PT-BR.

Isto não é um botão de "baixar a internet". Cada fonte precisa de aceite
explícito e gera um manifesto de proveniência. Os textos coletados servem para
curadoria e avaliação antes de qualquer treino; não são injetados diretamente
no adaptador em produção.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSource:
    key: str
    dataset_id: str
    config: str | None
    purpose: str
    license: str
    source_url: str
    acknowledgement: str
    blocked_without_manual_license_review: bool = False


SOURCES: dict[str, DatasetSource] = {
    "fineweb2_pt": DatasetSource(
        key="fineweb2_pt",
        dataset_id="HuggingFaceFW/fineweb-2",
        config="por_Latn",
        purpose="continued_pretraining_candidate",
        license="ODC-By-1.0 + Common Crawl Terms of Use",
        source_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        acknowledgement="fineweb2_terms",
    ),
    "carolina_pt": DatasetSource(
        key="carolina_pt",
        dataset_id="carolina-c4ai/corpus-carolina",
        config=None,
        purpose="continued_pretraining_candidate",
        license="CC-BY-4.0",
        source_url="https://huggingface.co/datasets/carolina-c4ai/corpus-carolina",
        acknowledgement="carolina_cc_by",
    ),
    "wikipedia_pt": DatasetSource(
        key="wikipedia_pt",
        dataset_id="wikimedia/wikipedia",
        config="20231101.pt",
        purpose="continued_pretraining_candidate",
        license="CC-BY-SA-3.0 + GFDL",
        source_url="https://huggingface.co/datasets/wikimedia/wikipedia",
        acknowledgement="wikipedia_sharealike",
    ),
    "culturax_pt": DatasetSource(
        key="culturax_pt",
        dataset_id="uonlp/CulturaX",
        config="pt",
        purpose="continued_pretraining_candidate",
        license="upstream mC4/OSCAR licenses; gated dataset terms",
        source_url="https://huggingface.co/datasets/uonlp/CulturaX",
        acknowledgement="culturax_terms",
    ),
    "pt_corpus_instruct": DatasetSource(
        key="pt_corpus_instruct",
        dataset_id="nicholasKluge/Pt-Corpus-Instruct",
        config=None,
        purpose="continued_pretraining_candidate_only",
        license="composite upstream licenses; includes non-commercial sources",
        source_url="https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct",
        acknowledgement="pt_corpus_instruct_license_review",
        blocked_without_manual_license_review=True,
    ),
}


def _extract_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or "").strip()


def collect(
    *,
    source: DatasetSource,
    output_path: Path,
    max_documents: int,
    max_bytes: int,
    accept_terms: set[str],
    manual_license_review: bool,
) -> dict[str, object]:
    if source.acknowledgement not in accept_terms:
        raise ValueError(
            f"{source.key} exige --accept-terms {source.acknowledgement}; leia {source.source_url}"
        )
    if source.blocked_without_manual_license_review and not manual_license_review:
        raise ValueError(
            f"{source.key} tem licenças upstream mistas; use --manual-license-review "
            "somente após revisão jurídica/uso pretendido"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("instale o extra de treino antes de coletar datasets") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"saída já existe: {output_path}; escolha outro nome")
    dataset = load_dataset(
        source.dataset_id,
        source.config,
        split="train",
        streaming=True,
    )
    written = 0
    bytes_written = 0
    seen_hashes: set[str] = set()
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for item in dataset:
            text = _extract_text(item)
            if len(text) < 160:
                continue
            encoded = text.encode("utf-8")
            if bytes_written + len(encoded) > max_bytes:
                break
            digest = hashlib.sha256(encoded).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            record = {
                "text": text,
                "source_key": source.key,
                "dataset_id": source.dataset_id,
                "dataset_config": source.config,
                "license": source.license,
                "source_url": source.source_url,
                "content_sha256": digest,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            bytes_written += len(encoded)
            if written >= max_documents:
                break
    return {
        "source": asdict(source),
        "output": str(output_path),
        "documents": written,
        "bytes": bytes_written,
        "accepted_terms": sorted(accept_terms),
        "manual_license_review": manual_license_review,
        "collected_at": datetime.now(UTC).isoformat(),
        "next_gate": (
            "curate, deduplicate against eval, and create a training manifest before training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta auditável de datasets PT-BR")
    parser.add_argument("--list", action="store_true", help="Lista fontes e termos exigidos")
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--output", default="keilinks_data/raw/pt_sample.jsonl")
    parser.add_argument("--max-documents", type=int, default=2_000)
    parser.add_argument("--max-mib", type=int, default=128)
    parser.add_argument("--accept-terms", action="append", default=[])
    parser.add_argument("--manual-license-review", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                {key: asdict(value) for key, value in SOURCES.items()}, ensure_ascii=False, indent=2
            )
        )
        return
    if not args.source:
        raise SystemExit("informe --source ou --list")
    if args.max_documents < 1 or args.max_mib < 1:
        raise SystemExit("--max-documents e --max-mib devem ser positivos")
    result = collect(
        source=SOURCES[args.source],
        output_path=Path(args.output),
        max_documents=args.max_documents,
        max_bytes=args.max_mib * 1024 * 1024,
        accept_terms=set(args.accept_terms),
        manual_license_review=args.manual_license_review,
    )
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
