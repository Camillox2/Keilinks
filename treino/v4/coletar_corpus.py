"""Coleta escalável e rastreável de texto PT-BR para o Keilinks 380M.

O objetivo é criar *shards* locais com procedência por documento, sem carregar
o corpus inteiro na RAM e sem transformar uma lista de datasets em um botão de
"baixar a internet". Cada execução registra os termos aceitos, a revisão do
dataset, filtros aplicados e hashes de conteúdo.

Somente fontes públicas com contrato de uso conhecido entram por padrão. Bases
gated, de licença composta ou sem suporte no ``datasets`` instalado ficam fora
da automação até que haja credencial/revisão específica.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from treino.v4.preparar_dados import clean_document

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dados" / "v4"


@dataclass(frozen=True)
class CorpusSource:
    key: str
    dataset_id: str
    config: str | None
    split: str
    license: str
    source_url: str
    acknowledgement: str
    purpose: str
    status: str = "public_supported"
    note: str = ""


SOURCES: dict[str, CorpusSource] = {
    "fineweb2_pt": CorpusSource(
        key="fineweb2_pt",
        dataset_id="HuggingFaceFW/fineweb-2",
        config="por_Latn",
        split="train",
        license="ODC-By-1.0 + Common Crawl Terms of Use",
        source_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        acknowledgement="fineweb2_terms",
        purpose="general_portuguese_pretraining",
        note="Filtra language_score e clusters MinHash grandes antes de gravar.",
    ),
    "wikipedia_pt": CorpusSource(
        key="wikipedia_pt",
        dataset_id="wikimedia/wikipedia",
        config="20231101.pt",
        split="train",
        license="CC-BY-SA-3.0 + GFDL",
        source_url="https://huggingface.co/datasets/wikimedia/wikipedia",
        acknowledgement="wikipedia_sharealike",
        purpose="encyclopedic_portuguese_pretraining",
        note="Título e artigo são preservados juntos; mantenha a atribuição no manifesto.",
    ),
    "carolina_pt": CorpusSource(
        key="carolina_pt",
        dataset_id="carolina-c4ai/corpus-carolina",
        config=None,
        split="train",
        license="CC-BY-4.0",
        source_url="https://huggingface.co/datasets/carolina-c4ai/corpus-carolina",
        acknowledgement="carolina_cc_by",
        purpose="brazilian_portuguese_pretraining",
        status="manual_import_required",
        note=(
            "A versão atual do pacote datasets local não suporta mais o script "
            "do dataset. Importe uma cópia revisada/manual com manifesto separado."
        ),
    ),
    "culturax_pt": CorpusSource(
        key="culturax_pt",
        dataset_id="uonlp/CulturaX",
        config="pt",
        split="train",
        license="termos gated e licenças upstream compostas",
        source_url="https://huggingface.co/datasets/uonlp/CulturaX",
        acknowledgement="culturax_terms",
        purpose="general_portuguese_pretraining",
        status="gated_manual_review_required",
        note="Exige login Hugging Face, aceite no site e revisão da licença de uso pretendido.",
    ),
    "gigaverbo_pt": CorpusSource(
        key="gigaverbo_pt",
        dataset_id="TucanoBR/GigaVerbo",
        config=None,
        split="train",
        license="licenças upstream mistas; dataset card declara license=other",
        source_url="https://huggingface.co/datasets/TucanoBR/GigaVerbo",
        acknowledgement="gigaverbo_license_review",
        purpose="general_portuguese_pretraining",
        status="manual_license_review_required",
        note=(
            "Apesar do filtro de qualidade, há fontes upstream com termos "
            "diferentes. Não é seguro tratá-lo como uma fonte de licença única."
        ),
    ),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SqliteDeduper:
    """Deduplicação exata no disco, adequada a milhões de documentos."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS seen (digest TEXT PRIMARY KEY)")
        self.pending = 0

    def add(self, digest: str) -> bool:
        try:
            self.connection.execute("INSERT INTO seen(digest) VALUES(?)", (digest,))
            self.pending += 1
            if self.pending >= 1_000:
                self.connection.commit()
                self.pending = 0
            return True
        except sqlite3.IntegrityError:
            return False

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class RawShardWriter:
    """Escreve JSONL em shards sem ultrapassar o orçamento da fonte."""

    def __init__(self, directory: Path, shard_bytes: int) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.shard_bytes = shard_bytes
        self.handle = None
        self.current_bytes = 0
        self.total_bytes = 0
        self.shards: list[dict[str, object]] = []
        self.records = 0

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.close()
        path = self.directory / f"shard-{len(self.shards):05d}.jsonl"
        self.handle = path.open("x", encoding="utf-8", newline="\n")
        self.shards.append({"path": str(path), "bytes": 0, "records": 0})
        self.current_bytes = 0

    def write(self, record: dict[str, object]) -> int:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        encoded = line.encode("utf-8")
        if self.handle is None or self.current_bytes + len(encoded) > self.shard_bytes:
            self._open_next()
        assert self.handle is not None
        self.handle.write(line)
        self.current_bytes += len(encoded)
        self.total_bytes += len(encoded)
        self.records += 1
        shard = self.shards[-1]
        shard["bytes"] = int(shard["bytes"]) + len(encoded)
        shard["records"] = int(shard["records"]) + 1
        return len(encoded)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def dataset_revision(dataset_id: str) -> str:
    """Resolve a revisão sem tornar uma falha de metadados fatal à coleta."""
    try:
        from huggingface_hub import HfApi

        return str(HfApi().dataset_info(dataset_id, timeout=20).sha or "")
    except Exception:
        return "unavailable"


def source_text(source: CorpusSource, item: dict[str, Any]) -> str:
    if source.key == "wikipedia_pt":
        title = str(item.get("title") or "").strip()
        text = str(item.get("text") or "").strip()
        return f"{title}\n\n{text}" if title else text
    return str(item.get("text") or "").strip()


def source_record_id(item: dict[str, Any]) -> str:
    for key in ("id", "url", "title"):
        value = str(item.get(key) or "").strip()
        if value:
            return value[:1_000]
    return ""


def accepted_by_source(
    source: CorpusSource,
    item: dict[str, Any],
    *,
    min_language_score: float,
    max_minhash_cluster_size: int,
) -> tuple[bool, str | None]:
    if source.key == "fineweb2_pt":
        try:
            if float(item.get("language_score", 0.0)) < min_language_score:
                return False, "low_language_score"
        except (TypeError, ValueError):
            return False, "invalid_language_score"
        try:
            if int(item.get("minhash_cluster_size", 1)) > max_minhash_cluster_size:
                return False, "large_minhash_cluster"
        except (TypeError, ValueError):
            return False, "invalid_minhash_cluster"
    return True, None


def stream_source(source: CorpusSource) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Instale datasets para coletar corpus") from exc
    kwargs: dict[str, object] = {"split": source.split, "streaming": True}
    if source.config:
        kwargs["name"] = source.config
    return load_dataset(source.dataset_id, **kwargs)


def collect_source(
    source: CorpusSource,
    *,
    run_dir: Path,
    budget_bytes: int,
    shard_bytes: int,
    max_documents: int,
    accepted_terms: set[str],
    min_language_score: float,
    max_minhash_cluster_size: int,
) -> dict[str, object]:
    if source.acknowledgement not in accepted_terms:
        raise ValueError(
            f"{source.key} exige --accept-terms {source.acknowledgement}; leia {source.source_url}"
        )
    if source.status != "public_supported":
        raise ValueError(f"{source.key} não entra na coleta automática: {source.note}")
    if budget_bytes < 1:
        raise ValueError("O orçamento de bytes deve ser positivo")

    output_dir = run_dir / source.key
    writer = RawShardWriter(output_dir, shard_bytes)
    deduper = SqliteDeduper(run_dir / "dedup.sqlite3")
    rejected: Counter[str] = Counter()
    seen = 0
    started = utc_now()
    revision = dataset_revision(source.dataset_id)
    try:
        for item in stream_source(source):
            seen += 1
            if writer.records >= max_documents or writer.total_bytes >= budget_bytes:
                break
            allowed, reason = accepted_by_source(
                source,
                item,
                min_language_score=min_language_score,
                max_minhash_cluster_size=max_minhash_cluster_size,
            )
            if not allowed:
                rejected[reason or "source_filter"] += 1
                continue
            text = source_text(source, item)
            if "\ufffd" in text:
                rejected["encoding_replacement"] += 1
                continue
            text = clean_document(text)
            if not text:
                rejected["quality_filter"] += 1
                continue
            digest = content_hash(text)
            record: dict[str, object] = {
                "text": text,
                "source_key": source.key,
                "dataset_id": source.dataset_id,
                "dataset_config": source.config,
                "dataset_revision": revision,
                "license": source.license,
                "source_url": source.source_url,
                "content_sha256": digest,
                "source_record_id": source_record_id(item),
            }
            if source.key == "fineweb2_pt":
                record["source_document_url"] = str(item.get("url") or "")[:2_048]
                record["language_score"] = float(item.get("language_score") or 0.0)
                record["minhash_cluster_size"] = int(item.get("minhash_cluster_size") or 0)
            elif source.key == "wikipedia_pt":
                record["title"] = str(item.get("title") or "")[:1_000]
                record["source_document_url"] = str(item.get("url") or "")[:2_048]
            estimated_line_bytes = len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 1
            if writer.total_bytes + estimated_line_bytes > budget_bytes:
                rejected["budget_reached"] += 1
                break
            if not deduper.add(digest):
                rejected["exact_duplicate"] += 1
                continue
            writer.write(record)
            if writer.records % 2_000 == 0:
                print(
                    f"{source.key}: {writer.records:,} documentos | "
                    f"{writer.total_bytes / 1024**3:.2f} GiB"
                )
    finally:
        writer.close()
        deduper.close()

    manifest = {
        "schema_version": 1,
        "kind": "keilinks_v4_raw_corpus",
        "status": "complete",
        "source": asdict(source),
        "dataset_revision": revision,
        "run_directory": str(run_dir),
        "shards": writer.shards,
        "started_at": started,
        "finished_at": utc_now(),
        "records_seen": seen,
        "records_written": writer.records,
        "bytes_written": writer.total_bytes,
        "budget_bytes": budget_bytes,
        "accepted_terms": sorted(accepted_terms),
        "filters": {
            "clean_document": "normalize, redact email/ip, minimum length/diversity",
            "min_language_score": min_language_score if source.key == "fineweb2_pt" else None,
            "max_minhash_cluster_size": max_minhash_cluster_size
            if source.key == "fineweb2_pt"
            else None,
            "exact_deduplication": "sqlite sha256 over cleaned text",
        },
        "rejections": dict(sorted(rejected.items())),
    }
    manifest_path = run_dir / f"{source.key}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_budget(values: list[str], sources: list[str]) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for value in values:
        try:
            key, amount = value.split("=", 1)
            gib = float(amount)
        except ValueError as exc:
            raise ValueError("Use --budget-gib fonte=numero, por exemplo fineweb2_pt=8") from exc
        if key not in SOURCES or gib <= 0:
            raise ValueError(f"Orçamento inválido: {value}")
        budgets[key] = int(gib * 1024**3)
    if not budgets:
        # Um piloto seguro por padrão. O comando de produção deve declarar os
        # orçamentos explicitamente, o que impede ocupar o disco sem perceber.
        return {key: 512 * 1024**2 for key in sources}
    missing = [key for key in sources if key not in budgets]
    if missing:
        raise ValueError(f"Falta --budget-gib para: {', '.join(missing)}")
    return budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coleta rastreável de corpus PT-BR para Keilinks V4"
    )
    parser.add_argument(
        "--list", action="store_true", help="Lista fontes, status e termos exigidos"
    )
    parser.add_argument("--source", action="append", choices=sorted(SOURCES))
    parser.add_argument("--run-id", help="Nome novo para dados/v4/raw/<run-id>")
    parser.add_argument(
        "--budget-gib",
        action="append",
        default=[],
        help="Orçamento por fonte: fineweb2_pt=8. Nunca substitui um shard já existente.",
    )
    parser.add_argument("--max-documents", type=int, default=100_000_000)
    parser.add_argument("--shard-mib", type=int, default=512)
    parser.add_argument("--accept-terms", action="append", default=[])
    parser.add_argument("--min-language-score", type=float, default=0.97)
    parser.add_argument(
        "--max-minhash-cluster-size",
        type=int,
        default=32,
        help="Evita clusters muito grandes sem descartar a maior parte do FineWeb-2.",
    )
    args = parser.parse_args()
    if not args.list and not args.source:
        parser.error("Informe ao menos um --source ou use --list")
    if args.max_documents < 1 or args.shard_mib < 1:
        parser.error("--max-documents e --shard-mib devem ser positivos")
    if not 0.0 <= args.min_language_score <= 1.0:
        parser.error("--min-language-score deve ficar entre 0 e 1")
    if args.max_minhash_cluster_size < 1:
        parser.error("--max-minhash-cluster-size deve ser positivo")
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
    run_id = args.run_id or f"public-pt-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    run_dir = DATA_ROOT / "raw" / run_id
    if run_dir.exists():
        raise SystemExit(f"A execução já existe: {run_dir}; use outro --run-id")
    budgets = parse_budget(args.budget_gib, args.source)
    run_dir.mkdir(parents=True, exist_ok=False)
    accepted_terms = set(args.accept_terms)
    all_manifests: list[dict[str, object]] = []
    try:
        for source_key in args.source:
            all_manifests.append(
                collect_source(
                    SOURCES[source_key],
                    run_dir=run_dir,
                    budget_bytes=budgets[source_key],
                    shard_bytes=args.shard_mib * 1024**2,
                    max_documents=args.max_documents,
                    accepted_terms=accepted_terms,
                    min_language_score=args.min_language_score,
                    max_minhash_cluster_size=args.max_minhash_cluster_size,
                )
            )
    except Exception as exc:
        (run_dir / "run.manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "run_id": run_id,
                    "error": str(exc),
                    "completed_sources": [item["source"]["key"] for item in all_manifests],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    run_manifest = {
        "schema_version": 1,
        "kind": "keilinks_v4_public_corpus_run",
        "status": "complete",
        "run_id": run_id,
        "created_at": utc_now(),
        "sources": [item["source"]["key"] for item in all_manifests],
        "documents": sum(int(item["records_written"]) for item in all_manifests),
        "bytes": sum(int(item["bytes_written"]) for item in all_manifests),
        "source_manifests": [f"{item['source']['key']}.manifest.json" for item in all_manifests],
    }
    (run_dir / "run.manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
