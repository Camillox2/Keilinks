"""Monta texto de pré-treino a partir de shards V4 rastreáveis.

O arquivo final mantém limites de documento (linha em branco) para que
``pretreinar.py`` possa fazer o split de validação por documento. Ele verifica
o hash que veio da coleta e intercala fontes, em vez de concatenar todo o web
corpus antes do material enciclopédico.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from treino.v4.coletar_corpus import SqliteDeduper, content_hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_paths(inputs: list[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix == ".jsonl":
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("shard-*.jsonl"))
        else:
            raise FileNotFoundError(path)
        for candidate in candidates:
            source = candidate.parent.name
            grouped.setdefault(source, []).append(candidate)
    if not grouped:
        raise FileNotFoundError("Nenhum shard-*.jsonl encontrado")
    return grouped


def records(paths: list[Path]) -> Iterator[dict[str, object]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {path}:{line_number}") from exc
                if isinstance(record, dict):
                    yield record


def validate_record(record: dict[str, object]) -> tuple[str | None, str | None]:
    text = record.get("text")
    digest = record.get("content_sha256")
    if not isinstance(text, str) or not isinstance(digest, str):
        return None, "missing_text_or_hash"
    text = text.strip()
    if len(text) < 160:
        return None, "too_short"
    if content_hash(text) != digest:
        return None, "hash_mismatch"
    if not str(record.get("source_key") or "").strip():
        return None, "missing_source"
    return text, None


def assemble(
    inputs: list[str],
    output: Path,
    *,
    max_bytes: int | None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Saída já existe: {output}")
    groups = shard_paths(inputs)
    iterators = {source: iter(records(paths)) for source, paths in groups.items()}
    active = list(sorted(iterators))
    temporary = output.with_suffix(output.suffix + ".tmp")
    dedup_path = output.with_suffix(output.suffix + ".dedup.sqlite3")
    if temporary.exists():
        raise FileExistsError(f"Arquivo temporário já existe: {temporary}")
    if dedup_path.exists():
        raise FileExistsError(f"Banco de deduplicação já existe: {dedup_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    deduper = SqliteDeduper(dedup_path)
    seen = accepted = written_bytes = 0
    rejections: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            while active:
                next_active: list[str] = []
                for source in active:
                    try:
                        record = next(iterators[source])
                    except StopIteration:
                        continue
                    next_active.append(source)
                    seen += 1
                    text, reason = validate_record(record)
                    if reason:
                        rejections[reason] += 1
                        continue
                    assert text is not None
                    digest = str(record["content_sha256"])
                    encoded = (text + "\n\n").encode("utf-8")
                    if max_bytes is not None and written_bytes + len(encoded) > max_bytes:
                        rejections["output_budget_reached"] += 1
                        active = []
                        break
                    if not deduper.add(digest):
                        rejections["exact_duplicate"] += 1
                        continue
                    handle.write(text)
                    handle.write("\n\n")
                    written_bytes += len(encoded)
                    accepted += 1
                    source_key = str(record.get("source_key") or source)
                    source_counts[source_key] += 1
                    source_bytes[source_key] += len(encoded)
                    if accepted % 10_000 == 0:
                        print(f"{accepted:,} documentos | {written_bytes / 1024**3:.2f} GiB")
                active = next_active if active else []
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        deduper.close()
    if accepted < 100:
        temporary.unlink(missing_ok=True)
        raise ValueError("Corpus final muito pequeno; espere a coleta acumular mais documentos")
    os.replace(temporary, output)
    digest = sha256_file(output)
    manifest = {
        "schema_version": 1,
        "kind": "keilinks_v4_pretrain_text",
        "output": str(output),
        "sha256": digest,
        "documents_seen": seen,
        "documents_written": accepted,
        "bytes_written": written_bytes,
        "sources": dict(sorted(source_counts.items())),
        "source_bytes": dict(sorted(source_bytes.items())),
        "rejections": dict(sorted(rejections.items())),
        "input_groups": {source: [str(path) for path in paths] for source, paths in groups.items()},
        "dedup_database": str(dedup_path),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monta o texto de pré-treino V4 a partir de shards"
    )
    parser.add_argument(
        "--input", action="append", required=True, help="Diretório de uma coleta ou shard JSONL"
    )
    parser.add_argument("--output", default="dados/v4/pretrain/pretrain_pt.txt")
    parser.add_argument(
        "--max-gib", type=float, help="Limita o arquivo final; omita para usar todos os shards"
    )
    args = parser.parse_args()
    if args.max_gib is not None and args.max_gib <= 0:
        parser.error("--max-gib deve ser positivo")
    return args


def main() -> None:
    args = parse_args()
    result = assemble(
        args.input,
        Path(args.output),
        max_bytes=int(args.max_gib * 1024**3) if args.max_gib else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
