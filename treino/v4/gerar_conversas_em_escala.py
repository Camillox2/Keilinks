"""Geração escalável de conversas SFT para a Keilinks V4.

Suporta milhões de exemplos sem manter o corpus inteiro em RAM:
- shards JSONL rotativos;
- deduplicação exata e aproximada em SQLite;
- retomada após interrupção;
- professor + crítico via Ollama;
- metas em estágios para auditoria antes de aumentar o volume.

O script não transforma 10 milhões de exemplos sintéticos em uma recomendação.
Use as metas de estágio e compare checkpoints antes de continuar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List

from treino.v4.dataset import hamming_distance, normalized_key, simhash64
from treino.v4.gerar_conversas_ollama import (
    GENERATOR_PROMPT,
    SYSTEM_IDENTITY,
    TAXONOMY,
    clean_messages,
    critic_reviews,
    deterministic_filters,
    ollama_chat,
    parse_json_object,
    review_passes,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "dados" / "v4" / "sft" / "generated_1b"
DEFAULT_STAGES = (100_000, 1_000_000, 10_000_000)


class SQLiteDeduper:
    """Índice persistente para não carregar hashes de milhões de exemplos na RAM."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS samples("
            "digest TEXT PRIMARY KEY, simhash TEXT NOT NULL, bucket INTEGER NOT NULL, "
            "category TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_bucket ON samples(bucket)"
        )
        self.connection.commit()

    def add(self, messages: Iterable[dict], category: str) -> bool:
        canonical = "\n".join(
            f"{message.get('role')}:{normalized_key(message.get('content', ''))}"
            for message in messages
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        value = simhash64(canonical)
        bucket = value >> 48
        if self.connection.execute(
            "SELECT 1 FROM samples WHERE digest=?", (digest,)
        ).fetchone():
            return False
        nearby = self.connection.execute(
            "SELECT simhash FROM samples WHERE bucket=? ORDER BY rowid DESC LIMIT 1000",
            (bucket,),
        ).fetchall()
        if any(hamming_distance(value, int(row[0], 16)) <= 3 for row in nearby):
            return False
        self.connection.execute(
            "INSERT INTO samples(digest,simhash,bucket,category,created_at) VALUES(?,?,?,?,?)",
            (digest, f"{value:016x}", bucket, category, time.time()),
        )
        return True

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def category_counts(self) -> Counter:
        return Counter(dict(self.connection.execute(
            "SELECT category, COUNT(*) FROM samples GROUP BY category"
        ).fetchall()))

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class ShardWriter:
    def __init__(self, root: Path, shard_size: int, existing_count: int) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.total = existing_count
        self.shard_index = existing_count // shard_size
        self.in_shard = existing_count % shard_size
        self.handle = None

    def _ensure_open(self) -> None:
        if self.handle is None:
            path = self.root / f"conversations-{self.shard_index:05d}.jsonl"
            self.handle = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._ensure_open()
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.total += 1
        self.in_shard += 1
        if self.in_shard >= self.shard_size:
            self.handle.flush()
            self.handle.close()
            self.handle = None
            self.shard_index += 1
            self.in_shard = 0

    def close(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None


def choose_category(rng: random.Random, counts: Counter) -> str:
    categories = list(TAXONOMY)
    weights = [1.0 / (1.0 + counts.get(category, 0)) ** 0.5 for category in categories]
    return rng.choices(categories, weights=weights, k=1)[0]


def next_stage(current: int, stages: tuple[int, ...]) -> int | None:
    return next((stage for stage in stages if current < stage), None)


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_record(item: dict, category: str, scenario: str, args: argparse.Namespace,
                 review: dict) -> dict:
    canonical = "\n".join(message["content"] for message in item["messages"])
    record_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return {
        "id": f"synthetic-scale-v4:{record_id}",
        "group_id": f"synthetic-scale-v4:{record_id}",
        "source": "synthetic_ollama_scale_v4",
        "teacher_model": args.model,
        "critic_model": args.critic_model or args.model,
        "category": category,
        "scenario": scenario,
        "quality_review": review,
        "license": "generated-locally-review-model-terms",
        "messages": [
            {"role": "system", "content": SYSTEM_IDENTITY},
            *item["messages"],
        ],
    }


def generate_batch(args: argparse.Namespace, rng: random.Random,
                   category: str, scenario: str) -> tuple[List[dict], List[dict]]:
    prompt = GENERATOR_PROMPT.format(
        identity=SYSTEM_IDENTITY,
        category=category,
        scenario=scenario,
        count=args.batch_size,
    )
    raw = ollama_chat(
        args.model,
        "Gere somente JSON válido com dados SFT diversos e de alta qualidade.",
        prompt,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    parsed = parse_json_object(raw)
    candidates, rejected = [], []
    for item in parsed.get("items", []):
        if not isinstance(item, dict):
            continue
        messages = clean_messages(item.get("messages"))
        reason = deterministic_filters(messages) if messages else "invalid_messages"
        if reason:
            rejected.append({"reason": reason, "item": item})
            continue
        candidates.append({"category": category, "messages": messages})
    reviews = critic_reviews(args.critic_model or args.model, candidates) if candidates else {}
    accepted = []
    for index, item in enumerate(candidates):
        review = reviews.get(index, {})
        if review_passes(review, args.min_score):
            accepted.append(build_record(item, category, scenario, args, review))
        else:
            rejected.append({"reason": "critic", "review": review, "item": item})
    rng.shuffle(accepted)
    return accepted, rejected


def run(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    stages = tuple(sorted(set(args.stages)))
    if args.target > stages[-1]:
        stages = (*stages, args.target)

    deduper = SQLiteDeduper(root / "dedup.sqlite3")
    current = deduper.count()
    writer = ShardWriter(root / "shards", args.shard_size, current)
    rejected_writer = ShardWriter(root / "rejected", args.shard_size, 0)
    counts = deduper.category_counts()
    rng = random.Random(args.seed + current)
    started = time.time()
    errors = 0

    try:
        while current < args.target:
            stage = next_stage(current, stages)
            if stage is None:
                break
            if stage > args.approved_stage:
                print(
                    f"Parada segura em {current:,}. Próxima meta: {stage:,}. "
                    f"Rode auditoria e continue com --approved-stage {stage}."
                )
                break

            category = choose_category(rng, counts)
            scenario = rng.choice(TAXONOMY[category])
            try:
                accepted, rejected = generate_batch(args, rng, category, scenario)
                errors = 0
            except Exception as exc:
                errors += 1
                print(f"[geração] erro {errors}: {exc}")
                if errors >= args.max_consecutive_errors:
                    raise RuntimeError("Muitos erros consecutivos no professor/crítico") from exc
                time.sleep(min(60, 5 * errors))
                continue

            for rejected_record in rejected:
                rejected_writer.write({
                    "timestamp": time.time(),
                    "category": category,
                    "scenario": scenario,
                    **rejected_record,
                })

            added = 0
            for record in accepted:
                if current >= args.target:
                    break
                if not deduper.add(record["messages"], record["category"]):
                    continue
                writer.write(record)
                current += 1
                added += 1
                counts[record["category"]] += 1

            deduper.commit()
            elapsed = max(time.time() - started, 1.0)
            rate = current / elapsed if current else 0.0
            manifest = {
                "format": "keilinks-sft-scale-v4",
                "target": args.target,
                "approved_stage": args.approved_stage,
                "accepted": current,
                "shard_size": args.shard_size,
                "shards": writer.shard_index + int(writer.in_shard > 0),
                "teacher_model": args.model,
                "critic_model": args.critic_model or args.model,
                "min_score": args.min_score,
                "categories": dict(counts),
                "elapsed_seconds": elapsed,
                "accepted_per_second": rate,
                "updated_at": time.time(),
            }
            write_manifest(root / "manifest.json", manifest)
            print(
                f"{current:,}/{args.target:,} | +{added} | {category}/{scenario} | "
                f"{rate:.3f} conversas/s"
            )
    finally:
        writer.close()
        rejected_writer.close()
        deduper.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geração SFT escalável e retomável para a Keilinks V4"
    )
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--critic-model")
    parser.add_argument("--target", type=int, default=10_000_000)
    parser.add_argument("--approved-stage", type=int, default=100_000)
    parser.add_argument("--stages", type=int, nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-score", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-consecutive-errors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    if args.target <= 0 or args.shard_size <= 0:
        parser.error("target e shard-size devem ser positivos")
    if args.approved_stage <= 0:
        parser.error("approved-stage deve ser positivo")
    return args


if __name__ == "__main__":
    run(parse_args())
