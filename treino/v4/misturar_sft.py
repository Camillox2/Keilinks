"""Cria um mix SFT balanceado e reproduzível.

Impede que dados sintéticos ou traduções automáticas dominem o ajuste.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from treino.v4.dataset import canonical_conversation, extract_messages, validate_messages

ROOT = Path(__file__).resolve().parents[2]


def load_records(path: Path, source_name: str) -> List[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            messages = extract_messages(record)
            if not validate_messages(messages):
                continue
            record = dict(record)
            record["messages"] = messages
            record.setdefault("source", source_name)
            records.append(record)
    return records


def deduplicate(records: Iterable[dict]) -> List[dict]:
    seen = set()
    result = []
    for record in records:
        canonical = canonical_conversation(record["messages"])
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append(record)
    return result


def take_random(records: List[dict], maximum: int, rng: random.Random) -> List[dict]:
    if maximum < 0 or len(records) <= maximum:
        return list(records)
    return rng.sample(records, maximum)


def build_mix(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    sources: Dict[str, List[dict]] = {
        "curated": load_records(Path(args.curated), "keilinks_curated_v4"),
        "oasst2": load_records(Path(args.oasst2), "oasst2_portuguese"),
        "alpaca": load_records(Path(args.alpaca), "alpaca_ptbr"),
        "dolly": load_records(Path(args.dolly), "dolly_ptbr"),
        "synthetic": load_records(Path(args.synthetic), "synthetic_ollama_v4"),
    }
    max_total = max(1, args.max_examples)
    synthetic_cap = int(max_total * args.synthetic_ratio)
    translation_cap_each = int(max_total * args.translation_ratio_each)

    selected = []
    selected.extend(sources["curated"])
    selected.extend(sources["oasst2"])
    selected.extend(take_random(sources["alpaca"], translation_cap_each, rng))
    selected.extend(take_random(sources["dolly"], translation_cap_each, rng))

    selected = deduplicate(selected)
    human_budget = max(0, max_total - synthetic_cap)
    if len(selected) > human_budget:
        curated_ids = {record.get("id") for record in sources["curated"]}
        anchors = [record for record in selected if record.get("id") in curated_ids]
        remainder = [record for record in selected if record.get("id") not in curated_ids]
        selected = anchors + take_random(remainder, max(0, human_budget - len(anchors)), rng)

    remaining = max(0, max_total - len(selected))
    synthetic_limit = min(synthetic_cap, remaining)
    selected.extend(take_random(sources["synthetic"], synthetic_limit, rng))
    selected = deduplicate(selected)
    rng.shuffle(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    categories = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for record in selected:
            counts[str(record.get("source", "unknown"))] += 1
            categories[str(record.get("category", "general"))] += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "output": str(output),
        "seed": args.seed,
        "max_examples": max_total,
        "synthetic_ratio_cap": args.synthetic_ratio,
        "translation_ratio_each_cap": args.translation_ratio_each,
        "available": {name: len(records) for name, records in sources.items()},
        "selected_total": len(selected),
        "selected_by_source": dict(counts),
        "selected_by_category": dict(categories),
    }
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix balanceado de SFT")
    parser.add_argument("--curated", default="dados/v4/seed_conversas.jsonl")
    parser.add_argument("--oasst2", default="dados/v4/sft/oasst2_pt.jsonl")
    parser.add_argument("--alpaca", default="dados/v4/sft/alpaca_ptbr.jsonl")
    parser.add_argument("--dolly", default="dados/v4/sft/dolly_ptbr.jsonl")
    parser.add_argument("--synthetic", default="dados/v4/sft/synthetic_ollama_v4.jsonl")
    parser.add_argument("--output", default="dados/v4/sft/all_sft.jsonl")
    parser.add_argument("--max-examples", type=int, default=200_000)
    parser.add_argument("--synthetic-ratio", type=float, default=0.25)
    parser.add_argument("--translation-ratio-each", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.synthetic_ratio <= 0.5:
        parser.error("--synthetic-ratio deve ficar entre 0 e 0.5")
    if not 0 <= args.translation_ratio_each <= 0.4:
        parser.error("--translation-ratio-each deve ficar entre 0 e 0.4")
    return args


if __name__ == "__main__":
    build_mix(parse_args())
