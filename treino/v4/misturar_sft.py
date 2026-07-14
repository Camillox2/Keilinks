"""Cria um mix SFT balanceado e reproduzível.

Impede que dados sintéticos ou traduções automáticas dominem o ajuste, mesmo
quando há poucos exemplos humanos disponíveis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

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
            if not record.get("source"):
                record["source"] = source_name
            records.append(record)
    return records


def load_many(paths: Sequence[Path], source_name: str) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        records.extend(load_records(path, source_name))
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
    if maximum <= 0:
        return []
    if len(records) <= maximum:
        return list(records)
    return rng.sample(records, maximum)


def cap_group_ratio(records: List[dict], predicate: Callable[[dict], bool],
                    ratio: float, rng: random.Random) -> List[dict]:
    """Garante count_grupo / count_total <= ratio sem depender de max_examples."""
    if ratio <= 0:
        return [record for record in records if not predicate(record)]
    if ratio >= 1:
        return records
    group = [record for record in records if predicate(record)]
    others = [record for record in records if not predicate(record)]
    if not group:
        return records
    if not others:
        return []
    maximum = math.floor((ratio / (1.0 - ratio)) * len(others))
    return others + take_random(group, maximum, rng)


def enforce_all_caps(records: List[dict], args: argparse.Namespace,
                     rng: random.Random) -> List[dict]:
    """Repete os cortes porque reduzir um grupo pode elevar a razão de outro."""
    predicates = [
        (
            lambda record: str(record.get("source", "")) == "alpaca_ptbr",
            args.translation_ratio_each,
        ),
        (
            lambda record: str(record.get("source", "")) == "dolly_ptbr",
            args.translation_ratio_each,
        ),
        (
            lambda record: str(record.get("source", "")) == "synthetic_ollama_v4",
            args.synthetic_ratio,
        ),
    ]
    result = list(records)
    for _ in range(12):
        before = len(result)
        for predicate, ratio in predicates:
            result = cap_group_ratio(result, predicate, ratio, rng)
        if len(result) == before:
            break
    return result


def build_mix(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    curated_paths = [
        Path(value.strip()) for value in args.curated.split(",") if value.strip()
    ]
    sources: Dict[str, List[dict]] = {
        "curated": load_many(curated_paths, "keilinks_curated_v4"),
        "oasst2": load_records(Path(args.oasst2), "oasst2_portuguese"),
        "alpaca": load_records(Path(args.alpaca), "alpaca_ptbr"),
        "dolly": load_records(Path(args.dolly), "dolly_ptbr"),
        "synthetic": load_records(Path(args.synthetic), "synthetic_ollama_v4"),
    }
    sources = {name: deduplicate(records) for name, records in sources.items()}
    max_total = max(1, args.max_examples)

    selected: List[dict] = []
    selected.extend(sources["curated"])
    selected.extend(sources["oasst2"])
    selected.extend(take_random(
        sources["alpaca"], int(max_total * args.translation_ratio_each), rng
    ))
    selected.extend(take_random(
        sources["dolly"], int(max_total * args.translation_ratio_each), rng
    ))
    selected.extend(take_random(
        sources["synthetic"], int(max_total * args.synthetic_ratio), rng
    ))
    selected = deduplicate(selected)

    if len(selected) > max_total:
        curated = [
            record for record in selected
            if str(record.get("source", "")).startswith("keilinks_curated")
        ]
        others = [record for record in selected if record not in curated]
        selected = curated[:max_total]
        selected.extend(take_random(others, max_total - len(selected), rng))

    selected = enforce_all_caps(deduplicate(selected), args, rng)
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

    total = max(len(selected), 1)
    ratios = {
        source: round(count / total, 6) for source, count in counts.items()
    }
    report = {
        "output": str(output),
        "seed": args.seed,
        "max_examples": max_total,
        "synthetic_ratio_cap": args.synthetic_ratio,
        "translation_ratio_each_cap": args.translation_ratio_each,
        "curated_paths": [str(path) for path in curated_paths],
        "available": {name: len(records) for name, records in sources.items()},
        "selected_total": len(selected),
        "selected_by_source": dict(counts),
        "selected_ratio_by_source": ratios,
        "selected_by_category": dict(categories),
    }
    if ratios.get("synthetic_ollama_v4", 0.0) > args.synthetic_ratio + 1e-9:
        raise RuntimeError("Mix sintético excedeu o limite configurado")
    for source in ("alpaca_ptbr", "dolly_ptbr"):
        if ratios.get(source, 0.0) > args.translation_ratio_each + 1e-9:
            raise RuntimeError(f"{source} excedeu o limite configurado")

    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix balanceado de SFT")
    parser.add_argument(
        "--curated",
        default=(
            "dados/v4/conversas_curadas_v4.jsonl,"
            "dados/v4/seed_conversas.jsonl"
        ),
        help="Lista de JSONL curados separada por vírgula.",
    )
    parser.add_argument("--oasst2", default="dados/v4/sft/oasst2_pt.jsonl")
    parser.add_argument("--alpaca", default="dados/v4/sft/alpaca_ptbr.jsonl")
    parser.add_argument("--dolly", default="dados/v4/sft/dolly_ptbr.jsonl")
    parser.add_argument(
        "--synthetic", default="dados/v4/sft/synthetic_ollama_v4.jsonl"
    )
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
