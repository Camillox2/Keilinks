"""Avaliação congelada de checkpoints Keilinks V4."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from api.runtime_v4 import V4Runtime


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    normalized = normalize(text)
    return any(normalize(phrase) in normalized for phrase in phrases)


def evaluate_response(case: dict, response: str) -> tuple[bool, list[str]]:
    failures = []
    required_any = case.get("required_any", [])
    required_all = case.get("required_all", [])
    forbidden_any = case.get("forbidden_any", [])
    if required_any and not contains_any(response, required_any):
        failures.append("required_any")
    normalized = normalize(response)
    for phrase in required_all:
        if normalize(phrase) not in normalized:
            failures.append(f"required_all:{phrase}")
    for phrase in forbidden_any:
        if normalize(phrase) in normalized:
            failures.append(f"forbidden:{phrase}")
    words = re.findall(r"\w+", response, flags=re.UNICODE)
    if len(words) < int(case.get("min_words", 2)):
        failures.append("too_short")
    if len(words) > int(case.get("max_words", 300)):
        failures.append("too_long")
    if len(words) >= 20 and len(set(word.lower() for word in words)) / len(words) < 0.35:
        failures.append("repetition")
    return not failures, failures


def run(args: argparse.Namespace) -> dict:
    runtime = V4Runtime(args.checkpoint, args.vocab, device=args.device)
    cases = []
    for line in Path(args.eval).open("r", encoding="utf-8"):
        if line.strip():
            cases.append(json.loads(line))
    results = []
    category_totals = Counter()
    category_passes = Counter()
    for index, case in enumerate(cases, 1):
        if case.get("web_required") and not args.with_web:
            continue
        answer = runtime.answer(
            case["prompt"],
            web_enabled=bool(args.with_web and case.get("web_required")),
            max_new_tokens=int(case.get("max_new_tokens", 180)),
            temperature=args.temperature,
            top_p=0.9,
        )
        passed, failures = evaluate_response(case, answer.text)
        category = case.get("category", "general")
        category_totals[category] += 1
        category_passes[category] += int(passed)
        results.append({
            "id": case.get("id", index),
            "category": category,
            "prompt": case["prompt"],
            "response": answer.text,
            "passed": passed,
            "failures": failures,
            "sources": answer.sources,
        })
        print(f"[{index}/{len(cases)}] {'PASS' if passed else 'FAIL'} {case.get('id')}: {failures}")
    total = len(results)
    passes = sum(int(result["passed"]) for result in results)
    report = {
        "checkpoint": args.checkpoint,
        "vocab": args.vocab,
        "total": total,
        "passed": passes,
        "score": passes / max(total, 1),
        "by_category": {
            category: {
                "total": category_totals[category],
                "passed": category_passes[category],
                "score": category_passes[category] / max(category_totals[category], 1),
            }
            for category in sorted(category_totals)
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação Keilinks V4")
    parser.add_argument("--checkpoint", default="checkpoints/v4-sft/keilinks_v4.pt")
    parser.add_argument("--vocab", default="dados/v4/pretrain/tokenizer.json")
    parser.add_argument("--eval", default="dados/v4/eval/keilinks_eval_v4.jsonl")
    parser.add_argument("--output", default="checkpoints/v4-sft/eval_report.json")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device")
    parser.add_argument("--with-web", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.70)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = run(arguments)
    sys.exit(0 if report["score"] >= arguments.min_score else 1)
