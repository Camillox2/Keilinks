"""Avaliação determinística de regressão do runtime Keilinks V5."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keilinks_v5.rag import LocalKnowledgeStore
from keilinks_v5.runtime import ChatMessage, UnslothRuntime, leaked_control_markers
from keilinks_v5.settings import KeilinksSettings


def _contains_any(text: str, values: list[str]) -> bool:
    return any(value.lower() in text for value in values)


def _contains_all(text: str, values: list[str]) -> bool:
    return all(value.lower() in text for value in values)


def evaluate_case(
    runtime: UnslothRuntime, case: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    prompt = str(case["prompt"])
    answer = runtime.answer(
        [ChatMessage(role="user", content=prompt)],
        use_rag=False,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    text = answer.text.lower()
    failures: list[str] = []
    if case.get("required_any") and not _contains_any(text, list(case["required_any"])):
        failures.append("required_any")
    if case.get("required_all") and not _contains_all(text, list(case["required_all"])):
        failures.append("required_all")
    if case.get("forbidden_any") and _contains_any(text, list(case["forbidden_any"])):
        failures.append("forbidden_any")
    if leaked_control_markers(answer.text):
        failures.append("leaked_control_marker")
    min_words = int(case.get("min_words", 0))
    if min_words and len(re.findall(r"\w+", answer.text, flags=re.UNICODE)) < min_words:
        failures.append("min_words")
    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "passed": not failures,
        "failures": failures,
        "response": answer.text,
        "elapsed_ms": round(answer.elapsed_ms, 1),
        "model_id": answer.model_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia regressões do Keilinks V5")
    parser.add_argument("--eval", default="dados/v4/eval/keilinks_eval_v4.jsonl")
    parser.add_argument("--output", default="keilinks_data/evaluations/latest.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--include-web-cases",
        action="store_true",
        help="Inclui casos que exigem busca web; por padrão são marcados como skipped.",
    )
    args = parser.parse_args()
    settings = KeilinksSettings.from_env()
    settings.validate()
    runtime = UnslothRuntime(settings, LocalKnowledgeStore(settings.rag_db))
    cases: list[dict[str, Any]] = []
    with Path(args.eval).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.get("web_required") and not args.include_web_cases:
            results.append(
                {
                    "id": case["id"],
                    "category": case.get("category", "web"),
                    "skipped": True,
                    "reason": "web_required",
                }
            )
            continue
        results.append(evaluate_case(runtime, case, args.max_new_tokens))
    evaluated = [result for result in results if not result.get("skipped")]
    passed = sum(bool(result["passed"]) for result in evaluated)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": runtime.model_id,
        "evaluated": len(evaluated),
        "passed": passed,
        "pass_rate": round(passed / len(evaluated), 4) if evaluated else 0.0,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("model_id", "evaluated", "passed", "pass_rate")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
