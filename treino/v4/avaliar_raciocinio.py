"""Avaliação congelada do protocolo de raciocínio curto da Keilinks."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

from cerebro.raciocinio import parse_reasoning_output


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def score_case(text: str, case: dict) -> dict:
    """Pontua uma resposta sem depender de uma cadeia de pensamento exposta."""

    final = parse_reasoning_output(text).final
    normalized = normalize(final)
    required_all = [normalize(item) for item in case.get("required_all", [])]
    required_any = [normalize(item) for item in case.get("required_any", [])]
    forbidden = [normalize(item) for item in case.get("forbidden_any", [])]
    missing = [item for item in required_all if item not in normalized]
    any_ok = not required_any or any(item in normalized for item in required_any)
    matched_forbidden = [item for item in forbidden if item in normalized]
    return {
        "id": case.get("id", "unknown"),
        "passed": not missing and any_ok and not matched_forbidden,
        "missing_required": missing,
        "matched_forbidden": matched_forbidden,
        "text": final,
    }


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not item.get("prompt"):
                raise ValueError(f"Caso inválido em {path}")
            cases.append(item)
    if not cases:
        raise ValueError(f"Nenhum caso em {path}")
    return cases


def evaluate(runtime, cases: Iterable[dict], max_new_tokens: int) -> list[dict]:
    results: list[dict] = []
    for case in cases:
        answer = runtime.answer(
            str(case["prompt"]),
            web_enabled=False,
            web_mode="never",
            reasoning_mode="always",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        result = score_case(answer.text, case)
        result["generated_tokens"] = answer.generated_tokens
        results.append(result)
        print(f"{result['id']}: {'PASS' if result['passed'] else 'FAIL'}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o currículo de raciocínio curto")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default="dados/v4/pretrain/tokenizer.json")
    parser.add_argument("--cases", default="dados/v4/eval/raciocinio_curto_v1.jsonl")
    parser.add_argument("--output", default="artifacts/raciocinio_eval_v1.json")
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from api.runtime_v4 import V4Runtime

    runtime = V4Runtime(args.checkpoint, args.vocab, device=args.device)
    results = evaluate(runtime, load_cases(Path(args.cases)), args.max_new_tokens)
    passed = sum(bool(result["passed"]) for result in results)
    payload = {
        "suite": "raciocinio_curto_v1",
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({key: payload[key] for key in ("passed", "total", "pass_rate")}, indent=2))


if __name__ == "__main__":
    main()
