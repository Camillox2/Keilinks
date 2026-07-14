"""Empacota o SFT usando o tokenizador V4 e assistant-only loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from treino.v4.dataset import build_packed_dataset
from treino.v4.tokenizador import TokenizadorV4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="dados/vocab_v4.json")
    parser.add_argument("--input", nargs="+", default=["dados/v4/sft/all_sft.jsonl"])
    parser.add_argument("--output", default="dados/v4/packed")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    args = parser.parse_args()

    tokenizer = TokenizadorV4(args.vocab)
    stats = build_packed_dataset(
        tokenizer,
        args.input,
        args.output,
        context_length=args.context,
        validation_ratio=args.validation_ratio,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
