"""Tokenizador BPE V4 com papéis de conversa explícitos.

Mantém compatibilidade com a implementação autoral da Keilinks, mas adiciona
``<sistema>`` como token especial e ferramentas reproduzíveis para construir o
vocabulário a partir do corpus local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from dados.tokenizador import Tokenizador

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB = ROOT / "dados" / "vocab_v4.json"
REQUIRED_SPECIALS = [
    "<pad>", "<unk>", "<inicio>", "<fim>",
    "<sistema>", "<vitor>", "<user>", "<keilinks>",
]


class TokenizadorV4(Tokenizador):
    def __init__(self, caminho: str | Path | None = None) -> None:
        super().__init__(None)
        self.SYSTEM = "<sistema>"
        self.ESPECIAIS = list(REQUIRED_SPECIALS)
        if caminho is not None:
            caminho = str(caminho)
            self.carregar(caminho)
            self._validate_specials(caminho)

    def _validate_specials(self, source: str) -> None:
        missing = [token for token in REQUIRED_SPECIALS if token not in self.vocab]
        if missing:
            raise ValueError(
                f"Vocabulário {source} não é V4; faltam tokens especiais: {missing}. "
                "Reconstrua com: python -m treino.v4.tokenizador build"
            )

    def _pre_tokenizar(self, texto: str):
        """Separa texto preservando todos os tokens especiais V4."""
        escaped = sorted((re.escape(token) for token in self.ESPECIAIS), key=len, reverse=True)
        partes = re.split(f"({'|'.join(escaped)})", texto)
        palavras = []
        especiais = set(self.ESPECIAIS)
        for parte in partes:
            if not parte:
                continue
            if parte in especiais:
                palavras.append(parte)
            else:
                palavras.extend(re.findall(r"\S+|\s+", parte))
        return palavras


def sample_text_file(path: Path, budget_bytes: int, rng: random.Random,
                     chunk_bytes: int = 2 * 1024 * 1024) -> List[str]:
    if not path.exists() or budget_bytes <= 0:
        return []
    size = path.stat().st_size
    if size <= budget_bytes:
        return [path.read_text(encoding="utf-8", errors="replace")]
    chunks = max(1, budget_bytes // chunk_bytes)
    samples = []
    with path.open("rb") as handle:
        for index in range(chunks):
            fraction = (index + rng.random()) / chunks
            offset = min(int(fraction * size), max(size - chunk_bytes, 0))
            handle.seek(offset)
            if offset:
                handle.readline()
            raw = handle.read(chunk_bytes)
            if raw:
                samples.append(raw.decode("utf-8", errors="replace"))
    return samples


def sample_sft_jsonl(path: Path, budget_bytes: int) -> List[str]:
    if not path.exists() or budget_bytes <= 0:
        return []
    texts = []
    used = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for message in record.get("messages", []):
                content = str(message.get("content", "")).strip()
                if content:
                    texts.append(content)
                    used += len(content.encode("utf-8"))
            if used >= budget_bytes:
                break
    return texts


def build_vocab(output: Path, pretrain_path: Path, sft_path: Path,
                vocab_size: int, sample_mb: int, seed: int) -> None:
    if vocab_size < len(REQUIRED_SPECIALS) + 256:
        raise ValueError("vocab_size pequeno demais")
    rng = random.Random(seed)
    budget = sample_mb * 1024 * 1024
    pretrain_budget = int(budget * 0.85)
    sft_budget = budget - pretrain_budget
    texts = sample_text_file(pretrain_path, pretrain_budget, rng)
    texts.extend(sample_sft_jsonl(sft_path, sft_budget))
    if not texts or sum(len(text) for text in texts) < 100_000:
        raise ValueError(
            "Amostra insuficiente. Baixe/prepare o corpus e gere as conversas antes do tokenizador."
        )
    tokenizer = TokenizadorV4()
    tokenizer.construir_vocab(
        textos,
        vocab_alvo=vocab_size,
        max_texto_mb=max(sample_mb, 16),
    )
    tokenizer._validate_specials("vocabulário recém-construído")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.salvar(str(output))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "path": str(output),
        "sha256": digest,
        "vocab_size": tokenizer.tam_vocab,
        "merges": len(tokenizer.merges),
        "sample_mb": sample_mb,
        "seed": seed,
        "special_tokens": REQUIRED_SPECIALS,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def inspect_vocab(path: Path) -> None:
    tokenizer = TokenizadorV4(path)
    probes = [
        "Olá, coração! Como você está?",
        "<sistema>Você é Keilinks.<fim><vitor>oi<fim><keilinks>Olá!<fim>",
        "Python: def soma(a: float, b: float) -> float:",
        "URL https://example.com e emoji 🤖✨",
    ]
    report = {
        "path": str(path),
        "vocab_size": tokenizer.tam_vocab,
        "merges": len(tokenizer.merges),
        "special_ids": {token: tokenizer.vocab[token] for token in REQUIRED_SPECIALS},
        "probes": [],
    }
    unk_id = tokenizer.vocab[tokenizer.UNK]
    for text in probes:
        ids = tokenizer.encode(text)
        report["probes"].append({
            "text": text,
            "tokens": len(ids),
            "unk": sum(token == unk_id for token in ids),
            "roundtrip": tokenizer.decode(ids),
        })
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenizador BPE da Keilinks V4")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", default=str(DEFAULT_VOCAB))
    build.add_argument("--pretrain", default="dados/v4/pretrain/pretrain_pt.txt")
    build.add_argument("--sft", default="dados/v4/sft/all_sft.jsonl")
    build.add_argument("--vocab-size", type=int, default=32_000)
    build.add_argument("--sample-mb", type=int, default=256)
    build.add_argument("--seed", type=int, default=42)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_vocab(
            Path(args.output), Path(args.pretrain), Path(args.sft),
            args.vocab_size, args.sample_mb, args.seed,
        )
    else:
        inspect_vocab(Path(args.vocab))


if __name__ == "__main__":
    main()
