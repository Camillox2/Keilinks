"""Tokenizador V4 com papéis explícitos e backend BPE moderno.

O backend padrão usa ``tokenizers`` (ByteLevel BPE em Rust), que é muito mais
rápido e estável para construir/tokenizar um corpus de bilhões de tokens. O
formato BPE autoral antigo continua legível para não invalidar checkpoints
legados, mas não é o caminho de novos pré-treinos.
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
DEFAULT_VOCAB = ROOT / "dados" / "v4" / "pretrain" / "tokenizer.json"
REQUIRED_SPECIALS = [
    "<pad>", "<unk>", "<inicio>", "<fim>",
    "<sistema>", "<vitor>", "<user>", "<keilinks>",
]


class TokenizadorV4(Tokenizador):
    def __init__(self, caminho: str | Path | None = None) -> None:
        super().__init__(None)
        self.SYSTEM = "<sistema>"
        self.ESPECIAIS = list(REQUIRED_SPECIALS)
        self._hf_tokenizer = None
        self.backend = "legacy_bpe"
        if caminho is not None:
            path = Path(caminho)
            if self._is_tokenizers_json(path):
                self._load_hf(path)
            else:
                self.carregar(str(path))
                self._validate_specials(str(path))

    @staticmethod
    def _is_tokenizers_json(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and isinstance(payload.get("model"), dict)

    def _load_hf(self, path: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Instale tokenizers para abrir o vocabulário V4 moderno") from exc
        tokenizer = Tokenizer.from_file(str(path))
        self._hf_tokenizer = tokenizer
        self.backend = "hf_bytelevel_bpe"
        self.vocab = tokenizer.get_vocab()
        self.vocab_inverso = {int(value): key for key, value in self.vocab.items()}
        self.tam_vocab = tokenizer.get_vocab_size()
        self.merges = []
        self._validate_specials(str(path))

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

    def encode(self, texto: str) -> List[int]:
        if self._hf_tokenizer is not None:
            return list(self._hf_tokenizer.encode(texto, add_special_tokens=False).ids)
        return super().encode(texto)

    def decode(self, tokens: Sequence[int]) -> str:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(list(tokens), skip_special_tokens=True)
        return super().decode(tokens)

    def salvar(self, caminho: str) -> None:
        if self._hf_tokenizer is not None:
            self._hf_tokenizer.save(caminho)
            return
        super().salvar(caminho)


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


def _build_hf_bytelevel_vocab(output: Path, texts: Iterable[str], vocab_size: int) -> TokenizadorV4:
    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError("Instale tokenizers para construir o vocabulário V4 moderno") from exc
    if output.exists():
        raise FileExistsError(f"Vocabulário já existe: {output}")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(REQUIRED_SPECIALS)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=REQUIRED_SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return TokenizadorV4(output)


def build_vocab(output: Path, pretrain_path: Path, sft_path: Path,
                vocab_size: int, sample_mb: int, seed: int,
                backend: str = "hf_bytelevel") -> None:
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
    if backend == "hf_bytelevel":
        tokenizer = _build_hf_bytelevel_vocab(output, texts, vocab_size)
    elif backend == "legacy":
        if output.exists():
            raise FileExistsError(f"Vocabulário já existe: {output}")
        tokenizer = TokenizadorV4()
        tokenizer.construir_vocab(
            texts,
            vocab_alvo=vocab_size,
            max_texto_mb=max(sample_mb, 16),
        )
        tokenizer._validate_specials("vocabulário recém-construído")
        output.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.salvar(str(output))
    else:
        raise ValueError(f"Backend desconhecido: {backend}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "path": str(output),
        "sha256": digest,
        "vocab_size": tokenizer.tam_vocab,
        "merges": len(tokenizer.merges),
        "backend": tokenizer.backend,
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
        "backend": tokenizer.backend,
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
    build.add_argument("--backend", choices=("hf_bytelevel", "legacy"), default="hf_bytelevel")
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_vocab(
            Path(args.output), Path(args.pretrain), Path(args.sft),
            args.vocab_size, args.sample_mb, args.seed, args.backend,
        )
    else:
        inspect_vocab(Path(args.vocab))


if __name__ == "__main__":
    main()
