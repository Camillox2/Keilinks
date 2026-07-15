"""Empacota milhões de conversas SFT sem carregar o dataset inteiro na RAM."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from treino.v4.dataset import (
    IGNORE_INDEX,
    canonical_conversation,
    encode_sft_record,
    extract_messages,
    split_name,
    validate_messages,
)
from treino.v4.tokenizador import TokenizadorV4


class ExactDeduper:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen(digest TEXT PRIMARY KEY)"
        )
        self.connection.commit()

    def add(self, canonical: str) -> bool:
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        try:
            self.connection.execute("INSERT INTO seen(digest) VALUES(?)", (digest,))
            return True
        except sqlite3.IntegrityError:
            return False

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for match in matches:
            path = Path(match)
            resolved = str(path.resolve())
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    if not paths:
        raise FileNotFoundError("Nenhum JSONL encontrado nos padrões informados")
    return paths


def iter_records(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield path, line_number, record


class SplitWriter:
    def __init__(self, path: Path, context: int, pad_id: int) -> None:
        self.path = path
        self.context = context
        self.pad_id = pad_id
        self.handle = path.open("wb")
        self.input_buffer: list[int] = []
        self.label_buffer: list[int] = []
        self.blocks = 0
        self.examples = 0
        self.target_tokens = 0

    def _write_block(self, pad: bool = False) -> None:
        if not self.input_buffer:
            return
        if pad:
            missing = self.context - len(self.input_buffer)
            self.input_buffer.extend([self.pad_id] * missing)
            self.label_buffer.extend([IGNORE_INDEX] * missing)
        if len(self.input_buffer) != self.context:
            raise RuntimeError("Bloco incompleto sem padding")
        np.asarray(self.input_buffer, dtype=np.int32).tofile(self.handle)
        np.asarray(self.label_buffer, dtype=np.int32).tofile(self.handle)
        self.blocks += 1
        self.input_buffer.clear()
        self.label_buffer.clear()

    def add(self, input_ids: list[int], labels: list[int]) -> None:
        if len(input_ids) != len(labels):
            raise ValueError("input_ids e labels com tamanhos diferentes")
        if len(input_ids) > self.context:
            input_ids = input_ids[-self.context:]
            labels = labels[-self.context:]
        if not any(label != IGNORE_INDEX for label in labels):
            return
        if len(self.input_buffer) + len(input_ids) > self.context:
            self._write_block(pad=True)
        self.input_buffer.extend(input_ids)
        self.label_buffer.extend(labels)
        self.examples += 1
        self.target_tokens += sum(label != IGNORE_INDEX for label in labels)
        if len(self.input_buffer) == self.context:
            self._write_block()

    def close(self) -> None:
        self._write_block(pad=True)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def interleaved_to_separate(path: Path, blocks: int, context: int,
                            input_target: Path, label_target: Path) -> None:
    """Converte [input block][label block] repetido em dois arquivos contíguos."""
    source = np.memmap(path, dtype=np.int32, mode="r", shape=(blocks, 2, context))
    input_mm = np.memmap(input_target, dtype=np.int32, mode="w+", shape=(blocks, context))
    label_mm = np.memmap(label_target, dtype=np.int32, mode="w+", shape=(blocks, context))
    chunk = 4096
    for start in range(0, blocks, chunk):
        end = min(start + chunk, blocks)
        input_mm[start:end] = source[start:end, 0, :]
        label_mm[start:end] = source[start:end, 1, :]
    input_mm.flush()
    label_mm.flush()
    del source, input_mm, label_mm


def pack(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = expand_inputs(args.inputs)
    tokenizer = TokenizadorV4(args.vocab)
    pad_id = int(tokenizer.vocab["<pad>"])
    train_temp = output / "train.interleaved.tmp"
    validation_temp = output / "validation.interleaved.tmp"
    dedup_path = output / "packing_dedup.sqlite3"
    for path in (train_temp, validation_temp, dedup_path):
        path.unlink(missing_ok=True)

    writers = {
        "train": SplitWriter(train_temp, args.context, pad_id),
        "validation": SplitWriter(validation_temp, args.context, pad_id),
    }
    deduper = ExactDeduper(dedup_path)
    stats = Counter()
    source_counts = Counter()

    try:
        for path, line_number, record in iter_records(paths):
            stats["read"] += 1
            messages = extract_messages(record)
            if not validate_messages(messages):
                stats["invalid"] += 1
                continue
            canonical = canonical_conversation(messages)
            if not deduper.add(canonical):
                stats["duplicates"] += 1
                continue
            example = encode_sft_record(tokenizer, record)
            if example is None:
                stats["invalid"] += 1
                continue
            split = split_name(record, messages, args.validation_ratio)
            writers[split].add(example.input_ids, example.labels)
            source_counts[str(record.get("source") or path.name)] += 1
            stats["accepted"] += 1
            if stats["accepted"] % args.commit_every == 0:
                deduper.commit()
                print(
                    f"{stats['accepted']:,} aceitas | {stats['duplicates']:,} duplicadas | "
                    f"train blocks {writers['train'].blocks:,}"
                )
    finally:
        for writer in writers.values():
            writer.close()
        deduper.close()

    for split, temporary in (("train", train_temp), ("validation", validation_temp)):
        writer = writers[split]
        if writer.blocks <= 0:
            raise ValueError(f"Split {split} ficou vazio")
        input_temp = output / f"{split}_input_ids.bin.tmp"
        label_temp = output / f"{split}_labels.bin.tmp"
        interleaved_to_separate(
            temporary,
            writer.blocks,
            args.context,
            input_temp,
            label_temp,
        )
        os.replace(input_temp, output / f"{split}_input_ids.bin")
        os.replace(label_temp, output / f"{split}_labels.bin")
        temporary.unlink(missing_ok=True)

    metadata = {
        "format": "keilinks-packed-sft-v4",
        "packer": "streaming-v1",
        "context_length": args.context,
        "dtype": "int32",
        "ignore_index": IGNORE_INDEX,
        "validation_ratio": args.validation_ratio,
        "vocab": str(Path(args.vocab)),
        "inputs": [str(path) for path in paths],
        "stats": {
            **dict(stats),
            "train_examples": writers["train"].examples,
            "validation_examples": writers["validation"].examples,
            "train_blocks": writers["train"].blocks,
            "validation_blocks": writers["validation"].blocks,
            "train_target_tokens": writers["train"].target_tokens,
            "validation_target_tokens": writers["validation"].target_tokens,
        },
        "selected_by_source": dict(source_counts),
    }
    temporary_metadata = output / "metadata.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_metadata, output / "metadata.json")
    print(json.dumps(metadata["stats"], ensure_ascii=False, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Packer SFT streaming para milhões de conversas")
    parser.add_argument("--vocab", default="dados/vocab_v4.json")
    parser.add_argument("--output", default="dados/v4/packed-scale")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--commit-every", type=int, default=10_000)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Arquivos ou globs JSONL; ex.: dados/v4/sft/generated_1b/shards/*.jsonl",
    )
    args = parser.parse_args()
    if args.context < 128:
        parser.error("context deve ser pelo menos 128")
    if not 0 < args.validation_ratio < 0.5:
        parser.error("validation-ratio deve ficar entre 0 e 0.5")
    return args


if __name__ == "__main__":
    pack(parse_args())
