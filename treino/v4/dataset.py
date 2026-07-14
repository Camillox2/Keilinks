"""Dataset V4: JSONL, deduplicação, split por grupo e packing.

No SFT, apenas tokens produzidos pelo papel assistant recebem loss. Sistema,
usuário e padding recebem -100.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class TokenizerLike(Protocol):
    def encode(self, text: str) -> List[int]: ...
    @property
    def vocab(self) -> Dict[str, int]: ...


@dataclass
class SFTExample:
    input_ids: List[int]
    labels: List[int]
    example_id: str
    source: str
    category: str


@dataclass
class BuildStats:
    read: int = 0
    accepted: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    rejected: int = 0
    train_examples: int = 0
    validation_examples: int = 0
    train_blocks: int = 0
    validation_blocks: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalized_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b\d+\b", "<n>", text)
    text = re.sub(r"[^a-z0-9áàâãéèêíìîóòôõúùûç<> ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def simhash64(text: str) -> int:
    words = re.findall(r"\w+", normalized_key(text), flags=re.UNICODE)
    if not words:
        return 0
    features = list(words)
    if len(words) >= 3:
        features += [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    vector = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def iter_jsonl(paths: Sequence[str | Path]) -> Iterator[dict]:
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {path}:{line_number}: {exc}") from exc
                if isinstance(item, dict):
                    yield item


def extract_messages(record: dict) -> List[dict]:
    messages = record.get("messages")
    if isinstance(messages, list):
        cleaned = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower().strip()
            content = normalize_text(message.get("content", ""))
            if role in {"system", "user", "assistant"} and content:
                cleaned.append({"role": role, "content": content})
        return cleaned
    user = normalize_text(record.get("prompt") or record.get("instruction") or record.get("pergunta") or "")
    assistant = normalize_text(record.get("response") or record.get("output") or record.get("resposta") or "")
    system = normalize_text(record.get("system") or "")
    result = []
    if system:
        result.append({"role": "system", "content": system})
    if user:
        result.append({"role": "user", "content": user})
    if assistant:
        result.append({"role": "assistant", "content": assistant})
    return result


def validate_messages(messages: Sequence[dict]) -> bool:
    return (len(messages) >= 2
            and any(msg["role"] == "user" for msg in messages)
            and messages[-1]["role"] == "assistant"
            and sum(len(msg["content"]) for msg in messages if msg["role"] == "assistant") >= 5)


def canonical_conversation(messages: Sequence[dict]) -> str:
    return "\n".join(f"{msg['role']}:{normalized_key(msg['content'])}" for msg in messages)


def split_name(record: dict, messages: Sequence[dict], validation_ratio: float = 0.02) -> str:
    group = str(record.get("group_id") or record.get("source_id") or "")
    if not group:
        group = normalized_key(next((m["content"] for m in messages if m["role"] == "user"), canonical_conversation(messages)))
    digest = hashlib.blake2b(group.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") / float(2**64 - 1)
    return "validation" if bucket < validation_ratio else "train"


def special_id(tokenizer: TokenizerLike, token: str, fallback: Optional[int] = None) -> int:
    value = tokenizer.vocab.get(token)
    if value is None:
        if fallback is None:
            raise KeyError(f"Token especial ausente no vocabulário: {token}")
        return fallback
    return int(value)


def encode_sft_record(tokenizer: TokenizerLike, record: dict) -> Optional[SFTExample]:
    messages = extract_messages(record)
    if not validate_messages(messages):
        return None
    input_ids, labels = [], []
    role_tokens = {"system": "<sistema>", "user": "<vitor>", "assistant": "<keilinks>"}
    for message in messages:
        segment = tokenizer.encode(f"{role_tokens[message['role']]}{message['content']}<fim>")
        if not segment:
            continue
        input_ids.extend(segment)
        labels.extend(segment if message["role"] == "assistant" else [IGNORE_INDEX] * len(segment))
    if len(input_ids) < 2:
        return None
    shifted_inputs, shifted_labels = input_ids[:-1], labels[1:]
    example_id = str(record.get("id") or hashlib.sha1(canonical_conversation(messages).encode()).hexdigest())
    return SFTExample(shifted_inputs, shifted_labels, example_id,
                      str(record.get("source") or "unknown"), str(record.get("category") or "general"))


def pack_examples(examples: Iterable[SFTExample], context_length: int, pad_id: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    input_buffer, label_buffer = [], []
    for example in examples:
        if len(example.input_ids) > context_length:
            example.input_ids = example.input_ids[-context_length:]
            example.labels = example.labels[-context_length:]
        if len(input_buffer) + len(example.input_ids) > context_length:
            padding = context_length - len(input_buffer)
            input_buffer.extend([pad_id] * padding)
            label_buffer.extend([IGNORE_INDEX] * padding)
            yield np.asarray(input_buffer, dtype=np.int32), np.asarray(label_buffer, dtype=np.int32)
            input_buffer, label_buffer = [], []
        input_buffer.extend(example.input_ids)
        label_buffer.extend(example.labels)
    if input_buffer:
        padding = context_length - len(input_buffer)
        input_buffer.extend([pad_id] * padding)
        label_buffer.extend([IGNORE_INDEX] * padding)
        yield np.asarray(input_buffer, dtype=np.int32), np.asarray(label_buffer, dtype=np.int32)


def _write_blocks(blocks, output_dir: Path, split: str, context_length: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(blocks)
    if count == 0:
        return 0
    input_mm = np.memmap(output_dir / f"{split}_input_ids.bin", dtype=np.int32,
                         mode="w+", shape=(count, context_length))
    label_mm = np.memmap(output_dir / f"{split}_labels.bin", dtype=np.int32,
                         mode="w+", shape=(count, context_length))
    for idx, (input_ids, labels) in enumerate(blocks):
        input_mm[idx], label_mm[idx] = input_ids, labels
    input_mm.flush(); label_mm.flush()
    return count


def build_packed_dataset(tokenizer: TokenizerLike, input_paths: Sequence[str | Path],
                         output_dir: str | Path, context_length: int = 2048,
                         validation_ratio: float = 0.02,
                         near_duplicate_hamming: int = 3) -> BuildStats:
    stats = BuildStats()
    seen_exact, simhash_buckets = set(), {}
    examples = {"train": [], "validation": []}
    for record in iter_jsonl(input_paths):
        stats.read += 1
        messages = extract_messages(record)
        if not validate_messages(messages):
            stats.rejected += 1; continue
        canonical = canonical_conversation(messages)
        exact = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if exact in seen_exact:
            stats.exact_duplicates += 1; continue
        seen_exact.add(exact)
        sh, bucket = simhash64(canonical), simhash64(canonical) >> 48
        nearby = simhash_buckets.setdefault(bucket, [])
        if any(hamming_distance(sh, old) <= near_duplicate_hamming for old in nearby[-2000:]):
            stats.near_duplicates += 1; continue
        nearby.append(sh)
        example = encode_sft_record(tokenizer, record)
        if example is None or not any(label != IGNORE_INDEX for label in example.labels):
            stats.rejected += 1; continue
        examples[split_name(record, messages, validation_ratio)].append(example)
        stats.accepted += 1
    pad_id = special_id(tokenizer, "<pad>", fallback=0)
    train_blocks = list(pack_examples(examples["train"], context_length, pad_id))
    val_blocks = list(pack_examples(examples["validation"], context_length, pad_id))
    out = Path(output_dir)
    stats.train_blocks = _write_blocks(train_blocks, out, "train", context_length)
    stats.validation_blocks = _write_blocks(val_blocks, out, "validation", context_length)
    stats.train_examples = len(examples["train"])
    stats.validation_examples = len(examples["validation"])
    metadata = {"format": "keilinks-packed-sft-v4", "context_length": context_length,
                "dtype": "int32", "ignore_index": IGNORE_INDEX, "stats": stats.to_dict()}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


class PackedBinaryDataset(Dataset):
    def __init__(self, directory: str | Path, split: str) -> None:
        self.directory = Path(directory)
        metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        self.context_length = int(metadata["context_length"])
        key = "train_blocks" if split == "train" else "validation_blocks"
        self.count = int(metadata["stats"][key])
        if self.count <= 0:
            raise ValueError(f"Split {split} vazio em {directory}")
        self.inputs = np.memmap(self.directory / f"{split}_input_ids.bin", dtype=np.int32,
                                mode="r", shape=(self.count, self.context_length))
        self.labels = np.memmap(self.directory / f"{split}_labels.bin", dtype=np.int32,
                                mode="r", shape=(self.count, self.context_length))

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return (torch.from_numpy(np.asarray(self.inputs[index]).copy()).long(),
                torch.from_numpy(np.asarray(self.labels[index]).copy()).long())
