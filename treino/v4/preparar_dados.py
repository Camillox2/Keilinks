"""Baixa e prepara dados locais para a Keilinks V4.

Arquivos grandes não entram no Git. Fontes padrão: FineWeb2 por_Latn,
Wikipedia PT, OpenAssistant OASST2 e bases PT-BR opcionais de instrução.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from treino.v4.dataset import (
    build_packed_dataset,
    canonical_conversation,
    normalize_text,
    normalized_key,
    simhash64,
    validate_messages,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dados" / "v4"
PRETRAIN_DIR = DATA_ROOT / "pretrain"
SFT_DIR = DATA_ROOT / "sft"
MANIFEST_DIR = DATA_ROOT / "manifests"


@dataclass
class SourceManifest:
    name: str
    dataset_id: str
    subset: str
    split: str
    license: str
    purpose: str
    downloaded_at: str
    records_seen: int = 0
    records_written: int = 0
    bytes_written: int = 0
    exact_duplicates: int = 0
    rejected: int = 0


class DeduplicatingTextWriter:
    def __init__(self, path: Path, max_bytes: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.max_bytes = max_bytes
        self.seen_exact: set[str] = set()
        self.simhash_buckets: Dict[int, list[int]] = {}
        self.bytes_written = self.written = self.duplicates = self.rejected = 0
        self.reached_limit = False

    def add(self, text: str, metadata: Optional[dict] = None) -> bool:
        text = clean_document(text)
        if not text:
            self.rejected += 1
            return False
        canonical = normalized_key(text)
        digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()
        if digest in self.seen_exact:
            self.duplicates += 1
            return False
        sh = simhash64(canonical)
        bucket = sh >> 48
        recent = self.simhash_buckets.setdefault(bucket, [])
        if any((sh ^ old).bit_count() <= 2 for old in recent[-500:]):
            self.duplicates += 1
            return False
        line = json.dumps({"text": text, **(metadata or {})}, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        if self.bytes_written + len(encoded) > self.max_bytes:
            self.reached_limit = True
            return False
        self.seen_exact.add(digest)
        recent.append(sh)
        self.handle.write(line)
        self.bytes_written += len(encoded)
        self.written += 1
        return True

    @property
    def full(self) -> bool:
        return self.reached_limit or self.bytes_written >= self.max_bytes

    def close(self) -> None:
        self.handle.close()


def clean_document(text: str) -> Optional[str]:
    text = normalize_text(text)
    if len(text) < 200:
        return None
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", "<email>", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>", text)
    text = re.sub(r"([!?.,])\1{4,}", r"\1\1", text)
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    text = "\n".join(lines).strip()
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < 40 or len(set(words)) / max(len(words), 1) < 0.12:
        return None
    if sum(ch.isalpha() or ch.isspace() for ch in text) / max(len(text), 1) < 0.65:
        return None
    return text


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_manifest(manifest: SourceManifest) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"{manifest.name}.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def download_fineweb2(target_bytes: int) -> SourceManifest:
    from datasets import load_dataset

    manifest = SourceManifest(
        "fineweb2_portuguese", "HuggingFaceFW/fineweb-2", "por_Latn",
        "train", "ODC-By-1.0", "pretrain", now_iso(),
    )
    writer = DeduplicatingTextWriter(PRETRAIN_DIR / "fineweb2_por.jsonl", target_bytes)
    dataset = load_dataset(
        manifest.dataset_id, name=manifest.subset,
        split=manifest.split, streaming=True,
    )
    try:
        for item in dataset:
            manifest.records_seen += 1
            writer.add(item.get("text", ""), {
                "source": manifest.name,
                "url": item.get("url", ""),
                "date": item.get("date", item.get("timestamp", "")),
            })
            if writer.full:
                break
            if manifest.records_seen % 10000 == 0:
                print(
                    f"FineWeb2: {writer.written:,} docs | "
                    f"{writer.bytes_written/1e9:.2f} GB"
                )
    finally:
        writer.close()
    manifest.records_written = writer.written
    manifest.bytes_written = writer.bytes_written
    manifest.exact_duplicates = writer.duplicates
    manifest.rejected = writer.rejected
    save_manifest(manifest)
    return manifest


def latest_wikipedia_pt_config() -> str:
    from datasets import get_dataset_config_names

    configs = sorted(
        config for config in get_dataset_config_names("wikimedia/wikipedia")
        if config.endswith(".pt")
    )
    if not configs:
        raise RuntimeError("Nenhuma configuração PT encontrada em wikimedia/wikipedia")
    return configs[-1]


def download_wikipedia(target_bytes: int) -> SourceManifest:
    from datasets import load_dataset

    subset = latest_wikipedia_pt_config()
    manifest = SourceManifest(
        "wikipedia_portuguese", "wikimedia/wikipedia", subset,
        "train", "CC-BY-SA-3.0/GFDL", "pretrain", now_iso(),
    )
    writer = DeduplicatingTextWriter(PRETRAIN_DIR / "wikipedia_pt.jsonl", target_bytes)
    dataset = load_dataset(
        manifest.dataset_id, name=subset, split="train", streaming=True
    )
    try:
        for item in dataset:
            manifest.records_seen += 1
            title = normalize_text(item.get("title", ""))
            writer.add(
                f"{title}\n\n{item.get('text', '')}",
                {"source": manifest.name, "title": title},
            )
            if writer.full:
                break
    finally:
        writer.close()
    manifest.records_written = writer.written
    manifest.bytes_written = writer.bytes_written
    manifest.exact_duplicates = writer.duplicates
    manifest.rejected = writer.rejected
    save_manifest(manifest)
    return manifest


def good_oasst_message(item: dict) -> bool:
    return (
        str(item.get("lang", "")).lower() in {"pt", "pt-br"}
        and not bool(item.get("deleted", False))
        and item.get("review_result", True) is not False
        and bool(normalize_text(item.get("text", "")))
    )


def oasst_conversation(item: dict, by_id: Dict[str, dict], max_messages: int = 8) -> List[dict]:
    chain: List[dict] = []
    current = item
    seen: set[str] = set()
    while current and len(chain) < max_messages:
        message_id = str(current.get("message_id") or "")
        if not message_id or message_id in seen or not good_oasst_message(current):
            return []
        seen.add(message_id)
        role = {"prompter": "user", "assistant": "assistant"}.get(current.get("role"))
        if role is None:
            return []
        chain.append({"role": role, "content": normalize_text(current.get("text", ""))})
        parent_id = current.get("parent_id")
        current = by_id.get(parent_id) if parent_id else None
    chain.reverse()
    while chain and chain[0]["role"] != "user":
        chain.pop(0)
    cleaned: List[dict] = []
    expected = "user"
    for message in chain:
        if message["role"] != expected:
            return []
        cleaned.append(message)
        expected = "assistant" if expected == "user" else "user"
    return cleaned if validate_messages(cleaned) else []


def download_oasst2() -> SourceManifest:
    from datasets import load_dataset

    manifest = SourceManifest(
        "oasst2_portuguese", "OpenAssistant/oasst2", "default",
        "train+validation", "Apache-2.0", "sft", now_iso(),
    )
    dataset = load_dataset(manifest.dataset_id)
    items = [
        dict(item)
        for split in ("train", "validation")
        for item in dataset[split]
    ]
    by_id = {
        item.get("message_id"): item for item in items if item.get("message_id")
    }
    output = SFT_DIR / "oasst2_pt.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            manifest.records_seen += 1
            if item.get("role") != "assistant":
                continue
            messages = oasst_conversation(item, by_id)
            if not messages:
                manifest.rejected += 1
                continue
            canonical = canonical_conversation(messages)
            key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if key in seen:
                manifest.exact_duplicates += 1
                continue
            seen.add(key)
            tree_id = item.get("message_tree_id") or item.get("message_id")
            record = {
                "id": f"oasst2:{item.get('message_id')}",
                "group_id": f"oasst2-tree:{tree_id}",
                "source": manifest.name,
                "source_id": tree_id,
                "category": "general",
                "license": manifest.license,
                "messages": messages,
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            handle.write(line)
            manifest.records_written += 1
            manifest.bytes_written += len(line.encode("utf-8"))
    save_manifest(manifest)
    return manifest


def download_instruction_dataset(dataset_id: str, output_name: str,
                                  license_name: str,
                                  field_map: Dict[str, str]) -> SourceManifest:
    from datasets import load_dataset

    manifest = SourceManifest(
        output_name, dataset_id, "default", "train",
        license_name, "sft", now_iso(),
    )
    dataset = load_dataset(dataset_id, split="train", streaming=True)
    output = SFT_DIR / f"{output_name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with output.open("w", encoding="utf-8") as handle:
        for item in dataset:
            manifest.records_seen += 1
            prompt = normalize_text(item.get(field_map["prompt"], ""))
            extra = (
                normalize_text(item.get(field_map.get("input", ""), ""))
                if field_map.get("input") else ""
            )
            response = normalize_text(item.get(field_map["response"], ""))
            if extra:
                prompt = f"{prompt}\n\n{extra}"
            if len(prompt) < 3 or len(response) < 10:
                manifest.rejected += 1
                continue
            key = hashlib.sha256(
                f"{normalized_key(prompt)}\n{normalized_key(response)}".encode()
            ).hexdigest()
            if key in seen:
                manifest.exact_duplicates += 1
                continue
            seen.add(key)
            record = {
                "id": f"{output_name}:{key[:16]}",
                "group_id": f"{output_name}:{key[:16]}",
                "source": output_name,
                "category": "instruction",
                "license": license_name,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            handle.write(line)
            manifest.records_written += 1
            manifest.bytes_written += len(line.encode("utf-8"))
    save_manifest(manifest)
    return manifest


def merge_sft() -> Path:
    output = SFT_DIR / "all_sft.jsonl"
    sources = sorted(
        path for path in SFT_DIR.glob("*.jsonl") if path.name != output.name
    )
    curated = [
        DATA_ROOT / "conversas_curadas_v4.jsonl",
        DATA_ROOT / "seed_conversas.jsonl",
    ]
    sources = [path for path in curated if path.exists()] + sources
    with output.open("w", encoding="utf-8") as destination:
        for source in sources:
            with source.open("r", encoding="utf-8", errors="replace") as handle:
                shutil.copyfileobj(handle, destination)
    return output


def export_pretrain_txt() -> Path:
    output = PRETRAIN_DIR / "pretrain_pt.txt"
    with output.open("w", encoding="utf-8") as destination:
        for source in sorted(PRETRAIN_DIR.glob("*.jsonl")):
            for line in source.open("r", encoding="utf-8", errors="replace"):
                try:
                    text = normalize_text(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
                if text:
                    destination.write(text + "\n\n")
    return output


def prepare_packed(vocab_path: str, context_length: int) -> None:
    from treino.v4.tokenizador import TokenizadorV4

    tokenizer = TokenizadorV4(vocab_path)
    stats = build_packed_dataset(
        tokenizer, [merge_sft()], DATA_ROOT / "packed",
        context_length=context_length,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--pretrain-gb", type=float, default=10.0)
    download.add_argument("--wiki-gb", type=float, default=1.5)
    download.add_argument(
        "--sources", default="fineweb2,wikipedia,oasst2,alpaca,dolly"
    )
    sub.add_parser("merge-sft")
    sub.add_parser("export-pretrain")
    packed = sub.add_parser("pack-sft")
    packed.add_argument("--vocab", default="dados/v4/pretrain/tokenizer.json")
    packed.add_argument("--context", type=int, default=2048)
    args = parser.parse_args()

    for directory in (PRETRAIN_DIR, SFT_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if args.command == "download":
        sources = {value.strip() for value in args.sources.split(",")}
        if "fineweb2" in sources:
            download_fineweb2(int(args.pretrain_gb * 1e9))
        if "wikipedia" in sources:
            download_wikipedia(int(args.wiki_gb * 1e9))
        if "oasst2" in sources:
            download_oasst2()
        if "alpaca" in sources:
            try:
                download_instruction_dataset(
                    "dominguesm/alpaca-data-pt-br", "alpaca_ptbr",
                    "verificar-dataset-card",
                    {"prompt": "instruction", "input": "input", "response": "output"},
                )
            except Exception as exc:
                print(f"[aviso] Alpaca falhou: {exc}")
        if "dolly" in sources:
            try:
                download_instruction_dataset(
                    "Gustrd/dolly-15k-libretranslate-pt", "dolly_ptbr",
                    "confirmar-no-card",
                    {"prompt": "instruction", "response": "output"},
                )
            except Exception as exc:
                print(f"[aviso] Dolly falhou: {exc}")
        merge_sft()
        export_pretrain_txt()
    elif args.command == "merge-sft":
        merge_sft()
    elif args.command == "export-pretrain":
        export_pretrain_txt()
    else:
        prepare_packed(args.vocab, args.context)


if __name__ == "__main__":
    main()
