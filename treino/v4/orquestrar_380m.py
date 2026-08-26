"""Orquestra o primeiro ciclo seguro do Core Keilinks 380M.

O processo espera uma coleta finalizada, monta o corpus, constrói o
tokenizador, prepara os binários e executa um burn-in antes do pré-treino
longo. Ele não baixa dados nem apaga artefatos: cada estágio concluído é
reutilizado apenas quando seu artefato e manifesto existem.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[orquestrar-380m] {message}", flush=True)


def require_free_space(path: Path, minimum_gib: float) -> None:
    available = shutil.disk_usage(path).free / 1024**3
    if available < minimum_gib:
        raise RuntimeError(
            f"Espaço livre insuficiente: {available:.1f} GiB; "
            f"são necessários ao menos {minimum_gib:.1f} GiB."
        )


def run_module(*args: str) -> None:
    command = [sys.executable, "-u", "-m", *args]
    log("Executando: " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON inválido em {path}")
    return value


def wait_for_collection(run_dir: Path, poll_seconds: int, max_wait_hours: float) -> dict[str, Any]:
    manifest_path = run_dir / "run.manifest.json"
    deadline = time.monotonic() + max_wait_hours * 3600
    while True:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("status") != "complete":
                raise RuntimeError(
                    f"Coleta terminou sem status complete: {manifest.get('status')!r}"
                )
            if int(manifest.get("documents", 0)) < 100:
                raise RuntimeError("Coleta finalizada com menos de 100 documentos")
            log(
                f"Coleta confirmada: {manifest['documents']:,} documentos, "
                f"{int(manifest.get('bytes', 0)) / 1024**3:.2f} GiB"
            )
            return manifest
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Manifesto de coleta não surgiu em {max_wait_hours} horas")
        log(f"Aguardando coleta: {run_dir}")
        time.sleep(poll_seconds)


def read_burn_in_health(log_path: Path, max_vram_gib: float) -> dict[str, float]:
    if not log_path.exists():
        raise RuntimeError(f"Log de burn-in ausente: {log_path}")
    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and "loss" in event:
                events.append(event)
    if len(events) < 3:
        raise RuntimeError("Burn-in produziu poucos pontos de telemetria")

    losses = [float(event["loss"]) for event in events]
    vrams = [float(event.get("vram_gb", 0.0)) for event in events]
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("Burn-in registrou loss não finita")
    if max(vrams) > max_vram_gib:
        raise RuntimeError(
            f"Burn-in excedeu a guarda de VRAM: {max(vrams):.2f} GiB > {max_vram_gib:.2f} GiB"
        )
    if losses[-1] > losses[0] * 1.05:
        raise RuntimeError(
            f"Burn-in não estabilizou: loss {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
    health = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "peak_vram_gb": max(vrams),
        "events": float(len(events)),
    }
    log("Burn-in saudável: " + json.dumps(health, ensure_ascii=False))
    return health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestra a coleta já iniciada e o pré-treino seguro do Core 380M"
    )
    parser.add_argument("--run-id", default="public-pt-380m-01")
    parser.add_argument("--corpus", default="dados/v4/pretrain/pretrain_pt.txt")
    parser.add_argument("--vocab", default="dados/v4/pretrain/tokenizer.json")
    parser.add_argument("--sft", default="dados/v4/sft/all_sft_380m.jsonl")
    parser.add_argument("--binary-dir", default="dados/v4/pretrain_binary")
    parser.add_argument("--output", default="checkpoints/v4-pretrain-380m")
    parser.add_argument("--burn-in-steps", type=int, default=400)
    parser.add_argument("--long-steps", type=int, default=160_000)
    parser.add_argument("--continue-after-burn-in", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=48.0)
    parser.add_argument("--minimum-free-gib", type=float, default=55.0)
    parser.add_argument("--max-vram-gib", type=float, default=7.4)
    args = parser.parse_args()
    if args.burn_in_steps < 60:
        parser.error("--burn-in-steps deve ser ao menos 60")
    if args.long_steps <= args.burn_in_steps:
        parser.error("--long-steps deve ser maior que --burn-in-steps")
    if args.poll_seconds < 10:
        parser.error("--poll-seconds deve ser ao menos 10")
    if args.minimum_free_gib <= 0 or args.max_vram_gib <= 0:
        parser.error("guardas de espaço e VRAM devem ser positivas")
    return args


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "dados" / "v4" / "raw" / args.run_id
    corpus = ROOT / args.corpus
    vocab = ROOT / args.vocab
    sft = ROOT / args.sft
    binary_dir = ROOT / args.binary_dir
    output = ROOT / args.output
    if not sft.exists():
        raise FileNotFoundError(f"SFT não encontrado: {sft}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run de coleta não encontrado: {run_dir}")
    require_free_space(ROOT, args.minimum_free_gib)
    wait_for_collection(run_dir, args.poll_seconds, args.max_wait_hours)

    corpus_manifest = corpus.with_suffix(corpus.suffix + ".manifest.json")
    if not (corpus.exists() and corpus_manifest.exists()):
        require_free_space(ROOT, args.minimum_free_gib)
        run_module("treino.v4.montar_corpus", "--input", str(run_dir), "--output", str(corpus))
    else:
        log(f"Montagem já concluída: {corpus}")

    if not vocab.exists():
        run_module(
            "treino.v4.tokenizador", "build", "--pretrain", str(corpus),
            "--sft", str(sft), "--output", str(vocab), "--vocab-size", "32000",
            "--sample-mb", "256", "--backend", "hf_bytelevel",
        )
    else:
        log(f"Tokenizador já concluído: {vocab}")

    metadata = binary_dir / "metadata.json"
    if not metadata.exists():
        require_free_space(ROOT, args.minimum_free_gib)
        run_module(
            "treino.v4.pretreinar", "--model", "core_380m_modern",
            "--profile", "rtx5050_380m", "--input", str(corpus),
            "--vocab", str(vocab), "--binary-dir", str(binary_dir),
            "--prepare-only", "--rebuild-binary",
        )
    else:
        log(f"Binários já concluídos: {binary_dir}")

    burn_checkpoint = output / "pretrain_final.pt"
    if not burn_checkpoint.exists():
        run_module(
            "treino.v4.pretreinar", "--model", "core_380m_modern",
            "--profile", "rtx5050_380m", "--input", str(corpus),
            "--vocab", str(vocab), "--binary-dir", str(binary_dir),
            "--output", str(output), "--steps", str(args.burn_in_steps),
        )
    else:
        log(f"Checkpoint de burn-in já existe: {burn_checkpoint}")

    health = read_burn_in_health(output / "pretrain_log.jsonl", args.max_vram_gib)
    (output / "burn_in_health.json").write_text(
        json.dumps(health, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.continue_after_burn_in:
        log("Burn-in concluído. Pré-treino longo exige --continue-after-burn-in.")
        return

    log(f"Iniciando pré-treino longo até o passo {args.long_steps:,}")
    run_module(
        "treino.v4.pretreinar", "--model", "core_380m_modern",
        "--profile", "rtx5050_380m", "--input", str(corpus),
        "--vocab", str(vocab), "--binary-dir", str(binary_dir),
        "--output", str(output), "--steps", str(args.long_steps),
        "--resume", str(burn_checkpoint),
    )


if __name__ == "__main__":
    main()
