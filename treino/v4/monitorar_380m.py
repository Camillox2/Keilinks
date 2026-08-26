"""Painel de terminal para a coleta e o pré-treino do Core Keilinks 380M.

Não controla nem interrompe o treino por padrão. ``--request-pause`` cria um
arquivo sentinela que o loop de pré-treino converte em checkpoint seguro após o
passo atual; ``--clear-pause`` apenas remove essa solicitação antes de retomar.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def human_size(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02}h {minutes:02}m"
    if hours:
        return f"{hours}h {minutes:02}m {seconds:02}s"
    return f"{minutes}m {seconds:02}s"


def color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def progress_bar(fraction: float, width: int = 28) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def panel_line(text: str = "") -> str:
    return "║ " + text[:62].ljust(62) + " ║"


def tail_lines(path: Path, maximum_bytes: int = 256 * 1024) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - maximum_bytes))
        data = handle.read().decode("utf-8", errors="replace")
    return data.splitlines()


def latest_training_event(log_path: Path) -> dict[str, Any] | None:
    for line in reversed(tail_lines(log_path)):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "step" in value and "loss" in value:
            return value
    return None


def collection_state(run_dir: Path, target_bytes: int) -> dict[str, Any]:
    manifest_path = run_dir / "run.manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = None
    shards = list(run_dir.rglob("shard-*.jsonl")) if run_dir.exists() else []
    bytes_written = sum(path.stat().st_size for path in shards)
    timestamps = [path.stat().st_ctime for path in shards if path.stat().st_size]
    elapsed = time.time() - min(timestamps) if timestamps else None
    rate = bytes_written / elapsed if elapsed and elapsed > 0 else None
    completed = bool(manifest and manifest.get("status") == "complete")
    final_bytes = int(manifest.get("bytes", bytes_written)) if manifest else bytes_written
    return {
        "bytes": final_bytes,
        "target": target_bytes,
        "fraction": min(final_bytes / max(target_bytes, 1), 1.0),
        "rate": rate,
        "eta": (target_bytes - bytes_written) / rate if rate and not completed else 0.0,
        "documents": int(manifest.get("documents", 0)) if manifest else None,
        "completed": completed,
        "shards": len(shards),
    }


def gpu_state() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        values = [item.strip() for item in result.stdout.strip().split(",")]
        if len(values) == 5:
            return {
                "name": values[0],
                "memory": f"{values[1]} / {values[2]} MiB",
                "utilization": f"{values[3]}%",
                "temperature": f"{values[4]}°C",
            }
    except (OSError, subprocess.SubprocessError):
        pass
    return {"name": "GPU indisponível", "memory": "—", "utilization": "—", "temperature": "—"}


def checkpoint_state(output_dir: Path) -> list[tuple[str, int, float]]:
    if not output_dir.exists():
        return []
    return sorted(
        (
            (path.name, path.stat().st_size, path.stat().st_mtime)
            for path in output_dir.glob("*.pt")
        ),
        key=lambda item: item[2],
        reverse=True,
    )


def stage_name(
    collection: dict[str, Any], corpus: Path, vocab: Path, binary_dir: Path,
    output_dir: Path, event: dict[str, Any] | None,
) -> str:
    if (output_dir / "pretrain_paused.pt").exists() and (output_dir / "PAUSE_REQUESTED").exists():
        return "PAUSADO COM CHECKPOINT"
    if event:
        return "BURN-IN" if int(event["step"]) < 400 else "PRÉ-TREINO LONGO"
    if not collection["completed"]:
        return "COLETANDO CORPUS"
    if not corpus.exists():
        return "MONTANDO CORPUS"
    if not vocab.exists():
        return "TREINANDO TOKENIZADOR"
    if not (binary_dir / "metadata.json").exists():
        return "TOKENIZANDO PARA BINÁRIO"
    if (output_dir / "pretrain_final.pt").exists():
        return "PRÉ-TREINO CONCLUÍDO"
    return "AGUARDANDO BURN-IN"


def render(args: argparse.Namespace, *, clear: bool) -> None:
    ansi = sys.stdout.isatty() and not args.no_color
    run_dir = resolve(f"dados/v4/raw/{args.run_id}")
    corpus = resolve(args.corpus)
    vocab = resolve(args.vocab)
    binary_dir = resolve(args.binary_dir)
    output_dir = resolve(args.output)
    collection = collection_state(run_dir, int(args.target_gib * 1024**3))
    event = latest_training_event(output_dir / "pretrain_log.jsonl")
    stage = stage_name(collection, corpus, vocab, binary_dir, output_dir, event)
    gpu = gpu_state()
    checkpoints = checkpoint_state(output_dir)
    free = shutil.disk_usage(ROOT).free

    if clear:
        print("\033[2J\033[H", end="")
    title = " KEILINKS • CORE 380M • PAINEL AO VIVO "
    print("╔" + "═" * 64 + "╗")
    print("║ " + color(title.center(62), "1;96", ansi) + " ║")
    print("╠" + "═" * 64 + "╣")
    print(panel_line(f"Etapa: {stage}"))
    fraction = float(collection["fraction"])
    amount = f"{human_size(float(collection['bytes']))} / {args.target_gib:.1f} GiB"
    print(panel_line(f"Corpus: [{progress_bar(fraction)}] {fraction * 100:5.1f}%"))
    print(panel_line(f"       {amount}"))
    if collection["completed"]:
        document_label = (
            f"{collection['documents']:,} documentos"
            if collection["documents"]
            else "manifesto concluído"
        )
        print(panel_line(f"       {document_label}"))
    else:
        rate = collection["rate"]
        rate_label = f"{human_size(float(rate))}/s" if rate else "calculando taxa"
        print(panel_line(f"       {rate_label} | ETA coleta: {human_duration(collection['eta'])}"))
    print("╠" + "─" * 64 + "╣")
    if event:
        step = int(event["step"])
        total = int(args.long_steps)
        speed = float(event.get("tokens_per_second", 0.0))
        remaining_tokens = max(total - step - 1, 0) * args.grad_accum_steps * args.context
        eta = remaining_tokens / speed if speed > 0 else None
        print(panel_line(
            f"Treino: passo {step:,}/{total:,}  loss {float(event['loss']):.4f} "
            f"LR {float(event.get('lr', 0.0)):.2e}"
        ))
        print(panel_line(
            f"        {speed:,.0f} tok/s | VRAM {float(event.get('vram_gb', 0.0)):.2f} GiB "
            f"| ETA {human_duration(eta)}"
        ))
    else:
        print(panel_line("Treino: ainda não iniciou; o orquestrador avança sozinho."))
    gpu_label = (
        f"GPU: {gpu['name'][:25]} | {gpu['memory']} | "
        f"uso {gpu['utilization']} | {gpu['temperature']}"
    )
    print(panel_line(gpu_label))
    print(panel_line(f"Disco livre: {human_size(free)}"))
    print("╠" + "─" * 64 + "╣")
    if checkpoints:
        print(panel_line("Checkpoints mais recentes:"))
        for name, size, modified in checkpoints[:3]:
            when = datetime.fromtimestamp(modified).strftime("%d/%m %H:%M")
            print(panel_line(f"  {name:<29} {human_size(size):>10}  {when}"))
    else:
        print(panel_line("Checkpoints: burn-in e, no longo, a cada 1.000 passos."))
    print("╠" + "─" * 64 + "╣")
    print(panel_line("Pausa segura: python -m treino.v4.monitorar_380m"))
    print(panel_line("               --request-pause"))
    print(panel_line("Ctrl+C encerra o painel; não interrompe coleta nem treino."))
    print("╚" + "═" * 64 + "╝")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Painel ao vivo do treino Keilinks 380M")
    parser.add_argument("--run-id", default="public-pt-380m-01")
    parser.add_argument("--corpus", default="dados/v4/pretrain/pretrain_pt.txt")
    parser.add_argument("--vocab", default="dados/v4/pretrain/tokenizer.json")
    parser.add_argument("--binary-dir", default="dados/v4/pretrain_binary")
    parser.add_argument("--output", default="checkpoints/v4-pretrain-380m")
    parser.add_argument("--target-gib", type=float, default=20.0)
    parser.add_argument("--long-steps", type=int, default=160_000)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--request-pause", action="store_true")
    parser.add_argument("--clear-pause", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval deve ser ao menos um segundo")
    if args.request_pause and args.clear_pause:
        parser.error("use apenas uma ação de pausa por vez")
    return args


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    args = parse_args()
    pause_file = resolve(args.output) / "PAUSE_REQUESTED"
    if args.request_pause:
        pause_file.parent.mkdir(parents=True, exist_ok=True)
        pause_file.write_text(
            f"solicitado em {datetime.now().isoformat()}\n", encoding="utf-8"
        )
        print(f"Pausa segura solicitada: {pause_file}")
        return
    if args.clear_pause:
        pause_file.unlink(missing_ok=True)
        print(f"Solicitação de pausa removida: {pause_file}")
        return

    try:
        while True:
            render(args, clear=not args.once and sys.stdout.isatty())
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nPainel encerrado; a coleta/treino continuam em segundo plano.")


if __name__ == "__main__":
    main()
