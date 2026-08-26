"""Mede janelas de contexto reais na GPU sem alterar o pré-treino.

O benchmark cria uma cópia temporária da configuração com a janela pedida.
Isso faz o cache RoPE e a validação do modelo refletirem 4k/8k/16k de verdade,
sem modificar o perfil usado pelo treinamento em andamento.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from treino.v4.config import TrainConfig, get_model_config, get_train_config
from treino.v4.modelo import KeilinksV4
from treino.v4.treinar import build_optimizer


def config_for_context(model_name: str, context: int):
    if context < 128:
        raise ValueError("O contexto precisa ter pelo menos 128 tokens")
    config = replace(get_model_config(model_name), context_length=context)
    config.validate()
    return config


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_training_case(model_name, profile, context, steps, warmup_steps,
                      compile_enabled, checkpoint_mode):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível")
    config = config_for_context(model_name, context)
    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats()
    train_config = get_train_config(profile)
    train_config = TrainConfig(
        **{**train_config.__dict__, "checkpoint_mode": checkpoint_mode}
    )
    model = optimizer = executable = x = y = None
    try:
        model = KeilinksV4(config).cuda()
        model.set_gradient_checkpointing(checkpoint_mode, train_config.checkpoint_every)
        model.train()
        optimizer = build_optimizer(model, train_config, torch.device("cuda"))
        executable = model
        if compile_enabled and hasattr(torch, "compile"):
            executable = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        x = torch.randint(0, config.vocab_size, (1, context), device="cuda")
        y = torch.randint(0, config.vocab_size, (1, context), device="cuda")
        for _ in range(warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = executable(x, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = executable(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {
            "workload": "train_forward_backward",
            "model": model_name,
            "context": context,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "compile": compile_enabled,
            "checkpoint": checkpoint_mode,
            "seconds": elapsed,
            "tokens_per_second": steps * context / elapsed,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "last_loss": float(loss.item()),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    finally:
        del executable, model, optimizer, x, y
        cleanup_cuda()


def run_inference_case(model_name, context, decode_tokens):
    """Mede prefill mais cache KV perto da janela completa, sem gradientes."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível")
    config = config_for_context(model_name, context)
    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats()
    model = prompt = generated = None
    try:
        model = KeilinksV4(config).cuda().eval()
        prompt_tokens = max(1, context - decode_tokens)
        prompt = torch.randint(0, config.vocab_size, (1, prompt_tokens), device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(prompt, max_new_tokens=decode_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {
            "workload": "inference_prefill_kv_cache",
            "model": model_name,
            "context": context,
            "prompt_tokens": prompt_tokens,
            "decode_tokens_requested": decode_tokens,
            "seconds": elapsed,
            "tokens_per_second": (prompt_tokens + decode_tokens) / elapsed,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "generated_tokens": int(generated.size(1) - prompt_tokens),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    finally:
        del generated, prompt, model
        cleanup_cuda()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="core_380m_modern")
    p.add_argument("--profile", default="rtx5050_380m")
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--decode-tokens", type=int, default=1)
    p.add_argument("--workload", choices=("train", "inference", "both"), default="train")
    p.add_argument("--output", default="checkpoints/v4/benchmark_rtx5050.json")
    p.add_argument("--compile", action="store_true", help="Mede também o caminho torch.compile")
    p.add_argument("--checkpoint", choices=("none", "selective", "full"), default="selective")
    p.add_argument(
        "--matrix",
        action="store_true",
        help="Testa todas as combinações; use apenas fora do treino",
    )
    args = p.parse_args()
    results = []
    cases = (
        [
            (compiled, mode)
            for compiled in (False, True)
            for mode in ("none", "selective", "full")
        ]
        if args.matrix
        else (
            [(False, args.checkpoint)]
            + ([(True, args.checkpoint)] if args.compile else [])
        )
    )
    for compiled, mode in cases:
        workloads = [args.workload] if args.workload != "both" else ["inference", "train"]
        for workload in workloads:
            # Compilar só é relevante para treino; evitar custo de compilação na
            # medição curta de inferência.
            if workload == "inference" and compiled:
                continue
            try:
                if workload == "train":
                    result = run_training_case(
                        args.model, args.profile, args.context, args.steps,
                        args.warmup_steps, compiled, mode,
                    )
                else:
                    result = run_inference_case(
                        args.model, args.context, args.decode_tokens,
                    )
                results.append(result)
                print(json.dumps(result, indent=2))
            except torch.OutOfMemoryError:
                cleanup_cuda()
                results.append({
                    "workload": workload,
                    "context": args.context,
                    "compile": compiled,
                    "checkpoint": mode,
                    "oom": True,
                })
            except Exception as exc:
                cleanup_cuda()
                results.append({
                    "workload": workload,
                    "context": args.context,
                    "compile": compiled,
                    "checkpoint": mode,
                    "error": str(exc),
                })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    valid = [result for result in results if "tokens_per_second" in result]
    if valid:
        fastest = max(valid, key=lambda result: result["tokens_per_second"])
        print("Melhor:", json.dumps(fastest, indent=2))


if __name__ == "__main__":
    main()
