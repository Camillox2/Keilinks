"""Protótipo experimental de recompensa para pesquisa V4.

Não é um motor de melhoria contínua de produção: não implementa DPO,
replay buffer, judge humano, gates nem rollback. Use o pipeline V5 de
feedback consentido e treino offline antes de promover qualquer checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from treino.v4.config import ModelConfig, TrainConfig, get_model_config
from treino.v4.modelo import KeilinksV4
from treino.v4.tokenizador import TokenizadorV4


@dataclass
class CandidateResponse:
    text: str
    tokens: List[int]
    log_probs: List[float]
    reward: float = 0.0
    failures: List[str] = None


class RuleBasedVerifier:
    """Verificador determinístico de qualidade, formato e segurança em PT-BR."""

    def __init__(self):
        self.prohibited_markers = [
            "<sistema>", "<vitor>", "<user>", "<fim>", "<keilinks>",
            "como um modelo de linguagem", "sou apenas uma ia",
        ]

    def evaluate(self, prompt: str, response: str) -> Tuple[float, List[str]]:
        failures = []
        reward = 1.0

        clean = response.strip()
        words = re.findall(r"\w+", clean, flags=re.UNICODE)

        # 1. Checagem de tamanho
        if len(words) < 3:
            failures.append("too_short")
            reward -= 0.5
        elif len(words) > 400:
            failures.append("too_long")
            reward -= 0.2

        # 2. Checagem de repetição / colapso
        if len(words) >= 15:
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
            if unique_ratio < 0.40:
                failures.append("high_repetition")
                reward -= 0.6

        # 3. Checagem de marcadores vazados
        for marker in self.prohibited_markers:
            if marker in clean.lower():
                failures.append(f"leaked_marker:{marker}")
                reward -= 0.4

        # 4. Checagem de caracteres de substituição UTF-8.
        # str.count("") é sempre positivo em Python e penalizava toda resposta.
        if clean.count("\ufffd") > 0:
            failures.append("utf8_corruption")
            reward -= 0.5

        # 5. Coerência básica
        if clean.endswith(("...", ",", "—", " e ", " que ")):
            failures.append("incomplete_sentence")
            reward -= 0.2

        reward = max(0.0, min(1.0, reward))
        return reward, failures


class SelfImproverEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        device: Optional[str] = None,
        beta_dpo: float = 0.1,
        beta_grpo_kl: float = 0.04,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = TokenizadorV4(vocab_path)
        self.verifier = RuleBasedVerifier()
        self.beta_dpo = beta_dpo
        self.beta_grpo_kl = beta_grpo_kl

        # Carrega modelo de política
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ModelConfig(**raw["config"])
        self.model = KeilinksV4(self.config)
        self.model.load_state_dict(raw["model"], strict=True)
        self.model.to(self.device)

        # Cria cópia de referência congelada (para DPO / GRPO KL)
        self.ref_model = KeilinksV4(self.config)
        self.ref_model.load_state_dict(raw["model"], strict=True)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def generate_group(
        self, prompt_text: str, group_size: int = 4, max_tokens: int = 150, temperature: float = 0.85
    ) -> List[CandidateResponse]:
        """Gera um grupo de G respostas candidatas para cálculo de vantagem relativa (GRPO)."""
        system = "<sistema>Você é Keilinks, uma IA brasileira acolhedora, honesta e inteligente.<fim>"
        full_prompt = f"{system}<vitor>{prompt_text}<fim><keilinks>"
        prompt_ids = self.tokenizer.encode(full_prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        candidates = []
        for _ in range(group_size):
            self.model.eval()
            with torch.no_grad():
                out = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.92,
                    eos_id=self.tokenizer.vocab.get("<fim>"),
                )
            generated_ids = out[0, len(prompt_ids):].tolist()
            text = self.tokenizer.decode(generated_ids).strip()
            if "<fim>" in text:
                text = text.split("<fim>")[0].strip()

            reward, failures = self.verifier.evaluate(prompt_text, text)
            candidates.append(
                CandidateResponse(
                    text=text,
                    tokens=generated_ids,
                    log_probs=[],
                    reward=reward,
                    failures=failures,
                )
            )

        return candidates

    def compute_grpo_loss(
        self,
        prompt_ids: List[int],
        candidates: List[CandidateResponse],
    ) -> torch.Tensor:
        """Calcula a perda GRPO com vantagens relativas normalizadas no grupo."""
        rewards = torch.tensor([c.reward for c in candidates], dtype=torch.float32, device=self.device)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        for idx, candidate in enumerate(candidates):
            if not candidate.tokens:
                continue
            adv = advantages[idx].item()
            seq = prompt_ids + candidate.tokens
            input_tensor = torch.tensor([seq[:-1]], dtype=torch.long, device=self.device)
            target_tensor = torch.tensor([seq[1:]], dtype=torch.long, device=self.device)

            # Mask everything except candidate response tokens
            mask = torch.zeros_like(target_tensor, dtype=torch.bool)
            start_pos = len(prompt_ids) - 1
            mask[:, start_pos:] = True

            logits, _ = self.model(input_tensor)
            with torch.no_grad():
                ref_logits, _ = self.ref_model(input_tensor)

            log_probs = F.log_softmax(logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # Cross entropy per token
            token_log_probs = log_probs.gather(2, target_tensor.unsqueeze(-1)).squeeze(-1)
            ref_token_log_probs = ref_log_probs.gather(2, target_tensor.unsqueeze(-1)).squeeze(-1)

            # GRPO surrogate objective
            ratio = torch.exp(token_log_probs - ref_token_log_probs)
            kl_div = torch.exp(ref_token_log_probs) * (ref_token_log_probs - token_log_probs)

            token_loss = -(ratio * adv - self.beta_grpo_kl * kl_div)
            masked_loss = (token_loss * mask).sum() / mask.sum().clamp_min(1)

            total_loss = total_loss + masked_loss
            count += 1

        return total_loss / max(count, 1)

    def train_step(self, optimizer: torch.optim.Optimizer, prompts: List[str], group_size: int = 4) -> float:
        """Executa um passo de treinamento GRPO em lote."""
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for prompt in prompts:
            system = "<sistema>Você é Keilinks, uma IA brasileira acolhedora, honesta e inteligente.<fim>"
            prompt_ids = self.tokenizer.encode(f"{system}<vitor>{prompt}<fim><keilinks>")
            candidates = self.generate_group(prompt, group_size=group_size)

            loss = self.compute_grpo_loss(prompt_ids, candidates)
            loss.backward()
            batch_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return batch_loss / max(len(prompts), 1)


def run_continuous_improvement_loop(
    checkpoint_path: str,
    vocab_path: str,
    output_dir: str = "checkpoints/v5-self-improved",
    iterations: int = 5,
    steps_per_iter: int = 50,
):
    """Executa o loop completo de auto-melhoria contínua."""
    engine = SelfImproverEngine(checkpoint_path, vocab_path)
    optimizer = torch.optim.AdamW(engine.model.parameters(), lr=1e-5, weight_decay=0.01)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seed_prompts = [
        "Explique como funciona a gravidade de forma simples.",
        "Como programar uma função recursiva em Python?",
        "Qual a capital do Brasil e conte um resumo histórico dela.",
        "Estou me sentindo um pouco cansado hoje, o que você sugere?",
        "O que é aprendizado por reforço com feedback humano?",
        "Me dê uma receita rápida de café da manhã brasileiro.",
        "Como manter a motivação para estudar todos os dias?",
    ]

    print(f"=== Iniciando Loop de Auto-Melhoria Keilinks V5 ({iterations} iterações) ===")

    for it in range(1, iterations + 1):
        print(f"\n--- Iteração {it}/{iterations} ---")
        for step in range(1, steps_per_iter + 1):
            batch = random.sample(seed_prompts, min(3, len(seed_prompts)))
            loss = engine.train_step(optimizer, batch, group_size=4)
            if step % 10 == 0:
                print(f"[Iter {it} | Step {step:>3}/{steps_per_iter}] GRPO Loss: {loss:.4f}")

        # Salva checkpoint intermediário
        save_file = out_path / f"keilinks_v5_iter_{it}.pt"
        torch.save(
            {
                "version": 5,
                "iteration": it,
                "model": engine.model.state_dict(),
                "config": engine.config.to_dict(),
            },
            save_file,
        )
        print(f"[Checkpoint] Salvo em: {save_file}")

    print("\n=== Auto-Melhoria concluída com sucesso! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Melhoria Contínua Keilinks V5")
    parser.add_argument("--checkpoint", default="checkpoints/v4-sft/keilinks_v4.pt")
    parser.add_argument("--vocab", default="dados/vocab_v4.json")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--i-understand-this-is-experimental",
        action="store_true",
        help="Obrigatório: este protótipo não substitui revisão humana, DPO auditado e gate de promoção.",
    )
    args = parser.parse_args()

    if not args.i_understand_this_is_experimental:
        raise SystemExit(
            "Bloqueado: não rode GRPO autoavaliado em produção. Use o feedback consentido, "
            "revisão humana e o pipeline V5 de preferências primeiro. Para pesquisa isolada, "
            "repita com --i-understand-this-is-experimental."
        )
    if not Path(args.checkpoint).exists() or not Path(args.vocab).exists():
        raise SystemExit(f"Checkpoint {args.checkpoint} ou vocab {args.vocab} não encontrado.")
    run_continuous_improvement_loop(
        args.checkpoint, args.vocab, iterations=args.iterations, steps_per_iter=args.steps
    )
