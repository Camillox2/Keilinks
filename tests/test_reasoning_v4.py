from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from api.runtime_v4 import V4Runtime
from cerebro.raciocinio import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    PLAN_CLOSE,
    PLAN_OPEN,
    format_reasoning_target,
    normalize_reasoning_mode,
    parse_reasoning_output,
    requires_reasoning,
)
from treino.v4.avaliar_raciocinio import score_case
from treino.v4.config import get_train_config
from treino.v4.dataset import canonical_conversation
from treino.v4.preparar_raciocinio import build_examples
from treino.v4.treinar import steps_for_epochs


class TestReasoningV4(unittest.TestCase):
    def test_runtime_adds_protocol_only_when_requested(self) -> None:
        class ByteTokenizer:
            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

            def decode(self, ids: list[int]) -> str:
                return bytes(ids).decode("utf-8", errors="ignore")

        runtime = object.__new__(V4Runtime)
        runtime.config = SimpleNamespace(context_length=8192)
        runtime.tokenizer = ByteTokenizer()
        runtime.system_prompt = "Sistema de teste."
        regular = runtime.build_prompt("Compare A e B", reasoning=False)
        careful = runtime.build_prompt("Compare A e B", reasoning=True)
        regular_text = bytes(regular).decode("utf-8", errors="ignore")
        careful_text = bytes(careful).decode("utf-8", errors="ignore")
        self.assertNotIn("MODO DE RACIOCÍNIO INTERNO", regular_text)
        self.assertIn("MODO DE RACIOCÍNIO INTERNO", careful_text)

    def test_protocol_keeps_only_final_text(self) -> None:
        target = format_reasoning_target(
            "Somar os valores e conferir o resultado.",
            "A soma é 42.",
        )
        parsed = parse_reasoning_output(target)
        self.assertTrue(parsed.has_complete_protocol)
        self.assertEqual(parsed.plan, "Somar os valores e conferir o resultado.")
        self.assertEqual(parsed.final, "A soma é 42.")
        self.assertNotIn(PLAN_OPEN, parsed.final)
        self.assertNotIn(ANSWER_CLOSE, parsed.final)

    def test_complete_protocol_exposes_a_short_summary_only(self) -> None:
        parsed = parse_reasoning_output(
            format_reasoning_target(
                "Conferir a entrada, calcular e validar o resultado.",
                "O valor final é 42.",
            )
        )
        self.assertTrue(parsed.has_complete_protocol)
        self.assertEqual(parsed.plan, "Conferir a entrada, calcular e validar o resultado.")
        self.assertNotIn("[[", parsed.plan)

    def test_incomplete_protocol_never_leaks_complete_plan(self) -> None:
        raw = f"{PLAN_OPEN}Conferir números.{PLAN_CLOSE} Resposta curta"
        parsed = parse_reasoning_output(raw)
        self.assertEqual(parsed.final, "Resposta curta")
        self.assertFalse(parsed.has_complete_protocol)
        self.assertNotIn(PLAN_CLOSE, parsed.final)
        self.assertNotIn(ANSWER_OPEN, parsed.final)

    def test_unclosed_plan_is_withheld(self) -> None:
        parsed = parse_reasoning_output(f"{PLAN_OPEN}Detalhes internos ainda em andamento")
        self.assertEqual(parsed.final, "")
        self.assertFalse(parsed.has_complete_protocol)

    def test_router_is_conservative_and_forcing_is_explicit(self) -> None:
        self.assertFalse(requires_reasoning("Oi, tudo bem?"))
        self.assertTrue(requires_reasoning("Compare as opções e escolha a melhor."))
        self.assertTrue(requires_reasoning("Calcule 18% de 430 e confira."))
        self.assertEqual(normalize_reasoning_mode("always"), "always")
        self.assertEqual(normalize_reasoning_mode("qualquer-texto"), "auto")

    def test_generated_curriculum_is_balanced_and_valid(self) -> None:
        examples = build_examples(seed=42)
        self.assertEqual(len(examples), 640)
        self.assertEqual(len({example["id"] for example in examples}), 640)
        self.assertEqual(
            len({canonical_conversation(example["messages"]) for example in examples}),
            640,
        )
        categories = {example["category"] for example in examples}
        self.assertEqual(
            categories,
            {
                "arithmetic",
                "percentage",
                "logical_conditions",
                "comparison",
                "research_and_uncertainty",
                "debugging",
            },
        )
        for example in examples[:20]:
            parsed = parse_reasoning_output(example["messages"][-1]["content"])
            self.assertTrue(parsed.has_complete_protocol)
            self.assertTrue(parsed.final)

    def test_8k_sft_profile_matches_operational_context(self) -> None:
        profile = get_train_config("rtx5050_sft_380m")
        self.assertEqual(profile.grad_accum_steps, 4)
        self.assertEqual(profile.checkpoint_mode, "full")
        self.assertEqual(profile.checkpoint_every, 1)
        self.assertEqual(steps_for_epochs(219, profile, 3), 165)

    def test_holdout_score_uses_final_answer_only(self) -> None:
        text = (
            f"{PLAN_OPEN}Fazer uma conta interna.{PLAN_CLOSE}"
            f"{ANSWER_OPEN}O resultado correto é 803.{ANSWER_CLOSE}"
        )
        result = score_case(text, {"id": "case", "required_all": ["803"]})
        self.assertTrue(result["passed"])
        self.assertEqual(result["text"], "O resultado correto é 803.")

    def test_frontend_sends_reasoning_mode(self) -> None:
        frontend = Path("interface/index.html").read_text(encoding="utf-8")
        self.assertIn('id="btn-raciocinio"', frontend)
        self.assertIn("reasoning_mode: reasoningMode", frontend)
        server = Path("api/servidor_v4.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(server.count("reasoning_mode=_reasoning_mode(payload)"), 2)
        self.assertIn('"raciocinio": answer.reasoning_mode', server)
        self.assertIn('"resumo_raciocinio": plan or None', server)
        self.assertIn("show_reasoning: showReasoning", frontend)
        self.assertIn("anexarResumoRaciocinio", frontend)


if __name__ == "__main__":
    unittest.main()
