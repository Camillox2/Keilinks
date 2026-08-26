from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from treino.v4.preparar_sft_conversacional import (
    DEFAULT_TOTAL,
    _default_paths,
    _record_from_redial_ptbr,
    allocate_targets,
    assistant_style_violation,
    build_conversation_mix,
    generate_neutral_anchors,
)


def _alphabetic_index(index: int) -> str:
    """Cria uma chave textual que o normalizador não reduz a <n>."""

    letters = "abcdefghijklmnopqrstuvwxyz"
    value = index
    parts: list[str] = []
    while True:
        value, remainder = divmod(value, len(letters))
        parts.append(letters[remainder])
        if value == 0:
            return "".join(reversed(parts))


def _record(source: str, index: int, *, synthetic: bool = False) -> dict[str, object]:
    tag = _alphabetic_index(index)
    return {
        "id": f"{source}-{index}",
        "group_id": f"{source}-group-{index}",
        "source": source,
        "synthetic": synthetic,
        "messages": [
            {"role": "user", "content": f"Pergunta de teste {source} {tag}."},
            {
                "role": "assistant",
                "content": f"Resposta neutra e verificável para o exemplo {tag}.",
            },
        ],
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class TestPrepararSftConversacional(unittest.TestCase):
    def test_default_targets_are_exact_and_conversation_first(self) -> None:
        targets = allocate_targets(DEFAULT_TOTAL)
        self.assertEqual(sum(targets.values()), DEFAULT_TOTAL)
        self.assertEqual(
            targets,
            {
                "human_dialogue": 1500,
                "translated_dialogue": 650,
                "human_instruction": 1600,
                "synthetic_instruction": 700,
                "short_reasoning": 350,
                "behavior_anchor": 200,
            },
        )
        self.assertEqual(
            targets["human_dialogue"] + targets["translated_dialogue"],
            2150,
        )

    def test_default_path_keys_match_builder_signature(self) -> None:
        self.assertEqual(
            set(_default_paths()),
            {
                "human_dialogue_paths",
                "translated_dialogue_paths",
                "human_instruction_paths",
                "synthetic_instruction_paths",
                "reasoning_paths",
                "anchor_paths",
            },
        )

    def test_style_gate_applies_to_assistant_not_user(self) -> None:
        user_slang = [
            {"role": "user", "content": "Bora resolver isso, vc consegue?"},
            {"role": "assistant", "content": "Sim. Vou explicar os próximos passos."},
        ]
        assistant_slang = [
            {"role": "user", "content": "Pode me ajudar?"},
            {"role": "assistant", "content": "Bora, mano. Vamos resolver."},
        ]
        self.assertIsNone(assistant_style_violation(user_slang))
        self.assertEqual(assistant_style_violation(assistant_slang), "assistant_slang")

    def test_authored_anchors_are_labeled_and_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "anchors.jsonl"
            report = generate_neutral_anchors(output, count=4)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["records"], 4)
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertTrue(record["synthetic"])
            self.assertEqual(record["source"], "keilinks_authored_neutral_v1")
            self.assertIsNone(assistant_style_violation(record["messages"]))

    def test_redial_translation_keeps_provenance_and_hides_internal_movie_ids(self) -> None:
        long_user = (
            "Quero uma recomendação de filme para assistir hoje "
            "com uma história envolvente. "
        )
        long_assistant = (
            "Posso sugerir uma opção e explicar por que ela combina "
            "com o que você descreveu. "
        )
        row = {
            "conversationId": 42,
            "initiatorWorkerId": 10,
            "respondentWorkerId": 11,
            "movieMentions": [{"movieId": "100", "movieName": "Filme de Exemplo (2020)"}],
            "messages_translated": [
                {"senderWorkerId": 10, "text": f"{long_user}Você conhece @100?"},
                {"senderWorkerId": 11, "text": f"{long_assistant}Sim, ele é uma boa escolha."},
                {"senderWorkerId": 10, "text": f"{long_user}Prefiro @Outro Filme sem terror."},
                {"senderWorkerId": 11, "text": f"{long_assistant}Nesse caso, priorize o drama."},
                {"senderWorkerId": 10, "text": f"{long_user}Obrigado pela explicação."},
                {"senderWorkerId": 11, "text": f"{long_assistant}Se quiser, comparo alternativas."},
            ],
        }

        record = _record_from_redial_ptbr(row, "revision")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertTrue(record["synthetic"])
        self.assertTrue(record["translated"])
        self.assertTrue(record["original_human_dialogue"])
        self.assertEqual(record["source"], "redial_ptbr_filtered_v2")
        self.assertEqual(record["messages"][0]["role"], "user")
        self.assertEqual(record["messages"][-1]["role"], "assistant")
        all_text = "\n".join(message["content"] for message in record["messages"])
        self.assertNotIn("@", all_text)
        self.assertIn("Filme de Exemplo", all_text)

    def test_mix_keeps_declared_buckets_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            human_dialogue = root / "human_dialogue.jsonl"
            translated_dialogue = root / "translated_dialogue.jsonl"
            human_instruction = root / "human_instruction.jsonl"
            synthetic_instruction = root / "synthetic_instruction.jsonl"
            reasoning = root / "reasoning.jsonl"
            anchors = root / "anchors.jsonl"
            output = root / "mixed.jsonl"

            _write_jsonl(
                human_dialogue,
                [_record("oasst", index) for index in range(30)],
            )
            _write_jsonl(
                translated_dialogue,
                [_record("redial_ptbr_filtered_v1", index, synthetic=True) for index in range(13)],
            )
            _write_jsonl(
                human_instruction,
                [_record("aya", index) for index in range(32)],
            )
            _write_jsonl(
                synthetic_instruction,
                [_record("tucano_sft", index, synthetic=True) for index in range(14)],
            )
            _write_jsonl(
                reasoning,
                [
                    _record("keilinks_reasoning_short_v1", index, synthetic=True)
                    for index in range(7)
                ],
            )
            _write_jsonl(
                anchors,
                [
                    _record("keilinks_curated_v4", index, synthetic=False)
                    for index in range(8)
                ],
            )

            report = build_conversation_mix(
                output,
                total=100,
                human_dialogue_paths=[human_dialogue],
                translated_dialogue_paths=[translated_dialogue],
                human_instruction_paths=[human_instruction],
                synthetic_instruction_paths=[synthetic_instruction],
                reasoning_paths=[reasoning],
                anchor_paths=[anchors],
            )
            records = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]

        by_bucket = Counter(record["mix_bucket"] for record in records)
        self.assertEqual(report["total"], 100)
        self.assertEqual(
            dict(by_bucket),
            {
                "human_dialogue": 30,
                "translated_dialogue": 13,
                "human_instruction": 32,
                "synthetic_instruction": 14,
                "short_reasoning": 7,
                "behavior_anchor": 4,
            },
        )
        self.assertEqual(report["synthetic_total"], 34)
        self.assertEqual(report["dialogue_total"], 43)
        self.assertEqual(len({record["id"] for record in records}), 100)


if __name__ == "__main__":
    unittest.main()
