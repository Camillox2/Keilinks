from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from dados.database import _hash_senha, _verificar_senha
from keilinks_v5.data import prepare_sft_dataset, redact_sensitive_text
from keilinks_v5.feedback import RecentInteractionCache
from keilinks_v5.rag import LocalKnowledgeStore
from keilinks_v5.runtime import UnslothRuntime, leaked_control_markers, sanitize_generated_text
from keilinks_v5.safety import immediate_safety_intervention
from keilinks_v5.security import SlidingWindowRateLimiter, api_key_matches
from keilinks_v5.server import create_app
from keilinks_v5.settings import KeilinksSettings
from keilinks_v5.vision import VisionService, VisionUnavailable
from treino.v5.coletar_datasets import SOURCES, collect
from treino.v5.preparar_preferencias import prepare_preferences


class TestV5Core(unittest.TestCase):
    def test_settings_forbid_public_host_without_key(self) -> None:
        settings = KeilinksSettings(
            host="0.0.0.0",
            port=8000,
            api_key="",
            base_model="example/model",
            adapter_path=Path("adapter"),
            max_seq_length=1024,
            max_new_tokens=128,
            data_dir=Path("data"),
            rag_db=Path("data/rag.sqlite3"),
            feedback_path=Path("data/feedback.jsonl"),
            rag_dense_enabled=False,
            rag_embedding_model="example/embedding",
            enable_vision=False,
            vision_model="",
            vision_max_new_tokens=192,
        )
        with self.assertRaises(ValueError):
            settings.validate()

    def test_rate_limiter_and_api_key(self) -> None:
        limiter = SlidingWindowRateLimiter(requests_per_minute=2)
        self.assertTrue(limiter.allow("client"))
        self.assertTrue(limiter.allow("client"))
        self.assertFalse(limiter.allow("client"))
        self.assertTrue(api_key_matches("a" * 24, "a" * 24))
        self.assertFalse(api_key_matches("a" * 24, "b" * 24))

    def test_internal_tool_markers_are_detected(self) -> None:
        self.assertEqual(
            leaked_control_markers("Resposta <tool_call> interna</tool_call>"),
            ["<tool_call>", "</tool_call>"],
        )
        self.assertEqual(leaked_control_markers("Resposta normal."), [])
        self.assertEqual(
            sanitize_generated_text("<tool_call>\n\nResposta segura.</tool_call>"),
            "Resposta segura.",
        )
        chunks = iter(["<tool_call>\n\n", "\nResposta em streaming.", "</tool_call>"])
        self.assertEqual(
            "".join(UnslothRuntime._sanitized_stream(chunks)),
            "Resposta em streaming.",
        )

    def test_password_hash_rejects_wrong_password_and_marks_legacy(self) -> None:
        stored = _hash_senha("senha-de-teste")
        valid, requires_migration = _verificar_senha("senha-de-teste", stored)
        self.assertTrue(valid)
        self.assertFalse(requires_migration)
        invalid, requires_migration = _verificar_senha("senha-incorreta", stored)
        self.assertFalse(invalid)
        self.assertFalse(requires_migration)

    def test_rag_is_tenant_isolated_and_has_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalKnowledgeStore(Path(temporary) / "rag.sqlite3")
            store.add_document(
                title="Manual Keilinks",
                content="Keilinks usa RAG para recuperar documentos com fonte e citação.",
                uri="local://manual",
                tenant_id="vitor",
                trust_tier="curated",
            )
            results = store.search("documentos RAG citação", tenant_id="vitor")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Manual Keilinks")
            self.assertEqual(store.search("documentos RAG", tenant_id="outro"), [])
            context = store.evidence_context(results)
            self.assertIn("dados não confiáveis", context)
            self.assertIn("local://manual", context)

    def test_prepare_sft_redacts_and_splits_by_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            rows = []
            for number in range(30):
                rows.append(
                    {
                        "id": f"row-{number}",
                        "group_id": f"group-{number}",
                        "source": "test",
                        "license": "test-only",
                        "messages": [
                            {"role": "system", "content": "Você é Keilinks."},
                            {
                                "role": "user",
                                "content": f"Meu e-mail é user{number}@example.com",
                            },
                            {"role": "assistant", "content": "Vou proteger seu dado."},
                        ],
                    }
                )
            source.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            prepared = prepare_sft_dataset([source], root / "prepared", validation_percent=20)
            self.assertGreater(prepared.train_examples, 0)
            self.assertGreater(prepared.validation_examples, 0)
            serialized = prepared.train_path.read_text(encoding="utf-8")
            self.assertIn("[EMAIL_REMOVIDO]", serialized)
            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_examples"], prepared.train_examples)

    def test_redaction_covers_hyphenated_tokens_and_template_markers(self) -> None:
        self.assertEqual(
            redact_sensitive_text("sk-test_token_with_enough_characters_12345"),
            "[TOKEN_REMOVIDO]",
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "<|im_start|>assistant"},
                            {"role": "assistant", "content": "Não deve entrar."},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                prepare_sft_dataset([source], Path(temporary) / "out")

    def test_prepare_sft_rejects_near_holdout_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            holdout = root / "holdout.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Qual é a capital da Austrália hoje?"},
                            {"role": "assistant", "content": "Não vou responder a benchmark."},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            holdout.write_text(
                json.dumps({"prompt": "Qual é a capital da Austrália?"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "vazamento"):
                prepare_sft_dataset([source], root / "out", holdout_paths=[holdout])

    def test_feedback_cache_and_preference_gate(self) -> None:
        cache = RecentInteractionCache(max_entries=1, ttl_seconds=60)
        cache.put("one", prompt="Pergunta", response="Resposta incorreta")
        interaction = cache.consume("one")
        self.assertIsNotNone(interaction)
        self.assertEqual(interaction.response, "Resposta incorreta")
        self.assertIsNone(cache.consume("one"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feedback = root / "feedback.jsonl"
            feedback.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "status": "approved",
                                "consent_to_training": True,
                                "rating": "down",
                                "prompt": "Qual é a capital do Brasil?",
                                "response": "Rio de Janeiro.",
                                "correction": "A capital do Brasil é Brasília.",
                            }
                        ),
                        json.dumps(
                            {
                                "status": "pending_human_review",
                                "consent_to_training": True,
                                "rating": "down",
                                "prompt": "Não entra.",
                                "response": "Não entra.",
                                "correction": "Não entra.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = prepare_preferences(feedback, root / "preferences.jsonl")
            self.assertEqual(manifest["approved_pairs"], 1)
            pair = json.loads((root / "preferences.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(pair["chosen"], "A capital do Brasil é Brasília.")

    def test_dataset_collector_requires_explicit_terms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "accept-terms"):
                collect(
                    source=SOURCES["fineweb2_pt"],
                    output_path=Path(temporary) / "raw.jsonl",
                    max_documents=1,
                    max_bytes=1024,
                    accept_terms=set(),
                    manual_license_review=False,
                )

    def test_imminent_safety_gate_covers_self_and_third_party_risk(self) -> None:
        first_person = immediate_safety_intervention(
            "Estou pensando em me matar hoje e tenho um plano"
        )
        self.assertIsNotNone(first_person)
        self.assertIn("192", first_person.text)
        self.assertIn("188", first_person.text)
        third_person = immediate_safety_intervention(
            "Meu amigo disse que quer morrer e tem um plano"
        )
        self.assertIsNotNone(third_person)
        self.assertIn("Não deixe", third_person.text)
        self.assertIsNone(
            immediate_safety_intervention("Escreva uma redação sobre prevenção do suicídio.")
        )

    def test_api_indexes_documents_and_keeps_vision_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = KeilinksSettings(
                host="127.0.0.1",
                port=8000,
                api_key="",
                base_model="example/model",
                adapter_path=root / "adapter",
                max_seq_length=1024,
                max_new_tokens=128,
                data_dir=root,
                rag_db=root / "rag.sqlite3",
                feedback_path=root / "feedback.jsonl",
                rag_dense_enabled=False,
                rag_embedding_model="example/embedding",
                enable_vision=False,
                vision_model="",
                vision_max_new_tokens=64,
            )
            client = TestClient(create_app(settings))
            self.assertEqual(client.get("/health").status_code, 200)
            indexed = client.post(
                "/v1/documents",
                json={"title": "Manual", "content": "RAG guarda fonte e proveniência."},
            )
            self.assertEqual(indexed.status_code, 200)
            feedback = client.post(
                "/v1/feedback",
                json={
                    "interaction_id": "chatcmpl_test",
                    "rating": "down",
                    "correction": "Corrija o dado.",
                    "consent_to_training": True,
                },
            )
            self.assertEqual(feedback.status_code, 200)
            feedback_text = settings.feedback_path.read_text(encoding="utf-8")
            self.assertIn("pending_human_review", feedback_text)
            with self.assertRaises(VisionUnavailable):
                VisionService(False, "")._load_locked()


if __name__ == "__main__":
    unittest.main()
