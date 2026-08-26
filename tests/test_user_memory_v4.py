from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api import servidor_v4
from dados import database
from treino.v4.exportar_feedback_consentido import export_approved_feedback


class TestUserMemoryV4(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(
            os.environ,
            {
                "KEILINKS_DB_PATH": str(self.root / "keilinks.sqlite3"),
                "KEILINKS_AUTH_SECRET": "m" * 48,
            },
            clear=False,
        )
        self.environment.start()
        database._INITIALIZED_PATHS.clear()
        database.inicializar_banco()
        self.client = servidor_v4.app.test_client()

    def tearDown(self) -> None:
        database._INITIALIZED_PATHS.clear()
        self.environment.stop()
        self.temporary.cleanup()

    def _register(self, username: str, name: str) -> dict:
        response = self.client.post(
            "/api/registrar",
            json={"username": username, "senha": "senha-segura", "nome": name},
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    @staticmethod
    def _headers(user: dict) -> dict[str, str]:
        return {"Authorization": f"Bearer {user['token']}"}

    def test_memory_is_explicit_user_scoped_and_relevant(self) -> None:
        alice = self._register("alice.mem", "Alice")
        bruno = self._register("bruno.mem", "Bruno")

        initial = self.client.get("/api/me/memory", headers=self._headers(alice))
        self.assertEqual(initial.status_code, 200)
        self.assertFalse(initial.get_json()["settings"]["memory_enabled"])

        enabled = self.client.put(
            "/api/me/memory/settings",
            headers=self._headers(alice),
            json={"memory_enabled": True, "history_enabled": True, "training_consent": False},
        )
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.get_json()["settings"]["memory_enabled"])

        created = self.client.post(
            "/api/me/memories",
            headers=self._headers(alice),
            json={"category": "preference", "content": "Prefiro respostas objetivas."},
        )
        self.assertEqual(created.status_code, 201)
        memory_id = created.get_json()["memory"]["id"]

        context = database.memoria_usuario_contexto(alice["id"], "Quero uma resposta objetiva")
        self.assertIn("Alice", context)
        self.assertIn("Prefiro respostas objetivas", context)
        self.assertNotIn("Bruno", context)

        foreign_read = self.client.get("/api/me/memory", headers=self._headers(bruno))
        self.assertEqual(foreign_read.status_code, 200)
        self.assertEqual(foreign_read.get_json()["memories"], [])

        foreign_delete = self.client.delete(
            f"/api/me/memories/{memory_id}", headers=self._headers(bruno)
        )
        self.assertEqual(foreign_delete.status_code, 404)
        self.assertEqual(len(database.memorias_usuario_listar(alice["id"])), 1)

    def test_unrelated_notes_do_not_get_injected_into_the_prompt(self) -> None:
        user = self._register("dora.mem", "Dora")
        self.client.put(
            "/api/me/memory/settings",
            headers=self._headers(user),
            json={"memory_enabled": True},
        )
        self.client.post(
            "/api/me/memories",
            headers=self._headers(user),
            json={"category": "note", "content": "Minha cidade natal é Florianópolis."},
        )

        context = database.memoria_usuario_contexto(user["id"], "Olá, como você está?")
        self.assertIn("Dora", context)
        self.assertNotIn("Florianópolis", context)

    def test_sensitive_memory_and_legacy_global_views_are_protected(self) -> None:
        user = self._register("carla.mem", "Carla")
        rejected = self.client.post(
            "/api/me/memories",
            headers=self._headers(user),
            json={"category": "note", "content": "Minha senha é supersecreta"},
        )
        self.assertEqual(rejected.status_code, 400)

        self.assertEqual(self.client.get("/api/historico").status_code, 401)
        protected = self.client.get("/api/memoria", headers=self._headers(user))
        self.assertEqual(protected.status_code, 200)
        self.assertIn("profile", protected.get_json())

    def test_feedback_requires_consent_then_waits_for_review(self) -> None:
        user = self._register("feedback.mem", "Feedback")
        payload = {
            "prompt": "Meu email é pessoa@example.com; explique como funciona uma API.",
            "response": (
                "Uma API permite que dois sistemas troquem dados por uma interface definida."
            ),
            "rating": "up",
        }
        blocked = self.client.post(
            "/api/me/training-feedback", headers=self._headers(user), json=payload
        )
        self.assertEqual(blocked.status_code, 409)
        self.assertTrue(blocked.get_json()["requires_training_consent"])

        enabled = self.client.put(
            "/api/me/memory/settings",
            headers=self._headers(user),
            json={"training_consent": True},
        )
        self.assertEqual(enabled.status_code, 200)
        queued = self.client.post(
            "/api/me/training-feedback", headers=self._headers(user), json=payload
        )
        self.assertEqual(queued.status_code, 201)
        feedback = queued.get_json()["feedback"]
        self.assertEqual(feedback["status"], "pending_human_review")
        self.assertTrue(feedback["redacted"])
        self.assertNotIn("pessoa@example.com", feedback["prompt"])
        self.assertEqual(queued.get_json()["summary"]["pending_human_review"], 1)

        reviewed = database.feedback_treino_revisar(feedback["id"], approved=True)
        self.assertIsNotNone(reviewed)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export_approved_feedback(Path(temporary))
            self.assertEqual(manifest["approved_sft_examples"], 1)
            self.assertEqual(manifest["approved_preference_pairs"], 0)

        self.client.put(
            "/api/me/memory/settings",
            headers=self._headers(user),
            json={"training_consent": False},
        )
        self.assertEqual(database.feedback_treino_aprovados(), [])

    def test_negative_feedback_with_correction_builds_preference_after_review(self) -> None:
        user = self._register("correction.mem", "Correção")
        self.client.put(
            "/api/me/memory/settings",
            headers=self._headers(user),
            json={"training_consent": True},
        )
        queued = self.client.post(
            "/api/me/training-feedback",
            headers=self._headers(user),
            json={
                "prompt": "Qual é a capital do Brasil?",
                "response": "Rio de Janeiro.",
                "rating": "down",
                "correction": "A capital do Brasil é Brasília.",
            },
        )
        self.assertEqual(queued.status_code, 201)
        feedback = queued.get_json()["feedback"]
        database.feedback_treino_revisar(feedback["id"], approved=True)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export_approved_feedback(Path(temporary))
            self.assertEqual(manifest["approved_sft_examples"], 1)
            self.assertEqual(manifest["approved_preference_pairs"], 1)

    def test_frontend_exposes_explicit_memory_controls(self) -> None:
        interface = (Path(__file__).resolve().parents[1] / "interface" / "index.html")
        source = interface.read_text(encoding="utf-8")
        self.assertIn("Memória e privacidade", source)
        self.assertIn("/api/me/memory/settings", source)
        self.assertIn("/api/me/memories", source)
        self.assertIn("/api/me/training-feedback", source)
        self.assertIn("Ativar contribuição", source)
        self.assertIn("automaticamente", source)


if __name__ == "__main__":
    unittest.main()
