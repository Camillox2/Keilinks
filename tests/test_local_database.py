from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dados import database


class TestLocalDatabase(unittest.TestCase):
    def test_sqlite_bootstrap_preserves_chat_and_knowledge_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "keilinks.sqlite3"
            with patch.dict(
                os.environ,
                {
                    "KEILINKS_DB_PATH": str(path),
                    "KEILINKS_AUTH_SECRET": "s" * 48,
                },
                clear=False,
            ):
                database.inicializar_banco()
                self.assertTrue(path.exists())
                self.assertTrue(
                    database.knowledge_adicionar(
                        "Como pesquisar fontes confiáveis?",
                        "Compare fontes primárias e cite a data de publicação.",
                        fonte="teste",
                    )
                )
                self.assertFalse(
                    database.knowledge_adicionar(
                        "Como pesquisar fontes confiáveis?",
                        "Resposta duplicada.",
                    )
                )
                found = database.knowledge_buscar("fontes confiáveis", limite=3)
                self.assertEqual(len(found), 1)
                self.assertIn("Compare", found[0]["resposta"])

                user = database.usuario_criar("vitor.teste", "senha-segura", "Vitor")
                self.assertIsNotNone(user)
                assert user is not None
                self.assertEqual(database.usuario_por_token(user["token"])["id"], user["id"])
                self.assertIsNone(database.usuario_login("vitor.teste", "errada"))
                self.assertEqual(
                    database.usuario_login("vitor.teste", "senha-segura")["id"], user["id"]
                )

                chat = database.chat_criar(user["id"], "Teste de contexto")
                database.conversa_salvar(
                    "Oi", "Oi! Como posso ajudar?", "modelo_v4", chat["id"], user["id"]
                )
                messages = database.chat_mensagens(chat["id"], user["id"])
                self.assertEqual(messages[0]["resposta"], "Oi! Como posso ajudar?")
                database.memoria_set("preferencia", "português")
                self.assertEqual(database.memoria_get("preferencia"), "português")
                self.assertTrue(database.chat_deletar(chat["id"], user["id"]))


if __name__ == "__main__":
    unittest.main()
