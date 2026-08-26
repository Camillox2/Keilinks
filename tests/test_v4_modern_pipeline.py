from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from busca.web_v4 import (
    SearchResult,
    _is_safe_public_url,
    _is_trusted_domain,
    deve_pesquisar,
    exige_fontes_atualizadas,
    tem_evidencia_suficiente,
)
from treino.v4.config import get_model_config
from treino.v4.montar_corpus import assemble
from treino.v4.tokenizador import REQUIRED_SPECIALS, TokenizadorV4, build_vocab


class TestModernV4Pipeline(unittest.TestCase):
    def test_operational_380m_keeps_sdpa_path(self) -> None:
        config = get_model_config("core_380m_modern")
        self.assertTrue(config.use_qk_norm)
        self.assertEqual(config.rope_theta, 500_000.0)
        self.assertEqual(config.attn_logit_softcapping, 0.0)
        self.assertEqual(config.final_logit_softcapping, 0.0)

    def test_conservative_web_router(self) -> None:
        self.assertFalse(deve_pesquisar("Oi, tudo bem?"))
        self.assertFalse(deve_pesquisar("Escreva uma historia curta sobre um robo."))
        self.assertTrue(deve_pesquisar("Quem e o presidente atual do Brasil?"))
        self.assertTrue(deve_pesquisar("Explique o que e computacao quantica."))
        self.assertFalse(deve_pesquisar("Qual e a capital do Brasil?", "never"))
        self.assertTrue(deve_pesquisar("Oi", "always"))

    def test_current_questions_need_independent_or_trusted_evidence(self) -> None:
        self.assertTrue(exige_fontes_atualizadas("Qual e o preco atual do dolar?"))
        self.assertFalse(exige_fontes_atualizadas("Explique computacao quantica."))
        wikipedia_only = [
            SearchResult(
                "Brasil",
                "https://pt.wikipedia.org/wiki/Brasil",
                "Texto enciclopedico suficientemente longo para o teste.",
                "wikipedia",
            )
        ]
        self.assertTrue(tem_evidencia_suficiente(wikipedia_only))
        self.assertFalse(
            tem_evidencia_suficiente(wikipedia_only, exige_atualidade=True)
        )
        independent = [
            SearchResult(
                "Banco Central",
                "https://www.bcb.gov.br/exemplo",
                "Texto oficial suficientemente longo para validar a fonte.",
                "duckduckgo",
            )
        ]
        self.assertTrue(
            tem_evidencia_suficiente(independent, exige_atualidade=True)
        )
        self.assertTrue(_is_trusted_domain("www.bcb.gov.br"))
        self.assertFalse(_is_trusted_domain("naowikipedia.org"))
        self.assertFalse(_is_safe_public_url("http://127.0.0.1/private"))
        self.assertFalse(_is_safe_public_url("http://localhost/private"))

    def test_hf_tokenizer_preserves_v4_specials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = root / "pretrain.txt"
            pretrain.write_text(
                ("Keilinks conversa em portugues brasileiro com clareza e empatia.\n\n") * 2_000,
                encoding="utf-8",
            )
            sft = root / "sft.jsonl"
            sft.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Ola"},
                            {"role": "assistant", "content": "Ola, como posso ajudar?"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "tokenizer.json"
            build_vocab(output, pretrain, sft, 512, 1, 42)
            tokenizer = TokenizadorV4(output)
            self.assertEqual(tokenizer.backend, "hf_bytelevel_bpe")
            self.assertTrue(all(token in tokenizer.vocab for token in REQUIRED_SPECIALS))
            self.assertGreater(len(tokenizer.encode("<vitor>Ola<fim>")), 2)

    def test_assemble_checks_hash_and_interleaves_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fineweb = root / "raw" / "run" / "fineweb2_pt"
            wiki = root / "raw" / "run" / "wikipedia_pt"
            fineweb.mkdir(parents=True)
            wiki.mkdir(parents=True)
            files = [fineweb / "shard-00000.jsonl", wiki / "shard-00000.jsonl"]
            handles = [path.open("w", encoding="utf-8") for path in files]
            try:
                for index in range(50):
                    for source, handle in zip(
                        ("fineweb2_pt", "wikipedia_pt"), handles, strict=True
                    ):
                        text = (
                            f"Documento {source} numero {index}. "
                            "Este texto possui vocabulario diverso para validar a montagem "
                            "do corpus. Ele contem informacoes suficientes para superar o "
                            "limite minimo de caracteres."
                        )
                        handle.write(
                            json.dumps(
                                {
                                    "text": text,
                                    "source_key": source,
                                    "content_sha256": hashlib.sha256(
                                        text.encode("utf-8")
                                    ).hexdigest(),
                                }
                            )
                            + "\n"
                        )
            finally:
                for handle in handles:
                    handle.close()
            output = root / "pretrain_pt.txt"
            manifest = assemble([str(root / "raw" / "run")], output, max_bytes=None)
            self.assertEqual(manifest["documents_written"], 100)
            self.assertEqual(set(manifest["sources"]), {"fineweb2_pt", "wikipedia_pt"})
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
