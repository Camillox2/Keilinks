"""Testes automatizados dos módulos atualizados da Keilinks V5.

Testa:
1. Instanciação do modelo Transformer com QK-Norm e Logit Soft-Capping.
2. Forward e backward pass com cálculo de loss.
3. Otimizador Muon e HybridOptimizer.
4. Motor de RAG Híbrido (BM25 + Dense + RRF).
5. Verificador determinístico de qualidade em PT-BR.
"""
from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from treino.v4.config import ModelConfig
from treino.v4.modelo import KeilinksV4
from treino.v4.muon import Muon, build_muon_hybrid_optimizer
from cerebro.hybrid_rag import BM25Retriever, HybridRAG
from treino.v4.self_improver import RuleBasedVerifier


class TestKeilinksV5(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            name="Keilinks Test 50M",
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            ff_dim=256,
            context_length=128,
            rope_theta=500_000.0,
            use_qk_norm=True,
            attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0,
        )

    def test_model_forward_backward(self):
        """Valida se o modelo roda forward e propaga gradientes com QK-Norm e Soft-capping."""
        model = KeilinksV4(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, 32))
        labels = torch.randint(0, self.config.vocab_size, (2, 32))

        logits, loss = model(input_ids, labels)
        self.assertEqual(logits.shape, (2, 32, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        # Verifica se parâmetros têm gradientes válidos
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradiente nulo em {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradiente infinito em {name}")

    def test_muon_hybrid_optimizer(self):
        """Valida se o otimizador Muon e o construtor híbrido executam o passo de otimização."""
        model = KeilinksV4(self.config)
        hybrid_opt = build_muon_hybrid_optimizer(model, lr_muon=0.02, lr_adam=1e-3, device_type="cpu")

        input_ids = torch.randint(0, self.config.vocab_size, (2, 16))
        labels = torch.randint(0, self.config.vocab_size, (2, 16))

        logits, loss = model(input_ids, labels)
        loss.backward()

        hybrid_opt.step()
        hybrid_opt.zero_grad()
        self.assertTrue(True)

    def test_hybrid_rag(self):
        """Valida indexação BM25 e busca híbrida."""
        rag = HybridRAG(cache_path=None)
        docs = [
            "A capital do Brasil é Brasília, inaugurada em 1960 por Juscelino Kubitschek.",
            "Python é uma linguagem de programação de alto nível com sintaxe expressiva.",
            "O aprendizado por reforço com feedback humano alinha modelos de linguagem.",
        ]
        rag.add_documents(docs)

        # Testa busca BM25 direta
        bm25_res = rag.bm25.search("Brasília Kubitschek")
        self.assertTrue(len(bm25_res) > 0)
        self.assertEqual(bm25_res[0][0], 0)  # Primeiro documento

        # Testa busca híbrida
        hybrid_res = rag.search("programação Python sintaxe")
        self.assertTrue(len(hybrid_res) > 0)
        self.assertIn("Python", hybrid_res[0]["text"])

    def test_rule_based_verifier(self):
        """Valida o verificador determinístico de qualidade e segurança."""
        verifier = RuleBasedVerifier()

        # Resposta de boa qualidade
        reward, fails = verifier.evaluate(
            "Qual a capital do Brasil?",
            "A capital do Brasil é Brasília, planejada por Lúcio Costa e Oscar Niemeyer."
        )
        self.assertGreaterEqual(reward, 0.8)
        self.assertEqual(len(fails), 0)

        # Resposta com repetição e marcador vazado
        reward_bad, fails_bad = verifier.evaluate(
            "Oi",
            "<sistema> erro erro erro erro erro erro erro erro erro erro erro erro erro erro erro"
        )
        self.assertLess(reward_bad, 0.6)
        self.assertIn("high_repetition", fails_bad)

    def test_vision_projector(self):
        """Valida o projetor MLP do córtex visual."""
        from cerebro.visao import VisionProjector
        projector = VisionProjector(vision_dim=768, text_dim=1152, compression_factor=2)
        # 16 patches de 768 dimensões
        dummy_visual_tokens = torch.randn(2, 16, 768)
        projected = projector(dummy_visual_tokens)
        # Comprimido por 4 (2x2) -> 4 patches de 1152 dimensões
        self.assertEqual(projected.shape, (2, 4, 1152))


if __name__ == "__main__":
    unittest.main()
