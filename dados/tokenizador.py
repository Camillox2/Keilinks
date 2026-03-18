"""
Tokenizador da Keilinks
Converte texto em números (tokens) que o modelo entende
Usando SentencePiece — funciona bem com português
"""

import os
import json


class Tokenizador:
    """Tokenizador simples baseado em caracteres para começar"""

    def __init__(self, caminho=None):
        self.vocab = {}
        self.vocab_inverso = {}
        self.tam_vocab = 0

        # Tokens especiais
        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.BOS = '<inicio>'  # início de fala
        self.EOS = '<fim>'     # fim de fala
        self.USR = '<vitor>'   # fala do Vitor
        self.KEI = '<keilinks>'  # fala da Keilinks

        if caminho and os.path.exists(caminho):
            self.carregar(caminho)

    def construir_vocab(self, textos):
        """Constrói vocabulário a partir de lista de textos"""
        chars = set()
        for texto in textos:
            chars.update(texto)

        tokens_especiais = [self.PAD, self.UNK, self.BOS, self.EOS, self.USR, self.KEI]
        todos = tokens_especiais + sorted(chars)

        self.vocab = {t: i for i, t in enumerate(todos)}
        self.vocab_inverso = {i: t for t, i in self.vocab.items()}
        self.tam_vocab = len(self.vocab)

        print(f"Vocabulário: {self.tam_vocab} tokens")

    def encode(self, texto):
        """Texto → lista de inteiros"""
        return [self.vocab.get(c, self.vocab[self.UNK]) for c in texto]

    def decode(self, tokens):
        """Lista de inteiros → texto"""
        return ''.join([
            self.vocab_inverso.get(t, self.UNK)
            for t in tokens
            if t not in [self.vocab.get(self.PAD), self.vocab.get(self.BOS)]
        ])

    def salvar(self, caminho):
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab}, f, ensure_ascii=False, indent=2)
        print(f"Tokenizador salvo em {caminho}")

    def carregar(self, caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.vocab_inverso = {int(v): k for k, v in self.vocab.items()}
        self.tam_vocab = len(self.vocab)
        print(f"Tokenizador carregado — {self.tam_vocab} tokens")
