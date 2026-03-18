"""
Tokenizador BPE da Keilinks
Byte Pair Encoding — subwords inteligentes ao inves de caracteres
"python" = 1 token, "programacao" = 2 tokens
"""

import os
import json
import re
from collections import Counter


class Tokenizador:
    """Tokenizador BPE (Byte Pair Encoding) para portugues"""

    def __init__(self, caminho=None):
        self.vocab = {}
        self.vocab_inverso = {}
        self.tam_vocab = 0
        self.merges = []  # Lista de pares merged (a, b) -> ab

        # Tokens especiais
        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.BOS = '<inicio>'
        self.EOS = '<fim>'
        self.USR = '<vitor>'
        self.KEI = '<keilinks>'
        self.ESPECIAIS = [self.PAD, self.UNK, self.BOS, self.EOS, self.USR, self.KEI]

        if caminho and os.path.exists(caminho):
            self.carregar(caminho)

    def _pre_tokenizar(self, texto):
        """Separa texto em palavras preservando tokens especiais"""
        # Extrai tokens especiais primeiro
        partes = re.split(r'(<pad>|<unk>|<inicio>|<fim>|<vitor>|<keilinks>)', texto)
        palavras = []
        for parte in partes:
            if parte in self.ESPECIAIS:
                palavras.append(parte)
            elif parte:
                # Separa em palavras + pontuacao + espacos
                tokens = re.findall(r'\S+|\s+', parte)
                for t in tokens:
                    palavras.append(t)
        return palavras

    def _palavra_para_chars(self, palavra):
        """Converte palavra em lista de caracteres (unidade base)"""
        if palavra in self.ESPECIAIS:
            return [palavra]
        return list(palavra)

    def _contar_pares(self, palavras_splits):
        """Conta pares adjacentes em todas as palavras"""
        pares = Counter()
        for splits, freq in palavras_splits.items():
            for i in range(len(splits) - 1):
                pares[(splits[i], splits[i + 1])] += freq
        return pares

    def _merge_par(self, palavras_splits, par):
        """Faz merge de um par em todas as palavras"""
        novo = {}
        a, b = par
        merged = a + b
        for splits, freq in palavras_splits.items():
            nova_splits = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and splits[i] == a and splits[i + 1] == b:
                    nova_splits.append(merged)
                    i += 2
                else:
                    nova_splits.append(splits[i])
                    i += 1
            novo[tuple(nova_splits)] = freq
        return novo

    def construir_vocab(self, textos, vocab_alvo=3000):
        """Constroi vocabulario BPE a partir de lista de textos"""
        print(f"Construindo vocab BPE (alvo: {vocab_alvo})...")

        # Conta frequencia de cada palavra
        freq_palavras = Counter()
        for texto in textos:
            palavras = self._pre_tokenizar(texto)
            for p in palavras:
                freq_palavras[p] += 1

        # Inicializa splits: cada palavra vira tupla de chars
        palavras_splits = {}
        for palavra, freq in freq_palavras.items():
            chars = tuple(self._palavra_para_chars(palavra))
            palavras_splits[chars] = freq

        # Coleta todos os caracteres unicos como vocab base
        vocab_base = set()
        for splits in palavras_splits:
            for s in splits:
                vocab_base.add(s)

        # Vocab base = especiais + caracteres
        todos = list(self.ESPECIAIS)
        for c in sorted(vocab_base):
            if c not in todos:
                todos.append(c)

        self.merges = []
        n_merges = vocab_alvo - len(todos)

        for i in range(n_merges):
            pares = self._contar_pares(palavras_splits)
            if not pares:
                break

            melhor = max(pares, key=pares.get)
            if pares[melhor] < 2:
                break

            palavras_splits = self._merge_par(palavras_splits, melhor)
            merged = melhor[0] + melhor[1]
            todos.append(merged)
            self.merges.append(melhor)

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_merges} merges — vocab: {len(todos)}")

        self.vocab = {t: i for i, t in enumerate(todos)}
        self.vocab_inverso = {i: t for t, i in self.vocab.items()}
        self.tam_vocab = len(self.vocab)
        print(f"Vocabulario BPE: {self.tam_vocab} tokens ({len(self.merges)} merges)")

    def _aplicar_bpe(self, palavra):
        """Aplica merges BPE a uma palavra"""
        if palavra in self.ESPECIAIS:
            return [palavra]

        splits = list(palavra)

        for a, b in self.merges:
            nova = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and splits[i] == a and splits[i + 1] == b:
                    nova.append(a + b)
                    i += 2
                else:
                    nova.append(splits[i])
                    i += 1
            splits = nova

        return splits

    def encode(self, texto):
        """Texto -> lista de inteiros"""
        palavras = self._pre_tokenizar(texto)
        tokens = []
        for palavra in palavras:
            subwords = self._aplicar_bpe(palavra)
            for sw in subwords:
                tokens.append(self.vocab.get(sw, self.vocab[self.UNK]))
        return tokens

    def decode(self, tokens):
        """Lista de inteiros -> texto"""
        partes = []
        for t in tokens:
            if t == self.vocab.get(self.PAD) or t == self.vocab.get(self.BOS):
                continue
            texto = self.vocab_inverso.get(t, self.UNK)
            partes.append(texto)
        return ''.join(partes)

    def salvar(self, caminho):
        """Salva vocab + merges"""
        data = {
            'vocab': self.vocab,
            'merges': [list(m) for m in self.merges],
        }
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizador BPE salvo em {caminho}")

    def carregar(self, caminho):
        """Carrega vocab + merges"""
        with open(caminho, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.vocab_inverso = {int(v): k for k, v in self.vocab.items()}
        self.tam_vocab = len(self.vocab)
        self.merges = [tuple(m) for m in data.get('merges', [])]
        print(f"Tokenizador BPE carregado — {self.tam_vocab} tokens, {len(self.merges)} merges")
