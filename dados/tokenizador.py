"""
Tokenizador BPE da Keilinks
Byte Pair Encoding — subwords inteligentes ao inves de caracteres
"python" = 1 token, "programacao" = 2 tokens
Otimizado pra vocab grande (32K+) com contagem incremental
"""

import os
import json
import re
import random
import time as _time
from collections import Counter, defaultdict


class Tokenizador:
    """Tokenizador BPE (Byte Pair Encoding) para portugues"""

    def __init__(self, caminho=None):
        self.vocab = {}
        self.vocab_inverso = {}
        self.tam_vocab = 0
        self.merges = []

        # Tokens especiais
        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.BOS = '<inicio>'
        self.EOS = '<fim>'
        self.USR = '<vitor>'
        self.USR2 = '<user>'
        self.KEI = '<keilinks>'
        self.ESPECIAIS = [self.PAD, self.UNK, self.BOS, self.EOS, self.USR, self.KEI, self.USR2]

        if caminho and os.path.exists(caminho):
            self.carregar(caminho)

    def _pre_tokenizar(self, texto):
        """Separa texto em palavras preservando tokens especiais"""
        partes = re.split(r'(<pad>|<unk>|<inicio>|<fim>|<vitor>|<user>|<keilinks>)', texto)
        palavras = []
        for parte in partes:
            if parte in self.ESPECIAIS:
                palavras.append(parte)
            elif parte:
                tokens = re.findall(r'\S+|\s+', parte)
                for t in tokens:
                    palavras.append(t)
        return palavras

    def _palavra_para_chars(self, palavra):
        """Converte palavra em lista de caracteres (unidade base)"""
        if palavra in self.ESPECIAIS:
            return [palavra]
        return list(palavra)

    def construir_vocab(self, textos, vocab_alvo=32000, max_texto_mb=15):
        """Constroi vocabulario BPE com contagem incremental.
        Em vez de recontar todos os pares a cada merge (lento),
        so atualiza as palavras afetadas pelo merge (rapido).
        """
        print(f"Construindo vocab BPE (alvo: {vocab_alvo})...")

        # Amostra o texto se for muito grande
        texto_total = '\n'.join(textos)
        tam_mb = len(texto_total.encode('utf-8')) / 1e6
        if tam_mb > max_texto_mb:
            print(f"  Texto: {tam_mb:.0f}MB — amostrando {max_texto_mb}MB pra construir vocab...")
            linhas = texto_total.split('\n')
            random.shuffle(linhas)
            amostrado = []
            tam_acum = 0
            for linha in linhas:
                amostrado.append(linha)
                tam_acum += len(linha.encode('utf-8'))
                if tam_acum >= max_texto_mb * 1e6:
                    break
            texto_total = '\n'.join(amostrado)
            print(f"  Amostrado: {len(amostrado):,} linhas ({tam_acum/1e6:.1f}MB)")

        # Conta frequencia de cada palavra
        freq_palavras = Counter()
        palavras = self._pre_tokenizar(texto_total)
        for p in palavras:
            freq_palavras[p] += 1

        # Filtra palavras com freq >= 2 (palavras unicas nao contribuem pra merges)
        total_unicas = len(freq_palavras)
        freq_palavras = {p: f for p, f in freq_palavras.items() if f >= 2}
        print(f"  Palavras unicas: {total_unicas:,} (usando {len(freq_palavras):,} com freq>=2)")

        # Inicializa: cada palavra vira lista de chars, indexada por ID
        # words[id] = [lista de splits]
        # word_freq[id] = frequencia
        words = {}
        word_freq = {}
        for idx, (palavra, freq) in enumerate(freq_palavras.items()):
            if palavra in self.ESPECIAIS:
                words[idx] = [palavra]
            else:
                words[idx] = list(palavra)
            word_freq[idx] = freq

        # Coleta caracteres base
        vocab_base = set()
        for splits in words.values():
            for s in splits:
                vocab_base.add(s)

        todos = list(self.ESPECIAIS)
        for c in sorted(vocab_base):
            if c not in todos:
                todos.append(c)

        # ─── Contagem incremental de pares ───────────────────────────────
        # pair_counts[(a,b)] = soma das frequencias das palavras que contem (a,b)
        # pair_to_words[(a,b)] = set de word_ids que contem esse par
        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)

        # Inicializa contagem
        for wid, splits in words.items():
            freq = word_freq[wid]
            for i in range(len(splits) - 1):
                par = (splits[i], splits[i + 1])
                pair_counts[par] += freq
                pair_to_words[par].add(wid)

        self.merges = []
        n_merges = vocab_alvo - len(todos)
        t0 = _time.time()

        for step in range(n_merges):
            if not pair_counts:
                break

            # Acha o par mais frequente
            melhor = max(pair_counts, key=pair_counts.get)
            if pair_counts[melhor] < 2:
                break

            a, b = melhor
            merged = a + b
            todos.append(merged)
            self.merges.append(melhor)

            # Atualiza APENAS as palavras que contem o par (a,b)
            affected = list(pair_to_words.pop(melhor, set()))
            del pair_counts[melhor]

            for wid in affected:
                splits = words[wid]
                freq = word_freq[wid]

                # Remove contagens antigas dos pares adjacentes nesta palavra
                for i in range(len(splits) - 1):
                    par = (splits[i], splits[i + 1])
                    if par != melhor:  # ja removemos o melhor
                        pair_counts[par] -= freq
                        if pair_counts[par] <= 0:
                            del pair_counts[par]
                        pair_to_words[par].discard(wid)
                        if not pair_to_words[par]:
                            del pair_to_words[par]

                # Aplica o merge
                nova = []
                i = 0
                while i < len(splits):
                    if i < len(splits) - 1 and splits[i] == a and splits[i + 1] == b:
                        nova.append(merged)
                        i += 2
                    else:
                        nova.append(splits[i])
                        i += 1
                words[wid] = nova

                # Adiciona contagens novas dos pares adjacentes
                for i in range(len(nova) - 1):
                    par = (nova[i], nova[i + 1])
                    pair_counts[par] += freq
                    pair_to_words[par].add(wid)

            if (step + 1) % 2000 == 0:
                elapsed = _time.time() - t0
                rate = (step + 1) / elapsed
                eta = (n_merges - step - 1) / rate if rate > 0 else 0
                print(f"  {step+1}/{n_merges} merges — vocab: {len(todos)} — "
                      f"{rate:.0f} merges/s — ETA: {eta/60:.1f}min")

        self.vocab = {t: i for i, t in enumerate(todos)}
        self.vocab_inverso = {i: t for t, i in self.vocab.items()}
        self.tam_vocab = len(self.vocab)
        total_time = _time.time() - t0
        print(f"Vocabulario BPE: {self.tam_vocab} tokens ({len(self.merges)} merges) em {total_time/60:.1f}min")

    def _build_merge_ranks(self):
        """Constroi dicionario de ranks pra BPE rapido: (a,b) -> prioridade"""
        self._merge_ranks = {}
        for i, (a, b) in enumerate(self.merges):
            self._merge_ranks[(a, b)] = i

    def _aplicar_bpe(self, palavra):
        """Aplica merges BPE usando ranks (O(n*log(n)) em vez de O(n*merges))"""
        if palavra in self.ESPECIAIS:
            return [palavra]

        if not hasattr(self, '_merge_ranks') or not self._merge_ranks:
            self._build_merge_ranks()

        splits = list(palavra)

        while len(splits) > 1:
            # Acha o par com menor rank (maior prioridade)
            melhor_rank = float('inf')
            melhor_idx = -1
            for i in range(len(splits) - 1):
                par = (splits[i], splits[i + 1])
                rank = self._merge_ranks.get(par, float('inf'))
                if rank < melhor_rank:
                    melhor_rank = rank
                    melhor_idx = i

            if melhor_rank == float('inf'):
                break  # nenhum par encontrado nos merges

            # Aplica o merge
            splits = splits[:melhor_idx] + [splits[melhor_idx] + splits[melhor_idx + 1]] + splits[melhor_idx + 2:]

        return splits

    def encode(self, texto):
        """Texto -> lista de inteiros. Usa cache pra palavras repetidas."""
        if not hasattr(self, '_cache_bpe'):
            self._cache_bpe = {}

        palavras = self._pre_tokenizar(texto)
        tokens = []
        unk_id = self.vocab[self.UNK]

        for palavra in palavras:
            if palavra in self._cache_bpe:
                tokens.extend(self._cache_bpe[palavra])
                continue

            subwords = self._aplicar_bpe(palavra)
            ids = [self.vocab.get(sw, unk_id) for sw in subwords]

            # Cache (limita tamanho pra nao explodir memoria)
            if len(self._cache_bpe) < 500000:
                self._cache_bpe[palavra] = ids

            tokens.extend(ids)

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
