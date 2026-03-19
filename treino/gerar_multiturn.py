"""
Gerador de dados Multi-Turn para Keilinks v2
Pega os pares single-turn e agrupa em conversas coerentes de 3-5 turnos.

Estrategias:
  1. Sub-tema: agrupa por palavras-chave compartilhadas (pares similares)
  2. Fluxo natural: saudacao -> tema -> follow-up -> despedida
  3. Emocional: sentimento -> reacao -> follow-up
  4. Identidade: perguntas sobre a Keilinks e o Vitor

Uso:
  python treino/gerar_multiturn.py
  python treino/gerar_multiturn.py --turnos 5
"""

import os
import re
import random
import time
import argparse
from collections import defaultdict, Counter

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Stopwords PT-BR (pra extração de keywords) ──────────────────────────

STOPWORDS = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da',
    'dos', 'das', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'pra',
    'com', 'sem', 'que', 'se', 'é', 'e', 'ou', 'mas', 'não', 'nao',
    'me', 'te', 'eu', 'tu', 'ele', 'ela', 'vc', 'voce', 'você',
    'meu', 'minha', 'seu', 'sua', 'esse', 'essa', 'isso', 'este',
    'como', 'qual', 'quem', 'quando', 'onde', 'quais', 'quanto',
    'foi', 'ser', 'ter', 'ir', 'fazer', 'faz', 'ta', 'tá', 'to',
    'ai', 'aí', 'la', 'lá', 'aqui', 'muito', 'mais', 'bem', 'so',
    'sobre', 'tipo', 'fala', 'pode', 'sabe', 'tem', 'são', 'era',
    'ja', 'já', 'tbm', 'tb', 'ne', 'né', 'po', 'ae', 'oi', 'ei',
    'the', 'of', 'and', 'in', 'to', 'is', 'it', 'at', 'on', 'an',
}

# Categorias macro
CATS = {
    'saudacao': ['oi', 'e ai', 'eai', 'fala', 'bom dia', 'boa noite', 'boa tarde',
                 'salve', 'ola', 'hey', 'beleza', 'como vai', 'tudo bem', 'opa'],
    'despedida': ['tchau', 'flw', 'falou', 'ate mais', 'ate logo',
                  'vou dormir', 'vou nessa', 'to indo', 'bye', 'tmj'],
    'sentimento_pos': ['to feliz', 'animado', 'consegui', 'passei', 'ganhei',
                       'show', 'incrivel', 'top', 'adorei', 'amei', 'deu certo', 'parabens'],
    'sentimento_neg': ['to triste', 'to mal', 'chateado', 'desanimado', 'cansado',
                       'exausto', 'nao consigo', 'puto', 'irritado', 'ansioso', 'saudade'],
    'identidade': ['quem é você', 'seu nome', 'se chama', 'você é',
                   'quem te criou', 'como funciona', 'foi treinada', 'keilinks'],
    'vitor': ['vitor', 'camillo', 'pai do', 'mãe do', 'irmã', 'keila',
              'namorada', 'família', 'adriano', 'juliene', 'natalia', 'vasco', 'retrowave'],
}


def extrair_keywords(texto):
    """Extrai palavras significativas do texto"""
    palavras = re.findall(r'[a-záàâãéêíóôõúç]+', texto.lower())
    return {p for p in palavras if len(p) > 3 and p not in STOPWORDS}


def categorizar_macro(texto):
    t = texto.lower()
    for cat, kws in CATS.items():
        if any(kw in t for kw in kws):
            return cat
    return None


def extrair_pares(caminho):
    pares = []
    with open(caminho, 'r', encoding='utf-8') as f:
        for linha in f:
            m = re.match(r'<vitor>(.*?)<fim><keilinks>(.*?)<fim>', linha.strip())
            if m:
                p, r = m.group(1).strip(), m.group(2).strip()
                if p and r and len(p) > 2 and len(r) > 2:
                    pares.append((p, r))
    return pares


def agrupar_por_subtema(pares):
    """
    Agrupa pares por keywords compartilhadas.
    Pares que falam do mesmo assunto especifico ficam juntos.
    """
    # Indexa cada keyword -> lista de indices de pares
    keyword_to_pares = defaultdict(list)
    pares_keywords = []

    for i, (p, r) in enumerate(pares):
        kws = extrair_keywords(p)
        pares_keywords.append(kws)
        for kw in kws:
            keyword_to_pares[kw].append(i)

    # Pega keywords que conectam 3+ pares (subtemas reais)
    # Mas não keywords com 1000+ pares (genéricas demais)
    subtemas = {}
    for kw, indices in keyword_to_pares.items():
        if 3 <= len(indices) <= 500:
            subtemas[kw] = indices

    # Agrupa: para cada subtema, pega os pares que compartilham MAIS keywords
    grupos = []
    usados = set()

    # Ordena subtemas por tamanho (menores = mais específicos = melhores)
    for kw in sorted(subtemas, key=lambda k: len(subtemas[k])):
        candidatos = [i for i in subtemas[kw] if i not in usados]
        if len(candidatos) < 3:
            continue

        # Dentro dos candidatos, pega os mais similares entre si
        # Score = keywords em comum com o grupo
        kw_grupo = pares_keywords[candidatos[0]]
        grupo = [candidatos[0]]

        for idx in candidatos[1:]:
            overlap = len(kw_grupo & pares_keywords[idx])
            if overlap >= 1:  # pelo menos 1 keyword em comum
                grupo.append(idx)
                kw_grupo = kw_grupo & pares_keywords[idx] | {kw}
                if len(grupo) >= 6:
                    break

        if len(grupo) >= 3:
            grupos.append([pares[i] for i in grupo])
            usados.update(grupo)

    return grupos


def gerar_conversas_tematicas(grupos, turnos_max):
    """Transforma cada grupo de pares similares em conversas multi-turn"""
    conversas = []
    for grupo in grupos:
        random.shuffle(grupo)
        i = 0
        while i + 2 < len(grupo):
            n = random.randint(3, min(turnos_max, len(grupo) - i))
            conversas.append(grupo[i:i + n])
            i += n
    return conversas


def gerar_fluxo_natural(pares, turnos_max):
    """Saudacao -> tema -> follow-up -> despedida"""
    saudacoes = []
    despedidas = []
    meios = defaultdict(list)  # cat -> pares
    gerais = []

    for p, r in pares:
        cat = categorizar_macro(p)
        if cat == 'saudacao':
            saudacoes.append((p, r))
        elif cat == 'despedida':
            despedidas.append((p, r))
        elif cat:
            meios[cat].append((p, r))
        else:
            gerais.append((p, r))

    random.shuffle(saudacoes)
    random.shuffle(despedidas)
    random.shuffle(gerais)

    conversas = []
    idx_s, idx_d, idx_g = 0, 0, 0

    # Cria conversas com fluxo natural
    for cat, lista in meios.items():
        random.shuffle(lista)
        i = 0
        while i < len(lista):
            conversa = []

            # Abre com saudacao (60%)
            if idx_s < len(saudacoes) and random.random() < 0.6:
                conversa.append(saudacoes[idx_s])
                idx_s += 1

            # Corpo: 2-3 pares do tema
            restante = len(lista) - i
            if restante < 2:
                break
            n = random.randint(2, min(3, restante))
            conversa.extend(lista[i:i + n])
            i += n

            # Fecha com despedida (25%)
            if idx_d < len(despedidas) and random.random() < 0.25 and len(conversa) < turnos_max:
                conversa.append(despedidas[idx_d])
                idx_d += 1

            if len(conversa) >= 3:
                conversas.append(conversa[:turnos_max])

    # Conversas so de gerais (agrupados sequencialmente)
    i = 0
    while i + 2 < len(gerais):
        n = random.randint(3, min(turnos_max, len(gerais) - i))

        # Verifica coerencia: pelo menos 1 keyword compartilhada entre pares adjacentes
        grupo = [gerais[i]]
        for j in range(1, n):
            if i + j >= len(gerais):
                break
            kw1 = extrair_keywords(grupo[-1][0])
            kw2 = extrair_keywords(gerais[i + j][0])
            if len(kw1 & kw2) >= 1:
                grupo.append(gerais[i + j])
            # Se nao tem overlap, ainda aceita (50% chance)
            elif random.random() < 0.5:
                grupo.append(gerais[i + j])

        if len(grupo) >= 3:
            conversa = []
            if idx_s < len(saudacoes) and random.random() < 0.4:
                conversa.append(saudacoes[idx_s])
                idx_s += 1
            conversa.extend(grupo[:turnos_max - len(conversa)])
            conversas.append(conversa[:turnos_max])

        i += n

    return conversas


def gerar_emocionais(pares, turnos_max):
    """Conversas emocionais com fluxo natural"""
    positivos = []
    negativos = []
    saudacoes = []
    gerais = []

    for p, r in pares:
        cat = categorizar_macro(p)
        if cat == 'sentimento_pos':
            positivos.append((p, r))
        elif cat == 'sentimento_neg':
            negativos.append((p, r))
        elif cat == 'saudacao':
            saudacoes.append((p, r))
        else:
            gerais.append((p, r))

    conversas = []
    random.shuffle(positivos)
    random.shuffle(negativos)
    random.shuffle(gerais)

    for lista in [positivos, negativos]:
        i = 0
        while i + 1 < len(lista):
            conversa = []
            # Saudacao + sentimentos + follow-up geral
            if saudacoes and random.random() < 0.5:
                conversa.append(random.choice(saudacoes))
            n = random.randint(2, min(3, len(lista) - i))
            conversa.extend(lista[i:i + n])
            i += n
            if gerais and len(conversa) < turnos_max and random.random() < 0.4:
                conversa.append(random.choice(gerais))
            if len(conversa) >= 3:
                conversas.append(conversa[:turnos_max])

    return conversas


def gerar_identidade(pares, turnos_max):
    """Conversas sobre quem a Keilinks é e o Vitor"""
    identidade = []
    vitor = []

    for p, r in pares:
        cat = categorizar_macro(p)
        if cat == 'identidade':
            identidade.append((p, r))
        elif cat == 'vitor':
            vitor.append((p, r))

    conversas = []
    random.shuffle(identidade)
    random.shuffle(vitor)

    # Combina identidade + vitor
    i_id, i_vt = 0, 0
    while i_id + 1 < len(identidade):
        conversa = []
        n = random.randint(2, min(3, len(identidade) - i_id))
        conversa.extend(identidade[i_id:i_id + n])
        i_id += n
        # Adiciona sobre o vitor (50%)
        if i_vt < len(vitor) and random.random() < 0.5:
            conversa.append(vitor[i_vt])
            i_vt += 1
        if len(conversa) >= 3:
            conversas.append(conversa[:turnos_max])

    # Conversas so sobre vitor
    i = 0
    while i + 2 < len(vitor):
        n = random.randint(3, min(turnos_max, len(vitor) - i))
        conversas.append(vitor[i:i + n])
        i += n

    return conversas


def formatar_conversa(turnos):
    return ''.join(f"<vitor>{p}<fim><keilinks>{r}<fim>" for p, r in turnos)


def main():
    parser = argparse.ArgumentParser(description='Gera dados multi-turn pra Keilinks')
    parser.add_argument('--turnos', type=int, default=4, help='Max turnos por conversa (3-6)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    turnos_max = max(3, min(6, args.turnos))

    caminho = 'dados/conversas.txt'
    if not os.path.exists(caminho):
        print("ERRO: dados/conversas.txt nao encontrado")
        return

    print("=" * 60)
    print("  Gerador Multi-Turn v2 — Keilinks")
    print("=" * 60)

    # Extrai pares
    print("\n  Extraindo pares...")
    t0 = time.time()
    pares = extrair_pares(caminho)
    print(f"  {len(pares):,} pares em {time.time()-t0:.1f}s")

    # ─── Estrategia 1: Sub-temas (keywords compartilhadas) ───────────
    print("\n  [1/4] Agrupando por sub-tema...")
    t1 = time.time()
    grupos_subtema = agrupar_por_subtema(pares)
    convs1 = gerar_conversas_tematicas(grupos_subtema, turnos_max)
    print(f"    {len(grupos_subtema):,} sub-temas -> {len(convs1):,} conversas ({time.time()-t1:.1f}s)")

    # ─── Estrategia 2: Fluxo natural ─────────────────────────────────
    print("\n  [2/4] Gerando fluxos naturais...")
    t2 = time.time()
    convs2 = gerar_fluxo_natural(pares, turnos_max)
    print(f"    {len(convs2):,} conversas ({time.time()-t2:.1f}s)")

    # ─── Estrategia 3: Emocionais ────────────────────────────────────
    print("\n  [3/4] Gerando conversas emocionais...")
    t3 = time.time()
    convs3 = gerar_emocionais(pares, turnos_max)
    print(f"    {len(convs3):,} conversas ({time.time()-t3:.1f}s)")

    # ─── Estrategia 4: Identidade ────────────────────────────────────
    print("\n  [4/4] Gerando conversas de identidade...")
    t4 = time.time()
    convs4 = gerar_identidade(pares, turnos_max)
    print(f"    {len(convs4):,} conversas ({time.time()-t4:.1f}s)")

    # Junta tudo
    todas = convs1 + convs2 + convs3 + convs4
    random.shuffle(todas)

    total_turnos = sum(len(c) for c in todas)
    media = total_turnos / len(todas) if todas else 0

    dist = Counter(len(c) for c in todas)
    print(f"\n  TOTAL: {len(todas):,} conversas multi-turn")
    print(f"  Media: {media:.1f} turnos/conversa")
    for n in sorted(dist):
        print(f"    {n} turnos: {dist[n]:,} ({dist[n]/len(todas)*100:.0f}%)")

    # Conta linhas antes
    with open(caminho, 'r', encoding='utf-8') as f:
        linhas_antes = sum(1 for _ in f)

    # Append no conversas.txt
    with open(caminho, 'a', encoding='utf-8') as f:
        f.write("\n# ═══ MULTI-TURN (gerado automaticamente) ═══\n")
        for conv in todas:
            f.write(formatar_conversa(conv) + '\n')

    # Backup separado
    with open('dados/multiturn.txt', 'w', encoding='utf-8') as f:
        for conv in todas:
            f.write(formatar_conversa(conv) + '\n')

    with open(caminho, 'r', encoding='utf-8') as f:
        linhas_depois = sum(1 for _ in f)

    # Exemplos
    print(f"\n  conversas.txt: {linhas_antes:,} -> {linhas_depois:,} (+{len(todas):,})")
    print(f"\n  Exemplos:")
    for i, conv in enumerate(todas[:6]):
        print(f"\n  --- Conversa {i+1} ({len(conv)} turnos) ---")
        for p, r in conv:
            print(f"    V: {p[:75]}")
            print(f"    K: {r[:75]}")

    print(f"\n{'='*60}")
    print(f"  CONCLUIDO — {time.time()-t0:.0f}s")
    print(f"  Proximo: python treino/treinar.py --modelo flash")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
