"""
Limpeza de dados de treino da Keilinks
Remove:
  - Duplicatas exatas
  - Respostas cortadas (terminam com reticencias ou no meio da frase)
  - Pares muito curtos (sem conteudo)
  - Pares em ingles
  - Linhas mal formatadas
  - Respostas identicas a pergunta
  - Excesso de repeticoes

Uso:
  python treino/limpar_dados.py
  python treino/limpar_dados.py --dry  (mostra o que faria sem alterar)
"""

import os
import re
import sys
import time
import argparse

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def eh_ingles(texto):
    """Detecta se texto é predominantemente em inglês"""
    palavras_en = {
        'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'been',
        'will', 'would', 'could', 'should', 'can', 'may', 'might',
        'this', 'that', 'these', 'those', 'with', 'from', 'into',
        'about', 'what', 'which', 'when', 'where', 'who', 'how',
        'your', 'they', 'them', 'their', 'there', 'here',
        'also', 'just', 'very', 'much', 'some', 'any', 'all',
        'because', 'however', 'although', 'therefore',
    }
    palavras = re.findall(r'[a-z]+', texto.lower())
    if len(palavras) < 5:
        return False
    en_count = sum(1 for p in palavras if p in palavras_en)
    return en_count / len(palavras) > 0.3


def resposta_cortada(resp):
    """Detecta respostas que foram cortadas no meio"""
    resp = resp.strip()
    if not resp:
        return True
    # Termina com virgula, preposicao solta, etc
    cortadas = [', ', ' e ', ' ou ', ' de ', ' do ', ' da ', ' em ', ' que ', ' com ']
    for c in cortadas:
        if resp.endswith(c.strip()):
            return True
    # Muito curta pra ser util
    if len(resp) < 5:
        return True
    return False


def par_repetitivo(pergunta, resposta):
    """Detecta quando pergunta e resposta sao quase iguais"""
    p = pergunta.lower().strip()
    r = resposta.lower().strip()
    if p == r:
        return True
    # Resposta contem a pergunta inteira e nada mais
    if len(r) < len(p) * 1.3 and p in r:
        return True
    return False


def limpar(dry_run=False):
    caminho = 'dados/conversas.txt'
    if not os.path.exists(caminho):
        print("ERRO: dados/conversas.txt nao encontrado")
        return

    print("=" * 60)
    print("  Limpeza de Dados — Keilinks")
    print("=" * 60)

    t0 = time.time()

    with open(caminho, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    total = len(linhas)
    print(f"\n  Linhas totais: {total:,}")

    # Contadores
    stats = {
        'duplicata': 0,
        'ingles': 0,
        'cortada': 0,
        'curta': 0,
        'repetitiva': 0,
        'mal_formatada': 0,
        'comentario': 0,
    }

    vistos = set()
    limpas = []

    for linha in linhas:
        l = linha.strip()

        # Comentarios e linhas vazias: mantem
        if not l or l.startswith('#'):
            stats['comentario'] += 1
            limpas.append(linha)
            continue

        # Linhas sem formato correto
        m = re.match(r'<vitor>(.*?)<fim><keilinks>(.*?)<fim>', l)
        if not m:
            # Linhas de identidade/personalidade no inicio do arquivo
            if len(l) > 5:
                limpas.append(linha)
            else:
                stats['mal_formatada'] += 1
            continue

        pergunta = m.group(1).strip()
        resposta = m.group(2).strip()

        # Duplicata exata
        chave = (pergunta.lower(), resposta.lower())
        if chave in vistos:
            stats['duplicata'] += 1
            continue
        vistos.add(chave)

        # Inglês
        if eh_ingles(pergunta) or eh_ingles(resposta):
            stats['ingles'] += 1
            continue

        # Resposta cortada
        if resposta_cortada(resposta):
            stats['cortada'] += 1
            continue

        # Par muito curto (pergunta E resposta curtas)
        if len(pergunta) < 3 and len(resposta) < 10:
            stats['curta'] += 1
            continue

        # Pergunta = Resposta
        if par_repetitivo(pergunta, resposta):
            stats['repetitiva'] += 1
            continue

        limpas.append(linha)

    removidas = total - len(limpas)

    print(f"\n  Resultado:")
    for motivo, n in sorted(stats.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f"    {motivo:20s}: {n:,}")
    print(f"    {'─'*35}")
    print(f"    {'REMOVIDAS':20s}: {removidas:,}")
    print(f"    {'MANTIDAS':20s}: {len(limpas):,}")
    print(f"    {'Reducao':20s}: {removidas/total*100:.1f}%")

    if dry_run:
        print(f"\n  [DRY RUN] Nenhuma alteracao feita.")
    else:
        # Backup
        backup = caminho + '.backup'
        with open(backup, 'w', encoding='utf-8') as f:
            f.writelines(linhas)
        print(f"\n  Backup salvo em: {backup}")

        with open(caminho, 'w', encoding='utf-8') as f:
            f.writelines(limpas)
        print(f"  Arquivo atualizado: {caminho}")

    print(f"\n  Tempo: {time.time()-t0:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true', help='Mostra o que faria sem alterar')
    args = parser.parse_args()
    limpar(dry_run=args.dry)
