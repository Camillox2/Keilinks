"""
Rebalanceamento de dados de treino da Keilinks
Problema: 58% dos dados são Wikipedia, modelo aprende a ser enciclopédia
Solução: Limita Wikipedia a ~20K linhas, mantém todas as conversas naturais

Uso:
  python treino/rebalancear_dados.py
  python treino/rebalancear_dados.py --dry  (mostra o que faria sem alterar)
  python treino/rebalancear_dados.py --wiki 15000  (limite de linhas wiki)
"""

import os
import re
import sys
import random
import argparse
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Padrões que indicam conteúdo enciclopédico/Wikipedia
PADROES_WIKI = [
    r'^<vitor>O que (é|e) ',
    r'^<vitor>Me fal[ea] sobre ',
    r'^<vitor>Quem (é|e) ',
    r'^<vitor>Resumir ',
    r'^<vitor>Defina ',
    r'^<vitor>Explique o conceito de ',
    r'^<vitor>O que significa ',
    r'^<vitor>Qual a definição de ',
    r'^<vitor>Descreva ',
]

# Respostas que parecem Wikipedia (municipio, freguesia, etc)
PADROES_RESP_WIKI = [
    r'é uma? freguesias? portugues',
    r'é um município brasileiro',
    r'é uma? cidades? brasileir',
    r'é uma? localidades?',
    r'censo de \d{4}',
    r'km² de área',
    r'densidade populacional',
    r'Ab urbe condita',
    r'foi um ano comum',
    r'foi uma? romancista',
    r'foi uma? escritor',
    r'foi uma? compositor',
    r'foi uma? político',
    r'nasceu em \d+ de',
    r'morreu em \d+ de',
]

PADRAO_WIKI_COMPILED = re.compile('|'.join(PADROES_WIKI), re.IGNORECASE)
PADRAO_RESP_COMPILED = re.compile('|'.join(PADROES_RESP_WIKI), re.IGNORECASE)


def eh_wikipedia(linha):
    """Detecta se uma linha é conteúdo estilo Wikipedia"""
    linha = linha.strip()
    if not linha.startswith('<vitor>'):
        return False

    # Checa pergunta
    if PADRAO_WIKI_COMPILED.search(linha):
        return True

    # Checa resposta
    m = re.search(r'<keilinks>(.*?)<fim>', linha)
    if m:
        resp = m.group(1)
        if PADRAO_RESP_COMPILED.search(resp):
            return True

    return False


def rebalancear(dry_run=False, limite_wiki=20000):
    caminho = 'dados/conversas.txt'
    if not os.path.exists(caminho):
        print("ERRO: dados/conversas.txt nao encontrado")
        return

    print("=" * 60)
    print("  Rebalanceamento de Dados — Keilinks")
    print("=" * 60)

    t0 = time.time()

    with open(caminho, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    total = len(linhas)
    print(f"\n  Linhas totais: {total:,}")

    # Separa em categorias
    wiki = []
    conversas = []
    outras = []  # comentários, identidade, linhas vazias

    for linha in linhas:
        l = linha.strip()
        if not l or l.startswith('#') or not l.startswith('<vitor>'):
            outras.append(linha)
        elif eh_wikipedia(linha):
            wiki.append(linha)
        else:
            conversas.append(linha)

    print(f"\n  Categorias:")
    print(f"    Wikipedia/enciclopédia: {len(wiki):,} ({len(wiki)/total*100:.1f}%)")
    print(f"    Conversas naturais:     {len(conversas):,} ({len(conversas)/total*100:.1f}%)")
    print(f"    Outras (identidade etc): {len(outras):,}")

    # Amostra aleatória do Wikipedia
    if len(wiki) > limite_wiki:
        random.seed(42)  # reprodutível
        wiki_mantidas = random.sample(wiki, limite_wiki)
        wiki_removidas = len(wiki) - limite_wiki
    else:
        wiki_mantidas = wiki
        wiki_removidas = 0

    # Monta arquivo final: identidade + conversas + wiki amostrada
    resultado = outras + conversas + wiki_mantidas
    random.seed(42)
    # Embaralha só as linhas de treino (mantém identidade no topo)
    identidade = [l for l in outras if l.strip()]
    vazias = [l for l in outras if not l.strip()]
    treino = conversas + wiki_mantidas
    random.shuffle(treino)
    resultado = identidade + ['\n'] + treino

    novo_total = len(resultado)
    wiki_pct = len(wiki_mantidas) / max(len(conversas) + len(wiki_mantidas), 1) * 100

    print(f"\n  Resultado:")
    print(f"    Conversas mantidas:  {len(conversas):,} (100%)")
    print(f"    Wikipedia mantidas:  {len(wiki_mantidas):,} (de {len(wiki):,})")
    print(f"    Wikipedia removidas: {wiki_removidas:,}")
    print(f"    {'─'*40}")
    print(f"    ANTES:  {total:,} linhas")
    print(f"    DEPOIS: {novo_total:,} linhas")
    print(f"    Redução: {total - novo_total:,} linhas ({(total-novo_total)/total*100:.1f}%)")
    print(f"    Proporção wiki: {wiki_pct:.1f}% (antes era {len(wiki)/max(len(wiki)+len(conversas),1)*100:.1f}%)")

    if dry_run:
        print(f"\n  [DRY RUN] Nenhuma alteração feita.")
    else:
        # Backup
        backup = caminho + '.pre_rebalance'
        with open(backup, 'w', encoding='utf-8') as f:
            f.writelines(linhas)
        print(f"\n  Backup salvo em: {backup}")

        with open(caminho, 'w', encoding='utf-8') as f:
            f.writelines(resultado)
        print(f"  Arquivo atualizado: {caminho}")

    # Mostra exemplos de conversas naturais mantidas
    print(f"\n  Exemplos de conversas naturais (o que o modelo vai aprender):")
    random.seed(123)
    exemplos = random.sample(conversas, min(5, len(conversas)))
    for ex in exemplos:
        ex = ex.strip()
        if len(ex) > 120:
            ex = ex[:120] + '...'
        print(f"    {ex}")

    print(f"\n  Tempo: {time.time()-t0:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true', help='Mostra o que faria sem alterar')
    parser.add_argument('--wiki', type=int, default=20000, help='Limite de linhas Wikipedia (default: 20000)')
    args = parser.parse_args()
    rebalancear(dry_run=args.dry, limite_wiki=args.wiki)
