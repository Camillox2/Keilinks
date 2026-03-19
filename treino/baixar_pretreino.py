"""
Download de dados para pré-treino da Keilinks
Baixa datasets de PT-BR do HuggingFace em modo streaming
(não precisa baixar tudo de uma vez, economiza disco)

Datasets:
  - CulturaX PT-BR: web brasileira limpa (~50GB total, pegamos ~2-3GB)
  - Wikipedia PT: conhecimento factual (~2GB)
  - OpenAssistant: conversas de assistente (fine-tune)

Uso:
  python treino/baixar_pretreino.py                    # baixa tudo
  python treino/baixar_pretreino.py --apenas wiki      # só Wikipedia
  python treino/baixar_pretreino.py --apenas culturax   # só CulturaX
  python treino/baixar_pretreino.py --apenas conversa   # só conversacional
  python treino/baixar_pretreino.py --limite-gb 2       # limita a 2GB de texto
"""

import os
import sys
import time
import argparse
import json
import re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def limpar_texto(texto):
    """Limpa texto pra pré-treino"""
    if not texto or len(texto.strip()) < 50:
        return None

    texto = texto.strip()

    # Remove URLs
    texto = re.sub(r'https?://\S+', '', texto)
    texto = re.sub(r'www\.\S+', '', texto)

    # Remove excesso de whitespace
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r' {3,}', ' ', texto)

    # Remove linhas muito curtas (lixo)
    linhas = texto.split('\n')
    linhas = [l for l in linhas if len(l.strip()) > 10 or not l.strip()]
    texto = '\n'.join(linhas)

    if len(texto.strip()) < 50:
        return None

    return texto.strip()


def baixar_culturax(saida, limite_bytes):
    """Baixa CulturaX PT-BR (web brasileira limpa e diversa)"""
    from datasets import load_dataset

    print("\n  [CulturaX PT-BR] Baixando via streaming...")
    print(f"  Limite: {limite_bytes / 1e9:.1f} GB")

    try:
        ds = load_dataset(
            "uonlp/CulturaX",
            "pt",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  ERRO ao carregar CulturaX: {e}")
        print("  Tentando dataset alternativo (OSCAR)...")
        return baixar_oscar(saida, limite_bytes)

    bytes_escritos = 0
    docs = 0
    t0 = time.time()

    with open(saida, 'w', encoding='utf-8') as f:
        for item in ds:
            texto = limpar_texto(item.get('text', ''))
            if not texto:
                continue

            f.write(texto + '\n\n')
            bytes_escritos += len(texto.encode('utf-8'))
            docs += 1

            if docs % 10000 == 0:
                elapsed = time.time() - t0
                gb = bytes_escritos / 1e9
                velocidade = bytes_escritos / max(elapsed, 1) / 1e6
                print(f"    {docs:,} docs | {gb:.2f} GB | {velocidade:.1f} MB/s")

            if bytes_escritos >= limite_bytes:
                break

    gb_final = bytes_escritos / 1e9
    print(f"  CulturaX: {docs:,} docs | {gb_final:.2f} GB | {time.time()-t0:.0f}s")
    return bytes_escritos


def baixar_oscar(saida, limite_bytes):
    """Fallback: OSCAR PT-BR"""
    from datasets import load_dataset

    print("\n  [OSCAR PT] Baixando via streaming...")

    try:
        ds = load_dataset(
            "oscar-corpus/OSCAR-2301",
            "pt",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  ERRO ao carregar OSCAR: {e}")
        print("  Tentando mc4...")
        return baixar_mc4(saida, limite_bytes)

    bytes_escritos = 0
    docs = 0
    t0 = time.time()

    with open(saida, 'w', encoding='utf-8') as f:
        for item in ds:
            texto = limpar_texto(item.get('text', ''))
            if not texto:
                continue

            f.write(texto + '\n\n')
            bytes_escritos += len(texto.encode('utf-8'))
            docs += 1

            if docs % 10000 == 0:
                gb = bytes_escritos / 1e9
                print(f"    {docs:,} docs | {gb:.2f} GB")

            if bytes_escritos >= limite_bytes:
                break

    print(f"  OSCAR: {docs:,} docs | {bytes_escritos/1e9:.2f} GB | {time.time()-t0:.0f}s")
    return bytes_escritos


def baixar_mc4(saida, limite_bytes):
    """Fallback 2: mC4 Portuguese"""
    from datasets import load_dataset

    print("\n  [mC4 PT] Baixando via streaming...")

    try:
        ds = load_dataset(
            "allenai/c4",
            "pt",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  ERRO: {e}")
        return 0

    bytes_escritos = 0
    docs = 0
    t0 = time.time()

    with open(saida, 'w', encoding='utf-8') as f:
        for item in ds:
            texto = limpar_texto(item.get('text', ''))
            if not texto:
                continue

            f.write(texto + '\n\n')
            bytes_escritos += len(texto.encode('utf-8'))
            docs += 1

            if docs % 10000 == 0:
                gb = bytes_escritos / 1e9
                print(f"    {docs:,} docs | {gb:.2f} GB")

            if bytes_escritos >= limite_bytes:
                break

    print(f"  mC4: {docs:,} docs | {bytes_escritos/1e9:.2f} GB | {time.time()-t0:.0f}s")
    return bytes_escritos


def baixar_wikipedia(saida, limite_bytes):
    """Baixa Wikipedia PT (conhecimento factual)"""
    from datasets import load_dataset

    print("\n  [Wikipedia PT] Baixando...")

    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.pt",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  ERRO: {e}")
        return 0

    bytes_escritos = 0
    docs = 0
    t0 = time.time()

    with open(saida, 'w', encoding='utf-8') as f:
        for item in ds:
            texto = item.get('text', '')
            if not texto or len(texto) < 200:
                continue

            # Limpa artigos muito curtos ou stubs
            texto = limpar_texto(texto)
            if not texto:
                continue

            f.write(texto + '\n\n')
            bytes_escritos += len(texto.encode('utf-8'))
            docs += 1

            if docs % 5000 == 0:
                gb = bytes_escritos / 1e9
                print(f"    {docs:,} artigos | {gb:.2f} GB")

            if bytes_escritos >= limite_bytes:
                break

    print(f"  Wikipedia: {docs:,} artigos | {bytes_escritos/1e9:.2f} GB | {time.time()-t0:.0f}s")
    return bytes_escritos


def baixar_conversacional(saida):
    """Baixa dados conversacionais pra fine-tune"""
    from datasets import load_dataset

    print("\n  [Conversacional] Baixando OpenAssistant + Dolly PT...")

    bytes_escritos = 0
    pares = 0
    t0 = time.time()

    with open(saida, 'w', encoding='utf-8') as f:
        # OpenAssistant - conversas em vários idiomas (filtrar PT)
        try:
            print("    OpenAssistant...")
            ds = load_dataset(
                "OpenAssistant/oasst2",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            for item in ds:
                lang = item.get('lang', '')
                if lang != 'pt-BR' and lang != 'pt':
                    continue

                texto = item.get('text', '').strip()
                role = item.get('role', '')

                if texto and len(texto) > 5:
                    # Salva como texto corrido pra pré-treino
                    f.write(texto + '\n')
                    bytes_escritos += len(texto.encode('utf-8'))
                    pares += 1

            print(f"    OpenAssistant: {pares} textos PT-BR")
        except Exception as e:
            print(f"    OpenAssistant falhou: {e}")

        # Dolly traduzido
        try:
            print("    Dolly PT-BR...")
            ds = load_dataset(
                "Gustrd/dolly-15k-libretranslate-pt",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            dolly_count = 0
            for item in ds:
                instrucao = item.get('instruction', '').strip()
                resposta = item.get('output', '') or item.get('response', '')
                resposta = resposta.strip() if resposta else ''

                if instrucao and resposta and len(resposta) > 10:
                    # Formato conversacional
                    f.write(f"<vitor>{instrucao}<fim><keilinks>{resposta}<fim>\n")
                    bytes_escritos += len(instrucao.encode('utf-8')) + len(resposta.encode('utf-8'))
                    dolly_count += 1

            print(f"    Dolly: {dolly_count} pares")
            pares += dolly_count
        except Exception as e:
            print(f"    Dolly falhou: {e}")

        # Alpaca PT-BR
        try:
            print("    Alpaca PT-BR...")
            ds = load_dataset(
                "dominguesm/alpaca-data-pt-br",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            alpaca_count = 0
            for item in ds:
                instrucao = item.get('instruction', '').strip()
                entrada = item.get('input', '').strip()
                resposta = item.get('output', '').strip()

                if entrada:
                    instrucao = f"{instrucao} {entrada}"

                if instrucao and resposta and len(resposta) > 10:
                    f.write(f"<vitor>{instrucao}<fim><keilinks>{resposta}<fim>\n")
                    bytes_escritos += len(instrucao.encode('utf-8')) + len(resposta.encode('utf-8'))
                    alpaca_count += 1

            print(f"    Alpaca: {alpaca_count} pares")
            pares += alpaca_count
        except Exception as e:
            print(f"    Alpaca falhou: {e}")

    print(f"  Conversacional: {pares:,} pares | {bytes_escritos/1e6:.1f} MB | {time.time()-t0:.0f}s")
    return bytes_escritos


def main():
    parser = argparse.ArgumentParser(description='Download de dados pra pré-treino')
    parser.add_argument('--apenas', choices=['culturax', 'wiki', 'conversa', 'oscar'], help='Baixa apenas um dataset')
    parser.add_argument('--limite-gb', type=float, default=10.0, help='Limite em GB pra datasets grandes (default: 10)')
    args = parser.parse_args()

    limite_bytes = int(args.limite_gb * 1e9)

    print("=" * 60)
    print("  Download de Dados — Pré-treino Keilinks")
    print("=" * 60)
    print(f"  Limite por dataset: {args.limite_gb} GB")

    os.makedirs('dados/pretreino', exist_ok=True)

    total_bytes = 0
    t_total = time.time()

    if args.apenas is None or args.apenas == 'culturax':
        total_bytes += baixar_culturax('dados/pretreino/culturax_pt.txt', limite_bytes)

    if args.apenas is None or args.apenas == 'wiki':
        total_bytes += baixar_wikipedia('dados/pretreino/wikipedia_pt.txt', limite_bytes)

    if args.apenas is None or args.apenas == 'conversa':
        total_bytes += baixar_conversacional('dados/pretreino/conversacional_pt.txt')

    if args.apenas == 'oscar':
        total_bytes += baixar_oscar('dados/pretreino/oscar_pt.txt', limite_bytes)

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD CONCLUÍDO")
    print(f"{'='*60}")
    print(f"  Total: {total_bytes/1e9:.2f} GB")
    print(f"  Tempo: {(time.time()-t_total)/60:.1f} min")
    print(f"  Salvos em: dados/pretreino/")
    print(f"\n  Próximo passo:")
    print(f"    python treino/pretreinar.py --modelo flash")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
