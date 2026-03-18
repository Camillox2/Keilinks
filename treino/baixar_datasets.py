"""
Baixa e processa datasets publicos para treino da Keilinks
Converte tudo pro formato <vitor>...<fim><keilinks>...<fim>
Tambem salva no MySQL (knowledge) para retrieval

Datasets:
  1. Alpaca PT-BR (~52K pares instrucao/resposta)
  2. Dolly PT-BR (~15K pares)
  3. OpenAssistant (conversas multilíngues, filtra PT)
  4. TechQA scraped (Stack Overflow PT, se disponivel)

Uso:
  python treino/baixar_datasets.py
  python treino/baixar_datasets.py --apenas alpaca
  python treino/baixar_datasets.py --limite 5000
"""

import os
import sys
import re
import json
import time
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ─── Utilidades ──────────────────────────────────────────────────────────────

def limpar_texto(texto):
    """Remove lixo, normaliza espacos"""
    if not texto:
        return ''
    texto = texto.strip()
    # Remove markdown excessivo
    texto = re.sub(r'\*{2,}', '', texto)
    texto = re.sub(r'#{1,6}\s*', '', texto)
    # Remove URLs longas
    texto = re.sub(r'https?://\S{100,}', '[link]', texto)
    # Normaliza espacos e quebras
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r'[ \t]{2,}', ' ', texto)
    return texto.strip()


def texto_valido(pergunta, resposta):
    """Filtra pares de baixa qualidade"""
    if not pergunta or not resposta:
        return False
    if len(pergunta) < 5 or len(resposta) < 10:
        return False
    if len(resposta) > 3000:
        return False  # muito longo pro contexto do modelo
    # Muito codigo sem explicacao
    if resposta.count('```') >= 4 and len(resposta.split()) < 20:
        return False
    # Resposta so em ingles (heuristica simples)
    palavras_pt = ['que', 'para', 'com', 'uma', 'por', 'como', 'mais',
                   'seu', 'sua', 'isso', 'este', 'esta', 'pode', 'tem',
                   'ser', 'foi', 'sao', 'nos', 'ele', 'ela', 'voce']
    palavras = resposta.lower().split()
    pt_count = sum(1 for p in palavras if p in palavras_pt)
    if len(palavras) > 10 and pt_count < 2:
        return False  # provavelmente ingles
    return True


def formatar_par(pergunta, resposta):
    """Formata no estilo Keilinks"""
    p = limpar_texto(pergunta)
    r = limpar_texto(resposta)
    if not texto_valido(p, r):
        return None
    return f"<vitor>{p}<fim><keilinks>{r}<fim>"


def salvar_no_mysql(pergunta, resposta, fonte, categoria='geral'):
    """Salva no banco knowledge do MySQL"""
    try:
        from dados.database import knowledge_adicionar
        knowledge_adicionar(
            pergunta[:500],
            resposta[:5000],
            fonte=fonte,
            categoria=categoria
        )
    except Exception:
        pass  # silencioso, nao para o processo


# ─── Alpaca PT-BR ────────────────────────────────────────────────────────────

def baixar_alpaca(limite=None):
    """Alpaca PT-BR — ~52K pares instrucao/resposta"""
    print("\n" + "=" * 60)
    print("  [1/4] Alpaca PT-BR")
    print("=" * 60)

    from datasets import load_dataset

    # Tenta varios repos conhecidos
    repos = [
        "dominguesm/alpaca-data-pt-br",
        "recogna-nlp/alpaca-ptbr",
    ]

    ds = None
    for repo in repos:
        try:
            print(f"  Baixando: {repo}...")
            ds = load_dataset(repo, split='train')
            print(f"  OK! {len(ds)} exemplos")
            break
        except Exception as e:
            print(f"  Falhou: {e}")
            continue

    if ds is None:
        print("  ERRO: Nenhum repo do Alpaca disponivel")
        return []

    pares = []
    salvos_mysql = 0
    total = min(len(ds), limite) if limite else len(ds)

    for i, ex in enumerate(ds):
        if limite and i >= limite:
            break

        # Alpaca tem 'instruction', 'input', 'output'
        instrucao = ex.get('instruction', '') or ex.get('instrucao', '') or ''
        inp = ex.get('input', '') or ex.get('entrada', '') or ''
        output = ex.get('output', '') or ex.get('saida', '') or ''

        # Combina instrucao + input como pergunta
        if inp and inp.strip():
            pergunta = f"{instrucao}\n{inp}"
        else:
            pergunta = instrucao

        par = formatar_par(pergunta, output)
        if par:
            pares.append(par)
            # Salva no MySQL a cada 10 (nao sobrecarrega)
            if len(pares) % 10 == 0:
                salvar_no_mysql(pergunta[:500], output[:5000], 'alpaca', 'instrucao')
                salvos_mysql += 1

        if (i + 1) % 5000 == 0:
            print(f"  Processados: {i+1}/{total} | Validos: {len(pares)} | MySQL: {salvos_mysql}")

    print(f"  Total: {len(pares)} pares validos | MySQL: {salvos_mysql}")
    return pares


# ─── Dolly PT-BR ─────────────────────────────────────────────────────────────

def baixar_dolly(limite=None):
    """Dolly — dataset de instrucoes multilíngue"""
    print("\n" + "=" * 60)
    print("  [2/4] Dolly / Databricks")
    print("=" * 60)

    from datasets import load_dataset

    repos = [
        ("databricks/databricks-dolly-15k", None),
    ]

    ds = None
    for repo, lang in repos:
        try:
            print(f"  Baixando: {repo}...")
            ds = load_dataset(repo, split='train')
            print(f"  OK! {len(ds)} exemplos")
            break
        except Exception as e:
            print(f"  Falhou: {e}")
            continue

    if ds is None:
        print("  ERRO: Dolly indisponivel")
        return []

    pares = []
    total = min(len(ds), limite) if limite else len(ds)

    for i, ex in enumerate(ds):
        if limite and i >= limite:
            break

        instrucao = ex.get('instruction', '')
        contexto = ex.get('context', '')
        output = ex.get('response', '')

        if contexto and contexto.strip():
            pergunta = f"{instrucao}\nContexto: {contexto[:300]}"
        else:
            pergunta = instrucao

        par = formatar_par(pergunta, output)
        if par:
            pares.append(par)
            if len(pares) % 10 == 0:
                cat = ex.get('category', 'geral')
                salvar_no_mysql(pergunta[:500], output[:5000], 'dolly', cat)

        if (i + 1) % 5000 == 0:
            print(f"  Processados: {i+1}/{total} | Validos: {len(pares)}")

    print(f"  Total: {len(pares)} pares validos")
    return pares


# ─── OpenAssistant (conversas) ───────────────────────────────────────────────

def baixar_oasst(limite=None):
    """OpenAssistant — conversas reais multilíngues"""
    print("\n" + "=" * 60)
    print("  [3/4] OpenAssistant (PT + EN)")
    print("=" * 60)

    from datasets import load_dataset

    try:
        print("  Baixando OpenAssistant...")
        ds = load_dataset("OpenAssistant/oasst1", split='train')
        print(f"  OK! {len(ds)} mensagens")
    except Exception as e:
        print(f"  Falhou: {e}")
        return []

    # Organiza por thread: agrupa mensagens pai/filho
    msgs_by_id = {}
    for ex in ds:
        msgs_by_id[ex['message_id']] = ex

    pares = []
    total_processados = 0

    for ex in ds:
        if limite and len(pares) >= limite:
            break

        # Pega apenas respostas (que tem parent)
        if not ex.get('parent_id'):
            continue

        parent = msgs_by_id.get(ex['parent_id'])
        if not parent:
            continue

        # Filtra: pega PT e EN (ingles tem conteudo valioso)
        lang = ex.get('lang', '')
        if lang not in ('pt-BR', 'pt', 'en', ''):
            continue

        pergunta = parent.get('text', '')
        resposta = ex.get('text', '')

        # Pega apenas respostas bem avaliadas
        rank = ex.get('rank', 99)
        if rank is not None and rank > 2:
            continue  # so top 2 respostas

        par = formatar_par(pergunta, resposta)
        if par:
            pares.append(par)
            if len(pares) % 10 == 0:
                salvar_no_mysql(pergunta[:500], resposta[:5000], 'oasst', 'conversa')

        total_processados += 1
        if total_processados % 5000 == 0:
            print(f"  Processados: {total_processados} | Validos: {len(pares)}")

    print(f"  Total: {len(pares)} pares validos")
    return pares


# ─── Wikipedia PT-BR (resumos) ───────────────────────────────────────────────

def baixar_wiki_pt(limite=None):
    """Wikipedia PT — artigos convertidos em Q&A"""
    print("\n" + "=" * 60)
    print("  [4/4] Wikipedia PT-BR (resumos)")
    print("=" * 60)

    from datasets import load_dataset

    try:
        print("  Baixando Wikipedia PT (pode demorar)...")
        ds = load_dataset("wikipedia", "20220301.pt", split='train', trust_remote_code=True)
        print(f"  OK! {len(ds)} artigos")
    except Exception as e:
        print(f"  Falhou: {e}")
        # Tenta alternativa
        try:
            print("  Tentando alternativa...")
            ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split='train')
            print(f"  OK! {len(ds)} artigos")
        except Exception as e2:
            print(f"  Falhou tambem: {e2}")
            return []

    pares = []
    max_items = min(len(ds), limite) if limite else min(len(ds), 100000)

    for i, ex in enumerate(ds):
        if i >= max_items:
            break

        titulo = ex.get('title', '')
        texto = ex.get('text', '')

        if not titulo or not texto or len(texto) < 100:
            continue

        # Pega primeiro paragrafo como resumo
        paragrafos = [p.strip() for p in texto.split('\n\n') if len(p.strip()) > 50]
        if not paragrafos:
            continue

        resumo = paragrafos[0][:800]

        # Cria pergunta/resposta
        pergunta = f"O que e {titulo}?"
        par = formatar_par(pergunta, resumo)
        if par:
            pares.append(par)
            if len(pares) % 10 == 0:
                salvar_no_mysql(pergunta[:500], resumo[:5000], 'wikipedia', 'conhecimento')

        # Segunda variacao: "me fala sobre X"
        if len(paragrafos) > 1 and len(pares) < max_items:
            resumo2 = ' '.join(paragrafos[:2])[:1000]
            pergunta2 = f"Me fala sobre {titulo}"
            par2 = formatar_par(pergunta2, resumo2)
            if par2:
                pares.append(par2)

        if (i + 1) % 10000 == 0:
            print(f"  Artigos: {i+1}/{max_items} | Pares: {len(pares)}")

    print(f"  Total: {len(pares)} pares validos")
    return pares


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Baixar datasets para Keilinks')
    parser.add_argument('--apenas', choices=['alpaca', 'dolly', 'oasst', 'wiki'],
                       help='Baixar apenas um dataset')
    parser.add_argument('--limite', type=int, default=None,
                       help='Limite de pares por dataset')
    parser.add_argument('--sem-wiki', action='store_true',
                       help='Pular Wikipedia (demora muito)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Keilinks — Download de Datasets")
    print("  Destino: dados/conversas.txt + MySQL knowledge")
    print("=" * 60)

    todos_pares = []

    if args.apenas:
        funcoes = {
            'alpaca': baixar_alpaca,
            'dolly': baixar_dolly,
            'oasst': baixar_oasst,
            'wiki': baixar_wiki_pt,
        }
        pares = funcoes[args.apenas](args.limite)
        todos_pares.extend(pares)
    else:
        # Baixa tudo
        todos_pares.extend(baixar_alpaca(args.limite))
        todos_pares.extend(baixar_dolly(args.limite))
        todos_pares.extend(baixar_oasst(args.limite))
        if not args.sem_wiki:
            todos_pares.extend(baixar_wiki_pt(args.limite))

    if not todos_pares:
        print("\n  Nenhum par valido baixado!")
        return

    # ─── Deduplicacao ────────────────────────────────────────────────────
    print(f"\n  Deduplicando {len(todos_pares)} pares...")
    visto = set()
    unicos = []
    for par in todos_pares:
        # Extrai pergunta pra deduplicar
        match = re.search(r'<vitor>(.+?)<fim>', par)
        if match:
            chave = match.group(1).lower().strip()[:100]
            if chave not in visto:
                visto.add(chave)
                unicos.append(par)
        else:
            unicos.append(par)

    print(f"  {len(todos_pares)} -> {len(unicos)} (removidos {len(todos_pares) - len(unicos)} duplicatas)")

    # ─── Salvar backup ───────────────────────────────────────────────────
    backup_path = os.path.join('dados', 'datasets_baixados.txt')
    os.makedirs('dados', exist_ok=True)
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unicos))
    tamanho_mb = os.path.getsize(backup_path) / 1e6
    print(f"  Backup: {backup_path} ({tamanho_mb:.1f} MB)")

    # ─── Merge com conversas.txt existente ───────────────────────────────
    conversas_path = os.path.join('dados', 'conversas.txt')
    existentes = set()

    if os.path.exists(conversas_path):
        with open(conversas_path, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        # Extrai perguntas existentes pra evitar duplicatas
        for match in re.finditer(r'<vitor>(.+?)<fim>', conteudo):
            existentes.add(match.group(1).lower().strip()[:100])
        print(f"  Conversas existentes: {len(existentes)} pares")

    novos = []
    for par in unicos:
        match = re.search(r'<vitor>(.+?)<fim>', par)
        if match:
            chave = match.group(1).lower().strip()[:100]
            if chave not in existentes:
                novos.append(par)
        else:
            novos.append(par)

    if novos:
        with open(conversas_path, 'a', encoding='utf-8') as f:
            f.write('\n' + '\n'.join(novos))
        print(f"  Adicionados {len(novos)} novos pares ao conversas.txt")
    else:
        print("  Nenhum par novo (tudo ja existia)")

    tamanho_total = os.path.getsize(conversas_path) / 1e6
    total_pares = len(existentes) + len(novos)
    print(f"\n{'=' * 60}")
    print(f"  CONCLUIDO")
    print(f"  Pares totais: ~{total_pares:,}")
    print(f"  conversas.txt: {tamanho_total:.1f} MB")
    print(f"  Backup em: {backup_path}")
    print(f"\n  Proximo passo: retreinar os modelos")
    print(f"  python treino/treinar.py --modelo flash")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
