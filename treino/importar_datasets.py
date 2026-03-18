"""
Importador de datasets externos para treino da Keilinks
Baixa Alpaca PT-BR, Dolly PT-BR, e outros datasets do HuggingFace
Salva em conversas.txt (formato treino) + knowledge no MySQL

Uso:
  python treino/importar_datasets.py                  # importa tudo
  python treino/importar_datasets.py --apenas alpaca  # so alpaca
  python treino/importar_datasets.py --apenas dolly   # so dolly
"""

import os
import sys
import re
import argparse
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from datasets import load_dataset
from dados.database import get_conn, knowledge_adicionar

CONVERSAS_PATH = os.path.join(BASE_DIR, 'dados', 'conversas.txt')
STATS = {'treino': 0, 'knowledge': 0, 'duplicados': 0, 'erros': 0}


def limpar_texto(texto):
    """Remove lixo, tags HTML, espaços extras"""
    if not texto or not isinstance(texto, str):
        return ''
    texto = re.sub(r'<[^>]+>', '', texto)  # HTML tags
    texto = re.sub(r'https?://\S+', '', texto)  # URLs
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Remove se muito curto ou muito longo
    if len(texto) < 10 or len(texto) > 2000:
        return ''
    return texto


def salvar_par_treino(pergunta, resposta, arquivo):
    """Salva no formato de treino da Keilinks"""
    p = limpar_texto(pergunta)
    r = limpar_texto(resposta)
    if not p or not r:
        return False
    arquivo.write(f"<vitor>{p}<fim><keilinks>{r}<fim>\n")
    return True


def salvar_knowledge_mysql(pergunta, resposta, fonte, categoria='geral'):
    """Salva no MySQL (tabela knowledge)"""
    p = limpar_texto(pergunta)
    r = limpar_texto(resposta)
    if not p or not r:
        return False
    try:
        knowledge_adicionar(p, r, fonte=fonte, categoria=categoria)
        return True
    except Exception:
        return False


# ─── ALPACA PT-BR ───────────────────────────────────────────────────────────

def importar_alpaca(arquivo):
    """Alpaca PT-BR: 52K pares instrução/resposta traduzidos"""
    print("\n[1/5] Baixando Alpaca PT-BR...")

    try:
        ds = load_dataset("dominguesm/alpaca-data-pt-br", split="train")
    except Exception:
        try:
            ds = load_dataset("recogna-nlp/alpaca-ptbr", split="train")
        except Exception as e:
            print(f"  ERRO ao baixar Alpaca: {e}")
            return

    print(f"  {len(ds)} exemplos encontrados")
    count = 0

    for row in ds:
        instrucao = row.get('instruction', '') or ''
        entrada = row.get('input', '') or ''
        saida = row.get('output', '') or ''

        # Monta a pergunta: instrução + input se tiver
        if entrada.strip():
            pergunta = f"{instrucao.strip()} {entrada.strip()}"
        else:
            pergunta = instrucao.strip()

        resposta = saida.strip()
        if not pergunta or not resposta:
            continue

        if salvar_par_treino(pergunta, resposta, arquivo):
            STATS['treino'] += 1
            count += 1

        # Salva no knowledge (amostra — não salva tudo pra não poluir)
        if count % 5 == 0:  # 1 a cada 5 vai pro knowledge
            if salvar_knowledge_mysql(pergunta, resposta, 'alpaca', 'instrucao'):
                STATS['knowledge'] += 1

        if count % 5000 == 0 and count > 0:
            print(f"  ... {count} pares processados")

    print(f"  Alpaca concluído: {count} pares de treino")


# ─── DOLLY PT-BR ────────────────────────────────────────────────────────────

def importar_dolly(arquivo):
    """Dolly traduzido para PT: ~15K pares alta qualidade"""
    print("\n[2/5] Baixando Dolly PT-BR...")

    try:
        ds = load_dataset("Gustrd/dolly-15k-libretranslate-pt", split="train")
    except Exception:
        try:
            ds = load_dataset("pablo-moreira/dolly-15k-pt", split="train")
        except Exception as e:
            print(f"  ERRO ao baixar Dolly: {e}")
            return

    print(f"  {len(ds)} exemplos encontrados")
    count = 0

    for row in ds:
        instrucao = row.get('instruction', '') or row.get('instrucao', '') or ''
        contexto = row.get('context', '') or row.get('contexto', '') or ''
        resposta = row.get('response', '') or row.get('resposta', '') or ''

        if contexto.strip():
            pergunta = f"{instrucao.strip()} Contexto: {contexto.strip()[:300]}"
        else:
            pergunta = instrucao.strip()

        resposta = resposta.strip()
        if not pergunta or not resposta:
            continue

        if salvar_par_treino(pergunta, resposta, arquivo):
            STATS['treino'] += 1
            count += 1

        if count % 3 == 0:  # 1 a cada 3 pro knowledge (dolly tem qualidade alta)
            if salvar_knowledge_mysql(pergunta, resposta, 'dolly', 'instrucao'):
                STATS['knowledge'] += 1

        if count % 5000 == 0 and count > 0:
            print(f"  ... {count} pares processados")

    print(f"  Dolly concluído: {count} pares de treino")


# ─── SQUAD PT-BR (Perguntas e respostas) ────────────────────────────────────

def importar_squad(arquivo):
    """SQuAD traduzido PT: perguntas/respostas baseadas em contexto"""
    print("\n[3/5] Baixando SQuAD PT-BR...")

    try:
        ds = load_dataset("Se77en/squad_v2_pt", split="train")
    except Exception:
        try:
            ds = load_dataset("deep-se/squad-pt", split="train")
        except Exception:
            try:
                ds = load_dataset("Se77ence/squad2-pt", split="train")
            except Exception as e:
                print(f"  ERRO ao baixar SQuAD: {e}")
                return

    print(f"  {len(ds)} exemplos encontrados")
    count = 0

    for row in ds:
        pergunta = row.get('question', '') or ''
        respostas = row.get('answers', {})

        # SQuAD tem respostas em formato especial
        if isinstance(respostas, dict):
            textos = respostas.get('text', [])
            resposta = textos[0] if textos else ''
        elif isinstance(respostas, list) and respostas:
            resposta = respostas[0] if isinstance(respostas[0], str) else ''
        else:
            continue

        if not pergunta.strip() or not resposta.strip():
            continue

        if salvar_par_treino(pergunta, resposta, arquivo):
            STATS['treino'] += 1
            count += 1

        if count % 10 == 0:  # SQuAD tem muitas respostas curtas, amostra menor
            if salvar_knowledge_mysql(pergunta, resposta, 'squad', 'qa'):
                STATS['knowledge'] += 1

        if count % 10000 == 0 and count > 0:
            print(f"  ... {count} pares processados")

    print(f"  SQuAD concluído: {count} pares de treino")


# ─── OPENASSISTANT (conversas multilíngues) ─────────────────────────────────

def importar_oasst(arquivo):
    """OpenAssistant: conversas reais com humanos, filtra PT"""
    print("\n[4/5] Baixando OpenAssistant...")

    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"  ERRO ao baixar OpenAssistant: {e}")
        return

    print(f"  {len(ds)} mensagens encontradas, filtrando PT...")

    # Agrupa por thread
    mensagens = {}
    for row in ds:
        msg_id = row.get('message_id', '')
        parent = row.get('parent_id', None)
        texto = row.get('text', '')
        lang = row.get('lang', '')

        mensagens[msg_id] = {
            'texto': texto,
            'parent': parent,
            'lang': lang,
        }

    # Monta pares pergunta/resposta (parent → child) em português
    count = 0
    for msg_id, msg in mensagens.items():
        parent_id = msg['parent']
        if not parent_id or parent_id not in mensagens:
            continue

        parent = mensagens[parent_id]

        # Filtra: pelo menos um em português
        if parent['lang'] != 'pt' and msg['lang'] != 'pt':
            # Aceita também 'pt-BR'
            if not (parent['lang'] or '').startswith('pt') and not (msg['lang'] or '').startswith('pt'):
                continue

        pergunta = parent['texto'].strip()
        resposta = msg['texto'].strip()

        if not pergunta or not resposta:
            continue

        if salvar_par_treino(pergunta, resposta, arquivo):
            STATS['treino'] += 1
            count += 1

        if count % 2 == 0:  # metade pro knowledge
            if salvar_knowledge_mysql(pergunta, resposta, 'openassistant', 'conversa'):
                STATS['knowledge'] += 1

    print(f"  OpenAssistant (PT) concluído: {count} pares de treino")


# ─── WIKIPEDIA PT-BR (artigos como knowledge) ──────────────────────────────

def importar_wikipedia(arquivo):
    """Wikipedia PT: artigos resumidos como pares Q&A"""
    print("\n[5/5] Baixando Wikipedia PT (amostra)...")

    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
    except Exception as e:
        print(f"  ERRO ao baixar Wikipedia: {e}")
        return

    count = 0
    max_artigos = 50000  # limita pra não demorar demais

    for row in ds:
        titulo = row.get('title', '').strip()
        texto = row.get('text', '').strip()

        if not titulo or not texto:
            continue

        # Pega primeiro parágrafo como resposta
        paragrafos = [p.strip() for p in texto.split('\n') if len(p.strip()) > 50]
        if not paragrafos:
            continue

        primeiro = paragrafos[0][:1000]

        # Cria par pergunta/resposta
        pergunta = f"O que é {titulo}?"
        resposta = primeiro

        if salvar_par_treino(pergunta, resposta, arquivo):
            STATS['treino'] += 1
            count += 1

        # Toda entrada da Wikipedia vai pro knowledge
        if salvar_knowledge_mysql(pergunta, resposta, 'wikipedia_dataset', 'enciclopedia'):
            STATS['knowledge'] += 1

        # Cria perguntas extras se o artigo for grande
        if len(paragrafos) > 2:
            p2 = paragrafos[1][:500]
            perguntas_extras = [
                f"Me fale sobre {titulo}",
                f"Quem é {titulo}?" if any(c.isupper() for c in titulo[1:]) else f"Explique {titulo}",
            ]
            for pe in perguntas_extras:
                if salvar_par_treino(pe, p2, arquivo):
                    STATS['treino'] += 1
                    count += 1

        if count % 5000 == 0 and count > 0:
            print(f"  ... {count} pares (de {max_artigos * 2} max)")

        if count >= max_artigos * 2:
            break

    print(f"  Wikipedia concluído: {count} pares de treino")


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Importa datasets para treino da Keilinks')
    parser.add_argument('--apenas', type=str, default=None,
                        help='Importar apenas: alpaca, dolly, squad, oasst, wikipedia')
    args = parser.parse_args()

    datasets_disponiveis = {
        'alpaca': importar_alpaca,
        'dolly': importar_dolly,
        'squad': importar_squad,
        'oasst': importar_oasst,
        'wikipedia': importar_wikipedia,
    }

    if args.apenas:
        if args.apenas not in datasets_disponiveis:
            print(f"Dataset '{args.apenas}' não existe. Opções: {list(datasets_disponiveis.keys())}")
            return
        selecionados = {args.apenas: datasets_disponiveis[args.apenas]}
    else:
        selecionados = datasets_disponiveis

    # Conta pares existentes
    pares_existentes = 0
    if os.path.exists(CONVERSAS_PATH):
        with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
            pares_existentes = sum(1 for line in f if '<vitor>' in line)

    print("=" * 60)
    print("  Keilinks — Importador de Datasets")
    print(f"  Pares de treino existentes: {pares_existentes:,}")
    print(f"  Datasets a importar: {list(selecionados.keys())}")
    print("=" * 60)

    # Abre arquivo em modo append
    with open(CONVERSAS_PATH, 'a', encoding='utf-8') as arquivo:
        for nome, func in selecionados.items():
            try:
                func(arquivo)
            except Exception as e:
                print(f"  ERRO em {nome}: {e}")
                STATS['erros'] += 1

    # Conta total final
    pares_final = 0
    with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
        pares_final = sum(1 for line in f if '<vitor>' in line)

    print("\n" + "=" * 60)
    print("  RESULTADO FINAL")
    print("=" * 60)
    print(f"  Novos pares de treino:   {STATS['treino']:,}")
    print(f"  Novos fatos no MySQL:    {STATS['knowledge']:,}")
    print(f"  Erros:                   {STATS['erros']}")
    print(f"  Total conversas.txt:     {pares_final:,} pares (antes: {pares_existentes:,})")
    print("=" * 60)
    print("\nPróximo passo: python treino/treinar.py --modelo flash")


if __name__ == '__main__':
    main()
