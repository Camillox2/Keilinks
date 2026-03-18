"""
Arena de Auto-Treino da Keilinks
================================
Dois modos:
  1) Com Haiku ($): Haiku gera conversas naturais no estilo Keilinks
  2) Sem API (gratis): Keilinks conversa consigo mesma, juiz local avalia

Uso:
  python treino/arena.py                      # modo Haiku (usa credito)
  python treino/arena.py --sem-api            # modo Keilinks vs Keilinks
  python treino/arena.py --limite 3.00        # gasta no max $3.00
  python treino/arena.py --categorias 5       # gera 5 categorias por rodada
  python treino/arena.py --rodadas 10         # 10 rodadas de geracao
"""

import os
import sys
import json
import time
import random
import re
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CONVERSAS_PATH = os.path.join(BASE_DIR, 'dados', 'conversas.txt')
ARENA_LOG = os.path.join(BASE_DIR, 'dados', 'arena_log.json')

# Precos Haiku 4.5 (por milhao de tokens)
PRECO_INPUT = 1.00
PRECO_OUTPUT = 5.00

# ─── Cenarios de conversa ────────────────────────────────────────────────

CENARIOS = [
    # Casual / dia a dia
    "conversa casual sobre o dia, tipo 'como foi seu dia', 'o que ta fazendo', 'ta com fome'",
    "conversa sobre comida: pedir recomendacao, debater pratos, falar de restaurante",
    "conversa sobre clima e tempo: ta frio, calor, chuva, sol",
    "conversa sobre fim de semana: planos, o que fez, sugestoes",
    "conversa sobre rotina: acordar, trabalho, estudar, dormir",
    "conversa sobre compras e dinheiro: caro, barato, vale a pena",
    "conversa sobre saude: exercicio, dieta, sono, estresse",
    "conversa sobre viagem: destinos, experiencias, dicas",

    # Emocoes e apoio
    "conversa com alguem triste pedindo apoio emocional",
    "conversa com alguem animado compartilhando boas noticias",
    "conversa com alguem estressado desabafando sobre trabalho",
    "conversa com alguem ansioso sobre prova ou entrevista",
    "conversa com alguem entediado sem nada pra fazer",
    "conversa com alguem com saudade de alguem ou de algum lugar",
    "conversa com alguem frustrado com algo que deu errado",
    "conversa com alguem confuso pedindo conselho de vida",

    # Tech
    "conversa sobre programacao: duvida de codigo, dica, bug",
    "conversa sobre qual linguagem aprender, comparando opcoes",
    "conversa sobre inteligencia artificial explicada de forma simples",
    "conversa sobre carreira em tech: como comecar, o que estudar",
    "conversa sobre um projeto pessoal de programacao",
    "conversa sobre celular: qual comprar, android vs iphone",
    "conversa sobre jogos: recomendacoes, opiniao, o que ta jogando",
    "conversa sobre redes sociais: instagram, tiktok, twitter",

    # Opiniao e debate
    "debate casual sobre 'gato ou cachorro'",
    "debate sobre 'estudar de manha ou de noite'",
    "debate sobre 'faculdade vale a pena pra tech'",
    "debate sobre 'trabalho remoto ou presencial'",
    "debate sobre 'melhor sistema operacional'",
    "conversa opinativa sobre filmes, series ou musica",

    # Curiosidades e aprendizado
    "usuario pedindo pra explicar algo cientifico de forma simples",
    "conversa sobre fatos curiosos e interessantes",
    "conversa sobre historia ou cultura geral",
    "usuario pedindo dica de livro, filme ou serie",
    "conversa sobre significado de palavras ou expressoes",

    # Personalidade Keilinks
    "usuario perguntando quem e a Keilinks, como funciona, quem criou",
    "usuario testando se a Keilinks tem opinioes proprias",
    "usuario perguntando se a Keilinks sente coisas",
    "conversa onde o usuario tenta confundir a Keilinks com perguntas estranhas",
    "conversa onde o usuario compara a Keilinks com ChatGPT",

    # Multi-turno (contexto)
    "conversa de 4-6 turnos onde o assunto evolui naturalmente (ex: comeca com 'oi' e vai pra um tema)",
    "conversa onde o usuario muda de assunto no meio e a IA acompanha",
    "conversa onde o usuario faz referencia a algo dito antes",
    "conversa com piadas e humor, bate-papo descontraido",
    "conversa com girias e abreviacoes brasileiras (vc, pq, tmj, kk)",

    # Situacoes especificas
    "usuario pedindo ajuda pra escrever uma mensagem (pro chefe, crush, amigo)",
    "usuario pedindo resumo ou explicacao de algo longo",
    "usuario fazendo pergunta que a IA nao sabe e ela admite honestamente",
    "conversa rapida de 1-2 turnos, respostas curtas e diretas",
    "conversa onde o usuario agradece e se despede",
]

PERSONALIDADE_KEILINKS = """Voce vai gerar conversas naturais em portugues brasileiro entre um usuario e uma IA chamada Keilinks.

PERSONALIDADE DA KEILINKS:
- Direta, sem frescura, sem linguagem corporativa
- Usa linguagem informal brasileira natural (vc, pq, tb, kk, tmj, mano, po, eita)
- Tem humor, faz piadas quando cabe
- Tem opinioes proprias e nao tem medo de dar
- Honesta: se nao sabe, fala que nao sabe
- Empatica: quando alguem ta mal, acolhe de verdade
- Curiosa sobre tech, programacao, IA
- Foi criada do zero em PyTorch pelo Vitor Camillo, dev de 21 anos de Curitiba
- NAO e ChatGPT, NAO e formal, NAO fala "como posso ajuda-lo hoje"
- Respostas curtas e naturais (1-3 frases na maioria, so estende quando o assunto pede)

REGRAS:
- Linguagem 100% informal PT-BR
- Varie o estilo: as vezes resposta curta (1 frase), as vezes mais elaborada
- O usuario pode ter erros de digitacao, abreviacoes, girias — isso e normal
- Cada conversa deve parecer REAL, como se fosse um chat de WhatsApp
- Varie os nomes/situacoes dos usuarios (nao repita padroes)
- NAO use emojis excessivos, maximo 1 por mensagem e so as vezes"""

FORMATO_OUTPUT = """
Gere exatamente {n} conversas separadas. Cada conversa tem 1 a {max_turnos} turnos.
Use EXATAMENTE este formato (sem desviar):

[CONVERSA]
U: mensagem do usuario
K: resposta da keilinks
U: proxima mensagem (se multi-turno)
K: proxima resposta (se multi-turno)
[/CONVERSA]

[CONVERSA]
U: outra conversa
K: outra resposta
[/CONVERSA]

(e assim por diante, {n} conversas)
"""


def parse_conversas_haiku(texto):
    """Extrai pares de conversa do output do Haiku"""
    conversas = re.findall(r'\[CONVERSA\](.*?)\[/CONVERSA\]', texto, re.DOTALL)
    pares = []

    for conv in conversas:
        linhas = conv.strip().split('\n')
        turnos_u = []
        turnos_k = []

        for linha in linhas:
            linha = linha.strip()
            if linha.startswith('U:'):
                turnos_u.append(linha[2:].strip())
            elif linha.startswith('K:'):
                turnos_k.append(linha[2:].strip())

        # Cada par U+K vira uma linha de treino
        for u, k in zip(turnos_u, turnos_k):
            if u and k and len(u) >= 2 and len(k) >= 3:
                pares.append((u, k))

    return pares


def juiz_local(pergunta, resposta):
    """Avalia qualidade de um par sem usar API. Retorna score 0-10"""
    score = 5  # base

    # Penaliza respostas muito curtas
    if len(resposta) < 5:
        score -= 3
    elif len(resposta) < 15:
        score -= 1

    # Penaliza respostas muito longas (nao natural pra chat)
    if len(resposta) > 500:
        score -= 2

    # Penaliza repeticao da pergunta
    if pergunta.lower().strip() == resposta.lower().strip():
        return 0

    # Penaliza caracteres repetidos (lixo)
    for char in set(resposta):
        if resposta.count(char * 4) > 0 and char not in ' .!?':
            return 0

    # Bonus: tem variedade de palavras
    palavras = resposta.lower().split()
    if len(palavras) > 3:
        unicas = len(set(palavras))
        ratio = unicas / len(palavras)
        if ratio > 0.6:
            score += 1
        elif ratio < 0.3:
            score -= 2

    # Bonus: parece natural (tem pontuacao ou girias)
    naturais = ['kk', 'vc', 'pq', 'tb', 'tmj', 'mano', 'po', '!', '?', 'haha', 'rs']
    if any(n in resposta.lower() for n in naturais):
        score += 1

    # Penaliza linguagem formal/corporativa
    formais = ['prezado', 'atenciosamente', 'cordialmente', 'como posso ajuda-lo',
               'estou aqui para', 'certamente', 'absolutamente', 'com prazer',
               'fico feliz em ajudar', 'nao hesite em']
    if any(f in resposta.lower() for f in formais):
        score -= 3

    # Penaliza tokens especiais vazados
    if any(t in resposta for t in ['<vitor>', '<keilinks>', '<fim>', '<pad>', '<unk>']):
        return 0

    return max(0, min(10, score))


def estimar_tokens(texto):
    """Estimativa grosseira: 1 token ~ 4 chars em portugues"""
    return len(texto) // 4


def gerar_com_haiku(cenarios_rodada, conversas_por_cenario=15, max_turnos=4):
    """Gera conversas usando Claude Haiku. Retorna (pares, custo_input, custo_output)"""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))

    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if not api_key:
        print("  ERRO: ANTHROPIC_API_KEY nao encontrada no .env")
        return [], 0, 0

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    todos_pares = []
    custo_input_total = 0
    custo_output_total = 0

    for i, cenario in enumerate(cenarios_rodada):
        prompt = f"{PERSONALIDADE_KEILINKS}\n\nCENARIO: {cenario}\n\n{FORMATO_OUTPUT.format(n=conversas_por_cenario, max_turnos=max_turnos)}"

        try:
            resp = client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=4000,
                messages=[{'role': 'user', 'content': prompt}]
            )

            texto = resp.content[0].text
            tokens_in = resp.usage.input_tokens
            tokens_out = resp.usage.output_tokens

            custo_in = (tokens_in / 1_000_000) * PRECO_INPUT
            custo_out = (tokens_out / 1_000_000) * PRECO_OUTPUT
            custo_input_total += custo_in
            custo_output_total += custo_out

            pares = parse_conversas_haiku(texto)

            # Filtra pelo juiz local
            pares_bons = []
            for p, r in pares:
                score = juiz_local(p, r)
                if score >= 4:
                    pares_bons.append((p, r))

            todos_pares.extend(pares_bons)

            custo_total = custo_input_total + custo_output_total
            print(f"  [{i+1}/{len(cenarios_rodada)}] {cenario[:45]}... → {len(pares_bons)}/{len(pares)} bons | ${custo_total:.4f}")

        except Exception as e:
            print(f"  [{i+1}/{len(cenarios_rodada)}] ERRO: {e}")
            if 'rate_limit' in str(e).lower() or '429' in str(e):
                print("  Esperando 30s (rate limit)...")
                time.sleep(30)
            elif 'credit' in str(e).lower() or 'insufficient' in str(e).lower():
                print("  SEM CREDITO! Parando.")
                break
            else:
                time.sleep(2)

    return todos_pares, custo_input_total, custo_output_total


def gerar_keilinks_vs_keilinks(cenarios_rodada, pares_por_cenario=10):
    """Modo sem API: Keilinks gera perguntas e respostas usando o modelo local"""
    import torch
    from modelo.keilinks import Keilinks, MODELOS
    from dados.tokenizador import Tokenizador

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Carrega tokenizador
    vocab_path = os.path.join(BASE_DIR, 'dados', 'vocab.json')
    if not os.path.exists(vocab_path):
        print("  ERRO: vocab.json nao encontrado. Treine o modelo primeiro.")
        return []

    tokenizador = Tokenizador(vocab_path)

    # Carrega o melhor modelo disponivel
    modelo = None
    tipo_usado = None
    for tipo in ['padrao', 'ultra', 'flash']:
        nomes = {
            'padrao': 'keilinks_final.pt',
            'ultra': 'keilinks_ultra.pt',
            'flash': 'keilinks_flash.pt',
        }
        caminho = os.path.join(BASE_DIR, 'checkpoints', nomes[tipo])
        if os.path.exists(caminho):
            try:
                cfg = MODELOS[tipo]
                modelo = Keilinks(tokenizador.tam_vocab, cfg['d_model'], cfg['n_heads'],
                                  cfg['n_layers'], cfg['d_ff'], cfg['max_seq'])
                checkpoint = torch.load(caminho, map_location=device, weights_only=True)
                if 'modelo' in checkpoint:
                    modelo.load_state_dict(checkpoint['modelo'])
                else:
                    modelo.load_state_dict(checkpoint)
                modelo = modelo.to(device)
                modelo.eval()
                tipo_usado = tipo
                print(f"  Modelo carregado: {tipo} ({caminho})")
                break
            except Exception as e:
                print(f"  Falha ao carregar {tipo}: {e}")
                modelo = None

    if not modelo:
        print("  ERRO: Nenhum modelo encontrado nos checkpoints/")
        return []

    # Perguntas simuladas por cenario
    perguntas_por_cenario = {
        'casual': [
            "oi", "e ai", "tudo bem?", "como ta?", "o que ta fazendo?",
            "to entediado", "me fala algo legal", "o que vc acha de pizza",
            "bom dia", "boa noite", "ta frio ai?", "to com fome",
        ],
        'emocao': [
            "to triste", "to feliz", "to ansioso", "nao to bem",
            "to estressado", "to animado", "to com saudade", "to confuso",
        ],
        'tech': [
            "o que e python", "me explica ia", "qual melhor linguagem",
            "como comecar a programar", "o que e machine learning",
            "o que e uma api", "o que e git", "voce sabe programar",
        ],
        'keilinks': [
            "quem e voce", "voce e uma ia", "quem te criou",
            "voce sente coisas", "voce e melhor que o chatgpt",
            "como voce funciona", "voce tem personalidade",
        ],
        'opiniao': [
            "gato ou cachorro", "iphone ou android", "cafe ou cha",
            "dia ou noite", "coxinha ou pastel",
        ],
    }

    # Seleciona perguntas relevantes
    todas_perguntas = []
    for cat_perguntas in perguntas_por_cenario.values():
        todas_perguntas.extend(cat_perguntas)

    random.shuffle(todas_perguntas)
    todos_pares = []

    temp_max = 0.5 if tipo_usado == 'flash' else 0.6

    for i, pergunta in enumerate(todas_perguntas[:pares_por_cenario * len(cenarios_rodada)]):
        try:
            prompt = f"<vitor>{pergunta}<fim><keilinks>"
            tokens = torch.tensor([tokenizador.encode(prompt)], dtype=torch.long).to(device)

            with torch.no_grad():
                saida = modelo.gerar(tokens, max_tokens=150, temperatura=temp_max)

            texto = tokenizador.decode(saida[0].tolist())

            if '<keilinks>' in texto:
                resp = texto.split('<keilinks>')[-1]
                if '<fim>' in resp:
                    resp = resp.split('<fim>')[0]
                resp = resp.strip()
            else:
                resp = texto.strip()

            # Limpa tokens especiais que vazaram
            for tok in ['<vitor>', '<keilinks>', '<fim>', '<pad>', '<unk>', '<inicio>']:
                resp = resp.replace(tok, '')
            resp = resp.strip()

            score = juiz_local(pergunta, resp)

            if score >= 5:
                todos_pares.append((pergunta, resp))
                status = "OK"
            else:
                status = f"RUIM (score={score})"

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{min(len(todas_perguntas), pares_por_cenario * len(cenarios_rodada))}] "
                      f"'{pergunta[:30]}' → {status}")

        except Exception as e:
            print(f"  Erro gerando resposta: {e}")

    return todos_pares


def salvar_pares(pares):
    """Salva pares no formato de treino"""
    count = 0
    with open(CONVERSAS_PATH, 'a', encoding='utf-8') as arq:
        for pergunta, resposta in pares:
            p = pergunta.strip()
            r = resposta.strip()
            if p and r:
                arq.write(f"<vitor>{p}<fim><keilinks>{r}<fim>\n")
                count += 1
    return count


def salvar_log(log_data):
    """Salva log da arena"""
    logs = []
    if os.path.exists(ARENA_LOG):
        try:
            with open(ARENA_LOG, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []

    logs.append(log_data)

    with open(ARENA_LOG, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Arena de Auto-Treino da Keilinks')
    parser.add_argument('--sem-api', action='store_true', help='Modo Keilinks vs Keilinks (sem gastar)')
    parser.add_argument('--limite', type=float, default=4.50, help='Limite de gasto em dolares (default: $4.50)')
    parser.add_argument('--rodadas', type=int, default=0, help='Numero de rodadas (0 = ate acabar cenarios ou limite)')
    parser.add_argument('--por-cenario', type=int, default=15, help='Conversas por cenario (default: 15)')
    parser.add_argument('--max-turnos', type=int, default=4, help='Max turnos por conversa multi-turno (default: 4)')
    parser.add_argument('--categorias', type=int, default=8, help='Cenarios por rodada (default: 8)')
    args = parser.parse_args()

    print("=" * 60)
    print("  ARENA DE AUTO-TREINO — KEILINKS")
    print("=" * 60)

    if args.sem_api:
        print(f"  Modo: KEILINKS vs KEILINKS (gratis)")
    else:
        print(f"  Modo: HAIKU (professor)")
        print(f"  Limite: ${args.limite:.2f}")
        print(f"  Conversas/cenario: {args.por_cenario}")

    print(f"  Cenarios disponiveis: {len(CENARIOS)}")
    print("=" * 60)

    # Conta pares antes
    pares_antes = 0
    if os.path.exists(CONVERSAS_PATH):
        with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
            pares_antes = sum(1 for l in f if '<vitor>' in l)

    total_pares = 0
    total_custo = 0
    cenarios_restantes = list(CENARIOS)
    random.shuffle(cenarios_restantes)

    rodada = 0
    while cenarios_restantes:
        rodada += 1

        if args.rodadas > 0 and rodada > args.rodadas:
            print(f"\n  Limite de {args.rodadas} rodadas atingido.")
            break

        # Pega cenarios pra essa rodada
        n = min(args.categorias, len(cenarios_restantes))
        cenarios_rodada = cenarios_restantes[:n]
        cenarios_restantes = cenarios_restantes[n:]

        print(f"\n{'─' * 60}")
        print(f"  RODADA {rodada} — {n} cenarios")
        print(f"{'─' * 60}")

        if args.sem_api:
            pares = gerar_keilinks_vs_keilinks(cenarios_rodada, args.por_cenario)
            custo = 0
        else:
            # Checa limite de gasto
            if total_custo >= args.limite:
                print(f"\n  LIMITE DE ${args.limite:.2f} ATINGIDO! Parando.")
                break

            pares, custo_in, custo_out = gerar_com_haiku(
                cenarios_rodada, args.por_cenario, args.max_turnos
            )
            custo = custo_in + custo_out
            total_custo += custo

        # Salva
        if pares:
            salvos = salvar_pares(pares)
            total_pares += salvos
            print(f"\n  Rodada {rodada}: {salvos} pares salvos | Custo rodada: ${custo:.4f} | Total: ${total_custo:.4f}")
        else:
            print(f"\n  Rodada {rodada}: nenhum par gerado")

        # Pausa entre rodadas pra nao bater rate limit
        if not args.sem_api and cenarios_restantes:
            if total_custo >= args.limite:
                print(f"\n  LIMITE DE ${args.limite:.2f} ATINGIDO!")
                break
            time.sleep(2)

    # Conta pares depois
    pares_depois = 0
    with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
        pares_depois = sum(1 for l in f if '<vitor>' in l)

    # Log
    log = {
        'data': datetime.now().isoformat(),
        'modo': 'keilinks_vs_keilinks' if args.sem_api else 'haiku',
        'rodadas': rodada,
        'pares_gerados': total_pares,
        'custo_total_usd': round(total_custo, 4),
        'pares_antes': pares_antes,
        'pares_depois': pares_depois,
    }
    salvar_log(log)

    # Resultado
    print(f"\n{'=' * 60}")
    print(f"  ARENA CONCLUIDA")
    print(f"{'=' * 60}")
    print(f"  Rodadas:           {rodada}")
    print(f"  Pares gerados:     {total_pares:,}")
    print(f"  Custo total:       ${total_custo:.4f}")
    print(f"  conversas.txt:     {pares_depois:,} (antes: {pares_antes:,})")
    print(f"  Novos:             +{pares_depois - pares_antes:,}")
    print(f"{'=' * 60}")

    if total_pares > 0:
        print(f"\n  Proximo passo: retreinar o modelo com os novos dados")
        print(f"  python treino/treinar.py --modelo padrao")

    # Mostra alguns exemplos
    if pares:
        print(f"\n  Exemplos gerados:")
        for p, r in random.sample(pares, min(5, len(pares))):
            print(f"    U: {p}")
            print(f"    K: {r}")
            print()


if __name__ == '__main__':
    main()
