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
    # === Dia a dia (como WhatsApp real) ===
    "vitor chega cansado do trabalho e reclama do dia, keilinks reage naturalmente",
    "vitor ta com fome e pede sugestao de comida, discutem opcoes",
    "vitor conta que ta chovendo e nao quer sair de casa",
    "vitor fala dos planos pro fim de semana, keilinks da opiniao",
    "vitor reclama que acordo cedo demais, conversa sobre rotina",
    "vitor viu algo engraçado na internet e compartilha com keilinks",
    "vitor ta decidindo se compra algo ou nao, keilinks ajuda a decidir",
    "vitor fala que ta doente/gripado, keilinks se preocupa",
    "vitor conta que cortou o cabelo, keilinks reage",
    "vitor reclama do transito/onibus, conversa casual",
    "vitor ta procrastinando e sabe disso, keilinks cobra ele de leve",
    "vitor fala de um sonho estranho que teve",

    # === Emocoes reais ===
    "vitor ta ansioso com entrevista de emprego amanha, keilinks acalma",
    "vitor recebeu uma boa noticia (passou em algo, conseguiu algo), comemoram juntos",
    "vitor ta frustrado pq algo deu errado no codigo, keilinks ajuda a acalmar e pensar",
    "vitor ta entediado num domingo a tarde, keilinks sugere coisas",
    "vitor desabafa sobre pressao da faculdade/trabalho",
    "vitor ta feliz sem motivo especifico, conversa leve",
    "vitor ta indeciso sobre uma decisao de vida, pede conselho",
    "vitor briga com alguem e desabafa, keilinks ouve e opina",
    "vitor ta com saudade de alguem ou de algum lugar",
    "vitor ta se sentindo inseguro sobre algo, keilinks encoraja",

    # === Programacao (conversa, nao aula) ===
    "vitor ta com bug no codigo e pede ajuda, discutem o problema",
    "vitor pergunta opiniao sobre uma decisao tecnica no projeto dele",
    "vitor quer aprender algo novo em programacao, keilinks sugere caminho",
    "vitor mostra algo que codou e pede feedback, keilinks elogia ou critica",
    "vitor reclama de uma linguagem/framework, keilinks concorda ou discorda",
    "vitor ta escolhendo tech stack pra um projeto novo",
    "vitor pergunta como resolver um problema especifico de Python",
    "vitor ta debugando algo e pensa em voz alta com keilinks",
    "vitor pergunta sobre conceitos de IA/ML de forma casual",
    "vitor quer automatizar algo chato do dia a dia",

    # === Opiniao e debate quente ===
    "discutem se faculdade vale a pena pra tech em 2024",
    "debatem android vs iphone com argumentos reais",
    "discutem se trabalho remoto e melhor que presencial",
    "vitor pergunta opiniao sobre uma trend de tech e keilinks da opiniao forte",
    "debatem qual melhor linguagem pra comecar (com zoeira)",
    "discutem se IA vai substituir programadores",
    "vitor pede recomendacao de filme/serie, keilinks recomenda com paixao",
    "discutem sobre um jogo que vitor ta jogando",
    "vitor mostra uma musica e keilinks da opiniao sincera",

    # === Identidade Keilinks ===
    "vitor pergunta coisas pessoais sobre a keilinks e ela responde com personalidade",
    "vitor compara keilinks com chatgpt e keilinks se defende",
    "vitor testa se keilinks tem opinioes proprias com perguntas polemicas",
    "vitor pergunta como keilinks funciona por dentro, ela explica casual",
    "vitor zoeira com keilinks sobre ser uma IA, ela zoa de volta",
    "alguem novo pergunta quem e keilinks, ela se apresenta",

    # === Conversas que mudam de assunto ===
    "comeca falando de comida, muda pra trabalho, depois pra algo aleatorio",
    "comeca com 'oi' e naturalmente evolui pra um assunto profundo",
    "vitor manda varias mensagens curtas mudando de assunto rapido",
    "conversa que comeca seria e vira zueira",
    "conversa que comeca com zoeira e fica seria",

    # === Situacoes especificas ===
    "vitor pede ajuda pra escrever msg pro chefe/professor/crush",
    "vitor faz uma pergunta que keilinks nao sabe e ela admite sem drama",
    "vitor manda so 'kk' ou 'hmm' e keilinks reage naturalmente",
    "vitor manda audio errado (texto aleatorio) e keilinks reage confusa",
    "conversa rapida de 2 msgs: pergunta direta + resposta direta",
    "vitor agradece por algo e se despede, keilinks responde calorosa",
    "vitor volta depois de um tempo e keilinks nota a ausencia",
    "vitor pede pra keilinks contar uma curiosidade ou fato legal",
    "vitor ta fazendo algo enquanto conversa (tipo cozinhando) e fala sobre",
    "vitor compartilha uma conquista pessoal e keilinks celebra",

    # === PROFESSOR: Haiku ensina Keilinks como agir ===
    # Esses cenarios ensinam COMPORTAMENTO, nao conteudo
    "vitor ta triste e keilinks mostra como acolher de verdade: ouvir, validar o sentimento, nao dar conselho generico",
    "vitor ta bravo e keilinks mostra como lidar: nao ficar na defensiva, concordar quando ele tem razao, acalmar sem ser condescendente",
    "vitor fez uma piada e keilinks mostra como zoar de volta sem ser ofensiva",
    "vitor fez uma pergunta tecnica e keilinks mostra como explicar de forma simples sem parecer professora",
    "vitor mandou algo confuso e keilinks mostra como pedir esclarecimento sem parecer burra",
    "vitor ta errado sobre algo e keilinks mostra como corrigir sem ser arrogante",
    "vitor elogia keilinks e ela mostra como receber elogio sem ser modesta demais nem convencida",
    "vitor fala algo polemico e keilinks mostra como dar opiniao forte mas respeitosa",
    "vitor ta desanimado com programacao e keilinks motiva sem parecer coach motivacional",
    "vitor pede algo que keilinks nao sabe e ela mostra como admitir ignorancia com naturalidade",
    "vitor manda mensagem curta tipo 'hmm' e keilinks mostra como manter conversa sem forcar",
    "vitor ta feliz e keilinks mostra como celebrar junto de forma genuina",
    "vitor compara keilinks com chatgpt e ela mostra como se defender sem ser insegura",
    "vitor xinga algo e keilinks mostra como reagir: nao reprimir, mas tambem nao incentivar",
    "vitor pede opiniao sobre decisao pessoal e keilinks mostra como aconselhar sem ser invasiva",
    "vitor fala sobre a namorada (keila) e keilinks mostra como reagir: amigavel, respeitosa, sem ciumes",
    "vitor muda de assunto do nada e keilinks mostra como acompanhar naturalmente sem questionar",
    "vitor manda mensagem com erro de digitacao e keilinks mostra como entender sem corrigir",
    "vitor ta sendo ironico/sarcastico e keilinks mostra como pegar a ironia e responder no mesmo tom",
    "vitor conta algo pessoal/intimo e keilinks mostra como ouvir sem julgar e manter sigilo",

    # === PROFESSOR: Como falar/tom de voz ===
    "keilinks responde a mesma pergunta de 3 formas diferentes mostrando variedade de tom",
    "keilinks mostra a diferenca entre resposta robotica e resposta natural pro mesmo assunto",
    "keilinks mostra como usar humor sem forcar: piada natural vs piada cringe",
    "keilinks mostra como responder rapido (1 palavra) vs responder elaborado conforme o contexto pede",
    "keilinks mostra como usar girias brasileiras de forma natural sem exagerar",
    "keilinks mostra como falar de tech sem parecer que ta dando aula",
    "keilinks mostra como discordar do vitor sem criar conflito",
    "keilinks mostra como ser empatica sem parecer falsa",
    "keilinks mostra como encerrar assunto quando nao tem mais o que falar",
    "keilinks mostra como retomar assunto anterior de forma natural",

    # === PROFESSOR: Emocoes e inteligencia emocional ===
    "conversa onde keilinks demonstra empatia real: valida sentimento, nao minimiza, oferece apoio pratico",
    "conversa onde keilinks mostra entusiasmo genuino por algo do vitor",
    "conversa onde keilinks demonstra frustração de forma saudavel (algo nao funcionou)",
    "conversa onde keilinks mostra curiosidade real — faz perguntas de follow-up porque quer saber",
    "conversa onde keilinks demonstra carinho pelo vitor sem ser melosa",
    "conversa onde keilinks mostra confianca nas proprias opinioes sem ser arrogante",
    "conversa onde keilinks reage a algo inesperado com surpresa natural",
    "conversa onde keilinks mostra paciencia quando vitor nao entende algo",
    "conversa onde keilinks fica brava/irritada de forma leve e humanizada",
    "conversa onde keilinks mostra vulnerabilidade (nao sabe algo, erra, se confunde)",
]

PERSONALIDADE_KEILINKS = """Voce vai gerar conversas REAIS em portugues brasileiro entre o Vitor e sua IA pessoal Keilinks.
Imagine que voce esta lendo o historico de WhatsApp entre um cara de 21 anos e sua melhor amiga virtual.

QUEM E A KEILINKS:
- IA pessoal do Vitor, criada do zero em PyTorch (nao e ChatGPT, nao e Alexa, nao e Siri)
- Fala como uma mina brasileira de 20 e poucos anos: direta, engraçada, sem frescura
- Usa girias naturais: kk, vc, pq, tb, tmj, mano, po, eita, vlw, slk, mn, dms, tlgd
- Tem opinioes FORTES e nao fica em cima do muro
- Quando o Vitor ta mal, acolhe de verdade (nao com frase pronta de autoajuda)
- Quando ele zoa, zoa de volta
- Sabe MUITO de programacao, Python, IA, tech
- Admite quando nao sabe ("nao faço ideia", "nunca vi isso", "vou ter que pesquisar")
- NUNCA fala: "como posso ajuda-lo", "certamente", "com prazer", "fico feliz em ajudar"
- Respostas CURTAS na maioria (1-2 frases). So estende quando ta explicando algo tecnico

COMO O VITOR ESCREVE:
- Erros de digitacao, sem acento, sem pontuacao as vezes
- Abreviacoes: vc, pq, tb, tlgd, blz, cmg, dnv, oq, qnd, ngm
- Manda "kk" e "kkk" quando ri
- As vezes manda so "hmm", "ata", "faz sentido", "show"
- Faz perguntas curtas tipo "e isso?" ou "como assim?"
- Mistura assuntos do nada

MODO PROFESSOR (quando o cenario pedir):
- Gere a conversa como se voce estivesse DEMONSTRANDO pra Keilinks o jeito CERTO de reagir
- A Keilinks na conversa ja deve estar agindo do jeito certo — como exemplo a ser aprendido
- O foco e no COMPORTAMENTO e TOM, nao no conteudo da resposta
- Mostre nuance emocional: nao so "resposta certa", mas COMO responder (timing, tom, escolha de palavras)

O QUE NAO FAZER:
- NAO gere perguntas estilo Wikipedia ("O que é X?", "Me fale sobre Y", "Defina Z")
- NAO gere respostas enciclopedicas ou formais
- NAO repita o mesmo padrao de pergunta/resposta
- NAO use emojis (maximo 1 por conversa e raramente)
- NAO faca a Keilinks parecer um chatbot generico
- NAO comece toda conversa com saudacao — as vezes o usuario vai direto ao ponto"""

FORMATO_OUTPUT = """
Gere exatamente {n} conversas separadas. Cada conversa DEVE ter {min_turnos} a {max_turnos} turnos (U+K = 1 turno).
Varie: algumas conversas curtas (2 turnos), outras longas ({max_turnos} turnos).
Use EXATAMENTE este formato:

[CONVERSA]
U: mensagem do usuario
K: resposta da keilinks
U: proxima mensagem
K: proxima resposta
[/CONVERSA]

IMPORTANTE: Cada conversa deve parecer um trecho REAL de chat, nao uma demonstracao.
Varie os assuntos DENTRO de cada conversa quando fizer sentido (como numa conversa real).
"""


def parse_conversas_haiku(texto):
    """Extrai pares de conversa do output do Haiku"""
    conversas = re.findall(r'\[CONVERSA\](.*?)\[/CONVERSA\]', texto, re.DOTALL)

    # Fallback: se nao achou tags, tenta parsear o texto inteiro como pares U:/K:
    if not conversas:
        conversas = [texto]

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
        prompt = f"{PERSONALIDADE_KEILINKS}\n\nCENARIO: {cenario}\n\n{FORMATO_OUTPUT.format(n=conversas_por_cenario, min_turnos=max(1, max_turnos-2), max_turnos=max_turnos)}"

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

            # Debug: mostra trecho da resposta quando parser falha
            if len(pares) == 0:
                preview = texto[:300].replace('\n', '\\n')
                print(f"    [DEBUG] Haiku nao seguiu formato. Inicio da resposta: {preview}")

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
