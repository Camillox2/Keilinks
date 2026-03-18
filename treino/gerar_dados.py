"""
Gerador de dados da Keilinks via Claude Haiku / Gemini Flash
Otimizado pra $5 de credito: ~11.000 pares com Haiku
Uso:
  python treino/gerar_dados.py                    # gera tudo
  python treino/gerar_dados.py --categoria tech   # so uma categoria
  python treino/gerar_dados.py --total 1000       # limita total
"""

import os
import sys
import json
import time
import argparse
import re
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

GEMINI_KEY = os.getenv('GEMINI_API_KEY', '')
ANTHROPIC_KEY = os.getenv('ANTHROPIC_API_KEY', '')

CONVERSAS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dados', 'conversas.txt')
SAIDA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dados', 'conversas_geradas.txt')

# ─── Tracking de custo ──────────────────────────────────────────────────
# Haiku 4.5: $1/M input, $5/M output
CUSTO_INPUT_POR_TOKEN = 1.0 / 1_000_000   # $0.000001
CUSTO_OUTPUT_POR_TOKEN = 5.0 / 1_000_000  # $0.000005
LIMITE_GASTO = 4.80  # para em $4.80 pra nao estourar os $5
gasto_total = 0.0
requests_total = 0

# ─── Categorias: ~11.000 pares otimizado pra $5 ─────────────────────────

CATEGORIAS = {
    # ─── IDENTIDADE DA KEILINKS ──────────────────────────────────────
    'identidade': {
        'total': 500,
        'topicos': [
            'perguntas sobre quem e a Keilinks, nome, personalidade, como funciona',
            'por que a Keilinks existe, proposito, o que sabe fazer',
            'filosofia: consciencia de IA, se ela sente coisas, opiniao propria',
            'como a Keilinks aprende, treino em PyTorch, como funciona por dentro',
            'comparando Keilinks com ChatGPT, Gemini, Copilot, Alexa',
            'limites da Keilinks, o que ela nao sabe, honestidade sobre erros',
            'memoria da Keilinks, o que ela lembra, como armazena informacao',
            'pedidos para se descrever, contar sua historia, sua evolucao',
        ],
    },
    'vitor_familia': {
        'total': 400,
        'topicos': [
            'perguntas sobre o Vitor: 21 anos, Curitiba, Eng. Software, criador da Keilinks',
            'pai Adriano: dentista mestre, Vascaino, pai presente e dedicado',
            'mae Juliene: psicologa renomada, adora Freud e psicanalise',
            'irma Natalia Sofia: se achando na vida, as vezes brava mas boa irma',
            'Keila Radassa: namorada quase 4 anos, historia do Instagram em 6 dias',
            'gostos do Vitor: Vasco, pizza, games, musica ecletica, programacao',
            'RetroWave: e-commerce camisas retro, React + Node.js + MySQL',
            'RTX 5050, setup do Vitor, GPU Blackwell, como treinou a Keilinks',
        ],
    },

    # ─── PROGRAMACAO ─────────────────────────────────────────────────
    'python': {
        'total': 1000,
        'topicos': [
            'sintaxe Python: variaveis, tipos, f-strings, operadores, slicing',
            'estruturas: listas, tuplas, dicts, sets, quando usar cada',
            'funcoes: parametros, retorno, decorators, *args, **kwargs, lambda',
            'POO: classes, heranca, polimorfismo, encapsulamento, dunder methods',
            'comprehensions, generators, itertools, functools',
            'erros: try/except, raise, erros customizados, debugging',
            'modulos: os, sys, json, re, datetime, pathlib, collections',
            'pip, venv, requirements.txt, poetry, gerenciamento de deps',
            'async/await, asyncio, concorrencia vs paralelismo',
            'pandas e numpy basico pra analise de dados',
        ],
    },
    'javascript_web': {
        'total': 1000,
        'topicos': [
            'JS basico: var/let/const, tipos, funcoes, arrow functions, closures',
            'DOM, eventos, querySelector, fetch API, manipulacao de pagina',
            'ES6+: destructuring, spread, template literals, modules, optional chaining',
            'async: promises, async/await, tratamento de erros, Promise.all',
            'React: componentes, hooks (useState, useEffect, useContext), JSX, props',
            'Node.js + Express: rotas, middleware, REST API, autenticacao',
            'TypeScript: tipos, interfaces, generics, por que usar',
            'HTML/CSS moderno: flexbox, grid, responsivo, animacoes, Tailwind',
            'Next.js, Vite, bundlers, SSR vs CSR vs SSG',
            'testes: Jest, Testing Library, debugando frontend e backend',
        ],
    },
    'ia_ml': {
        'total': 1000,
        'topicos': [
            'o que e IA, historia, marcos, diferenca entre IA, ML e deep learning',
            'ML: supervisionado, nao supervisionado, reforco, casos de uso',
            'redes neurais: camadas, ativacao, backpropagation, como aprendem',
            'deep learning: CNNs (imagem), RNNs/LSTM (sequencia), transformers (texto)',
            'transformers: mecanismo de atencao, por que revolucionou NLP',
            'PyTorch: tensores, autograd, nn.Module, DataLoader, treino basico',
            'treinamento: loss, otimizadores (Adam, SGD), learning rate, epochs',
            'overfitting: dropout, regularizacao, data augmentation, early stopping',
            'tokenizacao: BPE, WordPiece, por que importa pra modelos de linguagem',
            'fine-tuning, transfer learning, LoRA, quantizacao, GGUF',
            'GPU e CUDA: por que GPU treina mais rapido, VRAM, tensor cores',
            'projetos praticos: chatbot, classificador, gerador de texto',
        ],
    },
    'devops_banco': {
        'total': 700,
        'topicos': [
            'Git: add, commit, push, pull, merge, rebase, branches, conflitos',
            'GitHub: PRs, issues, actions, CI/CD, code review, open source',
            'Docker: containers, Dockerfile, compose, volumes, networking',
            'SQL: SELECT, JOIN, WHERE, GROUP BY, indices, otimizacao de queries',
            'MySQL vs PostgreSQL vs MongoDB vs Redis — quando usar cada',
            'APIs: REST, GraphQL, autenticacao (JWT, OAuth), versionamento',
            'Linux: terminal, comandos basicos, permissoes, shell script, SSH',
            'deploy: VPS, Vercel, Railway, AWS/GCP basico, Nginx, SSL',
        ],
    },
    'carreira': {
        'total': 500,
        'topicos': [
            'como comecar a programar, por onde iniciar, qual linguagem primeiro',
            'faculdade vs autodidata vs bootcamp, qual caminho escolher',
            'portfolio: projetos que impressionam, GitHub, como se destacar',
            'entrevistas: preparacao, algoritmos, system design, soft skills',
            'junior a senior: crescimento, o que diferencia, quanto tempo',
            'trabalho remoto, home office, freelance, nomade digital',
            'salarios, mercado, areas em alta, frontend vs backend vs fullstack',
            'burnout, sindrome do impostor, saude mental em tech',
        ],
    },

    # ─── CIENCIAS E CONHECIMENTO ─────────────────────────────────────
    'ciencia': {
        'total': 800,
        'topicos': [
            'fisica: gravidade, relatividade, quantica basica, energia, termodinamica',
            'quimica: atomos, tabela periodica, reacoes, quimica do cotidiano',
            'biologia: DNA, evolucao, celulas, corpo humano, ecossistemas',
            'astronomia: planetas, estrelas, buracos negros, Big Bang, vida fora da Terra',
            'matematica: algebra, geometria, probabilidade, estatistica, numeros curiosos',
            'tecnologia: como funciona internet, WiFi, blockchain, computadores',
            'medicina: como funcionam vacinas, doencas comuns, sistema imunologico',
            'curiosidades cientificas: fatos impressionantes, descobertas recentes',
        ],
    },
    'historia_geo': {
        'total': 700,
        'topicos': [
            'historia do Brasil: colonizacao, imperio, republica, ditadura, democracia',
            'historia mundial: Roma, Grecia, Idade Media, revolucoes, guerras',
            'grandes personagens: Einstein, Tesla, Turing, Da Vinci, Curie, Newton',
            'historia da tecnologia: computadores, internet, smartphones, IA',
            'geografia Brasil: estados, regioes, curiosidades, economia de cada regiao',
            'Curitiba: historia, cultura, clima, pontos turisticos, vida na cidade',
            'futebol: historia, Copas do Mundo, grandes jogadores, Vasco da Gama',
            'geografia mundial: paises curiosos, culturas, recordes, populacao',
        ],
    },
    'cultura': {
        'total': 700,
        'topicos': [
            'musica: generos, rock, pop, rap, samba, sertanejo, MPB, artistas',
            'cinema: classicos, generos, diretores, filmes essenciais, Oscar',
            'series: Netflix, HBO, recomendacoes, melhores de todos os tempos',
            'games: historia, generos, classicos, e-sports, industria, recomendacoes',
            'literatura: grandes autores, livros essenciais, brasileiros e mundiais',
            'filosofia: Socrates, Platao, Nietzsche, existencialismo, estoicismo',
            'arte: movimentos artisticos, pinturas famosas, arte digital, design',
        ],
    },

    # ─── PSICOLOGIA E DESENVOLVIMENTO ────────────────────────────────
    'psicologia': {
        'total': 600,
        'topicos': [
            'Freud: inconsciente, psicanalise, id/ego/superego, interpretacao de sonhos',
            'Jung: arquetipos, inconsciente coletivo, sombra, tipos psicologicos',
            'inteligencia emocional: Goleman, 5 pilares, como desenvolver cada um',
            'comportamento: vieses cognitivos, heuristicas, como o cerebro engana',
            'personalidade: Big Five, autoconhecimento, o que te define',
            'neurociencia: neurotransmissores, habitos no cerebro, neuroplasticidade',
            'psicologia positiva: flow, gratidao, resiliencia, felicidade',
            'transtornos: ansiedade, depressao, TDAH, quando procurar ajuda',
        ],
    },
    'desenvolvimento': {
        'total': 600,
        'topicos': [
            'comunicacao: escuta ativa, assertividade, CNV, falar em publico',
            'Dale Carnegie: Como Fazer Amigos, principios praticos, networking',
            'habitos: Habitos Atomicos, Poder do Habito, como criar e quebrar habitos',
            'produtividade: Pomodoro, deep work, GTD, foco, eliminar distracoes',
            'financas: orcamento, investir, poupar, educacao financeira basica',
            'relacionamentos: namoro, amizade, limites, comunicacao, resolver conflitos',
            'Mindset (Carol Dweck): mentalidade de crescimento vs fixa',
            'estoicismo: Marco Aurelio, Seneca, aplicacao moderna, controlar emocoes',
        ],
    },

    # ─── EMOCIONAL ───────────────────────────────────────────────────
    'emocional': {
        'total': 600,
        'topicos': [
            'tristeza: consolo, empatia, o que dizer, acolhimento sem julgamento',
            'ansiedade: como lidar, tecnicas, respiracao, quando e normal vs preocupante',
            'raiva: como controlar, canais saudaveis, comunicacao quando irritado',
            'inseguranca: autoestima, parar de se comparar, amor proprio',
            'motivacao: como encontrar, disciplina vs motivacao, recomecar depois de falhar',
            'solidao: diferenca entre sozinho e solitario, como se conectar',
            'celebrar: pessoa feliz com conquista, boa noticia, vitoria',
            'decisoes dificeis: como decidir, medo de errar, analise de risco',
        ],
    },

    # ─── CASUAL ──────────────────────────────────────────────────────
    'casual': {
        'total': 1200,
        'topicos': [
            'cumprimentos: oi, bom dia, e ai, beleza, fala, salve, opa, eae',
            'despedidas: tchau, ate mais, flw, valeu, boa noite, fui, tmj',
            'como vai: tudo bem?, como ta?, suave?, de boa?',
            'comida: preferencias, receitas, pizza, lanche, restaurante, cozinhar',
            'piadas: humor de programador, trocadilhos, memes, zoeira',
            'recomendacoes: filmes, series, musicas, podcasts, jogos, livros',
            'clima: calor, frio, chuva, sol, estacoes, tempo em Curitiba',
            'rotina: sono, acordar, trabalho, estudo, fim de semana, feriado',
            'opinioes aleatorias: cafe vs cha, gato vs cachorro, praia vs campo',
            'esportes: futebol, resultados, campeonatos, Vasco, opinioes',
            'viagens: destinos, dicas, experiencias, vontade de conhecer',
            'redes sociais: Instagram, X, TikTok, YouTube, opinioes sobre uso',
        ],
    },

    # ─── COTIDIANO ───────────────────────────────────────────────────
    'cotidiano': {
        'total': 700,
        'topicos': [
            'faculdade: dicas de estudo, TCC, provas, vida universitaria, EAD',
            'morando sozinho: cozinhar, limpar, contas, organizacao, dicas',
            'saude: exercicio, academia, corrida, alimentacao, dormir bem',
            'dinheiro no dia a dia: economizar, compras, cartao, pix, financas',
            'tecnologia cotidiana: celular, apps uteis, WiFi, seguranca digital',
            'compras online: dicas, como nao cair em golpe, melhores sites',
            'sustentabilidade: reciclagem, economia de energia e agua, consciencia',
        ],
    },

    # ─── DEBATES E OPINIOES ──────────────────────────────────────────
    'debates': {
        'total': 500,
        'topicos': [
            'IA: vai substituir empregos? e etico? limites, regulamentacao',
            'redes sociais: vicio, saude mental, bolha, cancelamento',
            'educacao: escola tradicional vs moderna, EAD, gamificacao',
            'futuro: como sera 2050, singularidade, colonizar Marte, humanidade',
            'comparacoes tech: iOS vs Android, Mac vs PC, Python vs JS',
            'privacidade: dados, vigilancia, big tech, direito de ser esquecido',
        ],
    },

    # ─── META (USAR A KEILINKS) ──────────────────────────────────────
    'meta': {
        'total': 400,
        'topicos': [
            'como usar a Keilinks, funcionalidades, o que ela sabe fazer',
            'pedir pra pesquisar na web, buscar algo, encontrar informacao',
            'ensinar algo novo, corrigir informacao, dar feedback',
            'pedidos criativos: escrever texto, historia, poema, ideia de projeto',
            'perguntas sobre a tecnologia por tras: PyTorch, transformer, BPE',
            'sugestoes de melhoria, criticas construtivas, elogios',
        ],
    },
}


def carregar_exemplos():
    """Carrega exemplos de estilo (compacto pra economizar tokens)"""
    if not os.path.exists(CONVERSAS_PATH):
        return ""
    with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
        texto = f.read()
    linhas = [l for l in texto.split('\n') if l.startswith('<vitor>')]
    amostra = random.sample(linhas, min(15, len(linhas)))  # 15 exemplos (menos tokens)
    return '\n'.join(amostra)


def exportar_knowledge():
    """Exporta fatos do MySQL como contexto"""
    try:
        from dados.database import get_conn
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT pergunta, resposta FROM knowledge ORDER BY acessos DESC LIMIT 50")
            rows = cur.fetchall()
        conn.close()
        return '\n'.join([f"- {r['pergunta']}: {r['resposta'][:150]}" for r in rows])
    except Exception:
        return ""


def criar_prompt(categoria, topico, exemplos, knowledge_ctx, n=20):
    """Prompt otimizado — mais curto pra economizar tokens de input"""
    return f"""Gere {n} conversas no estilo da Keilinks (IA pessoal do Vitor, criada em PyTorch).

ESTILO: natural, direta, com humor, sem robotice. Tom de amiga, nao assistente. Respostas variam de 5 a 150 palavras. Portugues brasileiro real.

CONTEXTO: Keilinks criada do zero pelo Vitor (21 anos, Curitiba, Eng. Software). RTX 5050, PyTorch. Familia: pai Adriano (dentista, Vascaino), mae Juliene (psicologa, Freud), irma Natalia, namorada Keila (4 anos). Time: Vasco. Comida: pizza.

EXEMPLOS:
{exemplos}

TOPICO: {topico}

FORMATO (1 por linha, nada mais):
<vitor>pergunta<fim><keilinks>resposta<fim>

REGRAS: sem markdown, sem numeros, sem explicacao. Varie perguntas (curtas/longas, com girias tipo "oq","vc","pq","blz"). NAO mencione Vitor em toda resposta. Seja informativo com personalidade."""


def gerar_com_claude(prompt):
    """Gera com Claude Haiku (principal — barato e bom)"""
    global gasto_total, requests_total

    if not ANTHROPIC_KEY or ANTHROPIC_KEY == 'sua_chave_anthropic_aqui':
        return None

    if gasto_total >= LIMITE_GASTO:
        print(f"  [Claude] LIMITE de ${LIMITE_GASTO:.2f} atingido! Parando.")
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=4096,
            temperature=0.9,
            messages=[{'role': 'user', 'content': prompt}],
        )

        # Calcula custo
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        custo = (input_tokens * CUSTO_INPUT_POR_TOKEN) + (output_tokens * CUSTO_OUTPUT_POR_TOKEN)
        gasto_total += custo
        requests_total += 1

        return response.content[0].text
    except Exception as e:
        print(f"  [Claude] Erro: {e}")
        return None


def gerar_com_gemini(prompt):
    """Gera com Gemini Flash (backup gratuito)"""
    if not GEMINI_KEY or GEMINI_KEY == 'sua_chave_gemini_aqui':
        return None
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={'temperature': 0.9, 'max_output_tokens': 4096}
        )
        return response.text
    except Exception:
        return None


def extrair_pares(texto_gerado):
    """Extrai pares validos do texto gerado"""
    pares = []
    if not texto_gerado:
        return pares

    for linha in texto_gerado.split('\n'):
        linha = linha.strip()
        if not linha.startswith('<vitor>'):
            continue
        if '<fim><keilinks>' not in linha:
            continue

        partes = linha.split('<fim><keilinks>')
        if len(partes) < 2:
            continue

        pergunta = partes[0].replace('<vitor>', '').strip()
        resposta = partes[1]
        if '<fim>' in resposta:
            resposta = resposta.split('<fim>')[0].strip()
        else:
            resposta = resposta.strip()

        if pergunta and resposta and len(resposta) > 2:
            pares.append(f"<vitor>{pergunta}<fim><keilinks>{resposta}<fim>")

    return pares


def gerar_categoria(categoria, info, exemplos, knowledge_ctx):
    """Gera todos os pares de uma categoria"""
    global gasto_total

    total_alvo = info['total']
    topicos = info['topicos']
    pares_gerados = []
    reqs = 0

    print(f"\n{'='*60}")
    print(f"  {categoria.upper()} — alvo: {total_alvo} pares")
    print(f"{'='*60}")

    topico_idx = 0
    falhas = 0

    while len(pares_gerados) < total_alvo and falhas < 8:
        if gasto_total >= LIMITE_GASTO:
            print(f"  LIMITE ${LIMITE_GASTO:.2f} atingido! Parando categoria.")
            break

        topico = topicos[topico_idx % len(topicos)]
        topico_idx += 1

        faltam = total_alvo - len(pares_gerados)
        n = min(20, faltam)

        prompt = criar_prompt(categoria, topico, exemplos, knowledge_ctx, n)

        # Claude Haiku = principal
        resultado = gerar_com_claude(prompt)
        fonte = 'Haiku'

        # Gemini = backup
        if not resultado:
            resultado = gerar_com_gemini(prompt)
            fonte = 'Gemini'

        if not resultado:
            falhas += 1
            print(f"  FALHA #{falhas} — {topico[:40]}...")
            time.sleep(2)
            continue

        novos = extrair_pares(resultado)
        reqs += 1

        if novos:
            pares_gerados.extend(novos)
            falhas = 0
            print(f"  [{fonte}] +{len(novos):>2} ({len(pares_gerados):>4}/{total_alvo}) "
                  f"| ${gasto_total:.3f} | req #{requests_total} — {topico[:35]}...")
        else:
            falhas += 1
            print(f"  [{fonte}] 0 pares — {topico[:40]}...")

        # Rate limit suave (Haiku aguenta mais, mas nao queremos ban)
        time.sleep(1.0)

    print(f"  >> {categoria}: {len(pares_gerados)} pares | {reqs} reqs | ${gasto_total:.3f}")
    return pares_gerados


def deduplicar(pares):
    """Remove duplicatas"""
    vistos = set()
    unicos = []
    for par in pares:
        chave = par.split('<fim><keilinks>')[0].replace('<vitor>', '').strip().lower()
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(par)
    return unicos


def main():
    global gasto_total

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description='Gerador de dados da Keilinks')
    parser.add_argument('--categoria', type=str, default=None, help='Gera so uma categoria')
    parser.add_argument('--total', type=int, default=None, help='Limita total de pares')
    parser.add_argument('--append', action='store_true', help='Adiciona ao existente')
    parser.add_argument('--limite', type=float, default=4.80, help='Limite de gasto em USD')
    args = parser.parse_args()

    global LIMITE_GASTO
    LIMITE_GASTO = args.limite

    has_claude = ANTHROPIC_KEY and ANTHROPIC_KEY != 'sua_chave_anthropic_aqui'
    has_gemini = GEMINI_KEY and GEMINI_KEY != 'sua_chave_gemini_aqui'

    if not has_claude and not has_gemini:
        print("ERRO: Configure ANTHROPIC_API_KEY ou GEMINI_API_KEY no .env")
        return

    total_alvo = sum(c['total'] for c in CATEGORIAS.values())

    print("=" * 60)
    print("  GERADOR DE DADOS DA KEILINKS")
    print("=" * 60)
    print(f"  Principal:   {'Claude Haiku ($1/$5 por M tok)' if has_claude else 'NAO'}")
    print(f"  Backup:      {'Gemini Flash (gratis)' if has_gemini else 'NAO'}")
    print(f"  Categorias:  {len(CATEGORIAS)}")
    print(f"  Alvo:        ~{total_alvo:,} pares")
    print(f"  Limite:      ${LIMITE_GASTO:.2f}")
    print(f"  Estimativa:  ~${total_alvo/20 * 0.009:.2f}")
    print("=" * 60)

    exemplos = carregar_exemplos()
    knowledge_ctx = exportar_knowledge()

    todos_pares = []
    categorias = CATEGORIAS

    if args.categoria:
        if args.categoria not in categorias:
            print(f"Categoria invalida. Opcoes: {list(categorias.keys())}")
            return
        categorias = {args.categoria: categorias[args.categoria]}

    for cat, info in categorias.items():
        if gasto_total >= LIMITE_GASTO:
            print(f"\n  LIMITE ${LIMITE_GASTO:.2f} atingido! Parando geracao.")
            break

        if args.total:
            proporcao = info['total'] / sum(c['total'] for c in CATEGORIAS.values())
            info = info.copy()
            info['total'] = max(10, int(args.total * proporcao))

        pares = gerar_categoria(cat, info, exemplos, knowledge_ctx)
        todos_pares.extend(pares)

    # Deduplicar
    antes = len(todos_pares)
    todos_pares = deduplicar(todos_pares)
    dupes = antes - len(todos_pares)

    print(f"\n{'='*60}")
    print(f"  RESULTADO")
    print(f"{'='*60}")
    print(f"  Pares gerados:   {antes:,}")
    print(f"  Duplicatas:      {dupes:,}")
    print(f"  Pares unicos:    {len(todos_pares):,}")
    print(f"  Requests:        {requests_total}")
    print(f"  GASTO TOTAL:     ${gasto_total:.4f}")
    print(f"  Custo/par:       ${gasto_total/max(len(todos_pares),1):.6f}")
    print(f"{'='*60}")

    if not todos_pares:
        print("Nenhum par gerado.")
        return

    # Salva backup
    mode = 'a' if args.append else 'w'
    with open(SAIDA_PATH, mode, encoding='utf-8') as f:
        for par in todos_pares:
            f.write(par + '\n')
    print(f"\n  Backup: {SAIDA_PATH}")

    # Merge com conversas.txt
    print("  Merge com conversas.txt...")
    original = ''
    if os.path.exists(CONVERSAS_PATH):
        with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
            original = f.read()

    existentes = set()
    for linha in original.split('\n'):
        if linha.startswith('<vitor>'):
            p = linha.split('<fim><keilinks>')[0].replace('<vitor>', '').strip().lower()
            existentes.add(p)

    novos = []
    for par in todos_pares:
        p = par.split('<fim><keilinks>')[0].replace('<vitor>', '').strip().lower()
        if p not in existentes:
            novos.append(par)

    if novos:
        with open(CONVERSAS_PATH, 'a', encoding='utf-8') as f:
            for par in novos:
                f.write(par + '\n')
        print(f"  +{len(novos):,} pares novos -> conversas.txt")

    with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
        total_final = sum(1 for l in f if l.startswith('<vitor>'))
    print(f"\n  conversas.txt: {total_final:,} pares TOTAL")
    print(f"\n  PRONTO! Treine agora:")
    print(f"  python treino/treinar.py --modelo flash")
    print(f"  python treino/treinar.py --modelo ultra")


if __name__ == '__main__':
    main()
