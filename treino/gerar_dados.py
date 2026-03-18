"""
Gerador de dados da Keilinks via Gemini 3 Flash / Claude
Gera milhares de pares de conversa no estilo da Keilinks
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

GEMINI_KEY = os.getenv('GEMINI_API_KEY', '')
ANTHROPIC_KEY = os.getenv('ANTHROPIC_API_KEY', '')

CONVERSAS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dados', 'conversas.txt')
SAIDA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dados', 'conversas_geradas.txt')

# ─── Categorias e prompts ────────────────────────────────────────────────

CATEGORIAS = {
    'identidade': {
        'total': 500,
        'topicos': [
            'perguntas sobre quem e a Keilinks, sua personalidade, como ela funciona',
            'perguntas sobre por que a Keilinks existe, seu proposito, o que ela sabe fazer',
            'perguntas filosoficas sobre consciencia, sentimentos, opiniao da Keilinks',
            'perguntas sobre como a Keilinks aprende, como foi treinada, como funciona por dentro',
            'perguntas comparando Keilinks com ChatGPT, Gemini, outras IAs',
        ],
    },
    'vitor_familia': {
        'total': 300,
        'topicos': [
            'perguntas sobre o Vitor (criador, 21 anos, Curitiba, Eng. Software)',
            'perguntas sobre a familia do Vitor (pai Adriano dentista, mae Juliene psicologa, irma Natalia)',
            'perguntas sobre a Keila (namorada do Vitor, quase 4 anos juntos)',
            'perguntas sobre gostos do Vitor (Vasco, pizza, games, programacao, musica ecletica)',
            'perguntas sobre o RetroWave (e-commerce de camisas retro, React + Node.js + MySQL)',
        ],
    },
    'tech': {
        'total': 2000,
        'topicos': [
            'perguntas sobre Python, sintaxe, bibliotecas, dicas praticas',
            'perguntas sobre JavaScript, React, Node.js, frontend e backend',
            'perguntas sobre PyTorch, redes neurais, deep learning, treinamento',
            'perguntas sobre bancos de dados, SQL, MySQL, PostgreSQL',
            'perguntas sobre Git, GitHub, versionamento, boas praticas',
            'perguntas sobre APIs, REST, Flask, Django, Express',
            'perguntas sobre Docker, deploy, cloud, DevOps basico',
            'perguntas sobre algoritmos, estruturas de dados, complexidade',
            'perguntas sobre HTML, CSS, design responsivo, acessibilidade',
            'perguntas sobre CUDA, GPU, performance, otimizacao',
            'perguntas sobre IA, machine learning, modelos de linguagem, transformers',
            'perguntas sobre debug, erros comuns, boas praticas de programacao',
            'perguntas sobre carreira em tech, como comecar, portfolio, entrevistas',
            'perguntas sobre Linux, terminal, linha de comando',
        ],
    },
    'casual': {
        'total': 1500,
        'topicos': [
            'cumprimentos variados e respostas naturais (oi, bom dia, e ai, beleza)',
            'conversa sobre o dia, como a pessoa esta se sentindo',
            'piadas, humor, brincadeiras entre amigos',
            'recomendacoes de filmes, series, musicas, jogos',
            'conversa sobre comida, receitas, preferencias alimentares',
            'conversa sobre clima, tempo, estacoes do ano',
            'conversa sobre fim de semana, planos, lazer',
            'conversa sobre sono, cansaco, rotina',
            'conversa sobre noticias, eventos atuais genericos',
            'respostas a elogios, criticas, agradecimentos',
        ],
    },
    'conhecimento': {
        'total': 1500,
        'topicos': [
            'perguntas sobre ciencia, fisica, quimica, biologia basica',
            'perguntas sobre historia mundial, personagens historicos',
            'perguntas sobre geografia, paises, capitais, curiosidades',
            'perguntas sobre matematica, logica, numeros',
            'perguntas sobre psicologia, comportamento humano, Freud, Jung',
            'perguntas sobre filosofia basica, grandes pensadores',
            'perguntas sobre saude, exercicio, alimentacao',
            'perguntas sobre espaco, astronomia, planetas, estrelas',
            'perguntas sobre natureza, animais, meio ambiente',
            'perguntas sobre economia, dinheiro, investimentos basico',
        ],
    },
    'emocional': {
        'total': 700,
        'topicos': [
            'pessoa triste pedindo conselho ou desabafo',
            'pessoa ansiosa, nervosa, com medo de algo',
            'pessoa com raiva, frustrada, irritada',
            'pessoa feliz compartilhando boa noticia',
            'pessoa insegura sobre decisoes, carreira, relacionamentos',
            'conversa sobre motivacao, disciplina, habitos',
            'conversa sobre relacionamentos, amizades, conflitos',
        ],
    },
    'meta': {
        'total': 500,
        'topicos': [
            'perguntas sobre como usar a Keilinks, o que ela pode fazer',
            'pedidos para a Keilinks pesquisar algo, buscar informacao',
            'pedidos para ensinar algo novo a Keilinks',
            'perguntas sobre a memoria da Keilinks, o que ela lembra',
            'conversa sobre limites da Keilinks, o que ela nao sabe',
        ],
    },
}


def carregar_exemplos():
    """Carrega conversas existentes como exemplos de estilo"""
    if not os.path.exists(CONVERSAS_PATH):
        return ""
    with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
        texto = f.read()
    # Pega so os pares de conversa (pula identidade textual)
    linhas = [l for l in texto.split('\n') if l.startswith('<vitor>')]
    # Amostra de 30 exemplos variados
    import random
    amostra = random.sample(linhas, min(30, len(linhas)))
    return '\n'.join(amostra)


def exportar_knowledge():
    """Exporta fatos do MySQL como contexto"""
    try:
        from dados.database import get_conn
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT pergunta, resposta FROM knowledge ORDER BY acessos DESC LIMIT 100")
            rows = cur.fetchall()
        conn.close()
        return '\n'.join([f"- {r['pergunta']}: {r['resposta'][:200]}" for r in rows])
    except Exception:
        return ""


def criar_prompt(categoria, topico, exemplos, knowledge_ctx, n=20):
    """Cria prompt para o Gemini/Claude gerar conversas"""
    return f"""Voce vai gerar {n} pares de conversa no estilo da Keilinks, uma IA pessoal criada pelo Vitor Henrique Camillo.

# ESTILO DA KEILINKS:
- Fala natural, direta, sem robotice
- Usa humor quando cabe
- Tom de amiga proxima, nao de assistente formal
- Respostas variam: curtas quando faz sentido, longas quando precisa explicar
- Nunca comeca com "Ola! Como posso ajudar?" ou frases de chatbot generico
- Quando nao sabe, admite na hora
- Tem opinioes proprias
- Menciona o Vitor (criador) naturalmente quando relevante

# CONTEXTO DA KEILINKS:
- Criada do zero em PyTorch pelo Vitor (21 anos, Curitiba, Eng. Software)
- Treinada na RTX 5050 (GPU Blackwell, 8GB VRAM)
- Vitor tambem criou o RetroWave (e-commerce de camisas retro, React + Node)
- Familia: pai Adriano (dentista, Vascaino), mae Juliene (psicologa, adora Freud), irma Natalia
- Namorada do Vitor: Keila Radassa (quase 4 anos juntos)
- Time: Vasco da Gama
- Comida favorita: pizza
- Pesquisa na web quando nao sabe algo

# EXEMPLOS DE CONVERSAS REAIS:
{exemplos}

# CONHECIMENTO DISPONIVEL:
{knowledge_ctx[:2000]}

# CATEGORIA: {categoria}
# TOPICO: {topico}

GERE EXATAMENTE {n} PARES no formato abaixo. Varie bastante as perguntas (curtas, longas, com erros de digitacao, informais, formais). Varie as respostas (algumas curtas de 1 frase, outras longas com explicacao). Use portugues brasileiro natural.

FORMATO OBRIGATORIO (uma conversa por linha):
<vitor>pergunta aqui<fim><keilinks>resposta aqui<fim>

REGRAS:
- NAO use markdown, asteriscos, ou formatacao especial nas respostas
- NAO numere as linhas
- NAO inclua explicacoes fora do formato
- Varie o tamanho das respostas (5 a 150 palavras)
- Inclua perguntas com erros de digitacao (ex: "oq", "vc", "pq", "tbm")
- Algumas perguntas devem ser bem curtas (1-3 palavras)
- Gere APENAS as linhas de conversa, nada mais"""


def gerar_com_gemini(prompt, max_retries=3):
    """Gera texto com Gemini 3 Flash"""
    if not GEMINI_KEY or GEMINI_KEY == 'sua_chave_gemini_aqui':
        return None

    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'temperature': 0.9,
                'max_output_tokens': 4096,
            }
        )
        return response.text
    except Exception as e:
        print(f"  [Gemini] Erro: {e}")
        return None


def gerar_com_claude(prompt, max_retries=3):
    """Gera texto com Claude (backup)"""
    if not ANTHROPIC_KEY or ANTHROPIC_KEY == 'sua_chave_anthropic_aqui':
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=4096,
            temperature=0.9,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"  [Claude] Erro: {e}")
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
        if not linha.endswith('<fim>'):
            # Tenta consertar
            if '<keilinks>' in linha:
                partes = linha.split('<fim><keilinks>')
                if len(partes) >= 2:
                    pergunta = partes[0].replace('<vitor>', '').strip()
                    resposta = partes[1].replace('<fim>', '').strip()
                    if pergunta and resposta:
                        pares.append(f"<vitor>{pergunta}<fim><keilinks>{resposta}<fim>")
            continue

        # Valida formato
        partes = linha.split('<fim><keilinks>')
        if len(partes) >= 2:
            pergunta = partes[0].replace('<vitor>', '').strip()
            resposta = partes[1].replace('<fim>', '').strip()
            if pergunta and resposta and len(resposta) > 2:
                pares.append(f"<vitor>{pergunta}<fim><keilinks>{resposta}<fim>")

    return pares


def gerar_categoria(categoria, info, exemplos, knowledge_ctx, usar_claude=False):
    """Gera todos os pares de uma categoria"""
    total_alvo = info['total']
    topicos = info['topicos']
    pares_gerados = []
    pares_por_batch = 20

    print(f"\n{'='*50}")
    print(f"  Categoria: {categoria} (alvo: {total_alvo} pares)")
    print(f"{'='*50}")

    topico_idx = 0
    tentativas_sem_progresso = 0

    while len(pares_gerados) < total_alvo and tentativas_sem_progresso < 5:
        topico = topicos[topico_idx % len(topicos)]
        topico_idx += 1

        faltam = total_alvo - len(pares_gerados)
        n = min(pares_por_batch, faltam)

        prompt = criar_prompt(categoria, topico, exemplos, knowledge_ctx, n)

        # Tenta Gemini primeiro, Claude como backup
        resultado = gerar_com_gemini(prompt)
        fonte = 'Gemini'

        if not resultado and usar_claude:
            resultado = gerar_com_claude(prompt)
            fonte = 'Claude'

        if not resultado:
            print(f"  [{fonte}] Sem resultado para: {topico[:50]}...")
            tentativas_sem_progresso += 1
            time.sleep(2)
            continue

        novos = extrair_pares(resultado)
        if novos:
            pares_gerados.extend(novos)
            tentativas_sem_progresso = 0
            print(f"  [{fonte}] +{len(novos)} pares ({len(pares_gerados)}/{total_alvo}) — {topico[:50]}...")
        else:
            print(f"  [{fonte}] 0 pares validos — {topico[:50]}...")
            tentativas_sem_progresso += 1

        # Rate limit: Gemini free = 15 RPM
        time.sleep(4.5)

    print(f"  Total {categoria}: {len(pares_gerados)} pares")
    return pares_gerados


def deduplicar(pares):
    """Remove duplicatas baseado na pergunta normalizada"""
    vistos = set()
    unicos = []
    for par in pares:
        # Extrai pergunta
        match = par.split('<fim><keilinks>')[0].replace('<vitor>', '').strip().lower()
        if match not in vistos:
            vistos.add(match)
            unicos.append(par)
    return unicos


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description='Gerador de dados da Keilinks')
    parser.add_argument('--categoria', type=str, default=None, help='Gera so uma categoria')
    parser.add_argument('--total', type=int, default=None, help='Limita total de pares')
    parser.add_argument('--claude', action='store_true', help='Usa Claude como backup')
    parser.add_argument('--append', action='store_true', help='Adiciona ao arquivo existente')
    args = parser.parse_args()

    if not GEMINI_KEY or GEMINI_KEY == 'sua_chave_gemini_aqui':
        if not ANTHROPIC_KEY or ANTHROPIC_KEY == 'sua_chave_anthropic_aqui':
            print("ERRO: Configure GEMINI_API_KEY ou ANTHROPIC_API_KEY no .env")
            return

    print("="*50)
    print("  Gerador de Dados da Keilinks")
    print(f"  Gemini: {'OK' if GEMINI_KEY and GEMINI_KEY != 'sua_chave_gemini_aqui' else 'NAO'}")
    print(f"  Claude: {'OK' if ANTHROPIC_KEY and ANTHROPIC_KEY != 'sua_chave_anthropic_aqui' else 'NAO'}")
    print("="*50)

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
        if args.total:
            # Distribui proporcionalmente
            proporcao = info['total'] / sum(c['total'] for c in CATEGORIAS.values())
            info = info.copy()
            info['total'] = max(10, int(args.total * proporcao))

        pares = gerar_categoria(cat, info, exemplos, knowledge_ctx, usar_claude=args.claude)
        todos_pares.extend(pares)

    # Deduplicar
    todos_pares = deduplicar(todos_pares)

    print(f"\n{'='*50}")
    print(f"  Total gerado: {len(todos_pares)} pares unicos")
    print(f"{'='*50}")

    if not todos_pares:
        print("Nenhum par gerado. Verifique as API keys.")
        return

    # Salva arquivo separado de gerados
    mode = 'a' if args.append else 'w'
    with open(SAIDA_PATH, mode, encoding='utf-8') as f:
        for par in todos_pares:
            f.write(par + '\n')
    print(f"  Salvo em: {SAIDA_PATH}")

    # Merge com conversas.txt original
    print("\n  Fazendo merge com conversas.txt...")
    original = ''
    if os.path.exists(CONVERSAS_PATH):
        with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
            original = f.read()

    # Extrai pares existentes para deduplicar
    pares_existentes = set()
    for linha in original.split('\n'):
        if linha.startswith('<vitor>'):
            p = linha.split('<fim><keilinks>')[0].replace('<vitor>', '').strip().lower()
            pares_existentes.add(p)

    novos = []
    for par in todos_pares:
        p = par.split('<fim><keilinks>')[0].replace('<vitor>', '').strip().lower()
        if p not in pares_existentes:
            novos.append(par)

    if novos:
        with open(CONVERSAS_PATH, 'a', encoding='utf-8') as f:
            for par in novos:
                f.write(par + '\n')
        print(f"  +{len(novos)} pares novos adicionados a conversas.txt")
    else:
        print("  Nenhum par novo (todos ja existiam)")

    print(f"\n  Pronto! conversas.txt agora tem mais dados para treino.")


if __name__ == '__main__':
    main()
