"""
Crawler de Livros e Conversacao da Keilinks
Busca conteudo sobre como conversar, autoajuda e psicologia.
Fontes:
  - Wikipedia (artigos sobre comunicacao, psicologia, livros famosos)
  - Open Library (resumos de livros)
  - Dataset interno de padroes de conversacao
Salva conteudo no MySQL como knowledge para treinar o modelo.
"""

import os
import sys
import re
import time
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dados.database import knowledge_adicionar, knowledge_existe, knowledge_total

HEADERS = {'User-Agent': 'Keilinks/2.0 (IA pessoal educacional)'}


def _log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [Livros] {msg}")


# ─── WIKIPEDIA: Artigos sobre conversacao e psicologia ───────────────────

WIKI_ARTIGOS = [
    # Comunicacao e conversacao
    'Comunicacao', 'Comunicacao interpessoal', 'Comunicacao nao violenta',
    'Escuta ativa', 'Empatia', 'Assertividade', 'Dialogo',
    'Linguagem corporal', 'Oratoria', 'Retorica',
    'Inteligencia emocional', 'Inteligencia social',
    'Habilidades sociais', 'Rapport',
    # Psicologia
    'Psicologia', 'Psicologia positiva', 'Psicologia social',
    'Autoestima', 'Resiliencia (psicologia)', 'Motivacao',
    'Ansiedade', 'Emocao', 'Sentimento',
    'Terapia cognitivo-comportamental', 'Psicanalise',
    'Sigmund Freud', 'Carl Jung', 'Carl Rogers',
    'Abraham Maslow', 'Hierarquia de necessidades de Maslow',
    # Autoajuda e desenvolvimento pessoal
    'Autoajuda', 'Desenvolvimento pessoal', 'Coaching',
    'Programacao neurolinguistica', 'Mindfulness',
    'Habito', 'Procrastinacao', 'Autodisciplina',
    # Livros famosos
    'Como Fazer Amigos e Influenciar Pessoas',
    'Os 7 Habitos das Pessoas Altamente Eficazes',
    'Inteligencia Emocional (livro)',
    'O Poder do Habito', 'Sapiens (livro)',
    'O Pequeno Principe', 'O Alquimista (romance)',
]


def buscar_wikipedia_pt(titulo: str) -> dict | None:
    """Busca resumo na Wikipedia PT"""
    try:
        search_url = "https://pt.wikipedia.org/w/api.php"
        params = {
            'action': 'query', 'list': 'search',
            'srsearch': titulo, 'srlimit': 1, 'format': 'json'
        }
        resp = requests.get(search_url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        resultados = resp.json().get('query', {}).get('search', [])
        if not resultados:
            return None

        titulo_real = resultados[0]['title']
        summary_url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(titulo_real)}"
        resp2 = requests.get(summary_url, headers=HEADERS, timeout=10)
        if resp2.status_code != 200:
            return None

        data = resp2.json()
        resumo = data.get('extract', '')
        if len(resumo) < 50:
            return None

        return {
            'titulo': titulo_real,
            'resumo': resumo[:1000],
            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
        }
    except Exception:
        return None


def crawl_wikipedia_conversacao() -> int:
    """Busca artigos de Wikipedia sobre conversacao e psicologia"""
    novos = 0

    for artigo in WIKI_ARTIGOS:
        chave = f"o que e {artigo.lower()}"
        if knowledge_existe(chave):
            continue

        resultado = buscar_wikipedia_pt(artigo)
        if not resultado:
            continue

        knowledge_adicionar(
            chave, resultado['resumo'],
            'wikipedia', 'geral', resultado['url'],
        )
        novos += 1
        time.sleep(0.5)

    if novos > 0:
        _log(f"Wikipedia conversacao: +{novos} artigos")
    return novos


# ─── OPEN LIBRARY: Resumos de livros ────────────────────────────────────

OPENLIB_BUSCAS = [
    'communication skills',
    'emotional intelligence',
    'self help conversation',
    'psychology behavior',
    'social skills',
    'assertiveness',
    'empathy',
    'active listening',
    'Dale Carnegie',
    'Daniel Goleman',
]


def buscar_open_library(termo: str, limite: int = 5) -> list[dict]:
    """Busca livros na Open Library"""
    try:
        url = "https://openlibrary.org/search.json"
        params = {'q': termo, 'limit': limite}
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        docs = resp.json().get('docs', [])
        resultados = []

        for doc in docs:
            titulo = doc.get('title', '')
            autor = ', '.join(doc.get('author_name', [])[:2])

            # Pega primeira frase se disponivel
            first = doc.get('first_sentence', [])
            descricao = ''
            if isinstance(first, list) and first:
                descricao = first[0] if isinstance(first[0], str) else ''

            # Tenta pegar descricao do livro
            if not descricao:
                key = doc.get('key', '')
                if key:
                    try:
                        dr = requests.get(f"https://openlibrary.org{key}.json", headers=HEADERS, timeout=8)
                        if dr.status_code == 200:
                            d = dr.json().get('description', '')
                            if isinstance(d, dict):
                                d = d.get('value', '')
                            if d:
                                descricao = d[:800]
                    except Exception:
                        pass

            if titulo and descricao and len(descricao) > 30:
                resultados.append({
                    'titulo': titulo,
                    'autor': autor,
                    'descricao': descricao,
                })

        return resultados
    except Exception as e:
        _log(f"Erro Open Library: {e}")
        return []


def crawl_open_library() -> int:
    """Crawla resumos de livros da Open Library"""
    novos = 0

    for termo in OPENLIB_BUSCAS:
        livros = buscar_open_library(termo, limite=3)

        for livro in livros:
            chave = f"sobre o livro {livro['titulo'].lower()[:80]}"
            if knowledge_existe(chave):
                continue

            texto = livro['descricao']
            if livro['autor']:
                texto = f"Livro de {livro['autor']}. {texto}"

            knowledge_adicionar(chave, texto, 'web', 'geral')
            novos += 1

        time.sleep(1)

    if novos > 0:
        _log(f"Open Library: +{novos} resumos de livros")
    return novos


# ─── DATASET DE CONVERSACAO ──────────────────────────────────────────────

CONHECIMENTO_CONVERSACAO = [
    # Como conversar
    ("como conversar melhor com pessoas",
     "Para conversar melhor: 1) Faca perguntas abertas em vez de sim/nao. 2) Escute de verdade antes de responder. 3) Mostre interesse genuino pelo outro. 4) Use o nome da pessoa. 5) Nao interrompa. 6) Compartilhe experiencias pessoais pra criar conexao. 7) Evite julgar. 8) Mantenha contato visual. A melhor conversa e quando os dois se sentem ouvidos."),

    ("como iniciar uma conversa",
     "Formas de iniciar conversa: 1) Comente algo do ambiente ('que lugar legal esse'). 2) Faca um elogio sincero. 3) Pergunte a opiniao da pessoa sobre algo. 4) Fale sobre algo em comum (evento, lugar, situacao). 5) Use humor leve. 6) Pergunte 'como voce conheceu X?' em festas. O segredo e ser natural e mostrar curiosidade real pela pessoa."),

    ("como manter uma conversa interessante",
     "Pra manter conversa fluindo: 1) Use a tecnica do 'sim, e...' - concorde e adicione algo. 2) Faca perguntas de acompanhamento ('e como foi isso?'). 3) Conte historias com comeco, meio e fim. 4) Varie os assuntos. 5) Compartilhe vulnerabilidades na medida certa. 6) Ria junto. 7) Evite monologo - equilibre falar e ouvir. 8) Demonstre entusiasmo quando algo te interessa."),

    ("como ser mais social",
     "Pra ser mais social: 1) Comece pequeno - cumprimente mais pessoas. 2) Aceite convites mesmo quando der preguica. 3) Pratique em ambientes seguros (trabalho, escola). 4) Lembre que todo mundo tem insegurancas. 5) Foque no outro, nao em si mesmo. 6) Pare de se comparar. 7) Seja voce mesmo - autenticidade atrai. 8) Socializar e habilidade, melhora com pratica."),

    ("o que e escuta ativa",
     "Escuta ativa e ouvir com atencao total. Nao e so esperar sua vez de falar. Envolve: 1) Manter contato visual. 2) Acenar com a cabeca. 3) Parafrasear ('entao voce ta dizendo que...'). 4) Fazer perguntas de esclarecimento. 5) Nao julgar. 6) Nao pensar na resposta enquanto o outro fala. 7) Dar espaco pro silencio. Quando alguem se sente realmente ouvido, a conversa muda completamente."),

    ("o que e empatia",
     "Empatia e a capacidade de se colocar no lugar do outro e sentir o que ele sente. Nao e concordar com tudo, e entender. Existem 3 tipos: 1) Cognitiva - entender o ponto de vista. 2) Emocional - sentir junto. 3) Compassiva - entender, sentir e querer ajudar. Pra desenvolver: ouca sem julgar, pergunte 'como voce se sente?', valide emocoes ('faz sentido voce se sentir assim')."),

    ("como lidar com pessoas dificeis",
     "Com pessoas dificeis: 1) Nao leve pro pessoal - geralmente o problema e delas. 2) Mantenha a calma - respirar fundo ajuda. 3) Estabeleca limites claros. 4) Use comunicacao nao violenta ('quando voce faz X, eu sinto Y'). 5) Escolha suas batalhas. 6) Tente entender a perspectiva delas. 7) Se nada funcionar, minimize o contato. 8) Cuide da sua saude mental primeiro."),

    ("como fazer amigos",
     "Pra fazer amigos: 1) Frequente os mesmos lugares (aula, academia, evento). 2) Tome a iniciativa - convide pra um cafe. 3) Seja confiavel e consistente. 4) Mostre interesse pela vida do outro. 5) Esteja presente nos momentos dificeis. 6) Nao force - amizade cresce naturalmente. 7) Seja voce mesmo. 8) Compartilhe interesses. Dale Carnegie ensina: 'Voce faz mais amigos em 2 meses se interessando pelos outros do que em 2 anos tentando fazer os outros se interessarem por voce'."),

    ("o que e inteligencia emocional",
     "Inteligencia emocional (conceito do Daniel Goleman) e a capacidade de reconhecer, entender e gerenciar suas emocoes e as dos outros. Tem 5 pilares: 1) Autoconsciencia - saber o que voce sente. 2) Autorregulacao - controlar impulsos. 3) Motivacao - se mover por objetivos internos. 4) Empatia - entender os outros. 5) Habilidades sociais - se relacionar bem. Pode ser desenvolvida em qualquer idade."),

    ("como lidar com ansiedade social",
     "Ansiedade social e o medo de ser julgado em situacoes sociais. Pra lidar: 1) Aceite que e normal sentir nervosismo. 2) Comece com situacoes menos assustadoras. 3) Foque na outra pessoa, nao em si. 4) Prepare alguns assuntos antes. 5) Chegue cedo em eventos (menos gente). 6) Celebre pequenas vitorias. 7) Lembre: ninguem ta prestando tanta atencao em voce quanto voce acha. 8) Se for muito forte, procure ajuda profissional."),

    ("como ser mais confiante",
     "Confianca se constroi: 1) Faca coisas que te assustam (comece pequeno). 2) Cuide da aparencia - nao por vaidade, por autoestima. 3) Mantenha postura aberta (ombros pra tras, cabeca erguida). 4) Pare de se comparar com outros. 5) Celebre conquistas, por menores que sejam. 6) Aceite erros como aprendizado. 7) Fale devagar e com clareza. 8) A confianca nao e 'eu sei que vou dar certo', e 'eu sei que vou ficar bem mesmo se der errado'."),

    ("como dar conselhos",
     "Pra dar bons conselhos: 1) Primeiro, escute tudo. 2) Pergunte 'voce quer um conselho ou quer desabafar?'. 3) Valide o sentimento antes ('entendo que isso e dificil'). 4) Compartilhe experiencia propria se tiver. 5) Sugira, nao imponha ('talvez voce pudesse...'). 6) Nao minimize o problema. 7) Nao diga 'eu avisei'. 8) Esteja disponivel depois."),

    ("como pedir desculpa de verdade",
     "Pedir desculpa de verdade: 1) Reconheca especificamente o que fez ('desculpa por ter gritado'). 2) Admita que foi errado sem justificativas. 3) Mostre que entende como o outro se sentiu. 4) Diga o que vai fazer diferente. 5) Nao espere perdao imediato. 6) Nao repita o erro. Uma desculpa real nunca tem 'mas' no meio. 'Desculpa, mas voce tambem...' nao e desculpa."),

    ("como elogiar alguem",
     "Elogios que funcionam: 1) Seja especifico ('adorei como voce explicou aquilo') em vez de generico ('voce e legal'). 2) Elogie o esforco, nao so o resultado. 3) Seja sincero - as pessoas sentem quando e falso. 4) Elogie na hora, nao depois. 5) Elogie caracteristicas, nao so aparencia. 6) Elogie em publico e critique em particular."),

    ("como dizer nao sem culpa",
     "Dizer nao: 1) Seja direto e educado ('nao vou conseguir'). 2) Nao invente desculpas elaboradas. 3) Ofereca alternativa se possivel ('nao posso hoje, mas semana que vem?'). 4) Lembre: dizer nao pra algo e dizer sim pra outra coisa. 5) Nao se desculpe demais. 6) Dizer nao e saudavel e necessario. 7) Quem te respeita vai entender. 8) Pratique - fica mais facil com o tempo."),

    ("como lidar com criticas",
     "Lidar com criticas: 1) Respire antes de reagir. 2) Separe critica construtiva de ataque pessoal. 3) Se for construtiva: agradeca e reflita. 4) Se for destrutiva: nao absorva. 5) Pergunte detalhes ('o que especificamente posso melhorar?'). 6) Nao se defenda imediatamente. 7) Lembre que feedback ajuda a crescer. 8) Nem toda opiniao sobre voce e verdade."),

    ("como ser um bom ouvinte",
     "Ser bom ouvinte: 1) Guarde o celular. 2) Olhe nos olhos. 3) Nao interrompa. 4) Faca sons de confirmacao ('hmm', 'entendo'). 5) Resuma o que ouviu ('entao voce ta dizendo que...'). 6) Pergunte como a pessoa se sente, nao so o que aconteceu. 7) Nao compare com sua experiencia imediatamente. 8) Esteja presente - nao pense na resposta enquanto ouve. A maioria das pessoas nao quer solucao, quer ser ouvida."),

    # Resumos de livros famosos
    ("resumo como fazer amigos e influenciar pessoas",
     "Como Fazer Amigos e Influenciar Pessoas (Dale Carnegie, 1936). Principios: 1) Nao critique nem condene. 2) Faca elogios sinceros. 3) Desperte desejo no outro. 4) Interesse-se genuinamente. 5) Sorria. 6) Lembre do nome das pessoas. 7) Seja bom ouvinte. 8) Fale sobre interesses do outro. 9) Faca o outro se sentir importante. 10) Evite discussoes. 11) Respeite opinioes. 12) Admita erros rapidamente. Um dos livros mais influentes sobre relacionamentos humanos."),

    ("resumo inteligencia emocional daniel goleman",
     "Inteligencia Emocional (Daniel Goleman, 1995). O QI nao e tudo - o QE (quociente emocional) importa mais pro sucesso. 5 habilidades: autoconsciencia (conhecer suas emocoes), autorregulacao (controlar impulsos), motivacao interna, empatia (ler emocoes dos outros) e habilidades sociais. Goleman mostra que essas habilidades podem ser aprendidas e desenvolvidas em qualquer idade. Revolucionou a forma como entendemos inteligencia."),

    ("resumo o poder do habito charles duhigg",
     "O Poder do Habito (Charles Duhigg, 2012). Todo habito tem 3 partes: deixa (gatilho), rotina (comportamento) e recompensa. Pra mudar um habito, mantenha a mesma deixa e recompensa mas troque a rotina. Habitos-chave (como exercicio) criam efeito cascata em outras areas. Forca de vontade e como musculo - cansa mas fortalece com uso. Mudanca acontece quando acreditamos que e possivel e temos apoio social."),

    ("resumo mindset carol dweck",
     "Mindset (Carol Dweck, 2006). Existem 2 mentalidades: fixa ('eu nasci assim, nao mudo') e de crescimento ('posso melhorar com esforco'). Pessoas com mindset de crescimento: aceitam desafios, persistem diante de obstaculos, veem esforco como caminho, aprendem com criticas, se inspiram no sucesso dos outros. Elogie o esforco, nao o talento. 'Ainda nao sei' e melhor que 'nao consigo'."),

    ("resumo habitos atomicos james clear",
     "Habitos Atomicos (James Clear, 2018). Mudancas de 1% por dia = 37x melhor em 1 ano. 4 leis pra criar habitos: 1) Torne obvio (deixe visivel). 2) Torne atraente (junte com algo que gosta). 3) Torne facil (regra dos 2 minutos). 4) Torne satisfatorio (recompense-se). Pra quebrar habitos ruins: inverta as leis. Foque em identidade ('sou uma pessoa saudavel') nao em metas ('quero perder 10kg')."),

    ("resumo comunicacao nao violenta marshall rosenberg",
     "Comunicacao Nao Violenta (Marshall Rosenberg, 1999). 4 passos: 1) Observacao sem julgamento ('quando voce chega atrasado' em vez de 'voce e irresponsavel'). 2) Sentimento ('eu fico frustrado'). 3) Necessidade ('porque preciso de organizacao'). 4) Pedido claro ('podemos combinar horarios?'). A CNV transforma conflitos em dialogo. Nao e ser passivo - e ser honesto sem ser agressivo."),
]


def importar_conhecimento_conversacao() -> int:
    """Importa dataset interno de conversacao pro MySQL"""
    novos = 0
    for pergunta, resposta in CONHECIMENTO_CONVERSACAO:
        if knowledge_existe(pergunta):
            continue
        knowledge_adicionar(pergunta, resposta, 'ensino', 'geral')
        novos += 1

    if novos > 0:
        _log(f"Dataset conversacao: +{novos} conhecimentos importados")
    return novos


# ─── CRAWL COMPLETO ──────────────────────────────────────────────────────

def crawl_livros_completo() -> int:
    """Executa crawl completo de todas as fontes de livros"""
    total = 0

    # 1. Dataset interno de conversacao (instantaneo)
    try:
        total += importar_conhecimento_conversacao()
    except Exception as e:
        _log(f"Erro dataset: {e}")

    # 2. Wikipedia sobre conversacao e psicologia
    try:
        total += crawl_wikipedia_conversacao()
    except Exception as e:
        _log(f"Erro Wikipedia: {e}")

    # 3. Open Library resumos de livros
    try:
        total += crawl_open_library()
    except Exception as e:
        _log(f"Erro Open Library: {e}")

    if total > 0:
        _log(f"Total livros: +{total} | Knowledge: {knowledge_total()}")
    return total


if __name__ == '__main__':
    print("Buscando conteudo sobre conversacao, autoajuda e psicologia...\n")
    novos = crawl_livros_completo()
    print(f"\nConcluido: {novos} novos conteudos salvos no MySQL")
    print(f"Total knowledge: {knowledge_total()}")
