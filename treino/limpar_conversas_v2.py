"""
Limpeza profunda do conversas.txt
Remove: inglês, Wikipedia, chatbot genérico, linhas sem formato, lixo traduzido
Mantém: conversas naturais estilo Keilinks
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERSAS = os.path.join(BASE_DIR, 'dados', 'conversas.txt')
SAIDA = os.path.join(BASE_DIR, 'dados', 'conversas_limpo.txt')

# Frases que indicam chatbot genérico traduzido
FRASES_CHATBOT = [
    'como um modelo de linguagem', 'como posso ajudá-lo', 'como posso ajuda-lo',
    'fico feliz em ajudar', 'certamente!', 'com prazer', 'não hesite em',
    'estou aqui para ajudar', 'sinto muito, mas como', 'como um assistente',
    'esta tarefa requer', 'não pode ser feita por um modelo',
    'como assistente de ia', 'i am an ai', 'as an ai model',
    'feliz em ajudá-lo', 'posso ajudá-lo', 'vou ficar feliz em',
    'certamente, aqui está', 'certamente! aqui', 'claro! aqui está',
    'como uma inteligência artificial', 'não tenho capacidade de',
    'infelizmente, como uma ia', 'sinto muito, não posso',
]

# Padrões Wikipedia
WIKI_PATTERNS = [
    'é uma freguesia', 'é um município brasileiro', 'é uma vila portuguesa',
    'é uma cidade portuguesa', 'é uma zona urbana', 'obteve o estatuto',
    'é sede da freguesia', 'censo de 20', 'habitantes (20',
    'é uma localidade', 'é um distrito', 'é uma paróquia',
    'nota: nos anos de', 'foi um político', 'foi um escritor',
    'é uma empresa', 'é um rio', 'é uma ilha',
]

# Palavras inglesas (se 3+ aparecem, é linha em inglês)
ENGLISH_MARKERS = [
    ' the ', ' is a ', ' was a ', ' are ', ' were ', ' this is ',
    ' you can ', ' we can ', ' they are ', ' i am ', ' however,',
    ' therefore', ' furthermore', ' in addition', ' which is ',
    ' has been ', ' will be ', ' would be ', ' should be ',
    ' could be ', ' there is ', ' there are ', ' it is ',
    'person 1:', 'person 2:', 'the following',
]

# Respostas que não são conversa (exercícios traduzidos)
PADROES_EXERCICIO = [
    'transitivo', 'intransitivo', 'a fração', 'a mediana',
    'o equivalente decimal', 'temperatura em fahrenheit',
    'sentimento positivo', 'sentimento negativo', 'o tom das frases',
    'as conjunções na frase', 'a principal ideia da passagem',
]


def eh_ingles(texto):
    t = texto.lower()
    count = sum(1 for m in ENGLISH_MARKERS if m in t)
    return count >= 2


def eh_wikipedia(texto):
    t = texto.lower()
    return any(p in t for p in WIKI_PATTERNS)


def eh_chatbot_generico(texto):
    t = texto.lower()
    return any(f in t for f in FRASES_CHATBOT)


def eh_exercicio_traduzido(texto):
    t = texto.lower()
    return any(p in t for p in PADROES_EXERCICIO)


def eh_resposta_longa_generica(pergunta, resposta):
    """Detecta respostas longas genéricas que não soam como Keilinks"""
    if len(resposta) > 500:
        # Keilinks fala curto. Respostas de 500+ chars são suspeitas
        # a menos que seja tech/código
        if '```' in resposta or 'def ' in resposta or 'import ' in resposta:
            return False  # código é ok
        # Se não tem gíria nenhuma e é longa, provavelmente é tradução
        girias = ['kk', 'vc', 'pq', 'tb', 'po', 'eita', 'mn', 'tmj', 'slk', 'mano']
        if not any(g in resposta.lower() for g in girias):
            return True
    return False


def pergunta_eh_instrucao_generica(pergunta):
    """Detecta perguntas tipo Alpaca/Dolly traduzidas"""
    p = pergunta.lower()
    instrucoes = [
        'crie um', 'crie uma', 'gere um', 'gere uma', 'escreva um', 'escreva uma',
        'projetar uma', 'construa uma', 'compile uma lista', 'categorize o',
        'classifique o', 'reescreva a', 'resuma o', 'dê-me um', 'forneça',
        'identifique o', 'descreva a', 'elabore', 'redija',
        'dado este artigo', 'com base na entrada', 'dado o seguinte',
        'analise o', 'compare e contraste', 'liste as',
    ]
    return any(i in p for i in instrucoes)


def limpar():
    with open(CONVERSAS, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    print(f"Total linhas original: {len(linhas):,}")

    boas = []
    stats = {
        'sem_formato': 0, 'ingles': 0, 'wikipedia': 0,
        'chatbot': 0, 'exercicio': 0, 'instrucao_generica': 0,
        'longa_generica': 0, 'curta_demais': 0, 'mantida': 0,
    }

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        # Deve ter formato <vitor>...<fim><keilinks>...<fim>
        if '<vitor>' not in linha or '<keilinks>' not in linha:
            # Remove <sistema> wrapper se tiver
            if '<sistema>' in linha and '<vitor>' in linha:
                # Extrai o conteúdo sem <sistema>
                match = re.search(r'<vitor>(.+)', linha)
                if match:
                    linha = '<vitor>' + match.group(1)
                else:
                    stats['sem_formato'] += 1
                    continue
            else:
                stats['sem_formato'] += 1
                continue

        # Extrair pergunta e resposta
        try:
            partes = linha.split('<vitor>', 1)[1]
            pergunta = partes.split('<fim>')[0].strip()
            resposta = partes.split('<keilinks>')[1]
            if '<fim>' in resposta:
                resposta = resposta.split('<fim>')[0].strip()
            else:
                resposta = resposta.strip()
        except (IndexError, ValueError):
            stats['sem_formato'] += 1
            continue

        if not pergunta or not resposta:
            stats['sem_formato'] += 1
            continue

        # Resposta muito curta
        if len(resposta) < 5:
            stats['curta_demais'] += 1
            continue

        # Filtros
        texto_completo = pergunta + ' ' + resposta

        if eh_ingles(texto_completo):
            stats['ingles'] += 1
            continue

        if eh_wikipedia(resposta):
            stats['wikipedia'] += 1
            continue

        if eh_chatbot_generico(resposta):
            stats['chatbot'] += 1
            continue

        if eh_exercicio_traduzido(resposta):
            stats['exercicio'] += 1
            continue

        if pergunta_eh_instrucao_generica(pergunta) and eh_resposta_longa_generica(pergunta, resposta):
            stats['instrucao_generica'] += 1
            continue

        if eh_resposta_longa_generica(pergunta, resposta):
            stats['longa_generica'] += 1
            continue

        # Boa! Salva no formato limpo (sem <sistema>, será adicionado depois)
        boas.append(f"<vitor>{pergunta}<fim><keilinks>{resposta}<fim>")
        stats['mantida'] += 1

    # Salva
    with open(SAIDA, 'w', encoding='utf-8') as f:
        f.write('\n'.join(boas) + '\n')

    print(f"\n  === RESULTADO DA LIMPEZA ===")
    print(f"  Sem formato válido: {stats['sem_formato']:,}")
    print(f"  Inglês:             {stats['ingles']:,}")
    print(f"  Wikipedia:          {stats['wikipedia']:,}")
    print(f"  Chatbot genérico:   {stats['chatbot']:,}")
    print(f"  Exercício traduz.:  {stats['exercicio']:,}")
    print(f"  Instrução genérica: {stats['instrucao_generica']:,}")
    print(f"  Resposta longa gen: {stats['longa_generica']:,}")
    print(f"  Curta demais:       {stats['curta_demais']:,}")
    print(f"  ---")
    print(f"  MANTIDAS:           {stats['mantida']:,}")
    print(f"  Salvo em: {SAIDA}")


if __name__ == '__main__':
    limpar()
