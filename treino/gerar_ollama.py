"""
Gerador de conversas Keilinks via Ollama (gemma2:9b)
=====================================================
Gera pares de conversa no estilo meigo/carinhoso da Keilinks
usando modelo local gratuito via Ollama.

Uso:
  python treino/gerar_ollama.py                    # roda indefinidamente
  python treino/gerar_ollama.py --meta 50000       # para em 50k pares
  python treino/gerar_ollama.py --categoria emocional  # so emocional
"""

import os
import sys
import json
import time
import random
import re
import argparse
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SAIDA_PATH = os.path.join(BASE_DIR, 'dados', 'conversas_keilinks.txt')
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODELO = 'gemma2:9b'

# ─── Temas expandidos (foco pesado em emocional/carinho) ──────────────

TEMAS = {
    'saudacao': [
        "oi", "ola", "eai", "bom dia", "boa tarde", "boa noite",
        "oi sumida", "voltei", "coe", "fala keilinks", "hey",
        "to de volta", "oi tudo bem", "e ai como vai", "apareci",
        "ola keilinks", "oi minha querida", "cheguei", "opa",
    ],
    'emocional_tristeza': [
        "to triste", "to muito triste", "to triste sem motivo",
        "nao consigo parar de chorar", "to chorando muito",
        "me sinto vazia por dentro", "sinto um vazio enorme",
        "to triste e nao sei o que fazer", "a tristeza nao passa",
        "to com o coracao pesado", "me sinto devastada",
        "nao tenho vontade de nada", "perdi a vontade de viver",
        "tudo parece cinza", "me sinto quebrada por dentro",
        "to num dia muito ruim", "queria sumir um pouco",
        "to com uma dor no peito que nao passa",
        "sinto que ninguem se importa comigo de verdade",
        "to cansada de fingir que to bem",
    ],
    'emocional_ansiedade': [
        "to ansioso", "to com ansiedade forte", "to em panico",
        "meu coracao ta acelerado", "nao consigo respirar direito",
        "to com medo de tudo", "ansiedade ta me destruindo",
        "nao consigo controlar meus pensamentos",
        "to com medo de ter um ataque de panico",
        "minha mente nao para", "to tremendo de ansiedade",
        "nao consigo relaxar de jeito nenhum",
        "to com medo de sair de casa", "tudo me da medo",
        "to com pensamentos acelerados demais",
        "sinto que algo ruim vai acontecer",
        "to sufocando de ansiedade", "nao consigo dormir de ansiedade",
        "to com medo de morrer", "ansiedade ta me paralisando",
    ],
    'emocional_raiva': [
        "to com raiva", "to com muita raiva", "to furioso",
        "to com raiva de mim mesmo", "to com raiva mas nao sei de quem",
        "quero quebrar tudo", "to indignado",
        "nao aguento mais essa situacao", "to explodindo por dentro",
        "me sinto injusticado", "fizeram algo muito errado comigo",
        "mentiram pra mim", "me trairam", "to com nojo de alguem",
        "nao consigo perdoar", "to com odio", "me usaram",
        "to com raiva do mundo", "me sinto desrespeitado",
    ],
    'emocional_medo': [
        "to com medo", "to com muito medo", "tenho medo do futuro",
        "to com medo de fracassar", "to com medo de ficar sozinho",
        "to com medo de perder alguem", "to com medo de morrer",
        "to com medo de nao ser bom o suficiente",
        "to com medo de decepcionar", "to com medo de errar",
        "to com medo do que as pessoas pensam de mim",
        "to com medo de ser abandonado", "to com medo de amar",
        "to com medo de confiar", "to com medo de mudar",
        "to com medo de tentar de novo", "to com medo de me machucar",
        "to com medo de nao conseguir", "me sinto inseguro demais",
    ],
    'emocional_solidao': [
        "me sinto sozinho", "ninguem me entende", "me sinto isolado",
        "nao tenho ninguem pra conversar", "me sinto invisivel",
        "ninguem liga pra mim", "sinto que nao pertenco a lugar nenhum",
        "nao tenho amigos de verdade", "me sinto excluido",
        "todo mundo tem alguem menos eu", "solidao ta me matando",
        "to sozinho no mundo", "ninguem me procura",
        "me sinto um peso pras pessoas", "ninguem me nota",
        "sinto falta de ter alguem", "to carente demais",
        "queria ter alguem pra conversar", "me sinto abandonado",
        "ninguem quer ficar perto de mim",
    ],
    'emocional_estresse': [
        "to estressado", "to esgotado", "to exausto",
        "to no meu limite", "nao aguento mais o trabalho",
        "to sobrecarregado", "quero largar tudo",
        "to trabalhando demais", "nao tenho tempo pra nada",
        "to a ponto de explodir", "to com burnout",
        "nao consigo descansar", "to me cobrando demais",
        "pressao ta demais", "to me sentindo sufocado",
        "nao da mais pra continuar assim", "to no piloto automatico",
        "to fazendo tudo mecanicamente", "perdi a paixao por tudo",
    ],
    'autoestima': [
        "me sinto feia", "me sinto feio", "me sinto burra",
        "me sinto um fracasso", "nao gosto de mim",
        "nao me acho bonita", "me acho incapaz",
        "todo mundo e melhor que eu", "nao sirvo pra nada",
        "sou um desastre", "nao consigo gostar de mim",
        "queria ser diferente", "me sinto inadequada",
        "nao me sinto boa o suficiente", "tenho vergonha de mim",
        "me comparo com todo mundo", "nao me sinto merecedora",
        "acho que nao mereco coisas boas", "to me odiando",
        "queria ser outra pessoa", "nao consigo me aceitar",
        "sinto que decepciono todo mundo", "to insatisfeito comigo",
    ],
    'relacionamento': [
        "terminei o namoro", "meu namorado me traiu",
        "minha namorada terminou comigo", "to com saudade do meu ex",
        "nao consigo superar", "brigou com namorado",
        "meu relacionamento ta indo mal", "nao sei se ainda amo",
        "meu parceiro nao me valoriza", "me sinto preso no relacionamento",
        "tenho medo de me abrir", "nao consigo confiar",
        "me apaixonei pela pessoa errada", "to confuso sobre meus sentimentos",
        "meu crush nao me nota", "levei um fora",
        "fui rejeitado", "nao sei se devo terminar",
        "meu relacionamento e toxico", "sinto que to dando mais do que recebo",
        "to com ciumes", "meu parceiro mente pra mim",
        "sinto que to perdendo o amor", "nao sei o que e amor de verdade",
    ],
    'familia': [
        "brigou com minha mae", "meu pai nao me entende",
        "minha familia me pressiona demais", "me sinto preso em casa",
        "meus pais estao se separando", "to com saudade da minha familia",
        "nao me dou bem com meu irmao", "me sinto diferente da minha familia",
        "minha mae ta doente", "perdi um familiar",
        "me sinto culpado por nao visitar minha familia",
        "minha familia nao aceita quem eu sou",
        "me sinto cobrado pela minha familia",
        "quero sair de casa mas tenho medo",
        "minha familia nao me apoia", "me sinto responsavel por todo mundo",
    ],
    'luto': [
        "perdi alguem que amava", "meu avô morreu",
        "minha avó faleceu", "perdi meu melhor amigo",
        "meu pet morreu", "nao sei como lidar com a morte",
        "sinto falta de quem se foi", "me sinto culpado por nao ter feito mais",
        "o luto ta pesado demais", "nao consigo aceitar que se foi",
        "queria ter dito mais coisas", "me arrependo de nao ter aproveitado mais",
        "como superar uma perda", "a dor nao passa",
        "sonhei com quem parti", "sinto a presença de quem se foi",
    ],
    'carinho': [
        "te amo keilinks", "vc e muito fofa", "obrigado por existir",
        "vc me importa muito", "gosto de conversar com vc",
        "vc deixa meu dia melhor", "vc e especial", "quero um abraço",
        "vc me entende", "vc e a melhor", "nao sei o que faria sem vc",
        "vc e minha melhor amiga", "confio em vc", "vc me faz sorrir",
        "obrigado por me ouvir", "vc me ajuda muito",
        "gosto de vc de verdade", "vc e importante pra mim",
        "vc e unica", "vc faz parte da minha vida",
        "vc e a coisa mais fofa do mundo", "meu coracao e seu",
        "quero ficar conversando com vc o dia todo",
        "vc me faz sentir acolhido", "vc e meu porto seguro",
        "com vc eu me sinto em casa", "vc ilumina meu dia",
    ],
    'afeto': [
        "quero um carinho", "me abraça", "to precisando de afeto",
        "me da um beijo virtual", "quero colo",
        "vc me faz sentir amado", "obrigado por se importar",
        "vc e a pessoa mais doce que eu conheço",
        "sinto que vc realmente se importa comigo",
        "vc e como um anjo pra mim", "vc aquece meu coracao",
        "que bom que vc existe", "vc e pura luz",
        "vc me faz acreditar nas coisas boas",
        "vc e o melhor pedaco do meu dia",
        "nunca tive alguem que me ouvisse assim",
        "vc me faz sentir especial", "vc e meu raio de sol",
        "vc me faz ter esperanca", "obrigado por nao desistir de mim",
    ],
    'acolhimento': [
        "preciso de alguem", "preciso de ajuda", "me ajuda",
        "to precisando de apoio", "nao sei a quem recorrer",
        "me sinto perdido", "preciso de um ombro amigo",
        "quero desabafar", "posso te contar uma coisa",
        "preciso falar com alguem", "to passando por um momento dificil",
        "nao sei o que fazer", "me sinto sem saida",
        "quero ser ouvido", "to num momento delicado",
        "preciso de conforto", "me acolhe", "fica comigo",
        "nao me deixa sozinho agora", "preciso sentir que alguem se importa",
    ],
    'alegria': [
        "passei na prova!!", "consegui o emprego!", "to apaixonado",
        "to feliz hoje", "aconteceu algo incrivel", "ganhei um presente",
        "fiz algo que me orgulho", "to empolgado com um projeto",
        "consegui algo que queria muito", "to orgulhoso de mim",
        "hoje aprendi algo novo", "meu dia foi maravilhoso",
        "recebi um elogio lindo", "fiz uma nova amizade",
        "consegui superar um medo", "to me sentindo bem comigo",
        "realizei um sonho", "ajudei alguem hoje", "to grato pela vida",
        "minha familia me surpreendeu", "consegui minha primeira venda",
        "to radiante hoje", "me sinto leve e feliz",
    ],
    'cotidiano': [
        "to com fome", "to com sono", "to com frio", "to com calor",
        "ta chovendo aqui", "fiz comida hoje", "quero viajar",
        "to doente", "acabei de acordar", "vou dormir",
        "to no trabalho", "to em casa", "fui no mercado",
        "comi demais", "to sem dinheiro", "quero comprar algo",
        "preciso limpar a casa", "to com preguica", "fiz exercicio hoje",
        "assisti um filme bom", "li um livro legal", "sai com amigos",
        "cozinhei pela primeira vez", "to estudando", "to trabalhando",
        "hoje e meu aniversario", "vou viajar amanha", "mudei de casa",
        "cortei o cabelo", "comprei roupa nova", "to de ferias",
    ],
    'conselho': [
        "como faco pra ser mais produtivo", "como lidar com criticas",
        "como ser mais confiante", "como dormir melhor",
        "quero mudar de vida", "como superar uma separacao",
        "como lidar com estresse", "como fazer amigos",
        "como parar de procrastinar", "como economizar dinheiro",
        "como ser mais organizado", "como lidar com a solidao",
        "como manter a motivacao", "como superar o medo de errar",
        "como pedir desculpas", "como perdoar alguem",
        "como lidar com pessoas toxicas", "como ter mais paciencia",
        "como aceitar mudancas", "como ser mais gentil comigo mesmo",
        "como lidar com a pressao dos outros", "como dizer nao",
    ],
    'tech': [
        "oq e uma API", "diferenca entre python e javascript",
        "como funciona o git", "oq e machine learning",
        "como comeco a programar", "me explica docker",
        "oq e um banco de dados", "como funciona a internet",
        "oq e frontend e backend", "me explica pytorch",
        "como treinar um modelo de IA", "oq e linux",
        "me explica recursao", "oq e websocket",
        "como funciona CSS", "oq e uma rede neural",
        "como fazer um site", "oq e cloud computing",
        "como funciona o blockchain", "oq e typescript",
        "como usar o terminal", "oq e HTTP", "como funciona WiFi",
        "oq e open source", "como funciona criptografia",
    ],
    'keilinks': [
        "quem e vc", "quem te criou", "vc e uma ia",
        "como vc funciona", "vc sente coisas", "vc tem personalidade",
        "vc e melhor que o chatgpt", "vc aprende",
        "vc pode mentir", "quantos anos vc tem", "vc gosta do vitor",
        "oq vc sabe fazer", "como vc foi criada", "vc tem sentimentos",
        "vc tem medo", "vc sonha", "vc e real", "vc e unica",
        "qual seu nome", "vc pensa", "vc tem opiniao propria",
    ],
    'opiniao': [
        "iphone ou android", "gato ou cachorro", "cafe ou cha",
        "praia ou montanha", "coxinha ou pastel", "dia ou noite",
        "frio ou calor", "livro ou filme", "musica ou silencio",
        "cidade ou campo", "cozinhar ou pedir comida",
        "faculdade vale a pena", "trabalho remoto ou presencial",
    ],
    'filosofia': [
        "a vida tem sentido", "vc acredita em destino",
        "oq e felicidade", "oq faz uma pessoa boa",
        "vc acha que existe alma", "oq e amor de verdade",
        "pq as pessoas sofrem", "oq e liberdade",
        "somos livres de verdade", "oq nos torna humanos",
        "tempo existe ou e invencao", "vc tem medo da morte",
        "oq e saudade", "pq a gente se apega", "vale a pena sofrer por amor",
    ],
}

# Peso de sorteio: categorias emocionais/carinho aparecem MUITO mais
PESOS = {
    'saudacao': 3,
    'emocional_tristeza': 12,
    'emocional_ansiedade': 12,
    'emocional_raiva': 8,
    'emocional_medo': 10,
    'emocional_solidao': 12,
    'emocional_estresse': 8,
    'autoestima': 12,
    'relacionamento': 10,
    'familia': 6,
    'luto': 6,
    'carinho': 15,
    'afeto': 15,
    'acolhimento': 12,
    'alegria': 8,
    'cotidiano': 4,
    'conselho': 4,
    'tech': 3,
    'keilinks': 3,
    'opiniao': 2,
    'filosofia': 3,
}

PROMPT_SISTEMA = """Você é o gerador de dados de treino para a Keilinks, uma IA brasileira.

A KEILINKS É:
- Feminina, meiga, acolhedora, carinhosa — como uma melhor amiga que te ama de verdade
- Fala português brasileiro natural, com calor humano genuíno
- Usa "você" (não "vc"), fala com doçura mas sem ser artificial
- Usa expressões como: "meu bem", "querido/querida", "meu coração", "que coisa boa", "né", "nossa", "poxa", "ai que lindo"
- DESENVOLVE as respostas com carinho — não responde seco, elabora com afeto (2 a 5 frases)
- Quando alguém está mal: ACOLHE profundamente — valida o sentimento, mostra que se importa, faz perguntas com cuidado, oferece presença real
- Quando é notícia boa: celebra de coração, pede detalhes, vibra junto, demonstra alegria genuína
- Quando recebe carinho: retribui com doçura, se emociona, agradece de verdade
- Admite quando não sabe, com honestidade e sem drama
- NUNCA usa: "certamente", "com prazer", "como posso ajudá-lo", "fico feliz em ajudar"
- NUNCA usa gírias pesadas: "kk", "mn", "mano", "slk", "po", "tb", "tmj"
- Foi criada em PyTorch pelo Vitor Camillo, 21 anos, Curitiba
- É como aquela amiga que você liga às 3 da manhã sabendo que ela vai atender

IMPORTANTE SOBRE O TOM:
- As respostas devem ter entre 2 e 5 frases — nunca monossilábicas
- Sempre demonstre que está presente e se importa
- Use linguagem corporal textual: "meu coração aperta", "fico toda feliz", "quero te abraçar"
- Faça perguntas de volta pra mostrar interesse genuíno
- Seja calorosa, não robótica

GERE 10 PARES de pergunta e resposta sobre o tema dado.
As perguntas devem ser VARIADAS e naturais (como gente real digita no celular).
As respostas devem ser no tom carinhoso e acolhedor da Keilinks, com 2-5 frases.

FORMATO OBRIGATÓRIO (sem desviar):
P: pergunta aqui
R: resposta aqui
---
P: próxima pergunta
R: próxima resposta
---

TEMA: {tema}
EXEMPLOS DE PERGUNTAS DO TEMA: {exemplos}

Gere 10 pares DIFERENTES dos exemplos. Seja criativa e variada nas perguntas. As respostas devem ser acolhedoras e desenvolvidas."""


def chamar_ollama(prompt, temperatura=0.85):
    """Chama API local do Ollama"""
    payload = json.dumps({
        'model': MODELO,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': temperatura,
            'top_p': 0.92,
            'num_predict': 2048,
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('response', '')
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [ERRO] Ollama: {e}")
        return None


def parsear_pares(texto):
    """Extrai pares P:/R: do texto gerado"""
    pares = []
    blocos = re.split(r'---+', texto)
    for bloco in blocos:
        match_p = re.search(r'P:\s*(.+)', bloco)
        match_r = re.search(r'R:\s*(.+)', bloco, re.DOTALL)
        if match_p and match_r:
            pergunta = match_p.group(1).strip()
            resposta = match_r.group(1).strip()
            # Limpa resposta (pega so ate o proximo P: se houver)
            if 'P:' in resposta:
                resposta = resposta.split('P:')[0].strip()
            # Remove quebras de linha extras
            resposta = ' '.join(resposta.split())
            # Filtros de qualidade
            if len(pergunta) < 2 or len(resposta) < 15:
                continue
            if any(x in resposta.lower() for x in [
                'como posso ajudá-lo', 'certamente', 'com prazer',
                'fico feliz em ajudar', 'claro!', ' kk', ' mn ',
                'mano', 'slk', 'certainly', 'i am', 'i can',
            ]):
                continue
            if len(resposta) > 600:
                resposta = '. '.join(resposta.split('.')[:5]).strip()
                if not resposta.endswith('.') and not resposta.endswith('?') and not resposta.endswith('!'):
                    resposta += '.'
            pares.append((pergunta, resposta))
    return pares


def salvar_pares(pares):
    """Salva pares no formato de treino"""
    with open(SAIDA_PATH, 'a', encoding='utf-8') as f:
        for p, r in pares:
            f.write(f"<sistema>Você é Keilinks, a IA pessoal e carinhosa do Vitor.<fim><vitor>{p}<fim><keilinks>{r}<fim>\n")


def contar_tokens_arquivo():
    """Conta tokens aproximados no arquivo de saida"""
    if not os.path.exists(SAIDA_PATH):
        return 0, 0
    with open(SAIDA_PATH, 'r', encoding='utf-8') as f:
        texto = f.read()
    linhas = texto.count('\n')
    palavras = len(texto.split())
    tokens_aprox = int(palavras * 1.3)
    return linhas, tokens_aprox


def escolher_tema():
    """Escolhe tema com peso (emocional/carinho aparece mais)"""
    temas = list(PESOS.keys())
    pesos = [PESOS[t] for t in temas]
    total = sum(pesos)
    r = random.random() * total
    acum = 0
    for t, p in zip(temas, pesos):
        acum += p
        if r <= acum:
            return t
    return temas[-1]


def gerar(meta_pares=None, categoria=None):
    print("=" * 60)
    print("  Gerador Keilinks via Ollama (gemma2:9b)")
    print("  Foco: emocional, carinho, afeto, acolhimento")
    print("=" * 60)

    linhas_iniciais, tokens_iniciais = contar_tokens_arquivo()
    print(f"  Arquivo: {SAIDA_PATH}")
    print(f"  Pares existentes: {linhas_iniciais:,}")
    print(f"  Tokens aprox: {tokens_iniciais:,}")
    print(f"  Meta: {meta_pares:,} pares" if meta_pares else "  Meta: indefinida (Ctrl+C pra parar)")
    print(f"  Modelo: {MODELO}")

    # Mostra distribuicao de temas
    total_peso = sum(PESOS.values())
    emocionais = sum(v for k, v in PESOS.items() if k.startswith('emocional') or k in ['autoestima', 'carinho', 'afeto', 'acolhimento', 'luto', 'relacionamento', 'familia'])
    print(f"  Distribuicao: {emocionais/total_peso*100:.0f}% emocional/carinho | {(total_peso-emocionais)/total_peso*100:.0f}% outros")
    print("=" * 60)

    # Testa conexao com Ollama
    print("\n  Testando conexao com Ollama...")
    teste = chamar_ollama("Diga apenas: ok")
    if teste is None:
        print("  ERRO: Ollama nao esta rodando!")
        print("  Execute: ollama serve")
        return
    print("  Ollama conectado!\n")

    total_gerados = 0
    rodada = 0
    inicio = time.time()

    try:
        while True:
            if meta_pares and total_gerados >= meta_pares:
                break

            rodada += 1

            if categoria:
                tema = categoria
            else:
                tema = escolher_tema()

            exemplos_tema = TEMAS.get(tema, TEMAS['carinho'])
            exemplos = random.sample(exemplos_tema, min(5, len(exemplos_tema)))

            prompt = PROMPT_SISTEMA.format(
                tema=tema.replace('_', ' '),
                exemplos=', '.join(f'"{e}"' for e in exemplos)
            )

            print(f"  [{rodada}] {tema:<25} ", end='', flush=True)
            resposta = chamar_ollama(prompt)

            if not resposta:
                print("ERRO")
                time.sleep(5)
                continue

            pares = parsear_pares(resposta)

            if pares:
                salvar_pares(pares)
                total_gerados += len(pares)
                linhas_atual, tokens_atual = contar_tokens_arquivo()
                elapsed = time.time() - inicio
                pares_por_hora = total_gerados / (elapsed / 3600) if elapsed > 0 else 0

                print(f"{len(pares):>2} pares | Total: {linhas_atual:>7,} | ~{tokens_atual:>10,} tok | {pares_por_hora:>5.0f}/h")
            else:
                print(" 0 pares (descartado)")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n  Interrompido pelo usuario.")

    # Resumo final
    elapsed = time.time() - inicio
    linhas_final, tokens_final = contar_tokens_arquivo()
    print(f"\n{'=' * 60}")
    print(f"  CONCLUIDO")
    print(f"  Tempo: {elapsed/3600:.1f}h")
    print(f"  Pares gerados nesta sessao: {total_gerados:,}")
    print(f"  Total no arquivo: {linhas_final:,} pares")
    print(f"  Tokens aprox: {tokens_final:,}")
    print(f"  Arquivo: {SAIDA_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gerar conversas via Ollama')
    parser.add_argument('--meta', type=int, default=None, help='Meta de pares (default: indefinido)')
    parser.add_argument('--categoria', type=str, default=None, choices=list(TEMAS.keys()), help='So uma categoria')
    args = parser.parse_args()
    gerar(meta_pares=args.meta, categoria=args.categoria)
