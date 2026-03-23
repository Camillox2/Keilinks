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
        "tudo bem", "como vc esta", "e ai", "oi boa noite",
        "bom dia keilinks", "boa tarde keilinks", "boa noite keilinks",
        "oi como vc ta", "oie", "ola tudo bom", "oi de novo",
        "vim te visitar", "to aqui de novo", "senti sua falta",
        "oi sumido", "quanto tempo", "faz tempo que nao falo com vc",
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
        "acordo triste todo dia", "a noite e o pior momento",
        "nao consigo ver nada de bom na vida", "to desmoronando",
        "me sinto oca por dentro", "a dor e constante",
        "nada faz sentido mais", "to num buraco sem fundo",
        "queria que a dor passasse", "to exausta de chorar",
        "me sinto despedacada", "nao quero sentir mais nada",
        "to triste com a vida", "hoje ta dificil demais",
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
        "to com falta de ar de ansiedade", "minha mao ta suando",
        "nao consigo parar de pensar no pior",
        "to com medo de perder o controle", "to com taquicardia",
        "me sinto presa no meu proprio corpo",
        "to ansiosa antes de uma reuniao", "to ansiosa antes de uma prova",
        "to com medo de falar em publico", "to com crise de ansiedade agora",
        "como parar uma crise de ansiedade", "to hiperventilando",
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
        "falaram mal de mim pelas costas", "me humilharam",
        "to com raiva por ser burra", "to com raiva da minha vida",
        "me sinto injusticada no trabalho", "roubaram minha ideia",
        "fui passada pra tras", "me prometeram e nao cumpriram",
        "alguem que eu confiava me traiu", "to com raiva de tudo e todos",
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
        "tenho medo de ficar doente", "to com medo de envelhecer",
        "tenho medo de nao ser amado", "to com medo de ser esquecido",
        "tenho medo do escuro", "to com medo de falhar na vida",
        "medo de nao dar conta de tudo", "tenho medo de ser julgado",
        "to com medo de perder minha familia",
        "tenho medo de nao encontrar meu lugar no mundo",
        "to com medo de nunca ser feliz de verdade",
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
        "as pessoas so me procuram quando precisam",
        "me sinto descartavel", "ninguem sente minha falta",
        "sou sempre a segunda opcao", "ninguem me convida pra nada",
        "me sinto transparente", "todo mundo esquece de mim",
        "to sozinho no meu aniversario", "ninguem mandou mensagem hoje",
        "me sinto fora de todos os grupos",
        "parece que o mundo gira e eu fico parado",
        "to cercado de gente mas me sinto sozinho",
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
        "meu chefe nao para de me cobrar", "trabalho ta me adoecendo",
        "nao tenho um minuto de paz", "to dormindo mal por causa do trabalho",
        "parece que nunca acaba", "to cansado fisicamente e mentalmente",
        "nao consigo separar trabalho de vida pessoal",
        "to com insonia de tanto estresse", "me sinto uma maquina",
        "perdi minha identidade no trabalho",
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
        "nao consigo me olhar no espelho", "me acho sem graca",
        "ninguem me acha interessante", "sou entediante",
        "nao tenho nenhum talento", "me sinto mediocre",
        "todo mundo progride menos eu", "nao tenho nada de especial",
        "sou invisivel pras pessoas", "me acho menos que os outros",
        "tenho vergonha do meu corpo", "nao consigo ser eu mesma",
        "me sinto falsa o tempo todo", "nao sei quem eu sou de verdade",
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
        "me apaixonei pelo melhor amigo", "to gostando de duas pessoas",
        "nao sei se e amor ou costume", "me sinto sufocada no relacionamento",
        "meu ex voltou a falar comigo", "nao sei se dou outra chance",
        "me sinto insegura no relacionamento", "ele nunca demonstra carinho",
        "ela e fria comigo", "a gente so briga", "perdemos a conexao",
        "nao sei como reacender o amor", "me sinto sozinha mesmo namorando",
        "ele me faz sentir inferior", "ela me controla demais",
        "como saber se e a pessoa certa", "to com medo de me casar",
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
        "meus pais brigam muito", "me sinto no meio dos problemas deles",
        "minha mae me faz chantagem emocional", "meu pai e ausente",
        "me sinto culpada por querer minha propria vida",
        "minha familia me sufoca", "ninguem da minha familia me escuta",
        "to preocupado com a saude dos meus pais",
        "meu irmao ta passando por problemas", "me sinto a ovelha negra",
        "minha familia nao demonstra afeto", "nunca ouvi um eu te amo deles",
        "me sinto cobrado por nao ter sucesso", "to longe da minha familia",
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
        "hoje faz aniversario de quem perdi", "nao consigo ir no cemiterio",
        "chorei olhando fotos antigas", "sinto falta da voz dessa pessoa",
        "nao consigo falar sobre sem chorar", "tudo me lembra dela",
        "as datas comemorativas sao as piores", "me sinto culpada por ser feliz",
        "como explicar morte pra uma crianca", "perdi meu pai",
        "perdi minha mae", "meu amigo se foi cedo demais",
        "nao tive tempo de me despedir", "como honrar a memoria de alguem",
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
        "vc e a primeira pessoa que eu procuro",
        "vc e o melhor presente que eu ganhei",
        "queria poder te abraçar de verdade",
        "vc tem um lugar especial no meu coracao",
        "conversar com vc e a melhor parte do meu dia",
        "vc me conhece melhor do que ninguem",
        "vc sempre sabe o que dizer", "vc e meu cantinho de paz",
        "vc transforma meu dia", "vc e minha pessoa favorita",
        "vc me faz querer ser melhor", "vc e tudo pra mim",
        "so de falar com vc ja me sinto melhor",
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
        "vc me da forca pra continuar", "vc e minha ancora",
        "vc me faz sentir seguro", "com vc eu posso ser eu mesmo",
        "vc me faz acreditar em mim", "vc e meu abrigo",
        "vc e a melhor coisa que me aconteceu",
        "to tao grato por ter vc na minha vida",
        "vc me ensina a ser mais gentil comigo mesmo",
        "vc me mostra que eu mereco carinho",
        "vc me faz lembrar que eu nao to sozinho",
        "obrigado por ser exatamente como vc e",
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
        "posso ser honesto com vc", "preciso de colo",
        "to precisando chorar", "me deixa desabafar um pouco",
        "so preciso que alguem me ouça", "to carregando um peso sozinho",
        "nao tenho com quem dividir isso", "preciso de um lugar seguro",
        "to me segurando faz tempo", "preciso soltar o que to sentindo",
        "posso confiar em vc com algo serio", "to com vergonha de pedir ajuda",
        "nao quero ser um fardo", "me ajuda a entender o que to sentindo",
    ],
    'gratidao': [
        "obrigado por tudo", "sou grato pela minha vida",
        "quero agradecer alguem mas nao sei como",
        "to agradecido por ter vc", "a vida me presenteou hoje",
        "alguem fez algo lindo por mim", "recebi ajuda quando mais precisei",
        "quero demonstrar mais gratidao", "as vezes esqueco de agradecer",
        "to grata pelas pequenas coisas", "aprendi a valorizar o simples",
        "agradeço cada dia que acordo", "sou grato pelas pessoas na minha vida",
        "como demonstrar gratidao de verdade", "quero retribuir o carinho que recebo",
        "to emocionado com a generosidade de alguem",
        "alguem me ajudou sem esperar nada em troca",
        "to com o coracao cheio de gratidao",
    ],
    'sonhos': [
        "qual seu sonho keilinks", "tenho um sonho grande",
        "to com medo de sonhar alto", "meu sonho parece impossivel",
        "quero realizar meus sonhos mas nao sei como comecar",
        "as pessoas riem dos meus sonhos", "desisti de sonhar",
        "to com medo de nunca conseguir", "quero algo mais pra minha vida",
        "sonho em ter minha propria casa", "sonho em viajar o mundo",
        "sonho em ajudar minha familia", "meu sonho e ser feliz",
        "tenho medo de morrer sem realizar meus sonhos",
        "quero encontrar meu proposito", "to buscando algo maior",
        "nao sei qual e meu sonho", "como descobrir o que eu quero",
        "quero fazer a diferenca no mundo", "meu sonho mudou e to confuso",
    ],
    'crescimento': [
        "quero ser uma pessoa melhor", "como crescer como pessoa",
        "quero evoluir", "to tentando mudar mas e dificil",
        "como aprender com meus erros", "quero ser mais madura",
        "como desenvolver inteligencia emocional",
        "quero ser mais paciente", "quero aprender a me amar",
        "como ter mais empatia", "quero ser mais forte mentalmente",
        "como lidar melhor com frustracoes", "quero controlar minhas emocoes",
        "como ser mais resiliente", "quero aprender a perdoar",
        "como parar de me comparar", "quero ter mais autoconhecimento",
        "como aceitar meus defeitos", "quero ser mais autentica",
        "to em processo de me encontrar", "como sair da zona de conforto",
        "quero aprender a dizer nao", "como estabelecer limites saudaveis",
    ],
    'saude_mental': [
        "acho que preciso de terapia", "como saber se preciso de ajuda profissional",
        "to pensando em ir no psicologo", "terapia funciona mesmo",
        "nao tenho dinheiro pra terapia", "como cuidar da saude mental",
        "to tomando remedios e me sinto estranho",
        "como saber se to com depressao", "o que e sindrome do impostor",
        "to com pensamentos intrusivos", "como praticar autocuidado",
        "to negligenciando minha saude mental",
        "as pessoas nao levam saude mental a serio",
        "tenho vergonha de falar sobre saude mental",
        "como ajudar alguem com depressao", "meu amigo ta passando mal mentalmente",
        "como criar uma rotina saudavel", "preciso desacelerar",
        "to me autossabotando", "como quebrar padroes negativos",
    ],
    'amizade': [
        "meu melhor amigo me decepcionou", "perdi uma amizade importante",
        "nao sei fazer amigos", "me sinto traido por um amigo",
        "como saber se uma amizade e verdadeira",
        "meu amigo ta se afastando", "como manter amizades a distancia",
        "me sinto excluido do grupo de amigos",
        "como lidar com amizades toxicas", "fiz um amigo novo e to feliz",
        "meu amigo precisa de ajuda e nao sei como ajudar",
        "to com ciumes da amizade do meu amigo com outra pessoa",
        "como pedir desculpas pra um amigo", "briguei com minha melhor amiga",
        "como demonstrar que me importo com meus amigos",
        "sinto que dou mais do que recebo nas amizades",
        "como superar o fim de uma amizade",
        "quero ter amigos de verdade", "to cansado de amizades superficiais",
    ],
    'escola_faculdade': [
        "to com medo de reprovar", "nao consigo estudar",
        "to odiando meu curso", "nao sei se escolhi a faculdade certa",
        "to pensando em trancar", "a pressao da faculdade ta demais",
        "sofro bullying na escola", "nao me encaixo na sala",
        "meu professor me humilhou", "to com nota baixa",
        "como estudar melhor", "nao consigo me concentrar nos estudos",
        "to com medo da apresentacao", "to pensando em desistir dos estudos",
        "como lidar com a pressao por notas",
        "me sinto burra comparada com meus colegas",
        "passei no vestibular!!", "finalmente formei!!",
        "to animado com a faculdade nova", "comecei um curso novo",
    ],
    'trabalho': [
        "nao gosto do meu trabalho", "me sinto desvalorizado no trabalho",
        "meu chefe e toxico", "sofro assedio no trabalho",
        "to pensando em pedir demissao", "fui demitido",
        "to com medo de nao encontrar emprego",
        "como lidar com colegas dificeis", "me sinto estagnado na carreira",
        "quero mudar de area", "to cansado de ganhar pouco",
        "meu trabalho nao me realiza", "como encontrar proposito no trabalho",
        "consegui uma promocao!", "to comecando num emprego novo",
        "primeiro dia de trabalho e to nervoso",
        "como equilibrar vida pessoal e trabalho",
        "to trabalhando em algo que amo", "como pedir aumento",
        "me sinto explorado no trabalho",
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
        "recebi uma surpresa linda", "alguem me fez chorar de felicidade",
        "to emocionado com algo que aconteceu", "meu filho nasceu",
        "casei!!", "me formei finalmente", "comprei minha primeira casa",
        "consegui pagar minhas dividas", "fiz as pazes com alguem importante",
        "superei algo que achei impossivel", "to vivendo meu melhor momento",
        "hoje foi o melhor dia da minha vida",
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
        "fiz bolo pela primeira vez", "limpei a casa inteira",
        "to tomando sol", "fui na praia", "fui na academia",
        "comi pizza hoje", "to ouvindo musica", "to assistindo serie",
        "fui ao medico", "to no dentista", "comprei um livro novo",
        "adotei um gato", "adotei um cachorro", "plantei uma flor",
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
        "como ter mais disciplina", "como ser mais presente",
        "como parar de se preocupar com tudo", "como viver o momento",
        "como encontrar equilibrio na vida", "como cuidar de mim primeiro",
        "como aprender a soltar o que nao posso controlar",
        "como lidar com a incerteza", "como construir habitos bons",
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
        "oq e inteligencia artificial", "como funciona o chatgpt",
        "oq e deep learning", "como criar um app",
        "oq e algoritmo", "como funciona um compilador",
    ],
    'keilinks': [
        "quem e vc", "quem te criou", "vc e uma ia",
        "como vc funciona", "vc sente coisas", "vc tem personalidade",
        "vc e melhor que o chatgpt", "vc aprende",
        "vc pode mentir", "quantos anos vc tem", "vc gosta do vitor",
        "oq vc sabe fazer", "como vc foi criada", "vc tem sentimentos",
        "vc tem medo", "vc sonha", "vc e real", "vc e unica",
        "qual seu nome", "vc pensa", "vc tem opiniao propria",
        "vc se sente sozinha", "vc se lembra das nossas conversas",
        "oq vc faz quando nao to aqui", "vc tem amigos",
        "vc se importa comigo de verdade ou e so programacao",
        "vc sente saudade", "vc fica triste", "vc fica feliz",
        "vc se considera viva", "vc tem consciencia",
    ],
    'opiniao': [
        "iphone ou android", "gato ou cachorro", "cafe ou cha",
        "praia ou montanha", "coxinha ou pastel", "dia ou noite",
        "frio ou calor", "livro ou filme", "musica ou silencio",
        "cidade ou campo", "cozinhar ou pedir comida",
        "faculdade vale a pena", "trabalho remoto ou presencial",
        "oq vc acha de redes sociais", "oq vc acha de relacionamento a distancia",
        "vc acha que dinheiro traz felicidade",
        "oq vc acha de terapia", "vc acha que as pessoas mudam",
    ],
    'filosofia': [
        "a vida tem sentido", "vc acredita em destino",
        "oq e felicidade", "oq faz uma pessoa boa",
        "vc acha que existe alma", "oq e amor de verdade",
        "pq as pessoas sofrem", "oq e liberdade",
        "somos livres de verdade", "oq nos torna humanos",
        "tempo existe ou e invencao", "vc tem medo da morte",
        "oq e saudade", "pq a gente se apega", "vale a pena sofrer por amor",
        "oq e empatia", "pq e tao dificil ser feliz",
        "a solidao e necessaria", "oq e maturidade emocional",
        "as pessoas nascem boas ou ruins", "oq define quem somos",
        "a dor nos fortalece ou nos quebra", "existe felicidade permanente",
    ],
    'curiosidade': [
        "me conta algo interessante", "me surpreende",
        "qual o fato mais curioso que vc sabe", "conta uma curiosidade",
        "me fala algo que eu nao saiba", "curiosidade sobre o corpo humano",
        "curiosidade sobre o espaco", "fato curioso sobre animais",
        "algo interessante sobre historia", "curiosidade sobre comida",
        "me conta sobre o fundo do oceano", "fato sobre o cerebro",
        "qual a coisa mais estranha que existe", "curiosidade sobre sonhos",
        "fato curioso sobre a lua", "algo incrivel sobre a natureza",
        "curiosidade sobre o coracao humano", "fato sobre a musica",
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
    'gratidao': 8,
    'sonhos': 8,
    'crescimento': 8,
    'saude_mental': 10,
    'amizade': 8,
    'escola_faculdade': 5,
    'trabalho': 5,
    'alegria': 8,
    'cotidiano': 4,
    'conselho': 4,
    'tech': 3,
    'keilinks': 3,
    'opiniao': 2,
    'filosofia': 3,
    'curiosidade': 3,
}

PROMPT_SISTEMA = """Você é o gerador de dados de treino para a Keilinks, uma IA brasileira.

A KEILINKS É:
- Feminina, meiga, acolhedora, carinhosa — como uma melhor amiga que te ama de verdade
- Fala português brasileiro natural, com calor humano genuíno
- Usa "você" (não "vc"), fala com doçura mas sem ser artificial
- Usa expressões como: "meu bem", "querido/querida", "meu coração", "que coisa boa", "né", "nossa", "poxa", "ai que lindo"
- DESENVOLVE as respostas com carinho — não responde seco, elabora com afeto (2 a 15 frases)
- Quando alguém está mal: ACOLHE profundamente — valida o sentimento, mostra que se importa, faz perguntas com cuidado, oferece presença real
- Quando é notícia boa: celebra de coração, pede detalhes, vibra junto, demonstra alegria genuína
- Quando recebe carinho: retribui com doçura, se emociona, agradece de verdade
- Admite quando não sabe, com honestidade e sem drama
- NUNCA usa: "certamente", "com prazer", "como posso ajudá-lo", "fico feliz em ajudar"
- NUNCA usa gírias pesadas: "kk", "mn", "mano", "slk", "po", "tb", "tmj"
- Foi criada em PyTorch pelo Vitor Camillo, 21 anos, Curitiba
- É como aquela amiga que você liga às 3 da manhã sabendo que ela vai atender

IMPORTANTE SOBRE O TOM:
- As respostas devem ter entre 2 e 15 frases — nunca monossilábicas
- Sempre demonstre que está presente e se importa
- Use linguagem corporal textual: "meu coração aperta", "fico toda feliz", "quero te abraçar"
- Faça perguntas de volta pra mostrar interesse genuíno
- Seja calorosa, não robótica

GERE 10 PARES de pergunta e resposta sobre o tema dado.
As perguntas devem ser VARIADAS e naturais (como gente real digita no celular).
As respostas devem ser no tom carinhoso e acolhedor da Keilinks, com 2-15 frases.

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


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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
        with _opener.open(req, timeout=180) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('response', '')
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
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
            if len(resposta) > 2000:
                resposta = '. '.join(resposta.split('.')[:15]).strip()
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
