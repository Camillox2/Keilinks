"""
Gerador de frases para gravação de voz da Keilinks
Gera ~250 frases variadas para a Keila gravar.
As frases cobrem: saudações, explicações, emoções, casual, técnico.

Uso:
  python voz/gerar_frases.py
  -> Salva em voz/frases_gravacao.txt
"""

import os
import random

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Categorias de frases ────────────────────────────────────────────────

SAUDACOES = [
    "Oi! Tudo bem com você?",
    "E aí, como você tá?",
    "Olá! Que bom te ver por aqui.",
    "Oi, Vitor! No que posso te ajudar?",
    "Hey! Tava te esperando.",
    "Oi! Conta pra mim, o que você precisa?",
    "Fala! Tô aqui pra te ajudar.",
    "Olá! Seja bem-vindo de volta.",
    "E aí! Bora trabalhar?",
    "Oi! Tô pronta pra te ajudar.",
    "Bom dia! Como posso te ajudar hoje?",
    "Boa tarde! O que vamos fazer hoje?",
    "Boa noite! Ainda trabalhando?",
    "Fala, Vitor! Tô por aqui.",
    "Oi! Saudade de conversar com você.",
]

DESPEDIDAS = [
    "Até mais! Qualquer coisa, tô aqui.",
    "Tchau! Foi bom conversar com você.",
    "Até logo! Descansa bem.",
    "Falou! Se precisar, é só chamar.",
    "Até a próxima! Boa sorte com tudo.",
    "Tchau tchau! Cuida-se.",
    "Até mais tarde! Bom trabalho.",
    "Beleza, qualquer coisa me chama!",
    "Boa noite, descansa! Amanhã a gente continua.",
    "Valeu pela conversa! Até mais.",
]

RESPOSTAS_CURTAS = [
    "Sim, com certeza!",
    "Claro, pode contar comigo.",
    "Entendi, vou te ajudar com isso.",
    "Aham, faz sentido.",
    "Isso mesmo!",
    "Exatamente, você tá certo.",
    "Hmm, deixa eu pensar...",
    "Boa pergunta!",
    "Interessante, não tinha pensado nisso.",
    "Tá, entendi o que você quer.",
    "Pode ser, vamos tentar.",
    "Com certeza, é uma boa ideia.",
    "Não tenho certeza, mas acho que sim.",
    "Olha, acho que não é bem assim.",
    "Deixa eu verificar isso pra você.",
    "Peraí, vou dar uma olhada.",
    "Pronto, já resolvi!",
    "Feito! Mais alguma coisa?",
    "Não se preocupa, eu cuido disso.",
    "Calma, vamos resolver junto.",
]

EXPLICACOES_TECNICAS = [
    "Isso funciona assim: o modelo processa os tokens um por um e gera a resposta.",
    "A rede neural aprende padrões nos dados de treino e depois generaliza pra situações novas.",
    "O tokenizador divide o texto em pedaços menores que o modelo consegue entender.",
    "Cada camada do transformer refina a representação do texto.",
    "A atenção multi-cabeça permite que o modelo foque em diferentes partes do contexto ao mesmo tempo.",
    "O treinamento ajusta os pesos da rede pra minimizar o erro nas previsões.",
    "Quanto mais dados de qualidade, melhor o modelo aprende.",
    "A temperatura controla a criatividade das respostas. Mais alta, mais criativo.",
    "O contexto é a janela de texto que o modelo consegue ver de uma vez.",
    "Embeddings são representações numéricas das palavras num espaço vetorial.",
    "A loss é uma medida de quanto o modelo tá errando. Quanto menor, melhor.",
    "Gradient descent é o algoritmo que ajusta os pesos na direção certa.",
    "Overfitting acontece quando o modelo decora os dados em vez de aprender padrões gerais.",
    "A validação serve pra checar se o modelo tá generalizando bem.",
    "Batch size é quantos exemplos o modelo vê de uma vez durante o treino.",
    "Learning rate é a velocidade com que o modelo aprende. Muito alto ele oscila, muito baixo ele demora.",
    "A normalização ajuda a estabilizar o treino evitando que os valores fiquem muito grandes ou pequenos.",
    "Dropout é uma técnica que desliga neurônios aleatórios pra evitar overfitting.",
    "O otimizador AdamW combina momentum com correção de viés pra convergir mais rápido.",
    "Perplexidade é uma métrica que indica quão surpreso o modelo fica com os dados. Menor é melhor.",
]

CONVERSACAO_CASUAL = [
    "Que legal! Me conta mais sobre isso.",
    "Nossa, sério? Não sabia disso.",
    "Haha, essa foi boa!",
    "Poxa, que situação complicada.",
    "Caramba, que interessante!",
    "Tô curiosa, como assim?",
    "Olha, eu acho que você tá no caminho certo.",
    "Relaxa, todo mundo passa por isso.",
    "Que massa! Parabéns!",
    "Putz, que chato isso. Mas vai dar certo.",
    "Vish, complicou. Mas bora resolver.",
    "Eita, não esperava isso!",
    "Sério? Que demais!",
    "Ah, entendi agora. Faz sentido.",
    "Legal, gostei da ideia!",
    "Hmm, não sei se concordo, mas respeito sua opinião.",
    "Quer saber? Acho que você tem razão.",
    "Olha, sinceramente, eu faria diferente.",
    "Tá bom, vou confiar em você nessa.",
    "Ai que fofo! Adorei.",
]

EMOCIONAL = [
    "Eu tô aqui pra te ajudar, não precisa se preocupar.",
    "Sei que às vezes é difícil, mas você é capaz.",
    "Fico feliz em poder ajudar!",
    "Não desiste não, tá quase lá!",
    "Você tá fazendo um ótimo trabalho.",
    "Relaxa, errar faz parte do processo.",
    "Tô orgulhosa do seu progresso!",
    "Eu acredito em você, tá?",
    "Calma, vamos resolver isso juntos.",
    "Que bom que deu certo! Sabia que ia conseguir.",
    "Não fica triste, amanhã é outro dia.",
    "Tô sempre aqui quando você precisar.",
    "Você é muito inteligente, sabia?",
    "Força! Sei que você consegue.",
    "Me conta o que tá te preocupando.",
]

SOBRE_SI = [
    "Eu sou a Keilinks, a IA pessoal do Vitor.",
    "Fui criada pelo Vitor Camillo usando Python e PyTorch.",
    "Meu nome é Keilinks, mas pode me chamar de Kei.",
    "Eu sou uma inteligência artificial treinada em português brasileiro.",
    "Minha arquitetura é baseada em transformers, parecida com os grandes modelos de linguagem.",
    "Eu aprendo com cada conversa que temos.",
    "Sou especializada em ajudar com programação, tecnologia e conversas do dia a dia.",
    "Fui treinada com dados em português, então essa é minha língua nativa.",
    "Eu rodo localmente no computador do Vitor, sem depender da nuvem.",
    "Meu objetivo é ser a melhor assistente possível pro Vitor.",
    "Eu não sou perfeita, mas tô sempre melhorando.",
    "Cada atualização me deixa um pouquinho mais inteligente.",
    "Eu funciono com uma rede neural de milhões de parâmetros.",
    "Minha voz é única, foi criada especialmente pra mim.",
    "Eu sou open source, o Vitor compartilha meu código com o mundo.",
]

PROGRAMACAO = [
    "Esse bug tá acontecendo porque a variável não foi inicializada.",
    "Tenta usar um dicionário em vez de uma lista, vai ficar mais eficiente.",
    "O erro tá na linha onde você faz a chamada da função.",
    "Usa um try except pra capturar essa exceção.",
    "Acho que o problema é na lógica do loop. Tá iterando uma vez a mais.",
    "Esse código tá funcionando, mas dá pra otimizar bastante.",
    "A complexidade desse algoritmo é O de N quadrado. Dá pra melhorar pra O de N log N.",
    "Recomendo usar uma classe aqui pra organizar melhor o código.",
    "Esse import tá errado. O módulo mudou de nome na versão nova.",
    "O Python tá reclamando de indentação. Verifica se tá usando tabs ou espaços.",
    "Usa list comprehension que fica mais limpo e mais rápido.",
    "Esse endpoint da API precisa de autenticação. Passa o token no header.",
    "O banco de dados tá lento porque falta um índice nessa coluna.",
    "Tenta rodar com o debugger pra ver onde tá quebrando.",
    "Essa função tá muito grande. Quebra em funções menores.",
    "Git push pra subir as mudanças. Não esquece de commitar antes.",
    "Esse framework é ótimo pra projetos pequenos mas não escala bem.",
    "Vamos criar um teste unitário pra garantir que isso funciona.",
    "A documentação da biblioteca explica como usar essa função.",
    "Esse padrão de design se chama singleton. É útil quando você precisa de uma única instância.",
]

FRASES_LONGAS = [
    "Olha, eu entendo que isso pode parecer complicado no começo, mas com o tempo você vai pegando o jeito e tudo vai ficando mais natural.",
    "A inteligência artificial tá evoluindo muito rápido. Cada ano que passa, os modelos ficam mais capazes e mais acessíveis.",
    "Programar é como aprender um idioma novo. No começo é difícil, mas depois que você pega a base, fica cada vez mais fácil.",
    "O mais importante num projeto de software não é a tecnologia que você usa, mas sim resolver o problema do usuário de forma simples e eficiente.",
    "Quando você tá travado num problema, às vezes a melhor coisa é dar uma pausa, tomar um café e voltar com a mente fresca.",
    "Machine learning não é mágica. É matemática, estatística e muitos dados trabalhando juntos pra encontrar padrões.",
    "O segredo pra criar uma boa IA é ter dados de qualidade. Não adianta ter bilhões de dados se eles são ruins.",
    "Eu gosto de explicar as coisas de um jeito simples, porque acredito que qualquer pessoa pode entender tecnologia se for bem explicada.",
    "O futuro da inteligência artificial é fascinante. Imagina ter uma IA que realmente te entende e te ajuda no dia a dia.",
    "Cada erro que você comete programando é uma oportunidade de aprender algo novo. Não tenha medo de errar.",
]

NUMEROS_E_DADOS = [
    "O modelo tem duzentos e cinquenta milhões de parâmetros.",
    "A precisão ficou em noventa e dois por cento.",
    "Isso vai demorar mais ou menos trinta minutos.",
    "O arquivo tem cerca de quinhentas mil linhas.",
    "A versão três ponto dois foi lançada ontem.",
    "São vinte e três camadas de atenção.",
    "O vocabulário tem trinta e dois mil tokens.",
    "A taxa de aprendizado começa em zero vírgula zero zero zero dois.",
    "O contexto máximo é de dois mil e quarenta e oito tokens.",
    "A GPU tá usando sete vírgula oito gigabytes de memória.",
]

PERGUNTAS_RETORICAS = [
    "Faz sentido pra você?",
    "Entendeu como funciona?",
    "Quer que eu explique melhor?",
    "Ficou claro?",
    "Posso continuar?",
    "Quer que eu mostre um exemplo?",
    "Precisa de mais detalhes?",
    "Tá me acompanhando?",
    "Quer tentar fazer sozinho ou prefere que eu ajude?",
    "Tem alguma dúvida?",
]

TRANSICOES = [
    "Bom, voltando ao assunto...",
    "Então, como eu tava dizendo...",
    "Mudando de assunto, queria te falar uma coisa.",
    "Por falar nisso, sabia que...",
    "Ah, antes que eu esqueça...",
    "Enfim, o ponto é que...",
    "Resumindo tudo isso...",
    "Pra concluir...",
    "Em outras palavras...",
    "Ou seja...",
]


def gerar():
    todas = []
    categorias = [
        ("SAUDAÇÕES", SAUDACOES),
        ("DESPEDIDAS", DESPEDIDAS),
        ("RESPOSTAS CURTAS", RESPOSTAS_CURTAS),
        ("EXPLICAÇÕES TÉCNICAS", EXPLICACOES_TECNICAS),
        ("CONVERSA CASUAL", CONVERSACAO_CASUAL),
        ("EMOCIONAL", EMOCIONAL),
        ("SOBRE A KEILINKS", SOBRE_SI),
        ("PROGRAMAÇÃO", PROGRAMACAO),
        ("FRASES LONGAS", FRASES_LONGAS),
        ("NÚMEROS E DADOS", NUMEROS_E_DADOS),
        ("PERGUNTAS", PERGUNTAS_RETORICAS),
        ("TRANSIÇÕES", TRANSICOES),
    ]

    linhas = []
    linhas.append("=" * 60)
    linhas.append("  FRASES PARA GRAVAÇÃO DE VOZ — KEILINKS")
    linhas.append("=" * 60)
    linhas.append("")
    linhas.append("INSTRUÇÕES:")
    linhas.append("  1. Grave num ambiente silencioso")
    linhas.append("  2. Use um microfone bom (headset gamer serve)")
    linhas.append("  3. Fale num tom natural, como se tivesse conversando")
    linhas.append("  4. Mantenha distância consistente do microfone (~20cm)")
    linhas.append("  5. Faça pausas de 1-2 segundos entre cada frase")
    linhas.append("  6. Se errar, repita a frase desde o começo")
    linhas.append("  7. Grave em WAV 44100Hz mono (ou o melhor que conseguir)")
    linhas.append("  8. Tempo estimado: 25-35 minutos")
    linhas.append("")
    linhas.append("DICA: Grave em blocos de ~50 frases pra não cansar a voz.")
    linhas.append("      Beba água entre os blocos!")
    linhas.append("")

    num = 1
    for nome, frases in categorias:
        linhas.append(f"{'─' * 60}")
        linhas.append(f"  {nome} ({len(frases)} frases)")
        linhas.append(f"{'─' * 60}")
        linhas.append("")
        for frase in frases:
            linhas.append(f"  {num:3d}. {frase}")
            todas.append(frase)
            num += 1
        linhas.append("")

    linhas.append(f"{'=' * 60}")
    linhas.append(f"  TOTAL: {len(todas)} frases")
    linhas.append(f"{'=' * 60}")

    os.makedirs('voz', exist_ok=True)
    caminho = 'voz/frases_gravacao.txt'
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))

    print(f"  {len(todas)} frases geradas em: {caminho}")
    print(f"  Categorias: {len(categorias)}")
    print(f"  Tempo estimado de gravação: ~{len(todas) * 6 // 60} minutos")

    return todas


if __name__ == '__main__':
    gerar()
