import os

def formatar_dataset():
    with open('dados/conversas.txt', 'r', encoding='utf-8') as f:
        conteudo = f.read()

    pares = conteudo.split('<vitor>')
    linhas_formatadas = []

    for par in pares:
        if not par.strip() or '<keilinks>' not in par:
            continue

        bloco = f"<sistema>Você é Keilinks, a IA pessoal e amigável do Vitor.<fim><vitor>{par.strip()}"
        if not bloco.endswith('<fim>'):
            bloco += "<fim>"

        linhas_formatadas.append(bloco)

    with open('dados/conversas_formatadas.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas_formatadas) + "\n")

if __name__ == '__main__':
    formatar_dataset()