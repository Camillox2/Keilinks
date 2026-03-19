import os

def formatar_dataset():
    # Usa dados limpos se existir, senão usa original
    entrada = 'dados/conversas_limpo.txt'
    if not os.path.exists(entrada):
        entrada = 'dados/conversas.txt'

    print(f"  Lendo: {entrada}")

    with open(entrada, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    linhas_formatadas = []

    for linha in linhas:
        linha = linha.strip()
        if not linha or '<vitor>' not in linha or '<keilinks>' not in linha:
            continue

        # Remove <sistema> se já tiver (evita duplicar)
        if '<sistema>' in linha:
            import re
            match = re.search(r'<vitor>.+', linha)
            if match:
                linha = match.group(0)
            else:
                continue

        bloco = f"<sistema>Você é Keilinks, a IA pessoal e amigável do Vitor.<fim>{linha}"
        if not bloco.endswith('<fim>'):
            bloco += '<fim>'

        linhas_formatadas.append(bloco)

    with open('dados/conversas_formatadas.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas_formatadas) + '\n')

    print(f"  Formatadas: {len(linhas_formatadas):,} linhas com <sistema>")
    print(f"  Salvo em: dados/conversas_formatadas.txt")

if __name__ == '__main__':
    formatar_dataset()