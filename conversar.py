"""
Conversar com a Keilinks
Execute: python conversar.py
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelo.keilinks import Keilinks
from dados.tokenizador import Tokenizador


def conversar():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_path = 'checkpoints/keilinks_final.pt'
    if not os.path.exists(checkpoint_path):
        print("Keilinks ainda não foi treinada.")
        print("Execute primeiro: python treino/treinar.py")
        return

    print("Carregando Keilinks...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    tokenizador = Tokenizador('dados/vocab.json')
    modelo = Keilinks(checkpoint['config']).to(device)
    modelo.load_state_dict(checkpoint['modelo'])
    modelo.eval()

    print("Keilinks online. Digite 'sair' para encerrar.\n")
    print("-" * 40)

    while True:
        entrada = input("Vitor: ").strip()
        if entrada.lower() in ['sair', 'exit', 'quit']:
            print("Keilinks: Até mais.")
            break
        if not entrada:
            continue

        prompt = f"<vitor>{entrada}<fim><keilinks>"
        tokens = torch.tensor([tokenizador.encode(prompt)], dtype=torch.long).to(device)

        saida = modelo.gerar(tokens, max_tokens=150, temperatura=0.8)
        texto = tokenizador.decode(saida[0].tolist())

        # Pega só a resposta da Keilinks
        if '<keilinks>' in texto:
            resposta = texto.split('<keilinks>')[-1]
            if '<fim>' in resposta:
                resposta = resposta.split('<fim>')[0]
        else:
            resposta = texto

        print(f"Keilinks: {resposta.strip()}\n")


if __name__ == '__main__':
    conversar()
