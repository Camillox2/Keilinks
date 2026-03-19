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

    # Tenta Flash primeiro, depois final (padrao)
    for path in ['checkpoints/keilinks_flash.pt', 'checkpoints/keilinks_final.pt', 'checkpoints/keilinks_ultra.pt']:
        if os.path.exists(path):
            checkpoint_path = path
            break
    else:
        print("Keilinks ainda não foi treinada.")
        print("Execute primeiro: python treino/treinar.py")
        return

    print(f"Carregando Keilinks ({checkpoint_path})...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizador = Tokenizador('dados/vocab.json')
    modelo = Keilinks(checkpoint['config']).to(device)
    # Filtra embedding_posicao de checkpoints antigos (agora usa RoPE)
    state = {k: v for k, v in checkpoint['modelo'].items() if 'embedding_posicao' not in k}
    modelo.load_state_dict(state, strict=False)
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
