"""
Mini re-treino da Keilinks com conversas aprendidas
Chamado automaticamente pelo servidor a cada N conversas
"""

import torch
import sys
import os
import argparse
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modelo.keilinks import Keilinks, MODELOS
from dados.tokenizador import Tokenizador

SAIDAS = {
    'flash':  'checkpoints/keilinks_flash.pt',
    'padrao': 'checkpoints/keilinks_final.pt',
    'ultra':  'checkpoints/keilinks_ultra.pt',
}

PASSOS_MINI   = 300   # poucos passos — só absorve o novo
LR_MINI       = 5e-5  # learning rate baixo — não esquece o que sabe
BATCH_MINI    = 4


def retreinar(tipo: str):
    os.chdir(BASE_DIR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    caminho_ckpt = SAIDAS[tipo]
    if not os.path.exists(caminho_ckpt):
        print(f"Modelo {tipo} não encontrado.")
        return

    # Combina conversas originais + aprendidas
    arquivos = ['dados/conversas.txt', 'dados/aprendizado.txt']
    texto_total = ''
    for arq in arquivos:
        if os.path.exists(arq):
            with open(arq, 'r', encoding='utf-8') as f:
                texto_total += f.read() + '\n'

    if len(texto_total) < 100:
        print("Dados insuficientes para re-treino.")
        return

    tokenizador = Tokenizador('dados/vocab.json')
    tokens = tokenizador.encode(texto_total)
    data = torch.tensor(tokens, dtype=torch.long)

    if len(data) < MODELOS[tipo]['contexto_max'] + 1:
        print("Poucos tokens para re-treino.")
        return

    # Carrega modelo existente
    ckpt = torch.load(caminho_ckpt, map_location=device, weights_only=False)
    modelo = Keilinks(ckpt['config']).to(device)
    modelo.load_state_dict(ckpt['modelo'])
    modelo.train()

    # LR muito baixo — absorve novo sem esquecer o antigo
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LR_MINI, weight_decay=0.01)
    contexto = ckpt['config']['contexto_max']

    print(f"[Auto-aprendizado] Re-treinando {tipo} por {PASSOS_MINI} passos...")

    for passo in range(PASSOS_MINI):
        ix = torch.randint(len(data) - contexto, (BATCH_MINI,))
        x = torch.stack([data[i:i+contexto] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+contexto+1] for i in ix]).to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device=='cuda')):
            _, loss = modelo(x, y)

        otimizador.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()

    # Salva de volta
    torch.save({'passo': ckpt['passo'], 'modelo': modelo.state_dict(), 'config': ckpt['config']},
               caminho_ckpt)
    print(f"[Auto-aprendizado] Concluido. Loss final: {loss.item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', choices=['flash', 'padrao', 'ultra'], default='flash')
    args = parser.parse_args()
    retreinar(args.modelo)
