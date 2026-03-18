"""
Loop de treino da Keilinks
Uso:
  python treino/treinar.py --modelo flash
  python treino/treinar.py --modelo padrao
  python treino/treinar.py --modelo ultra
"""

import torch
import torch.nn as nn
import sys
import os
import time
import math
import argparse

# Força UTF-8 no terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelo.keilinks import Keilinks, MODELOS
from dados.tokenizador import Tokenizador


CONFIGS_TREINO = {
    'flash':  {'batch': 24, 'passos': 5000,  'lr_max': 5e-4, 'lr_min': 5e-5},
    'padrao': {'batch': 12, 'passos': 8000,  'lr_max': 3e-4, 'lr_min': 3e-5},
    'ultra':  {'batch': 6,  'passos': 12000, 'lr_max': 2e-4, 'lr_min': 2e-5},
}

SAIDAS = {
    'flash':  'checkpoints/keilinks_flash.pt',
    'padrao': 'checkpoints/keilinks_final.pt',
    'ultra':  'checkpoints/keilinks_ultra.pt',
}

WARMUP    = 200
GRAD_CLIP = 1.0
AVALIAR   = 100
SALVAR    = 500


def lr_cosine(passo, total, lr_max, lr_min, warmup):
    if passo < warmup:
        return lr_max * passo / warmup
    p = (passo - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


def preparar_dados(caminho, tokenizador, contexto):
    with open(caminho, 'r', encoding='utf-8') as f:
        texto = f.read()
    tokens = tokenizador.encode(texto)
    data = torch.tensor(tokens, dtype=torch.long)
    corte = int(0.9 * len(data))
    return data[:corte], data[corte:]


def pegar_batch(data, batch_size, contexto, device):
    ix = torch.randint(len(data) - contexto, (batch_size,))
    x = torch.stack([data[i:i+contexto] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+contexto+1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def avaliar(modelo, val_data, batch_size, contexto, device, n=10):
    modelo.eval()
    losses = []
    for _ in range(n):
        x, y = pegar_batch(val_data, batch_size, contexto, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = modelo(x, y)
        losses.append(loss.item())
    modelo.train()
    return sum(losses) / len(losses)


def treinar(tipo_modelo: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg_treino = CONFIGS_TREINO[tipo_modelo]
    cfg_modelo = MODELOS[tipo_modelo].copy()
    BATCH    = cfg_treino['batch']
    PASSOS   = cfg_treino['passos']
    LR_MAX   = cfg_treino['lr_max']
    LR_MIN   = cfg_treino['lr_min']
    CONTEXTO = cfg_modelo['contexto_max']

    print("=" * 55)
    print(f"  Treinando: {cfg_modelo['nome']}")
    print(f"  Device:    {device.upper()}")
    if device == 'cuda':
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Batch:     {BATCH} | Passos: {PASSOS} | LR: {LR_MAX}")
    print("=" * 55)

    caminho_dados = 'dados/conversas.txt'
    if not os.path.exists(caminho_dados):
        print("\nERRO: 'dados/conversas.txt' não encontrado.")
        print("Execute primeiro: python dados/preparar_dados.py")
        return

    tokenizador = Tokenizador()
    with open(caminho_dados, 'r', encoding='utf-8') as f:
        texto = f.read()

    tokenizador.construir_vocab([texto])
    tokenizador.salvar('dados/vocab.json')

    treino_data, val_data = preparar_dados(caminho_dados, tokenizador, CONTEXTO)
    print(f"\n  Treino: {len(treino_data):,} tokens | Val: {len(val_data):,} tokens\n")

    cfg_modelo['vocab_size'] = tokenizador.tam_vocab
    cfg_modelo['eos_id'] = tokenizador.vocab.get(tokenizador.EOS, 1)

    modelo = Keilinks(cfg_modelo).to(device)

    # Retoma checkpoint se existir
    passo_inicial = 0
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_tmp = f'checkpoints/keilinks_{tipo_modelo}_tmp.pt'
    if os.path.exists(ckpt_tmp):
        ckpt = torch.load(ckpt_tmp, map_location=device, weights_only=False)
        ckpt_vocab = ckpt['config'].get('vocab_size', 0)
        novo_vocab = cfg_modelo['vocab_size']
        if ckpt_vocab != novo_vocab:
            # Vocab mudou — carrega pesos compativeis, inicializa tokens novos
            print(f"  Vocab mudou: {ckpt_vocab} -> {novo_vocab}, ajustando pesos...")
            state = ckpt['modelo']
            for key in ['embedding_token.weight', 'cabeca_saida.weight']:
                if key in state and state[key].shape[0] != novo_vocab:
                    old_w = state[key]
                    new_w = torch.zeros(novo_vocab, old_w.shape[1])
                    nn.init.normal_(new_w, 0.0, 0.02)
                    new_w[:old_w.shape[0]] = old_w
                    state[key] = new_w
            modelo.load_state_dict(state)
        else:
            modelo.load_state_dict(ckpt['modelo'])
        passo_inicial = ckpt['passo']
        print(f"  Retomando do passo {passo_inicial}\n")

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1
    )

    melhor_val = float('inf')
    inicio = time.time()

    for passo in range(passo_inicial, PASSOS):
        lr = lr_cosine(passo, PASSOS, LR_MAX, LR_MIN, WARMUP)
        for g in otimizador.param_groups:
            g['lr'] = lr

        x, y = pegar_batch(treino_data, BATCH, CONTEXTO, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device=='cuda')):
            _, loss = modelo(x, y)

        otimizador.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
        otimizador.step()

        if passo % AVALIAR == 0:
            loss_val = avaliar(modelo, val_data, BATCH, CONTEXTO, device)
            elapsed = time.time() - inicio
            tok_s = (passo * BATCH * CONTEXTO) / max(elapsed, 1)
            melhor = ''
            if loss_val < melhor_val:
                melhor_val = loss_val
                torch.save({'passo': passo, 'modelo': modelo.state_dict(), 'config': cfg_modelo},
                           SAIDAS[tipo_modelo].replace('.pt', '_melhor.pt'))
                melhor = ' [MELHOR]'

            print(f"[{passo:>5}/{PASSOS}] loss {loss.item():.4f} | val {loss_val:.4f} | "
                  f"lr {lr:.1e} | {tok_s:.0f} tok/s{melhor}")

        if passo % SALVAR == 0 and passo > 0:
            torch.save({'passo': passo, 'modelo': modelo.state_dict(),
                        'otimizador': otimizador.state_dict(), 'config': cfg_modelo},
                       ckpt_tmp)
            print(f"  >> checkpoint salvo (passo {passo})")

    # Usa o melhor checkpoint como final (evita overfitting)
    import shutil
    melhor_path = SAIDAS[tipo_modelo].replace('.pt', '_melhor.pt')
    if os.path.exists(melhor_path):
        shutil.copy(melhor_path, SAIDAS[tipo_modelo])
        print(f"\n  Usando melhor checkpoint como modelo final")
    else:
        torch.save({'passo': PASSOS, 'modelo': modelo.state_dict(), 'config': cfg_modelo},
                   SAIDAS[tipo_modelo])

    print(f"  Concluido em {(time.time()-inicio)/60:.1f} min")
    print(f"  Melhor val loss: {melhor_val:.4f}")
    print(f"  Salvo em: {SAIDAS[tipo_modelo]}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', choices=['flash', 'padrao', 'ultra'], default='padrao')
    args = parser.parse_args()
    treinar(args.modelo)
