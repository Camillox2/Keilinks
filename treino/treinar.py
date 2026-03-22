"""
Loop de treino da Keilinks v3
Otimizado pra RTX 5050 (8GB VRAM)
  - Gradient Accumulation (batch grande sem VRAM extra)
  - Gradient Checkpointing (economiza VRAM nos modelos maiores)
  - Mixed Precision bf16
  - Perplexity no log
  - Early Stopping (para quando val loss para de melhorar)
  - Label Smoothing (reduz overfitting)

Modelos:
  - Flash: 250M params (batch=2, ~6GB VRAM)
  - Pro:   275M params (batch=2, ~6GB VRAM)
  - Ultra: 300M params (batch=1, ~5GB VRAM)

Uso:
  python treino/treinar.py --modelo flash
  python treino/treinar.py --modelo padrao
  python treino/treinar.py --modelo ultra
  python treino/treinar.py --modelo ultra --grad-accum 16 --batch 1
"""

import torch
import torch.nn as nn
import sys
import os
import time
import math
import argparse

# Forca UTF-8 no terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelo.keilinks import Keilinks, MODELOS
from dados.tokenizador import Tokenizador


# ─── Configs otimizadas pra 8GB VRAM ─────────────────────────────────────
# batch = batch fisico (o que cabe na GPU)
# accum = passos de acumulacao (batch efetivo = batch * accum)
# grad_ckpt = ativa gradient checkpointing (troca VRAM por velocidade)

CONFIGS_TREINO = {
    'flash':  {'batch': 2,  'accum': 8,  'passos': 20000, 'lr_max': 2e-4, 'lr_min': 2e-5, 'grad_ckpt': True, 'label_smooth': 0.1},
    'padrao': {'batch': 2,  'accum': 8,  'passos': 20000, 'lr_max': 2e-4, 'lr_min': 2e-5, 'grad_ckpt': True, 'label_smooth': 0.1},
    'ultra':  {'batch': 1,  'accum': 16, 'passos': 25000, 'lr_max': 1.5e-4, 'lr_min': 1.5e-5, 'grad_ckpt': True, 'label_smooth': 0.1},
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
PACIENCIA = 2000  # Early stopping: para se val loss nao melhora por N passos


def lr_cosine(passo, total, lr_max, lr_min, warmup):
    if passo < warmup:
        return lr_max * passo / warmup
    p = (passo - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


def preparar_dados(caminho, tokenizador, contexto):
    print("  Tokenizando dados (isso pode demorar na primeira vez)...")
    t0 = time.time()

    with open(caminho, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    # Tokeniza em blocos pra mostrar progresso
    todos_tokens = []
    bloco = 10000
    for i in range(0, len(linhas), bloco):
        chunk = ''.join(linhas[i:i+bloco])
        todos_tokens.extend(tokenizador.encode(chunk))
        if (i + bloco) % 50000 == 0 or i + bloco >= len(linhas):
            elapsed = time.time() - t0
            pct = min(100, (i + bloco) / len(linhas) * 100)
            print(f"    {pct:.0f}% — {len(todos_tokens):,} tokens — {elapsed:.0f}s")

    data = torch.tensor(todos_tokens, dtype=torch.long)
    corte = int(0.9 * len(data))
    print(f"  Tokenização completa: {len(data):,} tokens em {time.time()-t0:.0f}s")
    return data[:corte], data[corte:]


def pegar_batch(data, batch_size, contexto, device):
    ctx = min(contexto, len(data) - 1)
    if ctx < 1:
        raise ValueError(f"Dados muito pequenos ({len(data)} tokens). Gere mais dados: python treino/gerar_dados.py")
    ix = torch.randint(len(data) - ctx, (batch_size,))
    x = torch.stack([data[i:i+ctx] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+ctx+1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def avaliar(modelo, val_data, batch_size, contexto, device, n=10):
    modelo.eval()
    losses = []
    bs = min(batch_size, max(1, len(val_data) // (contexto + 1)))
    for _ in range(n):
        x, y = pegar_batch(val_data, bs, contexto, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            _, loss = modelo(x, y)
        losses.append(loss.item())
    modelo.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss


def treinar(tipo_modelo: str, batch_override=None, accum_override=None, passos_override=None, usar_pretreino=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Otimizacoes NVIDIA: cudnn benchmark auto-tune kernels pra GPU
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    cfg_treino = CONFIGS_TREINO[tipo_modelo].copy()
    cfg_modelo = MODELOS[tipo_modelo].copy()

    # Overrides de linha de comando
    BATCH     = batch_override or cfg_treino['batch']
    ACCUM     = accum_override or cfg_treino['accum']
    PASSOS    = passos_override or cfg_treino['passos']
    LR_MAX    = cfg_treino['lr_max']
    LR_MIN    = cfg_treino['lr_min']
    GRAD_CKPT = cfg_treino['grad_ckpt']
    LABEL_SMOOTH = cfg_treino.get('label_smooth', 0.0)
    CONTEXTO  = cfg_modelo['contexto_max']

    batch_efetivo = BATCH * ACCUM

    print("=" * 60)
    print(f"  Keilinks Treino v3 — Otimizado pra 8GB VRAM")
    print(f"=" * 60)
    print(f"  Modelo:      {cfg_modelo['nome']}")
    print(f"  Device:      {device.upper()}")
    if device == 'cuda':
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:        {vram:.1f} GB")
    print(f"  Batch:       {BATCH} (fisico) x {ACCUM} (accum) = {batch_efetivo} (efetivo)")
    print(f"  Passos:      {PASSOS}")
    print(f"  Contexto:    {CONTEXTO}")
    print(f"  LR:          {LR_MAX} -> {LR_MIN}")
    print(f"  Grad Ckpt:   {'SIM' if GRAD_CKPT else 'NAO'}")
    print(f"  Label Smooth: {LABEL_SMOOTH}")
    print(f"  Early Stop:  paciencia {PACIENCIA} passos")
    print(f"  Mixed Prec:  bf16" if device == 'cuda' else "  Mixed Prec:  desativado")
    print("=" * 60)

    # Usa formatadas (com <sistema>) se existir, senão original
    if os.path.exists('dados/conversas_formatadas.txt'):
        caminho_dados = 'dados/conversas_formatadas.txt'
        print(f"  Dados:       conversas_formatadas.txt (com <sistema>)")
    else:
        caminho_dados = 'dados/conversas.txt'
        print(f"  Dados:       conversas.txt (sem <sistema>)")
    if not os.path.exists(caminho_dados):
        print("\nERRO: dados de treino nao encontrados.")
        print("Execute primeiro: python treino/gerar_dados.py")
        return

    tokenizador = Tokenizador()

    # Se usando pretreino, DEVE reutilizar o mesmo vocab (senão token IDs ficam errados)
    vocab_path = 'dados/vocab.json'
    if usar_pretreino and os.path.exists(vocab_path):
        print(f"  Vocab:       reutilizando {vocab_path} (mesmo do pré-treino)")
        tokenizador.carregar(vocab_path)
    elif os.path.exists(vocab_path):
        print(f"  Vocab:       reutilizando {vocab_path}")
        tokenizador.carregar(vocab_path)
    else:
        with open(caminho_dados, 'r', encoding='utf-8') as f:
            texto = f.read()
        vocab_alvo = cfg_modelo.get('vocab_size', 32000)
        tokenizador.construir_vocab([texto], vocab_alvo=vocab_alvo)
        tokenizador.salvar(vocab_path)

    treino_data, val_data = preparar_dados(caminho_dados, tokenizador, CONTEXTO)

    # Adapta contexto se dados forem pequenos demais
    dados_min = min(len(treino_data), len(val_data))
    if dados_min <= CONTEXTO:
        CONTEXTO_REAL = max(dados_min - 1, 32)
        print(f"\n  AVISO: Dados ({dados_min} tokens) < contexto ({CONTEXTO})")
        print(f"  Contexto adaptado: {CONTEXTO_REAL}")
        print(f"  Gere mais dados: python treino/gerar_dados.py\n")
    else:
        CONTEXTO_REAL = CONTEXTO

    print(f"\n  Treino: {len(treino_data):,} tokens | Val: {len(val_data):,} tokens")
    print(f"  Contexto efetivo: {CONTEXTO_REAL}\n")

    cfg_modelo['vocab_size'] = tokenizador.tam_vocab
    cfg_modelo['eos_id'] = tokenizador.vocab.get(tokenizador.EOS, 1)

    modelo = Keilinks(cfg_modelo).to(device)

    # Ativa gradient checkpointing se configurado
    if GRAD_CKPT:
        modelo.usar_grad_checkpoint = True
        print(f"  Gradient checkpointing ATIVADO (economiza ~40% VRAM)")

    # torch.compile nao funciona no Windows (precisa de Triton/Linux)
    # cudnn.benchmark ja esta ativado acima e da ~5-10% de ganho

    # Carrega checkpoint pré-treinado se --pretreino
    if usar_pretreino:
        ckpt_pretreino = f'checkpoints/keilinks_{tipo_modelo}_pretreino.pt'
        if os.path.exists(ckpt_pretreino):
            print(f"\n  Carregando modelo PRÉ-TREINADO: {ckpt_pretreino}")
            ckpt = torch.load(ckpt_pretreino, map_location=device, weights_only=False)
            state = ckpt['modelo']
            state = {k: v for k, v in state.items() if 'embedding_posicao' not in k}
            # Ajusta vocab se mudou
            novo_vocab = cfg_modelo['vocab_size']
            for key in ['embedding_token.weight', 'cabeca_saida.weight']:
                if key in state and state[key].shape[0] != novo_vocab:
                    old_w = state[key]
                    new_w = torch.zeros(novo_vocab, old_w.shape[1])
                    nn.init.normal_(new_w, 0.0, 0.02)
                    min_v = min(old_w.shape[0], novo_vocab)
                    new_w[:min_v] = old_w[:min_v]
                    state[key] = new_w
            modelo.load_state_dict(state, strict=False)
            print(f"  Modelo pré-treinado carregado! (passo {ckpt.get('passo', '?')})")
            print(f"  Agora fine-tuning em dados conversacionais...")
        else:
            print(f"\n  AVISO: checkpoint pré-treinado não encontrado: {ckpt_pretreino}")
            print(f"  Execute primeiro: python treino/pretreinar.py --modelo {tipo_modelo}")
            print(f"  Continuando do zero...\n")

    # Retoma checkpoint se existir
    passo_inicial = 0
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_tmp = f'checkpoints/keilinks_{tipo_modelo}_tmp.pt'
    if not usar_pretreino and os.path.exists(ckpt_tmp):
        ckpt = torch.load(ckpt_tmp, map_location=device, weights_only=False)
        ckpt_vocab = ckpt['config'].get('vocab_size', 0)
        novo_vocab = cfg_modelo['vocab_size']
        ckpt_dim = ckpt['config'].get('dim', 0)
        novo_dim = cfg_modelo['dim']

        if ckpt_dim != novo_dim:
            print(f"  Arquitetura mudou (dim {ckpt_dim} -> {novo_dim}), treinando do zero")
        elif ckpt_vocab != novo_vocab:
            print(f"  Vocab mudou: {ckpt_vocab} -> {novo_vocab}, ajustando pesos...")
            state = ckpt['modelo']
            for key in ['embedding_token.weight', 'cabeca_saida.weight']:
                if key in state and state[key].shape[0] != novo_vocab:
                    old_w = state[key]
                    new_w = torch.zeros(novo_vocab, old_w.shape[1])
                    nn.init.normal_(new_w, 0.0, 0.02)
                    min_v = min(old_w.shape[0], novo_vocab)
                    new_w[:min_v] = old_w[:min_v]
                    state[key] = new_w
            # Remove embedding_posicao de checkpoints antigos (agora usa RoPE)
            state = {k: v for k, v in state.items() if 'embedding_posicao' not in k}
            modelo.load_state_dict(state, strict=False)
            passo_inicial = ckpt['passo']
            print(f"  Retomando do passo {passo_inicial}")
        else:
            state = ckpt['modelo']
            state = {k: v for k, v in state.items() if 'embedding_posicao' not in k}
            modelo.load_state_dict(state, strict=False)
            passo_inicial = ckpt['passo']
            print(f"  Retomando do passo {passo_inicial}")

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Label smoothing: suaviza os targets pra evitar overconfidence/overfitting
    criterio = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH) if LABEL_SMOOTH > 0 else None

    melhor_val = float('inf')
    passo_melhor = 0  # passo do melhor val loss (pra early stopping)
    inicio = time.time()

    print(f"\n  Iniciando treino...\n")

    for passo in range(passo_inicial, PASSOS):
        lr = lr_cosine(passo, PASSOS, LR_MAX, LR_MIN, WARMUP)
        for g in otimizador.param_groups:
            g['lr'] = lr

        # ─── Gradient Accumulation ────────────────────────────────────
        otimizador.zero_grad(set_to_none=True)
        loss_acum = 0.0

        for micro in range(ACCUM):
            x, y = pegar_batch(treino_data, BATCH, CONTEXTO_REAL, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, _ = modelo(x, y)
                # Usa label smoothing se configurado, senao cross_entropy padrao
                if criterio is not None:
                    loss = criterio(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss_escalonada = loss / ACCUM

            loss_escalonada.backward()
            loss_acum += loss.item()

        loss_acum /= ACCUM

        torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
        otimizador.step()

        # ─── Avaliacao ────────────────────────────────────────────────
        if passo % AVALIAR == 0:
            loss_val = avaliar(modelo, val_data, BATCH, CONTEXTO_REAL, device)
            perplexity = math.exp(min(loss_val, 20))
            elapsed = time.time() - inicio
            tok_s = (passo * batch_efetivo * CONTEXTO_REAL) / max(elapsed, 1)
            melhor = ''

            if loss_val < melhor_val:
                melhor_val = loss_val
                passo_melhor = passo
                torch.save({'passo': passo, 'modelo': modelo.state_dict(), 'config': cfg_modelo},
                           SAIDAS[tipo_modelo].replace('.pt', '_melhor.pt'))
                melhor = ' *MELHOR*'

            # VRAM info
            vram_info = ''
            if device == 'cuda':
                vram_usada = torch.cuda.max_memory_allocated() / 1e9
                vram_info = f' | VRAM {vram_usada:.1f}G'

            print(f"[{passo:>5}/{PASSOS}] loss {loss_acum:.4f} | val {loss_val:.4f} | "
                  f"ppl {perplexity:.1f} | lr {lr:.1e} | {tok_s:.0f} tok/s{vram_info}{melhor}")

            # ─── Early Stopping ──────────────────────────────────────
            if passo - passo_melhor >= PACIENCIA and passo > WARMUP:
                print(f"\n  EARLY STOPPING: val loss nao melhorou por {PACIENCIA} passos")
                print(f"  Melhor val loss: {melhor_val:.4f} (passo {passo_melhor})")
                break

        # ─── Checkpoint ───────────────────────────────────────────────
        if passo % SALVAR == 0 and passo > 0:
            torch.save({'passo': passo, 'modelo': modelo.state_dict(),
                        'otimizador': otimizador.state_dict(), 'config': cfg_modelo},
                       ckpt_tmp)
            print(f"  >> checkpoint salvo (passo {passo})")

    # Usa o melhor checkpoint como final
    import shutil
    melhor_path = SAIDAS[tipo_modelo].replace('.pt', '_melhor.pt')
    if os.path.exists(melhor_path):
        shutil.copy(melhor_path, SAIDAS[tipo_modelo])
        print(f"\n  Melhor checkpoint copiado como modelo final")
    else:
        torch.save({'passo': PASSOS, 'modelo': modelo.state_dict(), 'config': cfg_modelo},
                   SAIDAS[tipo_modelo])

    melhor_ppl = math.exp(min(melhor_val, 20))
    print(f"\n{'='*60}")
    print(f"  CONCLUIDO")
    print(f"  Tempo:       {(time.time()-inicio)/60:.1f} min")
    print(f"  Val loss:    {melhor_val:.4f}")
    print(f"  Perplexity:  {melhor_ppl:.1f}")
    print(f"  Salvo em:    {SAIDAS[tipo_modelo]}")
    if device == 'cuda':
        print(f"  VRAM pico:   {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*60}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description='Treino da Keilinks')
    parser.add_argument('--modelo', choices=['flash', 'padrao', 'ultra'], default='padrao')
    parser.add_argument('--batch', type=int, default=None, help='Batch fisico (override)')
    parser.add_argument('--grad-accum', type=int, default=None, help='Passos de acumulacao (override)')
    parser.add_argument('--passos', type=int, default=None, help='Total de passos (override)')
    parser.add_argument('--pretreino', action='store_true', help='Carrega checkpoint pre-treinado antes de fine-tune')
    args = parser.parse_args()
    treinar(args.modelo, batch_override=args.batch, accum_override=args.grad_accum, passos_override=args.passos, usar_pretreino=args.pretreino)
