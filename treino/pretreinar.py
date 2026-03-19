"""
Pré-treino da Keilinks em texto PT-BR
O modelo aprende a LINGUA (gramática, fatos, raciocínio) antes de aprender a CONVERSAR.

Diferença do treino normal:
  - Treina em texto corrido (não em pares <vitor>/<keilinks>)
  - Muito mais dados (~2-5GB de texto)
  - Next-token prediction puro
  - Salva checkpoint pra fine-tune posterior

Uso:
  python treino/pretreinar.py --modelo flash
  python treino/pretreinar.py --modelo flash --passos 50000
  python treino/pretreinar.py --modelo flash --limite-tokens 2000000000  (2B tokens)
"""

import torch
import torch.nn as nn
import sys
import os
import time
import math
import argparse
import glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelo.keilinks import Keilinks, MODELOS
from dados.tokenizador import Tokenizador


# ─── Configs de pré-treino (mais passos, LR menor) ─────────────────────
CONFIGS_PRETREINO = {
    'flash':  {'batch': 2,  'accum': 8,  'passos': 50000, 'lr_max': 3e-4, 'lr_min': 1e-5, 'grad_ckpt': True},
    'padrao': {'batch': 2,  'accum': 8,  'passos': 50000, 'lr_max': 3e-4, 'lr_min': 1e-5, 'grad_ckpt': True},
    'ultra':  {'batch': 1,  'accum': 16, 'passos': 60000, 'lr_max': 2e-4, 'lr_min': 1e-5, 'grad_ckpt': True},
}

WARMUP    = 500
GRAD_CLIP = 1.0
AVALIAR   = 200
SALVAR    = 1000
PACIENCIA = 5000  # Mais paciencia no pretreino (dados maiores = curva mais lenta)


def lr_cosine(passo, total, lr_max, lr_min, warmup):
    if passo < warmup:
        return lr_max * passo / warmup
    p = (passo - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


def carregar_dados_pretreino(tokenizador, contexto, limite_tokens=None):
    """Carrega e tokeniza todos os arquivos de pré-treino"""
    pasta = 'dados/pretreino'
    if not os.path.exists(pasta):
        print("  ERRO: dados/pretreino/ não encontrado")
        print("  Execute: python treino/baixar_pretreino.py")
        return None, None

    arquivos = glob.glob(os.path.join(pasta, '*.txt'))
    if not arquivos:
        print("  ERRO: nenhum arquivo .txt em dados/pretreino/")
        return None, None

    print(f"  Arquivos encontrados: {len(arquivos)}")
    for arq in arquivos:
        tamanho = os.path.getsize(arq) / 1e6
        print(f"    {os.path.basename(arq)}: {tamanho:.1f} MB")

    # Tokeniza arquivo por arquivo (streaming pra não estourar RAM)
    todos_tokens = []
    t0 = time.time()

    for arq in arquivos:
        print(f"\n  Tokenizando {os.path.basename(arq)}...")
        t1 = time.time()

        with open(arq, 'r', encoding='utf-8', errors='replace') as f:
            # Lê em chunks de 10MB pra não estourar RAM
            chunk_size = 10 * 1024 * 1024  # 10MB
            tokens_arq = 0

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                tokens = tokenizador.encode(chunk)
                todos_tokens.extend(tokens)
                tokens_arq += len(tokens)

                if tokens_arq % 5000000 == 0:
                    print(f"    {tokens_arq/1e6:.1f}M tokens...")

                # Limite de tokens
                if limite_tokens and len(todos_tokens) >= limite_tokens:
                    print(f"    Limite de {limite_tokens/1e6:.0f}M tokens atingido")
                    break

        print(f"    {os.path.basename(arq)}: {tokens_arq/1e6:.1f}M tokens em {time.time()-t1:.0f}s")

        if limite_tokens and len(todos_tokens) >= limite_tokens:
            break

    total = len(todos_tokens)
    print(f"\n  Total: {total/1e6:.1f}M tokens em {time.time()-t0:.0f}s")

    if total < 10000:
        print("  ERRO: dados insuficientes pra pré-treino")
        return None, None

    # Split treino/validação (98/2 — mais dados pra treino no pretreino)
    data = torch.tensor(todos_tokens, dtype=torch.long)
    corte = int(0.98 * len(data))

    return data[:corte], data[corte:]


def pegar_batch(data, batch_size, contexto, device):
    ctx = min(contexto, len(data) - 1)
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
    return sum(losses) / len(losses)


def pretreinar(tipo_modelo: str, passos_override=None, limite_tokens=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    cfg_treino = CONFIGS_PRETREINO[tipo_modelo].copy()
    cfg_modelo = MODELOS[tipo_modelo].copy()

    BATCH     = cfg_treino['batch']
    ACCUM     = cfg_treino['accum']
    PASSOS    = passos_override or cfg_treino['passos']
    LR_MAX    = cfg_treino['lr_max']
    LR_MIN    = cfg_treino['lr_min']
    GRAD_CKPT = cfg_treino['grad_ckpt']
    CONTEXTO  = cfg_modelo['contexto_max']
    batch_efetivo = BATCH * ACCUM

    print("=" * 60)
    print(f"  Keilinks PRÉ-TREINO — Aprendendo Português")
    print("=" * 60)
    print(f"  Modelo:      {cfg_modelo['nome']}")
    print(f"  Device:      {device.upper()}")
    if device == 'cuda':
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:        {vram:.1f} GB")
    print(f"  Batch:       {BATCH} x {ACCUM} = {batch_efetivo}")
    print(f"  Passos:      {PASSOS}")
    print(f"  Contexto:    {CONTEXTO}")
    print(f"  LR:          {LR_MAX} -> {LR_MIN}")
    print(f"  Fase:        PRÉ-TREINO (texto corrido PT-BR)")
    print("=" * 60)

    # Tokenizador — constroi vocab nos dados de pretreino
    tokenizador = Tokenizador()

    # Carrega uma amostra pra construir vocab
    print("\n  Construindo vocabulário nos dados de pré-treino...")
    textos_amostra = []
    pasta = 'dados/pretreino'
    for arq in glob.glob(os.path.join(pasta, '*.txt')):
        with open(arq, 'r', encoding='utf-8', errors='replace') as f:
            # Lê primeiros 20MB de cada arquivo pra vocab
            textos_amostra.append(f.read(20 * 1024 * 1024))

    # Inclui dados conversacionais no vocab também
    conv_path = 'dados/conversas.txt'
    if os.path.exists(conv_path):
        with open(conv_path, 'r', encoding='utf-8') as f:
            textos_amostra.append(f.read())

    vocab_alvo = cfg_modelo.get('vocab_size', 32000)
    tokenizador.construir_vocab(textos_amostra, vocab_alvo=vocab_alvo)
    tokenizador.salvar('dados/vocab_pretreino.json')

    # Carrega e tokeniza dados
    treino_data, val_data = carregar_dados_pretreino(tokenizador, CONTEXTO, limite_tokens)
    if treino_data is None:
        return

    CONTEXTO_REAL = min(CONTEXTO, len(treino_data) - 1, len(val_data) - 1)

    print(f"\n  Treino: {len(treino_data)/1e6:.1f}M tokens | Val: {len(val_data)/1e6:.1f}M tokens")
    print(f"  Contexto: {CONTEXTO_REAL}")
    print(f"  Chinchilla ideal pra {cfg_modelo['nome']}: ~5B tokens")
    print(f"  Seus dados: {len(treino_data)/1e9:.2f}B tokens ({len(treino_data)/5e9*100:.1f}% do ideal)")

    # Modelo
    cfg_modelo['vocab_size'] = tokenizador.tam_vocab
    cfg_modelo['eos_id'] = tokenizador.vocab.get(tokenizador.EOS, 1)

    modelo = Keilinks(cfg_modelo).to(device)

    if GRAD_CKPT:
        modelo.usar_grad_checkpoint = True
        print(f"  Gradient checkpointing ATIVADO")

    # Retoma checkpoint se existir
    passo_inicial = 0
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_pretreino = f'checkpoints/keilinks_{tipo_modelo}_pretreino_tmp.pt'

    if os.path.exists(ckpt_pretreino):
        ckpt = torch.load(ckpt_pretreino, map_location=device, weights_only=False)
        ckpt_dim = ckpt['config'].get('dim', 0)
        if ckpt_dim == cfg_modelo['dim']:
            state = {k: v for k, v in ckpt['modelo'].items() if 'embedding_posicao' not in k}
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
            passo_inicial = ckpt['passo']
            print(f"  Retomando pré-treino do passo {passo_inicial}")
        else:
            print(f"  Arquitetura mudou, pré-treinando do zero")

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1
    )

    melhor_val = float('inf')
    passo_melhor = 0
    inicio = time.time()

    saida_final = f'checkpoints/keilinks_{tipo_modelo}_pretreino.pt'

    print(f"\n  Iniciando pré-treino...\n")

    for passo in range(passo_inicial, PASSOS):
        lr = lr_cosine(passo, PASSOS, LR_MAX, LR_MIN, WARMUP)
        for g in otimizador.param_groups:
            g['lr'] = lr

        otimizador.zero_grad(set_to_none=True)
        loss_acum = 0.0

        for micro in range(ACCUM):
            x, y = pegar_batch(treino_data, BATCH, CONTEXTO_REAL, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                _, loss = modelo(x, y)
                loss_escalonada = loss / ACCUM

            loss_escalonada.backward()
            loss_acum += loss.item()

        loss_acum /= ACCUM

        torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
        otimizador.step()

        if passo % AVALIAR == 0:
            loss_val = avaliar(modelo, val_data, BATCH, CONTEXTO_REAL, device)
            perplexity = math.exp(min(loss_val, 20))
            elapsed = time.time() - inicio
            tok_s = (passo * batch_efetivo * CONTEXTO_REAL) / max(elapsed, 1)
            melhor = ''

            if loss_val < melhor_val:
                melhor_val = loss_val
                passo_melhor = passo
                torch.save({
                    'passo': passo,
                    'modelo': modelo.state_dict(),
                    'config': cfg_modelo,
                    'fase': 'pretreino',
                }, saida_final)
                melhor = ' *MELHOR*'

            vram_info = ''
            if device == 'cuda':
                vram_usada = torch.cuda.max_memory_allocated() / 1e9
                vram_info = f' | VRAM {vram_usada:.1f}G'

            # ETA
            if passo > passo_inicial:
                s_por_passo = elapsed / (passo - passo_inicial)
                restante = (PASSOS - passo) * s_por_passo
                eta = f' | ETA {restante/3600:.1f}h'
            else:
                eta = ''

            print(f"[{passo:>6}/{PASSOS}] loss {loss_acum:.4f} | val {loss_val:.4f} | "
                  f"ppl {perplexity:.1f} | lr {lr:.1e} | {tok_s:.0f} tok/s{vram_info}{eta}{melhor}")

            # Early stopping
            if passo - passo_melhor >= PACIENCIA and passo > WARMUP:
                print(f"\n  EARLY STOPPING: val loss nao melhorou por {PACIENCIA} passos")
                print(f"  Melhor val loss: {melhor_val:.4f} (passo {passo_melhor})")
                break

        if passo % SALVAR == 0 and passo > 0:
            torch.save({
                'passo': passo,
                'modelo': modelo.state_dict(),
                'otimizador': otimizador.state_dict(),
                'config': cfg_modelo,
                'fase': 'pretreino',
            }, ckpt_pretreino)
            print(f"  >> checkpoint salvo (passo {passo})")

    # Copia vocab do pretreino como vocab principal
    import shutil
    shutil.copy('dados/vocab_pretreino.json', 'dados/vocab.json')

    melhor_ppl = math.exp(min(melhor_val, 20))
    print(f"\n{'='*60}")
    print(f"  PRÉ-TREINO CONCLUÍDO")
    print(f"{'='*60}")
    print(f"  Tempo:       {(time.time()-inicio)/3600:.1f}h")
    print(f"  Val loss:    {melhor_val:.4f}")
    print(f"  Perplexity:  {melhor_ppl:.1f}")
    print(f"  Salvo em:    {saida_final}")
    print(f"  Vocab:       dados/vocab.json")
    if device == 'cuda':
        print(f"  VRAM pico:   {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"\n  PRÓXIMO PASSO:")
    print(f"    python treino/treinar.py --modelo {tipo_modelo} --pretreino")
    print(f"{'='*60}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description='Pré-treino da Keilinks')
    parser.add_argument('--modelo', choices=['flash', 'padrao', 'ultra'], default='flash')
    parser.add_argument('--passos', type=int, default=None, help='Total de passos (override)')
    parser.add_argument('--limite-tokens', type=int, default=None, help='Limite de tokens (ex: 2000000000 pra 2B)')
    args = parser.parse_args()
    pretreinar(args.modelo, passos_override=args.passos, limite_tokens=args.limite_tokens)
