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
  python treino/pretreinar.py --modelo flash --rebuild-vocab  (força reconstruir vocab)
"""

import torch
import torch.nn as nn
import sys
import os
import time
import math
import argparse
import glob
import numpy as np

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


def tokenizar_para_disco(tokenizador, limite_tokens=None):
    """
    Tokeniza arquivos pra disco em 1 passagem com checkpoint a cada 100M tokens.
    Se parar no meio, retoma de onde parou.
    """
    import json as _json

    pasta = 'dados/pretreino'
    bin_treino = 'dados/pretreino/treino.bin'
    bin_val    = 'dados/pretreino/val.bin'
    bin_total  = 'dados/pretreino/total.txt'
    progress_file = 'dados/pretreino/tokenize_progress.json'

    if not os.path.exists(pasta):
        print("  ERRO: dados/pretreino/ não encontrado")
        print("  Execute: python treino/baixar_pretreino.py")
        return None, None, 0

    arquivos = sorted(glob.glob(os.path.join(pasta, '*.txt')))
    if not arquivos:
        print("  ERRO: nenhum arquivo .txt em dados/pretreino/")
        return None, None, 0

    print(f"  Arquivos encontrados: {len(arquivos)}")
    for arq in arquivos:
        print(f"    {os.path.basename(arq)}: {os.path.getsize(arq)/1e6:.1f} MB")

    # Se já terminou antes, reutiliza
    if os.path.exists(bin_treino) and os.path.exists(bin_val) and os.path.exists(bin_total):
        with open(bin_total) as f:
            total = int(f.read().strip())
        print(f"  Binários prontos — {total/1e6:.1f}M tokens (reutilizando)")
        corte = int(0.98 * total)
        treino = np.memmap(bin_treino, dtype=np.int32, mode='r', shape=(corte,))
        val    = np.memmap(bin_val,    dtype=np.int32, mode='r', shape=(total - corte,))
        return treino, val, total

    # --- Tokenização em 1 passagem com checkpoint ---

    # Carregar progresso anterior (se existir)
    progress = {}
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = _json.load(f)

    # Cada arquivo vira um .bin próprio (append)
    t0 = time.time()
    bytes_totais = sum(os.path.getsize(a) for a in arquivos)
    bytes_antes = 0
    total_tokens_global = 0
    checkpoint_intervalo = 100_000_000  # salva progresso a cada 100M tokens

    print("\n  Tokenizando para disco (1 passagem, checkpoint a cada 100M tokens)...")

    for arq in arquivos:
        nome = os.path.basename(arq)
        bin_arq = os.path.join(pasta, nome.replace('.txt', '.bin'))
        arq_size = os.path.getsize(arq)

        # Verificar progresso anterior deste arquivo
        pos_bytes_inicio = 0
        tokens_ja_feitos = 0
        if nome in progress:
            pos_bytes_inicio = progress[nome]['pos_bytes']
            tokens_ja_feitos = progress[nome]['tokens']
            if progress[nome].get('completo', False):
                print(f"\n  {nome}: já tokenizado ({tokens_ja_feitos/1e6:.1f}M tokens) — pulando")
                total_tokens_global += tokens_ja_feitos
                bytes_antes += arq_size
                continue
            else:
                print(f"\n  {nome}: retomando de {pos_bytes_inicio/1e6:.1f}MB ({tokens_ja_feitos/1e6:.1f}M tokens)")
        else:
            print(f"\n  {nome}: tokenizando do zero...")
            # Limpa bin anterior se existir sem progresso
            if os.path.exists(bin_arq):
                os.remove(bin_arq)

        tokens_arq = tokens_ja_feitos
        ultimo_checkpoint = tokens_ja_feitos
        pos_bytes = pos_bytes_inicio

        with open(arq, 'rb') as f, open(bin_arq, 'ab') as fb:
            if pos_bytes_inicio > 0:
                f.seek(pos_bytes_inicio)

            chunk_size = 10 * 1024 * 1024  # 10MB

            while True:
                raw = f.read(chunk_size)
                if not raw:
                    break

                chunk = raw.decode('utf-8', errors='replace')
                toks = np.array(tokenizador.encode(chunk), dtype=np.int32)

                fb.write(toks.tobytes())

                tokens_arq += len(toks)
                pos_bytes = f.tell()

                # Progresso visual
                bytes_agora = bytes_antes + pos_bytes
                pct = min(bytes_agora / bytes_totais * 100, 100)
                elapsed = time.time() - t0
                tokens_nesta_sessao = (tokens_arq - tokens_ja_feitos)
                tok_s = max(tokens_nesta_sessao / max(elapsed, 1), 1)
                print(f"\r    {nome}: {pct:.1f}% | {tokens_arq/1e6:.1f}M tokens | {tok_s/1e6:.1f}M tok/s   ", end='', flush=True)

                # Checkpoint a cada 100M tokens
                if tokens_arq - ultimo_checkpoint >= checkpoint_intervalo:
                    progress[nome] = {'pos_bytes': pos_bytes, 'tokens': tokens_arq, 'completo': False}
                    with open(progress_file, 'w') as fp:
                        _json.dump(progress, fp)
                    ultimo_checkpoint = tokens_arq

                # Limite de tokens
                if limite_tokens and (total_tokens_global + tokens_arq - tokens_ja_feitos) >= limite_tokens:
                    print(f"\n    Limite de {limite_tokens/1e6:.0f}M tokens atingido")
                    break

        # Marcar arquivo como completo
        progress[nome] = {'pos_bytes': pos_bytes, 'tokens': tokens_arq, 'completo': True}
        with open(progress_file, 'w') as fp:
            _json.dump(progress, fp)

        total_tokens_global += tokens_arq - tokens_ja_feitos
        bytes_antes += arq_size

        print(f"\n    {nome}: {tokens_arq/1e6:.1f}M tokens — concluído")

        if limite_tokens and total_tokens_global >= limite_tokens:
            break

    print(f"\n  Total: {total_tokens_global/1e6:.1f}M tokens em {time.time()-t0:.0f}s")

    if total_tokens_global < 10000:
        print("  ERRO: dados insuficientes")
        return None, None, 0

    # Combinar .bin de cada arquivo em treino.bin + val.bin
    print("\n  Combinando em treino.bin + val.bin...")
    corte = int(0.98 * total_tokens_global)

    mm_treino = np.memmap(bin_treino, dtype=np.int32, mode='w+', shape=(corte,))
    mm_val    = np.memmap(bin_val,    dtype=np.int32, mode='w+', shape=(total_tokens_global - corte,))

    pos = 0
    for arq in arquivos:
        nome = os.path.basename(arq)
        bin_arq = os.path.join(pasta, nome.replace('.txt', '.bin'))
        if not os.path.exists(bin_arq):
            continue

        n_tokens_arq = os.path.getsize(bin_arq) // 4  # int32 = 4 bytes
        mm_arq = np.memmap(bin_arq, dtype=np.int32, mode='r', shape=(n_tokens_arq,))

        # Copiar em blocos de 10M tokens pra não estourar RAM
        bloco_size = 10_000_000
        for i in range(0, n_tokens_arq, bloco_size):
            fim_bloco = min(i + bloco_size, n_tokens_arq)
            bloco = np.array(mm_arq[i:fim_bloco])
            fim_global = min(pos + len(bloco), total_tokens_global)
            n = fim_global - pos

            treino_fim = min(pos + n, corte)
            if pos < corte:
                qtd_treino = treino_fim - pos
                mm_treino[pos:treino_fim] = bloco[:qtd_treino]
            if pos + n > corte:
                val_ini = max(pos, corte)
                mm_val[val_ini - corte:pos + n - corte] = bloco[val_ini - pos:n]
            pos += n

            if pos >= total_tokens_global:
                break

        del mm_arq
        if pos >= total_tokens_global:
            break

    mm_treino.flush()
    mm_val.flush()

    with open(bin_total, 'w') as f:
        f.write(str(total_tokens_global))

    print(f"  Pronto: treino={corte/1e6:.1f}M | val={(total_tokens_global-corte)/1e6:.1f}M tokens")

    treino = np.memmap(bin_treino, dtype=np.int32, mode='r', shape=(corte,))
    val    = np.memmap(bin_val,    dtype=np.int32, mode='r', shape=(total_tokens_global - corte,))
    return treino, val, total_tokens_global


def pegar_batch(data, batch_size, contexto, device):
    """Pega batch direto do memmap — sem carregar tudo na RAM"""
    ctx = min(contexto, len(data) - 1)
    ix = np.random.randint(0, len(data) - ctx, size=(batch_size,))
    # np.stack copia do memmap, torch converte int32→int64 direto
    x = torch.as_tensor(np.stack([data[i:i+ctx]   for i in ix]), dtype=torch.long, device=device)
    y = torch.as_tensor(np.stack([data[i+1:i+ctx+1] for i in ix]), dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def avaliar(modelo, val_data, batch_size, contexto, device, criterio=None, n=10):
    modelo.eval()
    losses = []
    for _ in range(n):
        x, y = pegar_batch(val_data, batch_size, contexto, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits, _ = modelo(x)
            if criterio:
                loss = criterio(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    modelo.train()
    return sum(losses) / len(losses)


def pretreinar(tipo_modelo: str, passos_override=None, limite_tokens=None, rebuild_vocab=False):
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

    vocab_path = 'dados/vocab_pretreino.json'
    if os.path.exists(vocab_path) and not rebuild_vocab:
        print(f"\n  Vocab já existe ({vocab_path}) — reutilizando")
        tokenizador.carregar(vocab_path)
    else:
        print("\n  Construindo vocabulário nos dados de pré-treino...")
        textos_amostra = []
        pasta = 'dados/pretreino'
        for arq in glob.glob(os.path.join(pasta, '*.txt')):
            with open(arq, 'r', encoding='utf-8', errors='replace') as f:
                textos_amostra.append(f.read(20 * 1024 * 1024))

        conv_path = 'dados/conversas.txt'
        if os.path.exists(conv_path):
            with open(conv_path, 'r', encoding='utf-8') as f:
                textos_amostra.append(f.read())

        vocab_alvo = cfg_modelo.get('vocab_size', 32000)
        tokenizador.construir_vocab(textos_amostra, vocab_alvo=vocab_alvo)
        tokenizador.salvar(vocab_path)

    # Tokeniza pra disco (memmap) — não carrega na RAM
    treino_data, val_data, total_tokens = tokenizar_para_disco(tokenizador, limite_tokens)
    if treino_data is None:
        return

    CONTEXTO_REAL = min(CONTEXTO, len(treino_data) - 1, len(val_data) - 1)

    print(f"\n  Treino: {len(treino_data)/1e6:.1f}M tokens | Val: {len(val_data)/1e6:.1f}M tokens")
    print(f"  Contexto: {CONTEXTO_REAL}")
    print(f"  Chinchilla ideal pra {cfg_modelo['nome']}: ~5B tokens")
    print(f"  Seus dados: {total_tokens/1e9:.2f}B tokens ({total_tokens/5e9*100:.1f}% do ideal)")

    # Modelo
    cfg_modelo['vocab_size'] = tokenizador.tam_vocab
    cfg_modelo['eos_id'] = tokenizador.vocab.get(tokenizador.EOS, 1)

    modelo = Keilinks(cfg_modelo).to(device)

    if GRAD_CKPT:
        modelo.usar_grad_checkpoint = True
        print(f"  Gradient checkpointing ATIVADO")

    # Label smoothing pra melhor generalização
    criterio = nn.CrossEntropyLoss(label_smoothing=0.1)

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

            # Restaura optimizer do mesmo checkpoint (já carregado)
            otimizador_state = ckpt.get('otimizador', None)
        else:
            print(f"  Arquitetura mudou, pré-treinando do zero")
            otimizador_state = None
        del ckpt
    else:
        otimizador_state = None

    # torch.compile DEPOIS de carregar checkpoint (senão compila pesos errados)
    modelo_exec = modelo
    if hasattr(torch, 'compile') and device == 'cuda':
        try:
            modelo_exec = torch.compile(modelo)
            print(f"  torch.compile ATIVADO")
        except Exception as e:
            print(f"  torch.compile falhou ({e}), usando modo normal")

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1
    )

    if otimizador_state is not None:
        try:
            otimizador.load_state_dict(otimizador_state)
            print(f"  Optimizer restaurado do checkpoint")
        except Exception:
            print(f"  Optimizer incompatível, reiniciando")
        del otimizador_state

    melhor_val = float('inf')
    passo_melhor = 0
    inicio = time.time()

    saida_final = f'checkpoints/keilinks_{tipo_modelo}_pretreino.pt'

    # Log CSV pra plotar depois
    log_csv = f'checkpoints/pretreino_{tipo_modelo}_log.csv'
    if passo_inicial == 0 or not os.path.exists(log_csv):
        with open(log_csv, 'w') as f:
            f.write('passo,loss_treino,loss_val,perplexity,lr,tok_s\n')

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
                logits, _ = modelo_exec(x)  # sem targets pra não calcular loss interno
                loss = criterio(logits.view(-1, logits.size(-1)), y.view(-1))
                loss_escalonada = loss / ACCUM

            loss_escalonada.backward()
            loss_acum += loss.item()

        loss_acum /= ACCUM

        torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
        otimizador.step()

        if passo % AVALIAR == 0 and passo > passo_inicial:
            loss_val = avaliar(modelo, val_data, BATCH, CONTEXTO_REAL, device, criterio=criterio)
            perplexity = math.exp(min(loss_val, 20))
            elapsed = time.time() - inicio
            passos_feitos = max(passo - passo_inicial, 1)
            tok_s = (passos_feitos * batch_efetivo * CONTEXTO_REAL) / max(elapsed, 1)
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

            # Log CSV
            with open(log_csv, 'a') as f:
                f.write(f"{passo},{loss_acum:.6f},{loss_val:.6f},{perplexity:.2f},{lr:.2e},{tok_s:.0f}\n")

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
    parser.add_argument('--rebuild-vocab', action='store_true', help='Força reconstrução do vocabulário')
    args = parser.parse_args()
    pretreinar(args.modelo, passos_override=args.passos, limite_tokens=args.limite_tokens, rebuild_vocab=args.rebuild_vocab)
