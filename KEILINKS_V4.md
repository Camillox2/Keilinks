# Keilinks V4

A V4 separa pré-treino, SFT de personalidade, memória/RAG e dados atuais da web. Respostas da internet ou conversas isoladas não alteram os pesos automaticamente.

## Mudanças

- Transformer próprio com GQA, RoPE, RMSNorm, SwiGLU, SDPA e weight tying.
- Perfis de aproximadamente 380M, 500M, 800M e 1B parâmetros.
- Perfil recomendado para RTX 5050 8 GB: `core_380m`.
- `torch.compile`, AdamW fused/8-bit, BF16, TF32, prefetch e checkpointing seletivo.
- SFT com loss apenas nos tokens da Keilinks (`ignore_index=-100`).
- Split por grupo e deduplicação exata + SimHash.
- FineWeb2 PT, Wikipedia PT, OASST2 e fontes PT-BR baixadas localmente.
- Busca web com SearXNG, Brave ou DDG, cache e URLs citáveis.
- Retreino automático desligado por padrão.

## Instalação

Use WSL2/Ubuntu e mantenha o projeto em `/home/<usuario>/Keilinks`, não em `/mnt/c`.

```bash
bash scripts/setup_wsl_v4.sh
source .venv-v4/bin/activate
# instale PyTorch CUDA conforme pytorch.org
pip install -r requirements-v4.txt
```

Feche o Ollama antes de treinar para liberar VRAM.

## Benchmark da GPU

```bash
python -m treino.v4.benchmark_rtx5050 --model core_380m --context 1024 --steps 10
```

O resultado mede tokens/s, pico de VRAM e a melhor combinação de compile/checkpoint.

## Conversas curadas e corpus

```bash
python -m treino.v4.gerar_seed
python -m treino.v4.preparar_dados download --pretrain-gb 10 --wiki-gb 1.5
```

Para um alvo próximo de 1B, aumente gradualmente para dezenas de GB conforme disco e tempo. Os arquivos grandes não devem ser enviados ao GitHub.

## Pré-treino

```bash
python -m treino.v4.pretreinar --prepare-only --rebuild-binary
python -m treino.v4.pretreinar --model core_380m --profile rtx5050_380m
```

Treinar 1B do zero em 8 GB é experimental e muito lento. Para produção, prefira 380M/500M do zero ou QLoRA em uma base 3B–4B.

## SFT

```bash
python -m treino.v4.preparar_dados pack-sft --vocab dados/vocab.json --context 2048
python -m treino.v4.treinar --model core_380m --profile rtx5050_380m
```

Promova checkpoint novo somente após benchmark congelado e revisão humana.

## Servidor

```bash
export KEILINKS_ADMIN_TOKEN='uma-chave-longa'
export SEARXNG_URL='http://seu-searxng:8080' # opcional
# ou export BRAVE_SEARCH_API_KEY='...'
python -m api.servidor_v4
```

Sem provedor configurado, a busca tenta DDG e Wikipedia como fallback. Endpoints de ensino remoto exigem token.

## Licenças

- FineWeb2: ODC-By 1.0.
- OASST2: Apache 2.0.
- Wikipedia: CC BY-SA 3.0/GFDL.
- Alpaca/Dolly traduzidos: conferir o card antes de distribuição comercial.

Fatos atuais, preços, leis, notícias e cargos ficam no RAG/web, não nos pesos.
