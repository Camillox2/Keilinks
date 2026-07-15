# Keilinks V4

A V4 separa quatro responsabilidades:

1. **Pré-treino:** português, padrões gerais e conhecimento estável.
2. **SFT:** comportamento, identidade, segurança e estilo conversacional.
3. **Memória/RAG:** dados pessoais, documentos e conhecimento atualizável.
4. **Busca web:** notícias, preços, cargos, leis e fatos atuais com URLs de fonte.

Respostas da internet ou conversas isoladas não alteram os pesos automaticamente. Dados novos entram como candidatos para revisão.

## Principais mudanças

- Transformer decoder próprio com GQA, RoPE, RMSNorm, SwiGLU, SDPA e weight tying.
- Perfis de aproximadamente 380M, 500M, 800M e 1B parâmetros.
- Perfil inicial recomendado para RTX 5050 Laptop de 8 GB: `core_380m`.
- BF16, TF32, AdamW fused/8-bit, `torch.compile`, prefetch e checkpointing seletivo.
- SFT com loss apenas nos tokens do assistant (`ignore_index=-100`).
- Split por grupo, packing e deduplicação exata + SimHash.
- Tokenizador BPE V4 com `<sistema>`, `<vitor>`, `<user>` e `<keilinks>`.
- FineWeb2 PT, Wikipedia PT, OASST2 e fontes PT-BR opcionais baixadas localmente.
- Geração sintética local com professor + crítico via Ollama.
- Mix SFT com limite para dados sintéticos e traduções.
- Avaliação congelada antes de promover checkpoints.
- Busca por SearXNG, Brave, DDG e Wikipedia, com cache, ranking e URLs citáveis.
- Retreino automático desligado por padrão.

## 1. Ambiente

Use WSL2/Ubuntu e mantenha o projeto em `/home/<usuario>/Keilinks`, não em `/mnt/c`.

```bash
bash scripts/setup_wsl_v4.sh
source .venv-v4/bin/activate
# Instale a build CUDA indicada atualmente em pytorch.org.
pip install -r requirements-v4.txt
```

Feche o Ollama durante benchmark e treinamento para liberar VRAM.

## 2. Benchmark da RTX 5050

```bash
python -m treino.v4.benchmark_rtx5050 \
  --model core_380m \
  --context 1024 \
  --steps 10
```

Compare tokens/s e pico de VRAM entre eager/compile e checkpoint `none`, `selective` e `full`. Repita com contexto 2048 somente se houver margem de VRAM.

## 3. Conversas curadas

```bash
python -m treino.v4.gerar_seed
```

O arquivo gerado fica em `dados/v4/seed_conversas.jsonl` e não é enviado ao Git.

## 4. Download local do corpus

Comece pequeno:

```bash
python -m treino.v4.preparar_dados download \
  --pretrain-gb 0.25 \
  --wiki-gb 0.10 \
  --sources fineweb2,wikipedia,oasst2
```

Após revisar amostras e manifestos, faça a primeira coleta principal:

```bash
python -m treino.v4.preparar_dados download \
  --pretrain-gb 10 \
  --wiki-gb 1.5 \
  --sources fineweb2,wikipedia,oasst2,alpaca,dolly
```

Alpaca e Dolly são opcionais. Verifique o card e a licença atual antes de uso comercial ou redistribuição.

## 5. Conversas sintéticas locais

O Ollama pode gerar dados revisados em duas passagens:

```bash
ollama serve
python -m treino.v4.gerar_conversas_ollama \
  --model qwen3.5:4b \
  --target 50000 \
  --batch-size 8 \
  --min-score 7
```

A tag do modelo deve ser ajustada ao nome instalado no seu Ollama. Sintéticos não devem dominar o SFT.

## 6. Mix balanceado do SFT

```bash
python -m treino.v4.misturar_sft \
  --max-examples 200000 \
  --synthetic-ratio 0.25 \
  --translation-ratio-each 0.20
```

Isso cria `dados/v4/sft/all_sft.jsonl` e um manifesto com a distribuição por fonte e categoria.

## 7. Construir o tokenizador V4

```bash
python -m treino.v4.tokenizador build \
  --output dados/vocab_v4.json \
  --pretrain dados/v4/pretrain/pretrain_pt.txt \
  --sft dados/v4/sft/all_sft.jsonl \
  --vocab-size 32000 \
  --sample-mb 256

python -m treino.v4.tokenizador inspect \
  --vocab dados/vocab_v4.json
```

O vocabulário e seus metadados ficam locais. O hash é registrado nos binários de pré-treino.

## 8. Preparar o pré-treino

```bash
python -m treino.v4.pretreinar \
  --prepare-only \
  --rebuild-binary \
  --input dados/v4/pretrain/pretrain_pt.txt \
  --vocab dados/vocab_v4.json
```

O split é feito por documento antes da tokenização. Arquivos temporários são substituídos atomicamente apenas após a preparação terminar.

## 9. Smoke test e pré-treino real

Faça primeiro somente 20 passos:

```bash
python -m treino.v4.pretreinar \
  --model core_380m \
  --profile rtx5050_380m \
  --steps 20 \
  --output checkpoints/v4-smoke-pretrain
```

Depois do smoke test, o comando de treino completo é:

```bash
python -m treino.v4.pretreinar \
  --model core_380m \
  --profile rtx5050_380m \
  --output checkpoints/v4-pretrain
```

Treinar 800M–1B do zero em 8 GB continua experimental e muito lento. O pipeline suporta esses perfis, mas o benchmark real deve decidir o que é viável.

## 10. Empacotar o SFT

```bash
python -m treino.v4.pack_sft \
  --vocab dados/vocab_v4.json \
  --input dados/v4/sft/all_sft.jsonl \
  --output dados/v4/packed \
  --context 2048
```

Sistema, usuário e padding recebem label `-100`. Apenas a resposta da Keilinks gera gradiente.

## 11. SFT a partir do pré-treino

Smoke test:

```bash
python -m treino.v4.treinar \
  --model core_380m \
  --profile rtx5050_sft_380m \
  --data dados/v4/packed \
  --init-checkpoint checkpoints/v4-pretrain/pretrain_best.pt \
  --steps 20 \
  --output checkpoints/v4-smoke-sft
```

Treino SFT completo:

```bash
python -m treino.v4.treinar \
  --model core_380m \
  --profile rtx5050_sft_380m \
  --data dados/v4/packed \
  --init-checkpoint checkpoints/v4-pretrain/pretrain_best.pt \
  --output checkpoints/v4-sft
```

O script recusa inicialização aleatória por padrão.

## 12. Avaliar antes de promover

```bash
python -m treino.v4.avaliar_checkpoint \
  --checkpoint checkpoints/v4-sft/keilinks_v4.pt \
  --vocab dados/vocab_v4.json
```

O conjunto congelado cobre identidade, segurança, privacidade, factualidade, tecnologia, trabalho e respostas emocionais. Casos atuais podem ser testados separadamente com `--with-web`.

## 13. Servidor V4

```bash
export KEILINKS_ADMIN_TOKEN='uma-chave-longa-e-aleatoria'
export KEILINKS_V4_CHECKPOINT='checkpoints/v4-sft/keilinks_v4.pt'
export KEILINKS_V4_VOCAB='dados/vocab_v4.json'
export SEARXNG_URL='http://seu-searxng:8080' # opcional
# ou export BRAVE_SEARCH_API_KEY='...'
python -m api.servidor_v4
```

Quando o checkpoint V4 existe, o servidor não carrega os três modelos legados na VRAM. Sem V4, ele faz fallback para o servidor antigo. Endpoints de ensino e crawl exigem token fora do localhost.

## Licenças e dados grandes

- FineWeb2: confira o card atual e os termos dos conteúdos de origem.
- OASST2: confira o card atual antes da distribuição.
- Wikipedia: conteúdo sob licenças indicadas pelo projeto Wikimedia.
- Alpaca/Dolly traduzidos: opcionais; confirmar card e licença atual.

Corpora, binários, ambientes virtuais e checkpoints são ignorados pelo Git. O repositório guarda apenas código, amostras pequenas, manifestos de formato e avaliações congeladas.
