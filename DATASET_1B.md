# Plano de dados para Keilinks até 1B

## O número importante é tokens, não apenas conversas

Dez milhões de conversas com média de 300 a 600 tokens representam aproximadamente 3 a 6 bilhões de tokens. Para um modelo denso de 1B treinado do zero, isso ainda pode deixá-lo subtreinado. Como referência de planejamento, use cerca de 20 bilhões de tokens de pré-treino limpo como alvo de longo prazo e valide curvas de loss antes de aumentar o volume.

O SFT tem outra função: ensinar comportamento, diálogo, formato, segurança e estilo. Ele não deve ser usado como substituto do conhecimento geral aprendido no pré-treino.

## Metas recomendadas

### Pré-treino

- Primeiro smoke: 10 a 50 milhões de tokens.
- Primeira versão mensurável: 500 milhões a 2 bilhões de tokens.
- Core 380M: buscar vários bilhões de tokens limpos.
- Core 1B: alvo de longo prazo próximo de 20 bilhões de tokens, condicionado a tempo e compute.

### SFT conversacional

- Base curada manual: milhares de exemplos variados.
- Primeira geração auditada: 100 mil conversas.
- Segunda etapa: até 1 milhão, somente se o checkpoint melhorar.
- Capacidade máxima do pipeline: 10 milhões em shards.
- Não avançar automaticamente para 10 milhões sem avaliação e auditoria.

## Gerar em escala

O gerador usa shards de 100 mil, SQLite para deduplicação e retomada.

Primeiro estágio:

```bash
python -m treino.v4.gerar_conversas_em_escala \
  --model qwen3.5:4b \
  --critic-model qwen3.5:4b \
  --target 10000000 \
  --approved-stage 100000 \
  --shard-size 100000 \
  --batch-size 8 \
  --min-score 8
```

Depois de auditar amostras e comparar um checkpoint, liberar a etapa seguinte:

```bash
python -m treino.v4.gerar_conversas_em_escala \
  --model qwen3.5:4b \
  --critic-model qwen3.5:4b \
  --target 10000000 \
  --approved-stage 1000000
```

A etapa de 10 milhões só deve ser liberada se 1 milhão superar a etapa anterior nos benchmarks:

```bash
python -m treino.v4.gerar_conversas_em_escala \
  --model qwen3.5:4b \
  --critic-model qwen3.5:4b \
  --target 10000000 \
  --approved-stage 10000000
```

Arquivos locais:

```text
dados/v4/sft/generated_1b/
├── dedup.sqlite3
├── manifest.json
├── shards/
│   ├── conversations-00000.jsonl
│   ├── conversations-00001.jsonl
│   └── ...
└── rejected/
```

Esses arquivos são ignorados pelo Git.

## Packing streaming

O packer em escala não carrega os exemplos em listas Python:

```bash
python -m treino.v4.pack_sft_em_escala \
  --vocab dados/vocab_v4.json \
  --output dados/v4/packed-scale \
  --context 2048 \
  dados/v4/conversas_curadas_v4.jsonl \
  dados/v4/seed_conversas.jsonl \
  dados/v4/sft/oasst2_pt.jsonl \
  'dados/v4/sft/generated_1b/shards/*.jsonl'
```

Não use todos os shards sintéticos cegamente. Se os sintéticos dominarem o corpus, crie um subconjunto com hash determinístico e mantenha dados humanos e texto real no mix.

## Critérios para avançar de etapa

- Auditoria aleatória de pelo menos 1.000 exemplos por estágio.
- Menos de 1% de respostas factualmente inventadas na amostra.
- Sem dependência emocional, consciência fingida ou vazamento de marcadores.
- Diversidade lexical e temática maior que o estágio anterior.
- Validation loss não piora.
- Win rate contra o checkpoint anterior acima de 50% com intervalo de confiança.
- Segurança e factualidade não regridem.
- Dados sintéticos não substituem o corpus real de pré-treino.

## Referências de planejamento

- Chinchilla: https://arxiv.org/abs/2203.15556
- LIMA: https://arxiv.org/abs/2305.11206
- Textbooks Are All You Need: https://arxiv.org/abs/2306.11644
- Synthetic-data collapse analysis: https://arxiv.org/abs/2404.05090
