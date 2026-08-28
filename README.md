# Keilinks — Core 380M autoral em PT-BR

> IA generativa brasileira em desenvolvimento, com Transformer decoder próprio, pré-treinamento e SFT em PyTorch, otimizada para hardware local de consumo.

## Importante

O **Keilinks Core 380M não é um fine-tuning de Qwen, Llama ou outro LLM pronto**.

A arquitetura principal é implementada no próprio projeto em `treino/v4/modelo.py`, os pesos do Core são inicializados pelo Keilinks e o pipeline atual executa **pré-treinamento próprio + SFT conversacional**.

Modelos externos via Ollama, incluindo Qwen quando configurado, podem atuar apenas como **teacher/critic** para gerar e revisar uma parcela limitada de dados sintéticos. Eles ajudam a preparar os dados, mas **não são o cérebro do Keilinks Core 380M**.

## Core 380M

Perfil operacional atual:

- ~380M parâmetros;
- vocabulário de 32.000 tokens;
- 24 camadas;
- hidden size 1.152;
- 18 attention heads / 6 KV heads;
- Grouped Query Attention;
- RMSNorm;
- SwiGLU;
- RoPE;
- QK-Norm;
- weight tying;
- SDPA/Flash Attention quando disponível;
- KV cache para geração;
- contexto de 8.192 tokens;
- BF16, TF32, AdamW 8-bit, gradient checkpointing e `torch.compile` no perfil local.

O modelo é treinado na RTX 5050 Laptop de 8 GB usada como máquina-alvo do projeto.

## Pipeline

```text
Dados PT-BR com origem/licença conhecida
            │
            ▼
 limpeza + deduplicação + proveniência
            │
            ▼
      tokenizador próprio 32k
            │
            ▼
      PRETRAIN Core 380M
            │
            ▼
      checkpoint avaliado
            │
            ▼
       SFT conversacional
            │
            ▼
 avaliação + revisão + promoção
```

A meta configurada para o ciclo longo chega a **160.000 passos**, com aproximadamente **5,24 bilhões de tokens de exposição** no perfil operacional atual.

## Teacher/critic

A geração sintética opcional usa uma arquitetura de professor e crítico:

```text
LLM externo (teacher)
       │
       ▼
conversa candidata
       │
       ▼
LLM externo (critic)
       │
       ▼
filtros + deduplicação
       │
       ▼
parcela sintética do SFT
       │
       ▼
Keilinks Core 380M
```

Dados sintéticos são limitados para não dominar o conjunto. O projeto mantém dados humanos, traduzidos, sintéticos e avaliações congeladas identificados separadamente.

## SFT conversacional PT-BR

A versão V2 prepara 5.000 conversas únicas, combinando:

- 30% diálogo humano com feedback;
- 13% diálogo multi-turno traduzido e auditado;
- 32% instruções humanas;
- 14% instruções sintéticas;
- 7% raciocínio curto;
- 4% âncoras de comportamento Keilinks.

O pacote 8K possui cerca de 1,05 milhão de tokens supervisionados de treino, além da validação separada.

## Outros componentes

Ao redor do Core, o repositório também pesquisa e implementa:

- RAG local;
- pesquisa web com gate de evidência;
- memória pessoal isolada;
- SQLite local;
- visão com VLM separado;
- API e interface local;
- feedback consentido;
- filtros de segurança;
- testes e avaliações de regressão.

Esses recursos são componentes ao redor do modelo e não substituem o treinamento dos pesos do Core.

## Código principal

```text
treino/v4/modelo.py                    # Transformer autoral
treino/v4/config.py                    # perfis de modelo e treino
treino/v4/pretreinar.py                # pré-treinamento
treino/v4/treinar.py                   # SFT
treino/v4/orquestrar_380m.py           # ciclo do Core 380M
treino/v4/tokenizador.py               # tokenizador
treino/v4/gerar_conversas_ollama.py    # teacher/critic
treino/v4/preparar_sft_conversacional.py
busca/                                  # pesquisa e grounding
cerebro/                                # componentes auxiliares
api/                                    # serviços locais
tests/                                  # testes e regressões
```

## Documentação técnica completa

Veja [`README_V5.md`](README_V5.md) para benchmarks, comandos de treino, pipeline de dados e detalhes do ciclo 380M.

O repositório ainda contém experimentos anteriores com Qwen/QLoRA/Unsloth usados como baseline e comparação. **Eles não representam o Core 380M autoral atual.**

---

**Keilinks** — pesquisa prática em modelos de linguagem locais, PT-BR e treinamento autoral em hardware de consumo.
