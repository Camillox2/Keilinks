# Keilinks — Core 380M autoral em PT-BR

> IA generativa brasileira em desenvolvimento, com Transformer decoder próprio, pré-treinamento e SFT em PyTorch, otimizada para rodar e treinar localmente em uma RTX 5050 Laptop de 8 GB.

## O que é o Keilinks

O **Keilinks Core 380M** é a linha principal de pesquisa e desenvolvimento do projeto. O modelo **não é um fine-tuning de Qwen, Llama ou outro LLM pronto**: sua arquitetura Transformer é implementada no próprio repositório e os pesos do Core são inicializados e treinados pelo pipeline do Keilinks.

Modelos externos executados via Ollama, como Qwen, podem ser usados como **teacher/critic** para gerar, revisar e filtrar uma parcela limitada de dados sintéticos de conversa. Eles ajudam a preparar dados; **não são o cérebro do Keilinks Core 380M**.

O repositório ainda contém trilhas legadas e experimentais baseadas em Qwen/Unsloth. Elas servem como referência, baseline e ferramentas auxiliares, mas não representam a arquitetura autoral atual do Core 380M.

## Arquitetura atual

O modelo principal está em `treino/v4/modelo.py` e usa uma arquitetura decoder-only moderna:

- aproximadamente **380 milhões de parâmetros** no perfil operacional;
- vocabulário de **32.000 tokens**;
- **24 camadas** Transformer;
- dimensão oculta de **1.152**;
- **18 attention heads** e **6 KV heads** com Grouped Query Attention;
- **RMSNorm**;
- **SwiGLU**;
- **RoPE** com `theta=500000`;
- **QK-Norm**;
- weight tying entre embeddings e LM head;
- SDPA/Flash Attention automático quando disponível;
- KV cache na geração autoregressiva;
- gradient checkpointing configurável;
- BF16, TF32, AdamW 8-bit e `torch.compile` no perfil de treino da RTX 5050.

O perfil operacional `core_380m_modern` usa contexto nativo de **8.192 tokens**.

## Treino local na RTX 5050 8 GB

O objetivo do projeto é explorar até onde um modelo autoral pode ser levado em hardware de consumo sem esconder os limites do experimento.

Benchmarks locais do Core 380M mostraram:

| Janela / modo | Pico de VRAM | Vazão observada | Uso |
| --- | ---: | ---: | --- |
| 2.048 tokens, checkpoint seletivo | ~4,87 GB | ~2.181 tok/s | compatibilidade |
| 4.096 tokens, checkpoint completo | ~4,14 GB | ~2.214 tok/s | viável |
| 8.192 tokens, checkpoint completo | ~6,56 GB | ~2.154 tok/s | estável |
| 8.192 + `torch.compile` | ~5,45 GB | ~2.651 tok/s | perfil operacional |
| 16.384 tokens, treino | ~9,92 GB | ~790 tok/s | não usar nesta GPU |

O perfil `rtx5050_380m` usa batch físico 1 e quatro microbatches por atualização, preservando **32.768 tokens por passo de otimização**.

A meta configurada para o ciclo longo é de até **160.000 passos**, equivalente a aproximadamente **5,24 bilhões de tokens de exposição** no perfil atual. Isso é uma meta experimental; checkpoints só devem ser promovidos após avaliação.

## Pipeline de aprendizado

A linha principal segue esta ordem:

```text
Dados PT-BR com origem/licença conhecida
            │
            ▼
      limpeza + deduplicação
            │
            ▼
   tokenizador próprio 32k
            │
            ▼
     PRETRAIN do Core 380M
            │
            ▼
   checkpoint-base avaliado
            │
            ▼
     SFT conversacional
            │
            ▼
 avaliação + testes + revisão
            │
            ▼
      candidato promovido
```

O pré-treino cria a base linguística e estatística do modelo. O SFT é aplicado depois para ensinar comportamento conversacional, seguimento de instruções, formato de resposta e identidade do Keilinks.

## Teacher/critic com modelos externos

O arquivo `treino/v4/gerar_conversas_ollama.py` implementa uma pipeline opcional de geração sintética com **professor + crítico** via Ollama:

```text
LLM externo (teacher)
       │
       ▼
gera conversa candidata
       │
       ▼
LLM externo (critic)
       │
       ▼
filtros determinísticos + deduplicação
       │
       ▼
parcela sintética do dataset SFT
       │
       ▼
Keilinks Core 380M
```

Os dados sintéticos não devem dominar o conjunto. O pipeline mantém exemplos humanos, dados curados e avaliações congeladas separados para reduzir contaminação e reforço de erros do próprio gerador.

## SFT conversacional PT-BR

A versão conversacional V2 prepara **5.000 conversas únicas** com composição controlada:

| Categoria | Participação |
| --- | ---: |
| diálogo humano com feedback | 30% |
| diálogo multi-turno traduzido e auditado | 13% |
| instruções humanas | 32% |
| instruções sintéticas | 14% |
| raciocínio curto | 7% |
| âncoras de comportamento Keilinks | 4% |

O conjunto empacotado em contexto 8K possui cerca de **1,05 milhão de tokens supervisionados de treino**, além de validação separada.

O SFT só deve iniciar depois que o checkpoint de pré-treino passar os gates definidos para loss, estabilidade, VRAM, integridade de checkpoint e avaliações congeladas.

## Orquestração do primeiro ciclo 380M

O orquestrador espera uma coleta já concluída, monta o corpus, cria o tokenizador, prepara os binários e executa um burn-in antes de liberar o treino longo.

```powershell
python -m treino.v4.orquestrar_380m --run-id public-pt-380m-01
```

Depois de revisar o burn-in e seus logs, o treino longo precisa ser liberado explicitamente:

```powershell
python -m treino.v4.orquestrar_380m --run-id public-pt-380m-01 --continue-after-burn-in
```

O processo possui guardas para espaço em disco, VRAM, loss não finita e degradação evidente durante o burn-in.

## Preparar e treinar o SFT conversacional

Depois de um checkpoint de pré-treino aprovado:

```powershell
python -m treino.v4.preparar_sft_conversacional generate-anchors
python -m treino.v4.preparar_sft_conversacional build
python -m treino.v4.pack_sft_em_escala --context 8192 --output dados/v4/packed_sft_conversation_8k_v2 dados/v4/sft/all_sft_380m_conversation_8k_v2.jsonl
```

Treino SFT:

```powershell
python -m treino.v4.treinar --model core_380m_modern --profile rtx5050_sft_380m --data dados/v4/packed_sft_conversation_8k_v2 --epochs 3 --init-checkpoint checkpoints/v4-pretrain-380m/pretrain_best.pt --output checkpoints/v4-sft-conversation-8k-v2
```

## Dados, proveniência e segurança

O projeto trata dados como parte da arquitetura, não apenas como volume.

Princípios atuais:

- registrar origem e licença dos datasets;
- deduplicar exemplos e documentos;
- separar dados humanos, traduzidos e sintéticos;
- não usar conversas reais de usuários para treino automático;
- feedback só pode virar candidato a dado após consentimento e curadoria;
- remover ou redigir PII quando aplicável;
- manter conjuntos de avaliação fora do treino;
- não promover checkpoints apenas porque a loss caiu;
- preservar checkpoints e possibilidade de rollback.

## RAG, memória, web e visão

Além do Core de linguagem, o projeto experimenta componentes de produto que podem ser acoplados ao modelo:

- pesquisa web com gate de evidência e preferência por fontes verificáveis;
- RAG local;
- memória pessoal isolada;
- banco SQLite local;
- visão por VLM separado;
- feedback consentido;
- API e interface local;
- filtros de segurança e tratamento determinístico de cenários críticos.

Esses componentes são ferramentas ao redor do modelo e não substituem o aprendizado dos pesos do Core.

## Estado do projeto

O Keilinks é um projeto de pesquisa e engenharia em evolução. A prioridade atual é:

1. concluir e medir o pré-treino do **Core 380M**;
2. avaliar checkpoints em conjuntos congelados;
3. executar o SFT conversacional PT-BR somente após aprovação do pré-treino;
4. comparar qualidade antes/depois do SFT;
5. melhorar dados, avaliações e serving sem confundir automação com autoaperfeiçoamento seguro;
6. explorar perfis maiores apenas quando houver evidência de que o hardware e os dados justificam o custo.

## Estrutura relevante

```text
treino/v4/modelo.py                    # Transformer autoral
treino/v4/config.py                    # perfis 380M/500M/800M/1B
treino/v4/pretreinar.py                # pré-treinamento
treino/v4/treinar.py                   # SFT
treino/v4/orquestrar_380m.py           # ciclo seguro do Core 380M
treino/v4/tokenizador.py               # tokenizador
treino/v4/gerar_conversas_ollama.py    # teacher/critic sintético
treino/v4/preparar_sft_conversacional.py
busca/                                  # pesquisa e grounding
cerebro/                                # componentes auxiliares
api/                                    # serviços locais
tests/                                  # testes e regressões
```

## Nota sobre Qwen/Unsloth

O repositório mantém código de experiências anteriores com Qwen, QLoRA e Unsloth. Essa trilha foi útil como baseline e continua disponível para comparação, mas **não deve ser confundida com o Keilinks Core 380M autoral**.

O Core 380M usa a implementação Transformer do próprio projeto e segue seu próprio pipeline de pré-treinamento e SFT.

---

**Keilinks** — pesquisa prática em modelos de linguagem locais, PT-BR, treinamento autoral e engenharia responsável em hardware de consumo.
