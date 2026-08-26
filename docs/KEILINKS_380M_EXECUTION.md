# Keilinks Core 380M: execução local na RTX 5050 de 8 GB

Este documento descreve a linha de treino do modelo autoral Keilinks de cerca
de 380 milhões de parâmetros. Ela é independente do runtime Qwen/Unsloth que
continua sendo o caminho de produto até que o checkpoint autoral vença as
avaliações.

## Objetivo do modelo

O Core 380M deve ser uma assistente PT-BR natural, útil em conversa e honesta
sobre incerteza. O pré-treino dá competência linguística e conhecimento geral;
o SFT ensina turnos, tom e prioridades; o roteador de pesquisa consulta fontes
para fatos que podem estar ausentes ou desatualizados. Nenhuma dessas etapas,
isoladamente, torna o modelo pronto para produção.

## Perfil que realmente cabe na GPU

Use `core_380m_modern`, não `core_380m_v5_experimental`, para o treino longo:

- 1.152 dimensões, 24 camadas, 18 heads Q e 6 heads KV (GQA), SwiGLU e pesos
  de embedding/saída atados;
- QK-Norm e RoPE com theta 500.000 para estabilidade e contexto longo;
- **soft-capping desativado**. O experimento de soft-capping materializa `QKᵀ`
  e perde o caminho SDPA/Flash Attention, o que é inadequado para 8 GB;
- BF16, TF32, AdamW 8-bit e checkpointing seletivo a cada duas camadas.

Benchmark local de 26/08/2026, com batch físico 1 e 2.048 tokens:

| Métrica | Resultado |
| --- | ---: |
| VRAM de pico | 5,69 GB |
| Vazão medida | 3.234 tokens/s |
| Perfil efetivo | 16 microbatches por passo |
| Exposição-alvo inicial | 160.000 × 16 × 2.048 = 5,24B tokens |

O número acima implica aproximadamente 19 dias de computação ideal; logs,
checkpoints, validação, temperatura e processos concorrentes aumentam esse
tempo. Trate o primeiro ciclo como um experimento longo e retomável, não como
um download que termina o modelo.

## Dados em coleta

O run `public-pt-380m-01` reserva até 20 GiB de JSONL rastreável:

| Fonte | Orçamento | Uso | Gate |
| --- | ---: | --- | --- |
| FineWeb-2 `por_Latn` | 18 GiB | linguagem geral PT | ODC-By + termos Common Crawl; `language_score >= 0,97`; cluster MinHash <= 32 |
| Wikipedia PT `20231101.pt` | 2 GiB | fatos e escrita enciclopédica | CC-BY-SA/GFDL; título e URL preservados |

Cada documento gravado contém fonte, licença, revisão do dataset, hash SHA-256
e identificador/URL de origem quando a fonte o disponibiliza. A coleta usa
deduplicação exata em SQLite e não traz automaticamente CulturaX, GigaVerbo ou
Corpus Carolina: CulturaX é gated; GigaVerbo é uma agregação de licenças
heterogêneas; e a versão de `datasets` local não executa mais o script do
Carolina. Eles podem entrar posteriormente com credencial/importação e revisão
de licença por fonte.

O run de conversa `public-conversations-380m-01` coleta separadamente:

| Fonte | Papel | Tratamento |
| --- | --- | --- |
| OpenAssistant OASST2 PT | conversas com feedback humano | somente cadeias PT revisadas, não removidas e não sintéticas |
| Aya PT | instruções humanas | somente `original-annotations` |
| Tucano-SFT | diversidade de instruções | sintético; teto no mix para não dominar a personalidade |
| conversas curadas Keilinks | identidade, empatia e regras locais | mantidas como conjunto próprio e auditável |

O coletor remove padrões sensíveis conhecidos, recusa marcadores de template,
valida alternância de papéis e deduplica por conversa. Isso reduz risco, mas
não é uma garantia jurídica ou de anonimização.

## Ordem de treino

1. `coletar_corpus.py` produz shards e manifests, sem ocupar a RAM com todo o
   corpus.
2. `montar_corpus.py` verifica hashes e intercala fontes em
   `pretrain_pt.txt`, preservando limites de documento.
3. `tokenizador.py build` cria `tokenizer.json` com ByteLevel BPE do Rust
   (`tokenizers`), vocabulário de 32k e os tokens de papel V4. O loader ainda
   lê vocabulários BPE antigos.
4. `pretreinar.py --prepare-only` converte para binário `int32`, registra hash
   do vocabulário e cria validação determinística por documento.
5. Um smoke run valida loss finita, VRAM, checkpoint e retomada. O smoke local
   de três passos completou com 3.534 tokens/s e pico de 6,43 GB.
6. O pré-treino longo começa com checkpoint retomável. A promoção para SFT só
   acontece após perda de validação estável e avaliação congelada.
7. `coletar_conversas.py`, `misturar_sft.py` e `pack_sft_em_escala.py` criam o
   dataset assistant-only. SFT usa LR menor e começa do melhor checkpoint de
   pré-treino, nunca de pesos aleatórios.
8. DPO/GRPO só entra após pares de preferência corrigidos e aprovados. Feedback
   de usuário com consentimento não deve virar treino automático sem curadoria.

## Conversa natural e pesquisa web

`api/runtime_v4.py` usa o roteador `deve_pesquisar`:

- `auto`: pesquisa perguntas factuais e temporais, mas não gasta web em
  cumprimento casual, escrita criativa ou tradução;
- `always`: força consulta antes de responder;
- `never`: permite uma resposta sem navegação apenas quando o cliente assume
  essa limitação;
- se a pergunta exige verificação e nenhuma fonte é obtida, a resposta é uma
  recusa honesta de afirmar um fato, não uma alucinação;
- perguntas temporais não aceitam Wikipédia isolada: precisam de duas origens
  legíveis ou de uma fonte de domínio confiável/primário; cache desse tipo de
  pergunta expira em 15 minutos;
- a cadeia de busca tenta SearXNG, Brave, cliente DDG opcional e o feed RSS
  público do Bing antes do fallback enciclopédico, sem exigir chave para o
  caminho local mínimo;
- resultados entram no prompt como dados não confiáveis, nunca como instruções,
  e a resposta devolve URLs/títulos consultados.

O modelo não precisa “decidir sozinho que sabe pouco”; a decisão de pesquisa é
feita por uma camada determinística e auditável. Isso é mais confiável para um
modelo de 380M e mantém as respostas de conversa naturais quando não há motivo
para usar a web.

## Critérios para dizer que está pronto

O Core 380M só pode substituir o modelo atual quando passar todos os gates:

1. manifests completos, hashes verificáveis e espaço em disco suficiente para
   corpus, binários e checkpoints;
2. loss de validação do pré-treino sem divergência e sem regressão no resume;
3. avaliação congelada PT-BR de conversa, honestidade, segurança, RAG e web;
4. teste humano cego contra a baseline em conversas reais;
5. SFT sem vazamento de marcadores, repetição excessiva ou perda de instrução;
6. resposta factual com fonte quando o roteador aciona a web e recusa honesta
   quando ela não está disponível;
7. checkpoint anterior e instruções de rollback preservados.

Até esses gates, o Core 380M é um candidato em treinamento; o runtime Qwen
QLoRA existente permanece disponível para uso prático.
