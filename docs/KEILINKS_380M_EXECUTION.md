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
- BF16, TF32, AdamW 8-bit, `torch.compile` em `default` e checkpointing
  **completo**. O modo `reduce-overhead` foi benchmarkado, mas nesta
  instalação de PyTorch 2.11 falha com CUDA Graphs quando combinado com os
  quatro microbatches de 8k. `max-autotune-no-cudagraphs` evita esse defeito,
  mas teve cold-start desproporcional nesta GPU; o modo `default` é o caminho
  estável a validar no burn-in.

O perfil operacional usa contexto nativo de **8.192 tokens**, batch físico 1 e
quatro microbatches por atualização. Assim preserva 32.768 tokens por passo de
otimização (`1 × 8.192 × 4`), a mesma exposição do perfil 2k legado
(`1 × 2.048 × 16`), sem reduzir o lote efetivo.

Benchmark local de 26/08/2026, na RTX 5050 Laptop de 8 GB, com forward/backward
BF16 do Core 380M:

| Janela / modo | VRAM de pico | Vazão | Decisão |
| --- | ---: | ---: | --- |
| 2.048, seletivo | 4,87 GB | 2.181 tok/s | compatibilidade 2k |
| 4.096, completo | 4,14 GB | 2.214 tok/s | cabe, mas 8k é preferível |
| 8.192, completo | 6,56 GB | 2.154 tok/s | estável em cinco passos |
| 8.192, completo + `reduce-overhead` | 5,45 GB | 2.651 tok/s | probe; não usar no ciclo longo |
| 16.384, completo | 9,92 GB | 790 tok/s | não usar para treino nesta GPU |

O prefill de inferência cabe até 16k (3,47 GB de pico), mas isso não prova
qualidade posicional nem viabilidade de pré-treino. A meta de 160.000 passos
em 8k continua sendo 5,24B tokens e equivale a aproximadamente 22,9 dias
ideais na vazão compilada medida; logs, validações, temperatura e pausas podem
aumentar esse prazo. Trate o ciclo como experimento longo e retomável, não como
um download que termina o modelo.

## Raciocínio curto, resposta melhor e botão no chat

O Core não recebe uma cadeia de pensamento longa, opaca ou copiada da web. Ele
recebe um **currículo de plano curto**: dados relevantes, ferramenta necessária
(por exemplo, busca web) e uma checagem. A resposta final é treinada separada
do plano. O histórico persiste somente a resposta final; a interface pode,
quando o usuário habilitar **👁 Plano**, exibir o plano curto completo que foi
gerado para aquela resposta.

O arquivo `treino/v4/preparar_raciocinio.py` gera 640 exemplos PT-BR
determinísticos e verificáveis: contas, porcentagens, condições lógicas,
comparação por critério, investigação de erro e decisão de consultar a web.
Eles entram como aproximadamente 6,9% do SFT misturado atual (640 de 9.270
exemplos), suficiente para ensinar o formato sem substituir conversa humana,
empatia e instruções reais.

Prepare o estágio antes do SFT, mas **não** antes do pré-treino: pesos ainda
aleatórios não ganham capacidade de raciocínio apenas ao ver esses exemplos.
O pré-treino cria a base linguística; o SFT a ensina a usar o protocolo quando
for útil.

```powershell
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.preparar_raciocinio
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.pack_sft_em_escala `
  --context 8192 --output dados/v4/packed_sft_8k `
  dados/v4/sft/all_sft_380m_reasoning_8k.jsonl
```

Depois do checkpoint de pré-treino aprovado, o SFT deve usar
`core_380m_modern` e `rtx5050_sft_380m`: contexto 8.192, batch físico 1,
quatro microbatches, checkpointing completo e `torch.compile`.

```powershell
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.treinar `
  --model core_380m_modern --profile rtx5050_sft_380m `
  --data dados/v4/packed_sft_8k `
  --epochs 3 `
  --init-checkpoint checkpoints/v4-pretrain/pretrain_best.pt `
  --output checkpoints/v4-sft-reasoning-8k
```

O treino SFT calcula três épocas reais por padrão quando `--steps` não é
informado. No pacote atual de 219 blocos de treino, isso equivale a 165 passos
de otimização, não aos 10.000 passos máximos do perfil. Essa proteção evita
repetir o corpus curto mais de 180 vezes e destruir a generalização da conversa.
O warmup também é limitado a no máximo 10% do ciclo efetivo, para que um SFT
curto não passe inteiro apenas aquecendo a taxa de aprendizado.

Na interface, o botão **🧠 Raciocínio** envia `reasoning_mode=always` para a
próxima resposta. Desligado, o runtime usa `auto` e só pede plano em contas,
comparações, depuração e decisões mais complexas. A API ainda aceita `never`
para desabilitar o protocolo. **👁 Plano** controla `show_reasoning`; quando
ativo, mostra o máximo de 64 palavras do plano apenas se os marcadores de plano
e resposta foram fechados corretamente. Isso é uma explicação curta do modelo,
não um traçado interno token a token, e nunca exibe instruções cruas
recuperadas de RAG ou da web.

Antes de promover o checkpoint SFT, execute a avaliação congelada:

```powershell
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.avaliar_raciocinio `
  --checkpoint checkpoints/v4-sft-reasoning-8k/keilinks_v4.pt
```

Ela mede acerto em casos retidos de cálculo, condições, comparação, depuração
e consulta atual. O gate exige melhora ou ausência de regressão na avaliação
geral, nenhum marcador `[[PLANO]]`/`[[RESPOSTA]]` vazado ao usuário e respostas
web ainda citáveis. Isso mede se o protocolo melhorou comportamento; não é uma
alegação de que 640 exemplos transformaram um 380M em um modelo de fronteira.

## Acompanhamento e pausa segura

Abra o painel local com:

```powershell
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.monitorar_380m
```

Ele exibe a etapa atual, tamanho/taxa/ETA da coleta, disco, GPU, VRAM, loss,
LR, tokens por segundo, ETA do pré-treino e os três checkpoints mais recentes.
`Ctrl+C` fecha somente o painel.

- `pretrain_best.pt`: pesos com melhor validação, reavaliados a cada 250 passos;
- `pretrain_latest.pt`: pesos e otimizador, salvo a cada 1.000 passos para
  retomada;
- `pretrain_final.pt`: fechamento normal do ciclo;
- `pretrain_paused.pt`: checkpoint completo feito após uma pausa solicitada.

Para pedir uma pausa sem perder mais que o passo em andamento:

```powershell
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.monitorar_380m --request-pause
```

Depois, remova a solicitação e retome pelo orquestrador, que prefere o
checkpoint pausado quando ele existe:

```powershell
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.monitorar_380m --clear-pause
& .\.venv-unsloth\Scripts\python.exe -m treino.v4.orquestrar_380m --continue-after-burn-in
```

Em 100.000 passos o modelo terá visto aproximadamente 3,28 bilhões de tokens
(62,5% da exposição planejada). É um bom ponto de decisão, mas não um sinal
automático de que "já basta": compare `pretrain_best.pt`, a curva de validação
e a avaliação congelada antes de parar definitivamente ou seguir para SFT.

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

## Banco local sem MySQL

O servidor legado agora usa SQLite em `keilinks_data/keilinks.sqlite3`, criado
automaticamente no primeiro boot. Ele guarda usuários, chats, histórico,
memória, log do crawler e knowledge lexical (FTS5 quando disponível). O arquivo
e o segredo local de autenticação ficam fora do Git. Para mudar o local, defina
`KEILINKS_DB_PATH`. A API é local por padrão; exposição fora de localhost ainda
exige uma chave de API forte — SQLite não é banco para múltiplas instâncias ou
compartilhamento em rede.

## Linux/WSL e aceleração sem ilusão de hardware

O benchmark 8k acima foi feito no Windows e já usa BF16, TF32, GQA, AdamW
8-bit, checkpointing completo e `torch.compile`. A medição de 2.651 tok/s veio
de `reduce-overhead`, que falha com os quatro microbatches no PyTorch 2.11
instalado; ela não deve ser tratada como vazão contratada. O ciclo real usa
`default` e precisa registrar a primeira janela de 20 passos antes de fixarmos
uma taxa. Temperatura, validação, salvamento e concorrência do Windows também
podem variar a medição.

Na verificação de 26/08/2026, `wsl.exe` existe, mas não há uma distribuição
Linux registrada nem serviços WSL disponíveis; a consulta/ativação dos recursos
Windows exige privilégios de administrador. A instalação oficial de WSL pode
exigir reinicialização. Não reinicie durante a tokenização/preparo ativo: isso
interromperia o pipeline antes do checkpoint seguro. Assim que houver uma janela
segura, valide no Ubuntu/WSL2 com a mesma versão de PyTorch, o mesmo tokenizer e
o mesmo benchmark de 8k; somente mantenha Linux se a métrica real superar o
Windows.

- Não existe como liberar mais VRAM física por software. Memória compartilhada
  do Windows é paginação lenta, não substitui VRAM para pré-treino.
- CPU ajuda a tokenizar, carregar dados e manter a GPU alimentada; não acelera
  as multiplicações de matriz que dominam o treino do Transformer.
- Overclock não é aplicado pelo projeto: numa GPU de notebook ele aumenta risco
  térmico/instabilidade e pode piorar um treino de semanas. Só vale considerar
  manualmente depois de monitorar temperatura, potência e estabilidade.
- Ganhos reais adicionais vêm de kernels Linux que realmente passem no benchmark,
  uma GPU com mais VRAM, ou uma GPU cloud; MoE não reduz o custo quadrático da
  atenção de 8k e fica para depois da base e do SFT estáveis.

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
