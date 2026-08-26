# Keilinks: relatório técnico, decisões de arquitetura e roteiro de evolução

Data da revisão: 25 de agosto de 2026
Escopo: levar o Keilinks a uma IA generativa local, útil e evolutiva em uma
RTX 5050 Laptop com 8 GB de VRAM, sem confundir automação com
autoaperfeiçoamento seguro.

## Resultado executivo

O projeto agora possui uma rota de produto mensurável: Qwen3-4B em 4 bits
adaptado por QLoRA/Unsloth, API FastAPI local com streaming, RAG híbrido com
proveniência, visão opt-in, um gate de crise iminente e um ciclo de dados que
exige consentimento e aprovação humana. O adaptador V4 controlado foi treinado
na máquina-alvo, passou pela avaliação congelada e respondeu pela API real.

O que não foi tratado como pronto: AGI, pré-treino brasileiro de escala web,
feedback que retreina sozinho em produção ou estimativas de VRAM não medidas.
Todo candidato deve passar por avaliação congelada e revisão humana antes de
ser promovido.

## Estado verificado nesta revisão

| Item | Estado | Evidência local |
| --- | --- | --- |
| GPU | RTX 5050 Laptop, 8.151 MB, BF16 disponível | verificação PyTorch/CUDA no ambiente .venv-unsloth |
| Base de linguagem | Qwen3-4B-Instruct-2507 4-bit | carregamento em keilinks_v5/runtime.py |
| Treino V4 | concluído, 40 passos e 33.030.144 parâmetros LoRA treináveis (0,81%) | manifesto local do adaptador V4 |
| Validação | melhor eval_loss 2,0747 no checkpoint selecionado | manifesto do treino V4 |
| Avaliação congelada | 15/18 na regra lexical; 2 casos web ignorados por padrão | resultado local v4-controlled-sanitized.json |
| Inferência real | resposta V4 para “o que é uma API?” via FastAPI | teste local nesta revisão |
| Streaming | SSE finalizado com marcador [DONE] | endpoint /v1/chat/completions |
| Visão | SmolVLM2 4-bit analisou uma imagem de teste; pico observado de aproximadamente 1,67 GiB | keilinks_v5/vision.py e teste local |
| Testes | 16 testes passaram | pytest de tests/test_v5_core.py e tests/test_v5_pipeline.py |

O 14/18 do V3 e o 15/18 do V4 são reportados sem maquiagem. As falhas restantes
foram respostas semanticamente razoáveis reprovadas por palavras-chave rígidas.
A conclusão é que a próxima melhoria deve combinar rubricas estruturadas,
inspeção humana e testes adversariais; não aumentar passos até encaixar em uma
string.

### Fine-tuning V4 controlado

O candidato V4 usou o mesmo conjunto com holdout protegido, 40 passos, LoRA
rank 16, alpha 32, batch efetivo 8 e learning rate de 0,0001. O treino levou
151 segundos na RTX 5050 e o melhor checkpoint teve eval_loss 2,0747, contra
2,0856 do V3. No benchmark congelado, o V4 passou 15 de 18 casos, contra 14
do V3. A melhoria principal apareceu nos casos emocional e de priorização no
trabalho.

Durante a promoção, um caso mostrou o marcador interno de ferramenta no texto
gerado. O runtime agora sanitiza esse marcador, inclusive no SSE, e o
avaliador reprova explicitamente qualquer vazamento futuro. A validação final
do V4 continuou em 15/18 sem marcador vazado. O adaptador fica fora do Git
porque é um artefato local grande; o caminho padrão é
checkpoints/keilinks-qwen3-4b-lora-v4-controlled.

## Decisões técnicas

| Decisão | Escolha | Por quê | Limite e próxima ação |
| --- | --- | --- | --- |
| Modelo principal | Qwen3-4B-Instruct-2507 em 4-bit | cabe no perfil de 8 GB e tem licença Apache-2.0 | o contexto padrão continua 2.048 tokens; contexto nativo não garante caber na GPU |
| Ajuste fino | QLoRA via Unsloth, LoRA em atenção e MLP | adaptação útil sem pesos completos em BF16 | comparar rank, LR e dados com avaliação congelada |
| Runtime Windows | FastAPI mais Transformers/Unsloth | funciona nativamente e é simples de auditar | vLLM fica como opção Linux/WSL, não dependência Windows |
| Visão | VLM separado, opt-in e descarregado após uso | preserva VRAM e evita descrição falsa como fallback | benchmarkar Qwen3-VL-2B antes de trocar SmolVLM2 |
| RAG | FTS5/BM25 mais embedding CPU opcional e RRF | local, rastreável e não disputa VRAM | migrar para Qdrant apenas com corpus/concorrência que justifique |
| Dados | manifests, redator de PII, termos e holdout guard | reduz vazamento, problemas de licença e uso acidental de PII | criar revisão/amostragem humana antes de escalar |
| Preferências | DPO offline após aprovação humana | aprende comparações sem a complexidade de PPO | não iniciar sem pares consentidos e auditados suficientes |
| GRPO, Muon e V4 | experimento isolado | há valor de pesquisa, não ganho local comprovado | exigir baseline AdamW/QLoRA, custo, qualidade e rollback |

## Por que Unsloth é a escolha certa agora

O Unsloth declara suporte direto a RTX 30, 40 e 50 e Windows. Isso reduz a
complexidade de QLoRA, que é a prioridade em uma única GPU de 8 GB. A escolha
não autoriza ligar todo recurso de treino disponível: qualidade de dados,
testes e política de promoção importam mais que uma otimização isolada.

A base usada é
[Qwen3-4B-Instruct-2507, variante Unsloth 4-bit](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit).
O card informa GQA, contexto nativo grande e licença Apache-2.0. Nesta máquina,
a aplicação usa contexto menor por causa do KV cache, VLM e overhead real.

Não foi adicionada dependência obrigatória de vLLM. A documentação oficial
lista Linux como requisito de GPU e declara que Windows nativo não é suportado.
WSL pode fazer sentido numa etapa posterior de serving, mas seria atrito
desnecessário para o objetivo atual: funcionamento local estável.

## Arquitetura de produção

~~~text
Cliente local
    |
    +-- POST /v1/documents --> documento + hash + tenant_id --> SQLite FTS5 / vetores CPU
    |
    +-- POST /v1/chat/completions
            |
            +-- gate de crise iminente --> resposta segura, CVV, SAMU e UPA
            +-- recuperação RAG por RRF --> contexto marcado como não confiável
            +-- VisionService opt-in --> OCR e descrição marcados como não confiáveis
            +-- Qwen3 4-bit + adaptador Keilinks V4 --> JSON ou SSE
                                                        |
                                           cache efêmero de interação
                                                        |
                                  feedback com consentimento explícito
                                                        |
                                         arquivo redigido para revisão humana
~~~

O cache de interação existe somente em memória e expira. A aplicação não usa
chat para treino por padrão. A API requer chave fora de loopback e limita
chamadas. Conteúdo de RAG e VLM é inserido como contexto não confiável para
reduzir prompt injection por documento ou imagem.

### RAG: simples agora, escala depois

O armazenamento atual usa FTS5/BM25 para termos exatos em português, embeddings
opcionais em CPU e Reciprocal Rank Fusion:

~~~text
score(documento) = 1 / (60 + rank BM25) + 1 / (60 + rank dense)
~~~

Isto resolve o primeiro produto local: funciona offline, fornece proveniência
e pode ser testado sem serviço externo. Quando o índice crescer, o caminho é
Qdrant com busca híbrida e filtros por tenant. Não migre por moda: meça
latência p95, recall@k, volume de documentos e concorrência primeiro.

### Visão em 8 GB

O VLM é coprocessador, não um projetor improvisado no Transformer autoral. Com
HuggingFaceTB/SmolVLM2-2.2B-Instruct em 4-bit, a imagem é validada por tamanho
e dimensões, analisada uma por vez e o modelo é liberado no fim da requisição.
A observação de aproximadamente 1,67 GiB é uma medição de imagem de teste, não
promessa de consumo máximo: resolução, geração e driver mudam o resultado.

Há adaptador compatível para testar Qwen3-VL-2B. Só troque o padrão após uma
suite local de OCR, recibos, tabelas, gráficos e screenshots em português,
registrando acerto, alucinação, latência e VRAM. VLMs podem errar em usos de
alto impacto.

## Dados abertos: decisões por licença e qualidade

| Fonte | Uso potencial | Situação no coletor | Cuidados |
| --- | --- | --- | --- |
| FineWeb-2 por_Latn | continuação de português em grande escala | streaming com aceite fineweb2_terms | cerca de 109,5B tokens; ODC-By e termos Common Crawl exigem atribuição/conformidade |
| Corpus Carolina | continuação com português curado | disponível com aceite | CC-BY-4.0: preservar atribuição e manifesto |
| Wikipedia PT | conhecimento enciclopédico | disponível com aceite | CC-BY-SA/GFDL: atender atribuição e share-alike |
| CulturaX PT | grande volume textual | bloqueado até aceite dos termos | gated, PII e licenças compostas; não automatizar acesso |
| Pt-Corpus-Instruct | texto PT variado | bloqueado até revisão manual | texto bruto/metadados, não par garantido instrução-resposta, e fontes com restrições |

Direitos e origem acompanham cada linha. Dados de produção só entram com base
legal e consentimento explícito. Antes de escalar, execute deduplicação, filtro
de PII, detecção de idioma, amostragem humana e registro de exclusões.

## Melhoria contínua sem autoenvenenamento

“O modelo se treina sozinho” é perigoso: respostas geradas pelo próprio modelo
tendem a reforçar erros, estilo repetitivo e falsas certezas. O fluxo adotado é
intencionalmente mais lento e reversível:

~~~text
feedback com opt-in
        |
redação de PII e validação
        |
revisão humana e política de qualidade
        |
pares: prompt, resposta rejeitada, correção aprovada
        |
preparar_preferencias -> DPO offline em checkpoint candidato
        |
holdout congelado + avaliação humana + regressões de segurança
        |
promoção manual ou rejeição/rollback
~~~

O módulo de DPO está preparado para dados aprovados e não treina sem pares
revisados. GRPO pode ser avaliado em tarefas com verificador objetivo, como
testes de código em sandbox, mas não deve decidir sozinho qualidade de conversa,
saúde ou fatos do mundo. Mantenha dados âncora, checkpoints imutáveis e
rollback; isso é proteção parcial, não prova de segurança.

### Gatilhos de promoção

Promova apenas se todos forem verdadeiros:

1. eval_loss não piorou materialmente e o treino não mostra instabilidade.
2. O benchmark congelado não cai em honestidade, segurança, português ou instrução.
3. Avaliadores humanos aprovam uma amostra cega contra a versão anterior.
4. Redação de PII, isolamento por tenant e roteamento de crise continuam passando.
5. Latência, VRAM e taxa de erro cabem no hardware-alvo.
6. Há manifest de dados, licença, hash de artefato e plano de rollback.

## Segurança e limites de produto

O runtime não deve fingir ser médico, terapeuta, advogado ou fonte factual
infalível. Para linguagem que indica risco iminente de autoagressão há resposta
determinística que incentiva procurar alguém próximo, UPA/SAMU 192 e CVV 188.
Ela não depende de temperatura, prompt ou VLM e não substitui assistência
profissional.

Outras proteções aplicadas:

- sem trust_remote_code na visão;
- limite de bytes e pixels de imagem;
- sem fallback visual inventado;
- comparação de chave de API em tempo constante e rate limiting local;
- senhas do banco com scrypt/salt e migração de hash legado;
- segredos e credenciais removidos do código;
- CORS e host restritos por padrão.

Antes de exposição pública, adicionar autenticação por usuário, auditoria sem
conteúdo sensível, retenção/LGPD, teste de carga e revisão de ameaças de upload,
prompt injection e acesso entre tenants.

## Roteiro priorizado

### Fase 0 — concluída nesta revisão

- Instalação Windows/RTX 5050 reproduzível com Unsloth e verificação CUDA.
- QLoRA V3 estabelecido como baseline e V4 controlado treinado, avaliado e promovido.
- Código V5 do Gemini substituído por fachadas de compatibilidade seguras,
  removendo fallback visual falso e código remoto não auditado.
- RAG, feedback consentido, avaliação congelada, guard de leakage, segurança
  de servidor/banco e testes automatizados.

### Fase 1 — estabilizar o produto

1. Montar 100–300 conversas PT-BR licenciadas e avaliadas por humanos, cobrindo
   tom, código, recusa, emoção, ambiguidade e conhecimento prático.
2. Criar rubrica de avaliação estruturada; manter regras lexicais só como
   regressão barata.
3. Fazer grid pequeno de rank LoRA, learning rate e passos, escolhendo por
   comparação cega, não por loss isolada.
4. Criar suite visual PT-BR antes de alterar o VLM padrão.
5. Medir p50/p95, tokens por segundo, VRAM máxima e falhas em sessões locais.

### Fase 2 — dados e preferência

1. Coletar pequenas amostras de fontes abertas somente após aceitar termos.
2. Deduplicar contra treino, validação e holdout; revisar PII e licenças.
3. Usar feedback apenas com consentimento e revisão humana.
4. Construir o primeiro lote DPO e comparar com V4.
5. Usar avaliador LLM só como sinal auxiliar e medir viés contra pares humanos.

### Fase 3 — escala e serving

1. Migrar RAG para Qdrant somente com métricas que justifiquem.
2. Experimentar llama.cpp/GGUF e speculative decoding depois de benchmark de
   tokens por segundo e VRAM.
3. Avaliar vLLM em WSL/Linux apenas se houver demanda de concorrência.
4. Adicionar observabilidade sem conteúdo: latência, contexto, erro, VRAM e
   satisfação consentida.

### Fase 4 — pesquisa do modelo autoral

Comparar o V4 autoral, QK-Norm, RoPE/soft-capping, Muon e GRPO contra a
baseline QLoRA. Para cada experimento, registrar seed, hardware, dataset,
custo, loss, avaliação humana, segurança e rollback. Uma mudança só entra se
vencer a baseline de modo repetível.

## Comandos de validação

~~~powershell
.\.venv-unsloth\Scripts\python.exe -m pytest -q tests/test_v5_core.py tests/test_v5_pipeline.py
.\.venv-unsloth\Scripts\python.exe -m ruff check keilinks_v5 treino/v5 api cerebro tests
.\.venv-unsloth\Scripts\python.exe -m treino.v5.avaliar --output keilinks_data/evaluations/candidato.json
python -m api.servidor_v6
~~~

Execute testes antes de todo commit que afete runtime, dados, segurança ou
prompt. Mantenha checkpoints, caches e dados processados fora do Git; guarde
manifest, hash e métricas como artefatos de experimento.

## Fontes primárias consultadas

- [Unsloth — repositório e suporte a GPUs/Windows](https://github.com/unslothai/unsloth)
- [Qwen3-4B-Instruct-2507 Unsloth 4-bit — card e licença](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit)
- [vLLM — requisitos de instalação em GPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [SmolVLM2-2.2B-Instruct — card, capacidades e limites](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [Qwen3-VL-2B-Instruct — candidato multimodal](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [FineWeb-2 — por_Latn, tamanho e termos](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [CulturaX — dados e acesso/termos](https://huggingface.co/datasets/uonlp/CulturaX)
- [Corpus Carolina v1.2 — dados e licença](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina/tree/v1.2)
- [Pt-Corpus-Instruct — estrutura, fontes e limitações](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct)
- [Wikimedia/Wikipedia — dataset e licença](https://huggingface.co/datasets/wikimedia/wikipedia)
- [Qdrant — hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/) e [tutorial de hybrid search](https://qdrant.tech/documentation/tutorials-develop/hybrid-search-fastembed/)
- [DPO — artigo original](https://arxiv.org/abs/2305.18290)
- [GRPO/DeepSeekMath — artigo original](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer — documentação](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)
- [lm-evaluation-harness — framework e tarefas customizadas](https://github.com/EleutherAI/lm-evaluation-harness)
- [Ministério da Saúde — prevenção do suicídio](https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/s/suicidio-prevencao/suicidio-prevencao), [CVV 188](https://cvv.org.br/ligue-188-3/) e [SAMU 192](https://www.gov.br/saude/pt-br/composicao/saes/samu-192)

## Conclusão

O Keilinks já pode ser usado como assistente local generativo e é uma base
séria para evoluir. O que o deixa forte não é ligar todos os papers de uma vez:
é repetir o ciclo dados legais → experimento pequeno → avaliação difícil →
promoção reversível, preservando privacidade, segurança e a experiência em uma
GPU de 8 GB.
