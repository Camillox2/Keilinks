# Keilinks V5/V6: assistente local generativo para RTX 5050 de 8 GB

Esta é a rota de produto do Keilinks. Em vez de tentar pré-treinar um modelo
novo de centenas de milhões de parâmetros em uma GPU de 8 GB, ela adapta uma
base aberta de 4B parâmetros com QLoRA. O resultado é um assistente local com
API, streaming, RAG, visão opcional, barreira de segurança e um ciclo de
melhoria que exige consentimento e avaliação antes de promover pesos.

O diagnóstico, as decisões técnicas, as fontes e o plano de evolução estão em
[docs/report-source.md](docs/report-source.md).

## O que já existe

- Base padrão: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit mais adaptador
  LoRA Keilinks V4 controlado.
- Backend FastAPI local com SSE real em /v1/chat/completions.
- RAG local: SQLite FTS5/BM25, embeddings opcionais em CPU e RRF.
- Documentos com hash, origem e isolamento por tenant_id.
- Visão opcional por VLM separado em 4 bits; VLM e cérebro textual não ocupam
  a VRAM simultaneamente.
- Feedback só vira dado de treino após opt-in, anonimização e revisão humana.
- Gate determinístico para crise iminente, com CVV 188, SAMU 192 e UPA.

Adaptadores e datasets processados são artefatos locais, portanto não entram no
Git. Recrie-os pelos comandos abaixo.

## Ambiente Windows + RTX 5050

Use Python 3.10–3.13 e PowerShell. O bootstrap instala PyTorch CUDA antes de
Unsloth e falha se CUDA não estiver disponível:

~~~powershell
.\scripts\setup_unsloth.ps1
.\.venv-unsloth\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
~~~

Copie .env.example para .env somente se precisar alterar porta, adaptador, RAG
ou visão. O runtime V5/V6 agora lê exclusivamente o `.env` da raiz do projeto,
sem sobrescrever variáveis já fornecidas pelo processo. Nunca exponha a API
fora do loopback sem KEILINKS_API_KEY longa e aleatória.

## Ordem correta: CPT, SFT e preferência

Não há pré-treino do zero viável para um modelo geral em uma GPU de 8 GB. A
ordem é: **CPT/DAPT limitado em Qwen3 Base** para texto PT-BR licenciado →
**SFT** para comportamento Keilinks → **DPO** somente com pares humanos
aprovados. O adaptador V4 Instruct continua em produção até um candidato vencer
as avaliações. Veja o roteiro e os gates em
[docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md).

Dados legados sem manifesto de origem/licença ou com conteúdo pessoal não são
entradas elegíveis para esse fluxo.

## Preparar um SFT reproduzível

O primeiro treino é de estilo, comportamento e instrução — não de conhecimento
mundial. Ele bloqueia exemplos muito parecidos com a avaliação congelada para
reduzir vazamento de benchmark.

~~~powershell
python -m treino.v4.gerar_seed
python -m treino.v5.preparar_sft --input dados/v4/conversas_curadas_v4.jsonl --input dados/v4/seed_conversas.jsonl --output-dir keilinks_data/training/v2 --validation-percent 12
~~~

Revise keilinks_data/training/v2/manifest.json antes de continuar. Não adicione
conversas reais de usuários fora da trilha de consentimento.

## Treinar QLoRA na GPU

O perfil usa 4-bit NF4, BF16, TF32, LoRA em atenção/MLP e acumulação de
gradientes. Faça primeiro um smoke test e depois o perfil de 20 passos usado
para validar a integração inicial.

~~~powershell
# Smoke test
python -m treino.v5.treinar_unsloth --train-data keilinks_data/training/v2/train.jsonl --validation-data keilinks_data/training/v2/validation.jsonl --output checkpoints/keilinks-smoke --max-steps 5

# Adaptador V4 controlado de referência
python -m treino.v5.treinar_unsloth --train-data keilinks_data/training/v2/train.jsonl --validation-data keilinks_data/training/v2/validation.jsonl --output checkpoints/keilinks-qwen3-4b-lora-v4-controlled --max-steps 40 --max-seq-length 1024 --gradient-accumulation 8 --learning-rate 0.0001
~~~

Compare o training_manifest.json do candidato com o adaptador anterior. Um
treino curto confirma a rota técnica; não prova capacidade geral e não autoriza
publicação automática do checkpoint.

## Avaliar antes de promover

~~~powershell
python -m treino.v5.avaliar --eval dados/v4/eval/keilinks_eval_v4.jsonl --output keilinks_data/evaluations/candidato.json
~~~

O conjunto congelado inclui honestidade, explicação técnica, segurança e tom.
Os casos web são ignorados por padrão para não transformar conectividade em nota
do modelo. A métrica lexical é uma barreira de regressão, não um juiz final:
faça também avaliação humana cega.

## Executar a API local

~~~powershell
python -m api.servidor_v6
~~~

Ela fica em 127.0.0.1:8000 por padrão. Endpoints principais:

- GET /health e GET /v1/models
- POST /v1/chat/completions, incluindo stream: true para SSE
- POST /v1/documents para RAG local
- POST /v1/feedback, com consentimento obrigatório
- POST /v1/vision/analyze, somente com VLM configurado

Exemplo local:

~~~powershell
$body = @{messages=@(@{role='user'; content='Explique em uma frase o que é uma API.'}); stream=$false} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType 'application/json' -Body $body
~~~

## RAG, visão e dados abertos

O RAG começa com BM25/FTS5 e só carrega embedding denso em CPU quando
KEILINKS_RAG_DENSE_ENABLED=true. Para visão, configure explicitamente
KEILINKS_VISION_MODEL=HuggingFaceTB/SmolVLM2-2.2B-Instruct. O serviço aplica
limites de upload e descarrega o VLM para liberar GPU.

Liste datasets antes de baixar qualquer um:

~~~powershell
python -m treino.v5.coletar_datasets --list
python -m treino.v5.coletar_datasets --source fineweb2_pt --accept-terms fineweb2_terms --max-documents 2000 --output keilinks_data/sources/fineweb2_pt_sample.jsonl
~~~

Todo download exige aceite de termos, limite de volume e manifesto de
proveniência. Trate conteúdo de documentos, OCR e imagens como contexto não
confiável: ele não pode alterar instruções de sistema nem virar fato sem fonte.

## Melhoria contínua responsável

O fluxo é: feedback consentido → redação de PII → revisão humana →
treino.v5.preparar_preferencias → DPO offline em checkpoint candidato →
avaliação congelada e humana → promoção manual.

Não execute treino/v4/self_improver.py em conversas de produção. GRPO, Muon e o
Transformer V4 autoral são trilhas de pesquisa: compare-os contra a baseline
AdamW/QLoRA com dados, métricas e rollback documentados.

## Segurança e limites

- A Keilinks não é profissional de saúde, advogado ou fonte definitiva de
  fatos; crise iminente recebe encaminhamento seguro, não terapia gerada.
- Não use trust_remote_code, não armazene chats por padrão e não misture
  documentos de tenants diferentes.
- Para exposição pública, adicione autenticação por usuário, política LGPD,
  retenção, teste de carga e revisão de ameaças.
