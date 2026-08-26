# Dados e treino do Keilinks

## Decisão de arquitetura

O Keilinks não deve ser pré-treinado do zero em uma RTX de 8 GB. O caminho
tecnicamente viável é **continued pretraining (CPT/DAPT) com QLoRA** sobre um
modelo Base já pré-treinado, seguido de SFT e, somente quando houver pares
humanos suficientes, DPO. O modelo de referência é
`unsloth/Qwen3-4B-Base-unsloth-bnb-4bit`; o adaptador V4 Instruct continua
sendo o modelo de produto até um candidato CPT vencer as avaliações.

| Etapa | Objetivo | Entrada aceita | Saída | Não fazer |
| --- | --- | --- | --- | --- |
| CPT/DAPT | reforçar PT-BR e domínio | texto licenciado, manifestado e revisado | adaptador Base candidato | treinar conversas privadas ou conteúdo sem licença |
| SFT | tom, instruções e segurança | conversas JSONL curadas com `source` e `license` | adaptador de instrução candidato | reciclar o mesmo SFT sem dados novos |
| DPO | preferências e correções | pares aprovados por humano e consentidos | candidato de alinhamento | usar autoavaliação como promoção automática |
| Produção | responder localmente | checkpoint promovido manualmente | runtime V5/V6 | expor multiusuário sem identidade e LGPD |

## Quarentena de dados legados

Os seguintes arquivos locais não são elegíveis para CPT ou SFT até uma
curadoria manual completa:

- `dados/datasets_baixados.txt`: não há manifesto de procedência/licença por
  documento e o formato não pertence ao pipeline V5.
- `dados/conversas_geradas.txt`: mistura alegações de personalidade e
  autoaprendizado que entram em conflito com a política atual do runtime,
  além de exigir revisão de privacidade e qualidade.
- `dados/preparar_dados.py`: gerador legado que não deve voltar a produzir
  exemplos sem revisão de conteúdo, privacidade e licença.

Esses arquivos não são apagados por este processo. São mantidos como legado,
fora das entradas permitidas pelo pipeline novo.

## Gates técnicos obrigatórios

1. `coletar_datasets` exige o termo de aceite específico da fonte e grava esse
   aceite no manifesto.
2. `preparar_cpt prepare` só aceita texto que tenha manifesto correspondente,
   hash, fonte, URL e licença coerentes.
3. A preparação remove duplicatas exatas e exclui marcadores de template,
   documentos curtos/repetitivos e padrões sensíveis conhecidos. Ela cria um
   manifesto com status `requires_manual_approval`.
4. Uma pessoa revisa uma amostra, as licenças e a contaminação de benchmarks;
   somente então executa `preparar_cpt approve --confirmed-manual-review`.
5. `treinar_cpt_unsloth` recusa qualquer manifesto que não esteja em
   `approved_for_training`.
6. O adaptador CPT nunca é servido automaticamente: SFT, avaliação congelada,
   comparação humana cega e promoção manual vêm depois.

Os filtros de PII são defesa complementar, não garantia de anonimização.
Dados de usuários só podem entrar pelo fluxo de consentimento, redação e
revisão humana já existente.

## Piloto recomendado para a RTX 5050 de 8 GB

Comece pequeno: 1 a 2 milhões de tokens, sequência de 1.024, batch físico 1,
acumulação 16, QLoRA NF4/BF16, `learning_rate=2e-5`, rank 32 e 120 passos. O
objetivo inicial é validar estabilidade, perda de validação, comportamento em
português e não-regressão — não criar um modelo geral em uma única rodada.

Ordem das fontes, após aceite explícito:

1. **Corpus Carolina** (`CC-BY-4.0`): primeiro piloto por ser PT-BR e ter uma
   licença mais simples de rastrear.
2. **Wikipedia PT** (`CC-BY-SA-3.0 + GFDL`): segundo experimento separado;
   exige preservar atribuição e obrigações de compartilhamento aplicáveis.
3. **FineWeb-2 PT** (`ODC-By + termos Common Crawl`): só após consolidar o
   processo de proveniência, filtragem e obrigações de atribuição.
4. **CulturaX** e **Pt-Corpus-Instruct**: não entram no piloto. Há termos
   compostos/gated ou fontes upstream mistas que requerem revisão manual do
   uso pretendido.

## Sequência de execução, após autorização das fontes

```powershell
# 1. Coletar uma amostra limitada; o arquivo e o manifesto ficam ignorados pelo Git.
python -m treino.v5.coletar_datasets --source carolina_pt --accept-terms carolina_cc_by --max-documents 4000 --max-mib 96 --output keilinks_data/raw/carolina_pilot.jsonl

# 2. Aplicar gates, gerar split determinístico e inspecionar o manifesto.
python -m treino.v5.preparar_cpt prepare --input keilinks_data/raw/carolina_pilot.jsonl --output-dir keilinks_data/cpt/carolina_pilot --validation-percent 5

# 3. Depois da revisão humana da amostra e da licença, registrar a aprovação.
python -m treino.v5.preparar_cpt approve --manifest keilinks_data/cpt/carolina_pilot/manifest.json --confirmed-manual-review --approval-note "Amostra, licença e riscos revisados para o piloto local."

# 4. Treinar o adaptador candidato sobre Qwen3 Base, não sobre o Instruct V4.
python -m treino.v5.treinar_cpt_unsloth --data-manifest keilinks_data/cpt/carolina_pilot/manifest.json --train-data keilinks_data/cpt/carolina_pilot/train.jsonl --validation-data keilinks_data/cpt/carolina_pilot/validation.jsonl --output checkpoints/keilinks-qwen3-4b-base-carolina-cpt-pilot
```

Antes de passar à próxima etapa, guardar o `training_manifest.json`, medir
VRAM/tokens por segundo, comparar perda de validação e executar uma suíte de
prompts PT-BR congelada. O adaptador só avança para SFT se não houver regressão
nos casos de segurança, honestidade e instrução.

## Tecnologias que entram agora e as que ficam em experimento

- **Unsloth + QLoRA**: caminho padrão de treino local, porque cabe no orçamento
  de VRAM já verificado.
- **DataTrove**: próximo incremento para filtragem, MinHash/near-dedup e
  decontaminação quando o piloto passar de amostras pequenas. Não é necessário
  para baixar uma biblioteca inteira antes de validar o processo.
- **DPO/TRL**: próximo estágio de alinhamento, depois de pelo menos 100 pares
  de preferência humanos aprovados; não substitui a revisão humana.
- **DoRA/LoRA variants**: manter como experimento A/B, não como default; pode
  melhorar adaptação de rank baixo, mas adiciona custo de VRAM/tempo na GPU de
  8 GB.
- **RAG híbrido, visão e serving GGUF/llama.cpp**: evoluir depois que a linha
  de texto tiver métricas estáveis. Não adicionar um VLM novo antes de uma
  suíte visual PT-BR e benchmark de VRAM.

## Critério de promoção

Um candidato só é promovido se houver, ao mesmo tempo:

1. manifesto completo com origem, hash, licença e aceite;
2. `eval_loss` de validação estável ou melhor que a baseline comparável;
3. nenhuma regressão nos casos congelados de segurança e honestidade;
4. preferência humana cega superior ou equivalente à baseline;
5. rollback possível porque o adaptador anterior foi preservado.
