# Conversas da Keilinks V4

A base conversacional é composta em camadas. Ela não depende apenas de pares curtos de pergunta e resposta.

## Âncora curada e versionada

`dados/v4/conversas_curadas_v4.jsonl` contém diálogos multi-turno revisados manualmente. Eles ensinam a Keilinks a:

- manter contexto entre mensagens;
- pedir informação quando a pergunta está incompleta;
- aceitar correções do usuário;
- adaptar uma explicação quando a pessoa não entende;
- responder de forma curta quando recebe feedback de excesso de texto;
- conversar sobre programação, AWS, banco de dados, trabalho e estudo;
- ser acolhedora sem fingir sentimentos ou incentivar dependência emocional;
- distinguir memória, RAG, conhecimento estável e informação atual;
- admitir incerteza e pesquisar fatos mutáveis.

O arquivo é uma âncora de qualidade e personalidade. Ele não deve ser repetido artificialmente para aumentar volume.

## Seeds curadas adicionais

```bash
python -m treino.v4.gerar_seed
```

Esse comando cria `dados/v4/seed_conversas.jsonl`, com variações de identidade, segurança, tecnologia, privacidade e factualidade.

## Conversas humanas externas

O pipeline de dados reconstrói árvores do OASST2 em conversas multi-turno em português, mantendo todos os exemplos da mesma árvore no mesmo split para evitar vazamento entre treino e validação.

## Conversas sintéticas revisadas

```bash
python -m treino.v4.gerar_conversas_ollama \
  --model qwen3.5:4b \
  --target 50000 \
  --batch-size 8 \
  --min-score 7
```

O gerador usa duas passagens: um modelo professor cria o diálogo e um crítico independente atribui notas de relevância, português, correção, segurança e estilo. Duplicatas e respostas problemáticas são rejeitadas.

Dados sintéticos são limitados a 25% do mix final por padrão. O limite é verificado sobre a proporção realmente selecionada, e não apenas sobre o tamanho máximo solicitado.

## Montar o mix

```bash
python -m treino.v4.misturar_sft \
  --max-examples 200000 \
  --synthetic-ratio 0.25 \
  --translation-ratio-each 0.20
```

O mix inclui, quando disponíveis:

1. conversas multi-turno curadas da Keilinks;
2. seeds curadas;
3. OASST2 em português;
4. bases traduzidas opcionais com limite individual;
5. conversas sintéticas aprovadas pelo crítico.

## Validar antes do treino

```bash
python -m treino.v4.validar_conversas \
  --input dados/v4/conversas_curadas_v4.jsonl \
  --min-multiturn-ratio 1.0
```

O validador verifica JSON, IDs duplicados, alternância de papéis, término no assistant, vazamento de marcadores especiais e padrões comuns de segredos.

## Packing e loss

No packing, sistema e mensagens do usuário recebem label `-100`. Apenas os tokens produzidos pela Keilinks geram loss. Todos os turnos anteriores permanecem no contexto, permitindo que o modelo aprenda continuidade sem ser treinado para imitar o usuário.
