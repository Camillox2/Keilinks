# Pesquisa de fontes para SFT conversacional V2

Data: 26/08/2026
Público: manutenção técnica do Keilinks Core 380M

## Escopo e decisão

O objetivo foi aumentar a capacidade de conversa em PT-BR sem contar registros
duplicados como dados novos, sem esconder a proveniência de traduções e sem
incorporar corpus com licença, consentimento ou privacidade incompatíveis.

A decisão é usar um SFT de 5.000 registros com 43% de exemplos orientados a
diálogo. A parcela de conversa original de maior confiança continua sendo
OpenAssistant OASST2 PT. Para complementar continuidade multi-turno, foram
incluídos somente 400 diálogos de ReDial-PTBR e 250 de UltraChatBR, sempre
marcados como `synthetic=True` e `translated=True`.

## Evidências e proveniência

| Fonte | Evidência verificada | Decisão |
| --- | --- | --- |
| [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | O coletor local preserva apenas cadeias PT revisadas, não removidas e não sintéticas; o manifesto fixa a revisão do dataset. | 1.500 exemplos humanos no mix. |
| [ReDial-PTBR](https://huggingface.co/datasets/matheusrdgsf/re_dial_ptbr) | O card declara licença MIT, 10.347 diálogos e que as conversas de recomendação foram traduzidas para PT-BR pela Maritalk. | 400 exemplos no máximo; converter o iniciador em usuário e o recomendador em assistente; remover marcadores `@` de filmes. |
| [UltraChatBR](https://huggingface.co/datasets/recogna-nlp/UltrachatBR) | O card declara licença MIT e que o corpus é uma tradução de UltraChat. | 250 exemplos multi-turno aprovados por filtros; nunca contar como diálogo humano. |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | A página exige aceite e compartilhamento de contato; contém conversas reais, dados de moderação e termos de remoção/destruição. | Excluído: não coletar sem aceite explícito e pipeline jurídico/privacidade próprio. |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | A página declara gate de acesso, licença AI2 ImpACT, conteúdo tóxico possível e desidentificação, mas não garantia absoluta de adequação ao produto. | Excluído: não coletar dados de conversa real sem revisão de política e curadoria. |
| [MultiWOZ-PT](https://github.com/NLP-CISUC/Dialog-State-Tracking-PT) | O repositório descreve 1.000 diálogos manualmente adaptados, mas não expõe uma licença inequívoca no repositório. | Excluído até haver confirmação de licença reutilizável. |

## Controle de qualidade aplicado

O coletor aceita somente mensagens bem formadas, aplica a rotina já existente
de normalização e redação de dados sensíveis, remove marcadores de template,
deduplica conversas canônicas e recusa respostas de assistente com excesso de
gíria ou metatexto de modelo desatualado. Para ReDial, só entram conversas com
dois participantes conhecidos, usuário primeiro e assistente por último.

Resultados do conjunto final `all_sft_380m_conversation_8k_v2.jsonl`:

- 5.000 registros, 5.000 IDs únicos e 5.000 conversas canônicas únicas;
- 0 registro inválido, 0 violação do filtro de estilo e 0 marcador `@` restante
  em registros ReDial;
- 1.054.073 tokens supervisionados de treino e 34.264 de validação depois do
  empacotamento em contexto de 8.192;
- SHA-256 do JSONL: `70c6fba132d0c5046906835ca366dd85623ac65f3d61833c1d2eae67090d000b`.

## Limitações e gates de promoção

O conjunto não transforma o modelo em uma fonte factual confiável nem substitui
pesquisa na web. ReDial é limitado ao domínio de recomendações de filmes e
UltraChatBR é traduzido por máquina; por isso ambos são pequenos e não definem
a identidade do assistente. OASST1 foi excluído da amostra final porque sua
sobreposição canônica com OASST2 era de 730 exemplos.

O SFT só pode iniciar após o checkpoint de pré-treino passar os gates de loss,
VRAM, checkpoint e avaliações congeladas. Após SFT, a promoção requer avaliação
conversacional, checagem de comportamento de pesquisa web, ausência de
vazamento do protocolo de plano e comparação contra o checkpoint anterior.
Feedback de usuário continua fora do treino automático até haver consentimento
e curadoria explícitos.
