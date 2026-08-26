"""Gera um currículo SFT curto e verificável de raciocínio em PT-BR.

Os exemplos não são cadeias de pensamento longas coletadas na internet. Cada
um contém um plano interno conciso e uma resposta final cuja correção pode ser
verificada deterministicamente. O arquivo gerado fica fora do Git por ser um
artefato de dados reproduzível.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from cerebro.raciocinio import MAX_PLAN_WORDS, format_reasoning_target, parse_reasoning_output

SYSTEM_PROMPT = (
    "Você é Keilinks, uma IA brasileira. Em tarefas que exigem mais cuidado, "
    "faça um plano interno curto, verificável e sem inventar fatos entre "
    "[[PLANO]] e [[/PLANO]]. Em seguida entregue a resposta natural ao usuário "
    "entre [[RESPOSTA]] e [[/RESPOSTA]]."
)
SOURCE = "keilinks_reasoning_short_v1"


def money(value: Decimal) -> str:
    return f"R$ {value.quantize(Decimal('0.01')):.2f}".replace(".", ",")


def record(category: str, prompt: str, plan: str, answer: str, index: int) -> dict:
    target = format_reasoning_target(plan, answer)
    parsed = parse_reasoning_output(target)
    if parsed.final != answer or not parsed.has_complete_protocol:
        raise ValueError(f"Exemplo de raciocínio inválido: {category}:{index}")
    if len(parsed.plan.split()) > MAX_PLAN_WORDS:
        raise ValueError(f"Plano longo demais: {category}:{index}")
    canonical = f"{category}\n{prompt}\n{target}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "id": digest,
        "group_id": f"reasoning-short-v1-{index:04d}",
        "source": SOURCE,
        "dataset_id": "keilinks/reasoning-short-v1",
        "dataset_revision": "1",
        "license": "Keilinks-internal",
        "source_url": "local://treino/v4/preparar_raciocinio.py",
        "synthetic": True,
        "quality": "deterministic_verified",
        "reasoning_protocol": "short_plan_v1",
        "category": category,
        "content_sha256": digest,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
    }


def arithmetic_examples(rng: random.Random, start: int) -> list[dict]:
    examples: list[dict] = []
    scenarios = (
        "biblioteca escolar",
        "horta comunitária",
        "oficina de robótica",
        "feira de ciências",
        "estoque da papelaria",
        "laboratório de química",
        "campanha de doação",
        "equipe de manutenção",
        "clube de leitura",
        "curso de fotografia",
        "produção de camisetas",
        "arquivo do museu",
        "agenda da clínica",
        "central de atendimento",
        "aula de programação",
        "torneio de xadrez",
    )
    moments = (
        "no turno da manhã",
        "na conferência semanal",
        "durante o inventário",
        "no fechamento do projeto",
    )
    for offset in range(192):
        operation = offset % 3
        case = offset // 3
        context = f"{scenarios[case % len(scenarios)]} {moments[case // len(scenarios)]}"
        first = rng.randint(12, 480)
        second = rng.randint(3, 120)
        if operation == 0:
            prompt = f"Na {context}, quanto é {first} + {second}?"
            answer = f"{first} + {second} = {first + second}."
            plan = f"Identificar os dois valores. Somar {first} e {second}. Conferir o resultado."
        elif operation == 1:
            prompt = f"Na {context}, quanto é {first} × {second}?"
            answer = f"{first} × {second} = {first * second}."
            plan = f"Identificar os fatores. Multiplicar {first} por {second}. Conferir a conta."
        else:
            minuend = first + second
            prompt = f"Na {context}, quanto é {minuend} − {second}?"
            answer = f"{minuend} − {second} = {first}."
            plan = (
                f"Identificar minuendo e subtraendo. Calcular {minuend} menos {second}. Conferir."
            )
        examples.append(record("arithmetic", prompt, plan, answer, start + offset))
    return examples


def discount_examples(rng: random.Random, start: int) -> list[dict]:
    examples: list[dict] = []
    percentages = (5, 10, 20, 25)
    products = (
        "mochila",
        "teclado",
        "cadeira",
        "fone de ouvido",
        "caderno",
        "curso online",
        "luminária",
        "monitor",
        "livro técnico",
        "garrafa térmica",
        "impressora",
        "mesa de estudo",
        "mouse sem fio",
        "webcam",
        "calendário",
        "organizador",
    )
    campaigns = (
        "de volta às aulas",
        "de tecnologia",
        "de fim de semana",
        "da feira local",
        "de aniversário da loja",
        "para assinantes",
        "de renovação de estoque",
        "de compras sustentáveis",
    )
    for offset in range(128):
        product = products[offset % len(products)]
        campaign = campaigns[offset // len(products)]
        price = Decimal(rng.randrange(40, 260)) * Decimal("10")
        percentage = Decimal(rng.choice(percentages))
        discount = (price * percentage / Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        final_price = price - discount
        prompt = (
            f"Na campanha {campaign}, uma {product} custa {money(price)} e recebeu "
            f"{int(percentage)}% de desconto. "
            "Qual é o preço final?"
        )
        answer = f"O desconto é {money(discount)}; o preço final é {money(final_price)}."
        plan = (
            f"Usar o preço de {money(price)}. Calcular {int(percentage)}% de desconto. "
            "Subtrair o desconto do preço inicial e conferir os centavos."
        )
        examples.append(record("percentage", prompt, plan, answer, start + offset))
    return examples


def condition_examples(rng: random.Random, start: int) -> list[dict]:
    examples: list[dict] = []
    names = ("Ana", "Bruno", "Camila", "Diego", "Elisa", "Fábio")
    activities = (
        "atendimento ao cliente",
        "projeto de leitura",
        "oficina de robótica",
        "mutirão de limpeza",
        "curso de fotografia",
        "campanha de doação",
        "aula de programação",
        "equipe de esportes",
        "laboratório escolar",
        "organização do arquivo",
        "feira de ciências",
        "estoque da biblioteca",
        "plantão de dúvidas",
        "produção de conteúdo",
        "revisão de documentos",
        "treinamento de segurança",
        "clube de xadrez",
        "horta comunitária",
        "manutenção de computadores",
        "organização de eventos",
        "projeto de acessibilidade",
        "grupo de estudos",
        "planejamento financeiro",
        "equipe de suporte",
    )
    review_cycles = (
        "na avaliação mensal",
        "na revisão do projeto",
        "na checagem da semana",
        "no fechamento do trimestre",
    )
    for offset in range(96):
        activity = activities[offset % len(activities)]
        review_cycle = review_cycles[offset // len(activities)]
        name = rng.choice(names)
        delivered = bool(rng.randrange(2))
        target_met = bool(rng.randrange(2))
        delivered_text = "entregou" if delivered else "não entregou"
        target_text = "atingiu" if target_met else "não atingiu"
        prompt = (
            f"No {activity}, {review_cycle}, para receber o bônus a pessoa precisa entregar "
            "o relatório e atingir a meta. "
            f"{name} {delivered_text} o relatório e {target_text} a meta. Ela recebe o bônus?"
        )
        receives = delivered and target_met
        if receives:
            answer = f"Sim. {name} cumpriu as duas condições."
        else:
            missing = []
            if not delivered:
                missing.append("o relatório")
            if not target_met:
                missing.append("a meta")
            answer = f"Não. {name} precisa cumprir relatório e meta; falta {' e '.join(missing)}."
        plan = (
            "Listar as duas condições exigidas. Conferir relatório e meta separadamente. "
            "Concluir apenas se ambas forem verdadeiras."
        )
        examples.append(record("logical_conditions", prompt, plan, answer, start + offset))
    return examples


def comparison_examples(rng: random.Random, start: int) -> list[dict]:
    examples: list[dict] = []
    uses = (
        "backup de fotos",
        "arquivos da faculdade",
        "vídeos de aula",
        "documentos da equipe",
        "projetos de design",
        "gravações de podcast",
        "fotos de uma loja",
        "código-fonte",
        "relatórios de vendas",
        "músicas offline",
        "material de estudo",
        "arquivos de impressão",
        "vídeos de treinamento",
        "dados de pesquisa",
        "fotos de viagem",
        "arquivos de clientes",
        "projetos de arquitetura",
        "planilhas de orçamento",
        "portfólio de arte",
        "aulas gravadas",
        "documentos jurídicos",
        "arquivos de uma ONG",
        "registros de manutenção",
        "material de marketing",
    )
    audiences = (
        "para estudantes",
        "para uma equipe pequena",
        "para uma família",
        "para uso profissional",
    )
    for offset in range(96):
        use = uses[offset % len(uses)]
        audience = audiences[offset // len(uses)]
        worse_unit = rng.choice((4, 5, 6, 8))
        better_unit = rng.choice((2, 3))
        first_quantity = rng.choice((4, 5, 6, 8, 10))
        second_quantity = rng.choice((4, 5, 6, 8, 10))
        first_price = worse_unit * first_quantity
        second_price = better_unit * second_quantity
        first_is_better = bool(rng.randrange(2))
        if first_is_better:
            first_price, second_price = second_price, first_price
            first_unit, second_unit = better_unit, worse_unit
        else:
            first_unit, second_unit = worse_unit, better_unit
        prompt = (
            f"Para {use} {audience}, o plano A custa R$ {first_price} e entrega "
            f"{first_quantity} GB. "
            f"O plano B custa R$ {second_price} e entrega {second_quantity} GB. "
            "Qual tem menor custo por GB?"
        )
        chosen = "A" if first_is_better else "B"
        chosen_unit = first_unit if first_is_better else second_unit
        other_unit = second_unit if first_is_better else first_unit
        answer = (
            f"O plano {chosen}: custa R$ {chosen_unit} por GB, "
            f"contra R$ {other_unit} por GB do outro plano."
        )
        plan = (
            "Calcular preço dividido pela quantidade para cada plano. Comparar os dois custos "
            "unitários. Escolher o menor e conferir as unidades."
        )
        examples.append(record("comparison", prompt, plan, answer, start + offset))
    return examples


def research_examples(start: int) -> list[dict]:
    topics = (
        "a cotação do dólar comercial",
        "quem ocupa a presidência do Brasil",
        "o preço do plano mais barato de um serviço",
        "a previsão do tempo para amanhã em Recife",
        "as mudanças da versão mais recente do Python",
        "um alerta novo da Anvisa sobre um remédio",
        "o resultado do jogo de ontem",
        "o preço de uma passagem aérea para Lisboa",
    )
    contexts = (
        "hoje",
        "nesta semana",
        "neste mês",
        "na data desta consulta",
        "antes de eu tomar uma decisão",
        "com a fonte e a data verificadas",
        "sem confiar só em memória antiga",
        "considerando que a informação pode mudar",
    )
    examples: list[dict] = []
    for offset in range(64):
        topic = topics[offset % len(topics)]
        context = contexts[offset // len(topics)]
        prompt = f"Como posso confirmar {topic} {context}?"
        answer = (
            "Isso depende de informação atual. Vou pesquisar fontes confiáveis, conferir a data "
            "e citar o que sustentar a resposta antes de afirmar um valor ou fato."
        )
        plan = (
            "Identificar que o dado muda com o tempo. Pesquisar fontes atuais e confiáveis. "
            "Comparar data e contexto antes de responder com citação."
        )
        examples.append(record("research_and_uncertainty", prompt, plan, answer, start + offset))
    return examples


def debugging_examples(start: int) -> list[dict]:
    cases = (
        (
            "Meu script Python mostra ModuleNotFoundError. Como começo a corrigir?",
            "Confirme qual módulo falta e qual ambiente virtual executa o script. "
            "Instale a dependência nesse mesmo ambiente, rode um import simples e só "
            "depois repita o programa.",
        ),
        (
            "Minha API responde 404 depois de eu criar uma rota. O que verifico?",
            "Confira o caminho e método da rota, qual processo está escutando a porta e "
            "se o servidor foi reiniciado. Teste a rota diretamente antes de alterar mais código.",
        ),
        (
            "O teste passa na minha máquina, mas falha no CI. Qual é o primeiro plano?",
            "Compare versão de Python, dependências e variáveis de ambiente entre os dois "
            "locais. Reproduza o comando do CI localmente e reduza o erro a um caso mínimo.",
        ),
        (
            "O treino começou a dar loss NaN. Como investigar sem adivinhar?",
            "Pare a promoção do checkpoint, registre o passo e verifique dados, learning rate, "
            "gradientes e precisão. Retome apenas de um checkpoint saudável após um teste "
            "curto finito.",
        ),
    )
    contexts = (
        "num projeto novo",
        "depois de atualizar dependências",
        "no ambiente virtual da equipe",
        "em uma máquina Windows",
        "no CI Linux",
        "após restaurar um checkpoint",
        "sem alterar produção",
        "com logs limitados",
        "num serviço local",
        "antes de fazer deploy",
        "num teste mínimo",
        "com pouco espaço em disco",
        "após trocar de versão do Python",
        "num notebook de 8 GB de VRAM",
        "sem acesso à internet",
        "depois de retomar uma execução",
    )
    examples: list[dict] = []
    for offset in range(64):
        prompt, answer = cases[offset % len(cases)]
        prompt = f"{prompt} Contexto: {contexts[offset // len(cases)]}."
        plan = (
            "Reproduzir o sintoma. Separar configuração, dados e execução. Fazer uma verificação "
            "pequena que confirme a hipótese antes de aplicar uma mudança."
        )
        examples.append(record("debugging", prompt, plan, answer, start + offset))
    return examples


def build_examples(seed: int = 42) -> list[dict]:
    """Retorna 640 exemplos balanceados, independentes de serviços externos."""

    rng = random.Random(seed)
    groups: Iterable[list[dict]] = (
        arithmetic_examples(rng, 0),
        discount_examples(rng, 192),
        condition_examples(rng, 320),
        comparison_examples(rng, 416),
        research_examples(512),
        debugging_examples(576),
    )
    examples = [example for group in groups for example in group]
    rng.shuffle(examples)
    return examples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(records: Iterable[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, output)
    return count


def merge_jsonl(base: Path, generated: Path, output: Path) -> int:
    if not base.exists():
        raise FileNotFoundError(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for source in (base, generated):
            with source.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        target.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    os.replace(temporary, output)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara currículo de raciocínio curto da Keilinks"
    )
    parser.add_argument("--output", default="dados/v4/sft/reasoning_8k.jsonl")
    parser.add_argument("--manifest", default="dados/v4/sft/reasoning_8k.manifest.json")
    parser.add_argument("--base", default="dados/v4/sft/all_sft_380m.jsonl")
    parser.add_argument("--merged-output", default="dados/v4/sft/all_sft_380m_reasoning_8k.jsonl")
    parser.add_argument("--count", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 640:
        parser.error("--count deve ficar entre 1 e 640")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    examples = build_examples(args.seed)[: args.count]
    written = write_jsonl(examples, output)
    merged_count = None
    if not args.no_merge:
        merged_count = merge_jsonl(Path(args.base), output, Path(args.merged_output))
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "seed": args.seed,
        "examples": written,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "merged_output": None if args.no_merge else str(args.merged_output),
        "merged_examples": merged_count,
        "quality": "deterministic_verified",
        "max_plan_words": MAX_PLAN_WORDS,
        "note": "Currículo curto; não usar como cadeia de pensamento exposta ao usuário.",
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
