"""Protocolo de raciocínio curto para a Keilinks.

O objetivo não é expor uma cadeia de pensamento longa ao usuário. Para tarefas
mais difíceis, o modelo pode produzir um plano interno pequeno e verificável;
o runtime conserva somente a resposta final. Isso deixa o comportamento
auditável e permite um currículo SFT sem transformar raciocínio em texto de
enchimento.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

PLAN_OPEN = "[[PLANO]]"
PLAN_CLOSE = "[[/PLANO]]"
ANSWER_OPEN = "[[RESPOSTA]]"
ANSWER_CLOSE = "[[/RESPOSTA]]"
MAX_PLAN_WORDS = 64
VALID_REASONING_MODES = frozenset({"auto", "always", "never"})


@dataclass(frozen=True)
class ParsedReasoning:
    """Resultado seguro de uma resposta que pode conter o protocolo interno."""

    final: str
    plan: str
    has_complete_protocol: bool


def normalize_reasoning_mode(value: object) -> str:
    """Normaliza entrada de API sem deixar texto arbitrário entrar no prompt."""

    normalized = str(value or "auto").strip().lower()
    return normalized if normalized in VALID_REASONING_MODES else "auto"


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", normalized.lower()).strip()


def requires_reasoning(message: str) -> bool:
    """Roteia somente pedidos que tendem a ganhar com planejamento explícito.

    A decisão é conservadora para não desacelerar saudações, conversa casual ou
    uma resposta factual simples. O botão do frontend usa ``always`` quando a
    pessoa quer forçar o protocolo.
    """

    text = _plain(message)
    if not text:
        return False
    signals = (
        "passo a passo",
        "pense com calma",
        "raciocine",
        "compare",
        "comparacao",
        "melhor opcao",
        "qual opcao",
        "planeje",
        "plano para",
        "estrategia",
        "como resolver",
        "como consertar",
        "como corrigir",
        "debug",
        "erro de codigo",
        "calcule",
        "calculo",
        "estimativa",
        "trade-off",
        "tradeoff",
        "decisao",
    )
    if any(signal in text for signal in signals):
        return True
    numeric_terms = re.findall(r"\d+(?:[.,]\d+)?", text)
    math_signals = ("quanto", "percent", "vezes", "divid", "multiplic", "media")
    return len(numeric_terms) >= 2 and any(signal in text for signal in math_signals)


def _short_plan(plan: str, maximum_words: int = MAX_PLAN_WORDS) -> str:
    words = re.sub(r"\s+", " ", str(plan or "")).strip().split(" ")
    if not words or not words[0]:
        return ""
    return " ".join(words[:maximum_words])


def reasoning_instruction() -> str:
    """Instrução aplicada apenas quando o roteador solicita raciocínio."""

    return (
        "\n\nMODO DE RACIOCÍNIO INTERNO: antes da resposta, faça um plano curto, "
        "concreto e verificável de no máximo 64 palavras entre [[PLANO]] e "
        "[[/PLANO]]. Depois escreva a resposta ao usuário entre [[RESPOSTA]] e "
        "[[/RESPOSTA]]. O plano deve listar fatos dados, ferramenta necessária "
        "(por exemplo pesquisa web) e uma checagem; nunca invente fontes, números "
        "ou etapas. A resposta final deve ser natural, completa e não mencionar esse "
        "protocolo interno."
    )


def format_reasoning_target(plan: str, answer: str) -> str:
    """Formata um alvo SFT sempre com plano breve e resposta separada."""

    compact_plan = _short_plan(plan)
    final = str(answer or "").strip()
    if not compact_plan:
        raise ValueError("O plano de raciocínio não pode ser vazio")
    if not final:
        raise ValueError("A resposta final não pode ser vazia")
    return f"{PLAN_OPEN}\n{compact_plan}\n{PLAN_CLOSE}\n{ANSWER_OPEN}\n{final}\n{ANSWER_CLOSE}"


def parse_reasoning_output(text: str) -> ParsedReasoning:
    """Remove de forma defensiva o plano interno antes de exibir a resposta."""

    raw = str(text or "").strip()
    plan_match = re.search(
        re.escape(PLAN_OPEN) + r"\s*(.*?)\s*" + re.escape(PLAN_CLOSE),
        raw,
        flags=re.DOTALL,
    )
    answer_match = re.search(
        re.escape(ANSWER_OPEN) + r"\s*(.*?)\s*" + re.escape(ANSWER_CLOSE),
        raw,
        flags=re.DOTALL,
    )
    plan = _short_plan(plan_match.group(1)) if plan_match else ""
    if answer_match:
        final = answer_match.group(1).strip()
    elif ANSWER_OPEN in raw:
        # A geração pode acabar antes de fechar a resposta. Nesse caso, o texto
        # após a abertura de RESPOSTA ainda é conteúdo destinado ao usuário.
        final = raw.split(ANSWER_OPEN, 1)[1]
        final = final.replace(ANSWER_CLOSE, "").strip()
        final = re.sub(
            re.escape(PLAN_OPEN) + r".*?" + re.escape(PLAN_CLOSE),
            "",
            final,
            flags=re.DOTALL,
        ).strip()
    elif PLAN_OPEN in raw and PLAN_CLOSE not in raw:
        # Sem fechamento do plano não há uma fronteira segura: prefira a
        # mensagem de fallback do runtime a expor raciocínio parcial.
        final = ""
    else:
        # Se uma geração parar no meio do protocolo, a melhor resposta segura é
        # remover qualquer plano completo e preservar o texto restante.
        final = re.sub(
            re.escape(PLAN_OPEN) + r".*?" + re.escape(PLAN_CLOSE),
            "",
            raw,
            flags=re.DOTALL,
        )
        final = final.replace(ANSWER_OPEN, "").replace(ANSWER_CLOSE, "")
        final = final.replace(PLAN_OPEN, "").replace(PLAN_CLOSE, "").strip()
    return ParsedReasoning(
        final=final,
        plan=plan,
        has_complete_protocol=bool(plan_match and answer_match),
    )
