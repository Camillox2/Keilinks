"""Prepara um SFT PT-BR com prioridade para conversa natural.

O objetivo não é aumentar a contagem por duplicação.  O conjunto principal usa
conversas humanas verificadas como maior fatia e mantém cada origem rastreável.
Dados traduzidos ou escritos pelo assistente são sempre marcados como sintéticos
no registro e no manifesto; eles não podem se passar por conversa humana.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treino.v4.coletar_conversas import clean_messages, dataset_revision
from treino.v4.dataset import canonical_conversation

ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = ROOT / "dados" / "v4" / "sft"

# A IA deve entender fala informal do usuário, mas responder em PT-BR neutro.
# A lista é deliberadamente pequena: não pune uma palavra coloquial isolada,
# apenas impede que o estilo do assistente seja treinado como cheio de gírias.
ASSISTANT_SLANG = re.compile(
    r"\b(?:vc|vcs|blz|bora|mano|maninha|véi|vei|tá ligado|"
    r"show de bola|top demais|kkk+)\b",
    flags=re.IGNORECASE,
)
STALE_META = (
    "como modelo de linguagem",
    "sou apenas uma ia",
    "não tenho acesso à internet",
    "nao tenho acesso a internet",
    "não posso acessar a internet",
    "nao posso acessar a internet",
    "meu conhecimento vai até",
)
MOVIE_TOKEN = re.compile(r"@(\d+)")

# A distribuição prioriza conversar antes de ampliar conhecimento genérico.
# "translated_dialogue" contém somente diálogos multi-turno auditados, sempre
# marcados como tradução/sintético; portanto, não infla a cota humana.
TARGET_RATIOS: tuple[tuple[str, float], ...] = (
    ("human_dialogue", 0.30),
    ("translated_dialogue", 0.13),
    ("human_instruction", 0.32),
    ("synthetic_instruction", 0.14),
    ("short_reasoning", 0.07),
    ("behavior_anchor", 0.04),
)
DEFAULT_TOTAL = 5_000


@dataclass(frozen=True)
class NeutralScenario:
    """Um cenário comportamental, não uma fonte de fatos para pré-treino."""

    category: str
    request: str
    constraint: str
    diagnosis: str
    first_action: str
    evidence: str


NEUTRAL_SCENARIOS: tuple[NeutralScenario, ...] = (
    NeutralScenario(
        "clarification",
        "Quero organizar um projeto, mas ainda não sei por onde começar.",
        "Tenho pouco tempo esta semana e não quero criar uma lista impossível de cumprir.",
        "O problema parece ser falta de prioridade, não falta de vontade.",
        "escrever em uma frase o resultado que faria esta semana valer a pena",
        "uma lista curta de três entregas com prazo realista",
    ),
    NeutralScenario(
        "planning",
        "Estou com várias tarefas abertas e fico pulando de uma para outra.",
        "Algumas dependem de outras pessoas, então não controlo todos os prazos.",
        "Há tarefas executáveis agora e tarefas que precisam apenas de acompanhamento.",
        "separar o que você pode concluir hoje do que precisa de uma resposta externa",
        "menos trocas de contexto e uma pendência clara para cada dependência",
    ),
    NeutralScenario(
        "learning",
        "Quero aprender um assunto técnico sem me perder em vídeos aleatórios.",
        "Eu ainda não sei quais conceitos são realmente fundamentais.",
        "Antes de aprofundar, vale construir uma sequência mínima de conceitos e exercícios.",
        "escolher um objetivo prático pequeno que mostre se o conceito foi entendido",
        "conseguir explicar o conceito com suas palavras e resolver um exemplo simples",
    ),
    NeutralScenario(
        "debugging",
        "Meu código parou de funcionar e eu não sei qual mudança causou o problema.",
        "Não quero alterar várias coisas ao mesmo tempo e piorar o diagnóstico.",
        "O caminho mais seguro é isolar uma hipótese por vez e registrar o resultado.",
        "reproduzir o erro com o menor exemplo possível antes de tentar corrigir",
        "um erro reproduzível e um teste que passa depois da correção",
    ),
    NeutralScenario(
        "feedback",
        "Recebi uma crítica no trabalho e não sei se devo mudar tudo ou só uma parte.",
        "A crítica foi vaga e eu não quero interpretar intenção onde não existe.",
        "É melhor transformar a opinião em observações verificáveis antes de reagir.",
        "pedir um exemplo concreto do resultado que a pessoa gostaria de ver diferente",
        "uma alteração específica que possa ser revisada depois",
    ),
    NeutralScenario(
        "uncertainty",
        "Preciso tomar uma decisão, mas as informações que encontrei se contradizem.",
        "Uma fonte parece antiga e a outra não explica de onde tirou os números.",
        "A incerteza está nas fontes, portanto a próxima ação é verificar e não adivinhar.",
        "anotar quais afirmações precisam de fonte primária ou recente",
        "duas fontes confiáveis que concordem nos fatos relevantes",
    ),
    NeutralScenario(
        "web_research",
        "Quero comparar duas opções que mudam de preço e disponibilidade com frequência.",
        "Não quero decidir com base em uma informação desatualizada.",
        "Esta é uma pergunta que pede pesquisa atual, critérios explícitos e fontes citáveis.",
        "definir os critérios de comparação antes de pesquisar",
        "links recentes, data de consulta e uma recomendação condicionada aos critérios",
    ),
    NeutralScenario(
        "privacy",
        "Quero guardar detalhes de uma conversa, mas não sei o que é seguro registrar.",
        "Alguns detalhes são pessoais e talvez não sejam necessários para a próxima conversa.",
        "Memória útil deve conter preferência e contexto durável, não dados sensíveis por padrão.",
        "separar o que é preferência estável do que deve ficar apenas nesta conversa",
        "uma nota curta, revisável e sem informação desnecessariamente sensível",
    ),
    NeutralScenario(
        "correction",
        "Você me deu uma resposta que pareceu incompleta. "
        "Como posso te corrigir sem recomeçar tudo?",
        "Quero aproveitar o contexto que já expliquei.",
        "Uma boa correção aponta o trecho incorreto, o resultado esperado "
        "e qualquer restrição nova.",
        "descrever em uma frase o que ficou errado e mostrar um exemplo do que seria melhor",
        "uma resposta revisada que reconheça a correção e não repita o erro",
    ),
    NeutralScenario(
        "career",
        "Quero melhorar meu currículo, mas tenho medo de deixar o texto genérico.",
        "Minha experiência mistura tarefas técnicas e atendimento a pessoas.",
        "O texto fica mais forte quando mostra contexto, ação e resultado verificável.",
        "escolher uma experiência e escrever o problema, sua ação e o efeito observado",
        "uma frase específica que alguém de fora consiga entender e checar",
    ),
    NeutralScenario(
        "writing",
        "Tenho uma mensagem importante para enviar e estou deixando ela longa demais.",
        "Não quero soar frio nem perder um pedido essencial.",
        "A mensagem precisa de objetivo claro, contexto mínimo e uma próxima ação simples.",
        "escrever primeiro a frase que diz exatamente o que você está pedindo",
        "uma mensagem curta com pedido, prazo e espaço para resposta",
    ),
    NeutralScenario(
        "decision",
        "Estou dividido entre duas alternativas e cada uma resolve uma parte do problema.",
        "Não existe uma opção perfeita e eu não quero inventar certeza.",
        "A decisão melhora quando os critérios recebem peso conforme o impacto para você.",
        "listar três critérios e dizer qual deles você não aceita comprometer",
        "uma escolha explicada por critérios, com a principal troca reconhecida",
    ),
    NeutralScenario(
        "habits",
        "Quero criar uma rotina, mas sempre abandono quando o dia fica cheio.",
        "Meu horário muda bastante e metas rígidas costumam falhar.",
        "Uma rotina flexível precisa de um gatilho simples e uma versão mínima para dias difíceis.",
        "definir a menor versão da atividade que ainda conta como continuidade",
        "conseguir manter a sequência mesmo quando o tempo disponível for curto",
    ),
    NeutralScenario(
        "emotional_support",
        "Estou frustrado porque um plano em que eu investi tempo não deu certo.",
        "Não quero apenas ouvir que tudo vai ficar bem; preciso pensar no próximo passo.",
        "Reconhecer a frustração e separar o que aconteceu da sua capacidade "
        "ajuda a retomar o controle.",
        "anotar o que funcionou, o que não funcionou e qual hipótese pode ser testada depois",
        "uma ação pequena que reduza a incerteza sem exigir recomeçar tudo",
    ),
    NeutralScenario(
        "health_boundary",
        "Estou com uma dúvida de saúde e encontro orientações diferentes na internet.",
        "Não quero substituir uma avaliação profissional por uma resposta geral.",
        "Informação geral pode orientar perguntas, mas sintomas, diagnóstico "
        "e urgência precisam de profissional.",
        "registrar os sintomas, duração e sinais de alerta para levar a um serviço adequado",
        "saber qual profissional procurar e quais sinais exigem atendimento mais rápido",
    ),
    NeutralScenario(
        "finance_boundary",
        "Quero organizar meu orçamento, mas não sei se devo seguir uma regra pronta.",
        "Minha renda e meus gastos variam de mês para mês.",
        "Regras genéricas servem como ponto de partida, não como substituto "
        "do seu fluxo real de dinheiro.",
        "mapear por um mês os gastos fixos, variáveis e compromissos inadiáveis",
        "uma margem de segurança compatível com seus números reais",
    ),
    NeutralScenario(
        "collaboration",
        "Uma pessoa da equipe interpretou minha mensagem de outro jeito e agora o trabalho travou.",
        "Quero resolver sem assumir culpa por algo que não entendi completamente.",
        "Voltar ao objetivo compartilhado e descrever fatos observáveis reduz defensividade.",
        "propor uma conversa curta com o resultado esperado e os pontos ainda ambíguos",
        "um acordo escrito sobre responsável, prazo e critério de pronto",
    ),
    NeutralScenario(
        "creative_work",
        "Estou bloqueado em uma ideia criativa e todas as versões parecem ruins.",
        "Fico julgando enquanto ainda estou tentando gerar opções.",
        "Criar e avaliar são tarefas diferentes; separá-las dá espaço "
        "para alternativas aparecerem.",
        "produzir três rascunhos rápidos sem escolher o melhor durante a primeira etapa",
        "um critério claro para comparar os rascunhos depois",
    ),
    NeutralScenario(
        "technical_explanation",
        "Recebi uma explicação técnica, mas ainda não sei como aplicar no meu caso.",
        "O texto usa termos que eu entendo isoladamente, mas não juntos.",
        "Uma explicação útil começa pelo objetivo, mostra o mecanismo "
        "e termina em um exemplo pequeno.",
        "reformular a dúvida com seu caso concreto e o resultado que você espera",
        "um exemplo executável ou observável que confirme o entendimento",
    ),
    NeutralScenario(
        "natural_conversation",
        "Hoje eu só queria conversar um pouco, sem transformar tudo em um plano.",
        "Se eu pedir uma ideia ou uma ação, prefiro que você não presuma isso antes.",
        "Uma conversa de apoio pode começar por escuta e contexto, sem pressa de resolver.",
        "perguntar qual parte da situação está mais presente para você agora",
        "uma resposta que acompanhe o que foi dito e só ofereça sugestões quando forem bem-vindas",
    ),
    NeutralScenario(
        "memory_boundary",
        "Quero que você lembre uma preferência minha, mas não quero ficar preso a ela para sempre.",
        "Minha necessidade pode mudar nas próximas conversas.",
        "Uma memória saudável precisa ser opcional, editável e fácil de apagar.",
        "registrar a preferência junto de uma condição que indique quando ela deve ser revista",
        "uma nota curta que você consiga corrigir ou remover sem esforço",
    ),
)

NEUTRAL_FRAMES: tuple[tuple[str, str, str], ...] = (
    (
        "Entendi. Vamos reduzir isso a uma decisão que caiba no momento atual.",
        "Isso altera a ordem do plano, não o objetivo.",
        "Você não precisa resolver o restante agora; basta deixar o próximo passo claro.",
    ),
    (
        "Faz sentido querer avançar sem transformar a situação em algo maior do que ela é.",
        "Essa restrição é importante e deve entrar no plano, não ser tratada como detalhe.",
        "Se esse sinal não aparecer, vale ajustar o plano em vez de insistir por inércia.",
    ),
    (
        "Vamos separar o que é certo, o que ainda é hipótese e o que pode ser testado.",
        "Com essa condição, a opção mais simples passa a ser a mais segura.",
        "Registrar o resultado torna a próxima decisão mais fácil e menos baseada em impressão.",
    ),
    (
        "Você trouxe uma restrição útil; ela evita uma resposta genérica.",
        "Não é necessário esperar certeza total para dar um passo pequeno e reversível.",
        "O importante é observar um critério concreto, não apenas a sensação de que melhorou.",
    ),
)


def _stable_id(prefix: str, messages: Sequence[dict[str, str]]) -> str:
    canonical = canonical_conversation(messages)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def assistant_style_violation(messages: Sequence[dict[str, str]]) -> str | None:
    """Retorna a primeira razão para não ensinar estilo de assistente com gírias."""

    for message in messages:
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content") or "")
        lowered = text.lower()
        if "\ufffd" in text:
            return "replacement_character"
        if ASSISTANT_SLANG.search(text):
            return "assistant_slang"
        if any(fragment in lowered for fragment in STALE_META):
            return "stale_assistant_meta"
    return None


def _clean_record(record: dict[str, Any]) -> dict[str, Any] | None:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        return None
    messages = clean_messages(raw_messages)
    if not messages or assistant_style_violation(messages):
        return None
    cleaned = dict(record)
    cleaned["messages"] = messages
    cleaned.setdefault("id", _stable_id(str(cleaned.get("source") or "record"), messages))
    cleaned.setdefault("group_id", str(cleaned["id"]))
    return cleaned


def _load_jsonl(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Fonte SFT ausente: {path}")
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {path}:{line_number}") from exc
                if not isinstance(item, dict):
                    rejected["non_object"] += 1
                    continue
                cleaned = _clean_record(item)
                if cleaned is None:
                    rejected["invalid_or_style"] += 1
                    continue
                canonical = canonical_conversation(cleaned["messages"])
                if canonical in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(canonical)
                records.append(cleaned)
    return records, rejected


def _sample(records: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"Precisava de {count} exemplos, mas só há {len(records)} válidos")
    ordered = sorted(records, key=lambda item: str(item.get("id") or ""))
    chosen = random.Random(seed).sample(ordered, count)
    return sorted(chosen, key=lambda item: str(item.get("id") or ""))


def allocate_targets(total: int) -> dict[str, int]:
    """Distribui inteiros preservando exatamente a soma e as razões declaradas."""

    if total <= 0:
        raise ValueError("O total do SFT precisa ser positivo")
    raw = [(name, total * ratio) for name, ratio in TARGET_RATIOS]
    result = {name: math.floor(value) for name, value in raw}
    remaining = total - sum(result.values())
    for name, _ in sorted(raw, key=lambda item: (item[1] % 1, item[0]), reverse=True)[:remaining]:
        result[name] += 1
    return result


def _record_from_ultrachatbr(row: dict[str, Any], revision: str) -> dict[str, Any] | None:
    raw_turns = row.get("conversa")
    if isinstance(raw_turns, str):
        try:
            # Algumas linhas públicas têm escapes legados que geram SyntaxWarning
            # no parser. O dado será validado abaixo; não poluímos o terminal do
            # operador com centenas de avisos não acionáveis durante streaming.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                raw_turns = ast.literal_eval(raw_turns)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(raw_turns, list) or len(raw_turns) < 2:
        return None
    messages: list[dict[str, str]] = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            return None
        user = str(turn.get("humano") or "").strip()
        assistant = str(turn.get("assistente") or "").strip()
        if not user or not assistant:
            return None
        messages.extend((
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ))
    messages = clean_messages(messages)
    if len(messages) < 4 or assistant_style_violation(messages):
        return None
    joined = "\n".join(message["content"] for message in messages)
    if not 400 <= len(joined) <= 12_000:
        return None
    if len({message["content"].casefold() for message in messages}) != len(messages):
        return None
    source_id = str(row.get("conversation_id") or _stable_id("ultrachatbr", messages))
    return {
        "id": _stable_id("ultrachatbr_pt_filtered_v1", messages),
        "group_id": f"ultrachatbr:{source_id}",
        "source": "ultrachatbr_pt_filtered_v1",
        "source_id": source_id,
        "dataset_id": "recogna-nlp/UltrachatBR",
        "dataset_revision": revision,
        "source_url": "https://huggingface.co/datasets/recogna-nlp/UltrachatBR",
        "license": "MIT",
        "synthetic": True,
        "translated": True,
        "category": "filtered_multiturn_dialogue",
        "messages": messages,
    }


def _movie_title_map(raw_mentions: Any) -> dict[str, str]:
    """Converte IDs internos do ReDial em títulos legíveis quando disponíveis."""

    if isinstance(raw_mentions, dict):
        items: Iterable[Any] = raw_mentions.values()
    elif isinstance(raw_mentions, list):
        items = raw_mentions
    else:
        return {}
    titles: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        movie_id = item.get("movieId") or item.get("movie_id")
        movie_name = item.get("movieName") or item.get("movie_name")
        if movie_id is None or not isinstance(movie_name, str):
            continue
        name = movie_name.strip()
        if name and len(name) <= 240:
            titles[str(movie_id)] = name
    return titles


def _record_from_redial_ptbr(row: dict[str, Any], revision: str) -> dict[str, Any] | None:
    """Normaliza diálogo ReDial-PTBR sem ensinar marcadores internos de filme."""

    raw_turns = row.get("messages_translated")
    initiator = row.get("initiatorWorkerId")
    respondent = row.get("respondentWorkerId")
    if not isinstance(raw_turns, list) or len(raw_turns) < 6:
        return None
    if initiator is None or respondent is None or initiator == respondent:
        return None

    movie_titles = _movie_title_map(row.get("movieMentions"))

    def replace_movie_token(match: re.Match[str]) -> str:
        return movie_titles.get(match.group(1), "um filme")

    messages: list[dict[str, str]] = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            return None
        sender = turn.get("senderWorkerId")
        text = str(turn.get("text") or "").strip()
        if sender == initiator:
            role = "user"
        elif sender == respondent:
            role = "assistant"
        else:
            return None
        text = MOVIE_TOKEN.sub(replace_movie_token, text).replace("@", "")
        if not text:
            return None
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = f"{messages[-1]['content']}\n{text}"
        else:
            messages.append({"role": role, "content": text})

    messages = clean_messages(messages)
    if (
        len(messages) < 4
        or messages[0]["role"] != "user"
        or messages[-1]["role"] != "assistant"
        or assistant_style_violation(messages)
    ):
        return None
    joined = "\n".join(message["content"] for message in messages)
    if not 350 <= len(joined) <= 12_000:
        return None
    conversation_id = str(row.get("conversationId") or _stable_id("redial", messages))
    return {
        "id": _stable_id("redial_ptbr_filtered_v2", messages),
        "group_id": f"redial:{conversation_id}",
        "source": "redial_ptbr_filtered_v2",
        "source_id": conversation_id,
        "dataset_id": "matheusrdgsf/re_dial_ptbr",
        "dataset_revision": revision,
        "source_url": "https://huggingface.co/datasets/matheusrdgsf/re_dial_ptbr",
        "license": "MIT",
        "synthetic": True,
        "translated": True,
        "original_human_dialogue": True,
        "category": "translated_movie_recommendation_dialogue",
        "messages": messages,
    }


def collect_redial_ptbr(
    output: Path,
    *,
    limit: int,
    max_scan: int,
    accepted_terms: set[str],
) -> dict[str, Any]:
    """Coleta uma cota pequena de diálogos ReDial traduzidos para PT-BR."""

    if "redial_ptbr_mit" not in accepted_terms:
        raise ValueError(
            "ReDial-PTBR exige --accept-terms redial_ptbr_mit; "
            "leia o dataset card MIT antes de coletar."
        )
    if output.exists():
        raise FileExistsError(f"A saída já existe: {output}")
    if limit <= 0 or max_scan < limit:
        raise ValueError("--limit deve ser positivo e --max-scan precisa ser maior que --limit")
    from datasets import load_dataset

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Arquivo temporário já existe: {temporary}")
    revision = dataset_revision("matheusrdgsf/re_dial_ptbr")
    rejected: Counter[str] = Counter()
    accepted = 0
    seen: set[str] = set()
    dataset = load_dataset("matheusrdgsf/re_dial_ptbr", split="train", streaming=True)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for scanned, row in enumerate(dataset, 1):
                if scanned > max_scan or accepted >= limit:
                    break
                if not isinstance(row, dict):
                    rejected["non_object"] += 1
                    continue
                record = _record_from_redial_ptbr(row, revision)
                if record is None:
                    rejected["quality_gate"] += 1
                    continue
                canonical = canonical_conversation(record["messages"])
                if canonical in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(canonical)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                accepted += 1
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if accepted < limit:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Filtro aceitou apenas {accepted}/{limit} exemplos em {max_scan} linhas"
        )
    os.replace(temporary, output)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "output": str(output),
        "source": "matheusrdgsf/re_dial_ptbr",
        "license": "MIT",
        "dataset_revision": revision,
        "accepted": accepted,
        "max_scan": max_scan,
        "rejections": dict(sorted(rejected.items())),
        "gates": [
            "messages_translated only",
            "two known speakers; user first and assistant last",
            "movie marker removal",
            "assistant neutral-style filter",
            "length and duplicate filter",
        ],
        "provenance": (
            "human-human movie dialogues translated to PT-BR; "
            "always synthetic=True and translated=True"
        ),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def collect_ultrachatbr(
    output: Path,
    *,
    limit: int,
    max_scan: int,
    accepted_terms: set[str],
) -> dict[str, Any]:
    """Coleta só uma pequena camada filtrada do UltraChatBR via streaming."""

    if "ultrachatbr_mit" not in accepted_terms:
        raise ValueError(
            "UltrachatBR exige --accept-terms ultrachatbr_mit; "
            "leia o dataset card MIT antes de coletar."
        )
    if output.exists():
        raise FileExistsError(f"A saída já existe: {output}")
    if limit <= 0 or max_scan < limit:
        raise ValueError("--limit deve ser positivo e --max-scan precisa ser maior que --limit")
    from datasets import load_dataset

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Arquivo temporário já existe: {temporary}")
    revision = dataset_revision("recogna-nlp/UltrachatBR")
    rejected: Counter[str] = Counter()
    accepted = 0
    seen: set[str] = set()
    dataset = load_dataset("recogna-nlp/UltrachatBR", split="train", streaming=True)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for scanned, row in enumerate(dataset, 1):
                if scanned > max_scan or accepted >= limit:
                    break
                if not isinstance(row, dict):
                    rejected["non_object"] += 1
                    continue
                record = _record_from_ultrachatbr(row, revision)
                if record is None:
                    rejected["quality_gate"] += 1
                    continue
                canonical = canonical_conversation(record["messages"])
                if canonical in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(canonical)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                accepted += 1
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if accepted < limit:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Filtro aceitou apenas {accepted}/{limit} exemplos em {max_scan} linhas"
        )
    os.replace(temporary, output)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "output": str(output),
        "source": "recogna-nlp/UltrachatBR",
        "license": "MIT",
        "dataset_revision": revision,
        "accepted": accepted,
        "max_scan": max_scan,
        "rejections": dict(sorted(rejected.items())),
        "gates": [
            "multi-turn only",
            "valid role alternation",
            "assistant neutral-style filter",
            "stale assistant-meta filter",
            "length and duplicate filter",
        ],
        "provenance": "machine-translated UltraChat; always synthetic=True",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def generate_neutral_anchors(output: Path, *, count: int) -> dict[str, Any]:
    """Escreve âncoras de comportamento PT-BR neutro, declaradas como sintéticas."""

    maximum = len(NEUTRAL_SCENARIOS) * len(NEUTRAL_FRAMES)
    if count <= 0 or count > maximum:
        raise ValueError(f"--count deve ficar entre 1 e {maximum}")
    if output.exists():
        raise FileExistsError(f"A saída já existe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Arquivo temporário já existe: {temporary}")
    records: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(NEUTRAL_SCENARIOS):
        for frame_index, (opening, adaptation, closing) in enumerate(NEUTRAL_FRAMES):
            messages = [
                {"role": "user", "content": scenario.request},
                {
                    "role": "assistant",
                    "content": (
                        f"{opening} {scenario.diagnosis} Primeiro, sugiro "
                        f"{scenario.first_action}."
                    ),
                },
                {"role": "user", "content": scenario.constraint},
                {
                    "role": "assistant",
                    "content": (
                        f"{adaptation} Em vez de buscar uma resposta perfeita, "
                        f"faça uma ação pequena e reversível."
                    ),
                },
                {
                    "role": "user",
                    "content": "Como posso avaliar se esse plano está funcionando?",
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Use como sinal {scenario.evidence}. {closing} "
                        "Se quiser, posso ajudar a transformar isso em uma lista curta."
                    ),
                },
            ]
            messages = clean_messages(messages)
            if not messages or assistant_style_violation(messages):
                raise AssertionError("Âncora neutra violou a própria regra de estilo")
            records.append({
                "id": _stable_id("keilinks_authored_neutral_v1", messages),
                "group_id": f"authored-neutral:{scenario_index}:{frame_index}",
                "source": "keilinks_authored_neutral_v1",
                "source_id": f"{scenario_index}:{frame_index}",
                "license": "Keilinks-authored-training-data",
                "synthetic": True,
                "authoring": "assistant-authored neutral PT-BR behavior curriculum",
                "category": scenario.category,
                "messages": messages,
            })
    records = records[:count]
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "output": str(output),
        "records": len(records),
        "source": "keilinks_authored_neutral_v1",
        "synthetic": True,
        "style": "neutral_pt_br_no_assistant_slang",
        "categories": dict(Counter(record["category"] for record in records)),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _anchor_priority(record: dict[str, Any]) -> tuple[int, str]:
    source = str(record.get("source") or "")
    if source.startswith("keilinks_curated"):
        return (0, str(record.get("id") or ""))
    if source == "ultrachatbr_pt_filtered_v1":
        return (1, str(record.get("id") or ""))
    if source == "keilinks_authored_neutral_v1":
        return (2, str(record.get("id") or ""))
    return (3, str(record.get("id") or ""))


def build_conversation_mix(
    output: Path,
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = 42,
    human_dialogue_paths: Sequence[Path],
    translated_dialogue_paths: Sequence[Path],
    human_instruction_paths: Sequence[Path],
    synthetic_instruction_paths: Sequence[Path],
    reasoning_paths: Sequence[Path],
    anchor_paths: Sequence[Path],
) -> dict[str, Any]:
    """Monta o núcleo de SFT sem reamostragem/duplicação silenciosa."""

    targets = allocate_targets(total)
    buckets: dict[str, tuple[list[dict[str, Any]], Counter[str]]] = {
        "human_dialogue": _load_jsonl(human_dialogue_paths),
        "translated_dialogue": _load_jsonl(translated_dialogue_paths),
        "human_instruction": _load_jsonl(human_instruction_paths),
        "synthetic_instruction": _load_jsonl(synthetic_instruction_paths),
        "short_reasoning": _load_jsonl(reasoning_paths),
        "behavior_anchor": _load_jsonl(anchor_paths),
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    for index, (bucket, _) in enumerate(TARGET_RATIOS):
        records = buckets[bucket][0]
        if bucket == "behavior_anchor":
            ordered = sorted(records, key=_anchor_priority)
            if targets[bucket] > len(ordered):
                raise ValueError(
                    f"Âncoras insuficientes: precisava de {targets[bucket]}, há {len(ordered)}"
                )
            selected[bucket] = ordered[:targets[bucket]]
        else:
            selected[bucket] = _sample(records, targets[bucket], seed + index)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for bucket, _ in TARGET_RATIOS:
        for record in selected[bucket]:
            canonical = canonical_conversation(record["messages"])
            if canonical in seen:
                duplicates += 1
                continue
            seen.add(canonical)
            record = dict(record)
            record["mix_bucket"] = bucket
            merged.append(record)
    if len(merged) != total:
        raise RuntimeError(
            f"A deduplicação cruzada reduziu o mix para {len(merged)}/{total}; "
            "revise as fontes em vez de preencher com cópias."
        )

    random.Random(seed).shuffle(merged)
    if output.exists():
        raise FileExistsError(f"A saída já existe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Arquivo temporário já existe: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output)

    by_source = Counter(str(record.get("source") or "unknown") for record in merged)
    by_bucket = Counter(str(record["mix_bucket"]) for record in merged)
    synthetic_sources = {
        "tucano_sft",
        "keilinks_reasoning_short_v1",
        "redial_ptbr_filtered_v1",
        "redial_ptbr_filtered_v2",
        "ultrachatbr_pt_filtered_v1",
        "keilinks_authored_neutral_v1",
    }
    synthetic_total = sum(
        1 for record in merged
        if bool(record.get("synthetic", False)) or str(record.get("source")) in synthetic_sources
    )
    multi_turn_total = sum(1 for record in merged if len(record["messages"]) >= 4)
    dialogue_total = by_bucket["human_dialogue"] + by_bucket["translated_dialogue"]
    report = {
        "schema_version": 1,
        "status": "complete",
        "output": str(output),
        "seed": seed,
        "total": len(merged),
        "target_ratio": dict(TARGET_RATIOS),
        "target_count": targets,
        "actual_count_by_bucket": dict(by_bucket),
        "actual_ratio_by_bucket": {
            bucket: round(count / len(merged), 6) for bucket, count in by_bucket.items()
        },
        "actual_count_by_source": dict(by_source),
        "synthetic_total": synthetic_total,
        "synthetic_ratio": round(synthetic_total / len(merged), 6),
        "multi_turn_total": multi_turn_total,
        "multi_turn_ratio": round(multi_turn_total / len(merged), 6),
        "dialogue_total": dialogue_total,
        "dialogue_ratio": round(dialogue_total / len(merged), 6),
        "cross_bucket_duplicates": duplicates,
        "input_rejections": {
            bucket: dict(rejections) for bucket, (_, rejections) in buckets.items()
        },
        "style": "assistant responses filtered for excessive slang and stale model-meta",
        "provenance_note": (
            "translated and assistant-authored records remain synthetic in every stage"
        ),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _default_paths() -> dict[str, list[Path]]:
    public_01 = SFT_ROOT / "public-conversations-380m-01"
    public_02 = SFT_ROOT / "public-conversations-380m-02"
    candidates = SFT_ROOT / "candidates"
    return {
        "human_dialogue_paths": [
            public_01 / "oasst2_pt.jsonl",
            public_02 / "oasst1_pt.jsonl",
        ],
        "translated_dialogue_paths": [
            candidates / "redial_ptbr_filtered_v2.jsonl",
            candidates / "ultrachatbr_pt_filtered_v1.jsonl",
        ],
        "human_instruction_paths": [public_01 / "aya_pt.jsonl"],
        "synthetic_instruction_paths": [public_01 / "tucano_sft.jsonl"],
        "reasoning_paths": [SFT_ROOT / "reasoning_8k.jsonl"],
        "anchor_paths": [
            ROOT / "dados" / "v4" / "conversas_curadas_v4.jsonl",
            ROOT / "dados" / "v4" / "seed_conversas.jsonl",
            candidates / "keilinks_authored_neutral_v1.jsonl",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara SFT PT-BR com prioridade de diálogo e estilo neutro"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    redial = subparsers.add_parser("collect-redial-ptbr")
    redial.add_argument(
        "--output",
        default=str(SFT_ROOT / "candidates" / "redial_ptbr_filtered_v2.jsonl"),
    )
    redial.add_argument("--limit", type=int, default=400)
    redial.add_argument("--max-scan", type=int, default=8_500)
    redial.add_argument("--accept-terms", action="append", default=[])

    collect = subparsers.add_parser("collect-ultrachatbr")
    collect.add_argument(
        "--output",
        default=str(SFT_ROOT / "candidates" / "ultrachatbr_pt_filtered_v1.jsonl"),
    )
    collect.add_argument("--limit", type=int, default=250)
    collect.add_argument("--max-scan", type=int, default=12_000)
    collect.add_argument("--accept-terms", action="append", default=[])

    anchors = subparsers.add_parser("generate-anchors")
    anchors.add_argument(
        "--output",
        default=str(SFT_ROOT / "candidates" / "keilinks_authored_neutral_v1.jsonl"),
    )
    anchors.add_argument("--count", type=int, default=84)

    mix = subparsers.add_parser("build")
    mix.add_argument(
        "--output",
        default=str(SFT_ROOT / "all_sft_380m_conversation_8k_v2.jsonl"),
    )
    mix.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    mix.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "collect-redial-ptbr":
        report = collect_redial_ptbr(
            Path(args.output),
            limit=args.limit,
            max_scan=args.max_scan,
            accepted_terms=set(args.accept_terms),
        )
    elif args.command == "collect-ultrachatbr":
        report = collect_ultrachatbr(
            Path(args.output),
            limit=args.limit,
            max_scan=args.max_scan,
            accepted_terms=set(args.accept_terms),
        )
    elif args.command == "generate-anchors":
        report = generate_neutral_anchors(Path(args.output), count=args.count)
    else:
        paths = _default_paths()
        report = build_conversation_mix(
            Path(args.output),
            total=args.total,
            seed=args.seed,
            **paths,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
