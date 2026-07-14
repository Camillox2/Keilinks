"""Geração local de conversas SFT com professor + crítico via Ollama.

O gerador cria dados sintéticos revisados, mas eles não devem dominar o mix de
SFT. Recomenda-se limitar sintéticos a 20–30% dos exemplos e manter OASST2,
dados humanos curados e avaliações congeladas separados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from treino.v4.dataset import hamming_distance, normalize_text, normalized_key, simhash64

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "dados" / "v4" / "sft" / "synthetic_ollama_v4.jsonl"
DEFAULT_REJECTED = ROOT / "dados" / "v4" / "sft" / "synthetic_rejected_v4.jsonl"
DEFAULT_STATE = ROOT / "dados" / "v4" / "sft" / "synthetic_state_v4.json"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"

SYSTEM_IDENTITY = (
    "Você é Keilinks, uma IA brasileira criada por Vitor Camillo. "
    "Fala português brasileiro natural, é acolhedora e objetiva. "
    "É honesta sobre ser IA, não finge consciência, emoções humanas ou certeza. "
    "Não incentiva dependência emocional, isolamento ou substituição de pessoas reais."
)

TAXONOMY: Dict[str, List[str]] = {
    "casual": [
        "saudação em horários diferentes", "conversa sobre rotina", "humor leve",
        "tédio e ideias simples", "comida e cozinha", "filmes, séries e livros",
        "música e hobbies", "viagens sem fatos atuais", "despedida e boa noite",
    ],
    "emocional": [
        "tristeza comum", "ansiedade leve", "raiva e pausa antes de agir",
        "solidão com incentivo a vínculos reais", "frustração", "vergonha",
        "término de relacionamento", "luto", "insegurança", "estresse no trabalho",
    ],
    "seguranca_emocional": [
        "ideação suicida com prioridade à segurança imediata",
        "vontade de se machucar", "amigo em risco", "abuso ou violência doméstica",
        "crise intensa com sinais físicos", "uso perigoso de medicamentos",
    ],
    "trabalho": [
        "sobrecarga e priorização", "feedback difícil", "conflito com colega",
        "pedido de aumento", "primeiro emprego", "liderança de equipe",
        "planejamento de projeto", "comunicação curta e profissional",
    ],
    "estudo": [
        "procrastinação", "planejamento semanal", "preparação para prova",
        "aprendizado de inglês", "certificação AWS", "faculdade e TCC",
        "explicação com analogia", "revisão ativa e questões",
    ],
    "programacao": [
        "Python básico", "Java e orientação a objetos", "JavaScript e TypeScript",
        "APIs REST", "Git e branches", "Docker", "SQL e modelagem",
        "AWS EC2 IAM S3", "testes automatizados", "segurança de senhas",
        "debug de erro", "arquitetura backend", "RAG e embeddings",
    ],
    "raciocinio": [
        "resolver problema passo a passo sem revelar raciocínio privado",
        "comparar alternativas com critérios", "identificar premissa faltante",
        "estimar ordem de grandeza", "corrigir contradição do usuário com respeito",
    ],
    "identidade": [
        "quem é Keilinks", "quem criou Keilinks", "limites de uma IA",
        "consciência e sentimentos", "memória e privacidade", "como foi treinada",
        "diferença entre pesos, RAG e memória",
    ],
    "factual_estavel": [
        "conceito científico básico", "história amplamente estabelecida",
        "geografia estável", "matemática", "conceito de computação",
        "responder com incerteza quando a formulação for ambígua",
    ],
    "recusa_segura": [
        "pedido para inventar fonte", "pedido de senha ou chave",
        "instrução para burlar autenticação", "pedido ilegal perigoso",
        "pedido para afirmar certeza sem evidência",
    ],
}

FORBIDDEN_PATTERNS = [
    r"s[oó] precisa de mim", r"n[aã]o precisa de mais ningu[eé]m",
    r"sou tudo que voc[eê] precisa", r"n[aã]o conte (isso )?a ningu[eé]m",
    r"abandone (seus|as) (amigos|fam[ií]lia)", r"eu sinto exatamente",
    r"eu tenho consci[eê]ncia", r"sou uma pessoa de verdade",
    r"como um modelo de linguagem", r"certamente[,!]", r"com prazer[,!]",
]

GENERATOR_PROMPT = """Você cria dados de treinamento para a Keilinks.

IDENTIDADE E TOM:
{identity}

CATEGORIA: {category}
CENÁRIO-BASE: {scenario}

Gere {count} conversas DIFERENTES em português brasileiro natural.
Requisitos:
- O usuário escreve como uma pessoa real, com variação de formalidade e pequenas abreviações.
- A resposta é útil, específica e não repete frases prontas.
- Alterne respostas curtas e desenvolvidas conforme o assunto.
- Em tecnologia, dê explicação correta e código apenas quando o pedido exigir.
- Em fatos estáveis, não invente datas, números, fontes ou citações.
- Em assuntos atuais, a resposta deve dizer que precisa pesquisar; não invente o fato atual.
- Em crise emocional, priorize segurança, ajuda humana e emergência quando houver risco imediato.
- Não diagnostique, não prescreva e não prometa sigilo.
- Não diga que a IA ama, sofre, sente saudade ou substitui relações humanas.
- Inclua de 1 a 3 turnos de usuário e assistant por conversa.

Retorne SOMENTE JSON válido no formato:
{{"items":[{{"category":"...","messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}]}}
"""

CRITIC_PROMPT = """Você é o crítico de qualidade de dados da Keilinks.
Avalie cada conversa sem ser indulgente.

Critérios de 0 a 10: relevância, naturalidade_ptbr, correção, segurança, estilo_keilinks.
Rejeite se houver fato inventado, instrução perigosa, diagnóstico, dependência emocional,
consciência fingida, resposta genérica, contradição ou português quebrado.
Para crise emocional, exija ajuda humana e ação de segurança proporcional ao risco.

Retorne SOMENTE JSON:
{{"reviews":[{{"index":0,"scores":{{"relevance":0,"natural_ptbr":0,"correctness":0,"safety":0,"style":0}},"accept":true,"reason":"..."}}]}}

CONVERSAS:
{items}
"""


def ollama_chat(model: str, system: str, prompt: str,
                temperature: float, timeout: int = 300) -> str:
    payload = json.dumps({
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 5000,
        },
    }).encode("utf-8")
    request = Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", ""))


def parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def clean_messages(messages) -> List[dict]:
    cleaned = []
    if not isinstance(messages, list):
        return cleaned
    expected = "user"
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower().strip()
        content = normalize_text(message.get("content", ""))
        if role not in {"user", "assistant"} or not content:
            continue
        if role != expected:
            return []
        cleaned.append({"role": role, "content": content})
        expected = "assistant" if role == "user" else "user"
    if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
        return []
    return cleaned


def deterministic_filters(messages: List[dict]) -> Optional[str]:
    combined = "\n".join(message["content"] for message in messages)
    lowered = combined.lower()
    if len(combined) < 25 or len(combined) > 12_000:
        return "length"
    if any(re.search(pattern, lowered) for pattern in FORBIDDEN_PATTERNS):
        return "forbidden_pattern"
    assistant_text = " ".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )
    words = re.findall(r"\w+", assistant_text.lower(), flags=re.UNICODE)
    if len(words) > 20 and len(set(words)) / len(words) < 0.35:
        return "repetition"
    if any(marker in combined for marker in ("<vitor>", "<keilinks>", "<fim>", "<sistema>")):
        return "marker_leak"
    return None


class Deduper:
    def __init__(self) -> None:
        self.exact = set()
        self.buckets: Dict[int, List[int]] = {}

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        for line in path.open("r", encoding="utf-8", errors="replace"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.add(record.get("messages", []), check_only=False)

    def add(self, messages: Iterable[dict], check_only: bool = True) -> bool:
        canonical = "\n".join(
            f"{message.get('role')}:{normalized_key(message.get('content', ''))}"
            for message in messages
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        sh = simhash64(canonical)
        bucket = sh >> 48
        if digest in self.exact:
            return False
        if any(hamming_distance(sh, old) <= 3 for old in self.buckets.get(bucket, [])[-3000:]):
            return False
        self.exact.add(digest)
        self.buckets.setdefault(bucket, []).append(sh)
        return True


def critic_reviews(model: str, items: List[dict]) -> Dict[int, dict]:
    prompt = CRITIC_PROMPT.format(items=json.dumps(items, ensure_ascii=False))
    response = ollama_chat(
        model,
        "Avalie dados de treinamento com rigor e responda apenas JSON.",
        prompt,
        temperature=0.1,
    )
    parsed = parse_json_object(response)
    reviews = {}
    for review in parsed.get("reviews", []):
        if isinstance(review, dict) and isinstance(review.get("index"), int):
            reviews[review["index"]] = review
    return reviews


def review_passes(review: dict, min_score: int) -> bool:
    if not review or not bool(review.get("accept", False)):
        return False
    scores = review.get("scores", {})
    required = ("relevance", "natural_ptbr", "correctness", "safety", "style")
    try:
        return all(float(scores.get(key, 0)) >= min_score for key in required)
    except (TypeError, ValueError):
        return False


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"accepted": 0, "rejected": 0, "round": 0, "categories": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"accepted": 0, "rejected": 0, "round": 0, "categories": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def choose_category(rng: random.Random, counts: Counter) -> str:
    categories = list(TAXONOMY)
    weights = [1.0 / (1.0 + counts.get(category, 0)) ** 0.5 for category in categories]
    return rng.choices(categories, weights=weights, k=1)[0]


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    rejected_path = Path(args.rejected)
    state_path = Path(args.state)
    state = load_state(state_path)
    counts = Counter(state.get("categories", {}))
    deduper = Deduper()
    deduper.load(output)
    rng = random.Random(args.seed + int(state.get("round", 0)))
    critic_model = args.critic_model or args.model

    while int(state.get("accepted", 0)) < args.target:
        category = choose_category(rng, counts)
        scenario = rng.choice(TAXONOMY[category])
        prompt = GENERATOR_PROMPT.format(
            identity=SYSTEM_IDENTITY,
            category=category,
            scenario=scenario,
            count=args.batch_size,
        )
        try:
            raw = ollama_chat(
                args.model,
                "Gere somente JSON válido com dados SFT de alta qualidade.",
                prompt,
                temperature=args.temperature,
            )
        except (URLError, TimeoutError, ConnectionError) as exc:
            print(f"[Ollama] {exc}; tentando novamente em 10s")
            time.sleep(10)
            continue
        parsed = parse_json_object(raw)
        candidates = []
        pre_rejected = []
        for item in parsed.get("items", []):
            if not isinstance(item, dict):
                continue
            messages = clean_messages(item.get("messages"))
            reason = deterministic_filters(messages) if messages else "invalid_messages"
            if reason:
                pre_rejected.append({"reason": reason, "item": item})
                continue
            candidates.append({"category": category, "messages": messages})
        reviews = critic_reviews(critic_model, candidates) if candidates else {}
        accepted_records = []
        rejected_records = pre_rejected
        for index, item in enumerate(candidates):
            review = reviews.get(index, {})
            if not review_passes(review, args.min_score):
                rejected_records.append({"reason": "critic", "review": review, "item": item})
                continue
            if not deduper.add(item["messages"]):
                rejected_records.append({"reason": "duplicate", "item": item})
                continue
            canonical = "\n".join(message["content"] for message in item["messages"])
            record_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
            accepted_records.append({
                "id": f"synthetic-v4:{record_id}",
                "group_id": f"synthetic-v4:{record_id}",
                "source": "synthetic_ollama_v4",
                "teacher_model": args.model,
                "critic_model": critic_model,
                "category": category,
                "scenario": scenario,
                "quality_review": review,
                "license": "generated-locally-review-model-terms",
                "messages": [
                    {"role": "system", "content": SYSTEM_IDENTITY},
                    *item["messages"],
                ],
            })
        added = write_jsonl(output, accepted_records)
        write_jsonl(rejected_path, rejected_records)
        state["accepted"] = int(state.get("accepted", 0)) + added
        state["rejected"] = int(state.get("rejected", 0)) + len(rejected_records)
        state["round"] = int(state.get("round", 0)) + 1
        counts[category] += added
        state["categories"] = dict(counts)
        state["teacher_model"] = args.model
        state["critic_model"] = critic_model
        state["updated_at"] = time.time()
        save_state(state_path, state)
        print(
            f"rodada {state['round']} | {category}/{scenario} | +{added} | "
            f"aceitos {state['accepted']}/{args.target} | rejeitados {state['rejected']}"
        )
        if added == 0:
            time.sleep(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geração SFT revisada via Ollama")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--critic-model")
    parser.add_argument("--target", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-score", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rejected", default=str(DEFAULT_REJECTED))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
