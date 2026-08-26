"""Servidor recomendado da Keilinks V4.

Quando um checkpoint V4 está disponível, ele substitui a rota principal de chat
sem carregar os três modelos legados na VRAM. Sem checkpoint V4, o servidor
continua funcionando com o comportamento legado.
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from flask import Response, jsonify, request, stream_with_context

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from api import servidor as legacy
from api.runtime_v4 import V4Runtime
from busca.web_v4 import pesquisar as pesquisar_v4, precisa_buscar as precisa_buscar_v4
from cerebro.raciocinio import normalize_reasoning_mode

legacy.pesquisar = pesquisar_v4
legacy.precisa_buscar = precisa_buscar_v4
legacy.RETREINAR_A_CADA = 2**63 - 1

ADMIN_TOKEN = os.getenv("KEILINKS_ADMIN_TOKEN", "").strip()
WRITE_ENDPOINTS = {"/api/ensinar", "/api/crawl"}
CANDIDATES_PATH = BASE_DIR / "dados" / "v4" / "candidates" / "runtime_feedback.jsonl"
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "v4-sft" / "keilinks_v4.pt"
DEFAULT_VOCAB = BASE_DIR / "dados" / "v4" / "pretrain" / "tokenizer.json"

runtime: Optional[V4Runtime] = None
_original_chat = legacy.app.view_functions.get("chat")
_original_stream = legacy.app.view_functions.get("chat_stream")
_original_status = legacy.app.view_functions.get("status")


def _admin_token() -> str:
    value = request.headers.get("X-Keilinks-Admin", "").strip()
    if value:
        return value
    authorization = request.headers.get("Authorization", "")
    return authorization[7:].strip() if authorization.startswith("Bearer ") else ""


@legacy.app.before_request
def protect_write_endpoints():
    if request.path not in WRITE_ENDPOINTS:
        return None
    if not ADMIN_TOKEN and request.remote_addr in {"127.0.0.1", "::1", None}:
        return None
    if not ADMIN_TOKEN:
        return jsonify({"erro": "Defina KEILINKS_ADMIN_TOKEN para escrita remota."}), 503
    if not hmac.compare_digest(_admin_token(), ADMIN_TOKEN):
        return jsonify({"erro": "Token administrativo inválido."}), 403
    return None


def save_candidate(question: str, answer: str, source: str = "runtime") -> None:
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "source": source,
        "status": "pending_review",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }
    with CANDIDATES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_save_conversation(question: str, answer: str) -> None:
    save_candidate(question, answer, "legacy-approved-source")


legacy.salvar_conversa_txt = safe_save_conversation


def _initialize_data_systems() -> None:
    legacy.inicializar_banco()
    try:
        if legacy.knowledge_total() == 0:
            legacy.migrar_json_para_mysql(str(BASE_DIR))
    except Exception as exc:
        print(f"[V4 migração] {exc}")
    legacy.retrieval.carregar(
        str(BASE_DIR / "dados" / "conversas.txt"),
        str(BASE_DIR / "dados" / "aprendizado.txt"),
    )
    try:
        legacy.knowledge.iniciar_embeddings()
    except Exception as exc:
        print(f"[V4 embeddings] {exc}")


def initialize() -> None:
    global runtime
    checkpoint = Path(os.getenv("KEILINKS_V4_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    vocab = Path(os.getenv("KEILINKS_V4_VOCAB", str(DEFAULT_VOCAB)))
    if checkpoint.exists() and vocab.exists():
        _initialize_data_systems()
        runtime = V4Runtime(checkpoint, vocab)
        print(f"[Keilinks V4] checkpoint: {checkpoint}")
        print(f"[Keilinks V4] parâmetros: {runtime.model.parameter_count()/1e6:.1f}M")
        print(f"[Keilinks V4] device: {runtime.device}")
    else:
        print("[Keilinks V4] checkpoint/vocab ainda não disponível; usando legado.")
        legacy.inicializar()


def _authenticated_user(payload: dict):
    token = payload.get("token")
    if not token:
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            token = authorization[7:]
    return legacy.usuario_por_token(token) if token else None


def _history(chat_id, user_id) -> list[tuple[str, str]]:
    if not chat_id or not user_id:
        return []
    try:
        messages = legacy.chat_mensagens(chat_id, user_id) or []
    except Exception:
        return []
    result = []
    for item in messages[-6:]:
        question = item.get("pergunta", "")
        answer = item.get("resposta", "")
        if question and answer:
            result.append((question, answer))
    return result


def _semantic_context(message: str) -> tuple[str, float]:
    pieces = []
    score = 0.0
    try:
        retrieved, score = legacy.retrieval.buscar(message)
        if retrieved and score >= 0.20:
            pieces.append(retrieved)
    except Exception:
        pass
    try:
        knowledge = legacy.knowledge.buscar(message)
        if knowledge:
            pieces.append(knowledge)
    except Exception:
        pass
    return "\n".join(pieces)[:5000], score


def _reasoning_mode(payload: dict) -> str:
    """Aceita o nome em português e o nome estável do contrato da API."""

    return normalize_reasoning_mode(
        payload.get("reasoning_mode", payload.get("raciocinio", "auto"))
    )


def chat_v4():
    if runtime is None:
        if _original_chat is None:
            return jsonify({"erro": "Nenhum runtime disponível"}), 503
        return _original_chat()

    payload = request.get_json(force=True, silent=True) or {}
    message = str(payload.get("mensagem", "")).strip()
    if not message:
        return jsonify({"erro": "Mensagem vazia"}), 400
    user = _authenticated_user(payload)
    user_id = user["id"] if user else None
    chat_id = payload.get("chat_id")
    history = _history(chat_id, user_id)
    try:
        memory_context = legacy.memoria.gerar_contexto(user_id=user_id)
    except Exception:
        memory_context = ""
    semantic_context, semantic_score = _semantic_context(message)

    try:
        answer = runtime.answer(
            message,
            history=history,
            memory_context=memory_context,
            semantic_context=semantic_context,
            web_enabled=bool(payload.get("web_enabled", True)),
            web_mode=str(payload.get("web_mode", "auto")),
            reasoning_mode=_reasoning_mode(payload),
            max_new_tokens=min(int(payload.get("max_tokens", 256)), 512),
            temperature=float(payload.get("temperatura", 0.75)),
            top_p=float(payload.get("top_p", 0.9)),
        )
    except Exception as exc:
        print(f"[Keilinks V4 chat] {exc}")
        return jsonify({"erro": "Falha ao gerar resposta", "detalhe": str(exc)}), 500

    source_name = "modelo_v4_web" if answer.used_web else "modelo_v4"
    try:
        legacy.memoria.atualizar(message, answer.text, user_id=user_id)
    except Exception:
        pass
    try:
        legacy.conversa_salvar(
            message, answer.text, source_name,
            chat_id=chat_id, usuario_id=user_id,
        )
    except Exception as exc:
        print(f"[V4 salvar conversa] {exc}")
    if chat_id and user:
        try:
            messages = legacy.chat_mensagens(chat_id, user_id)
            if messages and len(messages) == 1:
                legacy.chat_atualizar_titulo(chat_id, message[:80])
        except Exception:
            pass

    return jsonify({
        "resposta": answer.text,
        "modelo": "keilinks-v4",
        "fonte": source_name,
        "usou_web": answer.used_web,
        "fontes": answer.sources,
        "confianca": 88 if answer.used_web else (75 if semantic_score >= 0.5 else 65),
        "prompt_tokens": answer.prompt_tokens,
        "generated_tokens": answer.generated_tokens,
        "raciocinio": answer.reasoning_mode,
        "usou_raciocinio": answer.used_reasoning,
        "pensamento": [
            "Runtime V4",
            f"Histórico: {len(history)} turnos",
            f"RAG score: {semantic_score:.2f}",
            f"Fontes web: {len(answer.sources)}",
            f"Raciocínio: {answer.reasoning_mode}",
        ],
    })


def chat_stream_v4():
    if runtime is None:
        if _original_stream is None:
            return jsonify({"erro": "Streaming indisponível"}), 503
        return _original_stream()

    payload = request.get_json(force=True, silent=True) or {}
    message = str(payload.get("mensagem", "")).strip()
    if not message:
        return jsonify({"erro": "Mensagem vazia"}), 400

    def generate_event():
        try:
            answer = runtime.answer(
                message,
                web_enabled=bool(payload.get("web_enabled", True)),
                web_mode=str(payload.get("web_mode", "auto")),
                reasoning_mode=_reasoning_mode(payload),
                max_new_tokens=min(int(payload.get("max_tokens", 256)), 512),
                temperature=float(payload.get("temperatura", 0.75)),
            )
            yield "data: " + json.dumps({
                "token": answer.text,
                "done": True,
                "fontes": answer.sources,
                "raciocinio": answer.reasoning_mode,
            }, ensure_ascii=False) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({
                "done": True,
                "erro": str(exc),
            }, ensure_ascii=False) + "\n\n"

    return Response(
        stream_with_context(generate_event()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def status_v4():
    if runtime is None:
        return _original_status() if _original_status else jsonify({"online": False})
    return jsonify({
        "online": True,
        "runtime": "v4",
        "device": str(runtime.device),
        "gpu": (
            __import__("torch").cuda.get_device_name(0)
            if __import__("torch").cuda.is_available() else "CPU"
        ),
        "model": runtime.config.name,
        "parameters": runtime.model.parameter_count(),
        "context_length": runtime.config.context_length,
        "knowledge": legacy.knowledge_total(),
        "retrieval": len(legacy.retrieval.pares),
        "auto_training": False,
    })


legacy.app.view_functions["chat"] = chat_v4
legacy.app.view_functions["chat_stream"] = chat_stream_v4
legacy.app.view_functions["status"] = status_v4
app = legacy.app


if __name__ == "__main__":
    initialize()
    app.run(
        host=os.getenv("KEILINKS_HOST", "0.0.0.0"),
        port=int(os.getenv("KEILINKS_PORT", "5000")),
        debug=False,
        threaded=True,
    )
