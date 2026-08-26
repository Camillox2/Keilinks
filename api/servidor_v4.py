"""Servidor recomendado da Keilinks V4.

Quando um checkpoint V4 está disponível, ele atende a rota principal de chat
sem carregar modelos alternativos na VRAM. Sem checkpoint V4, a API informa
claramente que aguarda a própria Keilinks em vez de trocar de modelo.
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import hmac
import json
import os
import sys
import time
from pathlib import Path

from flask import Response, jsonify, request, stream_with_context

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from api import servidor as legacy
from api.runtime_v4 import V4Runtime
from busca.web_v4 import pesquisar as pesquisar_v4
from busca.web_v4 import precisa_buscar as precisa_buscar_v4
from cerebro.raciocinio import normalize_reasoning_mode
from dados.database import (
    conversa_historico_usuario,
    memoria_usuario_atualizar,
    memoria_usuario_config,
    memoria_usuario_config_atualizar,
    memoria_usuario_contexto,
    memoria_usuario_criar,
    memoria_usuario_excluir,
    memorias_usuario_listar,
    usuario_atualizar_nome,
)

legacy.pesquisar = pesquisar_v4
legacy.precisa_buscar = precisa_buscar_v4
legacy.RETREINAR_A_CADA = 2**63 - 1

ADMIN_TOKEN = os.getenv("KEILINKS_ADMIN_TOKEN", "").strip()
WRITE_ENDPOINTS = {"/api/ensinar", "/api/crawl"}
CANDIDATES_PATH = BASE_DIR / "dados" / "v4" / "candidates" / "runtime_feedback.jsonl"
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "v4-sft" / "keilinks_v4.pt"
DEFAULT_VOCAB = BASE_DIR / "dados" / "v4" / "pretrain" / "tokenizer.json"

runtime: V4Runtime | None = None


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
            legacy.migrar_json_para_sqlite(str(BASE_DIR))
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
        print(
            "[Keilinks V4] checkpoint/vocab ainda não disponível; "
            "o chat ficará aguardando o checkpoint da própria Keilinks."
        )


def _authenticated_user(payload: dict):
    token = payload.get("token")
    if not token:
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            token = authorization[7:]
    return legacy.usuario_por_token(token) if token else None


def _history(chat_id, user_id, history_enabled: bool) -> list[tuple[str, str]]:
    if not chat_id or not user_id:
        return []
    try:
        messages = legacy.chat_mensagens(chat_id, user_id)
    except Exception as exc:
        raise ValueError("não foi possível validar este chat") from exc
    if messages is None:
        raise ValueError("chat não encontrado ou não pertence a esta conta")
    if not history_enabled:
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


def _show_reasoning(payload: dict) -> bool:
    """Exibe somente o plano curto completo quando o cliente pedir."""

    value = payload.get("show_reasoning", payload.get("mostrar_raciocinio", False))
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "sim", "on"}


def _request_context(payload: dict, message: str) -> dict:
    user = _authenticated_user(payload)
    user_id = user["id"] if user else None
    raw_chat_id = payload.get("chat_id")
    if raw_chat_id in {None, ""}:
        chat_id = None
    else:
        try:
            chat_id = int(raw_chat_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("chat_id inválido") from exc
        if chat_id <= 0:
            raise ValueError("chat_id inválido")
    if chat_id is not None and user_id is None:
        raise ValueError("é necessário autenticar para usar um chat salvo")

    settings = (
        memoria_usuario_config(user_id)
        if user_id is not None
        else {"memory_enabled": False, "history_enabled": False, "training_consent": False}
    )
    history = _history(chat_id, user_id, settings["history_enabled"])
    memory_context = (
        memoria_usuario_contexto(user_id, message)
        if user_id is not None
        else ""
    )
    semantic_context, semantic_score = _semantic_context(message)
    return {
        "user": user,
        "user_id": user_id,
        "chat_id": chat_id,
        "history": history,
        "memory_context": memory_context,
        "memory_settings": settings,
        "semantic_context": semantic_context,
        "semantic_score": semantic_score,
    }


def _persist_answer(message: str, answer, context: dict) -> str:
    """Persiste só a resposta final; plano de raciocínio nunca vai ao histórico."""

    source_name = "modelo_v4_web" if answer.used_web else "modelo_v4"
    user = context["user"]
    user_id = context["user_id"]
    chat_id = context["chat_id"]
    # Não extraímos palavras soltas nem inferimos dados pessoais de uma conversa.
    # A memória de longo prazo é criada apenas pelos endpoints explícitos abaixo.
    if user_id is None or not context["memory_settings"]["history_enabled"] or not chat_id:
        return source_name
    try:
        legacy.conversa_salvar(
            message, answer.text, source_name, chat_id=chat_id, usuario_id=user_id
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
    return source_name


def chat_v4():
    if runtime is None:
        return jsonify({
            "erro": (
                "O checkpoint conversacional da Keilinks ainda não está disponível. "
                "Nenhum modelo alternativo será usado."
            )
        }), 503

    payload = request.get_json(force=True, silent=True) or {}
    message = str(payload.get("mensagem", "")).strip()
    if not message:
        return jsonify({"erro": "Mensagem vazia"}), 400
    try:
        context = _request_context(payload, message)
    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400
    show_reasoning = _show_reasoning(payload)

    try:
        answer = runtime.answer(
            message,
            history=context["history"],
            memory_context=context["memory_context"],
            semantic_context=context["semantic_context"],
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

    source_name = _persist_answer(message, answer, context)
    plan = answer.reasoning_summary if show_reasoning else ""

    return jsonify({
        "resposta": answer.text,
        "modelo": "keilinks-v4",
        "fonte": source_name,
        "usou_web": answer.used_web,
        "fontes": answer.sources,
        "confianca": 88 if answer.used_web else (
            75 if context["semantic_score"] >= 0.5 else 65
        ),
        "prompt_tokens": answer.prompt_tokens,
        "generated_tokens": answer.generated_tokens,
        "raciocinio": answer.reasoning_mode,
        "usou_raciocinio": answer.used_reasoning,
        "resumo_raciocinio": plan or None,
        # Compatibilidade: pensamento agora é o plano realmente gerado, e não
        # uma lista de telemetria que poderia ser confundida com raciocínio.
        "pensamento": [plan] if plan else [],
        "telemetria": [
            "Runtime V4",
            f"Histórico: {len(context['history'])} turnos",
            f"RAG score: {context['semantic_score']:.2f}",
            f"Fontes web: {len(answer.sources)}",
            f"Raciocínio: {answer.reasoning_mode}",
        ],
    })


def chat_stream_v4():
    if runtime is None:
        return jsonify({
            "erro": (
                "O checkpoint conversacional da Keilinks ainda não está disponível. "
                "Nenhum modelo alternativo será usado."
            )
        }), 503

    payload = request.get_json(force=True, silent=True) or {}
    message = str(payload.get("mensagem", "")).strip()
    if not message:
        return jsonify({"erro": "Mensagem vazia"}), 400
    try:
        context = _request_context(payload, message)
    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400
    show_reasoning = _show_reasoning(payload)

    def generate_event():
        try:
            answer = runtime.answer(
                message,
                history=context["history"],
                memory_context=context["memory_context"],
                semantic_context=context["semantic_context"],
                web_enabled=bool(payload.get("web_enabled", True)),
                web_mode=str(payload.get("web_mode", "auto")),
                reasoning_mode=_reasoning_mode(payload),
                max_new_tokens=min(int(payload.get("max_tokens", 256)), 512),
                temperature=float(payload.get("temperatura", 0.75)),
                top_p=float(payload.get("top_p", 0.9)),
            )
            source_name = _persist_answer(message, answer, context)
            plan = answer.reasoning_summary if show_reasoning else ""
            yield "data: " + json.dumps({
                "token": answer.text,
                "done": True,
                "fontes": answer.sources,
                "fonte": source_name,
                "raciocinio": answer.reasoning_mode,
                "usou_raciocinio": answer.used_reasoning,
                "resumo_raciocinio": plan or None,
                "pensamento": [plan] if plan else [],
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
        return jsonify({
            "online": False,
            "runtime": "keilinks-v4-awaiting-checkpoint",
            "model": "Keilinks Core 380M",
            "auto_training": False,
            "memory": "user_scoped_explicit",
        })
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
        "memory": "user_scoped_explicit",
    })


def _memory_user():
    user = _authenticated_user({})
    if user is None:
        return None
    return user


def _memory_payload(user: dict) -> dict:
    return {
        "profile": {
            "id": user["id"],
            "username": user["username"],
            "nome": user.get("nome") or user["username"],
        },
        "settings": memoria_usuario_config(user["id"]),
        "memories": memorias_usuario_listar(user["id"]),
        "policy": {
            "automatic_extraction": False,
            "training": "opt_in_only; reviewed_offline_only",
            "description": (
                "A Keilinks usa somente memórias salvas ou confirmadas por você. "
                "Conversas não alteram os pesos automaticamente."
            ),
        },
    }


@legacy.app.route("/api/me/memory", methods=["GET"])
def memoria_pessoal_obter():
    user = _memory_user()
    if user is None:
        return jsonify({"erro": "Não autenticado"}), 401
    return jsonify(_memory_payload(user))


@legacy.app.route("/api/me/memory/settings", methods=["PUT"])
def memoria_pessoal_configurar():
    user = _memory_user()
    if user is None:
        return jsonify({"erro": "Não autenticado"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    fields = ("memory_enabled", "history_enabled", "training_consent")
    changes = {field: payload[field] for field in fields if field in payload}
    if not changes:
        return jsonify({"erro": "Nenhuma configuração de memória foi informada"}), 400
    try:
        settings = memoria_usuario_config_atualizar(user["id"], **changes)
    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400
    return jsonify({"settings": settings})


@legacy.app.route("/api/me/profile", methods=["PUT"])
def perfil_pessoal_atualizar():
    user = _memory_user()
    if user is None:
        return jsonify({"erro": "Não autenticado"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    try:
        profile = usuario_atualizar_nome(user["id"], payload.get("nome"))
    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400
    if profile is None:
        return jsonify({"erro": "Perfil não encontrado"}), 404
    return jsonify({"profile": profile})


@legacy.app.route("/api/me/memories", methods=["POST"])
def memoria_pessoal_criar():
    user = _memory_user()
    if user is None:
        return jsonify({"erro": "Não autenticado"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    try:
        memory = memoria_usuario_criar(
            user["id"], payload.get("content"), category=payload.get("category", "note")
        )
    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400
    return jsonify({"memory": memory}), 201


@legacy.app.route("/api/me/memories/<int:memory_id>", methods=["PATCH"])
def memoria_pessoal_atualizar(memory_id: int):
    user = _memory_user()
    if user is None:
        return jsonify({"erro": "Não autenticado"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    try:
        memory = memoria_usuario_atualizar(
            user["id"],
            memory_id,
            content=payload.get("content") if "content" in payload else None,
            category=payload.get("category") if "category" in payload else None,
        )
    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400
    if memory is None:
        return jsonify({"erro": "Memória não encontrada"}), 404
    return jsonify({"memory": memory})


@legacy.app.route("/api/me/memories/<int:memory_id>", methods=["DELETE"])
def memoria_pessoal_excluir(memory_id: int):
    user = _memory_user()
    if user is None:
        return jsonify({"erro": "Não autenticado"}), 401
    if not memoria_usuario_excluir(user["id"], memory_id):
        return jsonify({"erro": "Memória não encontrada"}), 404
    return jsonify({"ok": True})


def historico_pessoal_protegido():
    """Substitui a rota legada que expunha o histórico global sem autenticação."""

    user = _memory_user()
    if user is None:
        return jsonify({"erro": "Não autenticado"}), 401
    settings = memoria_usuario_config(user["id"])
    if not settings["history_enabled"]:
        return jsonify([])
    return jsonify(conversa_historico_usuario(user["id"], 50))


# Mantém a URL legada, mas elimina a exposição da memória global a visitantes.
legacy.app.view_functions["historico"] = historico_pessoal_protegido
legacy.app.view_functions["ver_memoria"] = memoria_pessoal_obter


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
