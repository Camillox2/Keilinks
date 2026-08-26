"""API FastAPI local, autenticável e com SSE de tokens reais."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .data import redact_sensitive_text
from .feedback import RecentInteractionCache
from .rag import LocalKnowledgeStore
from .runtime import ChatMessage, UnslothRuntime
from .security import SlidingWindowRateLimiter, api_key_matches
from .settings import KeilinksSettings
from .vision import VisionService, VisionUnavailable, VisualEvidence


class MessageInput(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[MessageInput] = Field(min_length=1, max_length=32)
    image: str | None = Field(default=None, max_length=14_000_000)
    use_rag: bool = True
    stream: bool = False
    max_new_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, messages: list[MessageInput]) -> list[MessageInput]:
        if messages[-1].role != "user":
            raise ValueError("a última mensagem deve ter role user")
        return messages


class DocumentInput(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=2_000_000)
    uri: str = Field(default="", max_length=2048)
    trust_tier: Literal["official", "curated", "user_provided"] = "user_provided"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackInput(BaseModel):
    interaction_id: str = Field(min_length=1, max_length=120)
    rating: Literal["up", "down"]
    correction: str = Field(default="", max_length=20_000)
    reason: str = Field(default="", max_length=2_000)
    consent_to_training: Literal[True]


class VisionRequest(BaseModel):
    image: str = Field(min_length=32, max_length=14_000_000)
    question: str = Field(min_length=1, max_length=4_000)


def _append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def create_app(settings: KeilinksSettings | None = None) -> FastAPI:
    settings = settings or KeilinksSettings.from_env()
    settings.validate()
    store = LocalKnowledgeStore(
        settings.rag_db,
        dense_enabled=settings.rag_dense_enabled,
        embedding_model=settings.rag_embedding_model,
    )
    runtime = UnslothRuntime(settings, store)
    vision = VisionService(
        settings.enable_vision,
        settings.vision_model,
        settings.vision_max_new_tokens,
    )
    limiter = SlidingWindowRateLimiter(requests_per_minute=30)
    feedback_lock = threading.Lock()
    recent_interactions = RecentInteractionCache()
    gpu_lock = threading.RLock()

    app = FastAPI(
        title="Keilinks V5",
        version="5.0.0a1",
        description="Assistente local em PT-BR com QLoRA, RAG e feedback opt-in.",
    )
    app.state.settings = settings
    app.state.store = store
    app.state.runtime = runtime
    app.state.vision = vision
    app.state.recent_interactions = recent_interactions

    def authorize(
        request: Request,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> str:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit")
        if not api_key_matches(settings.api_key, x_api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key inválida")
        return client

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "network_scope": "loopback" if settings.is_loopback else "authenticated_network",
            "runtime": runtime.status(),
            "rag": store.status(),
            "vision_enabled": settings.enable_vision,
        }

    @app.get("/v1/models")
    def models(_: str = Depends(authorize)) -> dict[str, object]:
        return {"data": [{"id": runtime.model_id, "owned_by": "keilinks"}]}

    @app.post("/v1/chat/completions")
    def chat(request_data: ChatRequest, _: str = Depends(authorize)):
        messages = [
            ChatMessage(role=message.role, content=message.content)
            for message in request_data.messages
        ]
        interaction_id = f"chatcmpl_{uuid.uuid4().hex}"
        visual_evidence = None

        def resolve_visual_context() -> tuple[VisualEvidence | None, str | None]:
            if not request_data.image:
                return None, None
            # O LLM pode estar residente após uma conversa anterior. Liberamos
            # a GPU antes de carregar o VLM e depois o runtime textual recarrega.
            runtime.unload()
            evidence = vision.inspect(request_data.image, messages[-1].content)
            return evidence, evidence.as_untrusted_context()

        try:
            if request_data.stream:

                def events():
                    output_parts: list[str] = []
                    try:
                        with gpu_lock:
                            evidence, visual_context = resolve_visual_context()
                            iterator, sources, model_id = runtime.stream(
                                messages,
                                use_rag=request_data.use_rag,
                                extra_system_context=visual_context,
                                max_new_tokens=request_data.max_new_tokens,
                                temperature=request_data.temperature,
                                top_p=request_data.top_p,
                            )
                            first_chunk: dict[str, object] = {
                                "id": interaction_id,
                                "object": "chat.completion.chunk",
                                "model": model_id,
                                "sources": sources,
                            }
                            if evidence is not None:
                                first_chunk["visual_evidence"] = evidence.to_dict()
                            yield runtime.sse_event(first_chunk)
                            for token in iterator:
                                if not token:
                                    continue
                                output_parts.append(token)
                                yield runtime.sse_event(
                                    {
                                        "id": interaction_id,
                                        "object": "chat.completion.chunk",
                                        "choices": [{"delta": {"content": token}, "index": 0}],
                                    }
                                )
                        if output_parts:
                            recent_interactions.put(
                                interaction_id,
                                prompt=messages[-1].content,
                                response="".join(output_parts),
                            )
                        yield runtime.sse_event(
                            {
                                "id": interaction_id,
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "delta": {},
                                        "index": 0,
                                        "finish_reason": "stop",
                                    }
                                ],
                            }
                        )
                    except (VisionUnavailable, RuntimeError, ValueError) as exc:
                        yield runtime.sse_event(
                            {
                                "id": interaction_id,
                                "object": "error",
                                "error": str(exc),
                            }
                        )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    events(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            with gpu_lock:
                visual_evidence, visual_context = resolve_visual_context()
                answer = runtime.answer(
                    messages,
                    use_rag=request_data.use_rag,
                    extra_system_context=visual_context,
                    max_new_tokens=request_data.max_new_tokens,
                    temperature=request_data.temperature,
                    top_p=request_data.top_p,
                )
                recent_interactions.put(
                    interaction_id,
                    prompt=messages[-1].content,
                    response=answer.text,
                )
        except VisionUnavailable as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        return {
            "id": interaction_id,
            "object": "chat.completion",
            "created": int(datetime.now(UTC).timestamp()),
            "model": answer.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.generated_tokens,
                "total_tokens": answer.prompt_tokens + answer.generated_tokens,
            },
            "sources": answer.sources,
            "elapsed_ms": round(answer.elapsed_ms, 1),
            "visual_evidence": visual_evidence.to_dict() if visual_evidence else None,
        }

    @app.post("/v1/documents")
    def add_document(document: DocumentInput, _: str = Depends(authorize)) -> dict[str, object]:
        source_id = store.add_document(
            title=document.title,
            content=document.content,
            uri=document.uri,
            trust_tier=document.trust_tier,
            metadata=document.metadata,
        )
        return {"source_id": source_id, "status": "indexed"}

    @app.post("/v1/feedback")
    def feedback(data: FeedbackInput, _: str = Depends(authorize)) -> dict[str, object]:
        interaction = recent_interactions.consume(data.interaction_id)
        payload = {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "interaction_id": data.interaction_id,
            "rating": data.rating,
            "correction": redact_sensitive_text(data.correction),
            "reason": redact_sensitive_text(data.reason),
            "consent_to_training": True,
            "status": "pending_human_review",
            "policy": "offline_only; never auto-train from this queue",
        }
        if interaction is not None:
            payload["prompt"] = redact_sensitive_text(interaction.prompt)
            payload["response"] = redact_sensitive_text(interaction.response)
        else:
            payload["source_context"] = (
                "não encontrado na janela efêmera; não utilizável para treino"
            )
        _append_jsonl(settings.feedback_path, payload, feedback_lock)
        return {"status": "recorded_for_review", "interaction_id": data.interaction_id}

    @app.post("/v1/vision/analyze")
    def analyze_vision(data: VisionRequest, _: str = Depends(authorize)) -> dict[str, object]:
        try:
            with gpu_lock:
                runtime.unload()
                return vision.inspect(data.image, data.question).to_dict()
        except (VisionUnavailable, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.exception_handler(ValueError)
    async def invalid_value_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


# Mantém compatibilidade com `uvicorn keilinks_v5.server:app`, sem carregar o
# modelo até a primeira geração.
app = create_app()
