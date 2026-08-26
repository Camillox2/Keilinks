"""Memória efêmera e consentimento explícito para feedback de treinamento.

Não persistimos conversas automaticamente. A resposta fica apenas numa janela
curta em memória para que, se a pessoa der consentimento explícito no endpoint
de feedback, seja possível registrar o par necessário para revisão humana.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class RecentInteraction:
    prompt: str
    response: str
    created_at: float


class RecentInteractionCache:
    """Cache limitado, com TTL e consumo único para reduzir retenção de chats."""

    def __init__(self, *, max_entries: int = 256, ttl_seconds: int = 3600) -> None:
        if max_entries < 1 or ttl_seconds < 1:
            raise ValueError("max_entries e ttl_seconds devem ser positivos")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, RecentInteraction] = OrderedDict()
        self._lock = threading.RLock()

    def _expire_locked(self, now: float) -> None:
        expired = [
            key for key, value in self._items.items() if now - value.created_at > self.ttl_seconds
        ]
        for key in expired:
            self._items.pop(key, None)

    def put(self, interaction_id: str, *, prompt: str, response: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            self._items[interaction_id] = RecentInteraction(prompt, response, now)
            self._items.move_to_end(interaction_id)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def consume(self, interaction_id: str) -> RecentInteraction | None:
        """Retorna e remove o chat, evitando uma segunda coleta da mesma conversa."""
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            return self._items.pop(interaction_id, None)
