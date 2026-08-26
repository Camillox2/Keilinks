"""Proteções mínimas para a API local da V5."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


def api_key_matches(expected: str, provided: str | None) -> bool:
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(expected, provided)


@dataclass
class SlidingWindowRateLimiter:
    """Rate limiter em memória, adequado a uso local e beta de um processo."""

    requests_per_minute: int = 30
    _events: defaultdict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(deque), init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def allow(self, identity: str) -> bool:
        now = time.monotonic()
        threshold = now - 60.0
        with self._lock:
            events = self._events[identity]
            while events and events[0] < threshold:
                events.popleft()
            if len(events) >= self.requests_per_minute:
                return False
            events.append(now)
            return True
