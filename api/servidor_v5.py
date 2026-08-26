"""Entrada de compatibilidade para a API FastAPI V6.

O antigo servidor Flask permanecia aberto por padrão e não fazia streaming de
tokens. Mantenha URLs de operação no contrato OpenAI-compatível V6.
"""

from __future__ import annotations

import uvicorn

from keilinks_v5.server import app, create_app
from keilinks_v5.settings import KeilinksSettings

__all__ = ["app", "create_app", "main"]


def main() -> None:
    settings = KeilinksSettings.from_env()
    settings.validate()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
