"""Entrada de compatibilidade para a API FastAPI V6.

O antigo servidor Flask permanecia aberto por padrão e não fazia streaming de
tokens. Mantenha URLs de operação no contrato OpenAI-compatível V6.
"""

from __future__ import annotations

from api.servidor_v6 import main
from keilinks_v5.server import app, create_app

__all__ = ["app", "create_app", "main"]

if __name__ == "__main__":
    main()
