"""Servidor Keilinks V5/V6, executável com: python -m api.servidor_v6."""

from __future__ import annotations

import uvicorn

from keilinks_v5.server import create_app
from keilinks_v5.settings import KeilinksSettings


def main() -> None:
    settings = KeilinksSettings.from_env()
    settings.validate()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
