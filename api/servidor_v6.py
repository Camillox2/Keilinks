"""Servidor Keilinks V5/V6, executável com: python -m api.servidor_v6."""

from __future__ import annotations

import argparse
from dataclasses import replace

import uvicorn

from keilinks_v5.server import create_app
from keilinks_v5.settings import KeilinksSettings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inicia a API local Keilinks V6")
    parser.add_argument("--host", help="Sobrescreve KEILINKS_HOST para esta execução")
    parser.add_argument("--port", type=int, help="Sobrescreve KEILINKS_PORT para esta execução")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = KeilinksSettings.from_env()
    if args.host:
        settings = replace(settings, host=args.host)
    if args.port is not None:
        settings = replace(settings, port=args.port)
    settings.validate()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
