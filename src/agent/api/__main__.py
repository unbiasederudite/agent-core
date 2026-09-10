"""CLI entrypoint for running the agent API server."""

import argparse
import sys
from pathlib import Path

import uvicorn

from agent.api.app import create_app
from agent.core.exceptions import ConfigError

_UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "loggers": {
        "uvicorn": {"handlers": [], "propagate": True},
        "uvicorn.error": {"handlers": [], "propagate": True},
        "uvicorn.access": {"handlers": [], "propagate": True},
    },
}


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, build the app, and start the uvicorn server.

    Args:
        argv: Arguments to parse. `None` uses `sys.argv`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m agent.api", description="Run the agent-core agent API."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the AppConfig JSON file."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    args = parser.parse_args(argv)

    try:
        app = create_app(args.config)
    except ConfigError as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        sys.exit(1)
    uvicorn.run(app, host=args.host, port=args.port, log_config=_UVICORN_LOG_CONFIG)


if __name__ == "__main__":
    main()
