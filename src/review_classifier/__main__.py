"""Entry point: ``python -m review_classifier``.

Configures logging and starts uvicorn. HOST, PORT, LOG_LEVEL and WORKERS are
read from the environment; model settings are documented in config.py.
"""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    level = os.getenv("LOG_LEVEL", "info").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "review_classifier.api:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_level=level.lower(),
        # The app writes its own access log line, with the request ID.
        access_log=False,
    )


if __name__ == "__main__":
    main()
