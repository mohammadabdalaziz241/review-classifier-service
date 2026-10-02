"""Database schema management.

    python -m review_classifier.db upgrade            # apply all migrations
    python -m review_classifier.db upgrade --wait 60  # first wait for the database
    python -m review_classifier.db current            # show the schema revision
    python -m review_classifier.db downgrade base     # remove everything (tests)

The database is taken from DATABASE_URL, e.g.
postgresql://reviews:password@localhost:5432/reviews
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from .store import MIGRATIONS_DIR, create_db_engine, current_revision, head_revision, redact

logger = logging.getLogger("review_classifier.db")


def _alembic_config(connection):
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def upgrade(engine: Engine, revision: str = "head") -> None:
    from alembic import command

    with engine.begin() as connection:
        command.upgrade(_alembic_config(connection), revision)


def downgrade(engine: Engine, revision: str) -> None:
    from alembic import command

    with engine.begin() as connection:
        command.downgrade(_alembic_config(connection), revision)


def wait_for_database(engine: Engine, timeout: float, interval: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with engine.connect():
                return
        except OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the review-classifier database schema.")
    parser.add_argument(
        "--url", default=os.environ.get("DATABASE_URL"), help="Default: $DATABASE_URL"
    )
    parser.add_argument(
        "--wait", type=float, default=0, metavar="SECONDS", help="Wait for the database first"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upgrade").add_argument("revision", nargs="?", default="head")
    sub.add_parser("downgrade").add_argument("revision")
    sub.add_parser("current")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.url:
        parser.error("set DATABASE_URL or pass --url")

    engine = create_db_engine(args.url)
    try:
        if args.wait:
            logger.info("waiting up to %.0fs for %s", args.wait, redact(args.url))
            wait_for_database(engine, args.wait)
        if args.command == "upgrade":
            upgrade(engine, args.revision)
        elif args.command == "downgrade":
            downgrade(engine, args.revision)
        current = current_revision(engine)
        print(f"database {redact(args.url)} at revision {current} (code head: {head_revision()})")
    except OperationalError as exc:
        print(f"Cannot reach the database at {redact(args.url)}: {exc.orig}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
