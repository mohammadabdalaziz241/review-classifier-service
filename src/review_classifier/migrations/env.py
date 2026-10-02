"""Alembic environment.

Migrations are run through ``python -m review_classifier.db``, which passes an
open connection in ``config.attributes["connection"]`` so the database URL never
goes through Alembic's ini-file parsing (which would mangle ``%`` in passwords).
"""

from alembic import context

config = context.config


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        raise RuntimeError("Run migrations with: python -m review_classifier.db upgrade")
    context.configure(connection=connection, target_metadata=None, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline (SQL script) mode is not supported.")
run_migrations_online()
