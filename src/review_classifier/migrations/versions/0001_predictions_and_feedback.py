"""Predictions and feedback.

Revision ID: 0001
Revises:
Create Date: 2026-10-02
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("endpoint", sa.String(32), nullable=False),
        sa.Column("batch_index", sa.Integer, nullable=False),
        sa.Column("batch_size", sa.Integer, nullable=False),
        sa.Column("model_id", sa.String(512), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("scores", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        sa.Column("text_sha256", sa.String(64), nullable=False),
        sa.Column("text_chars", sa.Integer, nullable=False),
        sa.Column("inference_ms", sa.Float, nullable=False),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_request_id", "predictions", ["request_id"])
    op.create_index(
        "ix_predictions_model_version_created_at", "predictions", ["model_version", "created_at"]
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
        sa.Column(
            "prediction_id",
            sa.Uuid,
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("text", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_index("ix_predictions_model_version_created_at", table_name="predictions")
    op.drop_index("ix_predictions_request_id", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_table("predictions")
