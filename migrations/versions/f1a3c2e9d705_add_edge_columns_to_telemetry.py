"""Add edge_flagged and edge_confidence columns to telemetry_readings

Adds two new columns to support the Feature 6 Edge AI Simulation pipeline:

  - edge_flagged    BOOLEAN   NOT NULL DEFAULT FALSE
      True when the edge node's local Z-score pre-screener flagged this reading
      before cloud transmission.  Server-side DEFAULT ensures all existing rows
      remain valid without any backfill — fully backward-compatible.

  - edge_confidence FLOAT(6)  NULLABLE
      Edge pre-screening confidence [0.0–1.0].  NULL for standard readings.

A plain index on edge_flagged is added to accelerate the priority-analytics
query that fetches all pre-screened anomalies in a batch sweep.

Revision ID: f1a3c2e9d705
Revises: d4e7f9a0b512
Create Date: 2026-06-25 17:10:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a3c2e9d705'
down_revision: Union[str, None] = 'd4e7f9a0b512'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # edge_flagged: Boolean with server-side default FALSE so existing rows
    # are unchanged without any UPDATE / backfill pass.
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'edge_flagged',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('FALSE'),
            comment='True when the edge node pre-screened this reading as anomalous (Feature 6)',
        ),
    )

    # edge_confidence: nullable float — only populated for edge-flagged readings.
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'edge_confidence',
            sa.Float(precision=6),
            nullable=True,
            comment='Edge pre-screening confidence [0.0–1.0]; NULL for standard readings',
        ),
    )

    # Index on edge_flagged to accelerate priority-analytics batch queries.
    op.create_index(
        'ix_telemetry_edge_flagged',
        'telemetry_readings',
        ['edge_flagged'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_telemetry_edge_flagged', table_name='telemetry_readings')
    op.drop_column('telemetry_readings', 'edge_confidence')
    op.drop_column('telemetry_readings', 'edge_flagged')
