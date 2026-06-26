"""Add economic impact columns to telemetry_readings

Adds two new nullable columns to support the Feature 3 Financial Engine:
  - revenue_loss_inr   FLOAT(8)  — estimated revenue loss in Indian Rupees
  - outage_risk_score  INTEGER   — composite transformer stress score [0–100]

Both columns are nullable so existing rows (written before Feature 3 was deployed)
remain valid without any backfill.  A partial index is added on is_anomalous=true
rows to accelerate the most common financial-reporting query pattern.

Revision ID: c9e4f1a2b803
Revises: b7f3c8d2e901
Create Date: 2026-06-25 11:52:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9e4f1a2b803'
down_revision: Union[str, None] = 'b7f3c8d2e901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # revenue_loss_inr: estimated monetary loss per anomaly event in Indian Rupees
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'revenue_loss_inr',
            sa.Float(precision=8),
            nullable=True,
            comment='Estimated revenue loss for this anomaly event (Indian Rupees)',
        ),
    )

    # outage_risk_score: composite transformer / substation stress score [0–100]
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'outage_risk_score',
            sa.Integer(),
            nullable=True,
            comment='Composite transformer/substation stress score [0–100]; NULL for normal readings',
        ),
    )

    # Partial index: financial reporting almost always filters anomalous rows.
    # Covering (meter_id, revenue_loss_inr DESC) lets the dashboard aggregate
    # top-loss meters without a sequential scan.
    op.create_index(
        'ix_telemetry_financial_anomalous',
        'telemetry_readings',
        ['meter_id', 'revenue_loss_inr'],
        unique=False,
        postgresql_where=sa.text('is_anomalous = true'),
    )


def downgrade() -> None:
    op.drop_index('ix_telemetry_financial_anomalous', table_name='telemetry_readings')
    op.drop_column('telemetry_readings', 'outage_risk_score')
    op.drop_column('telemetry_readings', 'revenue_loss_inr')
