"""Add ML analytics columns to telemetry_readings

Adds four new nullable columns to support the Feature 2 analytics pipeline:
  - is_anomalous        BOOLEAN  DEFAULT false
  - anomaly_type        VARCHAR(64)
  - anomaly_confidence  FLOAT
  - predicted_load_24h  FLOAT

All columns are nullable so existing rows (written before analytics ran) remain
valid and no backfill is required.

Revision ID: b7f3c8d2e901
Revises: 993a2a1551dd
Create Date: 2026-06-25 11:19:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7f3c8d2e901'
down_revision: Union[str, None] = '993a2a1551dd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # is_anomalous: boolean flag, defaults to false for backward compatibility
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'is_anomalous',
            sa.Boolean(),
            nullable=True,
            server_default=sa.text('false'),
            comment='True when flagged as anomalous by the ML pipeline',
        ),
    )

    # anomaly_type: short string label for the detected anomaly class
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'anomaly_type',
            sa.String(length=64),
            nullable=True,
            comment='Short label: voltage_sag | voltage_swell | low_power_factor | line_tapping | ml_outlier',
        ),
    )

    # anomaly_confidence: ML confidence score in [0.0, 1.0]
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'anomaly_confidence',
            sa.Float(precision=6),
            nullable=True,
            comment='Anomaly detection confidence [0.0 – 1.0]; NULL for normal readings',
        ),
    )

    # predicted_load_24h: 24-hour load forecast in Watts
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'predicted_load_24h',
            sa.Float(precision=6),
            nullable=True,
            comment='Forecasted aggregate load 24 hours from this reading (Watts)',
        ),
    )

    # Partial index to make querying anomalous rows efficient
    op.create_index(
        'ix_telemetry_anomalous',
        'telemetry_readings',
        ['meter_id', 'timestamp'],
        unique=False,
        postgresql_where=sa.text('is_anomalous = true'),
    )


def downgrade() -> None:
    op.drop_index('ix_telemetry_anomalous', table_name='telemetry_readings')
    op.drop_column('telemetry_readings', 'predicted_load_24h')
    op.drop_column('telemetry_readings', 'anomaly_confidence')
    op.drop_column('telemetry_readings', 'anomaly_type')
    op.drop_column('telemetry_readings', 'is_anomalous')
