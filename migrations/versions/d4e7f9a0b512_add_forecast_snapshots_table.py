"""Add forecast_snapshots table for Feature 5 Predictive Outage Forecasting

Creates the forecast_snapshots table that stores one predicted load snapshot
per meter per background sweep cycle.  Three indexes are added:

  ix_forecast_meter_generated  — composite (meter_id, generated_at DESC)
                                  for fast "latest snapshot per meter" lookup.
  ix_forecast_generated        — fleet-wide temporal scan.
  ix_forecast_high_risk        — partial index on high/critical rows only,
                                  accelerating the dashboard risk-zone filter.

Revision ID: d4e7f9a0b512
Revises: c9e4f1a2b803
Create Date: 2026-06-25 16:40:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd4e7f9a0b512'
down_revision: Union[str, None] = 'c9e4f1a2b803'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'forecast_snapshots',
        sa.Column(
            'id',
            sa.UUID(),
            server_default=sa.text('gen_random_uuid()'),
            nullable=False,
            comment='Globally unique forecast identifier',
        ),
        sa.Column(
            'meter_id',
            sa.String(length=64),
            nullable=False,
            comment='Opaque identifier of the meter being forecast',
        ),
        sa.Column(
            'generated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('NOW()'),
            nullable=False,
            comment='UTC moment this forecast snapshot was computed',
        ),
        sa.Column(
            'forecast_horizon',
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            comment='UTC end-of-window for this forecast (generated_at + 24 h)',
        ),
        sa.Column(
            'predicted_peak_w',
            sa.Float(precision=8),
            nullable=False,
            comment='Maximum predicted load in the 24-hour window (Watts)',
        ),
        sa.Column(
            'predicted_avg_w',
            sa.Float(precision=8),
            nullable=False,
            comment='Mean predicted load in the 24-hour window (Watts)',
        ),
        sa.Column(
            'outage_risk_score',
            sa.Integer(),
            nullable=False,
            comment='Composite outage risk score [0-100]',
        ),
        sa.Column(
            'risk_zone',
            sa.String(length=16),
            nullable=False,
            comment='Qualitative risk band: low | medium | high | critical',
        ),
        sa.Column(
            'capacity_threshold_w',
            sa.Float(precision=8),
            nullable=False,
            comment='Substation capacity limit used for risk scoring (Watts)',
        ),
        sa.Column(
            'model_name',
            sa.String(length=128),
            nullable=False,
            comment='Name of the forecaster model that produced this snapshot',
        ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Composite index: latest forecast per meter (primary read pattern)
    op.create_index(
        'ix_forecast_meter_generated',
        'forecast_snapshots',
        ['meter_id', sa.text('generated_at DESC')],
        unique=False,
    )

    # Fleet-wide temporal scan
    op.create_index(
        'ix_forecast_generated',
        'forecast_snapshots',
        [sa.text('generated_at DESC')],
        unique=False,
    )

    # meter_id standalone for single-meter lookups
    op.create_index(
        op.f('ix_forecast_snapshots_meter_id'),
        'forecast_snapshots',
        ['meter_id'],
        unique=False,
    )

    # Partial index: accelerates dashboard risk-zone filter (high/critical only)
    op.create_index(
        'ix_forecast_high_risk',
        'forecast_snapshots',
        ['meter_id', 'outage_risk_score'],
        unique=False,
        postgresql_where=sa.text("risk_zone IN ('high', 'critical')"),
    )


def downgrade() -> None:
    op.drop_index('ix_forecast_high_risk',          table_name='forecast_snapshots')
    op.drop_index(op.f('ix_forecast_snapshots_meter_id'), table_name='forecast_snapshots')
    op.drop_index('ix_forecast_generated',           table_name='forecast_snapshots')
    op.drop_index('ix_forecast_meter_generated',     table_name='forecast_snapshots')
    op.drop_table('forecast_snapshots')
