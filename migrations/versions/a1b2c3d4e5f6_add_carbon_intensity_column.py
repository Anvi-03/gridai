"""Add carbon_intensity_gco2_kwh column to telemetry_readings

Adds a new sustainability metric column to support the carbon intensity
simulation pipeline:

  - carbon_intensity_gco2_kwh  FLOAT(6)  NULLABLE
      Grid carbon intensity in gCO₂/kWh. Simulated value that fluctuates
      between 150 (high solar penetration during daytime 09–16h) and 800
      (coal-reliance at night 18–06h).

      Nullable with server_default=NULL so all existing rows remain valid
      without any UPDATE / backfill — fully backward-compatible with all
      pre-sustainability-feature clients.

Revision ID: a1b2c3d4e5f6
Revises: f1a3c2e9d705
Create Date: 2026-07-17 08:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f1a3c2e9d705'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # carbon_intensity_gco2_kwh: nullable float — populated at ingest time
    # by the simulator / seed script.  NULL for legacy readings that predate
    # the sustainability feature.
    op.add_column(
        'telemetry_readings',
        sa.Column(
            'carbon_intensity_gco2_kwh',
            sa.Float(precision=6),
            nullable=True,
            comment=(
                'Grid carbon intensity in gCO₂/kWh [150–800]. '
                'Lower during solar peak hours (09–16h), higher at night (18–06h). '
                'NULL for readings ingested before the sustainability feature was deployed.'
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column('telemetry_readings', 'carbon_intensity_gco2_kwh')
