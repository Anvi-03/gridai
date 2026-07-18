"""
GridPulse AI — Advisory API Router  (api/v1/advisory.py)

Exposes:
    GET /api/v1/advisory
        Fetches the latest telemetry carbon intensity and the current fleet
        load forecast, runs them through the Agentic Advisory Engine, and
        returns a structured operational recommendation.

    GET /api/v1/advisory/policy
        Returns the static microgrid policy tier configuration so operators
        and dashboards can display the tier hierarchy without hardcoding it.

Response Shape (GET /api/v1/advisory)
--------------------------------------
    AdvisoryResponse
    ├── status                  : "Normal" | "Warning" | "Critical"
    ├── advisory_active         : bool
    ├── advisory_message        : str   — human-readable recommendation
    ├── carbon_intensity        : float | None   — gCO₂/kWh from latest reading
    ├── predicted_load_kw       : float | None   — 24-h LSTM/Grid forecast (kW)
    ├── triggered_by            : list[str]      — which thresholds fired
    ├── load_threshold_kw       : float          — policy threshold used
    ├── carbon_threshold        : float          — policy threshold used
    ├── tier_actions            : dict           — per-tier recommended action
    └── generated_at            : str            — UTC ISO-8601 of this advisory

Design Decisions
----------------
• Read-mostly — the endpoint reads carbon_intensity from the latest
  telemetry row and predicted_load from the latest ForecastSnapshot.
  Both are indexed, single-row reads — sub-millisecond latency.

• Graceful degradation — if no telemetry or forecast data is available
  yet (e.g. cold start), the endpoint returns a "Normal" advisory with
  null metric fields rather than a 503, so dashboards never fail on boot.

• Zero ML at request time — all forecasts are pre-computed by the
  background ForecastingService.  This endpoint only reads cached results.

• Advisory logic is fully encapsulated in services.advisory — this router
  is intentionally thin (fetch → call → serialize).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from schemas import ForecastSnapshot, TelemetryReading
from services.advisory import (
    MICROGRID_POLICY_TIERS,
    get_grid_advisory,
)

logger = logging.getLogger("gridpulse.advisory.router")

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/advisory",
    tags=["Agentic Advisory"],
)


# ── Pydantic response models ──────────────────────────────────────────────────

class PolicyTierDetail(BaseModel):
    """Detail for a single microgrid policy tier."""
    label:       str = Field(description="Short tier label, e.g. 'Critical'.")
    description: str = Field(description="Comma-separated list of asset types in this tier.")
    shed:        bool = Field(description="True when this tier is eligible for load shedding.")
    note:        str  = Field(description="Operator guidance note for this tier.")


class PolicyTiersResponse(BaseModel):
    """Static microgrid load-shedding policy configuration."""
    tiers:              dict[int, PolicyTierDetail]
    load_threshold_kw:  float = Field(description="Load threshold (kW) that triggers an advisory.")
    carbon_threshold:   float = Field(description="Carbon intensity (gCO₂/kWh) that triggers an advisory.")


class AdvisoryResponse(BaseModel):
    """
    Structured grid advisory returned by GET /api/v1/advisory.

    Fields
    ------
    status          : "Normal" | "Warning" | "Critical"
    advisory_active : True when at least one threshold is breached.
    advisory_message: Human-readable (LLM-grade) operational recommendation.
    carbon_intensity: Most recent grid carbon intensity in gCO₂/kWh.
                      None if no telemetry data is available yet.
    predicted_load_kw: 24-hour peak predicted load in kilowatts.
                      None if no forecast data is available yet.
    triggered_by    : List of condition keys that fired: ["high_load", "high_carbon"].
    load_threshold_kw: The load threshold used for evaluation.
    carbon_threshold : The carbon intensity threshold used for evaluation.
    tier_actions    : Per-tier action map keyed by tier number (1–4).
    generated_at    : UTC ISO-8601 timestamp of when this advisory was generated.
    """
    status:             str              = Field(description="Normal | Warning | Critical")
    advisory_active:    bool             = Field(description="True when mitigation action is recommended.")
    advisory_message:   str              = Field(description="Detailed advisory recommendation.")
    carbon_intensity:   float | None     = Field(default=None, description="Latest carbon intensity (gCO₂/kWh).")
    predicted_load_kw:  float | None     = Field(default=None, description="24-h peak load forecast (kW).")
    triggered_by:       list[str]        = Field(default_factory=list, description="Conditions that triggered this advisory.")
    load_threshold_kw:  float            = Field(description="Load trigger threshold (kW).")
    carbon_threshold:   float            = Field(description="Carbon intensity trigger threshold (gCO₂/kWh).")
    tier_actions:       dict[int, str]   = Field(default_factory=dict, description="Per-tier recommended action.")
    generated_at:       str              = Field(description="UTC ISO-8601 advisory generation timestamp.")


# ── Data-fetch helpers ────────────────────────────────────────────────────────

async def _fetch_latest_carbon_intensity(db: AsyncSession) -> float | None:
    """
    Query the most recent carbon_intensity_gco2_kwh value from telemetry_readings.

    Returns None if the table is empty or no reading has a non-null value yet.
    Uses the ix_telemetry_timestamp index (timestamp DESC) — single-row scan.
    """
    stmt = (
        select(TelemetryReading.carbon_intensity_gco2_kwh)
        .where(TelemetryReading.carbon_intensity_gco2_kwh.is_not(None))
        .order_by(TelemetryReading.timestamp.desc())  # type: ignore[attr-defined]
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    return float(row) if row is not None else None


async def _fetch_latest_predicted_load_kw(db: AsyncSession) -> float | None:
    """
    Query the latest fleet peak predicted load from the forecast_snapshots table.

    Returns the maximum predicted_peak_w across all meters in the latest sweep,
    converted to kilowatts.  This is the worst-case (highest-risk) load forecast
    for the advisory — the value most likely to require mitigation.

    Returns None if no forecast snapshots exist yet.
    """
    # Subquery: the most recent generated_at timestamp across all meters
    latest_ts_sq = select(func.max(ForecastSnapshot.generated_at)).scalar_subquery()

    stmt = (
        select(func.max(ForecastSnapshot.predicted_peak_w))
        .where(ForecastSnapshot.generated_at == latest_ts_sq)
    )
    result = await db.execute(stmt)
    peak_w = result.scalar_one_or_none()
    if peak_w is None:
        return None
    return round(float(peak_w) / 1000.0, 3)  # Watts → kilowatts


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=AdvisoryResponse,
    summary="Agentic grid advisory — load & carbon mitigation recommendations",
    description=(
        "Fetches the latest carbon intensity from the telemetry stream and the "
        "current 24-hour load forecast, evaluates both against the microgrid "
        "policy thresholds, and returns a structured operational advisory with "
        "per-tier load-shedding recommendations. "
        "Zero ML inference occurs at request time — all data is pre-computed."
    ),
    responses={
        200: {"description": "Advisory successfully generated."},
    },
)
async def get_advisory(
    db: AsyncSession = Depends(get_db),
) -> AdvisoryResponse:
    """
    Agentic Advisory Engine endpoint.

    Flow
    ----
    1. Read the latest carbon_intensity_gco2_kwh from telemetry_readings.
    2. Read the latest fleet peak load forecast from forecast_snapshots (kW).
    3. Call get_grid_advisory(predicted_load_kw, carbon_intensity).
    4. Return the structured AdvisoryResponse.

    Graceful Degradation
    --------------------
    If no telemetry or forecast data is available (cold start / empty DB),
    the function uses a 0.0 sentinel value so the advisory engine always runs
    and returns a "Normal" status rather than raising a 503.
    """
    # ── Step 1: Fetch live metrics ────────────────────────────────────────────
    carbon_intensity  = await _fetch_latest_carbon_intensity(db)
    predicted_load_kw = await _fetch_latest_predicted_load_kw(db)

    # Use 0.0 sentinel for missing data so advisory engine always produces output
    ci_value   = carbon_intensity  if carbon_intensity  is not None else 0.0
    load_value = predicted_load_kw if predicted_load_kw is not None else 0.0

    logger.info(
        "Advisory requested: carbon_intensity=%.1f gCO₂/kWh | predicted_load=%.1f kW",
        ci_value, load_value,
    )

    # ── Step 2: Run advisory engine ───────────────────────────────────────────
    result = get_grid_advisory(
        predicted_load_kw=load_value,
        carbon_intensity=ci_value,
    )

    # ── Step 3: Build and return response ─────────────────────────────────────
    return AdvisoryResponse(
        status             = result.status,
        advisory_active    = result.advisory_active,
        advisory_message   = result.advisory_message,
        carbon_intensity   = carbon_intensity,       # None preserved for transparency
        predicted_load_kw  = predicted_load_kw,      # None preserved for transparency
        triggered_by       = result.triggered_by,
        load_threshold_kw  = result.load_threshold_kw,
        carbon_threshold   = result.carbon_threshold,
        tier_actions       = {str(k): v for k, v in result.tier_actions.items()},
        generated_at       = datetime.now(tz=timezone.utc).isoformat(),
    )


@router.get(
    "/policy",
    response_model=PolicyTiersResponse,
    summary="Microgrid load-shedding policy tier configuration",
    description=(
        "Returns the static microgrid policy tier hierarchy that governs "
        "which assets are protected, monitored, or shed during grid stress events. "
        "Use this endpoint to populate operator dashboards and advisory UIs "
        "without hardcoding tier definitions client-side."
    ),
)
async def get_policy_tiers() -> PolicyTiersResponse:
    """
    Return the MICROGRID_POLICY_TIERS configuration as a structured JSON response.

    This endpoint is static and read-only — no DB access required.
    """
    from services.advisory import CARBON_THRESHOLD, LOAD_THRESHOLD_KW

    return PolicyTiersResponse(
        tiers={
            tier_num: PolicyTierDetail(**tier_data)
            for tier_num, tier_data in MICROGRID_POLICY_TIERS.items()
        },
        load_threshold_kw=LOAD_THRESHOLD_KW,
        carbon_threshold=CARBON_THRESHOLD,
    )
