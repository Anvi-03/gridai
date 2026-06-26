"""
GridPulse AI — Forecasting Background Service  (services/forecasting_service.py)

Purpose
-------
Periodically sweeps all active meters in the database, extracts their recent
power-consumption history, runs the GridForecaster to produce 24-hour load
predictions, and writes a ForecastSnapshot row per meter to the
`forecast_snapshots` table.

The GET /api/v1/grid/forecast endpoint reads exclusively from this table,
so all ML computation is 100% decoupled from the API response path.

Architecture
------------
                ┌─────────────────────────────────────────┐
                │  schedule_forecast_sweep()               │
                │    asyncio.create_task(run_sweep())      │
                └──────────────┬──────────────────────────┘
                               │
                ┌──────────────▼──────────────────────────┐
                │  ForecastingService.run_forecast_sweep() │
                │                                          │
                │  1. Query all unique active meters       │
                │     (seen in last 24 h)                  │
                │                                          │
                │  2. For each meter (asyncio.Semaphore):  │
                │     a. Fetch last 96 telemetry rows      │
                │     b. Compute real power series (V×I×PF)│
                │     c. Call GridForecaster.predict_24h() │
                │     d. Compute risk score + zone         │
                │     e. Upsert ForecastSnapshot row       │
                │                                          │
                │  3. Commit all changes in one session    │
                └──────────────────────────────────────────┘

Concurrency Design
------------------
• asyncio.Semaphore(MAX_CONCURRENT_METERS=5) limits parallel DB queries
  so we never exhaust the connection pool (pool_size=50 in production but
  the semaphore provides a safety margin for test / dev environments).

• Each meter's history fetch + model inference + upsert is a single
  awaitable _process_meter() coroutine.  All meter coroutines are gathered
  concurrently, gated through the semaphore.

• The entire sweep operates within a single AsyncSession, committed once at
  the end.  If any individual meter fails, it is logged and skipped — the
  session continues for the remaining meters.

Database Safety
---------------
• DELETE-then-INSERT upsert: rather than ON CONFLICT (which would require a
  unique constraint on meter_id), we delete the previous snapshot for each
  meter and insert a fresh one.  This is safe because forecasts are
  disposable — the latest is always authoritative.
• All writes are flushed in a single batch and committed once to minimise
  round-trips.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from ml.load_forecaster import GridForecaster, get_forecaster, NOMINAL_CAPACITY_W
from schemas import ForecastSnapshot, TelemetryReading

logger = logging.getLogger("gridpulse.forecasting")

# ── Tunable parameters ────────────────────────────────────────────────────────

# Maximum meters processed concurrently — gates the asyncio.Semaphore.
# Keeps DB connection pool utilisation within safe limits.
MAX_CONCURRENT_METERS = 5

# Look-back window for history fetch: last 96 readings per meter
# (96 × 30-min intervals ≈ 48 h of history → good lagged-feature coverage)
HISTORY_WINDOW = 96
HISTORY_LOOKBACK_HOURS = 48

# Substation capacity used when the meter's own history is too sparse
# to derive a per-meter estimate.
DEFAULT_CAPACITY_W = NOMINAL_CAPACITY_W  # 6,555 W


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ForecastSweepSummary:
    """Compact summary returned from run_forecast_sweep() for logging."""
    total_meters:       int = 0
    snapshots_written:  int = 0
    high_risk_count:    int = 0     # meters in 'high' or 'critical' zone
    errors:             int = 0
    duration_ms:        float = 0.0
    meter_results:      dict[str, str] = field(default_factory=dict)  # meter_id → zone


# ── Core forecasting service ──────────────────────────────────────────────────

class ForecastingService:
    """
    Stateless background service that runs per-meter load forecasting and
    persists the results to `forecast_snapshots`.

    Usage (fire-and-forget from ingest route or lifespan)
    -------------------------------------------------------
        from services.forecasting_service import schedule_forecast_sweep
        schedule_forecast_sweep()   # non-blocking
    """

    def __init__(self) -> None:
        # Retrieve the active forecaster singleton — will be GridForecaster
        # unless overridden by env vars (ENABLE_LSTM / ENABLE_GRID_FORECASTER).
        forecaster = get_forecaster()

        # Prefer the full GridForecaster API; fall back gracefully if a
        # different forecaster type is active (e.g. MovingAverageForecaster).
        if isinstance(forecaster, GridForecaster):
            self._forecaster: GridForecaster = forecaster
        else:
            # Warm-up a dedicated GridForecaster for the sweep regardless of
            # the analytics-pipeline forecaster choice.
            logger.info(
                "ForecastingService: active forecaster is %s — "
                "creating a dedicated GridForecaster for sweep.",
                forecaster.name,
            )
            self._forecaster = GridForecaster()

        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_METERS)
        logger.info(
            "ForecastingService ready — forecaster=%s  concurrency=%d",
            self._forecaster.name, MAX_CONCURRENT_METERS,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    async def run_forecast_sweep(self) -> ForecastSweepSummary:
        """
        Execute a full fleet forecast sweep.

        Steps
        -----
        1. Open a single AsyncSession.
        2. Discover all meters active in the last 24 hours.
        3. Process each meter concurrently (gated by semaphore).
        4. Commit all upserted ForecastSnapshot rows.
        5. Return a summary for logging.

        This coroutine is safe to schedule via asyncio.create_task().
        """
        t0      = time.perf_counter()
        summary = ForecastSweepSummary()

        try:
            async with AsyncSessionLocal() as db:
                # ── 1. Discover active meters ─────────────────────────────────
                active_meters = await self._get_active_meters(db)
                summary.total_meters = len(active_meters)

                if not active_meters:
                    logger.info("ForecastingService: no active meters found — sweep skipped.")
                    return summary

                logger.info(
                    "ForecastingService: starting sweep for %d active meter(s).",
                    len(active_meters),
                )

                # ── 2. Process meters concurrently ────────────────────────────
                tasks = [
                    self._process_meter(db, meter_id)
                    for meter_id in active_meters
                ]
                results: list[ForecastSnapshot | None] = await asyncio.gather(
                    *tasks, return_exceptions=False
                )

                # ── 3. Collect valid snapshots + delete old rows ──────────────
                valid_snapshots: list[ForecastSnapshot] = []
                for meter_id, snap in zip(active_meters, results):
                    if snap is None:
                        summary.errors += 1
                        continue
                    valid_snapshots.append(snap)
                    summary.snapshots_written += 1
                    summary.meter_results[meter_id] = snap.risk_zone
                    if snap.risk_zone in ("high", "critical"):
                        summary.high_risk_count += 1

                if valid_snapshots:
                    # Delete stale snapshots for the meters we are about to refresh
                    meter_ids_to_update = [s.meter_id for s in valid_snapshots]
                    await db.execute(
                        delete(ForecastSnapshot).where(
                            ForecastSnapshot.meter_id.in_(meter_ids_to_update)
                        )
                    )
                    db.add_all(valid_snapshots)
                    await db.commit()

        except Exception as exc:
            logger.error(
                "ForecastingService: sweep failed with exception: %s",
                exc, exc_info=True,
            )
            summary.errors += 1

        summary.duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Forecast sweep complete | meters=%d  written=%d  "
            "high_risk=%d  errors=%d  duration=%.1f ms",
            summary.total_meters,
            summary.snapshots_written,
            summary.high_risk_count,
            summary.errors,
            summary.duration_ms,
        )
        return summary

    # ── Per-meter processing (runs inside semaphore gate) ─────────────────────

    async def _process_meter(
        self,
        db: AsyncSession,
        meter_id: str,
    ) -> ForecastSnapshot | None:
        """
        Fetch history, run GridForecaster, build a ForecastSnapshot object.

        The asyncio.Semaphore limits how many of these coroutines execute their
        DB-fetch concurrently, protecting the connection pool.

        Returns None on any per-meter failure (logged internally).
        """
        async with self._semaphore:
            try:
                # Fetch recent power history for this meter
                history_w = await self._fetch_meter_history(db, meter_id)

                # Run the 24-step forecast — cold-start / imputation handled inside
                predictions_w = self._forecaster.predict_next_24h(history_w)

                if not predictions_w:
                    logger.warning("ForecastingService: empty predictions for meter %s", meter_id)
                    return None

                predicted_peak_w = max(predictions_w)
                predicted_avg_w  = sum(predictions_w) / len(predictions_w)

                # Derive per-meter capacity: use 1.25× historical peak or the
                # default nominal capacity, whichever is larger.
                hist_peak        = max(history_w) if history_w else 0.0
                capacity_w       = max(DEFAULT_CAPACITY_W, hist_peak * 1.25)

                risk_score, risk_zone = self._forecaster.compute_risk_score(
                    predicted_peak_w, capacity_threshold_w=capacity_w
                )

                now           = datetime.now(tz=timezone.utc)
                horizon       = now + timedelta(hours=24)

                snapshot = ForecastSnapshot(
                    meter_id             = meter_id,
                    generated_at         = now,
                    forecast_horizon     = horizon,
                    predicted_peak_w     = round(predicted_peak_w, 2),
                    predicted_avg_w      = round(predicted_avg_w, 2),
                    outage_risk_score    = risk_score,
                    risk_zone            = risk_zone,
                    capacity_threshold_w = round(capacity_w, 2),
                    model_name           = self._forecaster.name,
                )

                logger.debug(
                    "Forecast | meter=%-22s  peak=%.1f W  cap=%.1f W  "
                    "risk=%d/100  zone=%s",
                    meter_id, predicted_peak_w, capacity_w, risk_score, risk_zone,
                )
                return snapshot

            except Exception as exc:
                logger.error(
                    "ForecastingService: error processing meter %s: %s",
                    meter_id, exc, exc_info=True,
                )
                return None

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _get_active_meters(self, db: AsyncSession) -> list[str]:
        """Return distinct meter IDs that have readings in the last 24 hours."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        stmt = (
            select(TelemetryReading.meter_id.distinct())
            .where(TelemetryReading.timestamp >= cutoff)
            .order_by(TelemetryReading.meter_id)
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def _fetch_meter_history(
        self,
        db: AsyncSession,
        meter_id: str,
    ) -> list[float]:
        """
        Fetch the last HISTORY_WINDOW real-power samples for *meter_id*.

        Real power (Watts) = voltage × current × power_factor.
        Rows are returned oldest → newest so the forecaster sees a
        chronological time series.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=HISTORY_LOOKBACK_HOURS)
        stmt = (
            select(
                TelemetryReading.voltage,
                TelemetryReading.current,
                TelemetryReading.power_factor,
            )
            .where(
                TelemetryReading.meter_id == meter_id,
                TelemetryReading.timestamp >= cutoff,
            )
            .order_by(TelemetryReading.timestamp.asc())  # type: ignore[attr-defined]
            .limit(HISTORY_WINDOW)
        )
        result = await db.execute(stmt)
        rows   = result.fetchall()

        return [
            float(r.voltage) * float(r.current) * float(r.power_factor)
            for r in rows
        ]


# ── Module-level singleton ────────────────────────────────────────────────────

_service: ForecastingService | None = None


def get_forecasting_service() -> ForecastingService:
    """Return the module-level ForecastingService singleton."""
    global _service
    if _service is None:
        _service = ForecastingService()
    return _service


# ── Fire-and-forget scheduler ─────────────────────────────────────────────────

def schedule_forecast_sweep() -> asyncio.Task:
    """
    Schedule a background forecast sweep without blocking the caller.

    Call this from:
      • The FastAPI ingest route (after committing telemetry).
      • The application lifespan handler (initial warm-up sweep).

    Returns the asyncio.Task so the caller can attach callbacks if needed.
    """
    service = get_forecasting_service()
    task    = asyncio.create_task(
        service.run_forecast_sweep(),
        name="forecast_sweep",
    )

    def _log_result(t: asyncio.Task) -> None:
        if t.cancelled():
            logger.warning("Forecast sweep task was cancelled.")
        elif t.exception():
            logger.error(
                "Forecast sweep task raised an exception: %s",
                t.exception(), exc_info=t.exception(),
            )

    task.add_done_callback(_log_result)
    logger.debug("Forecast sweep scheduled.")
    return task
