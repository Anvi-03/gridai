"""
GridPulse AI — Context Retriever  (services/context_retriever.py)

Purpose
-------
Execute a structured burst of fast aggregate SQL queries against PostgreSQL
and return a single, compact string block that captures the current health
state of the grid.  This string is injected verbatim into the LLM system
prompt, providing real, factual grounding ("RAG-Lite") so the copilot never
hallucinate numbers.

Design principles
-----------------
• **Read-only** — every query here is a SELECT.  No mutations.
• **Time-bounded** — all queries constrain to the last 24 hours so the
  context stays fresh and token-efficient.
• **Async** — runs inside the async request handler; uses AsyncSession so
  the event loop is never blocked.
• **Graceful degradation** — if any individual query fails it logs the error
  and returns a partial context block rather than crashing the copilot endpoint.
• **Strict separation** — this module contains ONLY database logic.  Zero LLM
  calls, zero prompt strings.

Output schema (plain text, injected into system prompt)
-------------------------------------------------------
=== GRID HEALTH SNAPSHOT [2026-06-25T12:17:00Z] ===
WINDOW: Last 24 hours

[FLEET OVERVIEW]
  Total readings ingested : 14,302
  Unique meters active    : 20
  Anomalous readings      : 47  (0.33%)
  Normal readings         : 14,255

[FINANCIAL SUMMARY]
  Total revenue loss (INR): ₹342.18
  Max single-event loss   : ₹22.95  (meter: METER-007)
  Avg loss per anomaly    : ₹7.28

[RISK SUMMARY]
  Max outage risk score   : 100/100  (meter: METER-007)
  Avg outage risk score   : 76/100
  High-risk meters (>70)  : 3

[TOP 5 ANOMALOUS METERS]
  #1  METER-007   — 12 events  ₹42.18 loss  risk 100/100  last: voltage_sag
  #2  METER-013   —  8 events  ₹28.44 loss  risk  88/100  last: line_tapping
  ...

[RECENT ANOMALY EVENTS (last 10)]
  2026-06-25T11:42:00Z  METER-007  line_tapping  V=160V  I=60A  PF=0.40  loss=₹7.59  risk=100
  ...

[FLEET ELECTRICAL AVERAGES]
  Avg voltage      : 229.8 V
  Avg current      : 15.2 A
  Avg power factor : 0.927
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import TelemetryReading

logger = logging.getLogger("gridpulse.context_retriever")

# Window for all context queries
CONTEXT_WINDOW_HOURS = 24


class ContextRetriever:
    """
    Executes a structured set of aggregate queries and returns a plain-text
    context block for injection into the LLM system prompt.

    Instantiate once; call `build_context(db)` per copilot request.
    """

    async def build_context(self, db: AsyncSession) -> str:
        """
        Run all summary queries and assemble the grid health snapshot string.

        Parameters
        ----------
        db : AsyncSession — an active async database session.

        Returns
        -------
        Multi-line string ready to be embedded in a system prompt.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=CONTEXT_WINDOW_HOURS)
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        sections: list[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        sections.append(
            f"=== GRID HEALTH SNAPSHOT [{now_str}] ===\n"
            f"WINDOW: Last {CONTEXT_WINDOW_HOURS} hours\n"
        )

        # ── Fleet overview ─────────────────────────────────────────────────────
        try:
            overview = await self._fleet_overview(db, cutoff)
            sections.append(overview)
        except Exception as exc:
            logger.error("Context retrieval — fleet overview failed: %s", exc)
            sections.append("[FLEET OVERVIEW]\n  (data unavailable)\n")

        # ── Financial summary ──────────────────────────────────────────────────
        try:
            financial = await self._financial_summary(db, cutoff)
            sections.append(financial)
        except Exception as exc:
            logger.error("Context retrieval — financial summary failed: %s", exc)
            sections.append("[FINANCIAL SUMMARY]\n  (data unavailable)\n")

        # ── Risk summary ───────────────────────────────────────────────────────
        try:
            risk = await self._risk_summary(db, cutoff)
            sections.append(risk)
        except Exception as exc:
            logger.error("Context retrieval — risk summary failed: %s", exc)
            sections.append("[RISK SUMMARY]\n  (data unavailable)\n")

        # ── Top anomalous meters ───────────────────────────────────────────────
        try:
            top_meters = await self._top_anomalous_meters(db, cutoff)
            sections.append(top_meters)
        except Exception as exc:
            logger.error("Context retrieval — top meters failed: %s", exc)
            sections.append("[TOP ANOMALOUS METERS]\n  (data unavailable)\n")

        # ── Recent anomaly events ──────────────────────────────────────────────
        try:
            recent = await self._recent_anomaly_events(db, cutoff)
            sections.append(recent)
        except Exception as exc:
            logger.error("Context retrieval — recent events failed: %s", exc)
            sections.append("[RECENT ANOMALY EVENTS]\n  (data unavailable)\n")

        # ── Electrical averages ────────────────────────────────────────────────
        try:
            elec = await self._electrical_averages(db, cutoff)
            sections.append(elec)
        except Exception as exc:
            logger.error("Context retrieval — electrical averages failed: %s", exc)
            sections.append("[FLEET ELECTRICAL AVERAGES]\n  (data unavailable)\n")

        return "\n".join(sections)

    # ── Individual query methods ───────────────────────────────────────────────

    async def _fleet_overview(self, db: AsyncSession, cutoff: datetime) -> str:
        # Count total and unique meters
        overview_stmt = select(
            func.count().label("total"),
            func.count(TelemetryReading.meter_id.distinct()).label("unique_meters"),
        ).where(TelemetryReading.timestamp >= cutoff)

        row = (await db.execute(overview_stmt)).one()
        total  = int(row.total or 0)
        unique = int(row.unique_meters or 0)

        # Count anomalous separately (SQLAlchemy-safe: filter + count)
        anom_stmt = select(func.count()).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
        )
        anomalous = int((await db.execute(anom_stmt)).scalar() or 0)
        normal    = total - anomalous
        pct       = (anomalous / total * 100) if total else 0.0

        return (
            "[FLEET OVERVIEW]\n"
            f"  Total readings ingested : {total:,}\n"
            f"  Unique meters active    : {unique}\n"
            f"  Anomalous readings      : {anomalous}  ({pct:.2f}%)\n"
            f"  Normal readings         : {normal:,}\n"
        )

    async def _financial_summary(self, db: AsyncSession, cutoff: datetime) -> str:
        stmt = select(
            func.sum(TelemetryReading.revenue_loss_inr).label("total_loss"),
            func.max(TelemetryReading.revenue_loss_inr).label("max_loss"),
            func.avg(TelemetryReading.revenue_loss_inr).label("avg_loss"),
            TelemetryReading.meter_id,
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
            TelemetryReading.revenue_loss_inr.isnot(None),
        ).group_by(TelemetryReading.meter_id).order_by(
            func.max(TelemetryReading.revenue_loss_inr).desc()
        ).limit(1)

        # Get aggregates across ALL anomalous rows first
        agg_stmt = select(
            func.coalesce(func.sum(TelemetryReading.revenue_loss_inr), 0).label("total_loss"),
            func.coalesce(func.avg(TelemetryReading.revenue_loss_inr), 0).label("avg_loss"),
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
            TelemetryReading.revenue_loss_inr.isnot(None),
        )
        agg = (await db.execute(agg_stmt)).one()
        total_loss = float(agg.total_loss or 0)
        avg_loss   = float(agg.avg_loss or 0)

        # Max loss + which meter
        max_stmt = select(
            func.max(TelemetryReading.revenue_loss_inr).label("max_loss"),
            TelemetryReading.meter_id,
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
            TelemetryReading.revenue_loss_inr.isnot(None),
        ).group_by(TelemetryReading.meter_id).order_by(
            func.max(TelemetryReading.revenue_loss_inr).desc()
        ).limit(1)

        max_row = (await db.execute(max_stmt)).first()
        max_loss = float(max_row.max_loss) if max_row else 0.0
        max_meter = max_row.meter_id if max_row else "N/A"

        return (
            "[FINANCIAL SUMMARY]\n"
            f"  Total revenue loss (INR): Rs.{total_loss:,.2f}\n"
            f"  Max single-event loss   : Rs.{max_loss:.2f}  (meter: {max_meter})\n"
            f"  Avg loss per anomaly    : Rs.{avg_loss:.2f}\n"
        )

    async def _risk_summary(self, db: AsyncSession, cutoff: datetime) -> str:
        # Max risk + meter
        max_stmt = select(
            func.max(TelemetryReading.outage_risk_score).label("max_risk"),
            TelemetryReading.meter_id,
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
            TelemetryReading.outage_risk_score.isnot(None),
        ).group_by(TelemetryReading.meter_id).order_by(
            func.max(TelemetryReading.outage_risk_score).desc()
        ).limit(1)

        max_row = (await db.execute(max_stmt)).first()
        max_risk   = int(max_row.max_risk)  if max_row else 0
        max_meter  = max_row.meter_id       if max_row else "N/A"

        # Avg risk across anomalous
        avg_stmt = select(
            func.coalesce(func.avg(TelemetryReading.outage_risk_score), 0).label("avg_risk"),
            func.count().label("high_risk_count"),
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
            TelemetryReading.outage_risk_score.isnot(None),
        )
        avg_row   = (await db.execute(avg_stmt)).one()
        avg_risk  = float(avg_row.avg_risk or 0)

        # Count of distinct meters where max risk > 70
        high_stmt = select(
            func.count(TelemetryReading.meter_id.distinct()).label("high_count")
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.outage_risk_score > 70,
        )
        high_count = int((await db.execute(high_stmt)).scalar() or 0)

        return (
            "[RISK SUMMARY]\n"
            f"  Max outage risk score   : {max_risk}/100  (meter: {max_meter})\n"
            f"  Avg outage risk score   : {avg_risk:.0f}/100\n"
            f"  High-risk meters (>70)  : {high_count}\n"
        )

    async def _top_anomalous_meters(self, db: AsyncSession, cutoff: datetime) -> str:
        stmt = select(
            TelemetryReading.meter_id,
            func.count().label("event_count"),
            func.coalesce(func.sum(TelemetryReading.revenue_loss_inr), 0).label("total_loss"),
            func.max(TelemetryReading.outage_risk_score).label("max_risk"),
            # Most recent anomaly type for that meter
            func.max(TelemetryReading.anomaly_type).label("last_type"),
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
        ).group_by(TelemetryReading.meter_id).order_by(
            func.count().desc()
        ).limit(5)

        rows = (await db.execute(stmt)).fetchall()

        if not rows:
            return "[TOP 5 ANOMALOUS METERS]\n  No anomalies detected in window.\n"

        lines = ["[TOP 5 ANOMALOUS METERS]"]
        for i, r in enumerate(rows, 1):
            loss  = float(r.total_loss or 0)
            risk  = int(r.max_risk or 0)
            lines.append(
                f"  #{i:<2} {r.meter_id:<20} — {r.event_count:>4} events  "
                f"Rs.{loss:>7.2f} loss  risk {risk:>3}/100  last: {r.last_type or 'unknown'}"
            )

        return "\n".join(lines) + "\n"

    async def _recent_anomaly_events(self, db: AsyncSession, cutoff: datetime) -> str:
        stmt = select(
            TelemetryReading.timestamp,
            TelemetryReading.meter_id,
            TelemetryReading.anomaly_type,
            TelemetryReading.voltage,
            TelemetryReading.current,
            TelemetryReading.power_factor,
            TelemetryReading.revenue_loss_inr,
            TelemetryReading.outage_risk_score,
        ).where(
            TelemetryReading.timestamp >= cutoff,
            TelemetryReading.is_anomalous == True,  # noqa: E712
        ).order_by(
            TelemetryReading.timestamp.desc()  # type: ignore[attr-defined]
        ).limit(10)

        rows = (await db.execute(stmt)).fetchall()

        if not rows:
            return "[RECENT ANOMALY EVENTS (last 10)]\n  No anomalies in window.\n"

        lines = ["[RECENT ANOMALY EVENTS (last 10)]"]
        for r in rows:
            ts    = str(r.timestamp)[:19] + "Z" if r.timestamp else "N/A"
            loss  = f"Rs.{float(r.revenue_loss_inr):.2f}" if r.revenue_loss_inr else "N/A"
            risk  = str(r.outage_risk_score)              if r.outage_risk_score is not None else "N/A"
            lines.append(
                f"  {ts}  {r.meter_id:<20}  {r.anomaly_type or 'unknown':<22}  "
                f"V={r.voltage:.0f}V  I={r.current:.0f}A  PF={r.power_factor:.2f}  "
                f"loss={loss}  risk={risk}"
            )

        return "\n".join(lines) + "\n"

    async def _electrical_averages(self, db: AsyncSession, cutoff: datetime) -> str:
        stmt = select(
            func.avg(TelemetryReading.voltage).label("avg_v"),
            func.avg(TelemetryReading.current).label("avg_i"),
            func.avg(TelemetryReading.power_factor).label("avg_pf"),
            func.min(TelemetryReading.voltage).label("min_v"),
            func.max(TelemetryReading.voltage).label("max_v"),
        ).where(TelemetryReading.timestamp >= cutoff)

        r = (await db.execute(stmt)).one()
        return (
            "[FLEET ELECTRICAL AVERAGES]\n"
            f"  Avg voltage      : {float(r.avg_v or 0):.1f} V "
            f"(min {float(r.min_v or 0):.1f} V / max {float(r.max_v or 0):.1f} V)\n"
            f"  Avg current      : {float(r.avg_i or 0):.1f} A\n"
            f"  Avg power factor : {float(r.avg_pf or 0):.3f}\n"
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_retriever: ContextRetriever | None = None


def get_context_retriever() -> ContextRetriever:
    """Return the module-level ContextRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = ContextRetriever()
    return _retriever
