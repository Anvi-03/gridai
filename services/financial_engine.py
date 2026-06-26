"""
GridPulse AI — Financial Impact Engine  (services/financial_engine.py)

Purpose
-------
Translate a raw anomaly detection result into quantified financial and
operational risk metrics that make the data actionable for grid operators
and compelling for executive dashboards.

Two primary outputs
-------------------
1. **revenue_loss_inr** (float, Indian Rupees)
   Estimated monetary loss caused by the anomalous reading compared with
   the baseline "should-be" power delivery at nominal grid parameters.

   Formula (per-reading basis):
     baseline_power_W  = V_nominal × I_nominal × PF_nominal
     actual_power_W    = V_actual  × I_actual  × PF_actual
     delta_power_W     = baseline_power_W − actual_power_W   (clamped ≥ 0)
     lost_energy_kWh   = delta_power_W / 1000 × READING_INTERVAL_H
     revenue_loss_inr  = lost_energy_kWh × TARIFF_INR_PER_KWH × LOSS_MULTIPLIER[anomaly_type]

   The `LOSS_MULTIPLIER` boosts the figure for higher-severity events:
     • line_tapping:      3.0× (unauthorised abstraction + criminal investigation)
     • voltage_sag/swell: 1.5× (equipment damage + downtime liability)
     • low_power_factor:  1.2× (reactive power penalty surcharge)
     • ml_outlier:        1.0× (base loss only)

2. **outage_risk_score** (int, 0–100)
   Sliding composite score representing the accumulated stress on the local
   transformer / substation segment.  Components:

     Component A — Voltage deviation severity (0–40 pts)
       Proportional to |V_actual − V_nominal| / V_nominal.  Reaches 40 at
       a ±20 % deviation (the ANSI C84.1 Range B limit).

     Component B — Current overload factor (0–35 pts)
       Proportional to max(0, I_actual − I_rated) / I_rated.  Reaches 35
       when current is 50 % above rated capacity (thermal overload zone).

     Component C — Power-factor degradation (0–25 pts)
       Linear from 0 (PF ≥ 0.95) to 25 (PF ≤ 0.50).  Low PF forces
       higher reactive current through the same conductors, accelerating
       insulation ageing.

     The three components are summed and clamped to [0, 100].

Design decisions
----------------
• **Pure Python / zero external deps** — all calculations are deterministic
  arithmetic; no ML libraries required.  This makes the engine trivially
  testable and auditable.
• **Dataclass return type** — `FinancialImpact` carries named, typed fields
  so the analytics service can unpack them without positional tuple magic.
• **Configurable constants at module top** — tariff rate, nominal parameters,
  etc. are named constants, easy to update via environment overrides in future.
• **Anomaly-type-aware multipliers** — ensures the INR loss figure is
  semantically correct per threat category, not just a flat per-kWh figure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger("gridpulse.financial")


# ─────────────────────────────────────────────────────────────────────────────
# Indian Grid Constants & Tariff Parameters
# ─────────────────────────────────────────────────────────────────────────────

# Indian commercial / industrial power tariff (CERC + state utility average).
# Source: Central Electricity Regulatory Commission (CERC) FY 2024-25.
# Range: ₹7.50 – ₹10.50 / kWh for HT commercial consumers.
# We use ₹9.00 as a realistic midpoint.
TARIFF_INR_PER_KWH: float = 9.00

# Nominal grid parameters (BIS IS 12360 / IEC 60038 for India: 230 V ±10%)
NOMINAL_VOLTAGE_V:   float = 230.0
NOMINAL_CURRENT_A:   float = 15.0   # representative residential/commercial feeder load
NOMINAL_PF:          float = 0.95   # target power factor per CERC regulations

# Assumed metering interval — each telemetry reading represents this many hours
# of continuous operating state.  At a 30-min sampling cadence: 0.5 h.
# Adjust in .env or config if your simulator interval differs.
READING_INTERVAL_H:  float = 0.5

# Rated current capacity for the local transformer segment (amperes)
RATED_CURRENT_A:     float = 30.0

# Revenue-loss multipliers per anomaly type (dimensionless)
# These reflect liability exposure beyond raw energy loss.
_LOSS_MULTIPLIERS: dict[str, float] = {
    "line_tapping":      3.0,   # Theft — full recovery cost + investigation overheads
    "voltage_sag":       1.5,   # Equipment damage liability + SLA penalty
    "voltage_swell":     1.5,   # Surge damage liability + insulation degradation
    "low_power_factor":  1.2,   # Reactive power surcharge per utility tariff
    "ml_outlier":        1.0,   # Statistical anomaly — base loss only
}
_DEFAULT_MULTIPLIER: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FinancialImpact:
    """
    Immutable result from FinancialEngine.calculate().

    Attributes
    ----------
    revenue_loss_inr  : Estimated revenue loss in Indian Rupees (≥ 0).
    outage_risk_score : Composite outage risk score in [0, 100].
    baseline_power_W  : Nominal expected power draw (Watts).
    actual_power_W    : Measured real power draw (Watts).
    lost_energy_kWh   : Energy not delivered / stolen this interval (kWh).
    tariff_inr_kwh    : Tariff rate applied (₹/kWh).
    multiplier        : Anomaly-type severity multiplier applied.
    risk_breakdown    : Dict with individual score components for audit.
    """
    revenue_loss_inr:  float
    outage_risk_score: int
    baseline_power_W:  float
    actual_power_W:    float
    lost_energy_kWh:   float
    tariff_inr_kwh:    float
    multiplier:        float
    risk_breakdown:    dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class FinancialEngine:
    """
    Deterministic financial-impact calculator for GridPulse anomaly events.

    All state is immutable after construction (constants only).
    Safe to share across coroutines / threads.

    Parameters
    ----------
    tariff_inr_per_kwh : Override the default Indian commercial tariff rate.
    reading_interval_h : Override the assumed metering interval in hours.
    nominal_voltage    : Override the nominal grid voltage.
    nominal_current    : Override the nominal feeder current.
    nominal_pf         : Override the nominal power factor.
    rated_current      : Override the rated transformer current capacity.
    """

    def __init__(
        self,
        tariff_inr_per_kwh: float = TARIFF_INR_PER_KWH,
        reading_interval_h: float = READING_INTERVAL_H,
        nominal_voltage:    float = NOMINAL_VOLTAGE_V,
        nominal_current:    float = NOMINAL_CURRENT_A,
        nominal_pf:         float = NOMINAL_PF,
        rated_current:      float = RATED_CURRENT_A,
    ) -> None:
        self._tariff    = tariff_inr_per_kwh
        self._interval  = reading_interval_h
        self._v_nom     = nominal_voltage
        self._i_nom     = nominal_current
        self._pf_nom    = nominal_pf
        self._i_rated   = rated_current

        logger.info(
            "FinancialEngine initialised — tariff=₹%.2f/kWh  interval=%.2f h  "
            "V_nom=%.1f V  I_nom=%.1f A  PF_nom=%.2f",
            self._tariff, self._interval,
            self._v_nom, self._i_nom, self._pf_nom,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate(
        self,
        voltage:      float,
        current:      float,
        power_factor: float,
        anomaly_type: str | None,
    ) -> FinancialImpact:
        """
        Compute revenue loss and outage risk for a single anomalous reading.

        Parameters
        ----------
        voltage       : Measured RMS voltage (Volts).
        current       : Measured RMS current (Amperes).
        power_factor  : Measured power factor [0.0 – 1.0].
        anomaly_type  : String label from the ML detector, or None.

        Returns
        -------
        FinancialImpact dataclass (all fields populated, immutable).
        """
        loss_inr, baseline_W, actual_W, lost_kWh, multiplier = \
            self._compute_revenue_loss(voltage, current, power_factor, anomaly_type)

        risk_score, breakdown = \
            self._compute_outage_risk(voltage, current, power_factor)

        impact = FinancialImpact(
            revenue_loss_inr  = round(loss_inr,  2),
            outage_risk_score = risk_score,
            baseline_power_W  = round(baseline_W, 2),
            actual_power_W    = round(actual_W, 2),
            lost_energy_kWh   = round(lost_kWh, 6),
            tariff_inr_kwh    = self._tariff,
            multiplier        = multiplier,
            risk_breakdown    = breakdown,
        )

        logger.debug(
            "FinancialImpact | type=%-20s  loss=₹%.2f  risk=%d/100  "
            "baseline=%.1f W  actual=%.1f W",
            anomaly_type or "none",
            impact.revenue_loss_inr,
            impact.outage_risk_score,
            impact.baseline_power_W,
            impact.actual_power_W,
        )
        return impact

    # ── Revenue-loss calculation ───────────────────────────────────────────────

    def _compute_revenue_loss(
        self,
        voltage:      float,
        current:      float,
        power_factor: float,
        anomaly_type: str | None,
    ) -> tuple[float, float, float, float, float]:
        """
        Returns (loss_inr, baseline_W, actual_W, lost_kWh, multiplier).

        For line_tapping specifically: the 'actual' draw is treated as the
        theft-inflated current at the anomalously low voltage — every Watt
        being illegitimately drawn maps to lost billable revenue.
        """
        baseline_W = self._v_nom * self._i_nom * self._pf_nom
        actual_W   = voltage * current * power_factor

        # For line-tapping the grid delivers MORE power than billed (current
        # spikes while voltage sags).  We compute actual draw and compare to
        # baseline billable consumption.
        if anomaly_type == "line_tapping":
            # Stolen = excess power drawn above normal billing baseline
            delta_W = max(0.0, actual_W - baseline_W)
        else:
            # For sag/swell/ML outlier — deficit in delivered power
            delta_W = max(0.0, baseline_W - actual_W)

        lost_kWh   = (delta_W / 1000.0) * self._interval
        multiplier = _LOSS_MULTIPLIERS.get(anomaly_type or "", _DEFAULT_MULTIPLIER)
        loss_inr   = lost_kWh * self._tariff * multiplier

        return loss_inr, baseline_W, actual_W, lost_kWh, multiplier

    # ── Outage-risk calculation ────────────────────────────────────────────────

    def _compute_outage_risk(
        self,
        voltage:      float,
        current:      float,
        power_factor: float,
    ) -> tuple[int, dict[str, float]]:
        """
        Returns (score_0_to_100, breakdown_dict).

        Component weights sum to 100 at their worst-case values:
            voltage_deviation : max 40 pts
            current_overload  : max 35 pts
            pf_degradation    : max 25 pts
        """

        # ── Component A: Voltage deviation (0–40 pts) ──────────────────────
        voltage_deviation_pct = abs(voltage - self._v_nom) / self._v_nom
        # Reaches maximum (40 pts) at 20 % deviation (ANSI C84.1 Range B)
        score_voltage = min(40.0, (voltage_deviation_pct / 0.20) * 40.0)

        # ── Component B: Current overload (0–35 pts) ───────────────────────
        overload_ratio = max(0.0, current - self._i_rated) / self._i_rated
        # Reaches maximum (35 pts) at 50 % above rated (thermal limit zone)
        score_current = min(35.0, (overload_ratio / 0.50) * 35.0)

        # ── Component C: Power-factor degradation (0–25 pts) ──────────────
        # Perfect score at PF ≥ 0.95; full score at PF ≤ 0.50
        pf_clamp  = max(0.50, min(0.95, power_factor))
        pf_range  = 0.95 - 0.50    # = 0.45
        pf_deficit = 0.95 - pf_clamp
        score_pf   = min(25.0, (pf_deficit / pf_range) * 25.0)

        total = math.floor(score_voltage + score_current + score_pf)
        total = max(0, min(100, total))

        breakdown = {
            "voltage_deviation_pts": round(score_voltage, 2),
            "current_overload_pts":  round(score_current, 2),
            "pf_degradation_pts":    round(score_pf, 2),
            "total_raw":             round(score_voltage + score_current + score_pf, 2),
        }

        return total, breakdown

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def engine_info(self) -> dict:
        """Return a summary dict for health-check / monitoring endpoints."""
        return {
            "tariff_inr_per_kwh":   self._tariff,
            "reading_interval_h":   self._interval,
            "nominal_voltage_V":    self._v_nom,
            "nominal_current_A":    self._i_nom,
            "nominal_pf":           self._pf_nom,
            "rated_current_A":      self._i_rated,
            "loss_multipliers":     _LOSS_MULTIPLIERS,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_engine: FinancialEngine | None = None


def get_financial_engine() -> FinancialEngine:
    """
    Return the module-level FinancialEngine singleton.
    Lazily initialised on first call; safe to import at module level.
    """
    global _engine
    if _engine is None:
        _engine = FinancialEngine()
    return _engine
