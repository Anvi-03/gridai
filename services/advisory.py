"""
GridPulse AI — Agentic Advisory Engine  (services/advisory.py)

Purpose
-------
Provides a rule-based, LLM-ready advisory layer that evaluates current grid
conditions — specifically predicted load and grid carbon intensity — against
operator-defined policy tiers and produces structured mitigation advisories.

Architecture
------------
                ┌──────────────────────────────────────────┐
                │  GET /api/v1/advisory                    │
                │                                          │
                │  1. Fetch latest carbon intensity from   │
                │     telemetry_readings (most recent row) │
                │  2. Fetch latest 24-h LSTM/Grid forecast │
                │     from forecast_snapshots              │
                │  3. Call get_grid_advisory(load, ci)     │
                │  4. Return AdvisoryResponse JSON         │
                └──────────────────────────────────────────┘

Design Decisions
----------------
• **Policy tiers as a constant dict** — `MICROGRID_POLICY_TIERS` lives here
  so both the advisory logic and any future LLM prompt builder can import the
  same ground truth without duplication.

• **Clean separation** — `get_grid_advisory()` is a pure function (no I/O)
  that is trivially unit-testable and can be swapped for a real LLM call
  later by replacing only the response-building block inside the function.

• **Dual trigger** — an advisory fires when EITHER:
    a. predicted_load_kw > LOAD_THRESHOLD_KW  (grid overload risk), OR
    b. carbon_intensity  > CARBON_THRESHOLD   (high-emission dispatch)
  This mirrors how a real grid operations team would set alert policies.

• **Status escalation** — if BOTH thresholds are breached simultaneously
  the status is escalated from "Warning" to "Critical" and the advisory
  urgency is reinforced accordingly.

• **Mock LLM interface** — `_compose_advisory_text()` is structured as a
  separate private helper so it can be replaced by a real Gemini / OpenAI
  call by passing a prompt to an LLM client.  The conditional logic currently
  inside acts as the MVP "LLM" that generates deterministic advisory text.

• **Reusability** — the `AdvisoryResult` dataclass is the single contract
  between the service layer and the API layer; the FastAPI response model
  maps directly from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

logger = logging.getLogger("gridpulse.advisory")

# ── Microgrid Policy Tiers ────────────────────────────────────────────────────
#
# Defines the operator-configured load-shedding priority order.  Tier 1 assets
# are NEVER shed; Tier 4 assets are the first to be curtailed.
#
# This constant is the single source of truth imported by the advisory engine,
# any future LLM prompt builder, and the API response.  Update here and the
# change propagates everywhere.

MICROGRID_POLICY_TIERS: dict[int, dict] = {
    1: {
        "label":       "Critical",
        "description": "Clinic, ICU, Server Rooms",
        "shed":        False,
        "note":        "Protected at all times — never curtailed under any condition.",
    },
    2: {
        "label":       "Essential",
        "description": "Security, Common Area Lighting",
        "shed":        False,
        "note":        "Curtailed only during declared grid emergencies above operator approval.",
    },
    3: {
        "label":       "Flexible",
        "description": "Residential AC, Washing Machines",
        "shed":        True,
        "note":        "Eligible for demand-response curtailment under high-load conditions.",
    },
    4: {
        "label":       "Deferrable",
        "description": "EV Charging Stations",
        "shed":        True,
        "note":        "First to be shed — high power draw, easily rescheduled to off-peak hours.",
    },
}

# ── Policy Thresholds ─────────────────────────────────────────────────────────

LOAD_THRESHOLD_KW: float = 700.0
"""
Grid load threshold in kilowatts above which shedding advisory is triggered.
At 700 kW the grid approaches its operational headroom ceiling, making
demand-response measures necessary to avoid transformer stress.
"""

CARBON_THRESHOLD: float = 500.0
"""
Carbon intensity threshold in gCO₂/kWh above which a clean-dispatch advisory
is triggered.  At > 500 gCO₂/kWh the grid is predominantly fossil-fuel driven;
deferring flexible loads to lower-carbon hours reduces scope-2 emissions.
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AdvisoryResult:
    """
    Structured result produced by `get_grid_advisory()`.

    Fields
    ------
    advisory_active       True when at least one threshold is breached and
                          an operational recommendation has been generated.
    status                "Normal" | "Warning" | "Critical"
    advisory_message      Human-readable (LLM-grade) recommendation text.
    triggered_by          Which conditions triggered this advisory (for logging
                          and API transparency).
    load_threshold_kw     The threshold that was evaluated (for reference).
    carbon_threshold      The carbon threshold that was evaluated.
    tier_actions          Per-tier recommended action ("protected" / "shed" / "monitor").
    """
    advisory_active:    bool
    status:             str
    advisory_message:   str
    triggered_by:       list[str]   = field(default_factory=list)
    load_threshold_kw:  float       = LOAD_THRESHOLD_KW
    carbon_threshold:   float       = CARBON_THRESHOLD
    tier_actions:       dict[int, str] = field(default_factory=dict)


# ── Private helpers ───────────────────────────────────────────────────────────

def _compose_advisory_text(
    predicted_load_kw: float,
    carbon_intensity:  float,
    status:            str,
    triggered_by:      list[str],
) -> str:
    """
    Generate a structured advisory recommendation string.

    MVP Implementation (Mock LLM)
    ------------------------------
    This function uses deterministic conditional logic to produce advisory text
    that mimics what an LLM would generate given the same context.  It is
    intentionally isolated so it can be replaced by a real Gemini/OpenAI call:

        prompt = build_prompt(predicted_load_kw, carbon_intensity, MICROGRID_POLICY_TIERS)
        response = gemini_client.generate(prompt)
        return response.text

    Parameters
    ----------
    predicted_load_kw : float
        Current 24-hour predicted peak load in kilowatts.
    carbon_intensity : float
        Latest grid carbon intensity reading in gCO₂/kWh.
    status : str
        "Warning" or "Critical" — drives urgency of language.
    triggered_by : list[str]
        Which thresholds fired (used to tailor the message).

    Returns
    -------
    str : Advisory recommendation text.
    """
    tier_4 = MICROGRID_POLICY_TIERS[4]
    tier_1 = MICROGRID_POLICY_TIERS[1]

    load_clause = (
        f"Predicted grid load of {predicted_load_kw:.1f} kW exceeds the "
        f"{LOAD_THRESHOLD_KW:.0f} kW operational threshold. "
        if "high_load" in triggered_by else ""
    )
    carbon_clause = (
        f"Carbon intensity at {carbon_intensity:.1f} gCO₂/kWh signals a "
        f"high-emission grid dispatch above the {CARBON_THRESHOLD:.0f} gCO₂/kWh clean-energy threshold. "
        if "high_carbon" in triggered_by else ""
    )

    if status == "Critical":
        urgency = (
            "CRITICAL GRID ADVISORY: Both load and carbon thresholds are breached simultaneously. "
            "Immediate demand-response action required. "
        )
    else:
        urgency = f"GRID ADVISORY ({status.upper()}): Threshold breach detected. Preventive action recommended. "

    action = (
        f"Recommended action: Initiate load-shedding of Tier 4 assets "
        f"({tier_4['description']}) to relieve grid stress and optimise carbon footprint. "
        f"Tier 1 assets ({tier_1['description']}) remain fully protected and must not be interrupted. "
        f"Tier 3 assets ({MICROGRID_POLICY_TIERS[3]['description']}) are eligible for "
        f"demand-response curtailment if Tier 4 shedding is insufficient. "
        f"Consider rescheduling Tier 4 loads to a low-carbon window "
        f"(typically 09:00–16:00 local time during peak solar hours) to reduce scope-2 emissions."
    )

    return urgency + load_clause + carbon_clause + action


def _build_tier_actions(status: str, triggered_by: list[str]) -> dict[int, str]:
    """
    Produce per-tier recommended actions based on advisory status.

    Returns a dict keyed by tier number → action label.
    """
    if not triggered_by:  # Normal — no action needed
        return {t: "normal_operations" for t in MICROGRID_POLICY_TIERS}

    actions: dict[int, str] = {}
    for tier_num, tier in MICROGRID_POLICY_TIERS.items():
        if tier_num == 1:
            actions[tier_num] = "protected_no_action"
        elif tier_num == 2:
            actions[tier_num] = "protected_monitor"
        elif tier_num == 3:
            # Tier 3 only shed during Critical (both thresholds breached)
            actions[tier_num] = "shed_if_tier4_insufficient" if status == "Critical" else "monitor_ready_to_curtail"
        else:  # Tier 4
            actions[tier_num] = "shed_immediately"
    return actions


# ── Public advisory function ──────────────────────────────────────────────────

def get_grid_advisory(
    predicted_load_kw: float,
    carbon_intensity: float,
) -> AdvisoryResult:
    """
    Evaluate current grid conditions and produce a structured operational advisory.

    This is a pure function (no I/O, no DB calls) — deterministic given the
    same inputs, trivially unit-testable, and safe to call from any async context.

    Trigger Logic
    -------------
    An advisory is raised when ANY of the following conditions hold:

        1. predicted_load_kw > LOAD_THRESHOLD_KW (700 kW)
           → Grid is approaching transformer headroom ceiling.
           → Shedding Tier 4 loads reduces demand before capacity breach.

        2. carbon_intensity > CARBON_THRESHOLD (500 gCO₂/kWh)
           → Grid dispatch is dominated by coal / gas generation.
           → Deferring Tier 4 flexible loads to a low-carbon window reduces
             scope-2 carbon footprint without impacting critical services.

    Status Escalation
    -----------------
        Both thresholds breached → "Critical"
        One  threshold breached  → "Warning"
        No threshold breached    → "Normal"  (advisory_active=False)

    Parameters
    ----------
    predicted_load_kw : float
        24-hour peak predicted load in kilowatts from the LSTM / Grid forecaster.
    carbon_intensity : float
        Most recent grid carbon intensity in gCO₂/kWh from the telemetry stream.

    Returns
    -------
    AdvisoryResult
        Fully structured advisory including status, human-readable message,
        triggered thresholds, and per-tier action recommendations.

    Examples
    --------
    >>> result = get_grid_advisory(predicted_load_kw=750.0, carbon_intensity=620.0)
    >>> result.status
    'Critical'
    >>> result.advisory_active
    True
    >>> result.tier_actions[4]
    'shed_immediately'
    >>> result.tier_actions[1]
    'protected_no_action'
    """
    triggered_by: list[str] = []

    load_breach   = predicted_load_kw > LOAD_THRESHOLD_KW
    carbon_breach = carbon_intensity  > CARBON_THRESHOLD

    if load_breach:
        triggered_by.append("high_load")
        logger.info(
            "Advisory trigger: load=%.1f kW exceeds threshold=%.0f kW",
            predicted_load_kw, LOAD_THRESHOLD_KW,
        )
    if carbon_breach:
        triggered_by.append("high_carbon")
        logger.info(
            "Advisory trigger: carbon_intensity=%.1f gCO₂/kWh exceeds threshold=%.0f",
            carbon_intensity, CARBON_THRESHOLD,
        )

    advisory_active = bool(triggered_by)

    # ── Determine status ──────────────────────────────────────────────────────
    if load_breach and carbon_breach:
        status = "Critical"
    elif advisory_active:
        status = "Warning"
    else:
        status = "Normal"

    # ── Compose advisory message (mock-LLM layer) ─────────────────────────────
    if advisory_active:
        message = _compose_advisory_text(
            predicted_load_kw=predicted_load_kw,
            carbon_intensity=carbon_intensity,
            status=status,
            triggered_by=triggered_by,
        )
    else:
        message = (
            f"Grid operating within normal parameters. "
            f"Predicted load: {predicted_load_kw:.1f} kW "
            f"(threshold: {LOAD_THRESHOLD_KW:.0f} kW). "
            f"Carbon intensity: {carbon_intensity:.1f} gCO₂/kWh "
            f"(threshold: {CARBON_THRESHOLD:.0f} gCO₂/kWh). "
            f"All Tier 1–4 assets running on normal operations schedule."
        )

    tier_actions = _build_tier_actions(status, triggered_by)

    logger.info(
        "Advisory computed: status=%s | load=%.1f kW | carbon=%.1f gCO₂/kWh | triggers=%s",
        status, predicted_load_kw, carbon_intensity, triggered_by,
    )

    return AdvisoryResult(
        advisory_active   = advisory_active,
        status            = status,
        advisory_message  = message,
        triggered_by      = triggered_by,
        load_threshold_kw = LOAD_THRESHOLD_KW,
        carbon_threshold  = CARBON_THRESHOLD,
        tier_actions      = tier_actions,
    )
