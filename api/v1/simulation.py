"""
GridPulse AI — Scenario Simulation Router (api/v1/simulation.py)

Exposes:
    POST /api/v1/simulation/trigger
        Triggers a deterministic scenario simulation and returns simulated
        telemetry, target transformer, affected meters, risk, and copilot review.
"""

from typing import List, Dict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/simulation",
    tags=["Scenario Simulation"],
)

# ── Pydantic I/O models ───────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    scenario: str = Field(
        description="The scenario key to simulate: heatwave | transformer_failure | ev_surge | heavy_rain | industrial_peak"
    )

class SimulationResponse(BaseModel):
    scenario: str
    target_transformer_id: str
    affected_meter_ids: List[str]
    simulated_telemetry: Dict[str, float]
    failure_probability: int
    estimated_loss_text: str
    copilot_analysis: str

# ── Deterministic Scenario Data ───────────────────────────────────────────────

SCENARIOS = {
    "heatwave": SimulationResponse(
        scenario="heatwave",
        target_transformer_id="transformer-alpha",
        affected_meter_ids=["METER-001", "METER-003", "METER-005"],
        simulated_telemetry={"voltage": 205.0, "current": 38.2, "power_factor": 0.82},
        failure_probability=82,
        estimated_loss_text="₹1.2 Lakh",
        copilot_analysis=(
            "### 🔍 Diagnosis Report\n\n"
            "**Root Cause**\n"
            "* High ambient temperature (heatwave) causing transformer oil temperature rise\n"
            "* Increased cooling overhead leading to line loss and voltage drop\n"
            "* Thermal stress on distribution substation alpha\n"
            "* **Failure Probability:** 82%\n\n"
            "**Estimated Loss**\n"
            "* **Financial Impact:** ₹1.2 Lakh\n\n"
            "**Recommendation**\n"
            "* Reduce transformer load by 15% immediately\n"
            "* Rotate load shedding across alpha sub-feeders"
        )
    ),
    "transformer_failure": SimulationResponse(
        scenario="transformer_failure",
        target_transformer_id="transformer-beta",
        affected_meter_ids=["METER-002", "METER-004", "METER-006"],
        simulated_telemetry={"voltage": 0.0, "current": 0.0, "power_factor": 0.0},
        failure_probability=100,
        estimated_loss_text="₹8.5 Lakh",
        copilot_analysis=(
            "### 🔍 Diagnosis Report\n\n"
            "**Root Cause**\n"
            "* Complete winding breakdown on transformer beta\n"
            "* Open circuit fault resulting in zero voltage delivery\n"
            "* Local substation protection relays tripped\n"
            "* **Failure Probability:** 100%\n\n"
            "**Estimated Loss**\n"
            "* **Financial Impact:** ₹8.5 Lakh\n\n"
            "**Recommendation**\n"
            "* Dispatch emergency crew to inspect and replace transformer beta\n"
            "* Reroute critical loads to adjacent substation segments"
        )
    ),
    "ev_surge": SimulationResponse(
        scenario="ev_surge",
        target_transformer_id="transformer-alpha",
        affected_meter_ids=["METER-001", "METER-002", "METER-003"],
        simulated_telemetry={"voltage": 195.0, "current": 55.0, "power_factor": 0.72},
        failure_probability=90,
        estimated_loss_text="₹2.5 Lakh",
        copilot_analysis=(
            "### 🔍 Diagnosis Report\n\n"
            "**Root Cause**\n"
            "* Uncoordinated EV fast-charging demand spike during evening peak\n"
            "* High feeder current causing voltage sag down to 195V\n"
            "* Severe transformer core saturation stress\n"
            "* **Failure Probability:** 90%\n\n"
            "**Estimated Loss**\n"
            "* **Financial Impact:** ₹2.5 Lakh\n\n"
            "**Recommendation**\n"
            "* Implement smart EV charging curtailment in Zone A\n"
            "* Encourage EV charging offset using differential pricing incentives"
        )
    ),
    "heavy_rain": SimulationResponse(
        scenario="heavy_rain",
        target_transformer_id="transformer-beta",
        affected_meter_ids=["METER-004", "METER-006"],
        simulated_telemetry={"voltage": 215.0, "current": 22.0, "power_factor": 0.88},
        failure_probability=35,
        estimated_loss_text="₹0.4 Lakh",
        copilot_analysis=(
            "### 🔍 Diagnosis Report\n\n"
            "**Root Cause**\n"
            "* Line insulation dampness during torrential rain\n"
            "* Micro-arcing at distribution terminal poles causing minor current leakage\n"
            "* Transformer housing humidity above threshold\n"
            "* **Failure Probability:** 35%\n\n"
            "**Estimated Loss**\n"
            "* **Financial Impact:** ₹0.4 Lakh\n\n"
            "**Recommendation**\n"
            "* Deploy crews to clear wet vegetation contact from overhead lines\n"
            "* Run insulation resistance check on transformer beta"
        )
    ),
    "industrial_peak": SimulationResponse(
        scenario="industrial_peak",
        target_transformer_id="transformer-alpha",
        affected_meter_ids=["METER-005", "METER-007", "METER-009"],
        simulated_telemetry={"voltage": 202.0, "current": 48.0, "power_factor": 0.78},
        failure_probability=75,
        estimated_loss_text="₹3.2 Lakh",
        copilot_analysis=(
            "### 🔍 Diagnosis Report\n\n"
            "**Root Cause**\n"
            "* Simultaneous industrial startup loads exceeding feeder capacity\n"
            "* Voltage instability with severe inductive load harmonics\n"
            "* Power factor degradation below regulatory limit\n"
            "* **Failure Probability:** 75%\n\n"
            "**Estimated Loss**\n"
            "* **Financial Impact:** ₹3.2 Lakh\n\n"
            "**Recommendation**\n"
            "* Enforce load scheduling rules for high-demand industrial units\n"
            "* Activate capacitor banks at Substation alpha to improve power factor"
        )
    ),
}

# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/trigger",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger a simulated grid scenario",
    description="Returns simulated telemetry, target transformer, and affected meters for a selected scenario."
)
async def trigger_simulation(body: SimulationRequest) -> SimulationResponse:
    scenario_key = body.scenario.lower().replace(" ", "_")
    if scenario_key not in SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scenario: '{body.scenario}'. Choose from: heatwave, transformer_failure, ev_surge, heavy_rain, industrial_peak."
        )
    return SCENARIOS[scenario_key]
