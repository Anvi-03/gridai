"""
GridPulse AI — Pydantic Validation Models
All data entering or leaving the API is validated through these models.

Design principles:
  • Strict typing + field-level validators — malformed payloads are rejected
    before they ever touch the database layer.
  • Explicit value ranges expressed as Annotated types — self-documenting and
    reflected in the auto-generated OpenAPI schema.
  • Separate Request / Response shapes — never expose internal DB fields
    (e.g. raw UUIDs) unless explicitly intended.
  • Backward-compatible edge metadata (Feature 6) — both edge_flagged and
    edge_confidence are optional in the request model so standard (non-edge)
    payloads continue to work without any changes.
"""
import uuid
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Annotated scalar types ────────────────────────────────────────────────────

Voltage = Annotated[
    float,
    Field(
        ge=0.0,
        le=500.0,
        description="RMS voltage in Volts. Residential range: 220–240 V.",
        examples=[230.5],
    ),
]

CurrentAmps = Annotated[
    float,
    Field(
        ge=0.0,
        le=10_000.0,
        description="RMS current in Amperes.",
        examples=[15.3],
    ),
]

PowerFactor = Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
        description="Dimensionless power factor. Healthy loads ≥ 0.8.",
        examples=[0.95],
    ),
]

MeterId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Alphanumeric meter identifier (hyphens & underscores allowed).",
        examples=["METER-001", "grid_node_42"],
    ),
]


# ── Single reading ────────────────────────────────────────────────────────────

class TelemetryReadingIn(BaseModel):
    """
    Pydantic model for a single meter reading arriving at the API.
    The client may optionally supply a timestamp; if omitted the server
    will use its own clock (via DB server_default).

    Edge AI fields (Feature 6)
    --------------------------
    edge_flagged    : Set to True by the edge node's local Z-score pre-screener
                      when a sudden voltage or current deviation is detected.
                      Defaults to False for standard (non-edge) payloads so
                      backward compatibility is fully preserved.
    edge_confidence : Pre-screening confidence in [0.0, 1.0].  Must be provided
                      when edge_flagged=True; ignored otherwise.
    """

    model_config = {"str_strip_whitespace": True, "frozen": True}

    meter_id: MeterId
    timestamp: datetime | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp. Defaults to server time if omitted.",
    )
    voltage: Voltage
    current: CurrentAmps
    power_factor: PowerFactor

    # ── Feature 6: Edge AI pre-screening metadata ────────────────────────────────
    # Both fields are optional so standard payloads don't need to change.
    edge_flagged: bool = Field(
        default=False,
        description=(
            "True when the edge node's local Z-score screener detected a "
            "voltage/current anomaly before cloud transmission."
        ),
    )
    edge_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Edge pre-screening confidence [0.0–1.0]. "
            "Required when edge_flagged=True; omitted for standard readings."
        ),
    )

    # ── Field-level validators ───────────────────────────────────────────────────

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | None) -> datetime | None:
        """Coerce naive datetimes to UTC; reject far-future / far-past values."""
        if v is None:
            return v
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        delta_seconds = abs((v - now).total_seconds())
        if delta_seconds > 86_400:  # reject timestamps > 24 h from now
            raise ValueError(
                f"Timestamp deviates from server time by {delta_seconds:.0f}s "
                "(max 86400 s / 24 h allowed)."
            )
        return v

    # ── Cross-field validators ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_apparent_power_sanity(self) -> "TelemetryReadingIn":
        """
        Soft sanity check: apparent power = voltage × current.
        If it exceeds 2.4 MW for a single meter we likely have corrupted data.
        """
        apparent_power_kva = (self.voltage * self.current) / 1000
        if apparent_power_kva > 2_400:
            raise ValueError(
                f"Apparent power {apparent_power_kva:.1f} kVA exceeds the "
                "sanity limit of 2 400 kVA for a single meter reading."
            )
        return self

    @model_validator(mode="after")
    def validate_edge_confidence_consistency(self) -> "TelemetryReadingIn":
        """
        Deterministic contract: if edge_flagged is True, edge_confidence must
        be supplied.  A flagged reading without a confidence score suggests
        a bug in the edge firmware / simulator and should be rejected.
        """
        if self.edge_flagged and self.edge_confidence is None:
            raise ValueError(
                "edge_confidence is required when edge_flagged=True. "
                "Provide a value in [0.0, 1.0] from the edge pre-screener."
            )
        return self


# ── Batch request body ────────────────────────────────────────────────────────

class TelemetryBatchRequest(BaseModel):
    """Envelope for a batch of readings from one or many meters."""

    model_config = {"frozen": True}

    readings: list[TelemetryReadingIn] = Field(
        min_length=1,
        description="One or more telemetry readings to ingest.",
    )

    @field_validator("readings")
    @classmethod
    def check_batch_size(
        cls, v: list[TelemetryReadingIn]
    ) -> list[TelemetryReadingIn]:
        from config import settings  # deferred import avoids circular reference

        if len(v) > settings.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch too large: {len(v)} readings received, "
                f"maximum allowed is {settings.MAX_BATCH_SIZE}."
            )
        return v


# ── Response models ───────────────────────────────────────────────────────────

class TelemetryReadingOut(BaseModel):
    """Shape returned to the client after a successful ingest."""

    id: uuid.UUID
    meter_id: str
    timestamp: datetime
    voltage: float
    current: float
    power_factor: float

    # ML analytics fields — populated asynchronously; None until analytics runs
    is_anomalous:       bool | None = Field(
        default=None,
        description="True when flagged by the ML anomaly detection pipeline.",
    )
    anomaly_type:       str | None = Field(
        default=None,
        description="Short anomaly label (voltage_sag, line_tapping, ml_outlier, …) or None.",
    )
    anomaly_confidence: float | None = Field(
        default=None,
        description="Anomaly detection confidence [0.0–1.0]; None for healthy readings.",
    )
    predicted_load_24h: float | None = Field(
        default=None,
        description="Forecasted aggregate load 24 hours from now (Watts).",
    )

    # Economic Impact fields (Feature 3) — populated only for anomalous readings
    revenue_loss_inr:   float | None = Field(
        default=None,
        description="Estimated revenue loss for this anomaly event (Indian Rupees).",
    )
    outage_risk_score:  int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Composite transformer/substation stress score [0–100]; None for normal readings.",
    )

    # Edge AI fields (Feature 6) — populated at ingest time from the edge payload
    edge_flagged: bool = Field(
        default=False,
        description="True when the edge node pre-screened this reading as anomalous.",
    )
    edge_confidence: float | None = Field(
        default=None,
        description="Edge pre-screening confidence [0.0–1.0]; None for standard readings.",
    )

    model_config = {"from_attributes": True}  # allow construction from ORM rows


class TelemetryBatchResponse(BaseModel):
    """Envelope returned for a batch ingest."""

    ingested: int = Field(description="Number of readings successfully committed.")
    readings: list[TelemetryReadingOut]


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


class ErrorDetail(BaseModel):
    """Structured error payload for 4xx / 5xx responses."""

    error: str
    detail: str | list | None = None
