"""
GridPulse AI — FastAPI Application Entry Point

Responsibilities:
  • App lifecycle (startup DB health-check, graceful shutdown).
  • CORS middleware configured for both dev and prod origins.
  • POST /api/v1/telemetry            — high-throughput async batch ingest.
  • GET  /api/v1/health               — liveness / readiness probe.
  • GET  /api/v1/telemetry            — paginated query of recent readings.
  • POST /api/v1/copilot/query        — GenAI Grid Copilot (Gemini Q&A).
  • GET  /api/v1/copilot/context      — live DB context snapshot (debug).
  • GET  /api/v1/copilot/health       — copilot engine configuration check.
  • GET  /api/v1/grid/forecast        — 24-hour predictive outage forecast.
  • GET  /api/v1/grid/forecast/health — forecasting subsystem health check.
  • GET  /api/v1/advisory             — Agentic Advisory Engine (load + carbon).
  • GET  /api/v1/advisory/policy      — Microgrid policy tier configuration.
  • (Feature 6) Edge-enriched payloads accepted natively via edge_flagged /
    edge_confidence fields — priority analytics path for edge-flagged reads.

All database I/O is fully async (asyncpg under the hood) so the event loop
is never blocked, even under heavy concurrent simulator traffic.
"""
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import check_database_connection, engine, get_db
from models import (
    ErrorDetail,
    HealthResponse,
    TelemetryBatchRequest,
    TelemetryBatchResponse,
    TelemetryReadingOut,
)
from schemas import Base, TelemetryReading
from services.analytics import get_analytics_service, schedule_analytics
from services.forecasting_service import get_forecasting_service, schedule_forecast_sweep
from api.v1.auth import router as auth_router
from api.v1.copilot import router as copilot_router
from api.v1.forecasting import router as forecasting_router
from api.v1.simulation import router as simulation_router
from api.v1.advisory import router as advisory_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("gridpulse.api")


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Tasks run once at startup before the first request is served:
      1. Verify the DB connection is alive.
      2. Create tables if they don't exist (dev convenience; use Alembic in prod).

    On shutdown the async engine is disposed cleanly.
    """
    logger.info("🚀 GridPulse AI starting up …")

    # Validate DB reachability
    if not await check_database_connection():
        logger.critical("Cannot reach PostgreSQL — aborting startup.")
        raise RuntimeError("Database unreachable on startup.")

    # Auto-create tables in development (idempotent).
    # In production, remove this block and rely solely on Alembic migrations.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables verified / created.")

    # Warm-start the ML singletons so the first request doesn't pay cold-start
    # latency.  The Isolation Forest trains on a 5 000-sample synthetic baseline
    # which takes ~100 ms — acceptable at startup, not at request time.
    logger.info("🤖 Warming up ML pipeline …")
    get_analytics_service()   # instantiates AnomalyDetector + Forecaster singletons
    logger.info("✅ ML pipeline ready.")

    # Warm-start the Predictive Forecasting service (Feature 5).
    # Instantiates the GridForecaster singleton and kicks off an initial sweep
    # so the /grid/forecast endpoint has data immediately after startup.
    logger.info("📈 Warming up Predictive Forecasting engine …")
    get_forecasting_service()   # initialises GridForecaster singleton
    schedule_forecast_sweep()   # fire-and-forget initial sweep
    logger.info("✅ Predictive Forecasting ready (initial sweep scheduled).")

    # Warm-start the GenAI Copilot singleton.
    # Non-fatal if GEMINI_API_KEY is missing — the copilot endpoints will return
    # HTTP 503 until the key is configured, but all other routes stay healthy.
    logger.info("🤖 Warming up GenAI Copilot …")
    try:
        from services.copilot_engine import get_copilot
        get_copilot()
        logger.info("✅ GenAI Copilot ready.")
    except ValueError as exc:
        logger.warning(
            "⚠️  GenAI Copilot not configured (%s). "
            "Set GEMINI_API_KEY in .env to enable /api/v1/copilot endpoints.",
            exc,
        )

    yield  # ← server is live and serving requests

    logger.info("🛑 GridPulse AI shutting down …")
    await engine.dispose()
    logger.info("Database connections closed.")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "High-throughput asynchronous telemetry ingestion engine "
        "for smart grid monitoring, anomaly detection, economic impact analysis, "
        "and GenAI-powered natural-language copilot."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth_router,        prefix=settings.API_PREFIX)
app.include_router(copilot_router,     prefix=settings.API_PREFIX)
app.include_router(forecasting_router, prefix=settings.API_PREFIX)
app.include_router(simulation_router,  prefix=settings.API_PREFIX)
app.include_router(advisory_router,    prefix=settings.API_PREFIX)


# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach X-Process-Time-Ms to every response for latency observability."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            error="InternalServerError",
            detail="An unexpected error occurred. Please try again.",
        ).model_dump(),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get(
    f"{settings.API_PREFIX}/health",
    response_model=HealthResponse,
    tags=["Observability"],
    summary="Liveness & readiness probe",
)
async def health_check() -> HealthResponse:
    """Returns 200 when the service and its database are reachable."""
    db_ok = await check_database_connection()
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version=settings.APP_VERSION,
        database="connected" if db_ok else "unreachable",
    )


@app.post(
    f"{settings.API_PREFIX}{settings.TELEMETRY_ENDPOINT}",
    response_model=TelemetryBatchResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Telemetry"],
    summary="Ingest a batch of meter readings",
    responses={
        422: {"model": ErrorDetail, "description": "Validation error in payload"},
        500: {"model": ErrorDetail, "description": "Database write failure"},
    },
)
async def ingest_telemetry(
    batch: TelemetryBatchRequest,
    db: AsyncSession = Depends(get_db),
) -> TelemetryBatchResponse:
    """
    Accepts 1 – MAX_BATCH_SIZE telemetry readings in a single request and
    writes them to PostgreSQL in one round-trip using `add_all`.

    The async session is injected by FastAPI's DI system; commit / rollback
    is handled automatically by the `get_db` dependency.

    Raises:
        422 — if any reading fails Pydantic validation.
        500 — if the database write fails.
    """
    now_utc = datetime.now(tz=timezone.utc)

    # Map Pydantic models → ORM objects (no DB call yet).
    # edge_flagged defaults to False and edge_confidence to None for standard
    # (non-edge) payloads — fully backward-compatible with pre-Feature-6 clients.
    orm_rows = [
        TelemetryReading(
            meter_id                  = r.meter_id,
            timestamp                 = r.timestamp or now_utc,
            voltage                   = r.voltage,
            current                   = r.current,
            power_factor              = r.power_factor,
            edge_flagged              = r.edge_flagged,
            edge_confidence           = r.edge_confidence,
            carbon_intensity_gco2_kwh = r.carbon_intensity_gco2_kwh,
        )
        for r in batch.readings
    ]

    try:
        db.add_all(orm_rows)
        await db.flush()   # flush to obtain DB-generated ids before commit
        # commit is handled by get_db after this function returns
    except Exception as exc:
        logger.error("DB write failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist telemetry readings.",
        )

    edge_count = sum(1 for r in batch.readings if r.edge_flagged)
    logger.info(
        "Ingested %d reading(s) from %d unique meter(s) [edge-flagged: %d].",
        len(orm_rows),
        len({r.meter_id for r in batch.readings}),
        edge_count,
    )

    # ── Fire-and-forget analytics + forecasting pipelines ────────────────────
    # Both tasks are scheduled BEFORE returning the HTTP 201 so the event
    # loop picks them up immediately.  The client never waits for ML results.
    schedule_analytics(orm_rows)
    schedule_forecast_sweep()   # refresh predictive snapshots after each ingest

    return TelemetryBatchResponse(
        ingested=len(orm_rows),
        readings=[TelemetryReadingOut.model_validate(row) for row in orm_rows],
    )


@app.get(
    f"{settings.API_PREFIX}{settings.TELEMETRY_ENDPOINT}",
    response_model=list[TelemetryReadingOut],
    tags=["Telemetry"],
    summary="Retrieve recent telemetry readings",
)
async def get_telemetry(
    meter_id: str | None = Query(default=None, description="Filter by meter ID"),
    limit: int = Query(default=50, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
) -> list[TelemetryReadingOut]:
    """
    Returns recent readings ordered by timestamp descending.
    Supports optional filtering by `meter_id` and standard limit/offset pagination.
    """
    stmt = select(TelemetryReading).order_by(
        TelemetryReading.timestamp.desc()  # type: ignore[attr-defined]
    )
    if meter_id:
        stmt = stmt.where(TelemetryReading.meter_id == meter_id)
    stmt = stmt.limit(limit).offset(offset)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [TelemetryReadingOut.model_validate(row) for row in rows]


@app.get(
    f"{settings.API_PREFIX}/stats",
    tags=["Telemetry"],
    summary="Aggregate statistics per meter",
)
async def get_stats(
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Returns average voltage, current, and power_factor per meter."""
    stmt = select(
        TelemetryReading.meter_id,
        func.count().label("total_readings"),
        func.avg(TelemetryReading.voltage).label("avg_voltage"),
        func.avg(TelemetryReading.current).label("avg_current"),
        func.avg(TelemetryReading.power_factor).label("avg_power_factor"),
        func.max(TelemetryReading.timestamp).label("last_seen"),
    ).group_by(TelemetryReading.meter_id).order_by(TelemetryReading.meter_id)

    result = await db.execute(stmt)
    return [
        {
            "meter_id": row.meter_id,
            "total_readings": row.total_readings,
            "avg_voltage": round(float(row.avg_voltage or 0), 3),
            "avg_current": round(float(row.avg_current or 0), 3),
            "avg_power_factor": round(float(row.avg_power_factor or 0), 4),
            "last_seen": row.last_seen,
        }
        for row in result
    ]


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug" if settings.DEBUG else "info",
    )
