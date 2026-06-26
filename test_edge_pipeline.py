"""
GridPulse AI — Feature 6 Edge AI Simulation Verification Test (test_edge_pipeline.py)

Performs comprehensive verification of the local filter, schema compatibility,
and end-to-end edge-enriched ingestion.

Run with:
    .venv\\Scripts\\python test_edge_pipeline.py
or:
    pytest test_edge_pipeline.py
"""

import sys
import asyncio
import pytest
import httpx
from datetime import datetime, timezone

# Force UTF-8 stdout on Windows to avoid encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Clear all config/singleton caches
import config as _cfg
_cfg.get_settings.cache_clear()

# Import the application elements
from main import app
from database import get_db_context
from schemas import TelemetryReading
from sqlalchemy import delete, select
from edge.local_filter import EdgeLocalFilter

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True, scope="module")
async def cleanup_db_engine():
    yield
    from database import engine
    await engine.dispose()


TRANSPORT = httpx.ASGITransport(app=app)
BASE_URL = "http://test"
TIMEOUT = httpx.Timeout(60.0)
TEST_METERS = ["M-OLD-01", "M-EDGE-99"]
SEPARATOR = "═" * 75


# ── 1. Pure-Python Edge Filter Isolation Test ───────────────────────────────────

def test_edge_filter_isolation():
    """
    Validation Criterion 1:
    Feed the edge filter a sequence of perfectly normal steady voltage readings.
    Assert it flags them as edge_flagged = False.
    Suddenly append a severe anomaly drop.
    Assert that the algorithm instantly flags it as edge_flagged = True with an associated confidence score.
    """
    print("\n[CRITERION 1] Running Pure-Python Edge Filter Isolation Test...")
    
    # Initialize the local filter (default window=48, threshold=3.0, min_samples=5)
    local_filter = EdgeLocalFilter(meter_id="M-TEST-FILTER")
    
    # 1. Feed 20 normal voltage readings (230.0 V, 10.0 A, 0.95 PF)
    for i in range(20):
        result = local_filter.update(voltage=230.0, current=10.0, power_factor=0.95)
        # Ensure it doesn't raise anomaly flags during warm-up or under flat-line baseline
        assert result.edge_flagged is False, (
            f"Expected edge_flagged=False on iteration {i}, got {result.edge_flagged}"
        )
        
    # 2. Suddenly append a severe anomaly drop (150.0 V)
    result_anom = local_filter.update(voltage=150.0, current=10.0, power_factor=0.95)
    
    # Assert it instantly flags and calculates a non-zero confidence score
    assert result_anom.edge_flagged is True, (
        f"Expected edge_flagged=True on voltage drop, got {result_anom.edge_flagged}"
    )
    assert result_anom.edge_confidence > 0.0, (
        f"Expected positive edge_confidence, got {result_anom.edge_confidence}"
    )
    
    print(f"   [OK] Local filter flagged anomaly: V=150.0 -> flagged={result_anom.edge_flagged}, confidence={result_anom.edge_confidence}")


# ── 2. Inbound Pydantic Schema Backward Compatibility Test ──────────────────────

@pytest.mark.asyncio
async def test_legacy_payload_compatibility():
    """
    Validation Criterion 2:
    Use httpx client to POST a legacy payload without edge details to /api/v1/telemetry.
    Assert the backend returns 201 Created (confirming backward compatibility isn't broken).
    """
    print("\n[CRITERION 2] Running Inbound Pydantic Schema Backward Compatibility Test...")
    
    # Pre-test cleanup
    async with get_db_context() as db:
        await db.execute(delete(TelemetryReading).where(TelemetryReading.meter_id.in_(TEST_METERS)))
        
    legacy_payload = {
        "readings": [
            {
                "meter_id": "M-OLD-01",
                "timestamp": "2026-06-26T12:00:00Z",
                "voltage": 230.0,
                "current": 4.5,
                "power_factor": 0.92
            }
        ]
    }
    
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE_URL, timeout=TIMEOUT) as client:
        response = await client.post("/api/v1/telemetry", json=legacy_payload)
        
    assert response.status_code == 201, (
        f"Expected 201 Created for legacy payload, got {response.status_code}: {response.text}"
    )
    
    print("   [OK] Legacy payload ingested successfully (HTTP 201)")


# ── 3. End-to-End Edge-Enriched Ingestion Test ──────────────────────────────────

@pytest.mark.asyncio
async def test_edge_enriched_ingestion():
    """
    Validation Criterion 3:
    Simulate an advanced meter node by POSTing a payload containing the new edge metadata.
    Assert the API yields a 201 Created response.
    Query the PostgreSQL database directly for M-EDGE-99 and assert that the edge_flagged
    boolean column and edge_confidence values match exactly what was transmitted.
    """
    print("\n[CRITERION 3] Running End-to-End Edge-Enriched Ingestion Test...")
    
    edge_payload = {
        "readings": [
            {
                "meter_id": "M-EDGE-99",
                "timestamp": "2026-06-26T12:05:00Z",
                "voltage": 160.0,
                "current": 12.0,
                "power_factor": 0.55,
                "edge_flagged": True,
                "edge_confidence": 0.89
            }
        ]
    }
    
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE_URL, timeout=TIMEOUT) as client:
        response = await client.post("/api/v1/telemetry", json=edge_payload)
        
    assert response.status_code == 201, (
        f"Expected 201 Created for edge-enriched payload, got {response.status_code}: {response.text}"
    )
    
    # Query database directly to verify persistence
    async with get_db_context() as db:
        stmt = select(TelemetryReading).where(TelemetryReading.meter_id == "M-EDGE-99")
        result = await db.execute(stmt)
        row = result.scalars().first()
        
        assert row is not None, "Telemetry reading for M-EDGE-99 was not found in the database"
        assert row.edge_flagged is True, f"Expected edge_flagged=True, got {row.edge_flagged}"
        assert abs(row.edge_confidence - 0.89) < 1e-4, f"Expected edge_confidence close to 0.89, got {row.edge_confidence}"
        
    print("   [OK] Edge-enriched payload ingested successfully and validated in DB.")


# ── Standalone execution block ──────────────────────────────────────────────────

async def run_standalone_verification() -> None:
    print(SEPARATOR)
    print("GridPulse AI — Feature 6 Verification Script")
    print(SEPARATOR)
    
    # Run tests
    test_edge_filter_isolation()
    await test_legacy_payload_compatibility()
    await test_edge_enriched_ingestion()
    
    # Cleanup post-test
    print("\n[CLEANUP] Removing test entries from database...")
    async with get_db_context() as db:
        await db.execute(delete(TelemetryReading).where(TelemetryReading.meter_id.in_(TEST_METERS)))
    print("   [OK] Cleanup completed.")
    
    print()
    print(SEPARATOR)
    print("ALL FEATURE 6 TESTS PASSED SUCCESSFULLY!")
    print(SEPARATOR)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_standalone_verification())
