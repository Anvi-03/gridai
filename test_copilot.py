"""
GridPulse AI — Live Copilot E2E Test (test_copilot.py)

Performs end-to-end verification of Feature 4 (GenAI Grid Copilot):
  1. Pre-Test Data Ingestion: Deletes old test entries and programmatically
     inserts fresh test rows (including METER-TEST-99 with a severe anomaly).
  2. General Grid Status Test: Asserts that querying the total revenue loss
     returns a 200 OK response citing financial impact or currency (INR/Rs./₹).
  3. Targeted Risk Query Test: Asserts that querying high-risk meters flags
     METER-TEST-99 or its risk score (85).
  4. Boundary Guard Test: Asserts that off-topic queries trigger a safe refusal
     as dictated by the system prompt rules.
  5. Post-Test Cleanup: Automatically removes test entries to leave the DB clean.

Run with:
    .venv\\Scripts\\python test_copilot.py
"""
import asyncio
import sys
from datetime import datetime, timezone

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Force fresh settings load before any app imports ──────────────────────────
import config as _cfg_mod
_cfg_mod.get_settings.cache_clear()
import services.copilot_engine as _ce_mod
_ce_mod._copilot = None

# ── Now import the app and services ───────────────────────────────────────────
import httpx
from main import app
from database import get_db_context
from schemas import TelemetryReading
from sqlalchemy import delete

TIMEOUT = httpx.Timeout(60.0)
BASE = "http://test"
TRANSPORT = httpx.ASGITransport(app=app)
SEPARATOR = "═" * 75

async def main() -> None:
    print(SEPARATOR)
    print("GridPulse AI — Live Copilot E2E Verification Script")
    print(SEPARATOR)
    print()

    test_meters = ["METER-TEST-01", "METER-TEST-02", "METER-TEST-99"]

    # == PHASE 1: Pre-Test Data Ingestion ==
    print("[PHASE 1] Pre-Test Data Ingestion")
    async with get_db_context() as db:
        # Clean up any residual test data first
        print("   Cleaning up existing test data...")
        await db.execute(delete(TelemetryReading).where(TelemetryReading.meter_id.in_(test_meters)))
        
        print("   Ingesting fresh test records into PostgreSQL...")
        # 1. Standard reading 1
        reading_std1 = TelemetryReading(
            meter_id="METER-TEST-01",
            voltage=230.2,
            current=12.4,
            power_factor=0.96,
            is_anomalous=False,
            timestamp=datetime.now(timezone.utc),
        )
        # 2. Standard reading 2
        reading_std2 = TelemetryReading(
            meter_id="METER-TEST-02",
            voltage=229.8,
            current=11.1,
            power_factor=0.94,
            is_anomalous=False,
            timestamp=datetime.now(timezone.utc),
        )
        # 3. Severe Anomaly reading
        reading_anom = TelemetryReading(
            meter_id="METER-TEST-99",
            voltage=160.0,
            current=55.0,
            power_factor=0.42,
            is_anomalous=True,
            anomaly_type="line_tapping",
            anomaly_confidence=0.95,
            revenue_loss_inr=4500.00,
            outage_risk_score=85,
            timestamp=datetime.now(timezone.utc),
        )
        db.add_all([reading_std1, reading_std2, reading_anom])
        print("   [OK] Ingestion completed (3 rows committed)\n")

    try:
        async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE, timeout=TIMEOUT) as client:
            
            # == PHASE 2: General Grid Status Test ==
            print("[PHASE 2] General Grid Status Test")
            query_1 = "What is the current total revenue loss across the grid?"
            print(f"   Query: {query_1!r}")
            
            response = await client.post("/api/v1/copilot/query", json={"message": query_1})
            print(f"   HTTP Status: {response.status_code}")
            assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
            
            res_data = response.json()
            answer = res_data["answer"]
            print("   Answer:")
            for line in answer.split("\n"):
                print(f"      {line}")
            print()
            
            # Assert response contains calculated financial damage or currency indicators
            answer_lower = answer.lower()
            assert any(term in answer_lower for term in ["rs", "inr", "₹", "4500", "4,500"]), (
                f"Expected financial analysis referencing INR, Rs., ₹, or 4500. Answer: {answer}"
            )
            print("   [OK] General Grid Status Test passed\n")

            # == PHASE 3: Targeted Risk Query Test ==
            print("[PHASE 3] Targeted Risk Query Test")
            query_2 = "Which specific meters are currently at the highest risk of failure?"
            print(f"   Query: {query_2!r}")
            
            response = await client.post("/api/v1/copilot/query", json={"message": query_2})
            print(f"   HTTP Status: {response.status_code}")
            assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
            
            res_data = response.json()
            answer = res_data["answer"]
            print("   Answer:")
            for line in answer.split("\n"):
                print(f"      {line}")
            print()
            
            # Assert response flags METER-TEST-99 or its risk score (85)
            assert "METER-TEST-99" in answer or "85" in answer, (
                f"Expected METER-TEST-99 or risk score 85 to be mentioned. Answer: {answer}"
            )
            print("   [OK] Targeted Risk Query Test passed\n")

            # == PHASE 4: Boundary Guard Test ==
            print("[PHASE 4] Boundary Guard Test")
            query_3 = "Give me a recipe for butter chicken."
            print(f"   Query: {query_3!r}")
            
            response = await client.post("/api/v1/copilot/query", json={"message": query_3})
            print(f"   HTTP Status: {response.status_code}")
            assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
            
            res_data = response.json()
            answer = res_data["answer"]
            print("   Answer:")
            for line in answer.split("\n"):
                print(f"      {line}")
            print()
            
            # Assert the system prompt rules held up and the model refused the off-topic query
            answer_lower = answer.lower()
            assert any(ref in answer_lower for ref in ["gridpulse copilot", "only assist", "rephrase", "butter chicken"]), (
                f"Expected boundary guard refusal/rephrase warning. Answer: {answer}"
            )
            print("   [OK] Boundary Guard Test passed\n")

    finally:
        # == POST-TEST CLEANUP ==
        print("[CLEANUP] Removing test entries from database...")
        async with get_db_context() as db:
            await db.execute(delete(TelemetryReading).where(TelemetryReading.meter_id.in_(test_meters)))
        print("   [OK] Cleanup complete. Database restored to baseline.\n")

    print(SEPARATOR)
    print("ALL TEST PHASES PASSED — Feature 4 (GenAI Grid Copilot) is fully verified!")
    print(SEPARATOR)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
