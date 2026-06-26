"""
GridPulse AI — Feature 2 (ML Anomaly Engine) Verification Test Suite
This script validates:
  1. Isolation Forest Sanity Test (normal readings are not flagged)
  2. Theft Detection Test (extreme readings flag line tapping)
  3. End-to-End Async Pipeline Test (HTTP ingest triggers background update in database)
"""

import asyncio
import sys
import time
import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from config import settings
from ml.anomaly_detector import get_detector
from schemas import TelemetryReading
from main import app

# Terminal colors using ANSI sequences
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")


def print_success(msg: str):
    print(f"{GREEN}[OK] {msg}{RESET}")


def print_failure(msg: str):
    print(f"{RED}[FAIL] {msg}{RESET}")


def print_info(msg: str):
    print(f"{YELLOW}[INFO] {msg}{RESET}")


# ── Test 1: Isolation Forest Sanity Test ─────────────────────────────────────

def run_sanity_test() -> bool:
    print_header("Test 1: Isolation Forest Sanity Test")
    try:
        detector = get_detector()
        # Normal reading: 230V, 5A, 0.95 PF
        result = detector.detect(voltage=230.0, current=5.0, power_factor=0.95)
        print_info(f"Normal telemetry: 230V, 5A, 0.95 PF -> is_anomaly={result.is_anomaly}, type={result.anomaly_type}, confidence={result.confidence}")
        
        assert result.is_anomaly == False, "Expected normal reading to not be anomalous"
        print_success("Isolation Forest Sanity Test passed.")
        return True
    except Exception as e:
        print_failure(f"Isolation Forest Sanity Test failed: {e}")
        return False


# ── Test 2: Theft Detection Test (Line Tampering) ────────────────────────────

def run_theft_test() -> bool:
    print_header("Test 2: Theft Detection Test (Line Tampering)")
    try:
        detector = get_detector()
        # Severe anomaly (degraded voltage + high current + terrible PF)
        # e.g., 160V, 60A, 0.4 PF
        result = detector.detect(voltage=160.0, current=60.0, power_factor=0.4)
        print_info(f"Degraded telemetry: 160V, 60A, 0.4 PF -> is_anomaly={result.is_anomaly}, type={result.anomaly_type}, confidence={result.confidence}")
        
        assert result.is_anomaly == True, "Expected degraded reading to be flagged as anomalous"
        assert result.anomaly_type == "line_tapping", f"Expected anomaly type 'line_tapping', got '{result.anomaly_type}'"
        print_success("Theft Detection Test passed.")
        return True
    except Exception as e:
        print_failure(f"Theft Detection Test failed: {e}")
        return False


# ── Test 3: End-to-End Async Pipeline Test ────────────────────────────────────

async def run_e2e_test() -> bool:
    print_header("Test 3: End-to-End Async Pipeline Test")
    
    # Generate a unique meter ID for this test runs
    test_meter_id = f"METER-ANOMALY-{int(time.time())}"
    
    # Anomalous payload: low voltage (160V) + high current (60A) + terrible PF (0.4)
    payload = {
        "readings": [
            {
                "meter_id": test_meter_id,
                "voltage": 160.0,
                "current": 60.0,
                "power_factor": 0.4
            }
        ]
    }
    
    try:
        # Start client with ASGITransport to trigger startup / shutdown events (warms up ML)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            print_info(f"Sending anomalous batch payload for meter '{test_meter_id}' to /api/v1/telemetry ...")
            response = await client.post("/api/v1/telemetry", json=payload)
            assert response.status_code == 201, f"Expected 201 Created, got {response.status_code}"
            
            resp_data = response.json()
            ingested = resp_data.get("readings", [])
            assert len(ingested) == 1, f"Expected 1 ingested reading, got {len(ingested)}"
            reading_id = ingested[0]["id"]
            print_success(f"Payload accepted. Ingested reading ID: {reading_id}")
            
            # Wait for background task to process the batch
            print_info("Waiting 1.5 seconds for background analytics pipeline to run...")
            await asyncio.sleep(1.5)
            
            # Query DB directly to check if changes were persisted
            print_info("Connecting to database to check persisted state...")
            engine = create_async_engine(settings.DATABASE_URL)
            async with engine.connect() as conn:
                stmt = text(
                    "SELECT is_anomalous, anomaly_type, anomaly_confidence, predicted_load_24h "
                    "FROM telemetry_readings WHERE id = :rid"
                )
                db_res = await conn.execute(stmt, {"rid": reading_id})
                row = db_res.fetchone()
                
                assert row is not None, f"Telemetry reading with ID {reading_id} not found in DB!"
                
                is_anomalous, anomaly_type, confidence, predicted_load = row
                print_info(f"Database row retrieved:")
                print(f"   is_anomalous       : {is_anomalous}")
                print(f"   anomaly_type       : {anomaly_type}")
                print(f"   anomaly_confidence : {confidence}")
                print(f"   predicted_load_24h : {predicted_load} W")
                
                assert is_anomalous == True, "Database record is_anomalous flag was NOT set to True!"
                assert anomaly_type == "line_tapping", f"Expected anomaly_type 'line_tapping', got '{anomaly_type}'"
                assert confidence == 1.0, f"Expected confidence 1.0, got {confidence}"
                assert predicted_load is not None and predicted_load >= 0.0, "Expected a valid predicted load"
                
                print_success("Database fields successfully updated by background analytics pipeline!")
                
            await engine.dispose()
            return True
            
    except Exception as e:
        print_failure(f"End-to-End Async Pipeline Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Main runner ───────────────────────────────────────────────────────────────

async def main():
    print(f"\n{BOLD}{YELLOW}==================================================")
    print("      GRIDPULSE AI SYSTEM CHECK — FEATURE 2 TESTS")
    print(f"=================================================={RESET}")
    
    t1_ok = run_sanity_test()
    t2_ok = run_theft_test()
    t3_ok = await run_e2e_test()
    
    print(f"\n{BOLD}{YELLOW}==================== TEST REPORT ===================={RESET}")
    print(f"Test 1: Isolation Forest Sanity:   {GREEN+'PASS'+RESET if t1_ok else RED+'FAIL'+RESET}")
    print(f"Test 2: Theft Detection:           {GREEN+'PASS'+RESET if t2_ok else RED+'FAIL'+RESET}")
    print(f"Test 3: E2E Async Pipeline:        {GREEN+'PASS'+RESET if t3_ok else RED+'FAIL'+RESET}")
    print(f"{BOLD}{YELLOW}====================================================={RESET}")
    
    if t1_ok and t2_ok and t3_ok:
        print(f"\n{BOLD}{GREEN}[SUCCESS] All Feature 2 Verification Tests Passed!{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{BOLD}{RED}[ERROR] Some Feature 2 Verification Tests Failed!{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    # Workaround for asyncio windows event loop policy warnings
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
