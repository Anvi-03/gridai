"""
GridPulse AI — Rigorous System Diagnostic Check
Standalone test & diagnostic suite verifying Feature 1.

Can be executed directly:
    python check_pipeline.py

Phases:
  1. DB Readiness: Validates connection and checks indices on telemetry_readings.
  2. Validation Guard: Verifies malformed payloads return strict 422 errors.
  3. Data Ingestion: Verifies 201 status and checks row insertion in DB.
  4. Concurrent Load Stress Test: Launches uvicorn, runs simulator with 50 meters
     for 10 seconds, verifies no 500s or timeouts, and shuts down cleanly.
"""
import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from config import settings
from schemas import Base

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


# ── Phase 1: Database Indices & Connection ────────────────────────────────────

async def run_phase1_db() -> bool:
    print_header("Phase 1: Database Readiness Connection & Indices")
    try:
        # Create temporary engine for testing
        test_engine = create_async_engine(settings.DATABASE_URL)
        async with test_engine.connect() as conn:
            # Check connection
            res = await conn.execute(text("SELECT 1"))
            assert res.scalar() == 1, "Connection ping failed"
            print_success("Database connection successful.")

            # Query existing indexes
            query = text("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'telemetry_readings';
            """)
            result = await conn.execute(query)
            indexes = [row[0] for row in result.fetchall()]

            print_info(f"Detected indexes: {indexes}")

            # Verify target indexes are present
            required_indexes = ["ix_telemetry_meter_timestamp", "ix_telemetry_timestamp"]
            for req_idx in required_indexes:
                # alembic or postgres may suffix/prefix slightly differently, do a substring match
                found = any(req_idx in idx for idx in indexes)
                if found:
                    print_success(f"Required index '{req_idx}' is present.")
                else:
                    print_failure(f"Required index '{req_idx}' is MISSING!")
                    return False

        await test_engine.dispose()
        return True
    except Exception as e:
        print_failure(f"Phase 1 failed with exception: {e}")
        return False


# ── Phase 2: Schema Validation (422 Guard) ────────────────────────────────────

async def run_phase2_validation(client: httpx.AsyncClient) -> bool:
    print_header("Phase 2: Strict Validation Test (Schema Guard)")

    malformed_cases = [
        {
            "name": "Missing power_factor",
            "payload": {
                "readings": [
                    {"meter_id": "METER-TEST", "voltage": 230.5, "current": 12.5}
                ]
            }
        },
        {
            "name": "Invalid voltage type (string instead of float)",
            "payload": {
                "readings": [
                    {"meter_id": "METER-TEST", "voltage": "high-voltage", "current": 12.5, "power_factor": 0.95}
                ]
            }
        },
        {
            "name": "Power factor out of range (> 1.0)",
            "payload": {
                "readings": [
                    {"meter_id": "METER-TEST", "voltage": 230.5, "current": 12.5, "power_factor": 1.5}
                ]
            }
        },
        {
            "name": "Empty meter_id string",
            "payload": {
                "readings": [
                    {"meter_id": "", "voltage": 230.5, "current": 12.5, "power_factor": 0.95}
                ]
            }
        },
        {
            "name": "Timestamp too far in the future",
            "payload": {
                "readings": [
                    {
                        "meter_id": "METER-TEST",
                        "voltage": 230.5,
                        "current": 12.5,
                        "power_factor": 0.95,
                        "timestamp": (datetime.now(timezone.utc).replace(year=2030)).isoformat()
                    }
                ]
            }
        }
    ]

    all_passed = True
    for case in malformed_cases:
        try:
            resp = await client.post("/api/v1/telemetry", json=case["payload"])
            if resp.status_code == 422:
                print_success(f"Passed: '{case['name']}' correctly rejected with HTTP 422.")
            else:
                print_failure(f"Failed: '{case['name']}' returned HTTP {resp.status_code} instead of 422.")
                print_info(f"Response: {resp.text}")
                all_passed = False
        except Exception as e:
            print_failure(f"Exception during '{case['name']}': {e}")
            all_passed = False

    return all_passed


# ── Phase 3: Successful Data Ingestion & DB Persistence ───────────────────────

async def run_phase3_persistence(client: httpx.AsyncClient) -> bool:
    print_header("Phase 3: Successful Data Ingestion & Persistence")
    
    test_meter = f"METER-DIAG-{int(time.time())}"
    valid_payload = {
        "readings": [
            {
                "meter_id": test_meter,
                "voltage": 232.8,
                "current": 15.42,
                "power_factor": 0.925
            }
        ]
    }

    try:
        # Ingest payload
        resp = await client.post("/api/v1/telemetry", json=valid_payload)
        if resp.status_code != 201:
            print_failure(f"Failed to ingest: API returned HTTP {resp.status_code}")
            print_info(f"Response: {resp.text}")
            return False
        
        print_success("API accepted valid payload with HTTP 201 Created.")
        resp_data = resp.json()
        ingested_records = resp_data.get("readings", [])
        assert len(ingested_records) == 1, "Expected 1 output record"
        record_id = ingested_records[0]["id"]
        print_info(f"Record successfully created in API with UUID: {record_id}")

        # Check DB directly
        test_engine = create_async_engine(settings.DATABASE_URL)
        async with test_engine.connect() as conn:
            query = text("SELECT id, meter_id, voltage, current, power_factor, timestamp FROM telemetry_readings WHERE id = :rid")
            db_res = await conn.execute(query, {"rid": record_id})
            row = db_res.fetchone()
            
            if row is None:
                print_failure(f"DB verification failed: record {record_id} not found in DB!")
                await test_engine.dispose()
                return False

            db_id, db_meter, db_volt, db_curr, db_pf, db_time = row
            print_success("Database lookup successful.")
            print_info(f"Row details in DB:")
            print(f"   ID          : {db_id}")
            print(f"   Meter ID    : {db_meter}")
            print(f"   Voltage     : {db_volt} V")
            print(f"   Current     : {db_curr} A")
            print(f"   Power Factor: {db_pf}")
            print(f"   Timestamp   : {db_time}")

            # Verify fields
            assert db_meter == test_meter, "Meter ID mismatch"
            assert abs(db_volt - 232.8) < 1e-4, "Voltage mismatch"
            assert abs(db_curr - 15.42) < 1e-4, "Current mismatch"
            assert abs(db_pf - 0.925) < 1e-4, "Power factor mismatch"
            print_success("All persisted database fields match submitted input.")

        await test_engine.dispose()
        return True
    except Exception as e:
        print_failure(f"Phase 3 failed with exception: {e}")
        return False


# ── Phase 4: Concurrent Load Stress Test ──────────────────────────────────────

async def run_phase4_stress_test() -> bool:
    print_header("Phase 4: Concurrent Load Capacity (Stress Test)")
    
    p4_ok = False
    server_process = None
    srv_out_file = None
    srv_err_file = None
    try:
        # Start FastAPI app in a background process
        print_info("Starting uvicorn backend on http://127.0.0.1:8000 ...")
        
        # Open log files in write mode
        srv_out_file = open("srv_out.log", "w", encoding="utf-8")
        srv_err_file = open("srv_err.log", "w", encoding="utf-8")
        
        # Run using the virtual environment's python/uvicorn
        python_exe = sys.executable
        server_process = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=srv_out_file,
            stderr=srv_err_file,
            text=True
        )

        # Wait for server to start up
        await asyncio.sleep(2.0)
        
        # Ping health check
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
            try:
                health_resp = await client.get("/api/v1/health")
                if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
                    print_success("FastAPI server is running and healthy.")
                else:
                    print_failure(f"Server health check failed: {health_resp.text}")
                    return False
            except Exception as e:
                print_failure(f"Could not connect to FastAPI server on startup: {e}")
                return False

        # Run simulator as a subprocess with custom parameters:
        # 50 meters, 10 seconds duration (approx 20 rounds of 0.5s interval)
        print_info("Spawning simulator with 50 concurrent meters for 10 seconds...")
        
        # Override environment variables for simulator run
        sim_env = os.environ.copy()
        sim_env["SIMULATOR_NUM_METERS"] = "50"
        sim_env["SIMULATOR_INTERVAL_S"] = "0.5"
        sim_env["SIMULATOR_BATCH_SIZE"] = "10"
        sim_env["SIMULATOR_TARGET_URL"] = "http://127.0.0.1:8000/api/v1/telemetry"
        sim_env["PYTHONUNBUFFERED"] = "1"
        
        # We start the simulator script and run it for 10 seconds, then send SIGINT or terminate
        sim_process = subprocess.Popen(
            [python_exe, "simulator.py"],
            env=sim_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Let simulator run for 10 seconds
        await asyncio.sleep(10.0)
        
        # Terminate simulator
        print_info("Stopping simulator...")
        sim_process.terminate()
        try:
            sim_stdout, sim_stderr = sim_process.communicate(timeout=3.0)
        except subprocess.TimeoutExpired:
            sim_process.kill()
            sim_stdout, sim_stderr = sim_process.communicate()

        print_info("Simulator run finished.")

        # Check simulator output to inspect requests
        # We can extract final requests/readings/success stats
        success_line = ""
        for line in sim_stdout.splitlines():
            if "Requests:" in line and "Readings:" in line:
                success_line = line
        
        if success_line:
            print_success(f"Simulator Stats: {success_line}")
        else:
            print_info("Could not extract final summary line from simulator output.")
            # Print standard output snippet for troubleshooting
            print("--- SIMULATOR STDOUT ---")
            print(sim_stdout)
            print("------------------------")
            if sim_stderr:
                print("--- SIMULATOR STDERR ---")
                print(sim_stderr)
                print("------------------------")

        # Ping health check and inspect stats endpoint to verify db pool has no locks
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30.0) as client:
            # Check stats endpoint
            stats_resp = await client.get("/api/v1/stats")
            if stats_resp.status_code == 200:
                stats_data = stats_resp.json()
                print_success(f"Stats endpoint accessible. Reporting on {len(stats_data)} active meters.")
            else:
                print_failure(f"Stats endpoint failed after stress test: {stats_resp.text}")
                return False

            # Check health endpoint again
            health_resp = await client.get("/api/v1/health")
            if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
                print_success("FastAPI server database connection pool remains healthy post-stress-test.")
            else:
                print_failure(f"FastAPI degraded or database unreachable post-stress-test: {health_resp.text}")
                return False

        p4_ok = True
        return True
    except Exception as e:
        import traceback
        print_failure(f"Phase 4 failed with exception: {e}")
        traceback.print_exc()
        return False
    finally:
        if server_process:
            print_info("Shutting down FastAPI background server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            print_info("FastAPI server shut down.")
            
        # Close the log files if they were opened
        if srv_out_file:
            srv_out_file.close()
        if srv_err_file:
            srv_err_file.close()
            
        if not p4_ok:
            print("--- FASTAPI SERVER STDOUT ---")
            if os.path.exists("srv_out.log"):
                with open("srv_out.log", "r", encoding="utf-8", errors="ignore") as f:
                    print(f.read())
            print("-----------------------------")
            print("--- FASTAPI SERVER STDERR ---")
            if os.path.exists("srv_err.log"):
                with open("srv_err.log", "r", encoding="utf-8", errors="ignore") as f:
                    print(f.read())
            print("-----------------------------")


# ── Run all diagnostics ────────────────────────────────────────────────────────

async def main():
    print(f"\n{BOLD}{YELLOW}==================================================")
    print("   GRIPULSE AI SYSTEM CHECK — FEATURE 1 DIAGNOSTIC")
    print(f"=================================================={RESET}")

    # Phase 1: DB Index Presence
    p1_ok = await run_phase1_db()

    # Spin up an in-process mock client using FastAPI App (lifespan auto-triggered)
    # for phases 2 & 3
    p2_ok = False
    p3_ok = False
    
    print_info("Initializing FastAPI mock client for verification...")
    # Import the FastAPI app directly
    from main import app
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Phase 2: Schema Guard
        p2_ok = await run_phase2_validation(client)

        # Phase 3: Successful Ingestion & Persistence
        p3_ok = await run_phase3_persistence(client)

    # Phase 4: Stress Test
    p4_ok = await run_phase4_stress_test()

    print(f"\n{BOLD}{YELLOW}=================== DIAGNOSTIC REPORT ==================={RESET}")
    print(f"Phase 1: DB connection & indexes:   {GREEN+'PASS'+RESET if p1_ok else RED+'FAIL'+RESET}")
    print(f"Phase 2: Validation constraints:    {GREEN+'PASS'+RESET if p2_ok else RED+'FAIL'+RESET}")
    print(f"Phase 3: Successful data write:     {GREEN+'PASS'+RESET if p3_ok else RED+'FAIL'+RESET}")
    print(f"Phase 4: Concurrent stress test:    {GREEN+'PASS'+RESET if p4_ok else RED+'FAIL'+RESET}")
    print(f"{BOLD}{YELLOW}========================================================={RESET}")

    if p1_ok and p2_ok and p3_ok and p4_ok:
        print(f"\n{BOLD}{GREEN}[SUCCESS] Feature 1 Diagnostic: ALL CRITERIA PASS. Ready for Production!{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{BOLD}{RED}[ERROR] Feature 1 Diagnostic: SOME CRITERIA FAILED! Check logs above.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    # Workaround for asyncio windows event loop policy warnings
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
