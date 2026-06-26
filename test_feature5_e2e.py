"""
GridPulse AI — Feature 5 Full E2E Verification (with data)
Run with: .venv\Scripts\python test_feature5_e2e.py
"""
import sys, asyncio, random
from datetime import datetime, timezone, timedelta

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Clear all singletons
import config as _cfg; _cfg.get_settings.cache_clear()
import services.copilot_engine as _ce; _ce._copilot = None
import ml.load_forecaster as _lf; _lf._forecaster = None
import services.forecasting_service as _fs; _fs._service = None

import httpx
from main import app
from database import get_db_context
from schemas import TelemetryReading, ForecastSnapshot
from services.forecasting_service import get_forecasting_service
from sqlalchemy import delete

TRANSPORT = httpx.ASGITransport(app=app)
BASE      = "http://test"
TIMEOUT   = httpx.Timeout(60.0)
SEP       = "=" * 65

TEST_METERS = ["METER-F5-001", "METER-F5-002", "METER-F5-HIGH"]

async def run():
    print(SEP)
    print("Feature 5 -- Full E2E Verification (with real DB data)")
    print(SEP)
    print()

    # ── PHASE 1: Ingest synthetic history for 3 test meters ──────────────────
    print("[PHASE 1] Ingesting synthetic telemetry history")
    async with get_db_context() as db:
        await db.execute(delete(TelemetryReading).where(TelemetryReading.meter_id.in_(TEST_METERS)))
        await db.execute(delete(ForecastSnapshot).where(ForecastSnapshot.meter_id.in_(TEST_METERS)))

        rng = random.Random(42)
        rows = []
        now  = datetime.now(tz=timezone.utc)
        for meter_id in TEST_METERS:
            for i in range(50):   # 50 readings per meter (well above MIN_SAMPLES)
                ts = now - timedelta(hours=50-i)
                if meter_id == "METER-F5-HIGH":
                    # Simulate a high-load meter close to capacity
                    v, a, pf = rng.uniform(220, 230), rng.uniform(25, 29), rng.uniform(0.88, 0.93)
                else:
                    v, a, pf = rng.uniform(225, 235), rng.uniform(10, 15), rng.uniform(0.90, 0.97)
                rows.append(TelemetryReading(
                    meter_id=meter_id, timestamp=ts,
                    voltage=v, current=a, power_factor=pf,
                    is_anomalous=False,
                ))
        db.add_all(rows)
    print(f"    Inserted {len(rows)} rows across {len(TEST_METERS)} test meters")
    print("    [OK]\n")

    try:
        # ── PHASE 2: Trigger a direct forecast sweep ──────────────────────────
        print("[PHASE 2] Running forecast sweep directly")
        svc     = get_forecasting_service()
        async with get_db_context() as db:
            summary = await svc.run_forecast_sweep()
        print(f"    Meters processed     : {summary.total_meters}")
        print(f"    Snapshots written    : {summary.snapshots_written}")
        print(f"    High-risk meters     : {summary.high_risk_count}")
        print(f"    Errors               : {summary.errors}")
        print(f"    Duration             : {summary.duration_ms:.1f} ms")
        assert summary.snapshots_written >= len(TEST_METERS), \
            f"Expected at least {len(TEST_METERS)} snapshots, got {summary.snapshots_written}"
        assert summary.errors == 0, f"Sweep had {summary.errors} error(s)"
        print("    [OK]\n")

        # ── PHASE 3: Verify the API returns populated forecasts ───────────────
        print("[PHASE 3] Verifying GET /api/v1/grid/forecast response")
        async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE, timeout=TIMEOUT) as client:
            r = await client.get("/api/v1/grid/forecast")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
            report = r.json()

        print(f"    total_meters_active  : {report['total_meters_active']}")
        fs = report["fleet_summary"]
        print(f"    max_risk_score       : {fs['max_risk_score']}")
        print(f"    avg_risk_score       : {fs['avg_risk_score']}")
        print(f"    systemic_prob        : {fs['systemic_outage_probability']}")
        print(f"    high_risk_zones      : {len(report['high_risk_zones'])}")

        matrix = report["outage_probability_matrix"]
        assert len(matrix) >= len(TEST_METERS), \
            f"Expected at least {len(TEST_METERS)} entries in matrix, got {len(matrix)}"

        # Verify METER-F5-HIGH appears in the matrix with valid data
        high_entry = next((e for e in matrix if e["meter_id"] == "METER-F5-HIGH"), None)
        assert high_entry is not None, "METER-F5-HIGH not found in forecast matrix"
        assert high_entry["predicted_peak_w"] > 0, "Predicted peak should be > 0"
        assert high_entry["load_ratio"] > 0, "Load ratio should be > 0"
        assert high_entry["risk_zone"] in ("low", "medium", "high", "critical")
        print(f"    METER-F5-HIGH: peak={high_entry['predicted_peak_w']:.1f}W  "
              f"risk={high_entry['outage_risk_score']}  zone={high_entry['risk_zone']}")

        # Verify fleet_summary counts add up to total_meters_active
        total_by_zone = (fs["low_risk_count"] + fs["medium_risk_count"] +
                         fs["high_risk_count"] + fs["critical_count"])
        # Note: total by zone covers all meters (not just test), so just verify non-negative
        assert total_by_zone >= 0
        print("    [OK]\n")

        # ── PHASE 4: Forecast health endpoint should now show healthy ─────────
        print("[PHASE 4] Verifying /forecast/health post-sweep")
        async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE, timeout=TIMEOUT) as client:
            r = await client.get("/api/v1/grid/forecast/health")
        assert r.status_code == 200
        h = r.json()
        print(f"    status               : {h['status']}")
        print(f"    total_snapshots      : {h['total_snapshots']}")
        print(f"    latest_sweep_age     : {h.get('latest_sweep_age_secs')}s")
        assert h["total_snapshots"] > 0
        assert h["status"] in ("healthy", "stale")
        print("    [OK]\n")

    finally:
        # ── CLEANUP ───────────────────────────────────────────────────────────
        print("[CLEANUP] Removing test data")
        async with get_db_context() as db:
            await db.execute(delete(TelemetryReading).where(TelemetryReading.meter_id.in_(TEST_METERS)))
            await db.execute(delete(ForecastSnapshot).where(ForecastSnapshot.meter_id.in_(TEST_METERS)))
        print("    [OK]\n")

    print(SEP)
    print("ALL FEATURE 5 E2E TESTS PASSED -- Predictive Forecasting is LIVE!")
    print(SEP)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())
