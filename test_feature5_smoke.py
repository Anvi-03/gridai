"""
GridPulse AI — Feature 5 Integration Smoke Test
Run with: .venv\Scripts\python test_feature5_smoke.py
"""
import sys
import asyncio

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Clear all singletons for a clean test run
import config as _cfg; _cfg.get_settings.cache_clear()
import services.copilot_engine as _ce; _ce._copilot = None
import ml.load_forecaster as _lf; _lf._forecaster = None
import services.forecasting_service as _fs; _fs._service = None

import httpx
from main import app

TRANSPORT = httpx.ASGITransport(app=app)
BASE      = "http://test"
TIMEOUT   = httpx.Timeout(60.0)
SEP       = "=" * 65


async def run():
    print(SEP)
    print("Feature 5 -- Predictive Forecasting Integration Test")
    print(SEP)
    print()

    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE, timeout=TIMEOUT) as client:

        # [1] Forecast health endpoint
        print("[1] GET /api/v1/grid/forecast/health")
        r = await client.get("/api/v1/grid/forecast/health")
        print(f"    HTTP {r.status_code}")
        assert r.status_code == 200, r.text
        h = r.json()
        print(f"    status={h['status']}  snapshots={h['total_snapshots']}")
        assert "status" in h
        assert "total_snapshots" in h
        print("    [OK]")
        print()

        # [2] Forecast report endpoint
        print("[2] GET /api/v1/grid/forecast")
        r = await client.get("/api/v1/grid/forecast")
        print(f"    HTTP {r.status_code}")
        assert r.status_code == 200, r.text
        report = r.json()

        print(f"    total_meters_active       : {report['total_meters_active']}")
        fs = report["fleet_summary"]
        print(f"    fleet_summary.max_risk    : {fs['max_risk_score']}")
        print(f"    fleet_summary.avg_risk    : {fs['avg_risk_score']}")
        print(f"    fleet_summary.systemic_p  : {fs['systemic_outage_probability']}")
        print(f"    high_risk_zones count     : {len(report['high_risk_zones'])}")
        print(f"    outage_prob_matrix entries: {len(report['outage_probability_matrix'])}")

        assert "generated_at" in report
        assert "high_risk_zones" in report
        assert "outage_probability_matrix" in report
        assert "fleet_summary" in report
        assert "predicted_peak_times" in report

        # Verify each matrix entry has all required fields
        for entry in report["outage_probability_matrix"]:
            assert "meter_id" in entry
            assert "outage_risk_score" in entry
            assert "risk_zone" in entry
            assert "predicted_peak_w" in entry
            assert "load_ratio" in entry
            assert entry["risk_zone"] in ("low", "medium", "high", "critical"), \
                f"Unexpected risk_zone: {entry['risk_zone']}"

        print("    [OK]")
        print()

        # [3] Forecast with risk_zone filter
        print("[3] GET /api/v1/grid/forecast?risk_zone=low")
        r = await client.get("/api/v1/grid/forecast?risk_zone=low")
        print(f"    HTTP {r.status_code}")
        assert r.status_code == 200, r.text
        filtered = r.json()
        # All returned items must be low zone
        for item in filtered["outage_probability_matrix"]:
            assert item["risk_zone"] == "low", f"Expected low, got {item['risk_zone']}"
        print("    [OK]")
        print()

        # [4] Forecast health after sweep
        print("[4] GET /api/v1/grid/forecast/health (post-sweep)")
        r = await client.get("/api/v1/grid/forecast/health")
        print(f"    HTTP {r.status_code}")
        assert r.status_code == 200
        h2 = r.json()
        print(f"    status={h2['status']}  snapshots={h2['total_snapshots']}  "
              f"age={h2.get('latest_sweep_age_secs')}s")
        print("    [OK]")
        print()

    print(SEP)
    print("ALL FEATURE 5 INTEGRATION TESTS PASSED")
    print(SEP)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())
