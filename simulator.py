"""
GridPulse AI — Edge-Aware Data Simulator  (simulator.py)

Simulates N concurrent smart meters that run local edge pre-screening
BEFORE transmitting telemetry to the cloud backend.

Architecture — Edge-to-Cloud Flow
──────────────────────────────────
                 ┌─────────────────────────────────────┐
                 │  Virtual Meter (asyncio task)        │
                 │                                      │
                 │  1. Generate raw V / I / PF reading  │
                 │  2. EdgeLocalFilter.update()  ◄──── Feature 6
                 │     • Rolling Z-score on V & I       │
                 │     • Pure Python, zero cloud deps    │
                 │  3. Build JSON payload                │
                 │     + edge_flagged: bool             │
                 │     + edge_confidence: float         │
                 │  4. Async HTTP POST → FastAPI         │
                 └─────────────────────────────────────┘

Edge Pre-Screening Behaviour
─────────────────────────────
• Each virtual meter owns its own EdgeLocalFilter instance (via the
  `get_edge_filter(meter_id)` registry — no shared state between meters).
• The filter warms up silently for the first MIN_SAMPLES_BEFORE_SCREENING
  readings (default: 5) before it begins flagging anomalies.
• Approximately 1-in-8 readings is deliberately injected with a synthetic
  anomaly (voltage sag or current spike) to exercise the edge-flag path.
• All readings — normal and edge-flagged — are sent to the cloud; the
  cloud layer decides what to do with the flag (skip redundant ML baseline
  for pre-screened anomalies, etc.).

Run independently of the server:
    python simulator.py

Behaviour
─────────
• Each virtual meter runs in its own asyncio task.
• Every SIMULATOR_INTERVAL_S seconds each meter sends a batch of
  SIMULATOR_BATCH_SIZE readings in a single POST.
• Voltage is sampled from a realistic 220–240 V band with ±2 V Gaussian jitter.
• Current and power_factor fluctuate over a simulated load cycle.
• A shared Stats object tracks throughput, latency, error rates, and
  edge-flag counts, printing a live summary every 5 seconds.
• The simulator runs until interrupted with Ctrl-C.
"""
import asyncio
import logging
import math
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import settings

    TARGET_URL = settings.SIMULATOR_TARGET_URL
    NUM_METERS = settings.SIMULATOR_NUM_METERS
    INTERVAL_S = settings.SIMULATOR_INTERVAL_S
    BATCH_SIZE = settings.SIMULATOR_BATCH_SIZE
except ImportError:
    TARGET_URL = "http://localhost:8000/api/v1/telemetry"
    NUM_METERS = 20
    INTERVAL_S = 0.5
    BATCH_SIZE = 10

# ── Edge filter import ────────────────────────────────────────────────────────
from edge.local_filter import get_edge_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gridpulse.simulator")

STATS_INTERVAL_S = 5.0   # how often to print the live summary

# Fraction of readings that are deliberately injected as edge anomalies
# so the edge-flag code path is exercised during every simulator run.
ANOMALY_INJECTION_RATE = 0.125   # ~1 in 8 readings


# ── Statistics tracker ────────────────────────────────────────────────────────

@dataclass
class Stats:
    """Thread-safe (single event-loop) counters for simulator telemetry."""

    requests_sent:      int   = 0
    readings_sent:      int   = 0
    success_count:      int   = 0
    error_count:        int   = 0
    total_latency_ms:   float = 0.0
    edge_flagged_count: int   = 0    # readings where edge pre-screener fired
    _latencies:         list[float] = field(default_factory=list)
    _lock:              asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record_success(
        self, latency_ms: float, batch_size: int, edge_flagged_in_batch: int
    ) -> None:
        async with self._lock:
            self.requests_sent      += 1
            self.readings_sent      += batch_size
            self.success_count      += 1
            self.total_latency_ms   += latency_ms
            self.edge_flagged_count += edge_flagged_in_batch
            self._latencies.append(latency_ms)

    async def record_error(self, batch_size: int) -> None:
        async with self._lock:
            self.requests_sent += 1
            self.readings_sent += batch_size
            self.error_count   += 1

    def summary(self) -> str:
        total       = self.success_count + self.error_count
        success_pct = (self.success_count / total * 100) if total else 0
        avg_lat     = (
            self.total_latency_ms / self.success_count if self.success_count else 0
        )
        p95_lat     = (
            sorted(self._latencies)[int(len(self._latencies) * 0.95)]
            if self._latencies else 0
        )
        edge_pct = (
            (self.edge_flagged_count / self.readings_sent * 100)
            if self.readings_sent else 0
        )
        return (
            f"Requests: {self.requests_sent:>6} | "
            f"Readings: {self.readings_sent:>7} | "
            f"OK: {self.success_count} ({success_pct:.1f}%) | "
            f"ERR: {self.error_count} | "
            f"⚡ Edge-flagged: {self.edge_flagged_count} ({edge_pct:.1f}%) | "
            f"Avg lat: {avg_lat:>6.1f}ms | "
            f"p95 lat: {p95_lat:>6.1f}ms"
        )


stats = Stats()


# ── Realistic data generation ─────────────────────────────────────────────────

def _generate_reading(meter_id: str, tick: int) -> dict:
    """
    Produce one realistic electrical measurement.

    Voltage: 230 V nominal ± Gaussian noise (σ=2 V) + slow sinusoidal drift
             simulating grid load variation over a 60-second cycle.
    Current: 10–30 A range with random walk.
    Power factor: 0.80–1.00 correlated with load.

    Synthetic anomaly injection (ANOMALY_INJECTION_RATE):
      • Voltage sag: drops voltage to 185–205 V (below the 207 V IEC sag limit)
      • Current spike: elevates current to 55–80 A (well above normal range)
    """
    inject_anomaly = random.random() < ANOMALY_INJECTION_RATE

    if inject_anomaly:
        anomaly_kind = random.choice(["voltage_sag", "current_spike"])
        if anomaly_kind == "voltage_sag":
            voltage = round(random.uniform(185.0, 205.0), 4)
            current = round(random.uniform(10.0, 30.0),   4)
        else:  # current_spike
            voltage_drift = 3.0 * math.sin(2 * math.pi * tick / 120)
            voltage = max(210.0, min(250.0, random.gauss(230.0 + voltage_drift, 2.0)))
            voltage = round(voltage, 4)
            current = round(random.uniform(55.0, 80.0), 4)
    else:
        # Normal reading
        voltage_drift = 3.0 * math.sin(2 * math.pi * tick / 120)
        voltage = max(210.0, min(250.0, random.gauss(230.0 + voltage_drift, 2.0)))
        voltage = round(voltage, 4)
        current = round(random.uniform(10.0, 30.0), 4)

    # Power factor: correlated with current — heavy loads tend toward ~0.85
    base_pf      = 1.0 - (current - 10.0) / 80.0
    power_factor = round(
        max(0.80, min(1.00, base_pf + random.gauss(0, 0.01))),
        4,
    )

    return {
        "meter_id":     meter_id,
        "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
        "voltage":      voltage,
        "current":      current,
        "power_factor": power_factor,
    }


def _build_edge_batch(meter_id: str, tick: int) -> tuple[dict, int]:
    """
    Build a telemetry batch payload enriched with edge pre-screening results.

    Each reading passes through the meter's EdgeLocalFilter before being
    added to the payload.  If the filter raises an edge flag, the reading is
    annotated with `edge_flagged=True` and `edge_confidence=<float>`.

    Returns
    -------
    (payload_dict, edge_flagged_count)
        payload_dict        — ready-to-POST JSON body
        edge_flagged_count  — number of readings in this batch where the
                              edge filter raised a flag (for stats tracking)
    """
    edge_filter        = get_edge_filter(meter_id)
    enriched_readings  = []
    edge_flagged_count = 0

    for _ in range(BATCH_SIZE):
        raw = _generate_reading(meter_id, tick)

        # ── Edge pre-screening (pure Python, no cloud calls) ──────────────────
        result = edge_filter.update(
            voltage      = raw["voltage"],
            current      = raw["current"],
            power_factor = raw["power_factor"],
        )

        # Enrich the payload with edge metadata
        raw["edge_flagged"]    = result.edge_flagged
        raw["edge_confidence"] = result.edge_confidence

        if result.edge_flagged:
            edge_flagged_count += 1
            logger.debug(
                "[%s] ⚡ EDGE FLAGGED  V=%.1f  I=%.1f  "
                "z_v=%.2f  z_i=%.2f  conf=%.3f",
                meter_id,
                raw["voltage"],
                raw["current"],
                result.z_voltage,
                result.z_current,
                result.edge_confidence,
            )

        enriched_readings.append(raw)

    return {"readings": enriched_readings}, edge_flagged_count


# ── Meter coroutine ───────────────────────────────────────────────────────────

async def run_meter(
    meter_id:   str,
    client:     httpx.AsyncClient,
    stop_event: asyncio.Event,
) -> None:
    """
    Simulates a single edge-aware smart meter.

    Each iteration:
      1. Generates BATCH_SIZE raw readings.
      2. Runs each reading through the local EdgeLocalFilter (pure Python).
      3. Enriches the payload with edge_flagged / edge_confidence fields.
      4. POSTs the enriched batch to the cloud ingest endpoint.
    """
    tick = 0
    while not stop_event.is_set():
        payload, edge_flagged_in_batch = _build_edge_batch(meter_id, tick)
        tick += 1
        t0   = time.perf_counter()

        try:
            response   = await client.post(TARGET_URL, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000

            if response.status_code == 201:
                await stats.record_success(latency_ms, BATCH_SIZE, edge_flagged_in_batch)
                logger.debug(
                    "[%s] OK %d readings | %d edge-flagged | %d ms",
                    meter_id,
                    BATCH_SIZE,
                    edge_flagged_in_batch,
                    int(latency_ms),
                )
            else:
                await stats.record_error(BATCH_SIZE)
                logger.warning(
                    "[%s] WARN  HTTP %d: %s",
                    meter_id,
                    response.status_code,
                    response.text[:200],
                )

        except httpx.ConnectError:
            await stats.record_error(BATCH_SIZE)
            logger.error(
                "[%s] FAIL Connection refused — is the server running at %s?",
                meter_id,
                TARGET_URL,
            )
        except httpx.TimeoutException:
            await stats.record_error(BATCH_SIZE)
            logger.warning("[%s] TIMEOUT Request timed out", meter_id)
        except Exception as exc:  # noqa: BLE001
            await stats.record_error(BATCH_SIZE)
            logger.exception("[%s] Unexpected error: %s", meter_id, exc)

        # Wait before the next burst (interruptible)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=INTERVAL_S)
        except asyncio.TimeoutError:
            pass


# ── Stats reporter ────────────────────────────────────────────────────────────

async def stats_reporter(stop_event: asyncio.Event) -> None:
    """Prints a one-line summary of simulator performance every 5 seconds."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=STATS_INTERVAL_S)
        except asyncio.TimeoutError:
            pass
        print(f"\nSTATS  {stats.summary()}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    stop_event = asyncio.Event()

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(*_):
        logger.info("\nSTOP Shutdown signal received — stopping meters …")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, OSError):
            signal.signal(sig, _shutdown)

    meter_ids = [f"METER-{i:03d}" for i in range(1, NUM_METERS + 1)]

    print(
        f"\nSTART GridPulse AI Edge-Aware Simulator\n"
        f"   Meters          : {NUM_METERS}\n"
        f"   Batch sz        : {BATCH_SIZE} readings/POST\n"
        f"   Interval        : {INTERVAL_S} s\n"
        f"   Target          : {TARGET_URL}\n"
        f"   Anomaly inject  : ~{ANOMALY_INJECTION_RATE*100:.0f}% of readings\n"
        f"   Edge screening  : Rolling Z-score (window=48, threshold=3-sigma)\n"
        f"   Approx throughput: ~{int(NUM_METERS * BATCH_SIZE / INTERVAL_S)} readings/s\n"
        f"   Press Ctrl-C to stop.\n"
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(
            max_connections=NUM_METERS + 5,
            max_keepalive_connections=NUM_METERS,
        ),
    ) as client:
        tasks = [
            asyncio.create_task(run_meter(mid, client, stop_event))
            for mid in meter_ids
        ]
        tasks.append(asyncio.create_task(stats_reporter(stop_event)))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    print(f"\nOK Simulator finished.\n{stats.summary()}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
