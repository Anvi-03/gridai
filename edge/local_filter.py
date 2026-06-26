"""
GridPulse AI — Edge Local Filter  (edge/local_filter.py)

Purpose
-------
A lightweight, pure-Python anomaly pre-screening utility designed to run
inside simulated edge meter nodes *before* data is transmitted to the cloud.

Design Constraints — Embedded System Fidelity
----------------------------------------------
• **Zero heavy dependencies** — only Python stdlib (`math`, `collections`).
  No numpy, no pandas, no scikit-learn.  This mirrors the execution
  environment of low-power ARM Cortex-M microcontrollers.
• **Fixed-size rolling window** — memory usage is bounded by WINDOW_SIZE.
  Uses `collections.deque(maxlen=N)` for O(1) append with automatic eviction.
• **Incremental statistics** — mean and variance are updated online using
  Welford's one-pass algorithm so we never iterate over the full window
  on each new sample.  This is O(1) per sample.
• **No floating-point division guards required by caller** — all edge cases
  (empty window, zero variance, cold-start) are handled internally.

Algorithm — Rolling Z-Score Screening
--------------------------------------
For each incoming sample x, the filter maintains:
    n        — number of samples seen so far (capped at WINDOW_SIZE)
    mean     — rolling exponential mean of the window
    var      — rolling variance (Welford's method)

    z_score  = (x - mean) / std_dev         if std_dev > 0
             = 0.0                           if std_dev == 0

    is_anomaly = |z_score| >= Z_THRESHOLD

Two signals are screened independently:
    1. voltage  — sudden voltage sag / swell events
    2. current  — sudden current spikes (potential line-tapping or fault)

The composite edge flag is raised if EITHER signal is anomalous.

Edge confidence is derived from the maximum absolute z-score, normalised
to [0.0, 1.0] using a sigmoid-like mapping:
    confidence = min(1.0, max_abs_z / (Z_THRESHOLD * 2))

This keeps confidence at 0.5 exactly at the threshold and approaches 1.0
as the z-score doubles the threshold — physically meaningful for severity.

Usage
-----
    from edge.local_filter import EdgeLocalFilter

    f = EdgeLocalFilter(meter_id="METER-001")
    result = f.update(voltage=195.0, current=62.0, power_factor=0.88)
    if result.edge_flagged:
        payload["edge_flagged"]    = True
        payload["edge_confidence"] = result.edge_confidence
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

# ── Tunable constants ─────────────────────────────────────────────────────────

# Number of most-recent samples to keep in the rolling window.
# At a 30-min interval cadence: 48 samples ≈ 24 h of local history.
# On an MCU with 8 kB SRAM: 48 × 2 floats × 8 bytes = ~768 bytes — safe.
WINDOW_SIZE: int = 48

# Z-score threshold above which a sample is flagged as anomalous.
# 3.0 is the classical 3-sigma rule; lower values (e.g. 2.5) are more
# sensitive but increase false-positive rate on noisy grids.
Z_THRESHOLD: float = 3.0

# Minimum number of samples before the filter starts screening.
# Below this, the window statistics are unreliable (too few data points
# to compute a meaningful standard deviation).
MIN_SAMPLES_BEFORE_SCREENING: int = 5


# ── Return type ───────────────────────────────────────────────────────────────

class EdgeFilterResult(NamedTuple):
    """
    Immutable result from a single EdgeLocalFilter.update() call.

    Attributes
    ----------
    edge_flagged    : True if the reading is an edge-detected anomaly.
    edge_confidence : Pre-screening confidence score in [0.0, 1.0].
                      0.0 → definitely normal; 1.0 → extreme deviation.
    z_voltage       : Computed voltage z-score (signed).
    z_current       : Computed current z-score (signed).
    samples_seen    : Number of samples processed so far (useful for debug).
    """
    edge_flagged:    bool
    edge_confidence: float
    z_voltage:       float
    z_current:       float
    samples_seen:    int


# ── Welford accumulator ───────────────────────────────────────────────────────

class _WelfordAccumulator:
    """
    Online mean and variance estimator using Welford's single-pass algorithm.

    Maintains a bounded sliding window via deque.  When the window is full,
    the oldest sample is removed and the running statistics are corrected
    using a *downdate* step (subtracting the evicted value from the
    accumulator before processing the new value).

    The downdate step is an approximation: exact online deletion from
    Welford's method requires storing all values, which defeats the purpose
    on a memory-constrained device.  Instead, we use the window-mean-based
    correction which is accurate when the window is long (n ≥ 10) and
    becomes approximate for very short windows.  This is acceptable for our
    use case: grid telemetry deviates slowly, so the correction error is
    negligible compared to the anomaly magnitude we are detecting.

    Reference: Knuth, "The Art of Computer Programming", Vol. 2, §4.2.2
    """

    __slots__ = ("_window", "_n", "_mean", "_M2")

    def __init__(self, maxlen: int) -> None:
        self._window: deque[float] = deque(maxlen=maxlen)
        self._n:    int   = 0      # total samples seen (capped at maxlen)
        self._mean: float = 0.0    # running mean
        self._M2:   float = 0.0    # running sum of squared deviations

    def push(self, x: float) -> None:
        """Add a new sample, evicting the oldest if the window is full."""
        if len(self._window) == self._window.maxlen:
            # Evict oldest: apply a mean-based downdate
            evicted   = self._window[0]
            old_mean  = self._mean
            self._n  -= 1
            if self._n > 0:
                self._mean = (self._mean * (self._n + 1) - evicted) / self._n
                # Approximate M2 correction using old mean
                self._M2 = max(0.0, self._M2 - (evicted - old_mean) * (evicted - self._mean))
            else:
                self._mean = 0.0
                self._M2   = 0.0

        # Welford update with the new value
        self._window.append(x)
        self._n    += 1
        delta       = x - self._mean
        self._mean += delta / self._n
        delta2      = x - self._mean
        self._M2   += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        """Sample variance (ddof=1); returns 0.0 when n < 2."""
        return self._M2 / (self._n - 1) if self._n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def n(self) -> int:
        return self._n


# ── Main filter class ─────────────────────────────────────────────────────────

class EdgeLocalFilter:
    """
    Per-meter rolling Z-score pre-screener running on the edge node.

    Each meter instance maintains its own independent rolling windows for
    voltage and current.  Power factor is not screened separately because
    it is a derived ratio and its anomalies are already captured by the
    current/voltage signals.

    Parameters
    ----------
    meter_id         : Identifier for logging only — no network calls.
    window_size      : Rolling window depth (default: WINDOW_SIZE = 48).
    z_threshold      : Flagging threshold in standard deviations (default: 3.0).
    min_samples      : Minimum samples before screening activates (default: 5).
    """

    def __init__(
        self,
        meter_id:   str,
        window_size: int   = WINDOW_SIZE,
        z_threshold: float = Z_THRESHOLD,
        min_samples: int   = MIN_SAMPLES_BEFORE_SCREENING,
    ) -> None:
        self._meter_id    = meter_id
        self._z_threshold = z_threshold
        self._min_samples = min_samples

        self._v_acc = _WelfordAccumulator(maxlen=window_size)
        self._i_acc = _WelfordAccumulator(maxlen=window_size)

    @property
    def meter_id(self) -> str:
        return self._meter_id

    @property
    def samples_seen(self) -> int:
        """Number of samples that have been pushed into the voltage window."""
        return self._v_acc.n

    def update(
        self,
        voltage:      float,
        current:      float,
        power_factor: float,  # accepted for API symmetry; stored for future use
    ) -> EdgeFilterResult:
        """
        Process a new measurement and return the screening result.

        This method is the only public interface callers need.  It updates the
        internal rolling windows, computes Z-scores, and returns an immutable
        EdgeFilterResult.

        Parameters
        ----------
        voltage      : RMS voltage (Volts).
        current      : RMS current (Amperes).
        power_factor : Power factor [0.0 – 1.0]  (accepted, not screened).

        Returns
        -------
        EdgeFilterResult (NamedTuple) — always returned, never raises.
        """
        # Update rolling windows *before* computing Z-scores so the new sample
        # is included in the distribution (online learning).
        self._v_acc.push(voltage)
        self._i_acc.push(current)

        n = self._v_acc.n   # same as self._i_acc.n after both pushes

        # ── Cold-start guard ──────────────────────────────────────────────────
        # During warm-up we lack enough history to compute a meaningful sigma.
        # Return a neutral result: flagged=False, confidence=0.0.
        if n < self._min_samples:
            return EdgeFilterResult(
                edge_flagged    = False,
                edge_confidence = 0.0,
                z_voltage       = 0.0,
                z_current       = 0.0,
                samples_seen    = n,
            )

        # ── Z-score computation ───────────────────────────────────────────────
        z_v = self._z_score(voltage, self._v_acc)
        z_i = self._z_score(current, self._i_acc)

        # ── Anomaly decision ──────────────────────────────────────────────────
        flagged = (abs(z_v) >= self._z_threshold) or (abs(z_i) >= self._z_threshold)

        # ── Confidence mapping ────────────────────────────────────────────────
        # Normalise the maximum absolute z-score to [0, 1].
        # At threshold → 0.5; at 2× threshold → 1.0; below threshold → < 0.5.
        max_abs_z  = max(abs(z_v), abs(z_i))
        confidence = min(1.0, max_abs_z / (self._z_threshold * 2.0))

        return EdgeFilterResult(
            edge_flagged    = flagged,
            edge_confidence = round(confidence, 4),
            z_voltage       = round(z_v, 4),
            z_current       = round(z_i, 4),
            samples_seen    = n,
        )

    def reset(self) -> None:
        """
        Reset internal state.  Call this after a meter goes offline and
        reconnects so stale history doesn't contaminate the new readings.
        """
        self._v_acc = _WelfordAccumulator(maxlen=self._v_acc._window.maxlen)
        self._i_acc = _WelfordAccumulator(maxlen=self._i_acc._window.maxlen)

    def diagnostics(self) -> dict:
        """
        Return a snapshot of internal state for debugging / telemetry.
        No external I/O — safe to call at any time.
        """
        return {
            "meter_id":       self._meter_id,
            "samples_seen":   self._v_acc.n,
            "voltage_mean":   round(self._v_acc.mean,     3),
            "voltage_std":    round(self._v_acc.std,      3),
            "current_mean":   round(self._i_acc.mean,     3),
            "current_std":    round(self._i_acc.std,      3),
            "z_threshold":    self._z_threshold,
            "window_size":    self._v_acc._window.maxlen,
            "screening_live": self._v_acc.n >= self._min_samples,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _z_score(x: float, acc: _WelfordAccumulator) -> float:
        """
        Compute the Z-score of *x* against the current rolling distribution.

        Returns 0.0 when the standard deviation is zero (flat signal) to
        avoid division-by-zero errors — common on constant-load simulators
        during warm-up.
        """
        std = acc.std
        if std == 0.0:
            return 0.0
        return (x - acc.mean) / std


# ── Module-level filter registry ──────────────────────────────────────────────

_filters: dict[str, EdgeLocalFilter] = {}


def get_edge_filter(
    meter_id:    str,
    window_size: int   = WINDOW_SIZE,
    z_threshold: float = Z_THRESHOLD,
) -> EdgeLocalFilter:
    """
    Return the EdgeLocalFilter singleton for *meter_id*, creating it if needed.

    The registry is module-level and process-local — exactly correct for
    the simulator where each meter is an async task in the same process.

    Parameters
    ----------
    meter_id    : The meter identifier string.
    window_size : Only used when creating a new filter instance.
    z_threshold : Only used when creating a new filter instance.
    """
    if meter_id not in _filters:
        _filters[meter_id] = EdgeLocalFilter(
            meter_id    = meter_id,
            window_size = window_size,
            z_threshold = z_threshold,
        )
    return _filters[meter_id]
