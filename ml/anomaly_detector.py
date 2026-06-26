"""
GridPulse AI — Anomaly Detector  (ml/anomaly_detector.py)

Strategy
--------
Two complementary layers run in sequence:

  1. **Deterministic guard-rails** — hard thresholds derived from grid engineering
     standards. Any reading that violates these is flagged immediately, regardless
     of the ML score.  This ensures we never miss catastrophic events (e.g. direct
     line-tapping, severe voltage sag) during the model's warm-up period.

  2. **Isolation Forest** — unsupervised ML model trained on a synthetic baseline
     of "normal" grid operation.  It assigns an anomaly score in [-1, +1]; values
     closer to -1 indicate outliers.  The model self-trains at startup and can be
     retrained online as more real data arrives.

Return value of `detect()`
--------------------------
    (is_anomaly: bool, confidence: float, anomaly_type: str | None)

    • confidence is in [0.0, 1.0] — higher means more certain the reading is bad.
    • anomaly_type is a short label (e.g. "voltage_sag", "line_tapping", "ml_outlier")
      or None when the reading is healthy.

Thread / async safety
---------------------
    The model is trained once at construction time and is read-only during
    inference — safe to call from multiple async worker coroutines concurrently.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("gridpulse.ml.anomaly")


# ── Anomaly type labels ───────────────────────────────────────────────────────

class AnomalyType:
    """Enumerated string constants for anomaly classification."""

    VOLTAGE_SAG       = "voltage_sag"        # voltage well below nominal band
    VOLTAGE_SWELL     = "voltage_swell"      # voltage spike above safe ceiling
    LOW_POWER_FACTOR  = "low_power_factor"   # highly inductive / capacitive load
    LINE_TAPPING      = "line_tapping"       # voltage sag + high current → theft
    ML_OUTLIER        = "ml_outlier"         # caught by Isolation Forest only


# ── Thresholds (grid-engineering constants) ───────────────────────────────────

# IEC 60038 / regional utility tolerance band: ±10 % of 230 V nominal
_VOLTAGE_NOMINAL       = 230.0   # V
_VOLTAGE_SAG_LIMIT     = 207.0   # V  (230 × 0.90) — ANSI C84.1 Range B lower
_VOLTAGE_SWELL_LIMIT   = 253.0   # V  (230 × 1.10)
_PF_HEALTHY_MIN        = 0.80    # dimensionless — industry minimum
_PF_CRITICAL_MIN       = 0.70    # below this → flag regardless of ML

# Line-tapping signature: voltage well below nominal AND current elevated
_LINE_TAP_VOLTAGE_MAX  = 210.0   # V  — abnormally low
_LINE_TAP_CURRENT_MIN  = 50.0    # A  — abnormally high for the voltage level

# Isolation Forest hyper-parameters
_IF_CONTAMINATION      = 0.05    # expected fraction of anomalies in training data
_IF_N_ESTIMATORS       = 200     # more trees → more stable scores
_IF_RANDOM_STATE       = 42

# How many synthetic normal samples to train on
_BASELINE_SAMPLES      = 5_000


# ── Detector class ────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Structured return value from AnomalyDetector.detect()."""
    is_anomaly:   bool
    confidence:   float        # [0.0, 1.0]
    anomaly_type: str | None   # None when healthy


class AnomalyDetector:
    """
    Unsupervised anomaly detector for smart-meter telemetry.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in real-world data.  Used to set the
        Isolation Forest decision threshold.  Default: 5 %.
    retrain_on_real_data : bool
        If True, calling `update_baseline(X)` will merge new real readings
        into the training set and refit the model.  Default: False (static
        baseline only — suitable for early-stage deployment).
    """

    def __init__(
        self,
        contamination: float = _IF_CONTAMINATION,
        retrain_on_real_data: bool = False,
    ) -> None:
        self._contamination       = contamination
        self._retrain_on_real     = retrain_on_real_data
        self._scaler              = StandardScaler()
        self._model: IsolationForest | None = None
        self._baseline_X: np.ndarray | None = None

        logger.info("Initialising AnomalyDetector — training on synthetic baseline …")
        t0 = time.perf_counter()
        self._train_on_synthetic_baseline()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("AnomalyDetector ready (trained in %.1f ms).", elapsed)

    # ── Synthetic baseline generation ─────────────────────────────────────────

    def _generate_baseline(self, n_samples: int = _BASELINE_SAMPLES) -> np.ndarray:
        """
        Produce a realistic synthetic dataset of *normal* grid readings.

        Each row is [voltage, current, power_factor].  The distributions are
        deliberately kept narrow (σ ≈ 5 V on voltage, σ ≈ 3 A on current) so
        that the Isolation Forest learns a tight decision boundary and is
        sensitive to deviations.
        """
        rng = np.random.default_rng(seed=42)

        voltage      = rng.normal(loc=230.0, scale=5.0,  size=n_samples)
        current      = rng.normal(loc=15.0,  scale=6.0,  size=n_samples)
        power_factor = np.clip(
            rng.normal(loc=0.93, scale=0.03, size=n_samples), 0.80, 1.0
        )

        # Clamp to physically valid ranges
        voltage      = np.clip(voltage,      0.0, 500.0)
        current      = np.clip(current,      0.0, 10_000.0)

        return np.column_stack([voltage, current, power_factor])

    def _train_on_synthetic_baseline(self) -> None:
        """Fit scaler + Isolation Forest on synthetic normal data."""
        X = self._generate_baseline()
        self._baseline_X = X
        self._fit(X)

    def _fit(self, X: np.ndarray) -> None:
        """(Re)fit the scaler and model on the provided dataset."""
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        self._model = IsolationForest(
            n_estimators=_IF_N_ESTIMATORS,
            contamination=self._contamination,
            random_state=_IF_RANDOM_STATE,
            n_jobs=-1,   # use all available CPU cores
        )
        self._model.fit(X_scaled)
        logger.debug("IsolationForest fitted on %d samples.", len(X))

    # ── Optional online retraining ─────────────────────────────────────────────

    def update_baseline(self, new_readings: list[tuple[float, float, float]]) -> None:
        """
        Merge new real-world readings into the training set and refit the model.

        Only active when `retrain_on_real_data=True`.  Useful for a periodic
        nightly job that incorporates the day's confirmed-normal readings.

        Parameters
        ----------
        new_readings : list of (voltage, current, power_factor) tuples
        """
        if not self._retrain_on_real:
            logger.debug("update_baseline() called but retrain_on_real_data=False — skipping.")
            return

        if not new_readings:
            return

        new_X = np.array(new_readings, dtype=float)
        combined = (
            np.vstack([self._baseline_X, new_X])
            if self._baseline_X is not None
            else new_X
        )
        self._baseline_X = combined
        logger.info(
            "Retraining AnomalyDetector on %d total samples (%d new).",
            len(combined), len(new_X),
        )
        self._fit(combined)

    # ── Core detection ─────────────────────────────────────────────────────────

    def detect(
        self,
        voltage:      float,
        current:      float,
        power_factor: float,
    ) -> DetectionResult:
        """
        Analyse a single meter reading for anomalies.

        Parameters
        ----------
        voltage      : RMS voltage (Volts)
        current      : RMS current (Amperes)
        power_factor : Dimensionless [0.0 – 1.0]

        Returns
        -------
        DetectionResult with:
            is_anomaly   — True if any detection layer flags this reading
            confidence   — [0.0, 1.0]; deterministic flags return 1.0
            anomaly_type — short label, or None for healthy readings
        """
        if self._model is None:
            raise RuntimeError("AnomalyDetector has not been initialised. Call _train_on_synthetic_baseline() first.")

        # ── Layer 1: Deterministic guard-rails ───────────────────────────────
        guard_result = self._check_deterministic_thresholds(voltage, current, power_factor)
        if guard_result is not None:
            return guard_result

        # ── Layer 2: Isolation Forest ────────────────────────────────────────
        return self._check_isolation_forest(voltage, current, power_factor)

    def _check_deterministic_thresholds(
        self,
        voltage:      float,
        current:      float,
        power_factor: float,
    ) -> DetectionResult | None:
        """
        Apply hard engineering thresholds.

        Returns a DetectionResult if a rule fires, else None (pass-through to ML).
        """

        # ── Line-tapping signature (check first — most dangerous) ────────────
        if voltage < _LINE_TAP_VOLTAGE_MAX and current > _LINE_TAP_CURRENT_MIN:
            logger.warning(
                "LINE TAPPING detected — voltage=%.1f V  current=%.1f A",
                voltage, current,
            )
            return DetectionResult(
                is_anomaly=True,
                confidence=1.0,
                anomaly_type=AnomalyType.LINE_TAPPING,
            )

        # ── Voltage sag ───────────────────────────────────────────────────────
        if voltage < _VOLTAGE_SAG_LIMIT:
            sag_pct = (_VOLTAGE_NOMINAL - voltage) / _VOLTAGE_NOMINAL
            confidence = min(1.0, sag_pct * 5)   # scales 0 → 1 over a 20 % sag
            logger.warning("VOLTAGE SAG — %.1f V (%.1f %% below nominal)", voltage, sag_pct * 100)
            return DetectionResult(
                is_anomaly=True,
                confidence=round(confidence, 4),
                anomaly_type=AnomalyType.VOLTAGE_SAG,
            )

        # ── Voltage swell ─────────────────────────────────────────────────────
        if voltage > _VOLTAGE_SWELL_LIMIT:
            swell_pct = (voltage - _VOLTAGE_NOMINAL) / _VOLTAGE_NOMINAL
            confidence = min(1.0, swell_pct * 5)
            logger.warning("VOLTAGE SWELL — %.1f V (%.1f %% above nominal)", voltage, swell_pct * 100)
            return DetectionResult(
                is_anomaly=True,
                confidence=round(confidence, 4),
                anomaly_type=AnomalyType.VOLTAGE_SWELL,
            )

        # ── Critical power-factor ─────────────────────────────────────────────
        if power_factor < _PF_CRITICAL_MIN:
            pf_deficit = _PF_HEALTHY_MIN - power_factor
            confidence = min(1.0, pf_deficit / 0.30)   # full confidence at PF=0.50
            logger.warning("LOW POWER FACTOR — %.3f", power_factor)
            return DetectionResult(
                is_anomaly=True,
                confidence=round(confidence, 4),
                anomaly_type=AnomalyType.LOW_POWER_FACTOR,
            )

        return None  # no deterministic rule fired

    def _check_isolation_forest(
        self,
        voltage:      float,
        current:      float,
        power_factor: float,
    ) -> DetectionResult:
        """
        Run the Isolation Forest on a single reading.

        score_samples() returns a raw anomaly score in (-∞, 0].  We normalise
        it to [0.0, 1.0] using the training-data score distribution stored on
        the fitted model.
        """
        X = np.array([[voltage, current, power_factor]], dtype=float)
        X_scaled = self._scaler.transform(X)

        # predict() returns +1 (normal) or -1 (anomaly)
        prediction = self._model.predict(X_scaled)[0]
        is_anomaly  = prediction == -1

        # score_samples() returns the anomaly score (lower = more anomalous)
        raw_score  = float(self._model.score_samples(X_scaled)[0])

        # Convert raw score to a [0, 1] confidence where 1 = certain anomaly.
        # The offset_  attribute is the threshold; scores below it are anomalies.
        threshold  = float(self._model.offset_)
        if is_anomaly:
            # How far below threshold are we?  Normalise by a ±0.15 window.
            confidence = min(1.0, max(0.0, (threshold - raw_score) / 0.15))
        else:
            confidence = 0.0

        if is_anomaly:
            logger.info(
                "ML OUTLIER detected — score=%.4f  threshold=%.4f  confidence=%.2f",
                raw_score, threshold, confidence,
            )

        return DetectionResult(
            is_anomaly=is_anomaly,
            confidence=round(confidence, 4),
            anomaly_type=AnomalyType.ML_OUTLIER if is_anomaly else None,
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def model_info(self) -> dict:
        """Return a summary dict useful for health-check / monitoring endpoints."""
        return {
            "model_type":       "IsolationForest",
            "n_estimators":     _IF_N_ESTIMATORS,
            "contamination":    self._contamination,
            "baseline_samples": len(self._baseline_X) if self._baseline_X is not None else 0,
            "retrain_enabled":  self._retrain_on_real,
            "thresholds": {
                "voltage_sag_limit_V":   _VOLTAGE_SAG_LIMIT,
                "voltage_swell_limit_V": _VOLTAGE_SWELL_LIMIT,
                "pf_critical_min":       _PF_CRITICAL_MIN,
                "line_tap_voltage_max_V":_LINE_TAP_VOLTAGE_MAX,
                "line_tap_current_min_A":_LINE_TAP_CURRENT_MIN,
            },
        }


# ── Module-level singleton (imported by analytics service) ────────────────────

_detector: AnomalyDetector | None = None


def get_detector() -> AnomalyDetector:
    """
    Return the module-level AnomalyDetector singleton.
    Lazily initialised on first call — safe to import at module level.
    """
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector
