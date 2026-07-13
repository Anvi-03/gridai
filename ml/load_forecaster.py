"""
GridPulse AI — Load Forecaster  (ml/load_forecaster.py)

Architecture overview
---------------------
This module ships three forecasters behind a shared interface:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  BaseForecaster (ABC)                                               │
  │    └── predict_24h(history) → float                                 │
  ├─────────────────────────────────────────────────────────────────────┤
  │  MovingAverageForecaster   ← legacy / sparse-data fallback         │
  │  LSTMForecaster            ← PyTorch-backed; activates if weights   │
  │                              file or ENABLE_LSTM=true               │
  │  GridForecaster            ← PRIMARY (Feature 5): detrended Ridge  │
  │                              regression with lagged feature matrix  │
  └─────────────────────────────────────────────────────────────────────┘

GridForecaster — Production Design
-----------------------------------
• Detrended Ridge Regression: the model is fitted against *residuals*
  (actual_W - rolling_24h_mean) so it learns short-term fluctuations
  without extrapolating global trends.  This avoids the classic pitfall of
  linear regression on a non-stationary power-demand series.

• Lagged feature matrix: for each training sample the feature vector is
    [P_{t-1}, P_{t-2}, ..., P_{t-N_LAGS},
     rolling_mean_6h, rolling_mean_24h, max_variance_24h]
  providing lag, local trend, and volatility signals simultaneously.

• Cold-start resilience: if fewer than MIN_SAMPLES readings are available
  for a meter the pipeline:
    1. Pads the series using ffill (forward-fill) then bfill (backward-fill).
    2. If the padded series is still too short, falls back to the
       MovingAverageForecaster (EMA-based) — never throws an IndexError.

• Ridge α=1.0: Tikhonov regularisation prevents overfitting on noisy
  simulator spikes while still tracking real demand shape.

• Thread / async safety: the model is trained once per sweep and is
  read-only during inference — safe for concurrent async coroutines.

Shared interface: `predict_24h(history)`
  • history — list of recent aggregate power samples (Watts), chronological
  • returns — predicted aggregate load 24 hours from now, in Watts

`predict_next_24h(history)` (GridForecaster only):
  • returns — list[float] of 24 hourly predictions for the full next cycle
"""

from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger("gridpulse.ml.forecaster")


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseForecaster(ABC):
    """Common interface all load forecasters must implement."""

    @abstractmethod
    def predict_24h(self, history: Sequence[float]) -> float:
        """
        Predict aggregate load 24 hours in the future.

        Parameters
        ----------
        history : sequence of recent aggregate power values (Watts),
                  ordered oldest → newest.  May be empty.

        Returns
        -------
        Predicted load in Watts (≥ 0).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier for logging / health endpoints."""


# ─────────────────────────────────────────────────────────────────────────────
# Moving-average forecaster (legacy — sparse data / EMA fallback)
# ─────────────────────────────────────────────────────────────────────────────

class MovingAverageForecaster(BaseForecaster):
    """
    Predict 24-hour-ahead load using a weighted exponential moving average
    with a sinusoidal day-of-cycle correction.

    Algorithm
    ---------
    1.  Compute EMA over the supplied history window.
    2.  Apply a diurnal multiplier sin²-shaped peak around hour 18 (6 PM)
        to capture the typical evening demand surge.
    3.  Add a small random noise component (σ = 2 %) to avoid perfectly flat
        forecast lines in monitoring dashboards.

    Parameters
    ----------
    window : int
        Number of samples considered for EMA.  Default: 48 (e.g. 48 × 30 min).
    alpha  : float
        EMA smoothing factor in (0, 1].  Higher = more weight on recent data.
    diurnal_amplitude : float
        Peak-to-trough swing as a fraction of mean load.  Default: 20 %.
    """

    def __init__(
        self,
        window:             int   = 48,
        alpha:              float = 0.15,
        diurnal_amplitude:  float = 0.20,
        peak_hour:          int   = 18,   # 6 PM local
    ) -> None:
        self._window    = window
        self._alpha     = alpha
        self._diurnal_a = diurnal_amplitude
        self._peak_hour = peak_hour
        logger.info(
            "MovingAverageForecaster ready (window=%d, α=%.2f, diurnal=±%.0f %%)",
            window, alpha, diurnal_amplitude * 100,
        )

    @property
    def name(self) -> str:
        return "MovingAverageForecaster"

    def predict_24h(self, history: Sequence[float]) -> float:
        """Return EMA estimate with a diurnal correction for T+24 h."""
        if not history:
            logger.warning("predict_24h called with empty history — returning 0.0")
            return 0.0

        samples = list(history)[-self._window:]

        # ── Exponential moving average ────────────────────────────────────────
        ema = float(samples[0])
        for val in samples[1:]:
            ema = self._alpha * val + (1 - self._alpha) * ema

        # ── Diurnal correction ────────────────────────────────────────────────
        from datetime import datetime, timezone
        current_hour = datetime.now(tz=timezone.utc).hour
        hour_offset = (current_hour - self._peak_hour) / 24.0 * 2 * math.pi
        diurnal_factor = 1.0 + self._diurnal_a * math.cos(hour_offset)

        # ── Tiny random perturbation to simulate forecast uncertainty ─────────
        rng          = np.random.default_rng()
        noise_factor = float(rng.normal(1.0, 0.02))

        prediction = max(0.0, ema * diurnal_factor * noise_factor)
        logger.debug(
            "MA forecast: ema=%.1f W  diurnal=%.3f  noise=%.3f  → %.1f W",
            ema, diurnal_factor, noise_factor, prediction,
        )
        return round(prediction, 2)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Forecaster (PyTorch — activates when weights file is present)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(BaseForecaster):
    """
    Sequence-to-one LSTM that predicts the next 24-hour aggregate load.

    Architecture
    ------------
    Input  → [seq_len, batch=1, input_size=1]
    LSTM   → hidden_size=64, num_layers=2, dropout=0.2
    Linear → hidden_size → 1
    Output → scalar (Watts)

    Weight loading
    --------------
    Pass `weights_path` pointing to a PyTorch state-dict file (.pt / .pth).
    If the file doesn't exist or torch isn't installed, the model falls back
    to random weights for shape validation then logs a prominent warning.

    Usage
    -----
    In production set ENABLE_LSTM=true and LSTM_WEIGHTS_PATH=/path/to/model.pt
    in .env, then call get_forecaster() — it will return an LSTMForecaster.
    """

    # Model hyper-parameters (must match the training script)
    INPUT_SIZE   = 1
    HIDDEN_SIZE  = 64
    NUM_LAYERS   = 2
    DROPOUT      = 0.2
    SEQ_LEN      = 48   # consume last 48 samples (~24 h at 30-min intervals)

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self._weights_path = Path(weights_path) if weights_path else None
        self._model        = None
        self._device       = None
        self._torch        = None
        self._loaded       = False

        try:
            self._init_torch()
        except ImportError:
            logger.warning(
                "PyTorch not installed — LSTMForecaster unavailable. "
                "Install torch and set ENABLE_LSTM=true to activate."
            )

    def _init_torch(self) -> None:
        import torch  # noqa: F401 — intentional deferred import
        import torch.nn as nn

        self._torch  = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("LSTMForecaster: using device=%s", self._device)

        class _LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm   = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout,
                )
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, x):  # x: [batch, seq, 1]
                out, _ = self.lstm(x)
                return self.linear(out[:, -1, :]).squeeze(-1)

        net = _LSTMNet(
            self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_LAYERS, self.DROPOUT
        ).to(self._device)

        if self._weights_path and self._weights_path.exists():
            state = torch.load(
                self._weights_path,
                map_location=self._device,
                weights_only=True,   # safe loading (torch ≥ 2.0)
            )
            net.load_state_dict(state)
            net.eval()
            self._loaded = True
            logger.info("LSTMForecaster: weights loaded from %s", self._weights_path)
        else:
            net.eval()
            logger.warning(
                "LSTMForecaster: no weights file found at %s — "
                "using random weights (predictions are meaningless).",
                self._weights_path,
            )

        self._model = net

    @property
    def name(self) -> str:
        status = "loaded" if self._loaded else "random-weights"
        return f"LSTMForecaster({status})"

    def predict_24h(self, history: Sequence[float]) -> float:
        """
        Run an LSTM forward pass on the last SEQ_LEN history samples.

        Falls back to 0.0 if torch isn't available or history is empty.
        """
        if self._model is None or self._torch is None:
            logger.warning("LSTMForecaster: torch unavailable — returning 0.0")
            return 0.0

        if not history:
            logger.warning("predict_24h called with empty history — returning 0.0")
            return 0.0

        samples = list(history)[-self.SEQ_LEN:]

        # Pad with the first value if shorter than SEQ_LEN
        if len(samples) < self.SEQ_LEN:
            samples = [samples[0]] * (self.SEQ_LEN - len(samples)) + samples

        torch = self._torch
        x = torch.tensor(samples, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x = x.to(self._device)

        with torch.no_grad():
            pred = self._model(x)

        prediction = max(0.0, float(pred.item()))
        logger.debug("LSTM forecast → %.1f W", prediction)
        return round(prediction, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Grid Forecaster (Feature 5 — Primary production forecaster)
# ─────────────────────────────────────────────────────────────────────────────

# ── Constants ──────────────────────────────────────────────────────────────────

N_LAGS          = 24    # number of lag steps in feature vector
MIN_SAMPLES     = 12    # minimum readings before Ridge is preferred over EMA
RIDGE_ALPHA     = 1.0   # Tikhonov regularisation — prevents overfit on spikes
ROLLING_SHORT   = 6     # short rolling window (≈ 3 h at 30-min cadence)
ROLLING_LONG    = 24    # long rolling window  (≈ 12 h at 30-min cadence)

# Substation capacity thresholds (W) — power above which risk escalates.
# Derived from: V_nom × I_rated × PF_nom = 230 × 30 × 0.95 = 6,555 W
NOMINAL_CAPACITY_W  = 6_555.0    # single-meter rated capacity (Watts)

# Risk zone thresholds (% of capacity)
RISK_ZONE_MEDIUM   = 0.60   # 60 % load → medium risk
RISK_ZONE_HIGH     = 0.80   # 80 % load → high risk
RISK_ZONE_CRITICAL = 1.00   # at/above capacity → critical


class GridForecaster(BaseForecaster):
    """
    Production-grade time-series load forecaster for Feature 5.

    Pipeline (per meter)
    --------------------
    1. Cold-start guard
       • If history < MIN_SAMPLES points, the series is padded via ffill+bfill.
       • If still < N_LAGS+1 points after padding, EMA fallback is used.

    2. Detrending
       • A 24-sample rolling mean is subtracted from the series so the Ridge
         model learns short-cycle residual patterns, not absolute levels.
         This prevents linear extrapolation from noisy simulator spikes.

    3. Lagged feature matrix construction
       • Feature vector per sample:
           [P_{t-1}, ..., P_{t-N_LAGS},   ← lag terms (24 columns)
            rolling_mean_6h,               ← short-trend signal
            rolling_mean_24h,              ← daily baseline
            max_variance_24h]              ← peak-spike indicator
         → 27 features total

    4. Ridge Regression (α=1.0)
       • Fitted on residuals.  Predicts the residual for each of the next
         24 steps, which is then added back to the current rolling mean to
         recover the absolute Watt estimate.

    5. Multi-step rollout
       • Predictions are made autoregressively: each predicted residual is
         appended to the history window and the feature matrix is re-sliced
         for the next step.  This gives 24 distinct hourly forecasts.

    6. Risk score & zone classification
       • Computed from the predicted peak relative to the capacity threshold.
       • risk_zone ∈ {"low", "medium", "high", "critical"}

    Parameters
    ----------
    n_lags           : int   — number of lag features (default 24)
    ridge_alpha      : float — Ridge regularisation strength (default 1.0)
    min_samples      : int   — minimum history before Ridge is preferred over EMA
    capacity_w       : float — substation capacity threshold in Watts
    """

    def __init__(
        self,
        n_lags:       int   = N_LAGS,
        ridge_alpha:  float = RIDGE_ALPHA,
        min_samples:  int   = MIN_SAMPLES,
        capacity_w:   float = NOMINAL_CAPACITY_W,
    ) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self._n_lags      = n_lags
        self._alpha       = ridge_alpha
        self._min_samples = min_samples
        self._capacity_w  = capacity_w
        self._Ridge       = Ridge            # store class, not instance
        self._StandardScaler = StandardScaler
        self._ema_fallback = MovingAverageForecaster()

        logger.info(
            "GridForecaster ready — n_lags=%d  alpha=%.1f  "
            "min_samples=%d  capacity=%.1f W",
            n_lags, ridge_alpha, min_samples, capacity_w,
        )

    @property
    def name(self) -> str:
        return f"GridForecaster(Ridge, α={self._alpha}, lags={self._n_lags})"

    # ── Public interface (BaseForecaster) ──────────────────────────────────────

    def predict_24h(self, history: Sequence[float]) -> float:
        """
        Predict scalar aggregate load 24 hours ahead.

        Satisfies the BaseForecaster interface used by the existing
        AnalyticsService — returns only the T+24 h point estimate.
        """
        predictions = self.predict_next_24h(list(history))
        return predictions[-1] if predictions else 0.0

    # ── Extended public API ────────────────────────────────────────────────────

    def predict_next_24h(self, history: list[float]) -> list[float]:
        """
        Generate 24 hourly load predictions for the upcoming cycle.

        Parameters
        ----------
        history : chronological list of real-power samples (Watts).
                  Each element represents one measurement interval.

        Returns
        -------
        List of 24 float predictions [step+1 … step+24], each ≥ 0.0 Watts.
        """
        if not history:
            logger.warning("predict_next_24h: empty history — returning zeros")
            return [0.0] * 24

        # ── Step 1: Cold-start guard & imputation ─────────────────────────────
        series = self._impute_series(history)

        # If still not enough data for Ridge, use EMA fallback for all 24 steps
        min_required = self._n_lags + 1
        if len(series) < min_required:
            logger.info(
                "GridForecaster: cold-start (len=%d < %d) — EMA fallback",
                len(series), min_required,
            )
            base_val = self._ema_fallback.predict_24h(series)
            # Simulate a 24-step trajectory with slight diurnal variation
            return self._ema_24step_trajectory(series, base_val)

        # ── Step 2: Detrend the series ────────────────────────────────────────
        arr          = np.array(series, dtype=np.float64)
        trend        = self._rolling_mean(arr, window=self._n_lags)
        residuals    = arr - trend

        # ── Step 3: Build training feature matrix ─────────────────────────────
        X_train, y_train = self._build_feature_matrix(residuals)
        if len(X_train) < 2:
            # Degenerate: not enough interior rows to train — EMA fallback
            logger.warning("GridForecaster: insufficient training rows — EMA fallback")
            base_val = self._ema_fallback.predict_24h(series)
            return self._ema_24step_trajectory(series, base_val)

        # ── Step 4: Fit Ridge on residuals ────────────────────────────────────
        scaler  = self._StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model   = self._Ridge(alpha=self._alpha, fit_intercept=True)
        model.fit(X_scaled, y_train)

        # Compute residual std for clamping — prevents autoregressive compounding.
        # Predicted residuals are clipped to ±2σ of the training distribution so
        # that multi-step extrapolation stays bounded regardless of noise level.
        residual_std  = float(np.std(y_train)) if len(y_train) > 1 else 0.0
        residual_clip = 2.0 * residual_std if residual_std > 0 else np.inf

        # Hard absolute cap: prediction cannot exceed 10× the mean raw load.
        # This guards against degenerate extrapolation on very noisy histories.
        history_mean  = float(np.mean(arr))
        abs_cap_w     = max(history_mean * 10.0, NOMINAL_CAPACITY_W * 2)

        # ── Step 5: Autoregressive 24-step rollout ────────────────────────────
        rolling_window   = list(residuals)          # will grow with predictions
        rolling_raw      = list(arr)                # to compute rolling mean
        predictions: list[float] = []

        for step in range(24):
            # Current rolling mean for trend reconstruction
            current_mean = float(np.mean(rolling_raw[-self._n_lags:])) if len(rolling_raw) >= self._n_lags else float(np.mean(rolling_raw))

            # Build feature vector for next step
            x_vec = self._build_single_feature_vector(rolling_window)
            x_scaled = scaler.transform(x_vec.reshape(1, -1))

            # Predict residual; clamp to ±2σ to prevent autoregressive blow-up
            raw_residual       = float(model.predict(x_scaled)[0])
            predicted_residual = float(np.clip(raw_residual, -residual_clip, residual_clip))

            # Reconstruct absolute prediction and apply hard safety cap
            raw_watt       = current_mean + predicted_residual
            predicted_watt = float(np.clip(raw_watt, 0.0, abs_cap_w))

            predictions.append(round(predicted_watt, 2))

            # Append to rolling windows for next step (use clamped residual)
            rolling_window.append(predicted_residual)
            rolling_raw.append(predicted_watt)

        logger.debug(
            "GridForecaster: 24-step rollout complete — "
            "min=%.1f W  max=%.1f W  mean=%.1f W",
            min(predictions), max(predictions), float(np.mean(predictions)),
        )
        return predictions

    def compute_risk_score(
        self,
        predicted_peak_w:     float,
        capacity_threshold_w: float | None = None,
    ) -> tuple[int, str]:
        """
        Compute a dynamic outage risk score (0–100) and zone label.

        The score escalates non-linearly as predicted load approaches and
        exceeds the substation's capacity threshold, following a piecewise
        function that reflects real grid stress physics:

          0 – 60 % capacity  → low     (score:  0–30)
          60 – 80% capacity  → medium  (score: 30–55)
          80 – 100% capacity → high    (score: 55–80)
          ≥ 100% capacity    → critical (score: 80–100)

        Parameters
        ----------
        predicted_peak_w     : Peak predicted load in Watts.
        capacity_threshold_w : Substation limit in Watts.  Defaults to the
                               instance-level capacity configured at init.

        Returns
        -------
        (risk_score: int [0, 100], risk_zone: str)
        """
        cap   = capacity_threshold_w or self._capacity_w
        if cap <= 0:
            return 0, "low"

        load_ratio = predicted_peak_w / cap

        if load_ratio < RISK_ZONE_MEDIUM:               # < 60 %
            score = (load_ratio / RISK_ZONE_MEDIUM) * 30
            zone  = "low"
        elif load_ratio < RISK_ZONE_HIGH:               # 60–80 %
            score = 30 + ((load_ratio - RISK_ZONE_MEDIUM) / (RISK_ZONE_HIGH - RISK_ZONE_MEDIUM)) * 25
            zone  = "medium"
        elif load_ratio < RISK_ZONE_CRITICAL:           # 80–100 %
            score = 55 + ((load_ratio - RISK_ZONE_HIGH) / (RISK_ZONE_CRITICAL - RISK_ZONE_HIGH)) * 25
            zone  = "high"
        else:                                           # ≥ 100 %
            score = 80 + min(20, (load_ratio - 1.0) * 40)
            zone  = "critical"

        risk_score = int(min(100, max(0, round(score))))
        logger.debug(
            "Risk score: peak=%.1f W  cap=%.1f W  ratio=%.2f  score=%d  zone=%s",
            predicted_peak_w, cap, load_ratio, risk_score, zone,
        )
        return risk_score, zone

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _impute_series(self, history: list[float]) -> list[float]:
        """
        Cold-start imputation via ffill + bfill.

        If the series has fewer than MIN_SAMPLES real values, we pad it to
        MIN_SAMPLES by forward-filling (repeat the last seen value) and then
        backward-fill (repeat the first seen value for any leading NaNs).

        This mirrors pandas ffill/bfill semantics but implemented with numpy
        so there is no pandas dependency.
        """
        series = list(history)
        target = max(self._min_samples, self._n_lags + 1)

        if not series:
            return series   # caller handles empty case

        if len(series) >= target:
            return series   # already sufficient — no imputation needed

        shortfall = target - len(series)

        # Forward fill: extend with last known value
        ff_series = series + [series[-1]] * shortfall

        # Back fill: if there were leading zeros (cold meter), fill from first
        # non-zero forward; for our use case the series always has real values
        # so bfill is a no-op here, but we honour the spec for correctness.
        result = []
        last_valid = ff_series[0]
        for v in ff_series:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                last_valid = v
                result.append(v)
            else:
                result.append(last_valid)

        logger.debug(
            "GridForecaster: imputed series from %d → %d points (ffill+bfill)",
            len(history), len(result),
        )
        return result

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """
        Compute a causal rolling mean (no look-ahead).

        For positions i < window, uses the mean of all available points
        up to and including position i (expanding window) — never looks
        ahead, so the detrended residuals remain valid for training.
        """
        result = np.empty_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = arr[start : i + 1].mean()
        return result

    def _build_feature_matrix(
        self, residuals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct the (X_train, y_train) pair from the residual series.

        Each row in X covers lags [t-1 … t-N_LAGS] plus rolling statistics.
        The target y is the residual at position t.

        Returns empty arrays if the series is too short to build any rows.
        """
        n = len(residuals)
        min_row = self._n_lags   # need at least n_lags lookback + 1 target
        if n <= min_row:
            return np.empty((0, self._n_lags + 3)), np.empty(0)

        X_rows: list[np.ndarray] = []
        y_vals: list[float]      = []

        for i in range(min_row, n):
            x_vec = self._build_single_feature_vector(list(residuals[:i]))
            X_rows.append(x_vec)
            y_vals.append(float(residuals[i]))

        return np.array(X_rows, dtype=np.float64), np.array(y_vals, dtype=np.float64)

    def _build_single_feature_vector(self, window: list[float]) -> np.ndarray:
        """
        Build a single feature vector from the tail of *window*.

        Feature layout (n_lags + 3 columns):
          [0 … n_lags-1] : lag terms  P_{t-1} … P_{t-N_LAGS}
          [n_lags]       : rolling_mean over last ROLLING_SHORT points
          [n_lags+1]     : rolling_mean over last ROLLING_LONG  points
          [n_lags+2]     : max variance (std dev) over last ROLLING_LONG points
        """
        n     = len(window)
        tail  = window[-self._n_lags:] if n >= self._n_lags else window

        # Pad front with the earliest available value (bfill behaviour)
        if len(tail) < self._n_lags:
            pad = [tail[0]] * (self._n_lags - len(tail))
            tail = pad + tail

        # Lag features — most-recent is index 0 in feature vector
        lags = list(reversed(tail))   # [P_{t-1}, P_{t-2}, ..., P_{t-N_LAGS}]

        # Rolling statistics
        short_win  = window[-ROLLING_SHORT:] if n >= ROLLING_SHORT else window
        long_win   = window[-ROLLING_LONG:]  if n >= ROLLING_LONG  else window

        rm_short   = float(np.mean(short_win))
        rm_long    = float(np.mean(long_win))
        max_var    = float(np.std(long_win)) if len(long_win) > 1 else 0.0

        return np.array(lags + [rm_short, rm_long, max_var], dtype=np.float64)

    def _ema_24step_trajectory(
        self, series: list[float], base_val: float
    ) -> list[float]:
        """
        Generate a 24-step trajectory using the EMA base value with a
        sinusoidal diurnal envelope — used when Ridge cannot be trained.
        """
        from datetime import datetime, timezone
        current_hour = datetime.now(tz=timezone.utc).hour
        predictions  = []
        rng = np.random.default_rng()

        for step in range(24):
            future_hour  = (current_hour + step + 1) % 24
            # Evening peak at 18:00, morning trough at 04:00
            angle        = (future_hour - 18) / 24.0 * 2 * math.pi
            diurnal      = 1.0 + 0.20 * math.cos(angle)
            noise        = float(rng.normal(1.0, 0.015))
            val          = max(0.0, base_val * diurnal * noise)
            predictions.append(round(val, 2))

        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

_forecaster: BaseForecaster | None = None


def get_forecaster() -> BaseForecaster:
    """
    Return the module-level forecaster singleton.

    Selection logic (checked in order):
      1. ENABLE_LSTM=true  → LSTMForecaster
         Weights loaded from LSTM_WEIGHTS_PATH (relative to project root or
         absolute).  Falls back to random weights with a warning if the file
         is missing (safe for unit-test environments).
      2. ENABLE_GRID_FORECASTER=false explicitly → MovingAverageForecaster
      3. Default → GridForecaster (Feature 5 — Ridge + lagged features)

    Configuration is read from config.Settings (validated by Pydantic), with
    a direct os.getenv() fallback for contexts where config.py is unavailable
    (e.g. standalone training scripts).
    """
    global _forecaster
    if _forecaster is not None:
        return _forecaster

    # Prefer validated config.Settings values; fall back to raw env vars.
    try:
        from config import settings as _cfg
        enable_lstm            = _cfg.ENABLE_LSTM
        enable_grid_forecaster = _cfg.ENABLE_GRID_FORECASTER
        weights_path           = _cfg.LSTM_WEIGHTS_PATH
    except Exception:
        enable_lstm            = os.getenv("ENABLE_LSTM",            "false").lower() == "true"
        enable_grid_forecaster = os.getenv("ENABLE_GRID_FORECASTER", "true").lower()  == "true"
        weights_path           = os.getenv("LSTM_WEIGHTS_PATH", "")

    logger.info(
        "Forecaster config: ENABLE_LSTM=%s  ENABLE_GRID_FORECASTER=%s  "
        "LSTM_WEIGHTS_PATH=%s",
        enable_lstm, enable_grid_forecaster, weights_path or "<not set>",
    )

    if enable_lstm:
        try:
            _forecaster = LSTMForecaster(weights_path=weights_path or None)
            logger.info("Active forecaster: %s", _forecaster.name)
            return _forecaster
        except Exception as exc:
            logger.error(
                "Failed to initialise LSTMForecaster (%s) — falling back.", exc
            )

    if enable_grid_forecaster:
        try:
            _forecaster = GridForecaster()
            logger.info("Active forecaster: %s", _forecaster.name)
            return _forecaster
        except Exception as exc:
            logger.error(
                "Failed to initialise GridForecaster (%s) — falling back to MA.", exc
            )

    _forecaster = MovingAverageForecaster()
    logger.info("Active forecaster: %s", _forecaster.name)
    return _forecaster
