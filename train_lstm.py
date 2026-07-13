"""
GridPulse AI -- LSTM Load Forecaster Training Script
=====================================================
train_lstm.py

Purpose
-------
Trains the LSTMForecaster architecture defined in ml/load_forecaster.py
on the pre-processed telemetry seed file (gridpulse_telemetry_db.csv)
and saves the final model weights to lstm_load_weights.pth.

Architecture contract  (must stay in sync with ml/load_forecaster.py)
----------------------------------------------------------------------
  INPUT_SIZE  = 1       univariate: aggregate real power (Watts)
  HIDDEN_SIZE = 64
  NUM_LAYERS  = 2
  DROPOUT     = 0.2
  SEQ_LEN     = 48      48-sample sliding window (~24 h at 30-min cadence)

Sliding-window label convention
--------------------------------
  feature  x  =  watts[i : i + SEQ_LEN]        shape [SEQ_LEN, 1]
  target   y  =  watts[i + SEQ_LEN]             scalar (Watts, 48 steps ahead)

  This matches the inference convention in LSTMForecaster.predict_24h()
  which feeds the last 48 samples and predicts the T+24h load.

Normalisation
-------------
  The Watts series is Z-score normalised (mean/std computed on the training
  split only, applied identically to the val split to prevent data leakage).
  Scaler parameters are saved to lstm_load_weights.scaler.json so that the
  production forecaster can invert the transform at inference time.

Train / val split
-----------------
  80 pct train / 20 pct val, split chronologically (no shuffle on val set).

Files produced
--------------
  lstm_load_weights.pth          final epoch state dict (production weights)
  lstm_load_weights_best.pth     best val-MSE epoch state dict
  lstm_load_weights.scaler.json  normalisation parameters
  lstm_load_weights.history.json per-epoch loss log

Usage
-----
  python train_lstm.py [OPTIONS]

  python train_lstm.py --data gridpulse_telemetry_db.csv --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gridpulse.train")


# ---------------------------------------------------------------------------
# Hyper-parameters  (MUST match ml/load_forecaster.py LSTMForecaster constants)
# ---------------------------------------------------------------------------

INPUT_SIZE  = 1    # univariate: aggregate Watts
HIDDEN_SIZE = 64   # LSTMForecaster.HIDDEN_SIZE
NUM_LAYERS  = 2    # LSTMForecaster.NUM_LAYERS
DROPOUT     = 0.2  # LSTMForecaster.DROPOUT
SEQ_LEN     = 48   # LSTMForecaster.SEQ_LEN


# ---------------------------------------------------------------------------
# Model definition  (mirrors _LSTMNet inside LSTMForecaster._init_torch)
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    """
    Sequence-to-one LSTM load forecaster.

    Identical architecture to the private _LSTMNet class inside
    LSTMForecaster._init_torch() in ml/load_forecaster.py.
    Any change to that class must be mirrored here, and vice versa,
    so that the saved state dict can be loaded at inference time.

    Input shape  : [batch, seq_len, 1]   (batch_first=True)
    Output shape : [batch]               scalar Watt prediction per sample
    """

    def __init__(
        self,
        input_size:  int   = INPUT_SIZE,
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, 1]  ->  [batch] scalar predictions."""
        out, _ = self.lstm(x)                          # [batch, seq, hidden]
        return self.linear(out[:, -1, :]).squeeze(-1)  # last timestep -> scalar


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WattsWindowDataset(Dataset):
    """
    Sliding-window PyTorch Dataset over a univariate Watts time series.

    Each sample produced:
        x : FloatTensor[seq_len, 1]  -- SEQ_LEN consecutive normalised Watt readings
        y : FloatTensor scalar       -- the reading SEQ_LEN steps ahead (target)

    This directly mirrors the inference convention in
    LSTMForecaster.predict_24h(), which consumes the last SEQ_LEN samples
    and predicts one step ahead in normalised space.

    Parameters
    ----------
    watts   : 1-D numpy array of real-power values (Watts), chronological,
              ALREADY sorted oldest -> newest.
    seq_len : Sliding window length. Default = SEQ_LEN (48).
    mean    : Z-score mean computed on the training split.
    std     : Z-score std  computed on the training split.
              If std <= 0, normalisation is skipped (identity transform).
    """

    def __init__(
        self,
        watts:   np.ndarray,
        seq_len: int   = SEQ_LEN,
        mean:    float = 0.0,
        std:     float = 1.0,
    ) -> None:
        self.seq_len = seq_len
        self.mean    = float(mean)
        self.std     = float(std) if float(std) > 0 else 1.0

        # Z-score normalise the entire split in-place (no copy needed for training)
        self.watts = (watts - self.mean) / self.std

        # Total valid windows: window [i : i+seq_len] needs target at [i+seq_len]
        if len(self.watts) < seq_len + 1:
            raise ValueError(
                f"Not enough data for a sliding window: need >= {seq_len + 1} "
                f"samples, got {len(self.watts)}."
            )
        self._n = len(self.watts) - seq_len

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_raw = self.watts[idx : idx + self.seq_len]  # [seq_len]
        y_raw = self.watts[idx + self.seq_len]         # scalar

        # Shape: [seq_len, 1] to match LSTMNet input [batch, seq, features]
        x = torch.tensor(x_raw, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(float(y_raw), dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_watts(csv_path: Path) -> np.ndarray:
    """
    Load gridpulse_telemetry_db.csv and compute aggregate real power.

    Real power (Watts) = voltage * current * power_factor
    Sorted ascending by timestamp to preserve the chronological sequence
    required by the LSTM training and inference pipeline.

    Returns
    -------
    1-D float64 numpy array of Watt values, oldest -> newest.
    """
    logger.info("Loading telemetry CSV: %s", csv_path)

    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    except FileNotFoundError:
        logger.error("Telemetry CSV not found: %s", csv_path)
        logger.error("Run seed_processed_data.py first to generate it.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to read CSV: %s", exc)
        sys.exit(1)

    required = {"timestamp", "voltage", "current", "power_factor"}
    missing  = required - set(df.columns)
    if missing:
        logger.error(
            "CSV is missing required columns: %s\n  Found: %s",
            missing, list(df.columns),
        )
        sys.exit(1)

    # Sort chronologically -- the forecaster contract requires oldest->newest
    df = df.sort_values("timestamp").reset_index(drop=True)

    watts = (
        df["voltage"] * df["current"] * df["power_factor"]
    ).to_numpy(dtype=np.float64)

    logger.info(
        "Loaded %d rows | P(W): min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
        len(watts), watts.min(), watts.max(), watts.mean(), watts.std(),
    )
    return watts


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """
    Run one complete forward-backward pass over *loader*.

    Gradient clipping (max_norm=1.0) prevents exploding gradients which
    are common when training LSTMs on noisy sensor data.

    Returns
    -------
    Average MSE loss over all batches (normalised space).
    """
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)    # [batch, seq_len, 1]
        y = y.to(device)    # [batch]

        optimiser.zero_grad()
        pred = model(x)                              # [batch]
        loss = criterion(pred, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """
    Compute the average MSE loss over a validation DataLoader.

    Returns
    -------
    Average MSE (normalised space).
    """
    model.eval()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        running_loss += criterion(pred, y).item()

    return running_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GridPulse LSTM load forecaster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="gridpulse_telemetry_db.csv",
        type=Path,
        metavar="PATH",
        help="Path to gridpulse_telemetry_db.csv produced by seed_processed_data.py.",
    )
    parser.add_argument(
        "--output",
        default="lstm_load_weights.pth",
        type=Path,
        metavar="PATH",
        help="Destination for the final model state dict.",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="INT",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        default=32,
        type=int,
        metavar="INT",
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        metavar="FLOAT",
        help="Adam optimiser learning rate.",
    )
    parser.add_argument(
        "--seq-len",
        default=SEQ_LEN,
        type=int,
        metavar="INT",
        help="Sliding window length. Must match LSTMForecaster.SEQ_LEN (48).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        metavar="INT",
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DIVIDER  = "=" * 72
    DIVIDER2 = "-" * 72

    logger.info("GridPulse AI -- LSTM Training Script")
    logger.info("  Data     : %s", args.data)
    logger.info("  Output   : %s", args.output)
    logger.info("  Epochs   : %d", args.epochs)
    logger.info("  Batch    : %d", args.batch)
    logger.info("  LR       : %g", args.lr)
    logger.info("  Seq len  : %d", args.seq_len)
    logger.info("  Device   : %s", device)

    # ------------------------------------------------------------------
    # 1. Load Watts series from telemetry CSV
    # ------------------------------------------------------------------
    watts = load_watts(args.data)

    # ------------------------------------------------------------------
    # 2. Chronological train / val split  (80 / 20)
    # ------------------------------------------------------------------
    split_idx  = int(len(watts) * 0.80)
    train_raw  = watts[:split_idx]
    val_raw    = watts[split_idx:]

    # Z-score scaler -- fit on train only (prevent data leakage into val)
    train_mean = float(train_raw.mean())
    train_std  = float(train_raw.std())

    logger.info(
        "Train/val split: %d / %d samples",
        len(train_raw), len(val_raw),
    )
    logger.info(
        "Scaler (train split): mean=%.4f W  std=%.4f W",
        train_mean, train_std,
    )

    train_ds = WattsWindowDataset(
        train_raw, seq_len=args.seq_len, mean=train_mean, std=train_std
    )
    val_ds = WattsWindowDataset(
        val_raw,   seq_len=args.seq_len, mean=train_mean, std=train_std
    )

    logger.info(
        "Sliding-window samples: train=%d  val=%d",
        len(train_ds), len(val_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,              # shuffling within epoch reduces temporal autocorrelation
        num_workers=0,             # 0 = main process  (safe on Windows)
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,             # val must stay chronological
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # 3. Instantiate model
    # ------------------------------------------------------------------
    model = _LSTMNet(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: _LSTMNet | params=%d | hidden=%d | layers=%d | dropout=%.1f",
        n_params, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    )

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print(
        f"  LSTM TRAINING  |  epochs={args.epochs}  batch={args.batch}  "
        f"lr={args.lr}  seq_len={args.seq_len}"
    )
    print(
        f"  Architecture   |  hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}  "
        f"dropout={DROPOUT}  params={n_params:,}"
    )
    print(f"  Device         |  {device}")
    print(DIVIDER)
    print(
        f"  {'Epoch':>6}  {'Train MSE':>10}  {'Train RMSE(W)':>13}  "
        f"{'Val MSE':>10}  {'Val RMSE(W)':>11}  {'Time':>6}"
    )
    print(f"  {DIVIDER2}")

    best_val_loss  = float("inf")
    best_epoch     = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, device
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed_s = time.perf_counter() - t0

        # Denormalise RMSE back to Watts for interpretable output
        train_rmse_w = (train_loss ** 0.5) * train_std
        val_rmse_w   = (val_loss   ** 0.5) * train_std

        print(
            f"  {epoch:>6}  {train_loss:>10.6f}  {train_rmse_w:>11.2f} W  "
            f"{val_loss:>10.6f}  {val_rmse_w:>9.2f} W  {elapsed_s:>4.1f}s"
        )

        history.append({
            "epoch":         epoch,
            "train_mse":     round(float(train_loss), 8),
            "val_mse":       round(float(val_loss),   8),
            "train_rmse_w":  round(float(train_rmse_w), 4),
            "val_rmse_w":    round(float(val_rmse_w),   4),
            "elapsed_s":     round(float(elapsed_s),    2),
        })

        # Track best validation epoch and save a checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_path     = args.output.with_stem(args.output.stem + "_best")
            torch.save(model.state_dict(), best_path)
            logger.debug("New best val MSE=%.6f at epoch %d -> %s", val_loss, epoch, best_path)

    print(f"  {DIVIDER2}")
    print(
        f"  Best val MSE : {best_val_loss:.6f}  "
        f"(epoch {best_epoch}, RMSE={best_val_loss**0.5 * train_std:.2f} W)"
    )
    print(DIVIDER + "\n")

    # ------------------------------------------------------------------
    # 5. Save final state dict  (as requested -- final epoch weights)
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info("Final weights saved  -> %s", args.output.resolve())

    # Scaler parameters -- inference must invert normalisation to get Watts
    scaler_path = args.output.with_suffix(".scaler.json")
    scaler_data = {
        "mean_w":   train_mean,
        "std_w":    train_std,
        "seq_len":  args.seq_len,
        "note":     "Apply z-score normalisation before inference: (x - mean_w) / std_w",
    }
    with open(scaler_path, "w", encoding="utf-8") as fh:
        json.dump(scaler_data, fh, indent=2)
    logger.info("Scaler params saved  -> %s", scaler_path.resolve())

    # Full per-epoch history log
    history_path = args.output.with_suffix(".history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    logger.info("Training history saved -> %s", history_path.resolve())

    logger.info(
        "Done. To activate in production:\n"
        "  Set ENABLE_LSTM=true and LSTM_WEIGHTS_PATH=%s in .env",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()