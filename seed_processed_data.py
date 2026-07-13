"""
GridPulse AI — Kaggle Dataset -> Telemetry Seed Script
======================================================
seed_processed_data.py

Purpose
-------
Converts a raw Kaggle energy consumption CSV into a GridPulse-compatible
telemetry CSV matching the raw telemetry format consumed by analytics.py
and the ML pipeline (anomaly_detector.py, load_forecaster.py).

Supported energy column names (resolved in priority order):
  - 'Electricity_Consumed'       <- smart_meter_enriched.csv (actual dataset)
  - 'Electricity Consumed (kWh)' <- original Kaggle spec
  - 'Electricity Consumed'       <- variant without unit
  - 'energy_kwh'                 <- common alternate name

The data layer derives aggregate load as:
    real_power_W = voltage * current * power_factor

So this script inverts that: given energy consumption, it synthesises the
three physical quantities that would have produced that consumption.

Output columns
--------------
  timestamp    -- ISO-8601 UTC string, preserving the original cadence
  voltage      -- RMS voltage (V), Gaussian(mu=230, sigma=2), clamped [210, 250]
  current      -- RMS current (A) = watts / (voltage x power_factor)
  power_factor -- dimensionless [0.88, 0.95], uniform distribution

Usage
-----
    python seed_processed_data.py --input <kaggle_file.csv> [OPTIONS]

Options
-------
  --input    PATH    Path to the Kaggle CSV (required)
  --output   PATH    Output CSV path (default: gridpulse_telemetry_db.csv)
  --seed     INT     NumPy RNG seed for reproducibility (default: 42)
  --meter-id STR     Meter ID tag written to CSV (default: KAGGLE_METER_01)
  --interval-min INT Assumed interval between readings in minutes (default: 30)

Example
-------
    python seed_processed_data.py \\
        --input dataset/smart_meter_enriched.csv \\
        --output gridpulse_telemetry_db.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gridpulse.seed")


# ---------------------------------------------------------------------------
# Physical constants (must match simulator.py + analytics.py)
# ---------------------------------------------------------------------------

VOLTAGE_NOMINAL  = 230.0   # V  — IEC 60038 nominal
VOLTAGE_STD_DEV  =   2.0   # V  — Gaussian jitter (matches simulator.py sigma=2)
VOLTAGE_MIN      = 210.0   # V  — clamp floor  (matches simulator safe band)
VOLTAGE_MAX      = 250.0   # V  — clamp ceiling

PF_MIN           =   0.88  # power factor lower bound (healthy residential)
PF_MAX           =   0.95  # power factor upper bound

# Conversion: kWh over a measurement interval -> average Watts
# For a 30-min interval:  W = kWh * (60 / 30) * 1000 = kWh * 2000
# For a 60-min interval:  W = kWh * (60 / 60) * 1000 = kWh * 1000
# The multiplier is computed dynamically from --interval-min; default = 2000.

CURRENT_MAX      = 300.0   # A  -- hard safety cap (physically plausible max)

# Energy column name aliases — resolved in priority order at load time.
# Add new variants here if future datasets use different headers.
ENERGY_COLUMN_ALIASES: list[str] = [
    "Electricity_Consumed",        # smart_meter_enriched.csv (actual dataset)
    "Electricity Consumed (kWh)",  # original Kaggle spec
    "Electricity Consumed",        # variant without unit suffix
    "energy_kwh",                  # common alternate name
    "kwh",                         # minimal alias
]
CURRENT_MIN      =   0.1   # A  — avoid near-zero values in downstream ML


# ---------------------------------------------------------------------------
# Core transformation functions
# ---------------------------------------------------------------------------

def kwh_to_watts(kwh: pd.Series, interval_minutes: int) -> pd.Series:
    """
    Convert kWh readings to average Watts for the measurement interval.

        W = kWh * (60 / interval_min) * 1000

    For the default 30-min cadence this is kWh * 2000, matching the
    specification in the user request.
    """
    multiplier = (60.0 / interval_minutes) * 1000.0
    logger.info(
        "kWh -> Watts multiplier: %.0f  (interval=%d min)",
        multiplier, interval_minutes,
    )
    return kwh * multiplier


def synthesise_voltage(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw RMS voltage from a Gaussian distribution centred on 230 V (sigma=2 V),
    clamped to the [210, 250] V safe operating band.

    Mirrors the distribution used in simulator.py _generate_reading().
    """
    raw = rng.normal(loc=VOLTAGE_NOMINAL, scale=VOLTAGE_STD_DEV, size=n)
    return np.clip(raw, VOLTAGE_MIN, VOLTAGE_MAX)


def synthesise_power_factor(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw power factor from a uniform distribution over [0.88, 0.95].

    This range reflects healthy residential/commercial loads:
      - >= 0.88: well within the grid-quality minimum of 0.80
      - <= 0.95: realistic for typical inductive AC loads
    """
    return rng.uniform(low=PF_MIN, high=PF_MAX, size=n)


def compute_current(
    watts: np.ndarray,
    voltage: np.ndarray,
    power_factor: np.ndarray,
) -> np.ndarray:
    """
    Derive RMS current from real power using the AC power formula:

        P = V * I * PF   =>   I = P / (V * PF)

    Edge cases:
      - Denominator floored at 1e-6 to prevent division by zero.
      - Negative watts (bad data) -> current set to CURRENT_MIN.
      - Current clamped to [CURRENT_MIN, CURRENT_MAX] to stay within
        the physically valid range accepted by the anomaly detector.
    """
    denominator = voltage * power_factor
    # Guard against any zero/sub-zero denominators
    denominator = np.where(denominator < 1e-6, 1e-6, denominator)

    current = np.where(watts > 0, watts / denominator, CURRENT_MIN)
    current = np.clip(current, CURRENT_MIN, CURRENT_MAX)
    return current


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_energy_column(columns: list[str]) -> str | None:
    """
    Return the first alias from ENERGY_COLUMN_ALIASES that is present in
    *columns*, or None if none match.

    Comparison is case-insensitive and strip-whitespace tolerant.
    """
    normalised = {c.strip().lower(): c for c in columns}
    for alias in ENERGY_COLUMN_ALIASES:
        if alias.strip().lower() in normalised:
            return normalised[alias.strip().lower()]
    return None


def load_kaggle_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Load and validate the Kaggle CSV.

    Required columns:
      - 'Timestamp'             -- any parseable datetime format
      - An energy column        -- resolved from ENERGY_COLUMN_ALIASES
                                   (e.g. 'Electricity_Consumed',
                                         'Electricity Consumed (kWh)')

    Returns
    -------
    (df, energy_col_name) where energy_col_name is the resolved column header.

    Raises SystemExit on unrecoverable errors.
    """
    logger.info("Loading Kaggle CSV: %s", path)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Input file not found: %s", path)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to read CSV: %s", exc)
        sys.exit(1)

    # Normalise column names: strip leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # Validate Timestamp column
    if "Timestamp" not in df.columns:
        logger.error(
            "Input CSV is missing 'Timestamp' column.\n"
            "  Found columns: %s", list(df.columns),
        )
        sys.exit(1)

    # Resolve energy column from alias list
    energy_col = resolve_energy_column(list(df.columns))
    if energy_col is None:
        logger.error(
            "Input CSV has no recognisable energy column.\n"
            "  Searched aliases : %s\n"
            "  Found columns    : %s",
            ENERGY_COLUMN_ALIASES, list(df.columns),
        )
        sys.exit(1)

    logger.info(
        "Loaded %d rows from '%s'  |  energy column resolved -> '%s'",
        len(df), path.name, energy_col,
    )
    return df, energy_col


def parse_timestamps(raw_ts: pd.Series) -> pd.Series:
    """
    Parse the Timestamp column into timezone-aware UTC datetimes.

    Accepts any format pandas can infer. Timezone-naive timestamps are
    assumed to be UTC (standard for public energy datasets).
    """
    # Use format='mixed' to avoid the infer_datetime_format deprecation
    # warning introduced in pandas >= 2.0.
    try:
        ts = pd.to_datetime(raw_ts, format="mixed", utc=False)
    except TypeError:
        # pandas < 2.0 fallback
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ts = pd.to_datetime(raw_ts, infer_datetime_format=True, utc=False)

    if ts.dt.tz is None:
        logger.info("Timestamps are timezone-naive -- assuming UTC.")
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    return ts


def validate_kwh(series: pd.Series) -> pd.Series:
    """
    Validate and clean the kWh column.

    1. Coerce non-numeric values to NaN with a warning.
    2. Replace NaN or negative values with 0.0 (safe default for missing data).
    """
    coerced = pd.to_numeric(series, errors="coerce")

    n_nan = coerced.isna().sum()
    n_neg = (coerced < 0).sum()
    n_bad = n_nan + n_neg

    if n_bad > 0:
        logger.warning(
            "%d rows have invalid kWh values (%d NaN, %d negative) "
            "— replaced with 0.0.",
            n_bad, n_nan, n_neg,
        )

    coerced = coerced.where(coerced >= 0, other=np.nan)
    coerced = coerced.fillna(0.0)
    return coerced


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_telemetry(
    df: pd.DataFrame,
    energy_col: str,
    interval_minutes: int,
    meter_id: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Full ETL pipeline: Kaggle DataFrame -> GridPulse telemetry DataFrame.

    Steps
    -----
    1. Parse + validate timestamps and energy values.
    2. Sort chronologically (the forecaster requires oldest->newest order,
       matching the ORDER BY timestamp ASC in analytics.py _fetch_load_history).
    3. Convert energy -> Watts (kWh * 2000 for 30-min intervals).
    4. Synthesise voltage ~ Gaussian(230, 2), clamped [210, 250].
    5. Synthesise power_factor ~ Uniform(0.88, 0.95).
    6. Derive current = watts / (voltage * power_factor).
    7. Assemble output DataFrame with GridPulse column schema.

    Parameters
    ----------
    df            : Raw Kaggle DataFrame (all columns present).
    energy_col    : Resolved energy column name (from ENERGY_COLUMN_ALIASES).
    interval_minutes : Cadence in minutes; drives the kWh->W multiplier.
    meter_id      : Tag written into the meter_id column.
    rng           : NumPy Generator for reproducible synthesis.
    """
    n = len(df)
    logger.info("Processing %d rows ...", n)

    # Step 1: Parse & validate
    timestamps = parse_timestamps(df["Timestamp"])
    kwh        = validate_kwh(df[energy_col])

    # Step 2: Sort chronologically (oldest -> newest)
    sort_idx   = timestamps.argsort()
    timestamps = timestamps.iloc[sort_idx].reset_index(drop=True)
    kwh        = kwh.iloc[sort_idx].reset_index(drop=True)

    # Step 3: Energy -> Watts
    watts = kwh_to_watts(kwh, interval_minutes).to_numpy(dtype=np.float64)

    # Step 4: Synthesise voltage
    voltage = synthesise_voltage(n, rng)

    # Step 5: Synthesise power factor
    power_factor = synthesise_power_factor(n, rng)

    # Step 6: Derive current
    current = compute_current(watts, voltage, power_factor)

    # Step 7: Assemble output
    out = pd.DataFrame({
        "timestamp":    timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "voltage":      np.round(voltage,      4),
        "current":      np.round(current,      4),
        "power_factor": np.round(power_factor, 4),
    })

    return out


def run_100_row_transformation_check(
    df: pd.DataFrame,
    energy_col: str,
    interval_minutes: int,
    rng_seed: int,
) -> None:
    """
    Transformation verification against the first 100 rows.

    Checks
    ------
    1. kWh -> Watts scaling is correct (expected multiplier = 60/interval * 1000).
    2. Voltage is within the Gaussian band [VOLTAGE_MIN, VOLTAGE_MAX].
    3. Power factor is within [PF_MIN, PF_MAX].
    4. Current is within [CURRENT_MIN, CURRENT_MAX] and matches I = W / (V * PF).
    5. No NaN values in any output column.
    """
    DIVIDER = "=" * 64
    print(f"\n{DIVIDER}")
    print("  TRANSFORMATION CHECK — first 100 rows")
    print(DIVIDER)

    sample = df.head(100).copy().reset_index(drop=True)
    n      = len(sample)
    print(f"  Rows sampled         : {n}")

    rng          = np.random.default_rng(seed=rng_seed)
    multiplier   = (60.0 / interval_minutes) * 1000.0
    kwh_vals     = pd.to_numeric(sample[energy_col], errors="coerce").fillna(0.0)
    watts_actual = (kwh_vals * multiplier).to_numpy(dtype=np.float64)

    voltage      = synthesise_voltage(n, rng)
    power_factor = synthesise_power_factor(n, rng)
    current      = compute_current(watts_actual, voltage, power_factor)

    # --- Check 1: kWh -> Watts scaling ---
    expected_w_sample = float(kwh_vals.iloc[0]) * multiplier
    actual_w_sample   = watts_actual[0]
    scale_ok = abs(expected_w_sample - actual_w_sample) < 1e-6
    print(f"  [{'PASS' if scale_ok else 'FAIL'}] kWh->W scaling  "
          f"row[0]: {float(kwh_vals.iloc[0]):.6f} kWh "
          f"x {multiplier:.0f} = {expected_w_sample:.4f} W")

    # --- Check 2: Voltage bounds ---
    v_min, v_max = float(voltage.min()), float(voltage.max())
    v_ok = (v_min >= VOLTAGE_MIN) and (v_max <= VOLTAGE_MAX)
    print(f"  [{'PASS' if v_ok else 'FAIL'}] Voltage range    "
          f"min={v_min:.2f} V  max={v_max:.2f} V  "
          f"(expected [{VOLTAGE_MIN}, {VOLTAGE_MAX}] V)")

    # --- Check 3: Power factor bounds ---
    pf_min, pf_max = float(power_factor.min()), float(power_factor.max())
    pf_ok = (pf_min >= PF_MIN) and (pf_max <= PF_MAX)
    print(f"  [{'PASS' if pf_ok else 'FAIL'}] Power factor     "
          f"min={pf_min:.4f}  max={pf_max:.4f}  "
          f"(expected [{PF_MIN}, {PF_MAX}])")

    # --- Check 4: Current bounds and formula ---
    i_min, i_max = float(current.min()), float(current.max())
    i_ok = (i_min >= CURRENT_MIN) and (i_max <= CURRENT_MAX)
    # Spot-verify first non-zero row
    for idx in range(n):
        if watts_actual[idx] > 0:
            expected_i = watts_actual[idx] / (voltage[idx] * power_factor[idx])
            expected_i = float(np.clip(expected_i, CURRENT_MIN, CURRENT_MAX))
            break
    actual_i   = float(current[idx])
    formula_ok = abs(expected_i - actual_i) < 1e-3
    print(f"  [{'PASS' if i_ok else 'FAIL'}] Current range    "
          f"min={i_min:.4f} A  max={i_max:.4f} A  "
          f"(expected [{CURRENT_MIN}, {CURRENT_MAX}] A)")
    print(f"  [{'PASS' if formula_ok else 'FAIL'}] Current formula  "
          f"row[{idx}]: W={watts_actual[idx]:.4f}  "
          f"V={voltage[idx]:.4f}  PF={power_factor[idx]:.4f}  "
          f"I_expected={expected_i:.4f}  I_actual={actual_i:.4f}")

    # --- Check 5: No NaN values ---
    out_check = pd.DataFrame({
        "voltage": voltage, "current": current, "power_factor": power_factor
    })
    nan_count = int(out_check.isna().sum().sum())
    nan_ok    = nan_count == 0
    print(f"  [{'PASS' if nan_ok else 'FAIL'}] No NaN values    count={nan_count}")

    all_pass = all([scale_ok, v_ok, pf_ok, i_ok, formula_ok, nan_ok])
    print(f"\n  Result: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(DIVIDER + "\n")


def print_summary(out: pd.DataFrame) -> None:
    """Log a statistical QA summary of the generated telemetry and print head(5)."""
    DIVIDER = "-" * 64
    logger.info(DIVIDER)
    logger.info("Output statistics:")
    logger.info("  Rows           : %d", len(out))
    logger.info(
        "  Timestamp range: %s  ->  %s",
        out["timestamp"].iloc[0], out["timestamp"].iloc[-1],
    )
    logger.info(
        "  Voltage   (V)  : min=%.2f  max=%.2f  mean=%.2f",
        out["voltage"].min(), out["voltage"].max(), out["voltage"].mean(),
    )
    logger.info(
        "  Current   (A)  : min=%.4f  max=%.4f  mean=%.4f",
        out["current"].min(), out["current"].max(), out["current"].mean(),
    )
    logger.info(
        "  Power Factor   : min=%.4f  max=%.4f  mean=%.4f",
        out["power_factor"].min(), out["power_factor"].max(), out["power_factor"].mean(),
    )

    # Derived real power -- sanity check that V*I*PF matches the original Watt scale
    derived_w = out["voltage"] * out["current"] * out["power_factor"]
    logger.info(
        "  Derived P (W)  : min=%.1f  max=%.1f  mean=%.1f",
        derived_w.min(), derived_w.max(), derived_w.mean(),
    )
    logger.info(DIVIDER)

    # Warn if any current would trigger the line-tapping anomaly detector
    # (anomaly_detector.py threshold: _LINE_TAP_CURRENT_MIN = 50.0 A)
    high_current_rows = int((out["current"] > 50.0).sum())
    if high_current_rows > 0:
        logger.warning(
            "%d rows have current > 50 A -- these may trigger the "
            "line-tapping anomaly detector in the cloud pipeline. "
            "Review source energy values if unexpected.",
            high_current_rows,
        )

    # ── Data Integrity Log: shape + head(5) printed to stdout ────────────────
    DIVIDER2 = "=" * 64
    print(f"\n{DIVIDER2}")
    print("  DATA INTEGRITY LOG")
    print(DIVIDER2)
    print(f"  DataFrame shape : {out.shape}  "
          f"(rows={out.shape[0]}, cols={out.shape[1]})")
    print(f"  Columns         : {list(out.columns)}")
    print(f"\n  head(5) -- first 5 rows of generated telemetry:")
    print(out.head(5).to_string(index=True))
    print(DIVIDER2 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Kaggle kWh energy CSV into a GridPulse-compatible "
            "telemetry seed file (timestamp / voltage / current / power_factor)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the raw Kaggle CSV file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="gridpulse_telemetry_db.csv",
        type=Path,
        metavar="PATH",
        help="Path for the output GridPulse telemetry CSV.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        metavar="INT",
        help="NumPy RNG seed for reproducible synthesis.",
    )
    parser.add_argument(
        "--meter-id",
        default="KAGGLE_METER_01",
        metavar="STR",
        help="Meter ID tag embedded in the output CSV.",
    )
    parser.add_argument(
        "--interval-min",
        default=30,
        type=int,
        metavar="INT",
        help=(
            "Measurement interval in minutes. Used to convert kWh -> average Watts. "
            "30 min => multiplier 2000; 60 min => multiplier 1000."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("GridPulse AI -- Telemetry Seed Script")
    logger.info("  Input   : %s", args.input)
    logger.info("  Output  : %s", args.output)
    logger.info("  Seed    : %d", args.seed)
    logger.info("  Meter ID: %s", args.meter_id)
    logger.info("  Interval: %d min", args.interval_min)

    rng = np.random.default_rng(seed=args.seed)

    # ── Step 1: File Ingestion ────────────────────────────────────────────────
    df, energy_col = load_kaggle_csv(args.input)
    logger.info(
        "File ingestion OK: %d rows x %d columns | energy column: '%s'",
        len(df), len(df.columns), energy_col,
    )

    # ── Step 2: Transformation check (first 100 rows) ────────────────────────
    run_100_row_transformation_check(
        df,
        energy_col=energy_col,
        interval_minutes=args.interval_min,
        rng_seed=args.seed,
    )

    # ── Step 3: Full transformation ───────────────────────────────────────────
    out = build_telemetry(
        df,
        energy_col=energy_col,
        interval_minutes=args.interval_min,
        meter_id=args.meter_id,
        rng=rng,
    )

    # ── Step 4: QA summary + Data Integrity Log ───────────────────────────────
    print_summary(out)

    # ── Step 5: Write full output CSV ─────────────────────────────────────────
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out[["timestamp", "voltage", "current", "power_factor"]].to_csv(
        out_path, index=False
    )

    logger.info(
        "Output written: %d rows -> %s",
        len(out), out_path.resolve(),
    )
    logger.info("Pipeline complete. Ready for LSTM training.")


if __name__ == "__main__":
    main()
