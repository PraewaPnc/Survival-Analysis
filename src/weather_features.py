"""Weather forecast feature aggregation and training-label generation.

First collapses the 72-hour hourly forecast file into one row per
(span_id, event_id) by aggregating each forecast variable into a single
summary statistic suitable as a model covariate:

    load_forecast → aggregate_forecast → save

It then joins those features back onto the transmission maintenance data and
generates a probabilistic fault-to-failure label per span × component × event:

    build_training_dataset → save
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
FORECAST_PATH = DATA_DIR / "weather_forecast_72hr.csv"
OUTPUT_PATH = DATA_DIR / "weather_aggregated.csv"
TRANSMISSION_PATH = DATA_DIR / "transmission_line_maintenance_data.csv"
TRAINING_PATH = DATA_DIR / "training_dataset.csv"

# Per-region forecast thresholds used to normalise the weather score:
# (wind_max threshold m/s, rain_total threshold mm).
REGION_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Northeast": (20.0, 150.0),
    "North": (22.0, 180.0),
    "Central": (18.0, 130.0),
    "South": (25.0, 300.0),
}

# Sea-level reference pressure (hPa) for the pressure-deficit term.
REFERENCE_PRESSURE_HPA = 1013.0

# Logistic label coefficients. The intercept was calibrated from the spec's
# nominal -2.8: forecast_rain_total is a 72-hour sum that routinely exceeds the
# region rain thresholds, so -2.8 produced a ~42% failure rate. -4.3 brought it
# to ~16%, but adding the age term to component_score lowered it to 14.8%, so the
# intercept was nudged to -4.2 to land back at ~16.1% (within the 15–20% band).
SCORE_INTERCEPT = -4.2
COMPONENT_WEIGHT = 2.5
WEATHER_WEIGHT = 2.0

# Reproducible Bernoulli label sampling.
RANDOM_STATE = 42

# Identifier columns carried through unchanged (one value per group).
# age_at_forecast_days and forecast_date are constant within a 72-hr event,
# so the first value per group is representative.
GROUP_KEYS: list[str] = ["span_id", "event_id"]
KEEP_COLS: list[str] = [
    "region", "event_type", "forecast_month",
    "age_at_forecast_days", "forecast_date",
]

# source column → (aggregation function, output column name)
AGG_SPEC: dict[str, tuple[str, str]] = {
    "forecast_gust_ms": ("max", "forecast_wind_max"),
    "forecast_wind_ms": ("mean", "forecast_wind_avg"),
    "forecast_rain_mm_hr": ("sum", "forecast_rain_total"),
    "forecast_pressure_hpa": ("min", "forecast_pressure_min"),
    "forecast_humidity_pct": ("max", "forecast_humidity_max"),
    "forecast_temp_c": ("max", "forecast_temp_max"),
    "forecast_confidence": ("mean", "forecast_confidence_avg"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_forecast(path: Path = FORECAST_PATH) -> pd.DataFrame:
    """Load the 72-hour hourly weather forecast CSV.

    Args:
        path: Path to the forecast CSV.

    Returns:
        Raw hourly forecast DataFrame (one row per forecast hour).
    """
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly forecasts into one row per (span_id, event_id).

    Each forecast variable is reduced to a single summary statistic
    (see AGG_SPEC), and the identifier columns in KEEP_COLS are carried
    through unchanged (assumed constant within each group).

    Args:
        df: Hourly forecast DataFrame from load_forecast().

    Returns:
        Aggregated DataFrame with GROUP_KEYS, KEEP_COLS, and the seven
        forecast summary columns.
    """
    agg_map = {src: func for src, (func, _) in AGG_SPEC.items()}
    rename_map = {src: out for src, (_, out) in AGG_SPEC.items()}

    # KEEP_COLS are constant within a group → take the first value.
    keep_map = {col: "first" for col in KEEP_COLS}

    grouped = (
        df.groupby(GROUP_KEYS, as_index=False)
        .agg({**keep_map, **agg_map})
        .rename(columns=rename_map)
    )

    ordered_cols = GROUP_KEYS + KEEP_COLS + list(rename_map.values())
    return grouped[ordered_cols]


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_aggregated(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Save the aggregated forecast features to CSV.

    Args:
        df: Aggregated DataFrame from aggregate_forecast().
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Training-label generation
# ---------------------------------------------------------------------------

def _sigmoid(x: pd.Series | np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid.

    Args:
        x: Linear predictor values.

    Returns:
        Element-wise sigmoid(x) in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def build_training_dataset(
    weather: pd.DataFrame,
    transmission: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Join weather features to maintenance data and label fault-to-failure.

    The aggregated weather features (one row per span × event) are joined onto
    the transmission maintenance data (one row per span × component) on
    ``span_id``, producing one row per span × component × event. A probabilistic
    ``fault_to_failure_flag`` is then drawn from a Bernoulli distribution whose
    success probability combines a component-health score and a region-normalised
    weather-severity score.

    Args:
        weather: Aggregated forecast features from aggregate_forecast().
        transmission: Raw transmission maintenance DataFrame.
        random_state: Seed for the Bernoulli label draw.

    Returns:
        Merged DataFrame with the generated probability ``failure_probability``
        and integer ``fault_to_failure_flag`` columns.
    """
    # The maintenance file ships its own fault_to_failure_flag; drop it so the
    # generated probabilistic label is unambiguous. region/event_type live in
    # the weather frame too — keep transmission's region as authoritative.
    trans = transmission.drop(columns=["fault_to_failure_flag"], errors="ignore")
    weather_cols = [c for c in weather.columns if c not in ("region",)]

    df = trans.merge(weather[weather_cols], on="span_id", how="inner")

    # Component age at the time of each forecast event (years), used both as a
    # model feature and as a term in the component-health score.
    df["age_at_forecast_years"] = df["age_at_forecast_days"] / 365

    # --- Component-health score (component-level covariates) ----------------
    component_score = (
        0.4 * (df["HI_score_last"] / 5)
        + 0.3 * (df["HI_trend_per_year"] / df["HI_trend_per_year"].max())
        + 0.2 * (df["permanent_faults"] / df["permanent_faults"].max())
        + 0.1 * (df["age_at_forecast_years"] / df["age_at_forecast_years"].max())
    )

    # --- Weather-severity score (event-level covariates) -------------------
    thresholds = df["region"].map(REGION_THRESHOLDS)
    wind_thresh = thresholds.map(lambda t: t[0])
    rain_thresh = thresholds.map(lambda t: t[1])

    weather_score = (
        0.4 * (df["forecast_wind_max"] / wind_thresh)
        + 0.35 * (df["forecast_rain_total"] / rain_thresh)
        + 0.25 * (1 - df["forecast_pressure_min"] / REFERENCE_PRESSURE_HPA)
    )

    # --- Probability + Bernoulli label -------------------------------------
    p = _sigmoid(
        COMPONENT_WEIGHT * component_score
        + WEATHER_WEIGHT * weather_score
        + SCORE_INTERCEPT
    )
    rng = np.random.RandomState(random_state)
    label = (rng.random(len(df)) < p).astype(int)

    df["failure_probability"] = p
    df["fault_to_failure_flag"] = label
    return df


def save_training_dataset(df: pd.DataFrame, path: Path = TRAINING_PATH) -> None:
    """Save the labelled training dataset to CSV.

    Args:
        df: Labelled DataFrame from build_training_dataset().
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_labels(
    weather_path: Path = OUTPUT_PATH,
    transmission_path: Path = TRANSMISSION_PATH,
    output: Path = TRAINING_PATH,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load → join → label → save the training dataset.

    Args:
        weather_path: Path to the aggregated weather features CSV.
        transmission_path: Path to the transmission maintenance CSV.
        output: Destination CSV path for the labelled training dataset.
        random_state: Seed for the Bernoulli label draw.
        verbose: Print failure rate, shape, and label distribution to stdout.

    Returns:
        The labelled training DataFrame.
    """
    weather = pd.read_csv(weather_path)
    transmission = pd.read_csv(transmission_path)

    df = build_training_dataset(weather, transmission, random_state=random_state)
    save_training_dataset(df, output)

    if verbose:
        rate = df["fault_to_failure_flag"].mean()
        counts = df["fault_to_failure_flag"].value_counts().sort_index()
        print(f"Training dataset saved to {output}")
        print(f"Shape: {df.shape}")
        print(f"Actual failure rate: {rate:.2%}")
        print("Label distribution:")
        print(counts.to_string())

    return df


# ---------------------------------------------------------------------------
# Convenience: run full aggregation
# ---------------------------------------------------------------------------

def run(
    path: Path = FORECAST_PATH,
    output: Path = OUTPUT_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load → aggregate → save the weather forecast features.

    Args:
        path: Path to the raw hourly forecast CSV.
        output: Destination CSV path for the aggregated result.
        verbose: Print shape and head to stdout.

    Returns:
        The aggregated DataFrame.
    """
    raw = load_forecast(path)
    agg = aggregate_forecast(raw)
    save_aggregated(agg, output)

    if verbose:
        print(f"Aggregated weather features saved to {output}")
        print(f"Shape: {agg.shape}")
        print(agg.head().to_string())

    return agg


if __name__ == "__main__":
    run()
