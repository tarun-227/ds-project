"""Feature engineering for next-day PM2.5 forecasting.

Usage (from partA-project/):
    python -m src.features

One row = (city, day t). Features use information available at the END of
day t only; the target is the observed PM2.5 on day t+1.

Leakage rules (tested in tests/test_features.py):
- Every pollutant feature is from day t or earlier (lags via groupby-shift,
  rolling windows that end at t).
- The "city level" feature is an expanding mean of the city's PAST observed
  PM2.5 (shifted by one day). It plays the role of target encoding, but it
  can never see the target and it works for cities that only appear in 2019/20.
- Calendar features (weekday, month, Diwali) describe the TARGET day t+1.
  They are known in advance, so using them is not leakage.
- Weather (NASA POWER reanalysis) is joined for day t (`*_t`, used by the
  main models) and for day t+1 (`*_next`). `*_next` is what a perfect 1-day
  weather forecast would give; it is used ONLY in the ablation in train.py,
  as an upper bound, never in the headline models.
- The calendar is complete per city (see clean.py), so shift(1) really is
  "yesterday" and shift(-1) really is "tomorrow".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_FILE, FEATURE_FILE, POOR_THRESHOLD, TARGET, TEST_YEAR,
    TRAIN_YEARS, VAL_YEAR, WEATHER_FILE,
)

# Diwali (main day) in the data period. Firecrackers push PM2.5 up for a few days.
DIWALI = pd.to_datetime([
    "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27",
])
DIWALI_WINDOW = 2  # flag days within +/- 2 days of Diwali

# Indian Meteorological Department seasons.
SEASON = {12: "winter", 1: "winter", 2: "winter",
          3: "summer", 4: "summer", 5: "summer",
          6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
          10: "post_monsoon", 11: "post_monsoon"}

OTHER_POLLUTANTS = ["NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]


def diwali_flag(dates: pd.Series) -> pd.Series:
    """1 if the date is within DIWALI_WINDOW days of a Diwali, else 0."""
    d = pd.to_datetime(dates).values.astype("datetime64[D]")
    diffs = np.abs(d[:, None] - DIWALI.values.astype("datetime64[D]")[None, :])
    return pd.Series((diffs.min(axis=1) <= np.timedelta64(DIWALI_WINDOW, "D")).astype("int8"),
                     index=dates.index)


WEATHER_VARS = ["temp_mean", "temp_range", "precip_mm", "wind_mean", "wind_max",
                "wind_dir", "humidity", "pressure_kpa"]


def add_weather(out: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join weather of day t (`_t`) and of the target day t+1 (`_next`)."""
    w = weather.copy()
    w["Date"] = pd.to_datetime(w["Date"])
    w["temp_range"] = w["temp_max"] - w["temp_min"]  # small range ~ stagnant, inversion-prone air
    w = w[["City", "Date"] + WEATHER_VARS]
    today = w.rename(columns={c: f"{c}_t" for c in WEATHER_VARS})
    nxt = w.rename(columns={c: f"{c}_next" for c in WEATHER_VARS}).rename(columns={"Date": "target_date"})
    out = out.merge(today, on=["City", "Date"], how="left")
    return out.merge(nxt, on=["City", "target_date"], how="left")


def build_features(clean: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per (City, Date=t) with features and next-day targets."""
    df = clean.sort_values(["City", "Date"]).reset_index(drop=True).copy()
    g = df.groupby("City", observed=True)
    pm = df[TARGET]  # forward-filled (<=3 days, past only) feature version

    out = pd.DataFrame({"City": df["City"].astype(str), "Date": df["Date"]})

    # --- autoregressive features: today and the recent past
    out["pm_t"] = pm
    out["pm_t_was_missing"] = df[f"{TARGET}_was_missing"]
    for lag in (1, 2, 6):  # lag 6 of t == same weekday as the target day t+1, a week back
        out[f"pm_lag{lag}"] = g[TARGET].shift(lag)
    for w in (3, 7, 30):
        out[f"pm_roll{w}"] = g[TARGET].transform(
            lambda s, w=w: s.rolling(w, min_periods=max(1, w // 2)).mean())
    out["pm_roll7_std"] = g[TARGET].transform(lambda s: s.rolling(7, min_periods=3).std())
    out["pm_diff1"] = out["pm_t"] - out["pm_lag1"]

    # --- leak-free "target encoding": the city's average of past observed values
    out["city_mean_past"] = g[f"{TARGET}_obs"].transform(
        lambda s: s.shift(1).expanding(min_periods=30).mean())

    # --- other pollutants today (co-emitted, e.g. NO2/CO from traffic)
    for col in OTHER_POLLUTANTS:
        if col in df.columns:
            out[col] = df[col]

    # --- calendar of the target day t+1 (known in advance)
    target_day = df["Date"] + pd.Timedelta(days=1)
    out["target_date"] = target_day
    out["dow"] = target_day.dt.dayofweek
    out["month"] = target_day.dt.month
    doy = target_day.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["season"] = target_day.dt.month.map(SEASON)
    out["diwali"] = diwali_flag(target_day)

    # --- targets: tomorrow's OBSERVED value (never imputed)
    out["y"] = g[f"{TARGET}_obs"].shift(-1)
    out["y_poor"] = (out["y"] > POOR_THRESHOLD).astype("float").where(out["y"].notna())

    if weather is not None:
        out = add_weather(out, weather)

    # Persistence baseline needs today's value; both are needed for a fair comparison.
    out = out[out["y"].notna() & out["pm_t"].notna()].reset_index(drop=True)
    out["split"] = assign_split(out["target_date"])
    return out[out["split"].notna()].reset_index(drop=True)


def assign_split(target_date: pd.Series) -> pd.Series:
    """Split on the year of the TARGET day: train 2015-18, val 2019, test 2020."""
    y = target_date.dt.year
    lo, hi = TRAIN_YEARS
    return pd.Series(np.select([y.between(lo, hi), y == VAL_YEAR, y == TEST_YEAR],
                               ["train", "val", "test"], default=None),
                     index=target_date.index).replace({None: np.nan})


NUMERIC_FEATURES = (
    ["pm_t", "pm_t_was_missing", "pm_lag1", "pm_lag2", "pm_lag6",
     "pm_roll3", "pm_roll7", "pm_roll30", "pm_roll7_std", "pm_diff1",
     "city_mean_past"] + OTHER_POLLUTANTS +
    ["dow", "month", "doy_sin", "doy_cos", "diwali"] +
    [f"{c}_t" for c in WEATHER_VARS]
)
# Oracle: tomorrow's actual weather (stand-in for a perfect forecast). Ablation only.
NEXT_WEATHER_FEATURES = [f"{c}_next" for c in WEATHER_VARS]
CATEGORICAL_FEATURES = ["season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main() -> None:
    clean = pd.read_csv(CLEAN_FILE, parse_dates=["Date"])
    weather = pd.read_csv(WEATHER_FILE, parse_dates=["Date"])
    feats = build_features(clean, weather)
    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(FEATURE_FILE, index=False)
    print(f"Saved {len(feats):,} rows x {len(FEATURES)} features to {FEATURE_FILE}")
    print(feats["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
