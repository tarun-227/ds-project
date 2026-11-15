"""Clean the raw CPCB city-day data.

Usage (from partA-project/):
    python -m src.clean

Every step records what it changed in a report dict, which is saved to
reports/cleaning_report.json so the numbers in the report are traceable.

Design decisions (and why):
- Duplicates on (City, Date): keep the first; a city has one reading per day.
- Impossible values (negative, or above config.PHYSICAL_MAX) -> NaN. They are
  sensor faults, not real pollution. Real extremes (Diwali, stubble burning)
  are kept on purpose: those are exactly the days we want to predict.
- PM2.5 spikes > SPIKE_MIN that are > SPIKE_RATIO x the city's previous-30-day
  median AND fall outside the Oct-Jan smog season -> NaN (e.g. Gurugram
  950 ug/m3 in late August, Guwahati 917 in May: no known event explains them
  and neighbouring days are ~40). Inside the smog season such jumps are real
  (Diwali, stubble burning) and kept.
- Columns with > MAX_MISSING_SHARE missing in the TRAIN years are dropped.
  The share is measured on train years only so the test period cannot
  influence a preprocessing decision.
- Each city is reindexed to a full daily calendar so a "lag 1" feature is
  really yesterday, not "the previous row that happened to exist".
- Feature gaps are forward-filled for at most FFILL_LIMIT days. Forward fill
  uses only the past; interpolation would use tomorrow's value and leak.
  A `<col>_was_missing` flag keeps that information for the model.
- The target is never imputed. `PM2.5_obs` keeps the original observation;
  rows where it is NaN are simply not used as training/eval targets.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_FILE, CLEANING_REPORT, FFILL_LIMIT, MAX_MISSING_SHARE,
    PHYSICAL_MAX, POLLUTANTS, RAW_FILE, SMOG_MONTHS, SPIKE_MIN, SPIKE_RATIO,
    TARGET, TRAIN_YEARS,
)


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["City"] = df["City"].astype("string").str.strip()
    for col in [c for c in POLLUTANTS + ["AQI"] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["City", "Date"])
    report["rows_missing_city_or_date"] = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates(subset=["City", "Date"], keep="first")
    report["duplicate_city_date_rows"] = n1 - len(df)
    return df


def remove_impossible(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    df = df.copy()
    removed = {}
    for col, cap in PHYSICAL_MAX.items():
        if col not in df.columns:
            continue
        bad = (df[col] < 0) | (df[col] > cap)
        removed[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["impossible_values_set_to_nan"] = removed
    return df


def remove_offseason_spikes(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Set isolated off-season PM2.5 spikes to NaN (see module docstring)."""
    df = df.sort_values(["City", "Date"]).copy()
    past_median = df.groupby("City")[TARGET].transform(
        lambda s: s.shift(1).rolling(30, min_periods=10).median())
    bad = ((df[TARGET] > SPIKE_MIN) & (df[TARGET] > SPIKE_RATIO * past_median)
           & ~df["Date"].dt.month.isin(SMOG_MONTHS))
    report["offseason_spikes_set_to_nan"] = (
        df.loc[bad, ["City", "Date", TARGET]].astype(str).values.tolist())
    df.loc[bad, TARGET] = np.nan
    return df


def drop_sparse_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    lo, hi = TRAIN_YEARS
    train = df[df["Date"].dt.year.between(lo, hi)]
    share = train[[c for c in POLLUTANTS if c in df.columns]].isna().mean()
    report["missing_share_train"] = share.round(3).to_dict()
    to_drop = [c for c, s in share.items() if s > MAX_MISSING_SHARE and c != TARGET]
    report["dropped_sparse_columns"] = to_drop
    return df.drop(columns=to_drop)


def complete_calendar(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Give every city one row per day between its first and last date."""
    parts = []
    for city, g in df.groupby("City", sort=True):
        full = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        g = g.set_index("Date").reindex(full)
        g["City"] = city
        parts.append(g.rename_axis("Date").reset_index())
    out = pd.concat(parts, ignore_index=True)
    report["calendar_rows_added"] = len(out) - len(df)
    return out


def impute_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    df = df.sort_values(["City", "Date"]).copy()
    df[f"{TARGET}_obs"] = df[TARGET]  # untouched target, never imputed
    pols = [c for c in POLLUTANTS if c in df.columns]
    before = df[pols].isna().sum()
    for col in pols:
        df[f"{col}_was_missing"] = df[col].isna().astype("int8")
    df[pols] = df.groupby("City")[pols].ffill(limit=FFILL_LIMIT)
    after = df[pols].isna().sum()
    report["ffilled_values"] = (before - after).astype(int).to_dict()
    report["still_missing_after_ffill"] = after.astype(int).to_dict()
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"raw_rows": len(raw), "raw_columns": list(raw.columns)}
    df = fix_types(raw)
    df = drop_duplicates(df, report)
    df = remove_impossible(df, report)
    df = complete_calendar(df, report)
    df = remove_offseason_spikes(df, report)
    df = drop_sparse_columns(df, report)
    df = impute_features(df, report)
    df["City"] = df["City"].astype("category")
    report["clean_rows"] = len(df)
    report["clean_columns"] = list(df.columns)
    return df.reset_index(drop=True), report


def main() -> None:
    raw = pd.read_csv(RAW_FILE)
    df, report = clean(raw)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    CLEANING_REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved {len(df):,} rows to {CLEAN_FILE}")
    print(f"Report: {CLEANING_REPORT}")


if __name__ == "__main__":
    main()
