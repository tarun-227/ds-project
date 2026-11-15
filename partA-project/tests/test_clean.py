"""Sanity tests for src/clean.py on a tiny hand-made dataset (no download needed)."""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean
from src.config import POLLUTANTS


@pytest.fixture
def raw():
    rows = [
        # City A: day 3 missing entirely, day 2 duplicated, one negative value
        ("A", "2017-01-01", 50.0),
        ("A", "2017-01-02", 60.0),
        ("A", "2017-01-02", 999.0),   # duplicate -> dropped (first kept)
        ("A", "2017-01-04", -5.0),    # impossible -> NaN
        ("A", "2017-01-05", 2000.0),  # above physical max -> NaN
        ("A", "2017-01-06", 80.0),
        ("B", "2017-01-01", 30.0),
        ("B", "2017-01-02", np.nan),
        ("B", "2017-01-03", 40.0),
    ]
    df = pd.DataFrame(rows, columns=["City", "Date", "PM2.5"])
    for col in POLLUTANTS:
        if col != "PM2.5":
            df[col] = 1.0
    df["AQI"] = 100.0
    df["AQI_Bucket"] = "Moderate"
    return df


def test_duplicates_removed_keep_first(raw):
    df, report = clean(raw)
    assert report["duplicate_city_date_rows"] == 1
    a = df[(df.City == "A") & (df.Date == "2017-01-02")]
    assert len(a) == 1 and a["PM2.5_obs"].iloc[0] == 60.0


def test_impossible_values_become_nan(raw):
    df, report = clean(raw)
    assert report["impossible_values_set_to_nan"]["PM2.5"] == 2
    a = df[df.City == "A"].set_index("Date")
    assert np.isnan(a.loc["2017-01-04", "PM2.5_obs"])
    assert np.isnan(a.loc["2017-01-05", "PM2.5_obs"])


def test_calendar_is_complete(raw):
    df, _ = clean(raw)
    for _, g in df.groupby("City", observed=True):
        days = g["Date"].sort_values().diff().dropna().dt.days
        assert (days == 1).all()


def test_ffill_uses_only_past(raw):
    """Gap on A 01-03..01-05 is filled from 01-02 (past), never from 01-06."""
    df, _ = clean(raw)
    a = df[df.City == "A"].set_index("Date")
    assert (a.loc["2017-01-03":"2017-01-05", "PM2.5"] == 60.0).all()
    assert a.loc["2017-01-03", "PM2.5_was_missing"] == 1


def test_target_never_imputed(raw):
    df, _ = clean(raw)
    b = df[df.City == "B"].set_index("Date")
    assert np.isnan(b.loc["2017-01-02", "PM2.5_obs"])  # target stays missing
    assert b.loc["2017-01-02", "PM2.5"] == 30.0        # feature filled from past


def _series(city, start, values):
    d = pd.date_range(start, periods=len(values), freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({"City": city, "Date": d, "PM2.5": values})
    for col in POLLUTANTS:
        if col != "PM2.5":
            df[col] = 1.0
    df["AQI"], df["AQI_Bucket"] = 100.0, "Moderate"
    return df


def test_offseason_spike_removed_but_smog_season_kept():
    calm = [40.0] * 30
    summer = _series("A", "2017-07-01", calm + [800.0, 40.0])   # August spike -> NaN
    winter = _series("B", "2017-10-15", calm + [800.0, 40.0])   # November spike -> kept
    df, report = clean(pd.concat([summer, winter], ignore_index=True))
    a = df[df.City == "A"].set_index("Date")["PM2.5_obs"]
    b = df[df.City == "B"].set_index("Date")["PM2.5_obs"]
    assert np.isnan(a.loc["2017-07-31"])
    assert b.loc["2017-11-14"] == 800.0
    assert len(report["offseason_spikes_set_to_nan"]) == 1
