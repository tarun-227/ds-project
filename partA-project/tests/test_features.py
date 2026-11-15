"""Leakage and correctness tests for src/features.py and the split (no download needed)."""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean
from src.config import PHYSICAL_MAX, POLLUTANTS
from src.features import FEATURES, NEXT_WEATHER_FEATURES, assign_split, build_features
from src.train import time_folds


def make_raw(n_days=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for city in ["A", "B"]:
        for d in pd.date_range("2018-11-01", periods=n_days, freq="D"):
            rows.append([city, d.strftime("%Y-%m-%d")] +
                        [rng.uniform(1, PHYSICAL_MAX[c] / 5) for c in POLLUTANTS])
    raw = pd.DataFrame(rows, columns=["City", "Date"] + POLLUTANTS)
    raw["AQI"], raw["AQI_Bucket"] = 100.0, "Moderate"
    return raw


def make_weather(raw):
    d = raw[["City", "Date"]].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    rng = np.random.default_rng(1)
    for c in ["temp_mean", "temp_max", "temp_min", "precip_mm", "wind_mean", "wind_max",
              "wind_dir", "humidity", "pressure_kpa"]:
        d[c] = rng.uniform(0, 30, len(d))
    return d


@pytest.fixture
def feats():
    raw = make_raw()
    return raw, build_features(clean(raw)[0], make_weather(raw))


def test_target_is_next_day_observation(feats):
    raw, f = feats
    r = raw.assign(Date=pd.to_datetime(raw.Date)).set_index(["City", "Date"])["PM2.5"]
    for _, row in f.sample(20, random_state=0).iterrows():
        assert row.y == pytest.approx(r[(row.City, row.Date + pd.Timedelta(days=1))])
        assert row.pm_t == pytest.approx(r[(row.City, row.Date)])


def test_features_do_not_use_the_future(feats):
    """Changing every value AFTER day t must not change any feature row for day <= t."""
    raw, f = feats
    cutoff = pd.Timestamp("2018-12-15")
    raw2 = raw.copy()
    later = pd.to_datetime(raw2.Date) > cutoff
    raw2.loc[later, POLLUTANTS] = 5.0  # plausible for every pollutant, but very different
    w2 = make_weather(raw2)
    w2.loc[w2.Date > cutoff + pd.Timedelta(days=1), w2.columns[2:]] = -1
    f2 = build_features(clean(raw2)[0], w2)
    a = f[f.Date <= cutoff].set_index(["City", "Date"])[FEATURES]
    b = f2[f2.Date <= cutoff].set_index(["City", "Date"])[FEATURES]
    pd.testing.assert_frame_equal(a, b.loc[a.index])


def test_oracle_weather_is_not_in_main_features():
    assert not set(NEXT_WEATHER_FEATURES) & set(FEATURES)
    assert "y" not in FEATURES and "y_poor" not in FEATURES


def test_split_by_target_year():
    d = pd.Series(pd.to_datetime(["2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01", "2014-12-31"]))
    assert assign_split(d).tolist()[:4] == ["train", "val", "val", "test"]
    assert pd.isna(assign_split(d).iloc[4])


def test_time_folds_never_validate_on_the_past():
    dates = pd.Series(pd.date_range("2015-01-01", periods=400).repeat(3))
    for tr, va in time_folds(dates, 5):
        assert dates.iloc[tr].max() < dates.iloc[va].min()
