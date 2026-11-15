"""Central configuration: paths, seeds, split years and cleaning thresholds.

Every other module imports from here so a single change propagates everywhere.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RAW_FILE = DATA_RAW / "city_day.csv"
CLEAN_FILE = DATA_PROCESSED / "city_day_clean.csv"
CLEANING_REPORT = REPORTS / "cleaning_report.json"
WEATHER_FILE = DATA_PROCESSED / "weather_daily.csv"

KAGGLE_DATASET = "rohanrao/air-quality-data-in-india"

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- problem definition
TARGET = "PM2.5"
# CPCB AQI breakpoint: PM2.5 (24h avg) > 90 ug/m3 is "Poor" or worse.
POOR_THRESHOLD = 90.0

# Time-based split (no shuffling -> no look-ahead leakage).
TRAIN_YEARS = (2015, 2018)  # inclusive
VAL_YEAR = 2019
TEST_YEAR = 2020

# ---------------------------------------------------------------- cleaning rules
POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

# Values above these are physically implausible for a 24-hour average
# (sensor faults, unit errors) and are set to NaN. Genuine extremes such as
# Diwali or crop-burning spikes in Delhi stay well below these limits.
PHYSICAL_MAX = {
    "PM2.5": 1000, "PM10": 1500, "NO": 500, "NO2": 500, "NOx": 600,
    "NH3": 500, "CO": 50, "SO2": 500, "O3": 500,
    "Benzene": 200, "Toluene": 300, "Xylene": 200,
}

# Isolated PM2.5 spikes outside the smog season (Oct-Jan) are treated as sensor
# faults: value > SPIKE_MIN and > SPIKE_RATIO x the city's median of the PREVIOUS
# 30 days (past-only, so no future information is used to clean day t).
SPIKE_MIN = 300.0
SPIKE_RATIO = 6.0
SMOG_MONTHS = (10, 11, 12, 1)

# Drop a pollutant column if more than this share is missing in the TRAIN years.
MAX_MISSING_SHARE = 0.5

# Forward-fill feature gaps of at most this many days (past values only).
FFILL_LIMIT = 3

# ---------------------------------------------------------------- modelling outputs
FEATURE_FILE = DATA_PROCESSED / "features.csv"
MODELS = ROOT / "models"
TABLES = REPORTS / "tables"
CV_FOLDS = 5
# Parallelism lives at ONE level (the hyperparameter search). Models inside it are
# single-threaded; nesting all-core jobs oversubscribes the CPU and can kill workers.
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
# Goals stated in the report introduction (checked in evaluate.py).
GOAL_MAE_IMPROVEMENT = 0.15   # >= 15% lower MAE than persistence on test
GOAL_RECALL = 0.80
GOAL_PRECISION = 0.60
