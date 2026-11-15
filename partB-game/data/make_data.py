"""Build data/delhi_pm25.csv for Level 2 from Part A's raw CPCB file.

Run from partB-game/:  python data/make_data.py
The output CSV is committed, so the game runs without Part A's data.
Source: Kaggle rohanrao/air-quality-data-in-india, city_day.csv (CPCB stations).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "partA-project" / "data" / "raw" / "city_day.csv"

df = pd.read_csv(RAW, parse_dates=["Date"])
d = df[(df.City == "Delhi") & (df.Date.dt.year <= 2019)][["Date", "PM2.5"]].dropna()
d = d[(d["PM2.5"] > 0) & (d["PM2.5"] < 1000)]
d["day_of_year"] = d.Date.dt.dayofyear
d.rename(columns={"PM2.5": "pm25"}).to_csv(HERE / "delhi_pm25.csv", index=False)
print(f"Saved {len(d)} Delhi days to {HERE / 'delhi_pm25.csv'}")
