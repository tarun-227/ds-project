"""Download daily historical weather per city from NASA POWER (free, no API key).

Usage (from partA-project/):
    python -m src.weather

Why: PM2.5 is driven by emissions AND by how well the air disperses them.
Wind, rain, temperature and humidity control dispersion, so they explain
day-to-day changes that pollutant lags cannot.

Source: NASA POWER daily point API (MERRA-2 reanalysis, ~0.5 degree grid),
https://power.larc.nasa.gov/docs/services/api/temporal/daily/ . One request
per city, cached to data/raw/weather/<city>.csv so reruns are offline.
(Open-Meteo was tried first but its free daily quota was exhausted.)
"""
from __future__ import annotations

import time
import pandas as pd
import requests  # uses certifi's CA bundle (urllib failed on an outdated system store)

from src.config import DATA_RAW, WEATHER_FILE

# Approximate city-centre coordinates (lat, lon).
CITY_COORDS = {
    "Ahmedabad": (23.03, 72.58), "Aizawl": (23.73, 92.72), "Amaravati": (16.51, 80.52),
    "Amritsar": (31.63, 74.87), "Bengaluru": (12.97, 77.59), "Bhopal": (23.26, 77.41),
    "Brajrajnagar": (21.82, 83.92), "Chandigarh": (30.73, 76.78), "Chennai": (13.08, 80.27),
    "Coimbatore": (11.02, 76.96), "Delhi": (28.61, 77.21), "Ernakulam": (9.98, 76.28),
    "Gurugram": (28.46, 77.03), "Guwahati": (26.14, 91.74), "Hyderabad": (17.39, 78.49),
    "Jaipur": (26.91, 75.79), "Jorapokhar": (23.71, 86.41), "Kochi": (9.93, 76.27),
    "Kolkata": (22.57, 88.36), "Lucknow": (26.85, 80.95), "Mumbai": (19.08, 72.88),
    "Patna": (25.59, 85.14), "Shillong": (25.58, 91.89), "Talcher": (20.95, 85.23),
    "Thiruvananthapuram": (8.52, 76.94), "Visakhapatnam": (17.69, 83.22),
}
# NASA POWER parameter -> our column name
DAILY_VARS = {
    "T2M": "temp_mean", "T2M_MAX": "temp_max", "T2M_MIN": "temp_min",
    "PRECTOTCORR": "precip_mm", "WS10M": "wind_mean", "WS10M_MAX": "wind_max",
    "WD10M": "wind_dir", "RH2M": "humidity", "PS": "pressure_kpa",
}
START, END = "20150101", "20200702"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
CACHE = DATA_RAW / "weather"


def fetch_city(city: str) -> pd.DataFrame:
    path = CACHE / f"{city}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])
    lat, lon = CITY_COORDS[city]
    params = {"parameters": ",".join(DAILY_VARS), "community": "AG",
              "latitude": lat, "longitude": lon, "start": START, "end": END,
              "format": "JSON"}
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, timeout=120)
        except requests.RequestException as exc:
            r = None
            print(f"  {city}: {exc}")
        if r is not None and r.status_code == 200:
            data = r.json()["properties"]["parameter"]
            break
        if attempt == 5:
            raise RuntimeError(f"weather download failed for {city}")
        wait = 20 * (attempt + 1)
        print(f"  {city}: retry in {wait}s")
        time.sleep(wait)
    df = pd.DataFrame(data).rename(columns=DAILY_VARS)
    df = df.replace(-999.0, float("nan"))  # POWER fill value
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename_axis("Date").reset_index()
    df.insert(0, "City", city)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(1)  # be polite to the free API
    return df


def main() -> None:
    frames = []
    for city in CITY_COORDS:
        frames.append(fetch_city(city))
        print(f"  {city}: {len(frames[-1])} days")
    out = pd.concat(frames, ignore_index=True)
    WEATHER_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WEATHER_FILE, index=False)
    print(f"Saved {len(out):,} rows to {WEATHER_FILE}")


if __name__ == "__main__":
    main()
