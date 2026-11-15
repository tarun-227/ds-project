"""Get the raw dataset into data/raw/city_day.csv and check it.

Usage (from partA-project/):
    python -m src.download_data

Uses the Kaggle API if it is installed and configured (~/.kaggle/kaggle.json).
Otherwise it prints the manual steps. Either way it then validates the file.
"""
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

from src.config import DATA_RAW, KAGGLE_DATASET, POLLUTANTS, RAW_FILE

EXPECTED_COLUMNS = {"City", "Date", "AQI", "AQI_Bucket", *POLLUTANTS}
MIN_ROWS = 25_000


def _unzip_if_needed() -> None:
    zipped = DATA_RAW / "city_day.csv.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as z:
            z.extractall(DATA_RAW)
        zipped.unlink()


def try_kaggle_download() -> bool:
    """Download city_day.csv via the Kaggle Python API (falls back to the CLI).

    Needs ~/.kaggle/kaggle.json. Returns True on success.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(KAGGLE_DATASET, "city_day.csv", path=str(DATA_RAW))
    except Exception as exc:  # no package, no credentials, no network
        print(f"Kaggle Python API failed ({exc}); trying the CLI.")
        if shutil.which("kaggle") is None:
            return False
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
               "-f", "city_day.csv", "-p", str(DATA_RAW)]
        if subprocess.run(cmd).returncode != 0:
            return False
    _unzip_if_needed()
    return RAW_FILE.exists()


def validate() -> pd.DataFrame:
    """Fail loudly if the file is missing or not the dataset we expect."""
    df = pd.read_csv(RAW_FILE)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"city_day.csv is missing columns: {sorted(missing)}")
    if len(df) < MIN_ROWS:
        sys.exit(f"Only {len(df)} rows; expected at least {MIN_ROWS}.")
    print(f"OK: {RAW_FILE} has {len(df):,} rows, {df['City'].nunique()} cities, "
          f"{df['Date'].min()} to {df['Date'].max()}.")
    return df


def main() -> None:
    _unzip_if_needed()
    if not RAW_FILE.exists() and not try_kaggle_download():
        print(
            "Could not download automatically. Do this once by hand:\n"
            f"  1. Open https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            "  2. Download city_day.csv (free Kaggle login needed)\n"
            f"  3. Put it at {RAW_FILE}\n"
            "  4. Re-run: python -m src.download_data"
        )
        sys.exit(1)
    validate()


if __name__ == "__main__":
    main()
