"""Append one CSV row per game event to playtest/logs/<session_id>.csv.

These logs are the playtest evidence for the report (choices, scores, reflections).
"""
import csv
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "playtest" / "logs"


def make_logger(session_id: str, player: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{session_id}.csv"

    def log(level: str, event: str, **details):
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "session_id", "player", "level", "event", "details"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), session_id, player,
                        level, event, json.dumps(details)])

    return log
