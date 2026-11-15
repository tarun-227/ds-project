"""Headless UI test: play through both levels with Streamlit's AppTest (no browser)."""
import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
LOGS = Path(__file__).resolve().parents[1] / "playtest" / "logs"


def click(at, label_start):
    btn = next(b for b in at.button if b.label.startswith(label_start))
    btn.click()
    return at.run(timeout=120)


def test_play_through_both_levels():
    before = set(LOGS.glob("*.csv")) if LOGS.exists() else set()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.text_input[0].input("pytest-bot").run()
    assert at.slider[0].value == 10                     # tempting overfit default
    at = click(at, "Submit degree 10")
    assert "overfitting" in at.info[0].value.lower()     # hint reacts to the choice
    at.slider[0].set_value(3).run()
    at = click(at, "Submit degree 3")
    assert not at.exception
    # Level 2 is now unlocked
    at.radio[0].set_value(at.radio[0].options[1]).run()
    assert "Too Simple" in at.header[0].value
    at = click(at, "🎲 Draw another sample")
    at.checkbox[0].check().run()
    at = click(at, "Submit depth 8")
    assert not at.exception and "variance" in at.info[0].value.lower()
    # remove the log this test wrote, so it never counts as playtest evidence
    for f in set(LOGS.glob("*.csv")) - before:
        f.unlink()
