"""Run the whole Part A pipeline end to end.

Usage (from partA-project/):
    python -m src.run_all            # everything (about 10 minutes on a laptop)
    python -m src.run_all --no-eda   # skip re-executing the notebooks

Steps: download/validate data -> clean -> weather -> features -> train
(CV tuning + val + test) -> evaluate (tables + figures) -> notebooks (01 EDA, 02 modelling, 03 evaluation).
Each step is also runnable on its own (python -m src.<step>).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from src import clean, download_data, evaluate, features, train, weather
from src.config import ROOT


def run_notebooks() -> None:
    """Execute every notebooks/0*.ipynb in order, in place, with THIS Python interpreter.

    A throw-away kernelspec is written to a temp dir (via JUPYTER_PATH), so the
    notebook never runs in some other environment's 'python3' kernel.
    """
    import nbformat
    from nbclient import NotebookClient

    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "kernels" / "parta"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "partA", "language": "python"}))
        os.environ["JUPYTER_PATH"] = tmp + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        for nb_path in sorted((ROOT / "notebooks").glob("0*.ipynb")):
            print(f"  executing {nb_path.name}", flush=True)
            nb = nbformat.read(nb_path, as_version=4)
            NotebookClient(nb, kernel_name="parta", timeout=600,
                           resources={"metadata": {"path": str(nb_path.parent)}}).execute()
            nbformat.write(nb, nb_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eda", action="store_true", help="skip re-running the notebooks")
    ap.add_argument("--only-eda", action="store_true", help="only re-run the notebooks")
    args = ap.parse_args()
    steps = [("download + validate", download_data.main), ("clean", clean.main),
             ("weather", weather.main), ("features", features.main),
             ("train", train.main), ("evaluate", evaluate.main)]
    if not args.no_eda:
        steps.append(("notebooks", run_notebooks))
    if args.only_eda:
        steps = [("notebooks", run_notebooks)]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n=== {name} ===", flush=True)
        fn()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. See reports/tables and reports/figures.")


if __name__ == "__main__":
    main()
