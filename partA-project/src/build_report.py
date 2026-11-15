"""Turn reports/report.md into reports/report.html and reports/report.pdf.

Usage (from partA-project/):
    python -m src.build_report

The PDF is printed by a headless Chrome or Edge (already on most machines),
so no LaTeX install is needed. Figures are referenced relative to reports/.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

from src.config import REPORTS

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.3pt; line-height: 1.42;
       color: #1d1d1f; max-width: 180mm; margin: auto; }
h1 { font-size: 20pt; margin: 0 0 4pt; color: #12355b; }
h2 { font-size: 13.5pt; color: #12355b; border-bottom: 1.5px solid #12355b; padding-bottom: 2pt;
     margin-top: 16pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; break-after: avoid; }
p, li { text-align: left; }
table { border-collapse: collapse; margin: 6pt 0 8pt; font-size: 8.8pt; break-inside: avoid; }
th, td { border: 1px solid #c9ced6; padding: 2.5pt 5pt; vertical-align: top; }
th { background: #eef2f7; text-align: left; }
img { max-width: 100%; display: block; margin: 6pt auto 2pt; break-inside: avoid; }
figure { margin: 4pt 0 8pt; break-inside: avoid; }
figcaption, .cap { font-size: 8.8pt; color: #555; text-align: center; margin-bottom: 8pt; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f4f6; padding: 0 2pt; }
pre { background: #f3f4f6; padding: 6pt; font-size: 8.5pt; overflow-x: auto; }
blockquote { border-left: 3px solid #12355b; margin: 6pt 0; padding: 2pt 8pt; background: #f6f8fb; }
.subtitle { color: #555; margin-bottom: 10pt; }
.pagebreak { break-before: page; }
"""


def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for name in ["google-chrome", "chromium", "chromium-browser", "chrome", "msedge"]:
        if shutil.which(name):
            return shutil.which(name)
    return next((c for c in candidates if Path(c).exists()), None)


def main() -> None:
    md_path = REPORTS / "report.md"
    html_path = REPORTS / "report.html"
    pdf_path = REPORTS / "report.pdf"
    body = markdown.markdown(md_path.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                         f"<title>Next-day PM2.5 forecasting report</title><style>{CSS}</style></head>"
                         f"<body>{body}</body></html>", encoding="utf-8")
    print(f"Wrote {html_path}")
    browser = find_browser()
    if browser is None:
        sys.exit("No Chrome/Edge found: open report.html in a browser and print it to PDF.")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
