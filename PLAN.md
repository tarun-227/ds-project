# Data Science Project + Learning Game — Build Plan

Two deliverables, graded separately:

| Part | What | Marks |
|---|---|---|
| A | Data science project (analysis is graded on how sound it is, not on model complexity) | 10 |
| B | A game where learners **discover** a DS concept by playing | 5 |

Folder layout:

```
ds-project/
├── PLAN.md                  ← this file
├── partA-project/
│   ├── README.md            ← how to run, results summary
│   ├── requirements.txt     ← pinned versions
│   ├── data/raw/            ← original download (never edited)
│   ├── data/processed/      ← output of the cleaning script
│   ├── notebooks/           ← 01_eda, 02_modelling, 03_evaluation
│   ├── src/                 ← config.py, clean.py, features.py, train.py, evaluate.py
│   ├── reports/             ← report.pdf + figures/
│   └── tests/               ← small sanity tests (no leakage, shapes, seeds)
└── partB-game/
    ├── README.md            ← how to play + learning objectives
    ├── requirements.txt
    ├── app/                 ← Streamlit game
    ├── levels/              ← level configs (datasets, goals, thresholds)
    └── playtest/            ← session logs, pre/post quiz results, peer feedback
```

---

## Part A — Data science project (10 marks)

### A1. Problem and data (1 mark)

**Suggested problem:** *Forecast the next day's PM2.5 level for Indian cities, and flag "Poor or worse" AQI days (PM2.5 > 90 µg/m³) at least 24 h ahead.*

- **Dataset:** "Air Quality Data in India (2015–2020)", built from CPCB station data (Kaggle, `rohanrao/air-quality-data-in-india`). It has about 29k city-day rows across 26 cities and 12 pollutants. Optionally join it with daily weather from Open-Meteo's historical API.
- **Why it suits the rubric:** it's real data from a public source, and it's messy. Missing values run from 10% to 60% depending on the column, there are sensor outliers, and the rows are time-ordered, so you have to deal with leakage. It's relevant to India and well past toy size.
- **Measurable goals (write these in the report intro):**
  1. Regression: beat a persistence baseline ("tomorrow = today") by **≥15% lower MAE** on a held-out 2020 test period.
  2. Classification: reach **recall ≥ 0.80** on "Poor+" days with precision ≥ 0.60.

*Backup topics if you'd rather do something else:* hospital readmission (UCI Diabetes 130-US), credit default (UCI Taiwan), or Indian crop yield. Don't use Iris or Titanic.

### A2. Data preparation and EDA (2 marks)

`notebooks/01_eda.ipynb` plus `src/clean.py`. **Give the reason for every decision in writing.**

- **Data types:** parse `Date`, make `City` categorical, fix numeric columns.
- **Duplicates:** check for duplicate (City, Date) rows and report how many.
- **Missing values:** plot missingness per column and per city (a missingno heatmap). Decide per column:
  - short gaps (≤3 days) → forward-fill or interpolate *within each city* (justification: pollution is autocorrelated);
  - long gaps → leave as NaN and add a `was_missing` flag;
  - drop columns that are more than 50% missing (e.g. Xylene), and say so.
- **Outliers:** flag physically impossible values (negative values, PM2.5 > 1000) and remove those. Keep real extremes such as Diwali and crop-burning spikes, because those days matter most. Show this with boxplots before and after.
- **Feature engineering:** lag features (t−1, t−2, t−7), 3-day and 7-day rolling means computed **only on past data**, day of week, month, season, festival flag (Diwali dates), and city target encoding fitted on the training data only.
- **Plots, each with a 2–3 line insight written under it:** seasonal pattern by month, the city comparison, a correlation heatmap of pollutants, the effect of the COVID lockdown in 2020, and an autocorrelation plot that justifies the lag features.

### A3. Modelling and analysis (2.5 marks)

`src/train.py`, with all seeds fixed in `src/config.py` (`SEED = 42`).

- **Split by time to avoid leakage:** train 2015–2018, validation 2019, test 2020 (held out until the very end). Don't use a random shuffle, and explain why in the report.
- **Approaches:**
  1. **Baseline:** persistence (tomorrow = today), plus a seasonal mean.
  2. **Linear model:** Ridge regression on the lag features. It's easy to interpret.
  3. **Tree ensemble:** Random Forest or LightGBM.
  4. *(optional)* Logistic regression and LightGBM classifiers for the Poor+ flag.
- **Tuning:** `TimeSeriesSplit` cross-validation (5 folds) on the training years, with `GridSearchCV` or `RandomizedSearchCV`. Report the mean ± std across folds.
- **Leakage guards:** fit every scaler, imputer and encoder inside an sklearn `Pipeline`, and have `tests/test_no_leakage.py` assert that no feature uses data from the target day.

### A4. Evaluation and interpretation (2 marks)

`notebooks/03_evaluation.ipynb` plus `src/evaluate.py`.

- **Metrics and why each fits:**
  - MAE, because it's in µg/m³ and easy to read.
  - RMSE, because it penalises missed spikes.
  - For the Poor+ flag: recall, precision, F1 and PR-AUC. The classes are imbalanced and missing a bad day costs more than a false alarm.
- **Comparison table:** every model on validation and test, with the improvement over baseline in %.
- **Error analysis:**
  - Error by city, by season and on festival days.
  - Residuals over time.
  - The worst 20 predictions, with a guess at why each went wrong.
  - Feature importance (SHAP or permutation).
- **Limitations, bias and next steps:**
  - Station coverage is uneven, so big cities are over-represented.
  - The 2020 lockdown is a distribution shift.
  - There's no weather data unless you join it.
  - Next steps: join weather data, try a spatial model, try probabilistic forecasts.

### A5. Reproducibility and code quality (1 mark)

- `README.md` covers: the problem, how to download the data (or a script that does it), `pip install -r requirements.txt`, and the single command `python -m src.run_all` that runs the whole pipeline end to end and regenerates the figures and tables.
- Pin the versions in `requirements.txt`, fix the seeds everywhere, put the code in a git repo with meaningful commits, and add docstrings and short comments.
- `pytest` runs the leakage and shape tests.
- **Final check:** clone into a fresh folder, set up a fresh virtual environment, and confirm it runs with no manual fixes.

### A6. Report and presentation (1.5 marks)

- **Report:** `reports/report.pdf`, 8–12 pages. Sections: Problem → Data → Cleaning and EDA → Method → Results → Error analysis → Limitations → Conclusion. Back every claim with a figure or a number.
- **Slides:** 8–10, for a 5–7 minute talk, plus a live demo that runs `run_all` or loads the saved model and predicts tomorrow for one city.
- **Viva prep:** be ready to explain why you split by time, why MAE, what leakage is and how you prevented it, what the residual plots show, and what you'd do with more time.

---

## Part B — Learning game (5 marks)

### Concept: **"Overfit or Not?" — a model-tuning puzzle game** (Streamlit + scikit-learn)

The player plays a junior data scientist. Each level gives them a **real dataset**, a few knobs to turn and a goal: *maximise the score on hidden test data*. The only score they see during play is the training score. When they submit, the game reveals the test score. The concept comes out of the gap between the two, so the player discovers it rather than being told.

### B1. Concepts and learning objectives (1.5 marks)

Write these in the game README and show them on the start screen. After playing, the learner should be able to:

| Level | Concept | Learning objective | Mechanic (real computation) |
|---|---|---|---|
| 1 | **Overfitting** | Explain why 100% training accuracy can mean worse test accuracy | Slider for polynomial degree (1–15) fitted to noisy real data. The game shows the train curve live and reveals test MSE after submit. |
| 2 | **Bias–variance trade-off** | Find the sweet spot of model complexity | Slider for decision tree `max_depth`. A bootstrap mode retrains on 20 resamples and draws the spread, so variance is visible, and plots the U-shaped test-error curve once the player has explored. |
| 3 | **Sampling and sample size** | Explain how a small or biased sample misleads | The player picks how many rows to "buy" from a limited budget, and from which region or group. A biased pick gives a confident but wrong estimate. |
| 4 | **Correlation vs causation** | Tell a confounded correlation apart from a causal effect | The player picks a policy (e.g. "ban ice cream to cut drownings") on data where temperature is a hidden confounder. A "run experiment" button randomises the treatment and shows the real effect. |
| 5 *(stretch)* | **Clustering** | See how k and feature scaling change clusters | The player places k centroids by hand and then runs k-means. The game shows inertia and silhouette, and compares scaled with unscaled features. |

Use real datasets: UCI Auto MPG, California Housing (bundled with sklearn), the Kaggle air-quality data from Part A (this reuse ties the two parts together), and a small synthetic confounder dataset for level 4 whose generating process is documented.

### B2. Game design and learning loop (1.5 marks)

- **Goal:** each level has a target, e.g. "test MSE < X" or "pick the right policy".
- **Rules:**
  - Limited attempts (3 submissions per level), or a "data budget".
  - A score made of the test metric minus a penalty for each attempt.
- **Feedback:** each submit shows the train score next to the test score, an animated chart, and a short "what just happened?" hint. The hint is triggered by **what the player did** (e.g. a high degree with a large train/test gap triggers the overfitting hint).
- **Progression:** passing a level unlocks the next one, and later levels mix concepts. Stars are awarded at 3 thresholds, and a leaderboard is stored locally.
- **Player decisions change the outcome:** nothing is pre-scripted. Every score is computed from the player's choices.
- **Reflection prompt** at the end of each level: one sentence in the player's own words. These answers are logged as evidence of learning.

### B3. Implementation (1 mark)

- **Stack:** Python, Streamlit, scikit-learn, Plotly. It runs with `streamlit run app/main.py`.
- Each level is a module in `levels/`, built on a shared `Level` base class (`setup`, `controls`, `evaluate`, `feedback`).
- **Every score is computed live.** There are no hard-coded answers, and the seeded random state is shown so runs can be reproduced.
- Keep the UI usable: one screen per level, sliders and buttons, a progress bar, and a "How to play" sidebar.
- **Logging:** each session writes `playtest/logs/<session_id>.csv` with the timestamp, level, the player's choices, the scores and the reflection text.
- Add a few `pytest` checks that every level loads and scores without errors.

### B4. Documentation and evidence of learning (1 mark)

- `README.md` covers: how to install and play, the controls, the learning objectives, and the concept each level teaches.
- **Playtest with 5 or more peers:**
  1. **Pre-quiz** with 5 conceptual questions (Google Form).
  2. They play for 10–15 minutes while logging is on.
  3. **Post-quiz** with the same concepts in different questions.
  4. A short feedback form: fun, clarity and confusion points, each on a 1–5 scale.
- `playtest/REPORT.md` holds the pre/post score table, the average gain, quotes from the reflections, one change made because of the feedback (before → after), and screenshots.

---

## Timeline (suggested 4 weeks)

| Week | Part A | Part B |
|---|---|---|
| 1 | Finalise the question and goals, download the data, set up the repo and requirements, run EDA and cleaning | Sketch the game, write the level design doc, set up the Streamlit skeleton |
| 2 | Feature engineering, baselines, time-split CV, Ridge, RF/LightGBM | Build levels 1–2 (overfitting, bias–variance) with logging |
| 3 | Tuning, evaluation, error analysis, SHAP, `run_all` script and tests | Build levels 3–4, polish, run the playtest with peers |
| 4 | Write the report, make the slides, test a fresh-clone run, rehearse the viva | Write the playtest report, iterate on feedback, finish the README |

## Rubric checklist (tick before submission)

**Part A**
- [ ] Question plus 2 measurable goals; the dataset's source, size and relevance are justified
- [ ] Missing values, duplicates, outliers and data types are handled, each with a written reason
- [ ] Engineered features are in place, and every plot has a written insight
- [ ] Baseline plus ≥2 models; time-based split; CV and tuning; no leakage (tested)
- [ ] Metrics are justified; comparison table; error analysis; limitations and bias
- [ ] README, pinned requirements and seeds; the pipeline runs end to end from a fresh clone
- [ ] Report, slides and a demo; viva questions rehearsed

**Part B**
- [ ] Learning objectives are stated in the README and in the game
- [ ] Goal, rules, feedback and progression are present; choices change the outcome
- [ ] Real data and live computation; runs without errors
- [ ] How-to-play guide plus playtest evidence (pre/post quiz, logs, feedback, one iteration)

## Next step

Confirm the topic for Part A (or pick a backup). Then start with **Week 1**: `requirements.txt`, the data download script and `01_eda.ipynb`, plus the Streamlit skeleton for the game.
