"""Evaluate saved predictions: comparison tables, goal check, error analysis, figures.

Usage (from partA-project/):
    python -m src.evaluate

Reads reports/tables/predictions.csv (written by src.train) and writes
tables to reports/tables/ and figures to reports/figures/.

Metric choices:
- MAE (ug/m3): easy to read, the headline goal is stated in it.
- RMSE: punishes big misses, i.e. the pollution spikes we most care about.
- Poor+ flag: precision, recall, F1 at the threshold chosen on 2019, and
  PR-AUC (threshold-free). Accuracy is NOT used: ~8-20% of days are Poor+,
  so "always clean" would already score >80%.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score,
    recall_score,
)

from src.config import (
    FEATURE_FILE, FIGURES, GOAL_MAE_IMPROVEMENT, GOAL_PRECISION, GOAL_RECALL,
    MODELS, N_JOBS, SEED, TABLES,
)
from src.features import FEATURES

LOCKDOWN = pd.Timestamp("2020-03-25")
REG_NAMES = {"persistence": "Persistence (baseline)", "seasonal_mean": "Seasonal mean (baseline)",
             "ridge": "Ridge", "random_forest": "Random forest", "lightgbm": "LightGBM"}
CLF_NAMES = {"persistence": "Persistence rule (today > 90)", "logistic": "Logistic regression",
             "lightgbm_clf": "LightGBM classifier",
             "lightgbm_reg_threshold": "LightGBM regressor + threshold"}
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def paired_bootstrap_gain(df, col, n=2000, seed=SEED):
    """95% CI for the % MAE gain over persistence, resampling whole DAYS (cities on
    the same day are correlated, so resampling rows would make the CI too narrow)."""
    rng = np.random.default_rng(seed)
    g = df.assign(e_m=np.abs(df[col] - df.y), e_p=np.abs(df.reg_persistence - df.y)) \
          .groupby("target_date")[["e_m", "e_p"]].sum()
    days = g.to_numpy()
    gains = []
    for _ in range(n):
        s = days[rng.integers(0, len(days), len(days))].sum(axis=0)
        gains.append(100 * (1 - s[0] / s[1]))
    return np.percentile(gains, [2.5, 97.5])


# ---------------------------------------------------------------- tables
def regression_table(p):
    rows = []
    for split in ["val", "test"]:
        q = p[p.split == split]
        base = mae(q.reg_persistence, q.y)
        for key, name in REG_NAMES.items():
            col = f"reg_{key}"
            row = {"split": split, "model": name, "MAE": mae(q[col], q.y),
                   "RMSE": rmse(q[col], q.y),
                   "MAE on Poor+ days": mae(q.loc[q.y > 90, col], q.loc[q.y > 90, "y"])}
            row["MAE gain vs persistence %"] = 100 * (1 - row["MAE"] / base)
            if key not in ("persistence", "seasonal_mean"):
                lo, hi = paired_bootstrap_gain(q, col)
                row["gain 95% CI"] = f"[{lo:.1f}, {hi:.1f}]"
            rows.append(row)
    return pd.DataFrame(rows).round(2)


def classification_table(p, thresholds):
    rows = []
    for split in ["val", "test"]:
        q = p[p.split == split]
        y = q.y_poor.astype(int)
        for key, name in CLF_NAMES.items():
            score = q[f"clf_{key}"]
            pred = (score > thresholds[key]).astype(int)
            rows.append({"split": split, "model": name, "threshold": thresholds[key],
                         "precision": precision_score(y, pred, zero_division=0),
                         "recall": recall_score(y, pred), "F1": f1_score(y, pred),
                         "PR-AUC": average_precision_score(y, score),
                         "Poor+ rate": y.mean()})
    return pd.DataFrame(rows).round(3)


# ---------------------------------------------------------------- figures
def fig_model_comparison(reg):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.barplot(data=reg, y="model", x="MAE", hue="split", ax=ax, palette=["#7aa6d8", "#1f4e8c"])
    ax.set_xlabel("MAE (µg/m³, lower is better)"); ax.set_ylabel("")
    ax.set_title("Next-day PM2.5: validation (2019) and test (2020) MAE")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8, padding=2)
    save(fig, "model_comparison")


def fig_pred_vs_actual(q, col):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for a, c, t in [(ax[0], "reg_persistence", "Persistence"), (ax[1], col, "Best model")]:
        a.scatter(q.y, q[c], s=4, alpha=0.3)
        m = max(q.y.max(), q[c].max())
        a.plot([0, m], [0, m], "r--", lw=1)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title(f"{t}: MAE {mae(q[c], q.y):.1f}"); a.set_xlabel("actual PM2.5")
    ax[0].set_ylabel("predicted PM2.5")
    fig.suptitle("Test 2020: predicted vs actual (log scale)")
    save(fig, "pred_vs_actual_test")


def fig_residuals_over_time(p, col):
    q = p.assign(err_model=(p[col] - p.y).abs(), err_pers=(p.reg_persistence - p.y).abs())
    d = q.groupby("target_date")[["err_model", "err_pers"]].mean().rolling(14, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(d.index, d.err_pers, label="persistence", color="#999999")
    ax.plot(d.index, d.err_model, label="best model", color="#1f4e8c")
    ax.axvline(LOCKDOWN, color="red", ls="--"); ax.text(LOCKDOWN, ax.get_ylim()[1] * 0.9, " lockdown", color="red")
    ax.axvline(pd.Timestamp("2020-01-01"), color="k", lw=0.8); ax.text(pd.Timestamp("2020-01-03"), ax.get_ylim()[1] * 0.8, "test →")
    ax.set_ylabel("abs. error, 14-day mean"); ax.legend(); ax.set_title("Error over time (val 2019 + test 2020)")
    save(fig, "residuals_over_time")


def fig_signed_residuals(q, col):
    r = q[col] - q.y
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(r.clip(-150, 150), bins=80); ax[0].axvline(0, c="r")
    ax[0].set_title(f"Residuals (pred − actual), median {r.median():.1f}")
    ax[1].scatter(q.y, r, s=4, alpha=0.3); ax[1].axhline(0, c="r")
    ax[1].set_xlabel("actual PM2.5"); ax[1].set_ylabel("residual")
    ax[1].set_title("Residual vs actual: big days are under-predicted")
    save(fig, "residuals_distribution_test")


def error_by(q, col, by):
    return q.assign(model=(q[col] - q.y).abs(), persistence=(q.reg_persistence - q.y).abs()) \
            .groupby(by, observed=True)[["model", "persistence"]].mean() \
            .assign(n=q.groupby(by, observed=True).size(),
                    mean_pm25=q.groupby(by, observed=True).y.mean())


def fig_error_by_city(q, col, train_cities):
    e = error_by(q, col, "City").sort_values("model")
    e["seen_in_training"] = e.index.isin(train_cities)
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(e))
    ax.barh(y - 0.2, e.persistence, 0.4, label="persistence", color="#bbbbbb")
    ax.barh(y + 0.2, e.model, 0.4, label="best model", color="#1f4e8c")
    ax.set_yticks(y, [f"{c}{'' if s else ' *'}" for c, s in zip(e.index, e.seen_in_training)])
    ax.set_xlabel("test MAE (µg/m³)"); ax.legend()
    ax.set_title("Test error by city (* = city not in training years)")
    save(fig, "error_by_city_test")
    return e.round(2)


def fig_error_by_season(p, col):
    e = error_by(p, col, ["split", "season"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    m = e.melt(id_vars=["split", "season"], value_vars=["model", "persistence"])
    m["group"] = m.split + " · " + m.season
    sns.barplot(data=m, x="group", y="value", hue="variable", ax=ax, palette=["#1f4e8c", "#bbbbbb"])
    ax.set_xlabel(""); ax.set_ylabel("MAE"); ax.tick_params(axis="x", rotation=30)
    ax.set_title("Error by season (2020 test has no post-monsoon)")
    save(fig, "error_by_season")
    return e.round(2)


def fig_pr_curves(p, thresholds):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a, split in zip(ax, ["val", "test"]):
        q = p[p.split == split]
        y = q.y_poor.astype(int)
        for key, name in CLF_NAMES.items():
            pr, rc, _ = precision_recall_curve(y, q[f"clf_{key}"])
            a.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, q[f'clf_{key}']):.2f})")
        a.axvline(GOAL_RECALL, ls=":", c="k"); a.axhline(GOAL_PRECISION, ls=":", c="k")
        a.set_xlabel("recall"); a.set_title(f"{split}: Poor+ day detection (base rate {y.mean():.1%})")
    ax[0].set_ylabel("precision"); ax[1].legend(fontsize=8, loc="lower left")
    save(fig, "pr_curves")


def fig_city_timeseries(q, col, city="Delhi"):
    d = q[q.City == city].sort_values("target_date")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(d.target_date, d.y, label="actual", color="k", lw=1.5)
    ax.plot(d.target_date, d[col], label="best model", color="#1f4e8c", lw=1.2)
    ax.plot(d.target_date, d.reg_persistence, label="persistence", color="#999999", lw=1, ls="--")
    ax.axhline(90, c="r", ls=":", lw=1); ax.axvline(LOCKDOWN, c="red", ls="--", lw=1)
    ax.set_title(f"{city}, test 2020: next-day forecasts"); ax.set_ylabel("PM2.5"); ax.legend()
    save(fig, "delhi_test_timeseries")


def fig_permutation_importance(feats):
    model = joblib.load(MODELS / "reg_lightgbm_trainonly.joblib")
    va = feats[feats.split == "val"]
    r = permutation_importance(model, va[FEATURES], va["y"], scoring="neg_mean_absolute_error",
                               n_repeats=5, random_state=SEED, n_jobs=N_JOBS)
    imp = pd.DataFrame({"feature": FEATURES, "MAE increase": r.importances_mean,
                        "std": r.importances_std}).sort_values("MAE increase", ascending=False)
    # pm_t is both a feature and the base the model adds its correction to, so shuffling
    # it destroys the forecast. Plot the other features so their sizes are visible.
    pm = imp.loc[imp.feature == "pm_t", "MAE increase"].iloc[0]
    top = imp[imp.feature != "pm_t"].head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature[::-1], top["MAE increase"][::-1], xerr=top["std"][::-1], color="#1f4e8c")
    ax.set_xlabel("increase in val MAE when shuffled (µg/m³)")
    ax.set_title("Permutation importance (LightGBM, 2019)\n"
                 f"not shown: pm_t (today's PM2.5) = {pm:.1f}")
    save(fig, "permutation_importance")
    return imp.round(3)


def fig_ablation(abl):
    fig, ax = plt.subplots(figsize=(8, 3))
    m = abl.melt(id_vars="features", value_vars=["val_gain_vs_persistence_%", "test_gain_vs_persistence_%"])
    m["variable"] = m.variable.str.split("_").str[0]
    sns.barplot(data=m, y="features", x="value", hue="variable", ax=ax, palette=["#7aa6d8", "#1f4e8c"])
    ax.axvline(100 * GOAL_MAE_IMPROVEMENT, c="r", ls="--"); ax.text(100 * GOAL_MAE_IMPROVEMENT, -0.6, " goal", color="r")
    ax.set_xlabel("MAE gain over persistence (%)"); ax.set_ylabel("")
    ax.legend(loc="lower right", title=None)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8, padding=2)
    save(fig, "ablation_features")


# ---------------------------------------------------------------- main
def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(TABLES / "predictions.csv", parse_dates=["Date", "target_date"])
    feats = pd.read_csv(FEATURE_FILE, parse_dates=["Date", "target_date"])
    thresholds = json.loads((TABLES / "thresholds.json").read_text())

    reg = regression_table(p)
    clf = classification_table(p, thresholds)
    reg.to_csv(TABLES / "regression_results.csv", index=False)
    clf.to_csv(TABLES / "classification_results.csv", index=False)

    # Best ML regressor chosen on VALIDATION only.
    ml = ["ridge", "random_forest", "lightgbm"]
    vq = p[p.split == "val"]
    best = min(ml, key=lambda k: mae(vq[f"reg_{k}"], vq.y))
    col = f"reg_{best}"
    tq = p[p.split == "test"]

    # Classification: choose on val by F1 among models that reach the recall goal.
    cv = clf[clf.split == "val"]
    best_clf = cv.sort_values("F1", ascending=False).iloc[0]["model"]
    ct = clf[(clf.split == "test") & (clf.model == best_clf)].iloc[0]
    test_gain = reg[(reg.split == "test") & (reg.model == REG_NAMES[best])].iloc[0]
    goals = {
        "best_regressor_by_val": REG_NAMES[best],
        "test_MAE_gain_vs_persistence_%": float(test_gain["MAE gain vs persistence %"]),
        "test_gain_95CI": test_gain.get("gain 95% CI"),
        "goal1_met (>= 15%)": bool(test_gain["MAE gain vs persistence %"] >= 100 * GOAL_MAE_IMPROVEMENT),
        "best_classifier_by_val": best_clf,
        "test_recall": float(ct.recall), "test_precision": float(ct.precision),
        "goal2_met (recall>=0.80 & precision>=0.60)": bool(ct.recall >= GOAL_RECALL and ct.precision >= GOAL_PRECISION),
    }

    # Error analysis
    train_cities = set(feats.loc[feats.split == "train", "City"])
    by_city = fig_error_by_city(tq, col, train_cities)
    by_city.to_csv(TABLES / "error_by_city_test.csv")
    fig_error_by_season(p, col).to_csv(TABLES / "error_by_season.csv", index=False)
    unseen = tq[~tq.City.isin(train_cities)]
    seen = tq[tq.City.isin(train_cities)]
    goals["test_MAE_cities_seen_in_training"] = round(mae(seen[col], seen.y), 2)
    goals["test_MAE_cities_new_in_2019_20"] = round(mae(unseen[col], unseen.y), 2)
    lock = {}
    for label, m in [("2020 before lockdown", tq.target_date < LOCKDOWN),
                     ("2020 lockdown onwards", tq.target_date >= LOCKDOWN)]:
        lock[label] = {"MAE model": round(mae(tq.loc[m, col], tq.loc[m, "y"]), 2),
                       "MAE persistence": round(mae(tq.loc[m, "reg_persistence"], tq.loc[m, "y"]), 2),
                       "mean PM2.5": round(float(tq.loc[m, "y"].mean()), 1),
                       "Poor+ rate": round(float((tq.loc[m, "y"] > 90).mean()), 3)}
    goals["lockdown"] = lock
    diw = vq[vq.diwali == 1]
    goals["val_2019_diwali_window"] = {"n": int(len(diw)),
                                       "MAE model": round(mae(diw[col], diw.y), 2),
                                       "MAE persistence": round(mae(diw.reg_persistence, diw.y), 2)}
    goals["test_median_residual"] = round(float((tq[col] - tq.y).median()), 2)
    goals["test_mean_residual_on_Poor+_days"] = round(float((tq[col] - tq.y)[tq.y > 90].mean()), 2)

    worst = tq.assign(abs_error=(tq[col] - tq.y).abs(), prediction=tq[col], today=tq.pm_t) \
              .nlargest(20, "abs_error")[["City", "target_date", "today", "y", "prediction", "abs_error"]]
    worst.round(1).to_csv(TABLES / "worst20_test.csv", index=False)

    # Figures
    fig_model_comparison(reg)
    fig_pred_vs_actual(tq, col)
    fig_residuals_over_time(p, col)
    fig_signed_residuals(tq, col)
    fig_pr_curves(p, thresholds)
    fig_city_timeseries(tq, col)
    fig_permutation_importance(feats).to_csv(TABLES / "permutation_importance.csv", index=False)
    if (TABLES / "ablation_features.csv").exists():
        fig_ablation(pd.read_csv(TABLES / "ablation_features.csv"))

    (TABLES / "summary.json").write_text(json.dumps(goals, indent=2, default=str))
    pd.set_option("display.width", 160)
    print(reg.to_string(index=False)); print(); print(clf.to_string(index=False)); print()
    print(json.dumps(goals, indent=2, default=str))


if __name__ == "__main__":
    main()
