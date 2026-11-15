"""Train, tune and save all models; write predictions for evaluation.

Usage (from partA-project/):
    python -m src.train

Protocol (no leakage, no peeking at test):
1. Hyperparameters are tuned with 5-fold *time-ordered* CV on the train years
   only (folds are split by date, so every validation fold is later than its
   training fold).
2. The tuned model is refit on all train years and scored on 2019 (validation).
   Model choice and the classification threshold are decided here.
3. Only then is each model refit on train+val and scored once on 2020 (test).

All preprocessing (imputation, scaling, one-hot) lives inside sklearn
Pipelines, so it is re-fitted on the training part of every CV fold.

The ML regressors predict the CHANGE from today (y - pm_t) and add it back.
Persistence is a very strong baseline for daily PM2.5, so learning the
correction to it is easier than learning the level from scratch (and lets
the trees extrapolate beyond the range of levels seen in training).
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CV_FOLDS, FEATURE_FILE, GOAL_RECALL, MODELS, N_JOBS, POOR_THRESHOLD, SEED, TABLES,
)
from src.features import (
    CATEGORICAL_FEATURES, FEATURES, NEXT_WEATHER_FEATURES, NUMERIC_FEATURES,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------- CV splitter
def time_folds(dates: pd.Series, n_splits: int = CV_FOLDS):
    """Expanding-window folds split on unique dates (all cities of a day stay together)."""
    days = np.sort(dates.unique())
    edges = np.linspace(0, len(days), n_splits + 2).astype(int)[1:]
    idx = np.arange(len(dates))
    for k in range(n_splits):
        cut, end = days[edges[k] - 1], days[edges[k + 1] - 1]
        tr = idx[dates.values <= cut]
        va = idx[(dates.values > cut) & (dates.values <= end)]
        yield tr, va


# ---------------------------------------------------------------- baselines
class Persistence(BaseEstimator, RegressorMixin):
    """Tomorrow = today. Needs no fitting."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["pm_t"].to_numpy()


class SeasonalMean(BaseEstimator, RegressorMixin):
    """Mean of the target per (city, month) in the training data.

    Falls back to the month mean, then the global mean, for unseen cities.
    """

    def fit(self, X, y):
        d = pd.DataFrame({"City": X["City"].values, "month": X["month"].values, "y": y})
        self.city_month_ = d.groupby(["City", "month"])["y"].mean()
        self.month_ = d.groupby("month")["y"].mean()
        self.global_ = float(d["y"].mean())
        return self

    def predict(self, X):
        keys = pd.MultiIndex.from_arrays([X["City"].values, X["month"].values])
        p = self.city_month_.reindex(keys).to_numpy()
        fb = self.month_.reindex(X["month"].values).to_numpy()
        p = np.where(np.isnan(p), fb, p)
        return np.where(np.isnan(p), self.global_, p)


class DeltaTarget(BaseEstimator, RegressorMixin):
    """Fit `estimator` on y - pm_t; predict pm_t + estimated change."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - X["pm_t"].to_numpy())
        return self

    def predict(self, X):
        return X["pm_t"].to_numpy() + self.estimator_.predict(X)


# ---------------------------------------------------------------- pipelines
def preprocessor(scale: bool, numeric=None) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric or NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def regression_searches():
    """(name, search object) for every tunable regressor."""
    ridge = DeltaTarget(Pipeline([
        ("prep", preprocessor(scale=True)), ("model", Ridge())]))
    rf = DeltaTarget(Pipeline([
        ("prep", preprocessor(scale=False)),
        ("model", RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1)),
    ]))
    lgbm = DeltaTarget(Pipeline([
        ("prep", preprocessor(scale=False)),
        ("model", LGBMRegressor(random_state=SEED, n_estimators=500, verbose=-1, n_jobs=1)),
    ]))
    p = "estimator__model__"
    return [
        ("ridge", ridge, {p + "alpha": np.logspace(-2, 4, 13)}, None),
        ("random_forest", rf, {
            p + "max_depth": [8, 12, 16, None],
            p + "min_samples_leaf": [5, 10, 20, 40],
            p + "max_features": [0.3, 0.5],
        }, 8),
        ("lightgbm", lgbm, {
            p + "objective": ["l2", "l1", "huber"],
            p + "learning_rate": loguniform(0.01, 0.1),
            p + "num_leaves": randint(8, 64),
            p + "min_child_samples": randint(10, 80),
            p + "subsample": uniform(0.6, 0.4),
            p + "subsample_freq": [1],
            p + "colsample_bytree": uniform(0.5, 0.5),
            p + "reg_lambda": loguniform(1e-3, 10),
        }, 25),
    ]


def classification_searches():
    logit = Pipeline([
        ("prep", preprocessor(scale=True)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    lgbm = Pipeline([
        ("prep", preprocessor(scale=False)),
        ("model", LGBMClassifier(random_state=SEED, n_estimators=400, verbose=-1, n_jobs=1,
                                 is_unbalance=True)),
    ])
    return [
        ("logistic", logit, {"model__C": np.logspace(-3, 2, 11)}, None),
        ("lightgbm_clf", lgbm, {
            "model__learning_rate": loguniform(0.01, 0.1),
            "model__num_leaves": randint(8, 64),
            "model__min_child_samples": randint(10, 80),
            "model__subsample": uniform(0.6, 0.4),
            "model__subsample_freq": [1],
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(1e-3, 10),
        }, 25),
    ]


def make_search(est, grid, n_iter, folds, scoring):
    if n_iter is None:
        return GridSearchCV(est, grid, cv=folds, scoring=scoring, n_jobs=N_JOBS)
    return RandomizedSearchCV(est, grid, n_iter=n_iter, cv=folds, scoring=scoring,
                              random_state=SEED, n_jobs=N_JOBS)


def threshold_for_recall(y_true, score, target_recall=GOAL_RECALL) -> float:
    """Largest threshold whose recall is still >= target (i.e. best precision at that recall)."""
    prec, rec, thr = precision_recall_curve(y_true, score)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_FILE, parse_dates=["Date", "target_date"])
    df = df.sort_values(["Date", "City"]).reset_index(drop=True)
    X_cols = FEATURES + ["City"]
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    trva = pd.concat([tr, va], ignore_index=True)
    folds = list(time_folds(tr["Date"]))

    preds = pd.concat([va, te], ignore_index=True)[
        ["City", "Date", "target_date", "split", "y", "y_poor", "pm_t", "season", "diwali"]]
    cv_rows, best_params = [], {}

    def record_cv(name, task, search, metric):
        i = search.best_index_
        res = search.cv_results_
        cv_rows.append({"task": task, "model": name, "metric": metric,
                        "cv_mean": -res["mean_test_score"][i] if metric == "MAE" else res["mean_test_score"][i],
                        "cv_std": res["std_test_score"][i]})
        best_params[name] = {k: (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()}

    # ---- regression baselines (persistence needs no training; seasonal mean is fitted)
    for name, est in [("persistence", Persistence()), ("seasonal_mean", SeasonalMean())]:
        maes = []
        for a, b in folds:
            m = est.fit(tr.loc[a, X_cols], tr.loc[a, "y"].to_numpy())
            maes.append(np.mean(np.abs(m.predict(tr.loc[b, X_cols]) - tr.loc[b, "y"])))
        cv_rows.append({"task": "regression", "model": name, "metric": "MAE",
                        "cv_mean": np.mean(maes), "cv_std": np.std(maes)})
        preds.loc[preds.split == "val", f"reg_{name}"] = est.fit(tr[X_cols], tr.y.to_numpy()).predict(va[X_cols])
        preds.loc[preds.split == "test", f"reg_{name}"] = est.fit(trva[X_cols], trva.y.to_numpy()).predict(te[X_cols])

    # ---- regression models
    for name, est, grid, n_iter in regression_searches():
        t0 = time.time()
        search = make_search(est, grid, n_iter, folds, "neg_mean_absolute_error")
        search.fit(tr[FEATURES], tr["y"])
        record_cv(name, "regression", search, "MAE")
        model_tr = search.best_estimator_  # already refit on all train years
        preds.loc[preds.split == "val", f"reg_{name}"] = model_tr.predict(va[FEATURES])
        model_final = clone_fit(model_tr, trva[FEATURES], trva["y"])
        preds.loc[preds.split == "test", f"reg_{name}"] = model_final.predict(te[FEATURES])
        joblib.dump(model_final, MODELS / f"reg_{name}.joblib")
        joblib.dump(model_tr, MODELS / f"reg_{name}_trainonly.joblib")
        print(f"[reg] {name:14s} CV MAE {cv_rows[-1]['cv_mean']:.2f} ± {cv_rows[-1]['cv_std']:.2f}"
              f"  ({time.time() - t0:.0f}s)")

    # ---- classification: Poor+ tomorrow
    thresholds = {"persistence": POOR_THRESHOLD}  # rule: today > 90
    preds["clf_persistence"] = preds["pm_t"]
    for name, est, grid, n_iter in classification_searches():
        t0 = time.time()
        search = make_search(est, grid, n_iter, folds, "average_precision")
        search.fit(tr[FEATURES], tr["y_poor"].astype(int))
        record_cv(name, "classification", search, "PR-AUC")
        model_tr = search.best_estimator_
        p_val = model_tr.predict_proba(va[FEATURES])[:, 1]
        thresholds[name] = threshold_for_recall(va["y_poor"].astype(int), p_val)
        preds.loc[preds.split == "val", f"clf_{name}"] = p_val
        model_final = clone_fit(model_tr, trva[FEATURES], trva["y_poor"].astype(int))
        preds.loc[preds.split == "test", f"clf_{name}"] = model_final.predict_proba(te[FEATURES])[:, 1]
        joblib.dump(model_final, MODELS / f"clf_{name}.joblib")
        print(f"[clf] {name:14s} CV PR-AUC {cv_rows[-1]['cv_mean']:.3f} ± {cv_rows[-1]['cv_std']:.3f}"
              f"  thr={thresholds[name]:.3f}  ({time.time() - t0:.0f}s)")

    # Best regressor turned into a classifier: threshold its forecast, tuned on val too.
    thresholds["lightgbm_reg_threshold"] = threshold_for_recall(
        va["y_poor"].astype(int), preds.loc[preds.split == "val", "reg_lightgbm"])
    preds["clf_lightgbm_reg_threshold"] = preds["reg_lightgbm"]

    # ---- ablation: which feature groups matter? (tuned LightGBM params, val + test)
    best_lgbm = joblib.load(MODELS / "reg_lightgbm_trainonly.joblib")
    weather_t = [c for c in NUMERIC_FEATURES if c.endswith("_t") and c != "pm_t"]
    groups = {
        "pollution + calendar (no weather)": [c for c in NUMERIC_FEATURES if c not in weather_t],
        "+ today's weather (main model)": NUMERIC_FEATURES,
        "+ tomorrow's actual weather (oracle forecast, upper bound)": NUMERIC_FEATURES + NEXT_WEATHER_FEATURES,
    }
    abl = []
    for label, num in groups.items():
        est = clone(best_lgbm)
        est.estimator.set_params(prep=preprocessor(scale=False, numeric=num))
        cols = num + CATEGORICAL_FEATURES
        row = {"features": label}
        for split, fit_df, ev in [("val", tr, va), ("test", trva, te)]:
            m = clone(est).fit(fit_df[cols], fit_df["y"])
            mae = float(np.mean(np.abs(m.predict(ev[cols]) - ev["y"])))
            pers = float(np.mean(np.abs(ev["pm_t"] - ev["y"])))
            row[f"{split}_MAE"] = round(mae, 2)
            row[f"{split}_gain_vs_persistence_%"] = round(100 * (1 - mae / pers), 1)
        abl.append(row)
        print(f"[ablation] {label}: {row}")
    pd.DataFrame(abl).to_csv(TABLES / "ablation_features.csv", index=False)

    preds.to_csv(TABLES / "predictions.csv", index=False)
    pd.DataFrame(cv_rows).round(4).to_csv(TABLES / "cv_results.csv", index=False)
    (TABLES / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    (TABLES / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(f"Saved predictions, CV results and models to {TABLES} and {MODELS}")


def clone_fit(model, X, y):
    return clone(model).fit(X, y)


if __name__ == "__main__":
    main()
