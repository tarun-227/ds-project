"""Level 1 — Overfitting.

Real data: sklearn's bundled diabetes dataset (442 patients). x = body-mass
index, y = disease progression after one year. The player sees only a small
training sample and picks a polynomial degree. Training error keeps falling
as the degree rises; the hidden test error, revealed on submit, does not.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from levels.base import Level, LevelState, Submission

N_TRAIN = 30
OFF_CHART = 100_000  # MSEs above this are shown as 'off the chart'


def fmt_mse(v: float) -> str:
    return f"{v:,.0f}" if v < OFF_CHART else "> 100,000 💥"
DEGREES = range(1, 16)
DEFAULT_DEGREE = 10  # "the previous analyst's model": great training error, poor test error


class OverfittingLevel(Level):
    key = "overfitting"
    title = "Level 1 · The Perfect Fit?"
    concept = "Overfitting"
    objective = ("Explain why a model that fits the training data almost perfectly "
                 "can predict new data worse than a simpler model.")
    goal = "Get the lowest error on the hidden test patients. You have 3 submissions."
    how_to_play = ("The previous analyst left you a degree-10 model with a tiny training error. "
                   "Drag the slider to change how wiggly the model is (polynomial degree). "
                   "You only see the 30 training patients and their error. "
                   "Press Submit to reveal how the model does on patients it never saw.")

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        data = load_diabetes(as_frame=True)
        x = data.frame["bmi"].to_numpy()
        y = data.frame["target"].to_numpy()
        idx = np.random.default_rng(seed).permutation(len(x))
        tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
        self.x_train, self.y_train = x[tr], y[tr]
        # Test patients are kept inside the training BMI range: we judge how well the
        # model interpolates, so the wild ends of a polynomial don't dominate the score.
        lo, hi = self.x_train.min(), self.x_train.max()
        te = te[(x[te] >= lo) & (x[te] <= hi)]
        self.x_test, self.y_test = x[te], y[te]
        # Reference points computed live (used for stars, never shown as "the answer").
        self._test_curve = {d: self.scores(d)[1] for d in DEGREES}
        self.best_test = min(self._test_curve.values())

    # ------------------------------------------------------------ computation
    def fit(self, degree: int):
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree),
                              LinearRegression())
        model.fit(self.x_train.reshape(-1, 1), self.y_train)
        return model

    def scores(self, degree: int) -> tuple[float, float]:
        m = self.fit(degree)
        tr = mean_squared_error(self.y_train, m.predict(self.x_train.reshape(-1, 1)))
        te = mean_squared_error(self.y_test, m.predict(self.x_test.reshape(-1, 1)))
        return float(tr), float(te)

    def stars(self, test_mse: float) -> int:
        ratio = test_mse / self.best_test
        return 3 if ratio <= 1.03 else 2 if ratio <= 1.10 else 1 if ratio <= 1.25 else 0

    def hint(self, degree: int, train_mse: float, test_mse: float) -> str:
        gap = test_mse / max(train_mse, 1e-9)
        if gap > 1.5 and degree >= 5:
            return ("Your training error is low but the test error jumped. The curve is "
                    "bending to pass through individual training points, including their "
                    "noise. That noise won't repeat for new patients. This is **overfitting**.")
        if degree <= 1 and self.stars(test_mse) < 3:
            return ("Both errors are fairly high and close together. A straight line may be "
                    "too simple to capture the pattern (**underfitting**). Try a bit more flexibility.")
        if self.stars(test_mse) >= 2:
            return ("Nice: train and test errors are close. The model learned the general "
                    "trend rather than memorising the training points.")
        return "Compare how far apart the train and test errors are. What happens to that gap as the degree changes?"

    def submit(self, state: LevelState, degree: int) -> Submission:
        tr, te = self.scores(degree)
        s = Submission({"degree": degree}, tr, te, self.stars(te), self.hint(degree, tr, te))
        state.submissions.append(s)
        state.attempts_left -= 1
        state.passed = state.passed or s.stars >= 1
        return s

    # ------------------------------------------------------------ UI
    def render(self, state: LevelState, log) -> None:  # pragma: no cover - UI
        import plotly.graph_objects as go
        import streamlit as st

        # Start from a tempting, overfit model so the default is NOT the answer:
        # the player has to discover that a lower training error isn't better.
        degree = st.slider("Model flexibility (polynomial degree)", 1, 15, DEFAULT_DEGREE,
                           key=f"{self.key}_degree")
        model = self.fit(degree)
        train_mse, _ = self.scores(degree)

        grid = np.linspace(self.x_train.min(), self.x_train.max(), 300)
        y_lo, y_hi = self.y_test.min() - 50, self.y_test.max() + 50
        fig = go.Figure()
        fig.add_scatter(x=self.x_train, y=self.y_train, mode="markers",
                        name="training patients", marker=dict(size=9))
        if state.submissions:
            fig.add_scatter(x=self.x_test, y=self.y_test, mode="markers",
                            name="hidden test patients", marker=dict(size=5, opacity=0.35))
        fig.add_scatter(x=grid, y=np.clip(model.predict(grid.reshape(-1, 1)), y_lo, y_hi),
                        mode="lines", name=f"model (degree {degree})", line=dict(width=3))
        fig.update_layout(xaxis_title="Body-mass index (standardised)",
                          yaxis_title="Disease progression", yaxis_range=[y_lo, y_hi],
                          height=420, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Training error (MSE)", fmt_mse(train_mse))

        if st.button(f"Submit degree {degree}", disabled=state.attempts_left == 0,
                     type="primary"):
            s = self.submit(state, degree)
            log(level=self.key, event="submit", degree=degree, train_mse=round(s.train_score, 1),
                test_mse=round(s.test_score, 1), stars=s.stars)
            st.rerun()

        if state.submissions:
            st.subheader("Your submissions")
            st.table([{"degree": s.choice["degree"], "train MSE": fmt_mse(s.train_score),
                       "test MSE": fmt_mse(s.test_score), "stars": "★" * s.stars or "–"}
                      for s in state.submissions])
            st.info(state.submissions[-1].hint)

        st.caption(f"Attempts left: {state.attempts_left}")
        if state.attempts_left == 0 or state.passed:
            done = state.attempts_left == 0
            if done:
                st.subheader("What was going on")
                ds = list(DEGREES)
                cfig = go.Figure()
                cfig.add_scatter(x=ds, y=[self.scores(d)[0] for d in ds], name="training error",
                                 line=dict(color="#2a6fdb", width=3))
                cfig.add_scatter(x=ds, y=list(self._test_curve.values()), name="test error",
                                 line=dict(color="#e8710a", width=3))
                cfig.update_layout(yaxis_type="log", xaxis_title="degree",
                                   yaxis_title="MSE (log scale)", height=320,
                                   margin=dict(t=20, b=10))
                st.plotly_chart(cfig, use_container_width=True)
                st.write("Test error for every degree. Training error only ever goes down "
                         "as the degree rises; test error doesn't.")
            text = st.text_area("In one sentence: why can a lower training error give a "
                                "worse test error?", value=state.reflection,
                                key=f"{self.key}_reflection")
            if st.button("Save reflection") and text.strip():
                state.reflection = text.strip()
                log(level=self.key, event="reflection", text=state.reflection)
                st.success("Saved. Next level unlocked!" if state.passed else "Saved.")
