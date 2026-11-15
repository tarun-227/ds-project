"""Level 2 — Bias–variance trade-off.

Real data: daily PM2.5 in Delhi 2015–2019 (CPCB, the same data as Part A).
x = day of the year, y = PM2.5. Air is dirty in winter and clean in the
monsoon, so there is a real seasonal pattern under a lot of day-to-day noise.

Each "analyst" gets a different random sample of N_TRAIN days and fits a
decision tree of the depth the player chose. The player's score is the
average test error over N_WORLDS analysts, which is exactly what the
bias–variance decomposition talks about:

    expected test MSE  ≈  bias²  +  variance  +  noise

- bias²    : how far the AVERAGE model is from the true seasonal curve
- variance : how much the models disagree with each other
- noise    : day-to-day randomness no model can predict

The "true" curve is estimated by a circular kernel smoother on all days.
Everything is computed live from the data with a fixed seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from levels.base import Level, LevelState, Submission

DATA = Path(__file__).resolve().parents[1] / "data" / "delhi_pm25.csv"
N_TRAIN = 100      # days each analyst gets to train on
N_TEST = 400       # held-out days used to score every model
N_WORLDS = 20      # analysts / alternative training samples
DEPTHS = range(1, 13)
SMOOTH_BW = 12.0   # days; bandwidth of the smoother that estimates the true curve


def circular_smoother(x_ref, y_ref, x_out, bw=SMOOTH_BW):
    """Gaussian kernel average on a 365-day circle (Dec 31 is next to Jan 1)."""
    d = np.abs(x_out[:, None] - x_ref[None, :])
    d = np.minimum(d, 365 - d)
    w = np.exp(-0.5 * (d / bw) ** 2)
    return (w @ y_ref) / w.sum(axis=1)


class BiasVarianceLevel(Level):
    key = "bias_variance"
    title = "Level 2 · Too Simple, Too Wild"
    concept = "Bias–variance trade-off"
    objective = ("Find the model complexity that balances bias (too simple to follow the "
                 "pattern) and variance (so flexible it changes with every sample), and "
                 "explain why the best test error sits between the two extremes.")
    goal = (f"Choose a tree depth. {N_WORLDS} analysts will each train it on their own "
            f"random {N_TRAIN} days; get the lowest AVERAGE error on hidden days. "
            "You have 3 submissions.")
    how_to_play = ("The dots are days of Delhi air pollution (PM2.5) in your training sample. "
                   "The slider sets how deep the decision tree may grow: deeper means more "
                   "steps in the curve. Press 'Draw another sample' to see what a different "
                   "analyst's data would give, or tick 'Show all analysts' to see all "
                   f"{N_WORLDS} models at once. Submit to reveal the hidden test error and "
                   "where it comes from.")

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        df = pd.read_csv(DATA)
        x = df["day_of_year"].to_numpy(dtype=float)
        y = df["pm25"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x))
        te, pool = idx[:N_TEST], idx[N_TEST:]
        self.x_test, self.y_test = x[te], y[te]
        self.x_pool, self.y_pool = x[pool], y[pool]
        self.samples = [rng.choice(len(pool), N_TRAIN, replace=False) for _ in range(N_WORLDS)]
        self.grid = np.arange(1, 366, dtype=float)
        self.truth_grid = circular_smoother(x, y, self.grid)
        self.truth_test = circular_smoother(x, y, self.x_test)
        self._cache: dict[int, dict] = {}
        # Stars compare AVOIDABLE error (test MSE minus noise): no depth can remove noise.
        self.best_avoidable = min(self.avoidable(d) for d in DEPTHS)

    # ------------------------------------------------------------ computation
    def fit(self, depth: int, world: int) -> DecisionTreeRegressor:
        s = self.samples[world]
        m = DecisionTreeRegressor(max_depth=depth, random_state=self.seed)
        m.fit(self.x_pool[s].reshape(-1, 1), self.y_pool[s])
        return m

    def train_mse(self, depth: int, world: int) -> float:
        s = self.samples[world]
        p = self.fit(depth, world).predict(self.x_pool[s].reshape(-1, 1))
        return float(np.mean((p - self.y_pool[s]) ** 2))

    def decompose(self, depth: int) -> dict:
        """Average test MSE over all worlds, split into bias², variance and noise."""
        if depth in self._cache:
            return self._cache[depth]
        preds = np.stack([self.fit(depth, w).predict(self.x_test.reshape(-1, 1))
                          for w in range(N_WORLDS)])            # (worlds, test days)
        mean_pred = preds.mean(axis=0)
        out = {
            "mse": float(np.mean((preds - self.y_test) ** 2)),
            "bias2": float(np.mean((mean_pred - self.truth_test) ** 2)),
            "variance": float(np.mean(preds.var(axis=0))),
            "noise": float(np.mean((self.y_test - self.truth_test) ** 2)),
            "train_mse": float(np.mean([self.train_mse(depth, w) for w in range(N_WORLDS)])),
        }
        self._cache[depth] = out
        return out

    def avoidable(self, depth: int) -> float:
        d = self.decompose(depth)
        return d["mse"] - d["noise"]

    def stars_for_depth(self, depth: int) -> int:
        r = self.avoidable(depth) / self.best_avoidable
        return 3 if r <= 1.05 else 2 if r <= 1.15 else 1 if r <= 1.30 else 0

    def hint(self, depth: int, d: dict) -> str:
        if self.stars_for_depth(depth) >= 2 and 2 <= depth <= 4:
            return ("Good balance: the models follow the seasonal pattern (low bias) without "
                    "jumping around from sample to sample (low variance).")
        if d["variance"] > d["bias2"] and depth >= 3:
            return (f"Most of your avoidable error is **variance** ({d['variance']:,.0f}) rather "
                    f"than bias ({d['bias2']:,.0f}). Each analyst's deep tree chased the "
                    "particular days in their sample, so the 20 models disagree wildly. "
                    "A simpler tree would be steadier.")
        if d["bias2"] > d["variance"] and depth <= 2:
            return (f"Most of your avoidable error is **bias** ({d['bias2']:,.0f}). The 20 "
                    "models agree with each other, but they are all wrong in the same way: "
                    "a tree this shallow can't follow the winter peak and monsoon dip.")
        if self.stars_for_depth(depth) >= 2:
            return ("Good balance: the models follow the seasonal pattern (low bias) without "
                    "jumping around from sample to sample (low variance).")
        return ("Look at the breakdown. Which part is bigger, bias or variance? Move the "
                "depth in the direction that shrinks the bigger one.")

    def submit(self, state: LevelState, depth: int) -> Submission:
        d = self.decompose(depth)
        s = Submission({"depth": depth}, d["train_mse"], d["mse"], self.stars_for_depth(depth),
                       self.hint(depth, d))
        state.submissions.append(s)
        state.attempts_left -= 1
        state.passed = state.passed or s.stars >= 1
        return s

    # ------------------------------------------------------------ UI
    def render(self, state: LevelState, log) -> None:  # pragma: no cover - UI
        import plotly.graph_objects as go
        import streamlit as st

        world = state.extra.setdefault("world", 0)
        depth = st.slider("Tree depth (model flexibility)", 1, max(DEPTHS), 8,
                          key=f"{self.key}_depth")
        c1, c2 = st.columns(2)
        if c1.button("🎲 Draw another sample"):
            state.extra["world"] = world = (world + 1) % N_WORLDS
            state.extra["draws"] = state.extra.get("draws", 0) + 1
            log(level=self.key, event="resample", depth=depth, world=world)
        show_all = c2.checkbox(f"Show all {N_WORLDS} analysts", key=f"{self.key}_all")

        s = self.samples[world]
        fig = go.Figure()
        if show_all:
            for w in range(N_WORLDS):
                fig.add_scatter(x=self.grid, y=self.fit(depth, w).predict(self.grid.reshape(-1, 1)),
                                mode="lines", line=dict(width=1, color="rgba(232,113,10,0.35)"),
                                showlegend=(w == 0), name="other analysts' models")
        fig.add_scatter(x=self.x_pool[s], y=self.y_pool[s], mode="markers",
                        name=f"your {N_TRAIN} training days", marker=dict(size=8, color="#2a6fdb"))
        fig.add_scatter(x=self.grid, y=self.fit(depth, world).predict(self.grid.reshape(-1, 1)),
                        mode="lines", name=f"your tree (depth {depth})",
                        line=dict(width=3, color="#1b3d7a"))
        if state.submissions:
            fig.add_scatter(x=self.grid, y=self.truth_grid, mode="lines",
                            name="true seasonal pattern", line=dict(dash="dash", color="#2ca02c"))
        fig.update_layout(xaxis_title="Day of year (1 = 1 Jan)", yaxis_title="PM2.5 (µg/m³)",
                          height=420, margin=dict(t=20, b=10), yaxis_range=[0, 500],
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Training error on your sample (MSE)", f"{self.train_mse(depth, world):,.0f}")

        if st.button(f"Submit depth {depth}", disabled=state.attempts_left == 0, type="primary"):
            sub = self.submit(state, depth)
            d = self.decompose(depth)
            log(level=self.key, event="submit", depth=depth, train_mse=round(sub.train_score),
                test_mse=round(sub.test_score), bias2=round(d["bias2"]),
                variance=round(d["variance"]), stars=sub.stars,
                resamples_before=state.extra.get("draws", 0), looked_at_all=bool(show_all))
            st.rerun()

        if state.submissions:
            st.subheader("Your submissions")
            rows = []
            for sub in state.submissions:
                d = self.decompose(sub.choice["depth"])
                rows.append({"depth": sub.choice["depth"], "train MSE": f"{sub.train_score:,.0f}",
                             "avg test MSE": f"{sub.test_score:,.0f}",
                             "bias²": f"{d['bias2']:,.0f}", "variance": f"{d['variance']:,.0f}",
                             "stars": "★" * sub.stars or "–"})
            st.table(rows)
            last = self.decompose(state.submissions[-1].choice["depth"])
            bars = go.Figure(go.Bar(
                x=[last["bias2"], last["variance"], last["noise"]],
                y=["bias² (too simple)", "variance (too wild)", "noise (unavoidable)"],
                orientation="h", marker_color=["#d62728", "#e8710a", "#999999"]))
            bars.update_layout(height=200, margin=dict(t=10, b=10), xaxis_title="error (MSE)")
            st.plotly_chart(bars, use_container_width=True)
            st.info(state.submissions[-1].hint)

        st.caption(f"Attempts left: {state.attempts_left}")
        if state.attempts_left == 0 or state.passed:
            if state.attempts_left == 0:
                st.subheader("What was going on")
                ds = list(DEPTHS)
                dd = [self.decompose(x) for x in ds]
                cfig = go.Figure()
                for k, name, col in [("train_mse", "training error", "#2a6fdb"),
                                     ("mse", "test error", "#000000"),
                                     ("bias2", "bias²", "#d62728"),
                                     ("variance", "variance", "#e8710a")]:
                    cfig.add_scatter(x=ds, y=[v[k] for v in dd], name=name,
                                     line=dict(color=col, width=3))
                cfig.update_layout(xaxis_title="tree depth", yaxis_title="MSE", height=340,
                                   margin=dict(t=20, b=10))
                st.plotly_chart(cfig, use_container_width=True)
                st.write("As depth grows, bias falls and variance rises. Test error is "
                         "lowest where their sum is lowest, not where training error is lowest.")
            text = st.text_area("In one sentence: why is the best depth neither the "
                                "shallowest nor the deepest?", value=state.reflection,
                                key=f"{self.key}_reflection")
            if st.button("Save reflection", key=f"{self.key}_save") and text.strip():
                state.reflection = text.strip()
                log(level=self.key, event="reflection", text=state.reflection)
                st.success("Saved. You've finished the game!" if state.passed else "Saved.")
