import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from levels import LEVELS
from levels.level1_overfitting import DEGREES, OverfittingLevel


def test_all_levels_construct():
    for cls in LEVELS:
        lvl = cls(seed=42)
        assert lvl.objective and lvl.goal and lvl.how_to_play


def test_overfitting_is_real():
    """Train error falls with degree while test error eventually rises: the
    concept must come out of the computation, not out of a script."""
    lvl = OverfittingLevel(seed=42)
    train = [lvl.scores(d)[0] for d in DEGREES]
    test = [lvl.scores(d)[1] for d in DEGREES]
    assert train[-1] < train[0]
    assert test[-1] > min(test) * 1.25


def test_submit_uses_attempts_and_scores_live():
    lvl = OverfittingLevel(seed=42)
    st = lvl.new_state()
    s = lvl.submit(st, 15)
    assert st.attempts_left == 2
    assert s.test_score > s.train_score
    assert "overfitting" in s.hint.lower()


# ---------------------------------------------------------------- level 2
from levels.level2_bias_variance import DEPTHS as BV_DEPTHS, BiasVarianceLevel


def test_bias_falls_and_variance_rises_with_depth():
    """The decomposition comes out of real fits on real data, not a script."""
    lvl = BiasVarianceLevel(seed=42)
    first, last = lvl.decompose(min(BV_DEPTHS)), lvl.decompose(max(BV_DEPTHS))
    assert last["bias2"] < first["bias2"]
    assert last["variance"] > first["variance"]
    assert last["train_mse"] < first["train_mse"]


def test_sweet_spot_is_in_the_middle():
    lvl = BiasVarianceLevel(seed=42)
    best = min(BV_DEPTHS, key=lvl.avoidable)
    assert min(BV_DEPTHS) < best < max(BV_DEPTHS)
    assert lvl.stars_for_depth(best) == 3
    assert lvl.stars_for_depth(max(BV_DEPTHS)) == 0


def test_decomposition_adds_up():
    """bias² + variance + noise should be roughly to the measured test MSE."""
    lvl = BiasVarianceLevel(seed=42)
    for d in (1, 3, 8):
        r = lvl.decompose(d)
        assert abs(r["bias2"] + r["variance"] + r["noise"] - r["mse"]) / r["mse"] < 0.15  # the true curve is estimated, so not exact


def test_level2_hint_reacts_to_choice():
    lvl = BiasVarianceLevel(seed=42)
    st = lvl.new_state()
    deep = lvl.submit(st, 12)
    shallow = lvl.submit(st, 1)
    assert "variance" in deep.hint.lower() and "bias" in shallow.hint.lower()
    assert st.attempts_left == 1
