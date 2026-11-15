"""Shared base class for all game levels.

A level separates *computation* (pure methods, unit-tested, no Streamlit)
from *rendering* (render(), which draws the UI). Scores are always computed
live from the player's choices: nothing is hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Submission:
    choice: dict
    train_score: float
    test_score: float
    stars: int
    hint: str


@dataclass
class LevelState:
    attempts_left: int
    submissions: list[Submission] = field(default_factory=list)
    passed: bool = False
    reflection: str = ""
    extra: dict = field(default_factory=dict)  # per-level UI state (e.g. which sample is shown)


class Level:
    key: str = "base"
    title: str = "Base level"
    concept: str = ""
    objective: str = ""          # learning objective shown to the player
    goal: str = ""               # what the player must achieve
    how_to_play: str = ""
    max_attempts: int = 3

    def __init__(self, seed: int = 42):
        self.seed = seed

    def new_state(self) -> LevelState:
        return LevelState(attempts_left=self.max_attempts)

    def render(self, state: LevelState, log) -> None:  # pragma: no cover - UI
        raise NotImplementedError
