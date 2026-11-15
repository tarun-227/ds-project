# Playtest report — Overfit or Not?

> **Status: TO FILL IN after running the playtest with peers.** Follow [PROTOCOL.md](PROTOCOL.md), then run
> `python playtest/analyze.py` from `partB-game/` and copy the tables from `playtest/summary.md` into the sections below.
> Every number here must come from real sessions (logs in `playtest/logs/`, quiz scores in `quiz_results.csv`).

## 1. Setup
- **Players:** _N_ peers (_course/year_), none had seen the game before.
- **Date(s):** _…_
- **Procedure:** pre-quiz (5 Qs) → 10–15 min play, no explanations given → post-quiz (5 Qs, same concepts, different questions) → feedback form.
- **Materials:** [quiz_pre.md](quiz_pre.md), [quiz_post.md](quiz_post.md), [answer_key.md](answer_key.md), [feedback_form.md](feedback_form.md).

## 2. Did they learn? Pre/post quiz
_Paste the pre/post table from summary.md._

| player | pre (/5) | post (/5) | gain |
|---|---|---|---|
| | | | |

**Mean gain: _…_** (_x_ of _N_ improved).
Per-question: which concept improved most? Which question was still often wrong?

Caveat: a small sample with no control group, and the post-quiz is not identical to the pre-quiz, so this is evidence, not proof.

## 3. Did they discover it by playing? Gameplay logs
_Paste the gameplay table from summary.md (choices in order, stars)._

What to look for and comment on:
- Level 1: how many submitted the default degree 10 first, saw 0 stars, then moved down?
- Level 2: how many pressed *Draw another sample* / *Show all analysts* before submitting? Did their second choice move towards depth 2–4?

## 4. Reflections in their own words
_Paste 3–5 reflections from summary.md and say which ones show the concept correctly and which show a misconception._

## 5. Feedback
| | fun | clarity | hints helpful | recommend |
|---|---|---|---|---|
| mean (1–5) | | | | |

Most common confusion: _…_

## 6. Change made because of feedback (before → after)
| Before | Feedback / evidence | After |
|---|---|---|
| Level 1 slider started at degree 1, which already earns ★★★ | Found in our own test play: a player could pass by clicking Submit straight away and never see overfitting | Slider starts at degree 10 ("the previous analyst's model"), which earns 0 stars. See `DEFAULT_DEGREE` in `levels/level1_overfitting.py` |
| _…from peer feedback…_ | _…_ | _…_ |

## 7. Screenshots
See `../docs/screenshots/`.
