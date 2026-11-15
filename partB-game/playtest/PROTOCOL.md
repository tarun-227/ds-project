# Playtest protocol (aim for 5+ peers, ~20 min each)

1. **Pre-quiz (3 min).** Give `quiz_pre.md` on paper or as a Google Form. No help, no game yet.
2. **Play (10–15 min).** Run `streamlit run app/main.py`. The player types a nickname (use the same one on both quizzes). Don't explain the concepts; only answer questions about the controls. Note down anything they get stuck on.
3. **Post-quiz (3 min).** Give `quiz_post.md`. It tests the same concepts with different questions.
4. **Feedback (2 min).** `feedback_form.md`: fun, clarity and confusion, each 1–5, plus one free-text comment.
5. **Record.** Add one row per player to `quiz_results.csv` (score each quiz with `answer_key.md`). The game writes its own log to `logs/<session>.csv` automatically.
6. **Analyse.** From `partB-game/`, run `python playtest/analyze.py`. It writes `playtest/summary.md` (pre/post table, average gain, stars, attempts and reflections from the logs). Copy the numbers into `REPORT.md`.
7. **Iterate.** Pick the most common confusion, change the game, and record the before/after in `REPORT.md`. If you can, have 1–2 more people play the changed version.

Ethics: nicknames only, no personal data. Tell players the logs are used for a course report.
