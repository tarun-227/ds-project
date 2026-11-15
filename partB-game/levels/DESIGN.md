# Level design

All levels follow one loop: **choose → see the training result → submit → see the hidden result → hint → reflect**.
The hint depends on what the player actually did, and every number is computed live with a fixed seed.

| # | Title | Concept | Learning objective | Player decision | What gets revealed | Status |
|---|---|---|---|---|---|---|
| 1 | The Perfect Fit? | Overfitting | Explain why a near-perfect training fit can predict new data worse | Polynomial degree 1–15, starting at a tempting degree 10 on 30 real patients (sklearn diabetes: BMI → progression) | Test MSE on held-out patients; at the end, the train and test error curves | **Built** |
| 2 | Too Simple, Too Wild | Bias–variance trade-off | Find the complexity sweet spot and explain bias vs variance | Decision tree `max_depth` 1–12 on 100 random days of Delhi PM2.5 (x = day of year); *Draw another sample*; *Show all 20 analysts* | Average test MSE over 20 analysts, split into bias², variance and noise; at the end, all curves vs depth | **Built** |
| 3 | The Data Budget | Sampling, sample size, sampling bias | Explain how a small or biased sample misleads | Spend a budget of N rows, choosing which cities or seasons to sample (Part A air-quality data) | The player's estimate of mean PM2.5 next to the true population mean, with a CI that narrows as N grows | Not built (out of scope for this submission) |
| 4 | Ban Ice Cream? | Correlation vs causation | Tell a confounded correlation from a causal effect | Choose a policy from observational data, then optionally "run an experiment" | A randomised experiment shows the real effect, and the temperature confounder is revealed | Not built (out of scope for this submission) |
| 5 | (stretch) Find the Groups | Clustering | See how k and feature scaling change clusters | Place k centroids, then run k-means; toggle scaling | Inertia and silhouette for each choice | Not built (out of scope for this submission) |

## Scoring
- Stars compare the player's test score with the best score achievable on that level, computed at runtime.
  - Level 1: test MSE within 3% of the best = ★★★, 10% = ★★, 25% = ★.
  - Level 2: *avoidable* error (test MSE minus noise) within 5% of the best = ★★★, 15% = ★★, 30% = ★. Noise is the same for every depth, so including it would make every choice look nearly equal (best 4,100 vs worst 5,800 MSE, of which ~2,500 is noise).
- 1 star passes the level and unlocks the next one.
- 3 submissions per level, so the player has to think before each try instead of brute-forcing the slider.

## Evidence of learning
- Every submission and reflection is logged to `playtest/logs/<session>.csv`.
- A pre-quiz and post-quiz (5 questions each, one per concept) go in `playtest/`.

## Design decisions from testing
- **The default must not be the answer.** In the first build, Level 1's slider started at degree 1, which already scores ★★★, so a player could pass without discovering anything. It now starts at degree 10 ("the previous analyst's model"): tiny training error, 0 stars. Level 2 starts at depth 8 for the same reason.
- **Level 2 sample size.** With 60 training days the U-curve was too flat to feel (every depth earned a star). With 100 days and avoidable-error stars, depth 2 is ★★★, depths 3–4 are ★★, 5 is ★, and depth 1 or ≥6 fail, so both "too simple" and "too wild" lose.
- **Why 20 analysts?** Variance is a property of the *training procedure*, not of one fitted model. Showing 20 models trained on 20 samples makes it visible; one model can't show it.
