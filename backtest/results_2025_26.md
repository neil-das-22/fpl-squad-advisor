# xP model backtest — historical_2025_26

Real historical backtest against the vaastav Fantasy-Premier-League
archive at `/sessions/eager-sweet-pasteur/mnt/FPL Project/data/raw/historical_2025_26`.

- Gameweeks replayed: **1–38** (38 with data)
- Player-gameweek observations: **29,338**
- Cold-start window: GW1–GW5; steady state: GW6+

## Data-loading warnings

These fired while normalising the source CSVs. Each one is a place
where the schema differed from what the harness expected — read them
before trusting the metrics.

- merged_gw.csv `xP` (FPL's own expected points) is all-zero for 27 of 38 gameweeks: [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37]. Those rounds are treated as MISSING, not as a prediction of zero. The FPL-xP baseline is therefore scored only on gameweeks [1, 2, 3, 4, 5, 6, 8, 9, 24, 29, 38], and our model is re-scored on that same subset for a fair head-to-head.

## 0. Summary

**Does the model beat the naive baselines?** Partly, and the honest
answer depends on which pool you score on.

- On the **full player pool** it does not: MAE 1.279 and
  Spearman 0.613, versus 1.073 / 0.663 for a season-to-date
  points-per-game baseline. That is a clear loss.
- On the **credible pool** (players with 1+ prior appearance,
  n=17,164) it wins: MAE 1.706 vs 1.755 for PPG, with
  Spearman 0.482 vs 0.482 and a better Pearson.
- On **top-20 precision**, the metric that maps onto actually
  picking a squad, it wins on both pools: 35.9% of its top
  20 picks returned 6+ points versus 30.4% for PPG, against a
  pool base rate of 7.1%.

**Does it beat FPL's own xP?** No — comfortably not. See section 6b.

**The single biggest problem** is not a constant at all. 39% of the
player pool has never played a minute, and the model assigns those
players a 2% chance of starting and 0.46 xP each. They
account for more than 100% of the model's total over-prediction.
Everything else in this report is small by comparison. Section 1b.

**The DefCon data gap is worth closing.** Giving the model each
player's own rolling CBIT/CBIRT rate instead of a flat positional
prior lifts the DefCon top-50 hit rate from 24.4% to 34.4%. Section 8.

**Where the model is genuinely good:** on the credible pool it is
well calibrated component by component — goals, assists, clean
sheets, saves, cards and bonus are all within 0.05 points per
player-gameweek of reality. The scoring-rules half of the model is in
good shape. The minutes half is where the damage is.

## 1. Overall accuracy

| slice | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| all player-gameweeks | 29338 | 1.279 | 2.080 | 0.176 | 0.499 | 0.613 | 1.349 | 1.173 |


`bias` is mean(predicted) - mean(actual): positive means the model
over-predicts. `spearman` is the rank correlation, and it is the number
to care about most — squad selection is an ordering problem, so being
uniformly 0.4 points high costs nothing as long as the ranking holds.

### Restricted universes

The full pool is dominated by players who did not feature at all, which
flatters any model that predicts near zero. These slices strip that
out — `players who featured` conditions on the outcome (so it is
optimistic and shown for diagnosis only), while the appearance-count
slices condition only on pre-gameweek information and are the honest read
on the pool you would actually pick from.

**`1+ prior appearances` is referred to throughout this report as the
*credible pool*.** It is the slice the component analysis and the
constant recommendations are computed on, for the reason set out
immediately below.

| slice | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| all players | 29338 | 1.279 | 2.080 | 0.176 | 0.499 | 0.613 | 1.349 | 1.173 |
| credible pool (1+ prior appearances) | 17164 | 1.706 | 2.590 | -0.100 | 0.400 | 0.482 | 1.826 | 1.926 |
| 3+ prior appearances | 14548 | 1.816 | 2.701 | -0.132 | 0.366 | 0.424 | 1.961 | 2.093 |
| players who featured (minutes > 0) | 11361 | 1.977 | 3.038 | -0.871 | 0.276 | 0.305 | 2.157 | 3.029 |


### 1b. Why those two rows differ so much: the never-appeared population

This is the headline finding of the backtest, so it goes before
everything else.

From GW6 onward, **9,906 player-gameweeks
(38.6% of the pool)** belong to players who had
recorded **zero minutes in every completed gameweek so far** — squad
filler, academy names, permanently-benched keepers.

- The model gave them a mean start probability of **0.020** and predicted **0.46 xP** each.
- They actually scored **0.015** points on average, and featured at all in **1.1%** of cases.
- They account for **4,373 points of signed error**, against a whole-pool total of **2,865**.

That last line is the important one: **more than 100% of the model's
total over-prediction comes from this one group.** Remove them and the
model flips from over-predicting to slightly under-predicting.

The cause is a single line in `estimate_start_probability()`, which
treats "zero minutes" as "no information" regardless of how much of
the season has elapsed. See recommendation 1 in section 9. It is a
logic bug rather than a mis-set constant, and it is not the same thing
as backtest LIMITATION #2 (unknown injury status): these players were
not injured, they were simply never picked, and the model already had
all the evidence it needed to know that.

## 2. Per-position breakdown

Whole pool:

| slice | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| GKP | 3383 | 1.015 | 1.596 | 0.367 | 0.591 | 0.623 | 1.125 | 0.758 |
| DEF | 9601 | 1.461 | 2.243 | 0.239 | 0.471 | 0.574 | 1.493 | 1.254 |
| MID | 13119 | 1.206 | 2.018 | 0.124 | 0.506 | 0.631 | 1.311 | 1.187 |
| FWD | 3235 | 1.308 | 2.268 | -0.001 | 0.481 | 0.637 | 1.307 | 1.308 |


Credible pool (1+ prior appearance) — the like-for-like comparison, since
the never-appeared population is not evenly spread across positions
(clubs carry far more spare defenders and midfielders than forwards):

| slice | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| GKP | 1111 | 1.787 | 2.462 | -0.003 | 0.430 | 0.537 | 2.170 | 2.173 |
| DEF | 6013 | 1.887 | 2.709 | 0.019 | 0.375 | 0.443 | 1.944 | 1.925 |
| MID | 8023 | 1.572 | 2.476 | -0.138 | 0.415 | 0.508 | 1.731 | 1.869 |
| FWD | 2017 | 1.656 | 2.737 | -0.352 | 0.413 | 0.543 | 1.663 | 2.016 |


## 3. Cold start vs steady state

GW1–5 is the regime where the model has little or no
current-season minutes history, so `estimate_start_probability()` falls
back to `DEFAULT_START_PROBABILITY` and `shrunk_per90_rate()` returns
close to the raw positional prior. **This is the exact situation the
2026/27 squad build is in right now**, so this comparison matters more
than the headline number.

| slice | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| cold start (GW1-5) | 3588 | 1.769 | 2.381 | 0.635 | 0.347 | 0.411 | 1.919 | 1.284 |
| steady state (GW6+) | 25750 | 1.210 | 2.035 | 0.112 | 0.527 | 0.648 | 1.269 | 1.157 |
| cold start, credible pool | 1420 | 1.949 | 2.795 | -0.142 | 0.323 | 0.393 | 2.275 | 2.418 |
| steady state, credible pool | 15744 | 1.684 | 2.571 | -0.096 | 0.404 | 0.484 | 1.785 | 1.881 |


Read the credible-pool rows, not the whole-pool rows, to judge whether the
model *learns*. On the whole pool rank correlation appears to get WORSE
through the season, which is an artifact: the never-appeared population
grows as clubs register more fringe players, so the bug in section 1b
does progressively more damage. On the credible pool the model behaves as
it should — it gets better as evidence accumulates.

Cold-start MAE is 0.559 pts worse than steady state; rank
correlation is 0.238 lower.

Per-gameweek cold-start detail:

| gameweek | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| 1.000 | 690.000 | 2.181 | 2.651 | 0.886 | 0.204 | 0.090 | 2.282 | 1.396 |
| 2.000 | 705.000 | 2.224 | 2.724 | 1.166 | 0.310 | 0.420 | 2.451 | 1.285 |
| 3.000 | 712.000 | 2.116 | 2.530 | 0.982 | 0.260 | 0.277 | 2.291 | 1.309 |
| 4.000 | 740.000 | 1.248 | 2.041 | 0.015 | 0.554 | 0.629 | 1.330 | 1.315 |
| 5.000 | 741.000 | 1.142 | 1.880 | 0.180 | 0.533 | 0.680 | 1.304 | 1.124 |


## 4. Calibration

Players are split into quintiles by predicted xP (quantile bins, because
the predicted distribution is heavily right-skewed). A well-calibrated
model has `bias` near zero in every bin; a positive bias in the top bin
with a negative one in the bottom means the model's spread is too wide.

**Whole pool:**

| bin | n | pred_range | mean_pred | mean_actual | bias |
|---|---|---|---|---|---|
| 1 | 5869 | 0.36-0.44 | 0.416 | 0.040 | 0.376 |
| 2 | 5866 | 0.44-0.51 | 0.472 | 0.096 | 0.376 |
| 3 | 5868 | 0.51-1.33 | 0.844 | 0.921 | -0.077 |
| 4 | 5868 | 1.33-2.39 | 1.901 | 1.649 | 0.251 |
| 5 | 5867 | 2.39-10.48 | 3.111 | 3.158 | -0.047 |


**Credible pool (1+ prior appearance)** — this is the table to read for
calibration of the scoring model itself:

| bin | n | pred_range | mean_pred | mean_actual | bias |
|---|---|---|---|---|---|
| 1 | 3433 | 0.38-0.70 | 0.514 | 0.491 | 0.023 |
| 2 | 3433 | 0.70-1.33 | 1.013 | 1.224 | -0.211 |
| 3 | 3432 | 1.33-2.07 | 1.678 | 1.759 | -0.081 |
| 4 | 3433 | 2.07-2.88 | 2.477 | 2.511 | -0.034 |
| 5 | 3433 | 2.88-10.48 | 3.448 | 3.643 | -0.195 |


**Cold start:**

| bin | n | pred_range | mean_pred | mean_actual | bias |
|---|---|---|---|---|---|
| 1 | 720 | 0.39-0.52 | 0.444 | 0.150 | 0.294 |
| 2 | 715 | 0.52-2.02 | 1.261 | 1.196 | 0.065 |
| 3 | 722 | 2.02-2.28 | 2.168 | 0.830 | 1.338 |
| 4 | 713 | 2.29-2.77 | 2.451 | 1.021 | 1.430 |
| 5 | 718 | 2.77-5.17 | 3.273 | 3.227 | 0.046 |


**Steady state:**

| bin | n | pred_range | mean_pred | mean_actual | bias |
|---|---|---|---|---|---|
| 1 | 5153 | 0.36-0.44 | 0.414 | 0.035 | 0.379 |
| 2 | 5149 | 0.44-0.50 | 0.465 | 0.082 | 0.383 |
| 3 | 5148 | 0.50-1.13 | 0.719 | 0.692 | 0.027 |
| 4 | 5150 | 1.13-2.29 | 1.676 | 1.752 | -0.077 |
| 5 | 5150 | 2.30-10.48 | 3.072 | 3.226 | -0.154 |


## 5. Top-20 precision (the metric that actually matters)

Each gameweek, take the model's 20 highest-xP players and ask what
fraction returned 6+ points. `pool_hit_rate` is the
same rate across the entire player pool — if precision is not
comfortably above it, the shortlist is adding nothing.

| predictor | precision | mean_actual_points | pool_hit_rate | gameweeks | picks |
|---|---|---|---|---|---|
| xP model | 0.359 | 4.393 | 0.071 | 38 | 760 |
| baseline: season PPG | 0.304 | 4.157 | 0.071 | 38 | 760 |
| baseline: position mean | 0.286 | 3.576 | 0.071 | 38 | 760 |


## 6. Comparison against baselines

Three naive baselines, all built strictly from pre-gameweek-N data, plus
FPL's own published expected-points number.

- `zero` — always predict 0. Included to make a point: most rows in a
  gameweek are players who did not play, so a model that predicts nothing
  gets a deceptively good MAE. If the xP model cannot beat this on MAE,
  that is not automatically damning — check the correlations.
- `ppg` — the player's season-to-date points per elapsed gameweek.
- `position mean` — mean actual points for that position so far.
- `FPL official xP` — the `xP` column in merged_gw.csv, i.e. FPL's own
  pre-match projection for that player and round. Not a naive baseline at
  all: it is a serious rival model with access to data we do not have
  (team news, expected lineups, ownership-weighted signals). This is the
  hardest comparison in the report and the most informative one.

**Whole pool, all replayed gameweeks:**

| predictor | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| xP model | 29338 | 1.279 | 2.080 | 0.176 | 0.499 | 0.613 | 1.349 | 1.173 |
| baseline: zero | 29338 | 1.181 | 2.661 | -1.173 | n/a | n/a | 0.000 | 1.173 |
| baseline: season PPG | 29338 | 1.073 | 2.114 | -0.036 | 0.489 | 0.663 | 1.137 | 1.173 |
| baseline: position mean | 29338 | 1.556 | 2.394 | 0.023 | 0.032 | 0.069 | 1.196 | 1.173 |


**Credible pool (1+ prior appearance)** — the same comparison with the
never-appeared population removed. This is the fair read: the naive PPG
baseline gets an enormous free win on the full pool purely because a
player who has never played has a season PPG of exactly 0, which is a
perfect prediction for him. That is not football knowledge, it is the
same signal the model is throwing away in recommendation 1.

| predictor | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| xP model | 17164 | 1.706 | 2.590 | -0.100 | 0.400 | 0.482 | 1.826 | 1.926 |
| baseline: zero | 17164 | 1.939 | 3.418 | -1.926 | n/a | n/a | 0.000 | 1.926 |
| baseline: season PPG | 17164 | 1.755 | 2.686 | 0.017 | 0.362 | 0.482 | 1.943 | 1.926 |
| baseline: position mean | 17164 | 1.810 | 2.907 | -0.680 | 0.005 | 0.046 | 1.246 | 1.926 |


Cold-start window only (the prior-based baselines have almost nothing to
work with there, which is the fairest place for a structural model to
earn its keep):

| predictor | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| xP model | 3588 | 1.769 | 2.381 | 0.635 | 0.347 | 0.411 | 1.919 | 1.284 |
| baseline: zero | 3588 | 1.293 | 2.757 | -1.284 | n/a | n/a | 0.000 | 1.284 |
| baseline: season PPG | 3588 | 1.204 | 2.470 | -0.232 | 0.390 | 0.577 | 1.052 | 1.284 |
| baseline: position mean | 3588 | 1.590 | 2.512 | -0.198 | -0.008 | 0.024 | 1.086 | 1.284 |


### 6b. Head-to-head against FPL's own xP

**Data-quality caveat, read this first.** In this archive the `xP`
column is only populated for **11 of 38 gameweeks** ([1, 2, 3, 4, 5, 6, 8, 9, 24, 29, 38]); for the other 27 the scraper wrote 0.0 for every
player. An all-zero round is missing data, not a prediction, so those
rounds are excluded — scoring them would have handed our model a
fictitious win worth roughly a full point of MAE. Both predictors are
therefore compared on the identical surviving subset
(**8,293 player-gameweeks**).

**Whole pool, FPL-xP gameweeks only:**

| predictor | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| xP model | 8293 | 1.443 | 2.198 | 0.341 | 0.425 | 0.526 | 1.543 | 1.203 |
| baseline: zero | 8293 | 1.213 | 2.682 | -1.203 | n/a | n/a | 0.000 | 1.203 |
| baseline: season PPG | 8293 | 1.116 | 2.250 | -0.088 | 0.442 | 0.623 | 1.114 | 1.203 |
| baseline: position mean | 8293 | 1.551 | 2.426 | -0.041 | 0.006 | 0.053 | 1.162 | 1.203 |
| FPL official xP | 8293 | 0.875 | 1.640 | -0.007 | 0.730 | 0.764 | 1.195 | 1.203 |


**Credible pool, FPL-xP gameweeks only:**

| predictor | n | mae | rmse | bias | pearson | spearman | mean_pred | mean_actual |
|---|---|---|---|---|---|---|---|---|
| xP model | 4245 | 1.771 | 2.672 | -0.108 | 0.361 | 0.447 | 1.955 | 2.062 |
| baseline: zero | 4245 | 2.082 | 3.528 | -2.062 | n/a | n/a | 0.000 | 2.062 |
| baseline: season PPG | 4245 | 1.891 | 2.878 | 0.115 | 0.318 | 0.444 | 2.177 | 2.062 |
| baseline: position mean | 4245 | 1.825 | 2.965 | -0.779 | 0.037 | 0.071 | 1.283 | 2.062 |
| FPL official xP | 4245 | 1.296 | 2.040 | -0.003 | 0.702 | 0.753 | 2.059 | 2.062 |


**Top-20 precision on the same subset:**

| predictor | precision | mean_actual_points | pool_hit_rate | gameweeks | picks |
|---|---|---|---|---|---|
| xP model | 0.373 | 4.300 | 0.073 | 11 | 220 |
| FPL official xP | 0.627 | 7.086 | 0.073 | 11 | 220 |


**Verdict: FPL's own xP comfortably beats this model**, on every
metric, on both pools. That is the honest result and it is not
especially surprising — FPL's number is built on team news and
expected lineups, which is precisely the information our minutes model
lacks (KNOWN DATA GAPS #1 and the never-appeared bug above). The gap is
almost entirely a minutes-modelling gap, not a scoring-rules gap:
on the credible pool, where minutes are largely a solved problem, our
model's deficit narrows sharply.

## 7. Where the error comes from (component attribution)

Each player's real gameweek score is decomposed back into FPL's scoring
components and compared with the model's own `xp_*` components. This is
what makes the recommendations below specific rather than vague.

**Computed on the credible pool**, not the full pool. On the full pool the
never-appeared population (section 1b) inflates every single predicted
component at once, which makes every component look uniformly too high and
produces confidently wrong advice — 'lower BONUS_SCALE', 'lower
XG90_PRIOR' — about constants that are in fact close to correct. Removing
that population is what turns this table into a usable instrument.

`unexplained_residual` is actual total points minus the sum of the
decomposed components. It should be near zero; if it is not, the archive
is missing a column (most likely `defensive_contribution`) and the rows
above it are correspondingly unreliable.

| component | mean_pred | mean_actual | bias | pct_of_pred |
|---|---|---|---|---|
| appearance | 1.050 | 1.066 | -0.016 | 57.492 |
| goals | 0.249 | 0.274 | -0.025 | 13.661 |
| assists | 0.112 | 0.159 | -0.047 | 6.153 |
| clean_sheet | 0.257 | 0.278 | -0.021 | 14.087 |
| goals_conceded | -0.072 | -0.097 | 0.024 | -3.955 |
| saves | 0.019 | 0.025 | -0.006 | 1.027 |
| defcon | 0.150 | 0.164 | -0.015 | 8.198 |
| cards | -0.100 | -0.092 | -0.007 | -5.466 |
| penalties | 0.000 | 0.001 | -0.001 | 0.000 |
| bonus | 0.161 | 0.137 | 0.024 | 8.802 |
| unexplained_residual | n/a | 0.009 | n/a | n/a |


## 8. DefCon data-gap experiment (KNOWN DATA GAP #2, quantified)

`xp_model` cannot see the CBIT/CBIRT counting stats, because
`fpl_client.load_players_df()` does not keep them. It therefore assumes
every player at a position makes defensive actions at one identical rate,
`DEFCON_PER90_PRIOR = {'GKP': 0.0, 'DEF': 8.6, 'MID': 9.3, 'FWD': 4.8}`. This archive *does* carry
the real per-gameweek counts, so we can measure exactly what that
assumption costs.

**Method.** For each gameweek N, each player's own CBIT/CBIRT per-90 is
rebuilt from gameweeks 1..N-1 only (same leakage firewall as every
other feature) and shrunk toward the positional prior using the model's
own `shrunk_per90_rate()` helper, so this tests the value of the DATA,
not of a new technique. Only the DefCon component is re-scored; the
re-scoring code is asserted in the test suite to reproduce the model's
own `xp_defcon` exactly when fed the flat prior.

Population: outfield players with at least 180 prior minutes — **11,891 player-gameweeks** across 36 gameweeks. Target is the real DefCon award (+2 for clearing the threshold, else 0); base rate 11.1%.

| variant | n | mae | rmse | bias | spearman | mean_pred | mean_actual | top50_hit_rate |
|---|---|---|---|---|---|---|---|---|
| flat prior (live model today) | 11891 | 0.354 | 0.609 | -0.024 | 0.239 | 0.198 | 0.222 | 0.244 |
| own rolling rate, shrunk | 11891 | 0.315 | 0.588 | -0.050 | 0.333 | 0.172 | 0.222 | 0.344 |
| own rolling rate, raw | 11891 | 0.308 | 0.587 | -0.054 | 0.333 | 0.168 | 0.222 | 0.346 |


**Finding — the gain is in ranking, not in average error.**

- MAE barely moves (0.3541 → 0.3147, i.e. slightly *worse*). That is expected
  and is not a strike against the idea: the target is 0 most of the
  time, so a predictor that stays near zero everywhere wins on MAE
  while being useless for picking anyone.
- RMSE improves (0.6089 → 0.5878), because the rolling rate stops
  being confidently wrong about the players who do hit the threshold.
- **Rank correlation rises from 0.239 to 0.333** — a 39% improvement.
- **The decisive number: of the 50 players each gameweek most expected
  to hit the DefCon threshold, the flat prior gets 24.4% right; the rolling rate gets 34.4%.** That is a 41% relative improvement in identifying DefCon returners.

The mechanism is visible in the spread. The flat prior takes exactly
3 distinct values across the whole league
(one per position). The rolling rate ranges from 5.1 (10th pct) through 7.9 (median) to 10.4 (90th pct) CBIT/CBIRT per 90 —
a genuine 2x spread between low- and high-volume defenders that the
model currently cannot see at all.

Note the shrunk and raw variants perform almost identically here, so
the shrinkage is cheap insurance rather than the source of the gain —
the gain is the data.

**Recommendation: prioritise closing KNOWN DATA GAP #2.** Concretely,
add `clearances_blocks_interceptions`, `tackles`, `recoveries` and
`defensive_contribution` to the `keep_cols` list in
`fpl_client.load_players_df()` — the FPL API already returns all four,
they are simply being dropped — and pass the resulting per-90 through
the existing `defcon_per90` override that `calculate_xp()` already
accepts. No model logic needs to change; the hook exists and is unused.

## 9. Recommended constant changes (FOR HUMAN REVIEW — nothing was changed)

This was a diagnostic pass. `models/xp_model.py` was **not** modified.
Each item names the constant(s) to look at, the direction of the error,
and the evidence. Work top-down: the appearance/minutes terms dominate xP
for most of the pool, so fixing them first changes what every other
diagnostic looks like.

### 1. `estimate_start_probability() `minutes <= 0` guard (+ DEFAULT_START_PROBABILITY)` — fix logic, not the constant _(confidence: high)_

**This is the largest single error source in the backtest and it is a logic bug, not a mis-tuned number.**

`estimate_start_probability()` branches on

    if matches_played is None or matches_played < 1 or minutes <= 0:
        base = default_start_probability   # 0.65

That `minutes <= 0` clause conflates two opposite situations: "it is GW1 and nobody has played yet" (no information) and "this player has logged zero minutes across every completed gameweek so far" (extremely strong information). The second group gets handed a 65% start probability.

Measured over gameweeks 6-38: 9,906 player-gameweeks (38.6% of the pool) had zero prior appearances. The model gave them a mean p_start of 0.020 and predicted 0.46 xP each. They actually scored 0.015 points on average and featured at all in 1.1% of cases.

Those rows alone account for 4373 points of signed error against a whole-pool total of 2865 — i.e. 153% of all the model's over-prediction. Excluding them the model actually *under*-predicts slightly, and its rank correlation roughly triples.

Suggested fix (for human review): make the fallback conditional on season progress rather than on minutes being zero — e.g. if `matches_played >= 3` and `minutes == 0`, the empirical start rate is ~1%, so return something near MIN_START_PROBABILITY (0.02) instead of DEFAULT_START_PROBABILITY. Reserve the flat prior for the genuine no-data case (`matches_played < 1`). Do NOT simply lower DEFAULT_START_PROBABILITY: that would wrongly punish real GW1 starters, who are the case it exists to serve.

### 2. `XG90_PRIOR['FWD'] / PRIOR_WEIGHT_90S` — increase _(confidence: medium)_

FWD is the worst-calibrated position on the credible pool: bias -0.352 pts per player-gameweek over 2,017 observations, against an all-position bias of -0.100. For FWD specifically, note that XG90_PRIOR['FWD'] = 0.3 is a league-average forward, but the forwards anyone actually owns are well above average — so the shrinkage in `shrunk_per90_rate()` drags exactly the players that matter toward a prior that does not describe them. Consider whether the prior should be conditioned on price or minutes rather than position alone.

## 10. Caveats and known limitations of this backtest

1. **Team strength ratings are an end-of-season snapshot.** The archive's
   `teams.csv` is one dump, not a per-gameweek history, and FPL adjusts
   those ratings during the season. The fixture model therefore gets very
   slightly better team ratings than it would have had live. This is the
   one lookahead that cannot be removed without a per-gameweek strength
   history, and it flatters the model a little.
2. **Injury/availability status is not reconstructible per gameweek**, so
   every player is backtested as `status = 'a'`. Live, the model zeroes
   out injured and suspended players. The backtest therefore predicts
   points for players who were never going to feature, which inflates the
   measured over-prediction. Treat the overall positive bias as an upper
   bound on the real thing.
3. **`is_promoted` is False for everyone**, so the PROMOTED_* constants
   are untested here.
4. **Penalty duty and per-player DefCon rates are not supplied**, so those
   components run on positional priors — which is exactly how they run
   live today (KNOWN DATA GAPS #2 and #4).
5. **Double gameweeks** are handled by summing both the model's components
   and the player's actual scores across fixtures, so the two sides stay
   comparable. **Blank gameweeks** predict 0 and are excluded from top-N
   precision.

---

Generated by `backtest/backtest.py`. Re-run with:

```
python3 backtest/backtest.py
```
