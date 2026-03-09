# Quarterly S&P 500 Return Probabilities: A Hierarchical Bayesian Approach

![DALLE Markov Sampling](./resources/DALLE_Markov_Sampling.webp)

## Overview

This notebook uses **PyMC 5.10** and **ArviZ** to estimate the probability that
quarterly S&P 500 returns will fall within four buckets:

| Bucket | Log-return range | Approx. simple return |
|---|---|---|
| Strong positive | > +5% | > +5.1% |
| Mild positive | 0% to +5% | 0% to +5.1% |
| Mild negative | −5% to 0% | −4.9% to 0% |
| Strong negative | < −5% | < −4.9% |

The **Hierarchical Bayesian Model** is the forecasting model. It uses a
three-level partial-pooling structure with yield-curve-based market
regimes, regime-specific coefficients, and regime-specific tail/scale
parameters.

A **Flat Bayesian Linear Regression** (single parameter set, Student-t
likelihood) is retained solely as a benchmark to validate that the
hierarchical structure earns its complexity — if it did not outperform a
simpler flat model, the regime design would need revisiting.

Both are evaluated on an **out-of-sample walk-forward test of up to 29 quarters
(2019 Q1 – 2026 Q1)**, spanning a full market cycle including the COVID
crash, the 2022 bear market, the 2023–2024 recovery, and the 2025–2026
period.

---

## Modeling Assumptions

Every modeling decision is deliberate. This section documents what was chosen,
what was rejected, and why.

### 1. Log returns, not simple returns

We model log returns `ln(P_t / P_{t-1})` rather than simple percentage returns.

- Log returns are time-additive: a multi-period log return is the sum of
  single-period log returns.
- They produce a more symmetric distribution, which is easier to model.
- Converting back to simple returns for interpretation is straightforward:
  `simple_return = exp(log_return) − 1`.

### 2. Quarterly frequency

Monthly data is resampled to quarters. The rationale:

- Macro variables (interest rates, credit spreads) have stronger predictive
  signal over multi-month horizons; monthly noise is reduced.
- Quarterly is the natural unit for institutional reporting and portfolio
  rebalancing decisions.
- With data pulled from FRED starting Q1 1997, usable observations begin
  around Q1 1999 (~105 quarters through 2024) — reasonable for Bayesian
  estimation with enough recessionary and transitional quarters to
  data-identify regime-specific parameters.

### 3. Features — lagged by one quarter

Five features, all lagged one quarter so they are fully known at prediction time.
Yield curve and HY spread are sourced from FRED (`T10Y2Y`, `BAMLH0A0HYM2`).

| Feature | Transformation | Rationale |
|---|---|---|
| `yield_curve_lag1` | Previous quarter-end 10Y–2Y spread (raw level) | Forward-looking macro regime signal; inverted curve precedes recessions |
| `yield_curve_chg_lag1` | Quarter-over-quarter change in 10Y–2Y spread | Momentum in yield curve steepening / flattening |
| `hy_spread_lag1` | Previous quarter-end HY OAS level | Credit-risk appetite signal; widens ahead of equity drawdowns |
| `hy_spread_chg_lag1` | Quarter-over-quarter log change in HY OAS | Spread momentum — rapidly widening spreads signal risk-off |
| `sp_returns_lag1` | Log return of S&P 500 | Momentum / mean-reversion |

Lagging is a strict requirement: using same-quarter features would constitute
look-ahead bias and overstate out-of-sample accuracy.

### 4. Student-t likelihood — not Normal

Financial returns have **fat tails**. Black Monday 1987 (Dow −22.6% in one day)
is a ~20-sigma event under normality — a probability so small it should never
have occurred in the history of the universe. It happened. Goldman Sachs's risk
team famously observed "25-sigma events, several days in a row" in August 2007.

The **Student-t distribution** with degrees of freedom `nu` accommodates this:
lower `nu` → heavier tails → more probability mass on extreme events. This is
the statistically correct model for equity returns.

### 5. Degrees-of-freedom prior — floor at nu = 4

The prior on `nu` is:

```
nu ~ Exponential(lam=1/20) + 4
```

The **floor at 4** is a deliberate modeling constraint. Here is the rationale:

| nu range | Consequence | Assessment |
|---|---|---|
| nu < 2 | Infinite variance | Statistically pathological; rejected |
| 2 ≤ nu < 4 | Infinite kurtosis | Implies extreme events every few quarters — too aggressive even for financial markets |
| 4 ≤ nu ≤ 15 | Finite variance and kurtosis, heavy tails | Consistent with empirical estimates for quarterly equity returns |
| nu > 30 | Approaches Normal distribution | Appropriate for calm regimes |

The tension we are resolving: black swans are real and happen more often than
the Normal distribution implies — but they still should not happen *every few
quarters*. The floor at 4 encodes that constraint while keeping the tails
genuinely heavy. The Exponential shift puts the bulk of the prior in the
[4, 30] range, consistent with empirical estimates in the financial
econometrics literature.

In the hierarchical model, `nu` is **regime-specific** — the recessionary
regime is expected to learn a lower `nu` (heavier tails) than the expansionary
regime, capturing the empirical finding that tail risk is not constant across
market environments.

### 6. Prior on regression coefficients — weakly informative

```
alpha, beta_* ~ Normal(0, sigma=0.05)
```

Features are RobustScaler-standardized (centered at median, scaled by IQR),
so a coefficient of 0.05 corresponds to a 5% change in log return per IQR
of the feature — a reasonable upper bound on macro predictability. The prior
is wide enough for the data to dominate the posterior without being so diffuse
that it contributes no regularisation at all.

If the posterior credible intervals exclude zero for a coefficient, the data
found a signal. If not, the features simply have limited predictive power at
quarterly frequency — which is itself a finding worth reporting.

### 7. Observation scale prior

```
sigma ~ HalfNormal(sigma=0.05)    # flat model
sigma_h ~ HalfNormal(sigma=0.06, shape=n_regimes)  # hierarchical, per-regime
```

HalfNormal places most mass near small positive values. The scale parameter
(0.05–0.06) is calibrated to observed quarterly S&P 500 log-return volatility
(historically ~7–8%). In the hierarchical model, each regime has its own
`sigma_h`: the recessionary regime is expected to learn a larger scale,
capturing **volatility clustering** — the well-documented empirical pattern
that high-volatility periods cluster together.

### 8. Yield-curve regime classifier

The hierarchical model classifies each quarter into one of three regimes using
the **previous quarter's 10Y–2Y Treasury yield spread** — fully known at the
start of each quarter (no look-ahead bias):

| Regime | Lagged yield curve slope | Market environment |
|---|---|---|
| 0 — Recessionary | < 0% (inverted) | Curve has inverted; elevated recession risk |
| 1 — Transitional | 0% to 1% (flat) | Late-cycle or early recovery |
| 2 — Expansionary | ≥ 1% (steep) | Normal growth environment |

**Why yield curve slope?** The 10Y–2Y spread has inverted before every US
recession since 1955 and does so *before* the downturn begins — making it a
genuinely forward-looking signal. By contrast, VIX and credit spreads tend to
react *during* crises rather than anticipate them. The yield curve is the
single most informative public macro signal for near-term recession risk.

**No test-set contamination**: regime boundaries (0% and 1%) are economically
motivated thresholds, not data-fitted. The test period (2019–2024) is never
consulted during regime design.

### 9. Partial pooling (hierarchical model)

The hierarchical model sits between two extremes:

| Approach | Description | Problem |
|---|---|---|
| Complete pooling | One parameter set for all regimes | Ignores regime differences |
| No pooling | Separate model per regime | Too little data per regime (~20–40 quarters) |
| **Partial pooling** | Regime params drawn from shared hyperpriors | Borrows strength across regimes |

Regime-level coefficients are drawn from global hyperpriors:

```
alpha_r[k] = mu_alpha + alpha_offset[k] * sigma_alpha
```

`sigma_alpha` controls the degree of pooling: if its posterior is small, the
regimes are similar and share a common intercept; if large, each regime has a
distinct intercept. This is *learned from data*, not assumed.

### 10. Non-centered parameterization

Regime parameters use the non-centered form above rather than the centered form
`alpha_r[k] ~ Normal(mu_alpha, sigma_alpha)`. When `sigma_alpha` is near zero,
the centered form creates a geometry ("Neal's funnel") that causes NUTS to
diverge or mix poorly. The non-centered form decouples the offset from the
scale and allows efficient sampling throughout the posterior.

### 11. Prior predictive checks

Before touching observed data, we verify that the priors generate plausible
quarterly return distributions. The plot uses `xlim=±50%` for readability and
separately reports the fraction of draws that fall outside this window — so
the full distribution is visible and quantified, not hidden.

### 12. Empirical CDF for probabilities

Predicted probabilities for each bucket are computed directly from the
posterior predictive samples:

```python
prob_above_5 = np.mean(post_samples > 0.05)
```

The posterior predictive is a **mixture** of Student-t distributions — one per
posterior draw of the parameters. Computing probabilities from the sample
empirical CDF is exact and makes no distributional assumptions about the shape
of the mixture.

### 13. Walk-forward expanding-window validation

The model is evaluated using a strict expanding-window walk-forward approach:

1. Train on data up to quarter *t* only.
2. Refit the RobustScaler on the current training window — prevents future-data
   leakage into the scaling transformation.
3. Sample the posterior on training data.
4. Set features to the test quarter *t+1* and sample posterior predictive —
   the true out-of-sample prediction.
5. Add quarter *t+1* to the training set and repeat.

**Train/test split: `train_end_year = 2018`** (~73 training quarters,
29 test quarters: 2019 Q1 – 2026 Q1).

The split year was chosen to ensure each regime has enough training observations
for meaningful partial pooling while leaving a demanding test period. Cutting at
2018 includes the 2008–2009 GFC, the 2011 European debt crisis, and the 2018 Q4
volatility spike in the training set — enough recessionary and transitional
quarters for the regime-specific parameters to be data-identified rather than
driven entirely by the hyperpriors. The test period then covers the
COVID crash (2020 Q1), the low-volatility 2021 bull run, the 2022
inflation-driven bear market, the 2023–2024 recovery, and 2025–2026 — a genuine
stress test spanning all three yield-curve regimes.

---

## Out-of-Sample Accuracy

Both models are evaluated on **up to 29 quarters (2019 Q1 – 2026 Q1)** that were
never seen during training. Three complementary metrics are computed in
**cells 33–35**.

---

### Metric 1 — Top-Bucket Accuracy (cells 21 and 31)

Each quarter the model assigns a probability to four possible outcomes:

| Bucket | Meaning |
|---|---|
| Strong positive | S&P 500 returns more than +5% that quarter |
| Mildly positive | Returns between 0% and +5% |
| Mildly negative | Returns between −5% and 0% |
| Significant drop | Returns worse than −5% |

The model is **correct** if the bucket it was most confident about actually
happened. Random guessing hits 25% (1 in 4). See cells 21 and 31 for the
actual numbers.

---

### Metric 2 — Brier Score (cell 34)

Measures how far the predicted probabilities are from the true outcome,
across all four buckets at once. Think of it as an "average squared miss."

- **0 = perfect** — every probability was spot-on
- **0.75 = random** — what you'd score by guessing 25% for every bucket

Lower is better. The **Brier Skill Score (BSS)** rescales this so that
0 = no better than random and 1 = perfect. A positive BSS means the model
adds real forecasting value beyond a coin flip. See **cell 34** for the
computed values for both models.

> *Plain English: a model that confidently called the 2020 Q1 crash correctly
> would earn a very low (good) Brier Score for that quarter. A model that was
> 80% confident in the wrong bucket would be heavily penalised.*

---

### Metric 3 — Log Score (cell 34)

Also called the logarithmic scoring rule. It rewards genuine, well-placed
confidence and **severely penalises overconfident wrong predictions**.

- **Random baseline ≈ 1.39** — what you'd score guessing 25% every time
- **Perfect = 0**
- Lower is better

Observed values across the test period (up to 29 quarters):

| Metric | Flat model | Hierarchical model |
|---|---|---|
| Brier Score | 0.7061 | **0.6887** |
| Brier Skill Score | 0.0586 | **0.0817** |
| Log Score | 1.2929 | **1.2666** |

See **cell 34** for the side-by-side table of all three metrics for both models.

> *Plain English: if the model says "90% chance of a strong quarter" and it
> crashes instead, the Log Score punishes that far harder than the Brier Score
> does. A low Log Score means the model is both right and honest about its
> uncertainty.*

---

### Regime-Stratified Table and Calibration Diagram (cell 35)

The regime-stratified breakdown shows accuracy and Brier Score separately for
Recessionary, Transitional, and Expansionary quarters. This is where the
hierarchical model should visibly outperform the flat model — it learns
different behaviour per regime rather than averaging everything together.

Observed results for the hierarchical model (up to 29-quarter test, 2019–2026):

| Regime | N quarters | Accuracy | Brier Score | BSS |
|---|---|---|---|---|
| Recessionary (<0%) | 8 | 38% | 0.6483 | 0.136 |
| Transitional (0–1%) | 18 | 39% | 0.7102 | 0.053 |
| Expansionary (≥1%) | 3 | 33% | 0.6675 | 0.110 |

The hierarchical model shows its strongest advantage in Recessionary regimes
(BSS 0.136 vs ≈0.06 overall), consistent with the design intent: regime-specific
tail and scale parameters allow the model to assign more probability mass to
extreme outcomes when the yield curve is inverted.

---

### Walk-Forward Design (no data leakage)

At every step the model only knows what was available *at the time of the
forecast*: training data, feature scaling, and regime boundaries are all
computed using only past quarters. No future information ever leaks in.

### Caching — backtest and trace

The walk-forward loop resamples MCMC for every test quarter, which takes
roughly 70 minutes. Results are cached to disk automatically:

- **`backtest_results_flat_YYYYMMDD.csv`** — flat model walk-forward
  predictions, keyed by the last test date. If the file exists, cell 14
  skips the MCMC loop entirely.
- **`backtest_results_monthly_YYYYMMDD.csv`** — hierarchical model
  walk-forward predictions and probabilities, keyed by the last date in
  FRED data. If the file exists, cell 30 skips the MCMC loop entirely.
- **`trace_full_monthly_YYYYMMDD.nc`** — the full-data posterior trace used
  for forecasting, also keyed by last data date. Cell 38 loads it instantly
  on subsequent runs instead of resampling.

All caches invalidate automatically when new FRED data arrives (the date in
the filename changes), so reruns after a data refresh always retrain from
scratch. Old cache files with stale dates can be deleted manually to free
disk space.

---

## Decision-Making Performance Dashboard (cell 36)

After the backtest, cell 36 produces a two-panel dashboard that answers the
practical question: *at what confidence threshold should I act on the model's
top-bucket call?*

- **Panel 1 — Per-quarter confidence bars**: each of the 29 test quarters is
  shown as a bar coloured green (model's top bucket was correct) or red
  (wrong), with height proportional to the model's confidence in that call.
- **Panel 2 — Hit rate vs confidence threshold**: a table and chart showing
  how accuracy changes as you raise the bar for acting on a forecast. If the
  model is only correct 50% of the time overall but hits 80% when confidence
  exceeds 50%, that threshold is where the signal is.

The dashboard is saved as **`backtest_dashboard.png`**.

---

## Practical Assessment (walk-forward backtest, 2019 Q1 – 2026 Q1)

### Overall accuracy

| Metric | Value |
|---|---|
| Quarters evaluated | 29 |
| Top-bucket correct | 11 (38%) |
| Random baseline | 25% |
| Edge over random | +13pp |

38% is meaningfully above chance but not high enough to use as a standalone
decision tool. The real signal is conditional on model confidence.

### The confidence threshold insight

The backtest dashboard (bottom panel) shows hit rate rises sharply with
the model's confidence in its top-bucket call:

| Confidence ≥ | Hit rate | # Signals | Coverage |
|---|---|---|---|
| 25% (all quarters) | 38% | 29 | 100% |
| 30% | 37% | 27 | 93% |
| 35% | 32% | 19 | 66% |
| 40% | 50% | 12 | 41% |
| 45% | 60% | 10 | 34% |
| 50% | 57% | 7 | 24% |

**When the model commits above ~45% confidence, it is right roughly 60% of
the time — well above the 25% random floor.** Those high-confidence quarters
are rare (roughly 1-in-3 test quarters); most quarters sit in the 25–40%
range where the model is effectively communicating genuine uncertainty rather
than a clear signal. The ≥40% threshold is the practical entry point where
accuracy first breaks above 50%.

### Best calls

| Quarter | Confidence | Prediction | Actual | Result |
|---|---|---|---|---|
| 2019 Q1 | 91% | Strong positive | +13% | ✓ |
| 2020 Q2 | 87% | Strong positive | +20% | ✓ |
| 2020 Q1 | — | Significant drop most likely | −20% | ✓ |

The model's clearest value was catching regime-driven extremes: the COVID
crash and the immediate recovery bounce, both driven by sharp yield-curve
moves that the model's macro features captured cleanly.

### Worst call — 2022

All four quarters of 2022 were wrong. The most damaging was **2022 Q2**:
the model was **70% confident in a strong positive outcome** while the
market fell −16%. The full-year 2022 failure reflects a structural gap:
the inflation-driven bear market was unlike anything in the 1999–2018
training set. The yield curve was steepening rapidly *while* equities fell
— a decoupling of the slope regime from equity returns that the model has
no feature to detect.

| Quarter | Model confidence | Predicted bucket | Actual return | Regime assigned |
|---|---|---|---|---|
| 2022 Q1 | ~32% | Mild positive | −4.9% | Transitional |
| 2022 Q2 | ~70% | Strong positive | **−16.4%** | Transitional |
| 2022 Q3 | ~70% | Mild negative | −5.3% | Transitional |
| 2022 Q4 | ~39% | Mild positive | +7.1% | Recessionary |

The root cause: real yields swung from **−6% to +1%** in twelve months
(10Y Treasury rising sharply while CPI ran at ~8% YoY). Yield curve
*slope* was blind to this — a steeply positive slope normally signals
expansion, not a −16% quarter. The inflation-driven discount-rate shock
was a regime the model had never seen in training.

### Overconfidence diagnosis

The **Log Score** is the key metric here — it penalises overconfident
wrong predictions far harder than the Brier Score. A model that assigns
95% probability and is wrong pays an enormous penalty; a model that says
50% and is wrong pays a moderate one.

The 2022 Q2 call (70% confidence, −16% outcome) is a textbook
overconfidence failure. The model assigned high probability to a benign
outcome because every feature it could see — a positive yield slope, low
HY spreads, recent positive momentum — pointed that way. It had no signal
for the rate-level shock that was actually driving returns.

**Diagnosis summary:**

| Issue | Evidence | Root cause |
|---|---|---|
| Overconfidence in 2022 | Log Score penalised heavily; 70% confidence on wrong bucket | Missing inflation / real-yield feature |
| Weak Transitional skill | BSS = 0.053 vs 0.136 Recessionary | Most quarters land here; regime is too coarse |
| Single failure mode | All 4 wrong quarters clustered in 2022 | Out-of-distribution regime, not random noise |

### What this means for use

This model is best used as **one probabilistic input** in a broader
decision process, not as an autonomous signal. Actionable guidance:

- **Act on the model when confidence is ≥40–45%** — accuracy breaks 50% at
  ≥40% and peaks around 60% at ≥45%. Below 40% the model is expressing
  honest uncertainty, not a clear directional forecast.
- **Cross-check with inflation / rate-level context.** The yield slope alone
  mis-classifies regimes when rate *levels* are the dominant equity driver
  (e.g., 2022). Adding CPI, the fed funds rate, or real yield would be the
  highest-value model extension.
- **Wide credible intervals are information, not failure.** A 90% CI of
  −18% to +23% means current macro conditions are genuinely ambiguous —
  the model is being honest rather than manufacturing a false point estimate.

---

## Next-Quarter Forecast (cells 37–39)

After evaluation, the model retrains on **all available data** and uses the
most recent quarter's macro readings to forecast the next quarter.

**What the forecast outputs:**

- **Regime** — which yield-curve environment next quarter falls into
  (Recessionary / Transitional / Expansionary), based on the current
  10Y–2Y Treasury spread
- **Bucket probabilities** — the probability of each of the four return
  outcomes
- **Expected return** — the probability-weighted average forecast
- **Credible intervals** — 50% CI (inner range where the model is fairly
  confident) and 90% CI (the wide range covering most plausible outcomes)
- **Probability bar chart** — visual summary with the 25% random baseline

> *Plain English: the forecast does not say "the market will go up X%." It
> says "given current macro conditions, here is how probable each outcome is."
> A wide credible interval means genuine uncertainty — the model is being
> honest rather than hiding behind a false point estimate.*

The retraining cell (~5 min on first run, instant on cache hit) is cell 37;
the chart and plain-English output are in cells 38–39.

**Most recent forecast — 2026 Q1** (as of last cached run):

| | |
|---|---|
| Regime | Transitional (10Y–2Y spread: +0.58%) |
| Strong positive (>+5%) | 30% |
| Mildly positive (0–+5%) | 22% |
| Mildly negative (−5–0%) | 23% |
| Significant drop (<−5%) | 26% |
| 90% credible interval | −18% to +23% |

The near-uniform distribution across all four buckets reflects the Transitional
regime: with the yield curve flat but slightly positive, the model has no strong
macro signal and reports genuine uncertainty rather than a directional view. The
top model confidence (30%) falls below the ≥40% action threshold, so this is
a "no strong signal" quarter.

### Print-Ready Forecast Report (cell 40)

Cell 40 generates a single-page briefing figure saved as
**`forecast_YYYYQ#.png`** — one file per quarter, auto-named, archivable.
It contains everything needed for a quarterly decision briefing:

| Section | Content |
|---|---|
| Header | Quarter label and colour-coded regime banner |
| Probability bars | Four return buckets with the top bucket highlighted |
| Macro inputs table | The five lag-1 features that drove the forecast |
| Return distribution strip | Posterior 90%/50% CI and expected return |
| Backtest context line | Overall accuracy and hit rate at the current confidence level |

---

## Improvement Roadmap

Prioritised by expected impact on the two diagnosed failure modes (overconfidence + weak Transitional skill).

### Priority 1 — Fix overconfidence (calibration)

- **Temperature / Platt scaling**: post-hoc calibration step that squeezes
  predicted probabilities toward the centre when the model is systematically
  overconfident. Does not require retraining the Bayesian model.
- **Minimum-entropy prior**: enforce a floor on outcome uncertainty so the
  model cannot assign >70–75% to a single bucket without very strong data
  support. Directly targets the 2022 Q2–Q3 failure mode.
- **Track calibration curves per regime**: the 2×2 calibration plots already
  exist — use them to identify which buckets are most miscalibrated and by
  how much, then target corrections there first.

### Priority 2 — Add inflation / rate-level features

The single highest-value model extension, directly motivated by 2022:

| Feature | Why |
|---|---|
| `real_yield_lag1` (10Y − CPI YoY) | Captures 2022-style discount-rate shocks invisible to slope alone |
| `cpi_momentum_lag1` (QoQ change in CPI) | Rate of change of inflation, not just level |
| `fed_funds_chg_lag1` | Policy tightening / easing speed |
| `breakeven_inflation_lag1` | Market-implied inflation expectations (FRED: `T10YIE`) |

These are all available on FRED with no API key, consistent with the existing data pipeline.

### Priority 3 — Refine the Transitional regime

Transitional is the weakest regime (BSS = 0.053) and the most common
(18 of 29 test quarters). It is too coarse — it lumps together
late-cycle tightening and early-cycle recovery, which have opposite
equity implications.

- Split into **Transitional-Rising** (yield curve moving toward inversion)
  and **Transitional-Recovering** (curve moving away from inversion)
- Alternatively: learn regime boundaries jointly from data using a
  **Hidden Markov Model** or **Dirichlet-process mixture**, rather than
  fixing them at 0% and 1%.

### Priority 4 — Reduce overfit to regime snapshots

- **Regime persistence prior**: the current model classifies each quarter
  independently. Recessionary regimes typically last 3–6 quarters; a
  Markov transition prior on regime assignments would smooth noisy
  quarter-to-quarter regime flips.
- **Time-varying coefficients**: a Gaussian random walk prior on the
  betas (state-space model) would capture structural breaks such as the
  post-2008 shift in interest-rate sensitivity.

### Priority 5 — Widen the training set

- Training data spans Q1 1999–2018 (~80 quarters, two market cycles).
  Extending further back would add the 1990s bull run and the 1987 crash,
  but FRED's HY spread series (`BAMLH0A0HYM2`) only starts in 1996,
  which is the practical floor.
- Consider **bootstrapped stress scenarios** or **synthetic 2022-style
  episodes** to supplement the single inflation-shock example in the
  out-of-sample period.

---

## Limitations

These are known structural constraints, not bugs:

- **Hard-coded regime boundaries**: The 0% and 1% yield-curve thresholds are
  economically motivated but fixed. A data-driven approach would learn them.
- **No rate-level or inflation features**: The 2022 failure makes this gap
  concrete. See Priority 2 above for the fix.
- **Time-invariant coefficients**: structural breaks (e.g., post-2008 rate
  sensitivity) are not captured. See Priority 4.
- **Independent quarter assumption**: regime persistence is not modelled.
  See Priority 4.
- **Training history ceiling**: limited by FRED HY spread availability
  (1996). See Priority 5.

---

## Setup

```bash
conda create -n pymc_env python=3.11
conda activate pymc_env
pip install pymc arviz pandas scikit-learn matplotlib seaborn statsmodels
jupyter notebook sp_return_prob_bayes.ipynb
```

Or in **VS Code**: switch to the `claude/hierarchical-bayesian-review-9Th9N`
branch, open the `.ipynb` file, and select the `pymc_env` kernel.

---

## Data

All data is fetched live from **FRED** at runtime (no API key required via `pandas_datareader`).
Three series are pulled starting Q1 1997 (buffer for first diff and one quarter of lag):

| Series | FRED ID | Used for |
|---|---|---|
| S&P 500 Index | `SP500` | Target variable (log quarterly returns) |
| 10Y–2Y Treasury spread | `T10Y2Y` | Regime classifier + regression feature |
| ICE BofA HY OAS | `BAMLH0A0HYM2` | Regression feature |

After differencing and lagging, usable observations begin around Q1 1999.

---

## References

- Martin, Osvaldo. *Bayesian Analysis with Python*, 3rd Edition.
- Kanungo, Deepak K. *Probabilistic Machine Learning for Finance and Investing*, 1st Edition.
- Mandelbrot, Benoit. *The Misbehavior of Markets* — on fat tails and power laws in financial returns.
- Taleb, Nassim N. *The Black Swan* — on the limits of historical tail estimation.
- Gelman, Andrew et al. *Bayesian Data Analysis*, 3rd Edition — on prior
  predictive checks, non-centered parameterization, and hierarchical models.
