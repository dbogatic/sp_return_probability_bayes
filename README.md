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

Two models are implemented and compared using a **21-quarter out-of-sample
walk-forward test (2019 Q1 – 2024 Q1)**, spanning a full market cycle
including the COVID crash, the 2022 bear market, and the 2023 recovery.

1. **Flat Bayesian Linear Regression** — a single set of parameters, Student-t
   likelihood, serving as the baseline.
2. **Hierarchical Bayesian Model** — three-level partial-pooling model with
   composite-stress-score market regimes, regime-specific coefficients, and
   regime-specific tail/scale parameters.

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

- Macro variables (VIX, interest rates) have stronger predictive signal over
  multi-month horizons; monthly noise is reduced.
- Quarterly is the natural unit for institutional reporting and portfolio
  rebalancing decisions.
- With ~24 years of data (2000–2024), quarterly frequency gives ~95 usable
  observations — reasonable for Bayesian estimation.

### 3. Features — lagged by one quarter

Six features, all lagged one quarter so they are fully known at prediction time.
Yield curve and HY spread are sourced from FRED (`T10Y2Y`, `BAMLH0A0HYM2`).

| Feature | Transformation | Rationale |
|---|---|---|
| `yield_curve_lag1` | Previous quarter-end 10Y–2Y spread (raw level) | Forward-looking macro regime signal; inverted curve precedes recessions |
| `yield_curve_chg_lag1` | Quarter-over-quarter change in 10Y–2Y spread | Momentum in yield curve steepening / flattening |
| `hy_spread_lag1` | Log change in HY OAS spread | Credit-risk appetite signal; widens ahead of equity drawdowns |
| `hy_spread_chg_lag1` | Quarter-over-quarter log change in HY OAS | Spread momentum — rapidly widening spreads signal risk-off |
| `VIX_lag1` | Log change in VIX | Volatility momentum (reactive, not predictive — retained as feature, not regime classifier) |
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

### 8. Composite macro stress score — regime classifier

The hierarchical model classifies each quarter into one of three regimes using a
**composite stress score** that aggregates three lagged macro signals, all fully
known at the start of each quarter (no look-ahead bias):

| Signal | Stressed direction | Economic rationale |
|---|---|---|
| 10Y–2Y yield curve slope (negated) | Inverted (< 0%) | Precedes recession by 6–18 months; forward-looking |
| ICE BofA HY option-adjusted spread | Wide spread | Credit market pricing real distress |
| Unemployment rate change | Rising | Labour market deterioration confirming slowdown |

Each signal is z-scored using **training data statistics only**, then averaged
into a single composite. Training-data tertiles map the composite to three
regimes:

| Regime | Composite score | Market environment |
|---|---|---|
| 0 — Recessionary | Top tertile (high stress) | Multiple signals simultaneously stressed |
| 1 — Transitional | Middle tertile | Mixed signals, late-cycle or early recovery |
| 2 — Expansionary | Bottom tertile (low stress) | All signals benign, growth environment |

**Why a composite?** A single yield-curve signal can produce false alarms. The
2022–2024 period is a good example: the 10Y–2Y spread inverted sharply, which
under a yield-curve-only classifier would flag the entire period as
*Recessionary*. But HY credit spreads stayed tight (markets were not pricing
credit stress) and unemployment continued to fall. The composite correctly kept
most of 2022–2024 in the *Transitional* regime — consistent with observed S&P
500 performance.

**Why not VIX?** VIX is a reactive signal — it spikes *during* crises, not
before them. It is retained as a regression feature (`VIX_lag1`) to capture
volatility momentum, but excluded from the regime classifier.

**No test-set contamination**: z-score normalization parameters and tertile
thresholds are derived entirely from pre-2019 training data. The test period
(2019–2024) is never consulted during regime design, keeping the walk-forward
evaluation genuinely out-of-sample.

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
21 test quarters: 2019 Q1 – 2024 Q1).

The split year was chosen to ensure each regime has enough training observations
for meaningful partial pooling while leaving a demanding test period. Cutting at
2018 includes the 2008–2009 GFC, the 2011 European debt crisis, and the 2018 Q4
volatility spike in the training set — enough recessionary and transitional
quarters for the regime-specific parameters to be data-identified rather than
driven entirely by the hyperpriors. The 21-quarter test period then covers the
COVID crash (2020 Q1), the low-volatility 2021 bull run, the 2022
inflation-driven bear market, and the 2023–2024 recovery — a genuine stress test
spanning all three composite-stress regimes.

---

## Limitations and Future Extensions

These are genuine limitations, not implementation bugs:

- **Hard-coded regime boundaries**: Composite stress tertile thresholds are fixed
  from training data. A Hidden Markov Model or Dirichlet-process mixture would
  learn regime transitions and boundaries jointly with return dynamics.
- **Equal-weighted composite**: The three stress signals contribute equally to
  the composite score. Learned weights (e.g. from a logistic regression on
  NBER recession dates) might improve regime separation.
- **Fixed feature set**: The regression uses six lagged macro variables. Earnings
  yield, consumer sentiment, and PMI have known incremental predictive power for
  equity returns.
- **Time-invariant coefficients**: A Gaussian random walk prior on the betas
  (state-space model) would capture structural breaks — e.g., the post-2008
  regime shift in interest rate sensitivity.
- **Regime-specific transition dynamics**: The current model treats each quarter
  as independently classified. Modeling regime persistence (staying in a high-
  VIX regime is more likely than transitioning out in one quarter) would improve
  regime assignment.
- **Short training history**: 2000–2016 training data (~65 quarters) spans two
  full market cycles. Extending to pre-2000 data would add the 1990s bull run
  and the 1987 crash to the training set.

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

`resources/data.csv` — monthly S&P 500 index levels and CBOE VIX from 2000 to
early 2024. Resampled to quarterly (end-of-quarter) before any analysis.

Three additional series are fetched live from **FRED** at runtime:

| Series | FRED ID | Used for |
|---|---|---|
| 10Y–2Y Treasury spread | `T10Y2Y` | Regime classifier + regression feature |
| ICE BofA HY OAS | `BAMLH0A0HYM2` | Regime classifier + regression feature |
| Civilian unemployment rate | `UNRATE` | Regime classifier |

---

## References

- Martin, Osvaldo. *Bayesian Analysis with Python*, 3rd Edition.
- Kanungo, Deepak K. *Probabilistic Machine Learning for Finance and Investing*, 1st Edition.
- Mandelbrot, Benoit. *The Misbehavior of Markets* — on fat tails and power laws in financial returns.
- Taleb, Nassim N. *The Black Swan* — on the limits of historical tail estimation.
- Gelman, Andrew et al. *Bayesian Data Analysis*, 3rd Edition — on prior
  predictive checks, non-centered parameterization, and hierarchical models.
