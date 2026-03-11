# Bayesian S&P 500 Return Probability Model

## Overview

This project estimates the probability that next-quarter S&P 500 returns will fall within
one of four discrete return buckets, using a hierarchical Bayesian model implemented in PyMC.

The output is a **probability distribution across four outcomes**, not a point forecast or
a directional signal. Wide credible intervals reflect genuine macro uncertainty rather than
model failure.

---

## Return Buckets

| Bucket | Return Range |
|---|---|
| Strong positive | > +5% |
| Mildly positive | 0% to +5% |
| Mildly negative | −5% to 0% |
| Significant drop | < −5% |

---

## Model Architecture

The model is implemented in **PyMC 5** with the following design:

- **Likelihood**: Student-t (accommodates fat tails in equity return distributions)
- **Structure**: Hierarchical partial pooling across three yield-curve regimes
- **Predictors**: Six lagged macro features (see [Model Inputs](#model-inputs))
- **Evaluation**: Walk-forward backtesting with a refit every 3 months
- **Parameterization**: Non-centered, to avoid sampling geometry issues near zero pooling

A flat Bayesian linear regression (single parameter set, same likelihood) is retained as a
benchmark throughout.

---

## Yield-Curve Regimes

Each month is classified into one of three regimes using the **prior month's 10Y–2Y Treasury
yield spread** — fully known at forecast time, no look-ahead bias.

| Regime | 10Y–2Y Spread | Description |
|---|---|---|
| 0 — Recessionary | < 0% | Inverted curve; elevated recession risk |
| 1 — Transitional | 0% to 1% | Flat curve; late-cycle or early recovery |
| 2 — Expansionary | > 1% | Steep curve; normal growth environment |

---

## Model Inputs

All six predictors are lagged one period. Same-period features would introduce look-ahead
bias and overstate out-of-sample accuracy.

| Feature | Description |
|---|---|
| `yield_curve_lag1` | Previous month-end 10Y–2Y spread (level) |
| `yield_curve_chg_lag1` | Month-over-month change in 10Y–2Y spread |
| `hy_spread_lag1` | Previous month-end HY OAS level |
| `hy_spread_chg_lag1` | Month-over-month log change in HY OAS |
| `mom_12_1` | 12-1 month equity price momentum |
| `real_yield_lag1` | 10Y Treasury minus CPI YoY (real yield) |

---

## Backtest Methodology

| Parameter | Value |
|---|---|
| Test period | 2020 Q4 – 2025 Q4 |
| Forecast horizon | Quarterly |
| Refit frequency | Every 3 months |
| Total forecasts evaluated | 21 quarters |
| Train/test split year | 2018 |

The walk-forward design trains only on data available prior to each forecast date.
The RobustScaler is refit on the training window at each step to prevent future-data
leakage into feature scaling.

---

## Model Variants Tested

Five structural configurations were compared in a single-pass sensitivity analysis
(n = 21 quarters). Variants B through E are **sensitivity benchmarks only** and are
not used in production forecasts.

| Variant | nu | σ_α prior | σ_β prior | σ_h prior | Notes |
|---|---|---|---|---|---|
| **A_baseline** ◄ | Estimated per regime | 0.03 | 0.02 | 0.04 | Current research baseline |
| B_fixed_nu7 | Fixed at 7 | 0.03 | 0.02 | 0.04 | Fixed-nu sensitivity test |
| C_loose | Estimated per regime | 0.10 | 0.05 | 0.04 | Looser pooling sensitivity test |
| D_fixed_nu7_loose | Fixed at 7 | 0.10 | 0.05 | 0.04 | Fixed-nu + loose pooling test |
| E_regime_sigma | Estimated per regime | 0.03 | 0.02 | 0.06 | Wider per-regime scale test |

`A_baseline` achieved the best log score across the sensitivity comparison.
`NU_FIXED = None` is the active configuration; setting `NU_FIXED = 7` in the config cell
reverts to the fixed-nu path if sampling divergences reappear.

---

## Backtest Results

Walk-forward evaluation over 21 quarters (2020 Q4 – 2025 Q4). Random baseline accuracy = 25%.

| Metric | Flat Model | Hierarchical Model |
|---|---|---|
| Top-bucket accuracy | 40% | **43%** |
| Brier Score | **0.7140** | 0.7834 |
| Brier Skill Score | **0.0479** | −0.0445 |
| Log Score | **1.3248** | 1.4699 |

The hierarchical model improved top-bucket directional accuracy (43% vs 40% vs 25% random).
However, its probabilistic calibration was conservative: the negative Brier Skill Score
indicates the predicted probability distributions were not better calibrated than a random
uniform forecast. The flat model scored better on both Brier and Log Score.

---

## Variant Comparison

Walk-forward comparison of `A_baseline` against `E_regime_sigma` (REFIT_EVERY = 3).

| Metric | A_baseline (σ_h = 0.04) | E_regime_sigma (σ_h = 0.06) |
|---|---|---|
| Top-bucket accuracy | **43%** | 33% |
| Brier Score | **0.7834** | 0.7950 |
| Brier Skill Score | **−0.0445** | −0.0599 |
| Log Score | **1.4699** | 1.4862 |

`A_baseline` outperforms `E_regime_sigma` on both accuracy and calibration metrics.
Widening the per-regime scale prior did not improve forecasts.

---

## Diagnostics

**Regime observation counts (full dataset):**

| Regime | Months |
|---|---|
| Recessionary (<0%) | 49 |
| Transitional (0–1%) | 114 |
| Expansionary (>1%) | 159 |

**Pooling diagnostics:**

- `sigma_alpha` posterior mean ≈ 0.012 (94% HDI: [0.0005, 0.039]). Near zero — regime
  intercepts are heavily pooled, meaning the three regimes share nearly the same mean
  return level.
- All `alpha_r` HDIs overlap across regime pairs. The model finds no credible difference
  in mean returns across regimes.
- Regime-specific `sigma_h` posteriors are distinct (Recessionary ≈ 0.030,
  Transitional ≈ 0.036, Expansionary ≈ 0.040), suggesting regimes affect **return
  volatility** more than mean returns.

---

## Example Forecast

**2026 Q2 forecast** — generated from features at 2026-01-31.

| | |
|---|---|
| Regime | Transitional (0–1%) |
| 10Y–2Y spread | +0.74% |
| Median quarterly return | +1.2% |
| 50% credible interval | −3.9% to +6.2% |
| 90% credible interval | −10.8% to +14.4% |

**Return bucket probabilities:**

| Bucket | Probability |
|---|---|
| > +5% (strong positive) | 30% |
| 0% to +5% (mildly positive) | 26% |
| −5% to 0% (mildly negative) | 24% |
| < −5% (significant drop) | 20% |

The wide credible interval reflects genuine uncertainty under current macro conditions.
This is a probability distribution, not a directional call.

---

## Key Findings

1. `A_baseline` (estimated nu, tight pooling, σ_h = 0.04) is the best-performing variant
   across both the sensitivity comparison and the walk-forward backtest.
2. Widening the per-regime scale prior (`E_regime_sigma`) did not improve forecasts —
   regime-specific volatility modelling at σ_h = 0.06 reduced accuracy and calibration.
3. Hierarchical accuracy of 43% vs 25% random suggests a modest but present directional
   signal from lagged macro predictors.
4. Probability forecasts remain conservative and near-uniform. The negative BSS indicates
   the model does not yet improve on random probability assignments, even when directional
   accuracy is above chance.

---

## Repository Structure

```
sp_return_probability_bayes/
├── sp_return_prob_bayes.ipynb   # Main notebook
├── requirements.txt
├── README.md
├── outputs/                     # Cached backtest CSVs and trace files
├── figures/                     # Saved forecast and dashboard PNGs
└── resources/                   # Supporting images
```

---

## Workflow

1. **Update data** — FRED series are fetched live at runtime via `pandas_datareader`.
   No manual download required.
2. **Run notebook** — execute all cells top to bottom. Cached results load automatically
   if sampler settings and data date are unchanged.
3. **Review outputs** — backtest metrics print in cells 33–36; the walk-forward dashboard
   saves to `figures/backtest_dashboard.png`; the one-page forecast report saves to
   `figures/forecast_YYYYQ#.png`.

---

## Limitations & Research Notes

- Short-horizon equity returns are difficult to predict with macro variables alone.
  The signal-to-noise ratio at quarterly frequency is low.
- The backtest covers only 21 quarters (2020–2025), spanning a limited number of
  distinct macro environments.
- Regime effects on mean returns are modest and statistically indistinguishable across
  regimes. The yield-curve regime primarily affects estimated volatility, not direction.
- Probability calibration remains conservative: predicted distributions are near-uniform
  and do not yet beat a random 25% allocation on Brier or Log Score metrics.
- This is a probabilistic research tool. It is not a deterministic timing model and
  should not be used as a standalone investment signal.

---

## Data Sources

All data is fetched from **FRED** at runtime via `pandas_datareader`. No API key is required.

| Series | FRED ID | Used for |
|---|---|---|
| S&P 500 Index | `SP500` | Target variable (monthly log returns) |
| 10Y–2Y Treasury spread | `T10Y2Y` | Regime classifier and regression feature |
| ICE BofA HY OAS | `BAMLH0A0HYM2` | Credit spread regression feature |
| CPI (All Urban Consumers) | `CPIAUCSL` | Real yield calculation |

---

## Setup

```bash
conda create -n pymc_env python=3.11
conda activate pymc_env
pip install -r requirements.txt
```

Open `sp_return_prob_bayes.ipynb` and select the `pymc_env` kernel.

---

## Future Improvements

The model currently uses a focused set of macro predictors:

- Yield curve slope and momentum
- High-yield credit spreads
- Equity price momentum
- Real yields

Additional factors that may improve probabilistic calibration include **valuation metrics**
(e.g. CAPE, earnings yield), **financial conditions indices**, and **volatility regime
indicators** (e.g. VIX term structure). Dynamic regime boundaries — replacing the fixed
0% and 1% thresholds with data-driven or soft-assignment methods — represent the
highest-priority structural improvement given current diagnostic findings.
