# Quarterly S&P 500 Return Probabilities: A Hierarchical Bayesian Approach

![DALLE Markov Sampling](./resources/DALLE_Markov_Sampling.webp)

## Summary

This Jupyter notebook leverages **PyMC 5.10** and **ArviZ** to forecast the probabilities of quarterly S&P 500 returns falling within specific ranges. It implements two complementary Bayesian models:

1. **Flat Bayesian Linear Regression** — a single set of parameters with Student's t-distribution errors, serving as the baseline.
2. **Hierarchical Bayesian Model** — a three-level partial-pooling model that classifies each quarter into a VIX-based market regime and learns regime-specific coefficients, while sharing information across regimes via global hyperpriors.

Developed with the assistance of **Chat GPT**, **GitHub Copilot**, and **Claude (Anthropic)**, the notebook marries Bayesian statistical methods with advanced visualization techniques to provide insightful forecasts on financial market trends.

## Key Components

### Data Preparation and Analysis

- Loads and processes monthly data on the S&P 500, VIX, and interest rates, converting it to a quarterly format.
- Calculates logarithmic returns for the S&P 500, along with changes in the VIX and interest rates.
- Introduces lagged features to enhance forecast accuracy by incorporating historical data.

### Feature Engineering and Model Training

- Selects influential features for modeling S&P 500 returns.
- Splits the data into training and testing sets, applying RobustScaler to mitigate outlier impacts.
- Utilizes robust scaling to ensure model accuracy and resilience.

### Prior Predictive Checks

- Performs prior predictive checks to assess the influence of priors on the resulting model and ensure they are not overly restrictive or too broad.
- Ensures that the specified priors allow for a reasonable range of outcomes, particularly for the heavy-tailed nature of financial returns.

### Flat Bayesian Model and Iterative Prediction

- Employs a Bayesian Linear Regression model with Student's t-distribution errors, specifying priors and utilizing MCMC sampling for posterior estimation.
- Refines predictions each quarter by updating the model with new data, improving forecast precision over time.
- Calculates predictive probabilities for specified ranges of S&P 500 returns, offering actionable insights.

### Hierarchical Bayesian Model

- Classifies each quarter into one of three **VIX-based market regimes**:
  - **Regime 0** — Low Volatility (VIX < 15): calm bull-market environment
  - **Regime 1** — Normal Volatility (15 ≤ VIX < 25): typical market conditions
  - **Regime 2** — High Volatility (VIX ≥ 25): stress / crisis environment
- Implements a **three-level hierarchy** with partial pooling:
  - *Level 2*: Global hyperpriors (`mu_*`, `sigma_*`) learned from the data control how much information is shared across regimes.
  - *Level 1*: Regime-specific parameters (`alpha_r`, `beta_vix_r`, `beta_rates_r`, `beta_sp_r`) drawn from the hyperpriors — each regime gets its own coefficients while still borrowing strength from the others.
  - *Level 0*: StudentT observation likelihood using the regime-appropriate mean.
- Uses **non-centered parameterization** to prevent NUTS sampling funnels and reduce divergences.
- Visualizes regime-specific coefficient posteriors (bar plots, forest plots) and hyperprior posteriors to show the learned degree of pooling.
- Runs the same expanding-window iterative prediction loop as the flat model, with regime indices passed at each step for comparison.

### Visualization and Performance Evaluation

- Visualizes MCMC trace plots and posterior predictive distributions for model transparency.
- Compares model predictions against actual returns to assess forecast accuracy.

### Results Compilation and Analysis

- Compiles predicted probabilities and actual returns for comprehensive performance analysis.
- Implements a function to evaluate the model's predictive effectiveness based on the highest probability ranges.

## Modeling Trade-offs and Future Plans

Both models are tailored to illustrate Bayesian approaches to financial forecasting and have been crafted with careful consideration of the historical outliers and heavy-tailed nature of financial returns observed in the period 2000–2024. They are more for demonstration than reliance on predicting exact return ranges.

The hierarchical model improves on the flat baseline by allowing the data to express regime-specific dynamics, but several further refinements are possible:

- **Regime-specific error variance** (`sigma_h` per regime) to capture volatility clustering within each environment.
- **Regime-specific degrees of freedom** (`nu_h` per regime) for heavier tails during crisis periods.
- **Time-varying coefficients** — a Gaussian random walk prior on the betas to capture structural breaks over time (state-space model).
- **Hidden Markov / mixture model** — treat regime membership as a latent variable and jointly infer regime transitions and return distributions, rather than using hard VIX thresholds.
- **Expanding the feature set** — yield curve slope, credit spreads, or earnings yield as additional predictors.
- **Continual model evaluation** against new quarterly data as it becomes available.

## Sources

- Martin, Osvaldo. *Bayesian Analysis with Python: A practical guide to probabilistic modeling*. 3rd Edition, Kindle Edition.
- Kanungo, Deepak K. *Probabilistic Machine Learning for Finance and Investing*. 1st Edition, Kindle Edition.
