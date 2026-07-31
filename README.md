# Causal Impact Analysis: Japanese Equity Markets & Political Shocks

An empirical quantitative research project applying **Synthetic Control Methods (SCM)** and **Quasi-Experimental Causal Inference** to measure the structural shock on the Nikkei 225 (`^N225`) following the Japanese Prime Minister election.

## Overview
Evaluating macro shocks on single equity indices is challenging due to concurrent global market movements. Standard regression models risk overfitting pre-treatment noise or picking up unobserved global trends. 

This project constructs a synthetic counterfactual for the Nikkei 225 using a weighted donor pool of major global equity indices (e.g., S&P 500, DAX, Hang Seng, FTSE MIB). To prevent overfitting and enforce convex combination constraints, model parameters are estimated using L1-regularized Lasso regression with non-negativity constraints (`positive=True`).

## Key Methodologies

* **Synthetic Control Framework**: Modeled the counterfactual baseline of the Nikkei 225 using global market indices prior to the intervention date.
* **L1 Regularization & Feature Selection**: Employed `LassoCV` with 5-fold cross-validation and non-negative coefficient constraints (`positive=True`) to enforce a sparse, convex donor pool and eliminate noisy control units.
* **Treatment Effect Estimation**: Quantified the **Average Treatment Effect (ATE)** and **Relative Treatment Effect (RTE)** post-election.
* **Permutation Inference / Placebo Testing**: Executed in-time placebo iterations across all non-treated donor indices to construct an empirical $p$-value distribution and evaluate true statistical significance against global baseline noise.

## Tech Stack
* **Language**: Python 3.x
* **Libraries**: `scikit-learn` (`LassoCV`), `pandas`, `numpy`, `statsmodels`, `yfinance`, `matplotlib`, `seaborn`

## Methodology & Pipeline Breakdown

1. **Data Ingestion**: Programmatically pulls historical daily price data via `yfinance` across global equity markets ($N=11$ major indices).
2. **Pre-Intervention Calibration**: Isolates pre-election trade data to fit scaler and train `LassoCV` model.
3. **Synthetic Asset Generation**: Projects pre-fit weights onto post-intervention feature matrix to generate the counterfactual pathway.
4. **Permutation Testing**: Runs placebo estimations on every non-treated ticker to generate an empirical density function of treatment effects for robust hypothesis testing.
