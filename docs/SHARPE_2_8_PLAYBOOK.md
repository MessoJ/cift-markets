# Sharpe 2.8 Playbook (Engineering + Evaluation)

This document is intentionally blunt:
**a 2.8 Sharpe target is not credible unless the evaluation loop is leakage-safe, friction-aware, and selection-bias-aware.**

## 1) Sharpe: compute it correctly
- Use per-period **excess** returns: $r_t = R_t - R_{f,t}$.
- Sharpe: $SR = \frac{\bar r}{\sigma(r)}$.
- Annualize (common assumption):
  $$SR_{ann} \approx \sqrt{m}\,SR$$
  where $m$ is periods/year (252 daily, 52 weekly, 12 monthly).

**Anti-pattern:** `mean * 252 / std` (this inflates Sharpe by ~$\sqrt{252}$).

### Autocorrelation reality (don’t over-annualize)
Annualizing via $\sqrt{m}$ is an i.i.d. approximation. In real trading PnL, returns are often autocorrelated (signal smoothing, overlap, execution effects), which can make naive Sharpe look better than it is.

Minimum rule: keep your sampling frequency and label horizon aligned (no overlapping labels without correction), and prefer a walk-forward OOS stream over single in-sample Sharpe.

## 2) Leakage control: purged + embargo CV
When labels use future information (e.g., forward return over horizon $H$), each sample is an *event interval*:
- $[t^{start}_i, t^{end}_i]$ where $t^{end}_i = t^{start}_i + H$ (or the last timestamp used for the label).

For a test fold spanning $[T_0, T_1]$:
- **Purge** training samples whose intervals overlap the test interval.
- **Embargo** training samples with start times in $(T_1, T_1 + \Delta]$.

### Nested selection (don’t tune on the test folds)
If you do hyperparameter search, feature selection, or “try a bunch of models and pick the best”, you need an **inner** selection loop. Otherwise the outer-fold Sharpe is still inflated.

#### Strict time holdout (required if you tune)
If you tune anything (hyperparameters, thresholds, features, data windows), keep a final, untouched time holdout and report it separately from CV.

- **Outer CV**: estimate generalization on the *training window* via purged+embargo CV.
- **Inner CV**: select hyperparameters/thresholds using only the training window.
- **Holdout**: fit once on the full training window (excluding label-overlap), then evaluate on the holdout.

## 3) Backtests must include frictions
Minimum:
- commissions/fees
- slippage/spread
- latency (at least “trade on next bar / next tick” semantics)

If turnover is high: add a basic market impact model or treat results as fantasy.

## 4) Selection bias: PSR/DSR
If you tried 50 variants and kept the best Sharpe, the “best” Sharpe is usually luck.

### Probabilistic Sharpe Ratio (PSR)
Returns the probability that the true Sharpe exceeds a benchmark $SR^*$.

Let $SR$ be observed Sharpe, $n$ samples, skew $\gamma_3$, kurtosis $\gamma_4$ (Pearson, normal=3):

$$
PSR=\Phi\!\left(\frac{(SR-SR^*)\sqrt{n-1}}{\sqrt{1-\gamma_3 SR + \frac{\gamma_4-1}{4}SR^2}}\right)
$$

### Deflated Sharpe Ratio (DSR)
DSR is PSR against an *inflated* hurdle that accounts for many tries (models/params/features).

### Probability of Backtest Overfitting (PBO)
If you search enough knobs, “the best backtest” is often a fluke. PSR/DSR helps, but if you’re doing large sweeps you should also track **how many trials** you effectively ran (variants × symbols × time windows × feature sets × seeds) and treat results as exploratory until confirmed on fresh, untouched data.

## 5) Implementation status in this repo
- Correct Sharpe + metrics: `cift/metrics/performance.py`
- Vectorized backtest uses shared Sharpe: `cift/core/data_processing.py`
- Reproducible PnL engine: `cift/backtest/engine.py`
- Purged+embargo splitter: `cift/ml/evaluation/splits.py`
- Walk-forward evaluator (OOS stream + PSR): `cift/ml/evaluation/walkforward.py`
- CLI entrypoint: `cift.cli:walkforward`

### Advanced Features (De Prado / mlfinlab style)
- **Fractional Differentiation**: `cift/ml/features.py:frac_diff_ffd` - preserves memory while achieving stationarity
- **Yang-Zhang Volatility**: `cift/ml/features.py:volatility_yang_zhang` - minimum-variance OHLC estimator
- **Amihud Illiquidity**: `cift/ml/features.py:amihud_illiquidity` - price impact measure
- **Triple Barrier Labeling**: `cift/ml/labeling.py:get_triple_barrier_labels` - PT/SL/Time barriers

### Models Available in Walk-Forward
- `baseline`: Deterministic signal from rolling mean returns
- `logreg`: Per-fold LogisticRegression with inner-loop tuning
- `xgboost`: Per-fold XGBoost classifier for complex pattern capture

### Run it
Your dataset must contain at least `timestamp` and `close` (CSV or Parquet).

Examples:
- Baseline (deterministic): `cift walkforward --data-path data.csv`
- Trainable baseline (logistic regression): `cift walkforward --data-path data.csv --model logreg`

Strict holdout + nested tuning (logreg):
`cift walkforward --data-path data.csv --model logreg --tune --holdout-bars 500 --c-grid "0.1,1.0,10.0" --threshold-grid "0.0,0.02,0.05"`

XGBoost with FracDiff + Volatility features:
`cift walkforward --data-path data.csv --model xgboost --use-fracdiff --fracdiff-d 0.4 --use-vol-features --vol-window 20 --holdout-bars 500`

Read:
- `metrics.*` are CV out-of-sample aggregate metrics.
- `holdout_*` are strict final holdout metrics.
- `selected_C` / `selected_threshold` show chosen parameters (when `--tune`).

## 6) Brutal reality on 2.8
Sustained 2.8 Sharpe is rare for retail equities strategies, especially at high turnover with public feeds and normal broker routing.
This repo’s goal must be to build a truthful loop so we can iterate honestly; it cannot “guarantee” 2.8.

## 7) What “credible progress” looks like
- Report OOS Sharpe from the walk-forward stream (not just best-fold).
- Report PSR/DSR alongside Sharpe.
- Re-run on a strictly held-out time range after decisions are frozen.
- Treat any jump in Sharpe as suspicious until it survives costs + embargo + revalidation.
