r"""Performance metrics.

These utilities intentionally avoid adding heavy dependencies.
They are used by backtests/analytics to keep Sharpe computation consistent.

Notes
-----
- Sharpe should be computed on *excess* returns.
- Annualization via :math:`\sqrt{m}` assumes roughly i.i.d. returns.
- PSR/DSR implementations follow the commonly-cited Bailey & López de Prado
  formulas (with skew/kurtosis adjustment) and are meant as a sanity check,
  not a guarantee of future performance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _rf_per_period(risk_free_rate_annual: float, periods_per_year: int) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    # Convert annual rate to per-period rate (compounded).
    return (1.0 + float(risk_free_rate_annual)) ** (1.0 / float(periods_per_year)) - 1.0


def annualized_sharpe(
    returns: Iterable[float] | np.ndarray,
    *,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Compute annualized Sharpe ratio from per-period returns.

    Parameters
    ----------
    returns:
        Per-period arithmetic returns.
    risk_free_rate_annual:
        Annual risk-free rate (e.g. 0.02 for 2%).
    periods_per_year:
        Sampling frequency (252 daily, 52 weekly, 12 monthly, etc.).
    ddof:
        Delta degrees of freedom for standard deviation.
    """
    r = _as_float_array(returns)
    if r.size < 2:
        return 0.0

    rf = _rf_per_period(risk_free_rate_annual, periods_per_year)
    excess = r - rf
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=ddof))
    if not math.isfinite(sigma) or sigma <= 0:
        return 0.0

    return math.sqrt(float(periods_per_year)) * (mu / sigma)


def prob_sharpe_ratio(
    sharpe: float,
    *,
    sharpe_benchmark: float = 0.0,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    Returns probability the true Sharpe exceeds `sharpe_benchmark`.
    Uses skew/kurtosis adjustment; kurtosis is *Pearson* (normal=3).
    """
    if n <= 1:
        return 0.0

    sr = float(sharpe)
    sr_star = float(sharpe_benchmark)
    g3 = float(skew)
    g4 = float(kurtosis)

    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * (sr**2)
    if denom <= 0 or not math.isfinite(denom):
        return 0.0

    z = ((sr - sr_star) * math.sqrt(n - 1.0)) / math.sqrt(denom)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_trials: int = 1,
    sharpe_null: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) as a PSR against an inflated hurdle.

    This is a pragmatic implementation: it converts the multiple-testing effect
    into a higher benchmark Sharpe using a normal-order-statistics approximation.
    """
    if n_trials <= 1:
        return prob_sharpe_ratio(
            sharpe,
            sharpe_benchmark=sharpe_null,
            n=n,
            skew=skew,
            kurtosis=kurtosis,
        )

    # Approx expected max of N standard normals (Gumbel-ish).
    # Reference: order-statistics approximations used in DSR discussions.
    n_trials_f = float(n_trials)
    z = math.sqrt(2.0) * math.erfcinv(2.0 / n_trials_f) if hasattr(math, "erfcinv") else None
    if z is None:
        # Fallback without erfcinv: use a conservative approximation.
        z = math.sqrt(2.0 * math.log(n_trials_f))

    # Inflate the benchmark by selection bias.
    sr_star = float(sharpe_null) + float(z) / math.sqrt(max(n - 1, 1))
    return prob_sharpe_ratio(
        sharpe,
        sharpe_benchmark=sr_star,
        n=n,
        skew=skew,
        kurtosis=kurtosis,
    )


def annualized_volatility(
    returns: Iterable[float] | np.ndarray,
    *,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    r = _as_float_array(returns)
    if r.size < 2:
        return 0.0
    sigma = float(np.std(r, ddof=ddof))
    if not math.isfinite(sigma) or sigma <= 0:
        return 0.0
    return math.sqrt(float(periods_per_year)) * sigma


def cagr(
    returns: Iterable[float] | np.ndarray,
    *,
    periods_per_year: int = 252,
) -> float:
    r = _as_float_array(returns)
    if r.size == 0:
        return 0.0
    total = float(np.prod(1.0 + r) - 1.0)
    years = float(r.size) / float(periods_per_year)
    if years <= 0:
        return 0.0
    base = 1.0 + total
    if base <= 0:
        return -1.0
    return base ** (1.0 / years) - 1.0


def max_drawdown(equity_curve: Iterable[float] | np.ndarray) -> float:
    eq = _as_float_array(equity_curve)
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    dd = (eq / (running_max + 1e-18)) - 1.0
    return float(np.min(dd))


def turnover(
    positions: Iterable[float] | np.ndarray,
) -> float:
    pos = _as_float_array(positions)
    if pos.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(pos))))
