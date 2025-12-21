from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from cift.metrics.performance import (
    annualized_sharpe,
    annualized_volatility,
    cagr,
    max_drawdown,
    turnover,
)


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray
    strategy_returns: np.ndarray
    positions: np.ndarray
    metrics: dict[str, Any]


def backtest_positions(
    returns: Iterable[float] | np.ndarray,
    positions: Iterable[float] | np.ndarray,
    *,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
    initial_capital: float = 100000.0,
    shift_positions: bool = True,
) -> BacktestResult:
    """Backtest a strategy defined by per-period returns and positions.

    Parameters
    ----------
    returns:
        Per-period arithmetic returns of the traded instrument.
    positions:
        Target position per period (e.g., -1..+1). If `shift_positions=True`,
        the position is shifted by 1 step to avoid look-ahead bias.
    commission_bps, slippage_bps:
        Round-trip costs applied on position changes, in basis points.
        Cost model is: `abs(delta_position) * (commission+slippage) / 10000`.
    """
    r = np.asarray(returns, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64)
    if r.ndim != 1 or p.ndim != 1:
        raise ValueError("returns and positions must be 1D arrays")
    if r.size != p.size:
        raise ValueError("returns and positions must have same length")
    if r.size == 0:
        return BacktestResult(
            equity=np.array([], dtype=np.float64),
            strategy_returns=np.array([], dtype=np.float64),
            positions=np.array([], dtype=np.float64),
            metrics={
                "total_return": 0.0,
                "cagr": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "final_portfolio_value": float(initial_capital),
            },
        )

    if shift_positions:
        p_eff = np.roll(p, 1)
        p_eff[0] = 0.0
    else:
        p_eff = p

    gross = r * p_eff

    # Transaction costs on position changes
    dp = np.diff(p_eff, prepend=p_eff[0])
    cost_bps = float(commission_bps) + float(slippage_bps)
    costs = np.abs(dp) * (cost_bps / 10000.0)
    net = gross - costs

    equity_rel = np.cumprod(1.0 + net)
    equity = equity_rel * float(initial_capital)

    metrics: dict[str, Any] = {}
    metrics["total_return"] = float(equity[-1] / float(initial_capital) - 1.0)
    metrics["cagr"] = float(cagr(net, periods_per_year=periods_per_year))
    metrics["annual_volatility"] = float(annualized_volatility(net, periods_per_year=periods_per_year))
    metrics["sharpe_ratio"] = float(
        annualized_sharpe(
            net,
            risk_free_rate_annual=risk_free_rate_annual,
            periods_per_year=periods_per_year,
        )
    )
    metrics["max_drawdown"] = float(max_drawdown(equity_rel))
    metrics["turnover"] = float(turnover(p_eff))
    metrics["final_portfolio_value"] = float(equity[-1])
    metrics["commission_bps"] = float(commission_bps)
    metrics["slippage_bps"] = float(slippage_bps)

    return BacktestResult(
        equity=equity,
        strategy_returns=net,
        positions=p_eff,
        metrics=metrics,
    )
