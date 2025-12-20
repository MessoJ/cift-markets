import numpy as np


def test_backtest_positions_shifts_positions_by_default():
    from cift.backtest.engine import backtest_positions

    # If we didn't shift, we'd earn +1% immediately.
    returns = np.array([0.01, 0.0, 0.0])
    signal = np.array([1.0, 1.0, 1.0])

    res = backtest_positions(returns, signal, commission_bps=0.0, slippage_bps=0.0, shift_positions=True)
    # First period position must be 0 to avoid look-ahead.
    assert res.positions[0] == 0.0
    assert res.strategy_returns[0] == 0.0


def test_backtest_positions_costs_reduce_performance():
    from cift.backtest.engine import backtest_positions

    returns = np.array([0.0, 0.01, 0.01, 0.0])
    # Flip from 0 to 1 at t=1 (effective after shift)
    signal = np.array([0.0, 1.0, 1.0, 1.0])

    res_free = backtest_positions(returns, signal, commission_bps=0.0, slippage_bps=0.0)
    res_cost = backtest_positions(returns, signal, commission_bps=10.0, slippage_bps=10.0)

    assert res_cost.metrics["final_portfolio_value"] < res_free.metrics["final_portfolio_value"]
