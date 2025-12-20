import numpy as np
import polars as pl
import pytest


def test_vectorized_backtest_sharpe_annualization_is_sqrt_252():
    # Constant position, so strategy_returns == returns (minus tiny commission on first change).
    # We keep commission at 0 to isolate Sharpe math.
    df = pl.DataFrame(
        {
            "returns": [0.01, -0.005, 0.002, 0.0, 0.004],
            "signal": [1, 1, 1, 1, 1],
        }
    )

    from cift.core.data_processing import run_vectorized_backtest

    _, metrics = run_vectorized_backtest(
        df,
        signal_column="signal",
        initial_capital=100000.0,
        commission_bps=0.0,
        risk_free_rate_annual=0.0,
        periods_per_year=252,
    )

    # Expected: sqrt(252) * mean/std
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.004], dtype=np.float64)
    expected = (np.mean(r) / np.std(r, ddof=1)) * np.sqrt(252)

    assert metrics["sharpe_ratio"] == pytest.approx(float(expected), rel=1e-12, abs=1e-12)
