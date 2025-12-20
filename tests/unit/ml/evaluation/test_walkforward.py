import numpy as np
import polars as pl

from cift.ml.evaluation.walkforward import run_walkforward


def _make_synthetic_prices(n: int = 400, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    # Autocorrelated returns: sign tends to persist.
    r = np.zeros(n, dtype=np.float64)
    r[0] = 0.0
    for i in range(1, n):
        prev = r[i - 1]
        drift = 0.002 if prev >= 0 else -0.002
        r[i] = drift + rng.normal(0.0, 0.001)

    close = 100.0 * np.cumprod(1.0 + r)
    ts = np.arange(n)

    return pl.DataFrame({"timestamp": ts, "close": close})


def test_walkforward_logreg_runs(tmp_path):
    df = _make_synthetic_prices()
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=1,
        n_splits=3,
        embargo_bars=1,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        threshold=0.05,
        model="logreg",
        n_lags=5,
        n_trials=10,
    )

    assert report.metrics["model"] == "logreg"
    assert "psr_sharpe_gt_0" in report.metrics
    assert "dsr_sharpe_gt_0" in report.metrics
    assert report.metrics["n_obs"] > 0


def test_walkforward_logreg_tune_and_holdout_runs(tmp_path):
    df = _make_synthetic_prices(n=650, seed=11)
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=1,
        n_splits=3,
        embargo_bars=1,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        model="logreg",
        n_lags=5,
        holdout_bars=120,
        tune=True,
        tune_splits=3,
        c_grid="0.1,1.0",
        threshold_grid="0.0,0.05",
    )

    assert report.metrics["model"] == "logreg"
    assert report.metrics["holdout_bars"] == 120
    assert report.metrics["tune"] is True
    assert "selected_C" in report.metrics
    assert "selected_threshold" in report.metrics
    assert "holdout_sharpe_ratio" in report.metrics


def test_walkforward_xgboost_with_fracdiff(tmp_path):
    """Test XGBoost model with FracDiff and volatility features."""
    df = _make_synthetic_prices(n=500, seed=42)
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=1,
        n_splits=3,
        embargo_bars=1,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        model="xgboost",
        n_lags=5,
        use_fracdiff=True,
        fracdiff_d=0.4,
        use_vol_features=True,
        vol_window=10,
    )

    assert report.metrics["model"] == "xgboost"
    assert report.metrics["use_fracdiff"] is True
    assert report.metrics["use_vol_features"] is True
    assert "psr_sharpe_gt_0" in report.metrics
    assert report.metrics["n_obs"] > 0


def test_walkforward_triple_barrier_logreg(tmp_path):
    """Test Triple Barrier labeling with logreg model."""
    df = _make_synthetic_prices(n=600, seed=99)
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=10,
        n_splits=3,
        embargo_bars=2,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        model="logreg",
        n_lags=5,
        use_triple_barrier=True,
        tb_pt=2.0,
        tb_sl=2.0,
        tb_min_ret=0.001,
    )

    assert report.metrics["model"] == "logreg"
    assert report.metrics.get("use_triple_barrier") is True
    assert "psr_sharpe_gt_0" in report.metrics
    assert report.metrics["n_obs"] > 0


def test_walkforward_meta_labeling_xgboost(tmp_path):
    """Test meta-labeling with XGBoost primary and meta models."""
    df = _make_synthetic_prices(n=600, seed=42)
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=5,
        n_splits=3,
        embargo_bars=2,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        model="xgboost",
        n_lags=5,
        use_meta_labeling=True,
        meta_model="xgboost",
        meta_use_sizing=True,
    )

    assert report.metrics["model"] == "xgboost"
    assert report.metrics.get("use_meta_labeling") is True
    assert report.metrics.get("meta_model") == "xgboost"
    assert "psr_sharpe_gt_0" in report.metrics
    assert report.metrics["n_obs"] > 0


def test_walkforward_sample_weights(tmp_path):
    """Test sample weights based on average uniqueness (De Prado)."""
    df = _make_synthetic_prices(n=600, seed=77)
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=5,
        n_splits=3,
        embargo_bars=2,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        model="logreg",
        n_lags=5,
        use_sample_weights=True,
    )

    assert report.metrics["model"] == "logreg"
    assert report.metrics.get("use_sample_weights") is True
    assert "psr_sharpe_gt_0" in report.metrics
    assert report.metrics["n_obs"] > 0


def test_walkforward_ta_features(tmp_path):
    """Test standard technical analysis features (RSI, MACD, etc.)."""
    df = _make_synthetic_prices(n=600, seed=123)
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=5,
        n_splits=3,
        embargo_bars=2,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        model="xgboost",
        n_lags=5,
        use_ta_features=True,
    )

    assert report.metrics["model"] == "xgboost"
    assert report.metrics.get("use_ta_features") is True
    assert "psr_sharpe_gt_0" in report.metrics
    assert report.metrics["n_obs"] > 0


def test_walkforward_full_de_prado_stack(tmp_path):
    """
    Full De Prado ML pipeline integration test:
    - FracDiff features for stationarity
    - Volatility features
    - Triple Barrier labeling
    - Meta-labeling for bet sizing
    - Sample weights based on uniqueness
    - XGBoost model
    - Tuning with inner purged CV
    - Strict holdout
    """
    df = _make_synthetic_prices(n=1000, seed=2024)
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=10,
        n_splits=3,
        embargo_bars=5,
        commission_bps=2.0,
        slippage_bps=2.0,
        periods_per_year=252,
        model="xgboost",
        n_lags=10,
        holdout_bars=100,
        use_fracdiff=True,
        fracdiff_d=0.4,
        use_vol_features=True,
        vol_window=20,
        use_triple_barrier=True,
        tb_pt=2.0,
        tb_sl=2.0,
        tb_min_ret=0.005,
        use_meta_labeling=True,
        meta_model="xgboost",
        meta_use_sizing=True,
        use_sample_weights=True,
    )

    # Verify all features are recorded
    assert report.metrics["model"] == "xgboost"
    assert report.metrics.get("use_fracdiff") is True
    assert report.metrics.get("use_vol_features") is True
    assert report.metrics.get("use_triple_barrier") is True
    assert report.metrics.get("use_meta_labeling") is True
    assert report.metrics.get("use_sample_weights") is True
    assert report.metrics.get("holdout_bars") == 100
    
    # Verify we got meaningful results
    assert "psr_sharpe_gt_0" in report.metrics
    assert "dsr_sharpe_gt_0" in report.metrics
    assert report.metrics["n_obs"] > 0
    
    # Holdout metrics should exist
    assert "holdout_sharpe_ratio" in report.metrics or "holdout_n_obs" in report.metrics


def test_walkforward_full_stack_brutal(tmp_path):
    """Test the full 'brutal' stack: Micro features, Vol Target, Tuning."""
    df = _make_synthetic_prices(n=600, seed=999)
    # Add High/Low/Volume for micro features
    # Synthetic High/Low
    df = df.with_columns([
        (pl.col("close") * 1.01).alias("high"),
        (pl.col("close") * 0.99).alias("low"),
        pl.lit(1000).alias("volume"),
    ])
    p = tmp_path / "data.csv"
    df.write_csv(p)

    report = run_walkforward(
        data_path=str(p),
        timestamp_col="timestamp",
        close_col="close",
        horizon_bars=5,
        n_splits=3,
        embargo_bars=2,
        commission_bps=1.0,
        slippage_bps=1.0,
        periods_per_year=252,
        model="xgboost",
        n_lags=5,
        use_ta_features=True,
        use_micro_features=True,
        vol_target=0.15,
        tune_model=True,
    )

    assert report.metrics["model"] == "xgboost"
    assert report.metrics.get("use_micro_features") is True
    assert report.metrics.get("vol_target") == 0.15
    assert report.metrics.get("tune_model") is True
    assert "psr_sharpe_gt_0" in report.metrics
