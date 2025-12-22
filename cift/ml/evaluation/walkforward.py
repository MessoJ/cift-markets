from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cift.backtest.engine import backtest_positions
from cift.metrics.performance import deflated_sharpe_ratio, prob_sharpe_ratio
from cift.ml.evaluation.splits import PurgedKFold, build_forward_return_events
from cift.ml.features import (
    frac_diff_ffd,
    get_microstructure_features,
    get_technical_features,
)
from cift.ml.labeling import (
    apply_meta_model_sizing,
    compute_sample_weights,
    get_meta_labels,
    get_triple_barrier_labels,
)


@dataclass(frozen=True)
class WalkForwardReport:
    metrics: dict[str, Any]
    fold_metrics: list[dict[str, Any]]


def _load_df(path: str) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        return pl.read_parquet(str(p))
    return pl.read_csv(str(p))


def _sample_skew(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    mu = float(np.mean(x))
    m2 = float(np.mean((x - mu) ** 2))
    if not np.isfinite(m2) or m2 <= 0:
        return 0.0
    m3 = float(np.mean((x - mu) ** 3))
    if not np.isfinite(m3):
        return 0.0
    return float(m3 / (m2**1.5))


def _sample_kurtosis_pearson(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 3.0
    mu = float(np.mean(x))
    m2 = float(np.mean((x - mu) ** 2))
    if not np.isfinite(m2) or m2 <= 0:
        return 3.0
    m4 = float(np.mean((x - mu) ** 4))
    if not np.isfinite(m4):
        return 3.0
    k = float(m4 / (m2**2))
    return k if np.isfinite(k) and k > 0 else 3.0


def _build_features(
    df: pl.DataFrame,
    *,
    n_lags: int,
    use_fracdiff: bool = False,
    fracdiff_d: float = 0.4,
    use_vol_features: bool = False,
    vol_window: int = 20,
    use_ta_features: bool = False,
    use_micro_features: bool = False,
) -> tuple[pl.DataFrame, list[str]]:
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")

    feature_cols: list[str] = []
    exprs: list[pl.Expr] = []

    # Lagged returns (lag_0 includes the current bar return; position shifting handles trade timing).
    for lag in range(n_lags):
        name = f"ret_lag_{lag}"
        feature_cols.append(name)
        exprs.append(pl.col("returns").shift(lag).alias(name))

    w = max(2, n_lags)
    feature_cols.append(f"ret_mean_{w}")
    exprs.append(pl.col("returns").rolling_mean(window_size=w).shift(1).alias(f"ret_mean_{w}"))
    feature_cols.append(f"ret_std_{w}")
    exprs.append(pl.col("returns").rolling_std(window_size=w).shift(1).alias(f"ret_std_{w}"))

    df = df.with_columns(exprs)

    # Optional: Fractional Differentiation on close prices
    if use_fracdiff and "close" in df.columns:
        close_arr = df["close"].to_numpy()
        fd = frac_diff_ffd(close_arr, d=float(fracdiff_d))
        df = df.with_columns([pl.Series("fracdiff_close", fd)])
        feature_cols.append("fracdiff_close")

    # Optional: Volatility features (rolling std of returns)
    if use_vol_features:
        vol_col_name = f"vol_{vol_window}"
        df = df.with_columns(
            [pl.col("returns").rolling_std(window_size=vol_window).shift(1).alias(vol_col_name)]
        )
        feature_cols.append(vol_col_name)

        # Volatility momentum (change in volatility)
        vol_mom_name = f"vol_mom_{vol_window}"
        df = df.with_columns(
            [
                (pl.col(vol_col_name) / pl.col(vol_col_name).shift(vol_window) - 1.0).alias(
                    vol_mom_name
                )
            ]
        )
        feature_cols.append(vol_mom_name)

    # Optional: Technical Analysis features (RSI, MACD, BB, ATR, MFI)
    if use_ta_features:
        # Ensure we have required columns
        req_cols = ["close", "high", "low", "volume"]
        if all(c in df.columns for c in req_cols):
            df = get_technical_features(df)
            # Add the new columns to feature_cols
            ta_cols = [
                "rsi_14",
                "macd_line",
                "macd_signal",
                "macd_hist",
                "bb_width",
                "bb_pct",
                "atr_14",
                "mfi_14",
            ]
            feature_cols.extend(ta_cols)

            # Shift TA features by 1 to avoid lookahead bias (calculated on close)
            # Actually, get_technical_features calculates on current bar.
            # If we trade at Open of next bar based on Close of this bar, we don't need to shift IF we are careful.
            # But standard practice in this pipeline is to shift features that use 'close' to ensure no leakage.
            # Let's shift them all by 1.
            shift_exprs = [pl.col(c).shift(1).alias(c) for c in ta_cols]
            df = df.with_columns(shift_exprs)
        else:
            # If missing columns, skip but warn (or just skip silently for now)
            pass

    # Optional: Microstructure features
    if use_micro_features:
        req_cols = ["close", "high", "low"]
        if all(c in df.columns for c in req_cols):
            df = get_microstructure_features(df)
            micro_cols = ["hl_spread", "ker_14", "vol_bp_14"]
            feature_cols.extend(micro_cols)
            # Shift by 1
            shift_exprs = [pl.col(c).shift(1).alias(c) for c in micro_cols]
            df = df.with_columns(shift_exprs)

    df = df.drop_nulls(feature_cols)
    return df, feature_cols


def _parse_float_grid(values: str) -> list[float]:
    parts = [p.strip() for p in str(values).split(",") if p.strip()]
    if not parts:
        return []
    out: list[float] = []
    for p in parts:
        out.append(float(p))
    return out


def _fit_predict_logreg(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    C: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for model=logreg") from e

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=300, solver="lbfgs", C=float(C))
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    return clf.predict_proba(X_test_scaled)[:, 1]


def _fit_predict_xgboost(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    max_depth: int = 3,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    sample_weight: np.ndarray | None = None,
    tune: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    try:
        import xgboost as xgb
    except Exception as e:  # pragma: no cover
        raise RuntimeError("xgboost is required for model=xgboost") from e

    early_stopping_rounds = kwargs.pop("early_stopping_rounds", None)

    # Hyperparameter Tuning (Random Search)
    if tune:
        # Simple 80/20 split for validation
        split_idx = int(len(X_train) * 0.8)
        if split_idx > 10:  # Only tune if enough data
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
            w_tr = sample_weight[:split_idx] if sample_weight is not None else None
            w_val = sample_weight[split_idx:] if sample_weight is not None else None

            param_grid = {
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "n_estimators": [100, 300, 500],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.2],
            }

            best_score = float("inf")
            best_params = {
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
            }

            import random

            from sklearn.metrics import log_loss

            # Try 10 random combinations
            for _ in range(10):
                params = {k: random.choice(v) for k, v in param_grid.items()}

                clf = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                    n_jobs=1,  # Avoid thread contention in parallel folds
                    early_stopping_rounds=10,
                    **params,
                )

                try:
                    clf.fit(
                        X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False
                    )

                    val_proba = clf.predict_proba(X_val)[:, 1]
                    score = log_loss(y_val, val_proba, sample_weight=w_val)

                    if score < best_score:
                        best_score = score
                        best_params = params
                        if hasattr(clf, "best_iteration"):
                            best_params["n_estimators"] = clf.best_iteration + 10
                except Exception:
                    continue

            # Apply best params
            max_depth = best_params["max_depth"]
            n_estimators = best_params["n_estimators"]
            learning_rate = best_params["learning_rate"]
            kwargs.update(
                {
                    k: v
                    for k, v in best_params.items()
                    if k not in ["max_depth", "n_estimators", "learning_rate"]
                }
            )

            # If tuning happened, we should probably use early stopping for the final fit too
            if early_stopping_rounds is None:
                early_stopping_rounds = 10

    eval_set = None

    if early_stopping_rounds is not None:
        # Simple 90/10 split for validation (or reuse split if we want, but let's keep it consistent)
        split_idx = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        w_tr = sample_weight[:split_idx] if sample_weight is not None else None
        w_val = sample_weight[split_idx:] if sample_weight is not None else None

        eval_set = [(X_val, y_val)]

        clf = xgb.XGBClassifier(
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=eval_set, verbose=False)
    else:
        clf = xgb.XGBClassifier(
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
            **kwargs,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)

    return clf.predict_proba(X_test)[:, 1]


def _signal_from_proba(proba: np.ndarray, *, threshold: float) -> np.ndarray:
    upper = 0.5 + float(threshold)
    lower = 0.5 - float(threshold)
    return np.where(proba > upper, 1.0, np.where(proba < lower, -1.0, 0.0)).astype(np.float64)


def _fit_predict_meta_model(
    *,
    X_train: np.ndarray,
    meta_labels_train: np.ndarray,
    X_test: np.ndarray,
    model: str = "xgboost",
) -> np.ndarray:
    """
    Fit a meta-model to predict whether primary model's prediction is correct.

    Returns probability that primary model is correct (used for bet sizing).
    """
    # Ensure we have both classes
    unique_labels = np.unique(meta_labels_train)
    if len(unique_labels) < 2:
        # Degenerate case: return 0.5 (neutral)
        return np.full(len(X_test), 0.5, dtype=np.float64)

    if model == "xgboost":
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("xgboost is required for meta-model") from e

        clf = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=50,  # Smaller for meta-model to avoid overfitting
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        clf.fit(X_train, meta_labels_train)
        return clf.predict_proba(X_test)[:, 1]
    else:
        # Default to logistic regression
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as e:
            raise RuntimeError("scikit-learn is required for meta-model") from e

        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=300, solver="lbfgs", C=1.0),
        )
        clf.fit(X_train, meta_labels_train)
        return clf.predict_proba(X_test)[:, 1]


def _oos_metrics_from_stream(
    *,
    strategy_returns: np.ndarray,
    positions: np.ndarray,
    periods_per_year: int,
    n_trials: int,
) -> dict[str, Any]:
    bt_all = backtest_positions(
        returns=np.asarray(strategy_returns, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
        commission_bps=0.0,
        slippage_bps=0.0,
        periods_per_year=periods_per_year,
        initial_capital=100000.0,
        shift_positions=False,
    )
    sharpe = float(bt_all.metrics.get("sharpe_ratio", 0.0))

    oos = np.asarray(bt_all.strategy_returns, dtype=np.float64)
    skew = _sample_skew(oos)
    kurt = _sample_kurtosis_pearson(oos)

    psr = prob_sharpe_ratio(
        sharpe,
        sharpe_benchmark=0.0,
        n=int(len(oos)),
        skew=skew,
        kurtosis=kurt,
    )
    dsr = deflated_sharpe_ratio(
        sharpe,
        n=int(len(oos)),
        skew=skew,
        kurtosis=kurt,
        n_trials=int(max(1, n_trials)),
        sharpe_null=0.0,
    )

    m: dict[str, Any] = dict(bt_all.metrics)
    m["psr_sharpe_gt_0"] = float(psr)
    m["dsr_sharpe_gt_0"] = float(dsr)
    m["n_obs"] = int(len(oos))
    m["skew"] = float(skew)
    m["kurtosis_pearson"] = float(kurt)
    return m


def run_walkforward(
    *,
    data_path: str,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    horizon_bars: int = 1,
    n_splits: int = 5,
    embargo_bars: int = 0,
    commission_bps: float = 1.0,
    slippage_bps: float = 1.0,
    periods_per_year: int = 252,
    threshold: float = 0.0,
    model: str = "baseline",
    n_lags: int = 10,
    n_trials: int = 1,
    holdout_bars: int = 0,
    tune: bool = False,
    tune_splits: int = 3,
    c_grid: str = "0.1,1.0,10.0",
    threshold_grid: str = "0.0,0.02,0.05",
    use_fracdiff: bool = False,
    fracdiff_d: float = 0.4,
    use_vol_features: bool = False,
    vol_window: int = 20,
    use_triple_barrier: bool = False,
    tb_pt: float = 2.0,
    tb_sl: float = 2.0,
    tb_min_ret: float = 0.0,
    use_meta_labeling: bool = False,
    meta_model: str = "xgboost",
    meta_use_sizing: bool = True,
    meta_threshold: float = 0.5,
    use_sample_weights: bool = False,
    use_ta_features: bool = False,
    use_micro_features: bool = False,
    vol_target: float = 0.0,
    tune_model: bool = False,
) -> WalkForwardReport:
    """Leakage-safe walk-forward evaluation.

    Modes
    -----
    baseline:
        Deterministic signal from rolling mean returns.
    logreg:
        Per-fold LogisticRegression on lagged returns features.
    xgboost:
        Per-fold XGBoost classifier for more complex pattern capture.

    Notes
    -----
    - Split logic is *purged + embargo* to avoid label overlap leakage.
    - Backtest uses shift_positions=True so today's signal trades next bar.
        - `threshold` is interpreted as:
        - baseline: absolute return-mean threshold
        - logreg/xgboost: probability dead-zone width around 0.5
        - If `holdout_bars>0`, the last `holdout_bars` are reserved as a strict time holdout.
            Any training rows whose *labels* would touch the holdout window are excluded.
        - If `tune=True` (logreg/xgboost), we run an inner purged+embargo CV grid search over
            hyperparameters and threshold using only the pre-holdout window.
        - `use_fracdiff`: Apply fractional differentiation (De Prado) to close prices.
        - `use_vol_features`: Add volatility and volatility momentum features.
        - `use_triple_barrier`: Use Triple Barrier Method (PT/SL/Time) instead of fixed-horizon labels.
        - `tb_pt`, `tb_sl`: Profit-take and stop-loss multipliers (barrier = mult * vol).
    """
    df = _load_df(data_path)
    for col in (timestamp_col, close_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.sort(timestamp_col)
    df = df.with_columns([pl.col(close_col).pct_change().alias("returns")])
    df = df.drop_nulls(["returns"])

    # Create volatility column for triple barrier (rolling std of returns)
    tb_vol_window = max(20, horizon_bars * 2)
    df = df.with_columns(
        [
            pl.col("returns").rolling_std(window_size=tb_vol_window).alias("volatility"),
        ]
    )
    df = df.drop_nulls(["volatility"])

    # Labeling: Triple Barrier or Fixed Horizon
    if use_triple_barrier:
        df = get_triple_barrier_labels(
            df,
            close_col=close_col,
            vol_col="volatility",
            pt=float(tb_pt),
            sl=float(tb_sl),
            horizon=int(horizon_bars),
            min_ret=float(tb_min_ret),
        )
        # Use barrier_return as the forward return for label computation
        df = df.with_columns(
            [
                pl.col("barrier_return").alias("fwd_return"),
            ]
        )
    else:
        # Fixed horizon forward return label (shift -horizon)
        df = df.with_columns(
            [
                (pl.col(close_col).shift(-horizon_bars) / pl.col(close_col) - 1.0).alias(
                    "fwd_return"
                ),
            ]
        )
    # Do not drop null fwd_return yet: we need returns/features for holdout backtesting.

    model_norm = str(model).strip().lower()
    if model_norm not in {"baseline", "logreg", "xgboost"}:
        raise ValueError("model must be one of: baseline, logreg, xgboost")

    feature_cols: list[str] = []
    if model_norm == "baseline":
        df = df.with_columns(
            [
                pl.col("returns").rolling_mean(window_size=max(2, horizon_bars)).alias("ret_mean"),
            ]
        )
        df = df.with_columns(
            [
                pl.when(pl.col("ret_mean") > threshold)
                .then(1.0)
                .when(pl.col("ret_mean") < -threshold)
                .then(-1.0)
                .otherwise(0.0)
                .alias("signal"),
            ]
        )
    else:
        df, feature_cols = _build_features(
            df,
            n_lags=n_lags,
            use_fracdiff=use_fracdiff,
            fracdiff_d=fracdiff_d,
            use_vol_features=use_vol_features,
            vol_window=vol_window,
            use_ta_features=use_ta_features,
            use_micro_features=use_micro_features,
        )

    # Build arrays.
    returns_all = df["returns"].to_numpy()
    fwd_all = df["fwd_return"].to_numpy()

    # For training/validation we need valid labels.
    label_ok = np.isfinite(fwd_all)
    y_all = (fwd_all > 0.0).astype(np.int32)

    X_all: np.ndarray | None = None
    if model_norm in {"logreg", "xgboost"}:
        X_all = df.select(feature_cols).to_numpy()

    # Determine holdout boundary (in this post-feature-engineering index space).
    holdout_bars_i = int(max(0, holdout_bars))
    split_idx = len(df)
    if holdout_bars_i > 0:
        if holdout_bars_i >= len(df) - 10:
            raise ValueError("holdout_bars too large for dataset")
        split_idx = len(df) - holdout_bars_i

    # Volatility Targeting Pre-calculation
    vol_scaler = np.ones(len(df), dtype=np.float64)
    if vol_target > 0.0:
        # Calculate rolling volatility (20-day)
        # We use rolling_std(20).shift(1) to ensure no lookahead bias.
        temp_vol = df.select(
            pl.col("returns").rolling_std(window_size=20).shift(1).fill_null(0.0).alias("vol")
        )["vol"].to_numpy()

        # Annualize
        ann_vol = temp_vol * np.sqrt(periods_per_year)

        # Avoid div by zero
        ann_vol = np.where(ann_vol < 1e-4, 1e-4, ann_vol)

        # Target / Realized
        raw_scaler = vol_target / ann_vol

        # Clip leverage (max 5x to prevent explosions)
        vol_scaler = np.clip(raw_scaler, 0.0, 5.0)

    # Prevent label leakage across the split by removing the last `horizon_bars` rows from training.
    safe_train_end = max(0, split_idx - int(max(1, horizon_bars)))

    train_universe = np.arange(safe_train_end)
    train_universe = train_universe[label_ok[:safe_train_end]]

    holdout_universe = (
        np.arange(split_idx, len(df)) if holdout_bars_i > 0 else np.array([], dtype=int)
    )
    # Treat each bar as 1 unit; horizon and embargo expressed in bars.
    bar_index = np.arange(len(df), dtype=np.float64)
    events = build_forward_return_events(bar_index, horizon=float(horizon_bars))

    # If we have a holdout, we do CV on the pre-holdout window only.
    eval_indices = train_universe if holdout_bars_i > 0 else np.arange(len(df))[label_ok]
    eval_positions = np.arange(eval_indices.size, dtype=np.int64)
    eval_events = [events[int(i)] for i in eval_indices]

    # Compute sample weights based on label overlap (De Prado)
    sample_weights_all: np.ndarray | None = None
    if use_sample_weights and len(df) > 0:
        t_start = np.arange(len(df), dtype=np.int64)
        t_end = t_start + int(horizon_bars)
        sample_weights_all = compute_sample_weights(t_start, t_end)

    # Optional inner-loop tuning for logreg.
    selected_C = 1.0
    selected_threshold = float(threshold)
    effective_trials = int(max(1, n_trials))

    if model_norm == "logreg" and bool(tune):
        if X_all is None:
            raise RuntimeError("Feature matrix missing")

        C_values = _parse_float_grid(c_grid) or [1.0]
        thr_values = _parse_float_grid(threshold_grid) or [float(threshold)]
        effective_trials = int(max(1, len(C_values) * len(thr_values)))

        inner_splitter = PurgedKFold(n_splits=int(max(2, tune_splits)), embargo=float(embargo_bars))

        best_score = -1e18
        best: tuple[float, float] = (selected_C, selected_threshold)

        for C in C_values:
            for thr in thr_values:
                fold_sharpes: list[float] = []
                # Inner CV uses only train_universe.
                for inner_train_pos, inner_test_pos in inner_splitter.split(
                    eval_positions, eval_events
                ):
                    inner_train_idx = eval_indices[inner_train_pos]
                    inner_test_idx = eval_indices[inner_test_pos]

                    X_train = X_all[inner_train_idx]
                    y_train = y_all[inner_train_idx]
                    X_test = X_all[inner_test_idx]
                    inner_weights = (
                        sample_weights_all[inner_train_idx]
                        if sample_weights_all is not None
                        else None
                    )

                    proba = _fit_predict_logreg(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        C=float(C),
                        sample_weight=inner_weights,
                    )
                    pos = _signal_from_proba(proba, threshold=float(thr))
                    rets = returns_all[inner_test_idx]

                    bt = backtest_positions(
                        returns=rets,
                        positions=pos,
                        commission_bps=commission_bps,
                        slippage_bps=slippage_bps,
                        periods_per_year=periods_per_year,
                        initial_capital=100000.0,
                        shift_positions=True,
                    )
                    fold_sharpes.append(float(bt.metrics.get("sharpe_ratio", 0.0)))

                score = float(np.mean(fold_sharpes)) if fold_sharpes else -1e18
                if score > best_score:
                    best_score = score
                    best = (float(C), float(thr))

        selected_C, selected_threshold = best

    # Outer CV OOS stream using selected params.
    splitter = PurgedKFold(n_splits=n_splits, embargo=float(embargo_bars))
    fold_metrics: list[dict[str, Any]] = []
    all_cv_returns: list[float] = []
    all_cv_positions: list[float] = []

    for fold, (train_pos, test_pos) in enumerate(
        splitter.split(eval_positions, eval_events), start=1
    ):
        train_idx = eval_indices[train_pos]
        test_idx = eval_indices[test_pos]

        # Get sample weights for this fold's training data
        train_weights = sample_weights_all[train_idx] if sample_weights_all is not None else None

        if model_norm == "baseline":
            test_returns = returns_all[test_idx]
            test_positions = df["signal"].to_numpy()[test_idx]
        elif model_norm == "logreg":
            if X_all is None:
                raise RuntimeError("Feature matrix missing")

            proba = _fit_predict_logreg(
                X_train=X_all[train_idx],
                y_train=y_all[train_idx],
                X_test=X_all[test_idx],
                C=float(selected_C),
                sample_weight=train_weights,
            )
            primary_positions = _signal_from_proba(proba, threshold=float(selected_threshold))

            if use_meta_labeling:
                # Meta-labeling: train secondary model to filter/size bets
                # Create meta-labels from training data primary predictions
                train_proba = _fit_predict_logreg(
                    X_train=X_all[train_idx],
                    y_train=y_all[train_idx],
                    X_test=X_all[train_idx],  # Predict on training set for meta-labels
                    C=float(selected_C),
                    sample_weight=train_weights,
                )
                train_primary_side = _signal_from_proba(
                    train_proba, threshold=float(selected_threshold)
                )
                train_actual_rets = returns_all[train_idx]
                meta_labels_train = get_meta_labels(train_primary_side, train_actual_rets)

                # Train meta-model
                meta_proba_test = _fit_predict_meta_model(
                    X_train=X_all[train_idx],
                    meta_labels_train=meta_labels_train,
                    X_test=X_all[test_idx],
                    model=meta_model,
                )

                # Apply meta-model sizing/filtering
                test_positions = apply_meta_model_sizing(
                    primary_side=primary_positions,
                    meta_proba=meta_proba_test,
                    threshold=float(meta_threshold),
                    use_sizing=bool(meta_use_sizing),
                )
            else:
                test_positions = primary_positions

            test_returns = returns_all[test_idx]
        else:  # xgboost
            if X_all is None:
                raise RuntimeError("Feature matrix missing")

            proba = _fit_predict_xgboost(
                X_train=X_all[train_idx],
                y_train=y_all[train_idx],
                X_test=X_all[test_idx],
                max_depth=3,
                n_estimators=100,
                learning_rate=0.1,
                sample_weight=train_weights,
                tune=tune_model,
            )
            primary_positions = _signal_from_proba(proba, threshold=float(selected_threshold))

            if use_meta_labeling:
                # Meta-labeling: train secondary model to filter/size bets
                train_proba = _fit_predict_xgboost(
                    X_train=X_all[train_idx],
                    y_train=y_all[train_idx],
                    X_test=X_all[train_idx],
                    max_depth=3,
                    n_estimators=100,
                    learning_rate=0.1,
                    sample_weight=train_weights,
                    tune=tune_model,
                )
                train_primary_side = _signal_from_proba(
                    train_proba, threshold=float(selected_threshold)
                )
                train_actual_rets = returns_all[train_idx]
                meta_labels_train = get_meta_labels(train_primary_side, train_actual_rets)

                meta_proba_test = _fit_predict_meta_model(
                    X_train=X_all[train_idx],
                    meta_labels_train=meta_labels_train,
                    X_test=X_all[test_idx],
                    model=meta_model,
                )

                test_positions = apply_meta_model_sizing(
                    primary_side=primary_positions,
                    meta_proba=meta_proba_test,
                    threshold=float(meta_threshold),
                    use_sizing=bool(meta_use_sizing),
                )
            else:
                test_positions = primary_positions

            test_returns = returns_all[test_idx]

        # Apply Volatility Targeting
        if vol_target > 0.0:
            test_positions = test_positions * vol_scaler[test_idx]

        bt = backtest_positions(
            returns=test_returns,
            positions=test_positions,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
            initial_capital=100000.0,
            shift_positions=True,
        )

        m = dict(bt.metrics)
        m["fold"] = fold
        m["n_train"] = int(len(train_idx))
        m["n_test"] = int(len(test_idx))
        fold_metrics.append(m)

        all_cv_returns.extend(bt.strategy_returns.tolist())
        all_cv_positions.extend(bt.positions.tolist())

    cv_metrics = _oos_metrics_from_stream(
        strategy_returns=np.asarray(all_cv_returns, dtype=np.float64),
        positions=np.asarray(all_cv_positions, dtype=np.float64),
        periods_per_year=periods_per_year,
        n_trials=effective_trials,
    )

    metrics: dict[str, Any] = dict(cv_metrics)
    metrics["n_splits"] = int(n_splits)
    metrics["horizon_bars"] = int(horizon_bars)
    metrics["embargo_bars"] = int(embargo_bars)
    metrics["model"] = model_norm
    metrics["holdout_bars"] = int(holdout_bars_i)
    metrics["tune"] = bool(tune) if model_norm in {"logreg", "xgboost"} else False

    metrics["use_triple_barrier"] = bool(use_triple_barrier)
    if use_triple_barrier:
        metrics["tb_pt"] = float(tb_pt)
        metrics["tb_sl"] = float(tb_sl)
        metrics["tb_min_ret"] = float(tb_min_ret)

    if model_norm in {"logreg", "xgboost"}:
        metrics["n_lags"] = int(n_lags)
        metrics["n_trials"] = int(effective_trials)
        metrics["selected_threshold"] = float(selected_threshold)
        metrics["use_fracdiff"] = bool(use_fracdiff)
        metrics["use_vol_features"] = bool(use_vol_features)
        metrics["use_ta_features"] = bool(use_ta_features)
        metrics["use_micro_features"] = bool(use_micro_features)
        metrics["vol_target"] = float(vol_target)
        metrics["tune_model"] = bool(tune_model)
        metrics["use_meta_labeling"] = bool(use_meta_labeling)
        metrics["use_sample_weights"] = bool(use_sample_weights)
        if use_meta_labeling:
            metrics["meta_model"] = str(meta_model)
            metrics["meta_use_sizing"] = bool(meta_use_sizing)
            metrics["meta_threshold"] = float(meta_threshold)
        if model_norm == "logreg":
            metrics["selected_C"] = float(selected_C)

    # Final holdout evaluation (single pass) if requested.
    if holdout_bars_i > 0:
        if model_norm == "baseline":
            holdout_returns = returns_all[holdout_universe]
            holdout_positions = df["signal"].to_numpy()[holdout_universe]
        elif model_norm == "logreg":
            if X_all is None:
                raise RuntimeError("Feature matrix missing")
            if train_universe.size < 10:
                raise ValueError("Not enough training data before holdout")

            # Get sample weights for train universe
            holdout_train_weights = (
                sample_weights_all[train_universe] if sample_weights_all is not None else None
            )

            proba = _fit_predict_logreg(
                X_train=X_all[train_universe],
                y_train=y_all[train_universe],
                X_test=X_all[holdout_universe],
                C=float(selected_C),
                sample_weight=holdout_train_weights,
            )
            primary_holdout_positions = _signal_from_proba(
                proba, threshold=float(selected_threshold)
            )

            if use_meta_labeling:
                # Create meta-labels from train data
                train_proba = _fit_predict_logreg(
                    X_train=X_all[train_universe],
                    y_train=y_all[train_universe],
                    X_test=X_all[train_universe],
                    C=float(selected_C),
                    sample_weight=holdout_train_weights,
                )
                train_primary_side = _signal_from_proba(
                    train_proba, threshold=float(selected_threshold)
                )
                train_actual_rets = returns_all[train_universe]
                meta_labels_train = get_meta_labels(train_primary_side, train_actual_rets)

                meta_proba_holdout = _fit_predict_meta_model(
                    X_train=X_all[train_universe],
                    meta_labels_train=meta_labels_train,
                    X_test=X_all[holdout_universe],
                    model=meta_model,
                )

                holdout_positions = apply_meta_model_sizing(
                    primary_side=primary_holdout_positions,
                    meta_proba=meta_proba_holdout,
                    threshold=float(meta_threshold),
                    use_sizing=bool(meta_use_sizing),
                )
            else:
                holdout_positions = primary_holdout_positions

            holdout_returns = returns_all[holdout_universe]
        else:  # xgboost
            if X_all is None:
                raise RuntimeError("Feature matrix missing")
            if train_universe.size < 10:
                raise ValueError("Not enough training data before holdout")

            # Sample weights for xgboost holdout
            holdout_train_weights_xgb = (
                sample_weights_all[train_universe] if sample_weights_all is not None else None
            )

            proba = _fit_predict_xgboost(
                X_train=X_all[train_universe],
                y_train=y_all[train_universe],
                X_test=X_all[holdout_universe],
                sample_weight=holdout_train_weights_xgb,
                tune=tune_model,
            )
            primary_holdout_positions = _signal_from_proba(
                proba, threshold=float(selected_threshold)
            )

            if use_meta_labeling:
                train_proba = _fit_predict_xgboost(
                    X_train=X_all[train_universe],
                    y_train=y_all[train_universe],
                    X_test=X_all[train_universe],
                    sample_weight=holdout_train_weights_xgb,
                    tune=tune_model,
                )
                train_primary_side = _signal_from_proba(
                    train_proba, threshold=float(selected_threshold)
                )
                train_actual_rets = returns_all[train_universe]
                meta_labels_train = get_meta_labels(train_primary_side, train_actual_rets)

                meta_proba_holdout = _fit_predict_meta_model(
                    X_train=X_all[train_universe],
                    meta_labels_train=meta_labels_train,
                    X_test=X_all[holdout_universe],
                    model=meta_model,
                )

                holdout_positions = apply_meta_model_sizing(
                    primary_side=primary_holdout_positions,
                    meta_proba=meta_proba_holdout,
                    threshold=float(meta_threshold),
                    use_sizing=bool(meta_use_sizing),
                )
            else:
                holdout_positions = primary_holdout_positions

            holdout_returns = returns_all[holdout_universe]

        # Apply Volatility Targeting
        if vol_target > 0.0:
            holdout_positions = holdout_positions * vol_scaler[holdout_universe]

        holdout_bt = backtest_positions(
            returns=holdout_returns,
            positions=holdout_positions,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
            initial_capital=100000.0,
            shift_positions=True,
        )

        holdout_stream = _oos_metrics_from_stream(
            strategy_returns=np.asarray(holdout_bt.strategy_returns, dtype=np.float64),
            positions=np.asarray(holdout_bt.positions, dtype=np.float64),
            periods_per_year=periods_per_year,
            n_trials=1,
        )
        for k, v in holdout_stream.items():
            metrics[f"holdout_{k}"] = v
        metrics["holdout_n_obs_raw"] = int(len(holdout_returns))

    return WalkForwardReport(metrics=metrics, fold_metrics=fold_metrics)
