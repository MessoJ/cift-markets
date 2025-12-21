
import numpy as np
import polars as pl


def get_triple_barrier_labels(
    df: pl.DataFrame,
    close_col: str = "close",
    vol_col: str = "volatility",
    pt: float = 1.0,
    sl: float = 1.0,
    horizon: int = 10,
    min_ret: float = 0.0
) -> pl.DataFrame:
    """
    Triple Barrier Method Labeling.

    Args:
        df: Polars DataFrame with prices and volatility.
        close_col: Name of close price column.
        vol_col: Name of volatility column (used for dynamic barriers).
        pt: Profit Take multiplier (barrier = pt * vol).
        sl: Stop Loss multiplier (barrier = sl * vol).
        horizon: Max holding period in bars.
        min_ret: Minimum return to consider a label non-zero.

    Returns:
        DataFrame with 'label' (1, -1, 0) and 'ret' (realized return).
    """
    # Convert to numpy for speed in loop (Triple Barrier is path-dependent)
    # Polars doesn't easily support "first of 3 events" logic without complex expressions.

    prices = df[close_col].to_numpy()
    vols = df[vol_col].to_numpy()
    df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(len(df))

    n = len(prices)
    labels = np.zeros(n, dtype=np.int32)
    out_rets = np.zeros(n, dtype=np.float64)

    # This loop is slow in Python. For production, use Numba or Cython.
    # For now, we optimize by iterating only valid start points.

    for i in range(n - horizon):
        p0 = prices[i]
        vol = vols[i]

        # Dynamic barriers
        upper = p0 * (1 + pt * vol)
        lower = p0 * (1 - sl * vol)

        # Path
        path = prices[i+1 : i+1+horizon]

        # Find first touch
        # 1. Touch Upper
        # 2. Touch Lower

        # Vectorized check for this window
        touch_upper = np.where(path >= upper)[0]
        touch_lower = np.where(path <= lower)[0]

        first_upper = touch_upper[0] if len(touch_upper) > 0 else horizon + 1
        first_lower = touch_lower[0] if len(touch_lower) > 0 else horizon + 1

        if first_upper == horizon + 1 and first_lower == horizon + 1:
            # Time Limit hit
            ret = (path[-1] - p0) / p0
            label = 0
            # Optional: Label based on sign if return is significant
            if ret > min_ret: label = 1
            elif ret < -min_ret: label = -1

        elif first_upper < first_lower:
            # PT hit first
            ret = (path[first_upper] - p0) / p0
            label = 1
        else:
            # SL hit first
            ret = (path[first_lower] - p0) / p0
            label = -1

        labels[i] = label
        out_rets[i] = ret

    # Pad the end
    labels[n-horizon:] = 0
    out_rets[n-horizon:] = 0.0

    return df.with_columns([
        pl.Series("label", labels),
        pl.Series("barrier_return", out_rets)
    ])


def get_meta_labels(
    primary_side: np.ndarray,
    actual_returns: np.ndarray,
) -> np.ndarray:
    """
    Create meta-labels for training a secondary (meta) model.

    Meta-labeling (De Prado, AFML Ch. 3) trains a secondary model to predict
    whether the primary model's prediction is CORRECT. This allows:
    1. Better bet sizing (filter out bad bets)
    2. Maintain high recall on primary model, let meta-model handle precision

    Args:
        primary_side: Primary model's directional predictions (+1, -1, 0).
        actual_returns: Realized returns for each observation.

    Returns:
        np.ndarray: Meta-labels (1 = primary was correct, 0 = incorrect).

    Example:
        If primary predicted +1 and actual return > 0: meta_label = 1
        If primary predicted +1 and actual return < 0: meta_label = 0
        If primary predicted 0 (no bet): meta_label = 0 (no bet to evaluate)
    """
    primary = np.asarray(primary_side, dtype=np.float64)
    returns = np.asarray(actual_returns, dtype=np.float64)

    # Meta-label = 1 if sign(primary) == sign(return) AND primary != 0
    # This means: did the primary model correctly predict direction?
    primary_correct = (primary * returns) > 0  # Both same sign = correct
    no_bet = primary == 0

    meta_labels = np.where(no_bet, 0, primary_correct.astype(int))
    return meta_labels


def apply_meta_model_sizing(
    primary_side: np.ndarray,
    meta_proba: np.ndarray,
    threshold: float = 0.5,
    use_sizing: bool = True,
) -> np.ndarray:
    """
    Apply meta-model predictions for bet sizing / filtering.

    Args:
        primary_side: Primary model's directional predictions (+1, -1, 0).
        meta_proba: Meta-model's probability that primary is correct.
        threshold: Minimum meta_proba to take the bet (0.5 = neutral).
        use_sizing: If True, size = primary_side * meta_proba.
                    If False, binary filter: take bet if meta_proba >= threshold.

    Returns:
        np.ndarray: Final position sizes.
    """
    primary = np.asarray(primary_side, dtype=np.float64)
    proba = np.asarray(meta_proba, dtype=np.float64)

    if use_sizing:
        # Continuous bet sizing: position = side * confidence
        # De Prado: size ∝ (meta_proba - 0.5) * 2 for scaling to [-1, 1]
        # Or simpler: size = side * meta_proba (0.5 = half size, 1.0 = full size)
        positions = primary * proba
    else:
        # Binary filter: only bet if meta_model is confident
        bet_filter = proba >= threshold
        positions = np.where(bet_filter, primary, 0.0)

    return positions


def compute_sample_weights(
    t_start: np.ndarray,
    t_end: np.ndarray,
    method: str = "avg_uniqueness",
) -> np.ndarray:
    """
    Compute sample weights based on label overlap (De Prado, AFML Ch. 4).

    In financial ML, observations with overlapping labels are not IID.
    Observations that span unique time periods should get higher weights.

    Average Uniqueness Method:
    For each observation i, compute what fraction of its lifetime is unique
    (not overlapping with other observations). The weight is proportional
    to this average uniqueness.

    Args:
        t_start: Start timestamps (bar indices) for each observation.
        t_end: End timestamps (bar indices) for each observation's label horizon.
        method: Weighting method, "avg_uniqueness" (default) or "return_attr".

    Returns:
        np.ndarray: Sample weights (sum to len(t_start)).
    """
    t_start = np.asarray(t_start, dtype=np.int64)
    t_end = np.asarray(t_end, dtype=np.int64)
    n = len(t_start)

    if n == 0:
        return np.array([], dtype=np.float64)

    # Build concurrency matrix: for each time t, count overlapping observations
    t_min = int(t_start.min())
    t_max = int(t_end.max())

    # Count how many observations are active at each time point
    concurrency = np.zeros(t_max - t_min + 1, dtype=np.float64)

    for i in range(n):
        for t in range(int(t_start[i]) - t_min, int(t_end[i]) - t_min + 1):
            if 0 <= t < len(concurrency):
                concurrency[t] += 1.0

    # For each observation, compute average uniqueness
    # Uniqueness at time t = 1 / concurrency[t]
    # Average uniqueness = mean(uniqueness) over the observation's lifetime

    weights = np.zeros(n, dtype=np.float64)

    for i in range(n):
        uniqueness_sum = 0.0
        count = 0
        for t in range(int(t_start[i]) - t_min, int(t_end[i]) - t_min + 1):
            if 0 <= t < len(concurrency) and concurrency[t] > 0:
                uniqueness_sum += 1.0 / concurrency[t]
                count += 1

        if count > 0:
            weights[i] = uniqueness_sum / count  # Average uniqueness
        else:
            weights[i] = 1.0

    # Normalize weights to sum to n (sklearn convention)
    total = weights.sum()
    if total > 0:
        weights = weights * (n / total)
    else:
        weights = np.ones(n, dtype=np.float64)

    return weights


def compute_return_attribution_weights(
    returns: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
) -> np.ndarray:
    """
    Compute sample weights based on return attribution (De Prado, AFML Ch. 4).

    Weights are proportional to the absolute return contribution of each
    observation, adjusted for overlap.

    Args:
        returns: Array of returns for each observation.
        t_start: Start timestamps for each observation.
        t_end: End timestamps for each observation.

    Returns:
        np.ndarray: Sample weights (sum to len(returns)).
    """
    returns = np.asarray(returns, dtype=np.float64)

    # First get uniqueness weights
    uniqueness = compute_sample_weights(t_start, t_end, method="avg_uniqueness")

    # Weight by absolute return (more impactful observations)
    abs_ret = np.abs(returns)

    # Combine: weight = uniqueness * abs_return
    weights = uniqueness * abs_ret

    # Normalize
    n = len(returns)
    total = weights.sum()
    if total > 0:
        weights = weights * (n / total)
    else:
        weights = np.ones(n, dtype=np.float64)

    return weights
