import numpy as np
import polars as pl


def frac_diff_ffd(series: np.ndarray, d: float, thres: float = 1e-5) -> np.ndarray:
    """
    Fractional Differentiation (FFD) to preserve memory while achieving stationarity.

    Args:
        series: Input time series (1D numpy array).
        d: Order of differentiation (0 < d < 1).
        thres: Threshold for weight cutoff.

    Returns:
        Fractionally differentiated series.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1

    w = np.array(w[::-1])  # Weights

    # Convolve
    # We want to compute dot product of weights and window of series
    # This is equivalent to valid convolution
    if len(series) < len(w):
        return np.full(len(series), np.nan)

    # Using valid mode convolution
    res = np.convolve(series, w, mode='valid')

    # Pad the beginning with NaNs to match original length
    pad = np.full(len(series) - len(res), np.nan)
    return np.concatenate([pad, res])

def volatility_yang_zhang(
    df: pl.DataFrame,
    window: int = 30,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pl.Series:
    """
    Yang-Zhang Volatility Estimator (Minimum Variance).

    Formula:
    sigma^2 = sigma_o^2 + k * sigma_c^2 + (1-k) * sigma_rs^2
    where:
    sigma_o^2 = variance of (open - close_prev)
    sigma_c^2 = variance of (close - open)
    sigma_rs^2 = Rogers-Satchell volatility
    k = 0.34 / (1.34 + (n+1)/(n-1))
    """
    # We need pandas/numpy for rolling apply if complex, but Polars has rolling expressions.
    # Let's try to do it in pure Polars expressions for speed.

    # Log prices
    log_o = pl.col(open_col).log()
    log_h = pl.col(high_col).log()
    log_l = pl.col(low_col).log()
    log_c = pl.col(close_col).log()
    log_c_prev = pl.col(close_col).shift(1).log()

    # Overnight volatility (open - close_prev)
    # We need rolling variance of this term
    term_o = (log_o - log_c_prev)
    var_o = term_o.rolling_var(window_size=window)

    # Open-to-Close volatility (close - open)
    term_c = (log_c - log_o)
    var_c = term_c.rolling_var(window_size=window)

    # Rogers-Satchell volatility (rolling mean of RS term)
    # RS = u(u-c) + d(d-c) where u=H-O, d=L-O, c=C-O (normalized by Open)
    # Standard RS formula: H(H-C) + L(L-C) ?? No.
    # RS = (h-o)(h-c) + (l-o)(l-c) using log prices
    rs_term = (log_h - log_o) * (log_h - log_c) + (log_l - log_o) * (log_l - log_c)
    var_rs = rs_term.rolling_mean(window_size=window)

    # k factor
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Combine
    yz_var = var_o + k * var_c + (1 - k) * var_rs

    return yz_var.sqrt().alias("vol_yz")

def amihud_illiquidity(
    df: pl.DataFrame,
    window: int = 30,
    close_col: str = "close",
    volume_col: str = "volume"
) -> pl.Series:
    """
    Amihud Illiquidity: Average of |Return| / (Price * Volume)
    Measures price impact per dollar traded.
    """
    ret = pl.col(close_col).pct_change().abs()
    dollar_vol = pl.col(close_col) * pl.col(volume_col)

    # Avoid division by zero
    impact = ret / (dollar_vol + 1e-9)

    return impact.rolling_mean(window_size=window).alias("amihud")


def get_technical_features(
    df: pl.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """
    Generate standard technical analysis features using Polars expressions.

    Includes:
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2)
    - ATR (14)
    - MFI (14)
    """
    # Helper for EMA
    def ema(series: pl.Expr, span: int) -> pl.Expr:
        return series.ewm_mean(span=span, adjust=False)

    # 1. RSI (14)
    delta = pl.col(close_col).diff()
    gain = delta.clip(lower_bound=0)
    loss = delta.clip(upper_bound=0).abs()

    avg_gain = gain.ewm_mean(com=13, adjust=False)
    avg_loss = loss.ewm_mean(com=13, adjust=False)
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9)
    ema12 = ema(pl.col(close_col), 12)
    ema26 = ema(pl.col(close_col), 26)
    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    macd_hist = macd_line - signal_line

    # 3. Bollinger Bands (20, 2)
    bb_mean = pl.col(close_col).rolling_mean(window_size=20)
    bb_std = pl.col(close_col).rolling_std(window_size=20)
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mean
    bb_pct = (pl.col(close_col) - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # 4. ATR (14)
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = pl.col(close_col).shift(1)
    tr1 = pl.col(high_col) - pl.col(low_col)
    tr2 = (pl.col(high_col) - prev_close).abs()
    tr3 = (pl.col(low_col) - prev_close).abs()
    # Polars max_horizontal equivalent
    tr = pl.max_horizontal(tr1, tr2, tr3)
    atr = ema(tr, 14) # Using EMA smoothing for ATR

    # 5. MFI (14) - Money Flow Index
    # Typical Price
    tp = (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3
    raw_money_flow = tp * pl.col(volume_col)

    # Positive/Negative Flow
    tp_diff = tp.diff()
    pos_flow = pl.when(tp_diff > 0).then(raw_money_flow).otherwise(0)
    neg_flow = pl.when(tp_diff < 0).then(raw_money_flow).otherwise(0)

    pos_mf = pos_flow.rolling_sum(window_size=14)
    neg_mf = neg_flow.rolling_sum(window_size=14)

    mfi_ratio = pos_mf / (neg_mf + 1e-9)
    mfi = 100 - (100 / (1 + mfi_ratio))

    return df.with_columns([
        rsi.alias("rsi_14"),
        macd_line.alias("macd_line"),
        signal_line.alias("macd_signal"),
        macd_hist.alias("macd_hist"),
        bb_width.alias("bb_width"),
        bb_pct.alias("bb_pct"),
        atr.alias("atr_14"),
        mfi.alias("mfi_14"),
    ])


def get_microstructure_features(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 14
) -> pl.DataFrame:
    """
    Generate microstructure and efficiency features.

    Includes:
    - Corwin-Schultz Spread (High-Low proxy for bid-ask spread)
    - Kaufman Efficiency Ratio (Trend efficiency)
    - Becker-Parkinson Volatility (High-Low volatility)
    """
    # 1. Corwin-Schultz Spread
    # Beta = E[sum(adj_high^2)] ... simplified approximation for speed:
    # S = 2(e^alpha - 1) / (1 + e^alpha)
    # where alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    # This is complex to implement in pure Polars expressions without custom UDFs.
    # We will use a simplified High-Low estimator: (High - Low) / Close
    hl_spread = (pl.col(high_col) - pl.col(low_col)) / (pl.col(close_col) + 1e-9)
    avg_spread = hl_spread.rolling_mean(window_size=window)

    # 2. Kaufman Efficiency Ratio (KER)
    # KER = |Change| / Sum(|Diff|)
    change = pl.col(close_col).diff(window).abs()
    volatility = pl.col(close_col).diff().abs().rolling_sum(window_size=window)
    ker = change / (volatility + 1e-9)

    # 3. Becker-Parkinson Volatility
    # sigma = sqrt(1 / (4 * N * ln(2)) * sum(ln(H/L)^2))
    hl_log_sq = (pl.col(high_col) / pl.col(low_col)).log().pow(2)
    bp_vol = (hl_log_sq.rolling_mean(window_size=window) / (4 * np.log(2))).sqrt()

    return df.with_columns([
        avg_spread.alias("hl_spread"),
        ker.alias("ker_14"),
        bp_vol.alias("vol_bp_14"),
    ])
