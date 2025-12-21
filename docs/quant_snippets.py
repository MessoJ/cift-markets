import numpy as np
import pandas as pd


def get_weights_ffd(d, thres):
    """
    Generates weights for the Fast Fractional Differentiation (FFD).
    Source: Lopez de Prado, Advances in Financial Machine Learning.
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Constant width window (FFD) method.
    Note: This method preserves memory better than standard differencing.
    """
    # 1) Compute weights for the longest series
    w = get_weights_ffd(d, thres)
    width = len(w) - 1

    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)

    df = pd.concat(df, axis=1)
    return df

def get_triple_barrier_labels(prices, events, pt=1, sl=1, vertical_barrier_bars=100, trgt=0.01):
    """
    Simplified Triple Barrier Method.

    :param prices: pd.Series of close prices.
    :param events: pd.DatetimeIndex of entry times.
    :param pt: Profit Taking multiplier (e.g., 2x volatility).
    :param sl: Stop Loss multiplier (e.g., 2x volatility).
    :param vertical_barrier_bars: Number of bars for the time limit.
    :param trgt: Target volatility or fixed target return (e.g., 0.01 for 1%).
    :return: pd.DataFrame with 'ret' (return) and 'bin' (label: -1, 0, 1).
    """
    out = pd.DataFrame(index=events)
    out['t1'] = np.nan # Vertical barrier timestamp
    out['pt'] = pt * trgt
    out['sl'] = -sl * trgt

    for loc in events:
        # 1. Vertical Barrier
        idx = prices.index.get_loc(loc)
        if idx + vertical_barrier_bars < len(prices):
            t1 = prices.index[idx + vertical_barrier_bars]
        else:
            t1 = prices.index[-1]

        # Slice price path
        path = prices[loc:t1]

        # 2. Horizontal Barriers
        # Returns relative to initial price
        returns = (path / prices[loc]) - 1

        # Check touch
        # First touch of PT
        touch_pt = returns[returns > out.loc[loc, 'pt']].index.min()
        # First touch of SL
        touch_sl = returns[returns < out.loc[loc, 'sl']].index.min()

        # Determine which happened first
        first_touch = pd.to_datetime(t1)
        label = 0 # Time limit

        if pd.notna(touch_pt) and pd.notna(touch_sl):
            if touch_pt < touch_sl:
                first_touch = touch_pt
                label = 1
            else:
                first_touch = touch_sl
                label = -1
        elif pd.notna(touch_pt):
            first_touch = touch_pt
            label = 1
        elif pd.notna(touch_sl):
            first_touch = touch_sl
            label = -1

        out.loc[loc, 't1'] = first_touch
        out.loc[loc, 'bin'] = label
        out.loc[loc, 'ret'] = returns[first_touch]

    return out
