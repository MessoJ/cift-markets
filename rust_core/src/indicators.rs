/// High-performance ML indicators in Rust
/// 
/// Implements common technical indicators optimized for:
/// - Batch processing (training data)
/// - Real-time streaming (live inference)
/// - SIMD operations where beneficial
/// 
/// Performance targets:
/// - RSI(14) over 10000 bars: <1ms
/// - MACD over 10000 bars: <2ms
/// - Bollinger Bands over 10000 bars: <1ms
/// - All indicators over 10000 bars: <5ms

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ============================================================================
// RSI - RELATIVE STRENGTH INDEX
// ============================================================================

/// Calculate RSI with Wilder's smoothing (standard method)
/// 
/// Formula: RSI = 100 - (100 / (1 + RS))
/// where RS = Exponential Moving Average of Gains / Exponential Moving Average of Losses
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() <= period {
        return vec![50.0; prices.len()]; // Default to neutral
    }

    let mut gains = Vec::with_capacity(prices.len() - 1);
    let mut losses = Vec::with_capacity(prices.len() - 1);
    
    // Calculate price changes
    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change >= 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    let mut rsi_values = vec![50.0; period]; // Warm-up period
    
    // Initial averages (SMA for first period)
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;
    
    // First RSI value
    let rs = if avg_loss > 1e-10 { avg_gain / avg_loss } else { 100.0 };
    rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
    
    // Subsequent values using Wilder's smoothing (EMA)
    let alpha = 1.0 / period as f64;
    
    for i in period..gains.len() {
        avg_gain = (1.0 - alpha) * avg_gain + alpha * gains[i];
        avg_loss = (1.0 - alpha) * avg_loss + alpha * losses[i];
        
        let rs = if avg_loss > 1e-10 { avg_gain / avg_loss } else { 100.0 };
        rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
    }
    
    rsi_values
}

// ============================================================================
// MACD - MOVING AVERAGE CONVERGENCE DIVERGENCE
// ============================================================================

/// Calculate MACD with configurable periods
/// 
/// Returns: (macd_line, signal_line, histogram)
/// Standard settings: fast=12, slow=26, signal=9
pub fn macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if prices.len() < slow_period {
        let n = prices.len();
        return (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    }

    // Calculate EMAs
    let ema_fast = ema(prices, fast_period);
    let ema_slow = ema(prices, slow_period);
    
    // MACD line = Fast EMA - Slow EMA
    let mut macd_line = Vec::with_capacity(prices.len());
    for i in 0..prices.len() {
        macd_line.push(ema_fast[i] - ema_slow[i]);
    }
    
    // Signal line = EMA of MACD line
    let signal_line = ema(&macd_line, signal_period);
    
    // Histogram = MACD line - Signal line
    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();
    
    (macd_line, signal_line, histogram)
}

/// Exponential Moving Average
fn ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() || period == 0 {
        return vec![0.0; prices.len()];
    }

    let mut ema_values = Vec::with_capacity(prices.len());
    let alpha = 2.0 / (period + 1) as f64;
    
    // SMA for initial value
    let initial: f64 = prices.iter().take(period).sum::<f64>() / period.min(prices.len()) as f64;
    
    // Warm-up with SMA approximation
    for i in 0..period.min(prices.len()) {
        let sma: f64 = prices[..=i].iter().sum::<f64>() / (i + 1) as f64;
        ema_values.push(sma);
    }
    
    // EMA calculation
    if prices.len() > period {
        let mut prev_ema = initial;
        for i in period..prices.len() {
            let current_ema = alpha * prices[i] + (1.0 - alpha) * prev_ema;
            ema_values.push(current_ema);
            prev_ema = current_ema;
        }
    }
    
    ema_values
}

// ============================================================================
// BOLLINGER BANDS
// ============================================================================

/// Calculate Bollinger Bands
/// 
/// Returns: (upper_band, middle_band, lower_band, bandwidth, percent_b)
/// Standard settings: period=20, num_std=2.0
pub fn bollinger_bands(
    prices: &[f64],
    period: usize,
    num_std: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    
    if n < period {
        return (
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.5; n],  // Neutral percent_b
        );
    }

    let mut upper = Vec::with_capacity(n);
    let mut middle = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);
    let mut bandwidth = Vec::with_capacity(n);
    let mut percent_b = Vec::with_capacity(n);
    
    // Warm-up period
    for i in 0..period - 1 {
        let window = &prices[..=i];
        let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
        let std_dev = variance.sqrt();
        
        let u = mean + num_std * std_dev;
        let l = mean - num_std * std_dev;
        
        upper.push(u);
        middle.push(mean);
        lower.push(l);
        bandwidth.push(if mean > 1e-10 { (u - l) / mean } else { 0.0 });
        percent_b.push(if (u - l) > 1e-10 { (prices[i] - l) / (u - l) } else { 0.5 });
    }
    
    // Rolling calculation
    for i in (period - 1)..n {
        let window = &prices[i + 1 - period..=i];
        
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();
        
        let u = mean + num_std * std_dev;
        let l = mean - num_std * std_dev;
        
        upper.push(u);
        middle.push(mean);
        lower.push(l);
        bandwidth.push(if mean > 1e-10 { (u - l) / mean } else { 0.0 });
        percent_b.push(if (u - l) > 1e-10 { (prices[i] - l) / (u - l) } else { 0.5 });
    }
    
    (upper, middle, lower, bandwidth, percent_b)
}

// ============================================================================
// ATR - AVERAGE TRUE RANGE
// ============================================================================

/// Calculate Average True Range (volatility indicator)
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len().min(low.len()).min(close.len());
    
    if n < 2 {
        return vec![0.0; n];
    }

    // Calculate True Range
    let mut tr = Vec::with_capacity(n);
    tr.push(high[0] - low[0]); // First bar
    
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr.push(hl.max(hc).max(lc));
    }
    
    // Apply Wilder's smoothing (EMA with 1/period)
    let mut atr_values = Vec::with_capacity(n);
    
    // Initial ATR (SMA of first period)
    let initial_atr: f64 = tr.iter().take(period).sum::<f64>() / period.min(n) as f64;
    
    for i in 0..period.min(n) {
        let sma: f64 = tr[..=i].iter().sum::<f64>() / (i + 1) as f64;
        atr_values.push(sma);
    }
    
    let alpha = 1.0 / period as f64;
    let mut prev_atr = initial_atr;
    
    for i in period..n {
        let current_atr = (1.0 - alpha) * prev_atr + alpha * tr[i];
        atr_values.push(current_atr);
        prev_atr = current_atr;
    }
    
    atr_values
}

// ============================================================================
// STOCHASTIC OSCILLATOR
// ============================================================================

/// Calculate Stochastic Oscillator (%K and %D)
/// 
/// Returns: (k_values, d_values)
/// Standard settings: k_period=14, k_slow=3, d_period=3
pub fn stochastic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_period: usize,
    k_slow: usize,
    d_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = high.len().min(low.len()).min(close.len());
    
    if n < k_period {
        return (vec![50.0; n], vec![50.0; n]);
    }

    // Calculate raw %K
    let mut raw_k = Vec::with_capacity(n);
    
    for i in 0..k_period - 1 {
        raw_k.push(50.0); // Neutral default
    }
    
    for i in (k_period - 1)..n {
        let window_high = &high[i + 1 - k_period..=i];
        let window_low = &low[i + 1 - k_period..=i];
        
        let highest = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lowest = window_low.iter().cloned().fold(f64::INFINITY, f64::min);
        
        let k = if (highest - lowest) > 1e-10 {
            100.0 * (close[i] - lowest) / (highest - lowest)
        } else {
            50.0
        };
        
        raw_k.push(k);
    }
    
    // Smooth %K (Slow Stochastic)
    let k_values = sma(&raw_k, k_slow);
    
    // %D is SMA of %K
    let d_values = sma(&k_values, d_period);
    
    (k_values, d_values)
}

/// Simple Moving Average
fn sma(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() || period == 0 {
        return vec![0.0; values.len()];
    }

    let mut result = Vec::with_capacity(values.len());
    
    for i in 0..values.len() {
        let start = if i >= period { i + 1 - period } else { 0 };
        let window = &values[start..=i];
        let avg = window.iter().sum::<f64>() / window.len() as f64;
        result.push(avg);
    }
    
    result
}

// ============================================================================
// VWAP - VOLUME WEIGHTED AVERAGE PRICE
// ============================================================================

/// Calculate cumulative VWAP
pub fn vwap(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = high.len().min(low.len()).min(close.len()).min(volume.len());
    
    if n == 0 {
        return Vec::new();
    }

    let mut cum_pv = 0.0;
    let mut cum_vol = 0.0;
    let mut vwap_values = Vec::with_capacity(n);
    
    for i in 0..n {
        let typical_price = (high[i] + low[i] + close[i]) / 3.0;
        cum_pv += typical_price * volume[i];
        cum_vol += volume[i];
        
        let vwap = if cum_vol > 1e-10 { cum_pv / cum_vol } else { close[i] };
        vwap_values.push(vwap);
    }
    
    vwap_values
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

/// Python wrapper for ML indicators
#[pyclass]
pub struct FastIndicators;

#[pymethods]
impl FastIndicators {
    #[new]
    fn new() -> Self {
        FastIndicators
    }

    /// Calculate RSI (Relative Strength Index)
    /// 
    /// Args:
    ///     prices: List of closing prices
    ///     period: RSI period (default: 14)
    /// 
    /// Returns: List of RSI values (0-100)
    #[staticmethod]
    #[pyo3(signature = (prices, period = 14))]
    fn rsi(prices: Vec<f64>, period: usize) -> Vec<f64> {
        rsi(&prices, period)
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    /// 
    /// Args:
    ///     prices: List of closing prices
    ///     fast_period: Fast EMA period (default: 12)
    ///     slow_period: Slow EMA period (default: 26)
    ///     signal_period: Signal line period (default: 9)
    /// 
    /// Returns: Tuple of (macd_line, signal_line, histogram)
    #[staticmethod]
    #[pyo3(signature = (prices, fast_period = 12, slow_period = 26, signal_period = 9))]
    fn macd(
        prices: Vec<f64>,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        macd(&prices, fast_period, slow_period, signal_period)
    }

    /// Calculate Bollinger Bands
    /// 
    /// Args:
    ///     prices: List of closing prices
    ///     period: Moving average period (default: 20)
    ///     num_std: Number of standard deviations (default: 2.0)
    /// 
    /// Returns: Tuple of (upper_band, middle_band, lower_band, bandwidth, percent_b)
    #[staticmethod]
    #[pyo3(signature = (prices, period = 20, num_std = 2.0))]
    fn bollinger_bands(
        prices: Vec<f64>,
        period: usize,
        num_std: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        bollinger_bands(&prices, period, num_std)
    }

    /// Calculate Average True Range (ATR)
    /// 
    /// Args:
    ///     high: List of high prices
    ///     low: List of low prices
    ///     close: List of closing prices
    ///     period: ATR period (default: 14)
    /// 
    /// Returns: List of ATR values
    #[staticmethod]
    #[pyo3(signature = (high, low, close, period = 14))]
    fn atr(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: usize) -> Vec<f64> {
        atr(&high, &low, &close, period)
    }

    /// Calculate Stochastic Oscillator
    /// 
    /// Args:
    ///     high: List of high prices
    ///     low: List of low prices
    ///     close: List of closing prices
    ///     k_period: %K lookback period (default: 14)
    ///     k_slow: %K smoothing period (default: 3)
    ///     d_period: %D smoothing period (default: 3)
    /// 
    /// Returns: Tuple of (%K values, %D values)
    #[staticmethod]
    #[pyo3(signature = (high, low, close, k_period = 14, k_slow = 3, d_period = 3))]
    fn stochastic(
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        k_period: usize,
        k_slow: usize,
        d_period: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        stochastic(&high, &low, &close, k_period, k_slow, d_period)
    }

    /// Calculate VWAP (Volume Weighted Average Price)
    /// 
    /// Args:
    ///     high: List of high prices
    ///     low: List of low prices
    ///     close: List of closing prices
    ///     volume: List of volumes
    /// 
    /// Returns: List of VWAP values
    #[staticmethod]
    fn vwap(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, volume: Vec<f64>) -> Vec<f64> {
        vwap(&high, &low, &close, &volume)
    }

    /// Calculate EMA (Exponential Moving Average)
    /// 
    /// Args:
    ///     prices: List of prices
    ///     period: EMA period
    /// 
    /// Returns: List of EMA values
    #[staticmethod]
    fn ema(prices: Vec<f64>, period: usize) -> Vec<f64> {
        ema(&prices, period)
    }

    /// Calculate SMA (Simple Moving Average)
    /// 
    /// Args:
    ///     prices: List of prices
    ///     period: SMA period
    /// 
    /// Returns: List of SMA values
    #[staticmethod]
    fn sma(prices: Vec<f64>, period: usize) -> Vec<f64> {
        sma(&prices, period)
    }

    /// Calculate all common indicators at once (more efficient)
    /// 
    /// Args:
    ///     high: High prices
    ///     low: Low prices
    ///     close: Close prices
    ///     volume: Volume
    /// 
    /// Returns: Dict with all indicator values
    #[staticmethod]
    fn all_indicators(
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    ) -> std::collections::HashMap<String, Vec<f64>> {
        let mut result = std::collections::HashMap::new();
        
        // RSI
        result.insert("rsi_14".to_string(), rsi(&close, 14));
        
        // MACD
        let (macd_line, signal_line, histogram) = macd(&close, 12, 26, 9);
        result.insert("macd_line".to_string(), macd_line);
        result.insert("macd_signal".to_string(), signal_line);
        result.insert("macd_histogram".to_string(), histogram);
        
        // Bollinger Bands
        let (upper, middle, lower, bandwidth, percent_b) = bollinger_bands(&close, 20, 2.0);
        result.insert("bb_upper".to_string(), upper);
        result.insert("bb_middle".to_string(), middle);
        result.insert("bb_lower".to_string(), lower);
        result.insert("bb_bandwidth".to_string(), bandwidth);
        result.insert("bb_percent_b".to_string(), percent_b);
        
        // ATR
        result.insert("atr_14".to_string(), atr(&high, &low, &close, 14));
        
        // Stochastic
        let (k, d) = stochastic(&high, &low, &close, 14, 3, 3);
        result.insert("stoch_k".to_string(), k);
        result.insert("stoch_d".to_string(), d);
        
        // VWAP
        result.insert("vwap".to_string(), vwap(&high, &low, &close, &volume));
        
        // EMAs
        result.insert("ema_9".to_string(), ema(&close, 9));
        result.insert("ema_20".to_string(), ema(&close, 20));
        result.insert("ema_50".to_string(), ema(&close, 50));
        
        // SMAs
        result.insert("sma_20".to_string(), sma(&close, 20));
        result.insert("sma_50".to_string(), sma(&close, 50));
        result.insert("sma_200".to_string(), sma(&close, 200));
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        // Trending up prices should give RSI > 50
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let rsi_values = rsi(&prices, 14);
        
        assert!(rsi_values.last().unwrap() > &60.0);
    }

    #[test]
    fn test_macd() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let (macd_line, signal_line, histogram) = macd(&prices, 12, 26, 9);
        
        assert_eq!(macd_line.len(), prices.len());
        assert_eq!(signal_line.len(), prices.len());
        assert_eq!(histogram.len(), prices.len());
    }

    #[test]
    fn test_bollinger_bands() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0).collect();
        let (upper, middle, lower, bandwidth, percent_b) = bollinger_bands(&prices, 20, 2.0);
        
        // Upper should be above middle, middle above lower
        for i in 19..prices.len() {
            assert!(upper[i] > middle[i]);
            assert!(middle[i] > lower[i]);
        }
        
        // Percent B should be between 0 and 1 most of the time
        for p in percent_b.iter().skip(19) {
            assert!(*p >= -0.5 && *p <= 1.5); // Some tolerance for outliers
        }
    }

    #[test]
    fn test_atr() {
        let high: Vec<f64> = (0..50).map(|i| 102.0 + (i as f64 * 0.1).sin()).collect();
        let low: Vec<f64> = (0..50).map(|i| 98.0 + (i as f64 * 0.1).sin()).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        
        let atr_values = atr(&high, &low, &close, 14);
        
        assert_eq!(atr_values.len(), 50);
        // ATR should be positive and roughly ~4 (high-low range)
        assert!(atr_values.last().unwrap() > &1.0);
        assert!(atr_values.last().unwrap() < &10.0);
    }

    #[test]
    fn test_stochastic() {
        let high: Vec<f64> = (0..50).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        
        let (k, d) = stochastic(&high, &low, &close, 14, 3, 3);
        
        assert_eq!(k.len(), 50);
        assert_eq!(d.len(), 50);
        
        // Values should be 0-100
        for v in k.iter().chain(d.iter()) {
            assert!(*v >= 0.0 && *v <= 100.0);
        }
    }
}
