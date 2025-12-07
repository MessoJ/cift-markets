/// High-performance market data processor
/// Implements SIMD-optimized calculations where possible
pub struct MarketDataProcessor {
    // State can be added as needed
}

impl MarketDataProcessor {
    pub fn new() -> Self {
        MarketDataProcessor {}
    }

    /// Calculate Volume-Weighted Average Price (VWAP)
    /// Input: Vec of (price, volume) tuples
    /// Returns: VWAP
    pub fn calculate_vwap(&self, ticks: &[(f64, f64)]) -> f64 {
        if ticks.is_empty() {
            return 0.0;
        }

        let mut total_pv = 0.0;
        let mut total_volume = 0.0;

        for &(price, volume) in ticks {
            total_pv += price * volume;
            total_volume += volume;
        }

        if total_volume > 0.0 {
            total_pv / total_volume
        } else {
            0.0
        }
    }

    /// Calculate Order Flow Imbalance (OFI)
    /// Input: bid volumes, ask volumes (matched by price level)
    /// Returns: OFI metric (-1 to 1, where negative = sell pressure, positive = buy pressure)
    pub fn calculate_ofi(&self, bid_volumes: &[f64], ask_volumes: &[f64]) -> f64 {
        if bid_volumes.is_empty() && ask_volumes.is_empty() {
            return 0.0;
        }

        let total_bid: f64 = bid_volumes.iter().sum();
        let total_ask: f64 = ask_volumes.iter().sum();
        let total = total_bid + total_ask;

        if total > 0.0 {
            (total_bid - total_ask) / total
        } else {
            0.0
        }
    }

    /// Calculate weighted OFI (distance-weighted)
    /// Closer price levels have more weight
    pub fn calculate_weighted_ofi(&self, bid_volumes: &[f64], ask_volumes: &[f64]) -> f64 {
        if bid_volumes.is_empty() && ask_volumes.is_empty() {
            return 0.0;
        }

        let mut weighted_bid = 0.0;
        let mut weighted_ask = 0.0;

        // Weight by 1/distance from best price
        for (i, &volume) in bid_volumes.iter().enumerate() {
            let weight = 1.0 / (i + 1) as f64;
            weighted_bid += volume * weight;
        }

        for (i, &volume) in ask_volumes.iter().enumerate() {
            let weight = 1.0 / (i + 1) as f64;
            weighted_ask += volume * weight;
        }

        let total = weighted_bid + weighted_ask;
        if total > 0.0 {
            (weighted_bid - weighted_ask) / total
        } else {
            0.0
        }
    }

    /// Calculate microprice
    /// Formula: (bid * ask_volume + ask * bid_volume) / (bid_volume + ask_volume)
    /// This is a volume-weighted midpoint
    pub fn calculate_microprice(
        &self,
        best_bid: f64,
        best_ask: f64,
        bid_volume: f64,
        ask_volume: f64,
    ) -> f64 {
        let total_volume = bid_volume + ask_volume;
        if total_volume > 0.0 {
            (best_bid * ask_volume + best_ask * bid_volume) / total_volume
        } else {
            (best_bid + best_ask) / 2.0 // Fallback to midpoint
        }
    }

    /// Calculate effective spread
    pub fn calculate_effective_spread(&self, trade_price: f64, midpoint: f64) -> f64 {
        2.0 * (trade_price - midpoint).abs()
    }

    /// Calculate realized spread (post-trade)
    pub fn calculate_realized_spread(
        &self,
        trade_price: f64,
        midpoint_pre: f64,
        midpoint_post: f64,
    ) -> f64 {
        2.0 * (trade_price - midpoint_pre).signum() * (midpoint_post - midpoint_pre)
    }

    /// Calculate book pressure
    /// Ratio of bid volume to ask volume at top N levels
    pub fn calculate_book_pressure(&self, bid_volumes: &[f64], ask_volumes: &[f64]) -> f64 {
        let total_bid: f64 = bid_volumes.iter().sum();
        let total_ask: f64 = ask_volumes.iter().sum();

        if total_ask > 0.0 {
            total_bid / total_ask
        } else {
            f64::MAX
        }
    }

    /// Calculate book slope (price impact)
    /// Measures how much price moves per unit of volume
    pub fn calculate_book_slope(
        &self,
        prices: &[f64],
        volumes: &[f64],
        is_bid: bool,
    ) -> f64 {
        if prices.len() < 2 || volumes.is_empty() {
            return 0.0;
        }

        let cumulative_volume: Vec<f64> = volumes
            .iter()
            .scan(0.0, |acc, &v| {
                *acc += v;
                Some(*acc)
            })
            .collect();

        let total_volume = cumulative_volume.last().copied().unwrap_or(0.0);
        if total_volume == 0.0 {
            return 0.0;
        }

        // Calculate price change from best to worst level
        let price_change = if is_bid {
            prices[0] - prices[prices.len() - 1]
        } else {
            prices[prices.len() - 1] - prices[0]
        };

        price_change / total_volume
    }

    /// Calculate exponential moving average (EMA)
    pub fn calculate_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() || period == 0 {
            return Vec::new();
        }

        let alpha = 2.0 / (period + 1) as f64;
        let mut ema_values = Vec::with_capacity(prices.len());

        // Start with simple moving average
        let initial_sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;
        ema_values.push(initial_sma);

        // Calculate EMA
        for &price in prices.iter().skip(period) {
            let prev_ema = *ema_values.last().unwrap();
            let new_ema = alpha * price + (1.0 - alpha) * prev_ema;
            ema_values.push(new_ema);
        }

        ema_values
    }

    /// Calculate Relative Strength Index (RSI)
    pub fn calculate_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return Vec::new();
        }

        let mut rsi_values = Vec::new();
        let mut gains = Vec::new();
        let mut losses = Vec::new();

        // Calculate price changes
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(change.abs());
            }
        }

        // Calculate RSI
        for i in period..gains.len() {
            let avg_gain: f64 = gains[i - period..i].iter().sum::<f64>() / period as f64;
            let avg_loss: f64 = losses[i - period..i].iter().sum::<f64>() / period as f64;

            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                let rs = avg_gain / avg_loss;
                100.0 - (100.0 / (1.0 + rs))
            };

            rsi_values.push(rsi);
        }

        rsi_values
    }

    /// Calculate volatility (standard deviation of returns)
    pub fn calculate_volatility(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return Vec::new();
        }

        let mut volatilities = Vec::new();

        // Calculate log returns
        let mut returns = Vec::new();
        for i in 1..prices.len() {
            if prices[i - 1] > 0.0 {
                let log_return = (prices[i] / prices[i - 1]).ln();
                returns.push(log_return);
            }
        }

        // Calculate rolling standard deviation
        for i in period..=returns.len() {
            let window = &returns[i - period..i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;
            volatilities.push(variance.sqrt());
        }

        volatilities
    }
}

impl Default for MarketDataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap() {
        let processor = MarketDataProcessor::new();
        let ticks = vec![(100.0, 10.0), (101.0, 20.0), (102.0, 30.0)];
        let vwap = processor.calculate_vwap(&ticks);
        assert!((vwap - 101.0).abs() < 0.1);
    }

    #[test]
    fn test_ofi() {
        let processor = MarketDataProcessor::new();
        let bids = vec![100.0, 80.0, 60.0];
        let asks = vec![50.0, 40.0, 30.0];
        let ofi = processor.calculate_ofi(&bids, &asks);
        assert!(ofi > 0.0); // More bid pressure
    }

    #[test]
    fn test_microprice() {
        let processor = MarketDataProcessor::new();
        let microprice = processor.calculate_microprice(100.0, 101.0, 50.0, 50.0);
        assert!((microprice - 100.5).abs() < 0.01);
    }
}
