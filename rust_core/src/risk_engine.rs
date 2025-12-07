/// High-performance risk engine for trading validation
/// Performs sub-microsecond risk checks
pub struct RiskEngine {
    max_position_size: f64,
    max_notional: f64,
    max_leverage: f64,
}

impl RiskEngine {
    pub fn new(max_position_size: f64, max_notional: f64, max_leverage: f64) -> Self {
        RiskEngine {
            max_position_size,
            max_notional,
            max_leverage,
        }
    }

    /// Check if order passes all risk checks
    /// Returns: (passed: bool, reason: String)
    pub fn check_order(
        &self,
        _symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        current_position: f64,
        account_value: f64,
    ) -> (bool, String) {
        // Position size check
        let new_position = match side {
            "buy" => current_position + quantity,
            "sell" => current_position - quantity,
            _ => current_position,
        };

        if new_position.abs() > self.max_position_size {
            return (
                false,
                format!(
                    "Position size {} exceeds max {}",
                    new_position.abs(),
                    self.max_position_size
                ),
            );
        }

        // Notional value check
        let order_notional = quantity * price;
        if order_notional > self.max_notional {
            return (
                false,
                format!(
                    "Order notional {} exceeds max {}",
                    order_notional, self.max_notional
                ),
            );
        }

        // Leverage check
        let position_notional = new_position.abs() * price;
        let leverage = if account_value > 0.0 {
            position_notional / account_value
        } else {
            f64::MAX
        };

        if leverage > self.max_leverage {
            return (
                false,
                format!("Leverage {} exceeds max {}", leverage, self.max_leverage),
            );
        }

        // Buying power check (simplified)
        if side == "buy" {
            let required_capital = order_notional;
            if required_capital > account_value * 2.0 {
                // Allow 2x margin
                return (false, "Insufficient buying power".to_string());
            }
        }

        (true, "OK".to_string())
    }

    /// Calculate maximum order size allowed given current constraints
    pub fn max_order_size(
        &self,
        _symbol: &str,
        side: &str,
        price: f64,
        current_position: f64,
        account_value: f64,
    ) -> f64 {
        // Max based on position size limit
        let max_from_position = match side {
            "buy" => self.max_position_size - current_position,
            "sell" => self.max_position_size + current_position,
            _ => 0.0,
        };

        // Max based on notional limit
        let max_from_notional = if price > 0.0 {
            self.max_notional / price
        } else {
            0.0
        };

        // Max based on leverage limit
        let max_from_leverage = if price > 0.0 {
            (account_value * self.max_leverage) / price
        } else {
            0.0
        };

        // Max based on buying power (2x margin)
        let max_from_capital = if side == "buy" && price > 0.0 {
            (account_value * 2.0) / price
        } else {
            f64::MAX
        };

        // Return the minimum of all constraints
        max_from_position
            .min(max_from_notional)
            .min(max_from_leverage)
            .min(max_from_capital)
            .max(0.0)
    }

    /// Calculate portfolio-level risk metrics
    pub fn calculate_var(
        &self,
        positions: &[(String, f64, f64)], // (symbol, quantity, price)
        volatilities: &std::collections::HashMap<String, f64>,
        confidence_level: f64,
    ) -> f64 {
        // Simplified VaR calculation (parametric VaR)
        let total_value: f64 = positions
            .iter()
            .map(|(_, qty, price)| qty.abs() * price)
            .sum();

        // Average volatility (simplified, should use correlation matrix)
        let avg_volatility: f64 = positions
            .iter()
            .filter_map(|(symbol, _, _)| volatilities.get(symbol))
            .sum::<f64>()
            / positions.len() as f64;

        // Z-score for confidence level (e.g., 1.645 for 95%)
        let z_score = match confidence_level {
            0.95 => 1.645,
            0.99 => 2.326,
            _ => 1.96,
        };

        total_value * avg_volatility * z_score
    }

    /// Check if portfolio is within risk limits
    pub fn check_portfolio_risk(
        &self,
        positions: &[(String, f64, f64)],
        account_value: f64,
    ) -> (bool, String) {
        let total_notional: f64 = positions
            .iter()
            .map(|(_, qty, price)| qty.abs() * price)
            .sum();

        let portfolio_leverage = if account_value > 0.0 {
            total_notional / account_value
        } else {
            f64::MAX
        };

        if portfolio_leverage > self.max_leverage {
            return (
                false,
                format!(
                    "Portfolio leverage {} exceeds max {}",
                    portfolio_leverage, self.max_leverage
                ),
            );
        }

        // Check concentration risk (no single position > 30% of portfolio)
        for (symbol, qty, price) in positions {
            let position_notional = qty.abs() * price;
            let concentration = if total_notional > 0.0 {
                position_notional / total_notional
            } else {
                0.0
            };

            if concentration > 0.3 {
                return (
                    false,
                    format!(
                        "Position {} concentration {:.1}% exceeds 30%",
                        symbol,
                        concentration * 100.0
                    ),
                );
            }
        }

        (true, "OK".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_checks() {
        let engine = RiskEngine::new(1000.0, 100_000.0, 5.0);

        // Valid order
        let (passed, _) = engine.check_order("AAPL", "buy", 100.0, 150.0, 0.0, 50_000.0);
        assert!(passed);

        // Exceeds position size
        let (passed, _) = engine.check_order("AAPL", "buy", 1500.0, 150.0, 0.0, 500_000.0);
        assert!(!passed);

        // Exceeds notional
        let (passed, _) = engine.check_order("AAPL", "buy", 1000.0, 200.0, 0.0, 500_000.0);
        assert!(!passed);
    }

    #[test]
    fn test_max_order_size() {
        let engine = RiskEngine::new(1000.0, 100_000.0, 5.0);

        let max_size = engine.max_order_size("AAPL", "buy", 150.0, 0.0, 50_000.0);
        assert!(max_size > 0.0);
        assert!(max_size <= 1000.0); // Limited by max position size
    }
}
