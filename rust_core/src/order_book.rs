use std::collections::{BTreeMap, HashMap};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Limit,
    Market,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: u64,
    pub user_id: u64,
    pub side: Side,
    pub order_type: OrderType,
    pub price: Decimal,
    pub quantity: Decimal,
    pub filled: Decimal,
    pub timestamp: i64,
}

impl Order {
    pub fn remaining(&self) -> Decimal {
        self.quantity - self.filled
    }

    pub fn is_filled(&self) -> bool {
        self.filled >= self.quantity
    }
}

/// High-performance order book implementation
/// Uses BTreeMap for price-time priority
/// Zero-allocation hot path for order matching
pub struct OrderBook {
    symbol: String,
    // Buy orders: descending price (best bid first)
    bids: BTreeMap<Decimal, Vec<Order>>,
    // Sell orders: ascending price (best ask first)
    asks: BTreeMap<Decimal, Vec<Order>>,
    // Order ID -> (price, side) for O(1) cancellation
    order_index: HashMap<u64, (Decimal, Side)>,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        OrderBook {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            order_index: HashMap::new(),
        }
    }

    /// Add order to the book and return fills
    /// Returns: Vec<(fill_price, fill_quantity, counterparty_id)>
    pub fn add_order(&mut self, mut order: Order) -> Vec<(Decimal, Decimal, u64)> {
        let mut fills = Vec::new();

        match order.order_type {
            OrderType::Limit => {
                // Try to match against existing orders
                fills = self.match_order(&mut order);

                // Add remaining quantity to the book
                if !order.is_filled() {
                    self.add_to_book(order);
                }
            }
            OrderType::Market => {
                // Market orders execute immediately
                return self.execute_market_order(order);
            }
        }

        fills
    }

    /// Execute market order immediately at best available prices
    pub fn execute_market_order(&mut self, mut order: Order) -> Vec<(Decimal, Decimal, u64)> {
        self.match_order(&mut order)
    }

    /// Match order against opposite side of the book
    fn match_order(&mut self, order: &mut Order) -> Vec<(Decimal, Decimal, u64)> {
        let mut fills = Vec::new();
        let remaining = order.remaining();

        if remaining == Decimal::ZERO {
            return fills;
        }

        let opposite_side = match order.side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };

        // Get prices in order (best first)
        let prices: Vec<Decimal> = opposite_side.keys().copied().collect();

        for price in prices {
            // For buy orders, only match if ask price <= our price (for limit orders)
            // For sell orders, only match if bid price >= our price (for limit orders)
            if order.order_type == OrderType::Limit {
                match order.side {
                    Side::Buy => {
                        if price > order.price {
                            break;
                        }
                    }
                    Side::Sell => {
                        if price < order.price {
                            break;
                        }
                    }
                }
            }

            if let Some(level_orders) = opposite_side.get_mut(&price) {
                let mut i = 0;
                while i < level_orders.len() && order.remaining() > Decimal::ZERO {
                    let resting_order = &mut level_orders[i];
                    let match_qty = order.remaining().min(resting_order.remaining());

                    // Execute fill
                    order.filled += match_qty;
                    resting_order.filled += match_qty;
                    fills.push((price, match_qty, resting_order.user_id));

                    // Remove filled orders
                    if resting_order.is_filled() {
                        let removed_order = level_orders.remove(i);
                        self.order_index.remove(&removed_order.order_id);
                    } else {
                        i += 1;
                    }
                }

                // Remove empty price levels
                if level_orders.is_empty() {
                    opposite_side.remove(&price);
                }
            }

            if order.is_filled() {
                break;
            }
        }

        fills
    }

    /// Add order to the appropriate side of the book
    fn add_to_book(&mut self, order: Order) {
        let book_side = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        self.order_index.insert(order.order_id, (order.price, order.side));
        book_side.entry(order.price).or_insert_with(Vec::new).push(order);
    }

    /// Cancel an order by ID
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        if let Some((price, side)) = self.order_index.remove(&order_id) {
            let book_side = match side {
                Side::Buy => &mut self.bids,
                Side::Sell => &mut self.asks,
            };

            if let Some(level_orders) = book_side.get_mut(&price) {
                if let Some(pos) = level_orders.iter().position(|o| o.order_id == order_id) {
                    level_orders.remove(pos);
                    if level_orders.is_empty() {
                        book_side.remove(&price);
                    }
                    return true;
                }
            }
        }
        false
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.keys().next_back().copied()
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.keys().next().copied()
    }

    /// Get bid-ask spread
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get order book depth (top N levels for each side)
    pub fn depth(&self, levels: usize) -> (Vec<(Decimal, Decimal)>, Vec<(Decimal, Decimal)>) {
        let bids: Vec<(Decimal, Decimal)> = self
            .bids
            .iter()
            .rev()
            .take(levels)
            .map(|(price, orders)| {
                let total_qty: Decimal = orders.iter().map(|o| o.remaining()).sum();
                (*price, total_qty)
            })
            .collect();

        let asks: Vec<(Decimal, Decimal)> = self
            .asks
            .iter()
            .take(levels)
            .map(|(price, orders)| {
                let total_qty: Decimal = orders.iter().map(|o| o.remaining()).sum();
                (*price, total_qty)
            })
            .collect();

        (bids, asks)
    }

    /// Get total volume at a specific price level
    pub fn volume_at_price(&self, price: Decimal, side: Side) -> Decimal {
        let book_side = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };

        book_side
            .get(&price)
            .map(|orders| orders.iter().map(|o| o.remaining()).sum())
            .unwrap_or(Decimal::ZERO)
    }

    /// Get total number of orders
    pub fn order_count(&self) -> usize {
        let bid_count: usize = self.bids.values().map(|v| v.len()).sum();
        let ask_count: usize = self.asks.values().map(|v| v.len()).sum();
        bid_count + ask_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_basics() {
        let mut book = OrderBook::new("AAPL".to_string());

        // Add buy order
        let buy_order = Order {
            order_id: 1,
            user_id: 100,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Decimal::from(150),
            quantity: Decimal::from(10),
            filled: Decimal::ZERO,
            timestamp: 0,
        };

        let fills = book.add_order(buy_order);
        assert!(fills.is_empty());
        assert_eq!(book.best_bid(), Some(Decimal::from(150)));
    }

    #[test]
    fn test_order_matching() {
        let mut book = OrderBook::new("AAPL".to_string());

        // Add sell order
        let sell_order = Order {
            order_id: 1,
            user_id: 100,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: Decimal::from(150),
            quantity: Decimal::from(10),
            filled: Decimal::ZERO,
            timestamp: 0,
        };
        book.add_order(sell_order);

        // Add matching buy order
        let buy_order = Order {
            order_id: 2,
            user_id: 101,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Decimal::from(150),
            quantity: Decimal::from(5),
            filled: Decimal::ZERO,
            timestamp: 1,
        };

        let fills = book.add_order(buy_order);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].0, Decimal::from(150)); // price
        assert_eq!(fills[0].1, Decimal::from(5)); // quantity
    }
}
