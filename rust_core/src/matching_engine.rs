use crate::order_book::{OrderBook, Order, Side};
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// Multi-symbol matching engine
/// Manages order books for multiple symbols with concurrent access
pub struct MatchingEngine {
    /// Symbol -> OrderBook mapping
    books: HashMap<String, Arc<RwLock<OrderBook>>>,
}

impl MatchingEngine {
    pub fn new() -> Self {
        MatchingEngine {
            books: HashMap::new(),
        }
    }

    /// Get or create order book for symbol
    pub fn get_or_create_book(&mut self, symbol: String) -> Arc<RwLock<OrderBook>> {
        self.books
            .entry(symbol.clone())
            .or_insert_with(|| Arc::new(RwLock::new(OrderBook::new(symbol))))
            .clone()
    }

    /// Process order for a symbol
    pub fn process_order(&mut self, symbol: String, order: Order) -> Vec<(rust_decimal::Decimal, rust_decimal::Decimal, u64)> {
        let book = self.get_or_create_book(symbol);
        let mut book_guard = book.write();
        book_guard.add_order(order)
    }

    /// Cancel order for a symbol
    pub fn cancel_order(&mut self, symbol: &str, order_id: u64) -> bool {
        if let Some(book) = self.books.get(symbol) {
            let mut book_guard = book.write();
            book_guard.cancel_order(order_id)
        } else {
            false
        }
    }

    /// Get best bid/ask for symbol
    pub fn best_prices(&self, symbol: &str) -> Option<(Option<rust_decimal::Decimal>, Option<rust_decimal::Decimal>)> {
        self.books.get(symbol).map(|book| {
            let book_guard = book.read();
            (book_guard.best_bid(), book_guard.best_ask())
        })
    }

    /// Get order book depth for symbol
    pub fn get_depth(&self, symbol: &str, levels: usize) -> Option<(Vec<(rust_decimal::Decimal, rust_decimal::Decimal)>, Vec<(rust_decimal::Decimal, rust_decimal::Decimal)>)> {
        self.books.get(symbol).map(|book| {
            let book_guard = book.read();
            book_guard.depth(levels)
        })
    }

    /// Get number of active symbols
    pub fn symbol_count(&self) -> usize {
        self.books.len()
    }
}

impl Default for MatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order_book::OrderType;
    use rust_decimal::Decimal;

    #[test]
    fn test_multi_symbol_matching() {
        let mut engine = MatchingEngine::new();

        // Add order for AAPL
        let aapl_order = Order {
            order_id: 1,
            user_id: 100,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Decimal::from(150),
            quantity: Decimal::from(10),
            filled: Decimal::ZERO,
            timestamp: 0,
        };

        engine.process_order("AAPL".to_string(), aapl_order);

        // Add order for GOOGL
        let googl_order = Order {
            order_id: 2,
            user_id: 101,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: Decimal::from(2800),
            quantity: Decimal::from(5),
            filled: Decimal::ZERO,
            timestamp: 1,
        };

        engine.process_order("GOOGL".to_string(), googl_order);

        assert_eq!(engine.symbol_count(), 2);

        // Check best prices
        if let Some((bid, _ask)) = engine.best_prices("AAPL") {
            assert_eq!(bid, Some(Decimal::from(150)));
        }
    }
}
