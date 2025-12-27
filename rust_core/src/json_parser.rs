/// High-performance JSON parsing for WebSocket messages
/// 
/// Uses simd-json for 10x faster parsing than standard serde_json
/// Specifically optimized for Finnhub, Alpaca, and Polygon message formats
/// 
/// Performance targets:
/// - Parse trade message: <1μs
/// - Parse quote message: <1μs
/// - Batch parse 1000 messages: <1ms

use simd_json::prelude::*;
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;
use parking_lot::RwLock;

// ============================================================================
// MESSAGE TYPES
// ============================================================================

/// Finnhub trade message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FinnhubTrade {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "p")]
    pub price: f64,
    #[serde(rename = "v")]
    pub volume: f64,
    #[serde(rename = "t")]
    pub timestamp: i64,
    #[serde(rename = "c")]
    pub conditions: Option<Vec<String>>,
}

/// Finnhub WebSocket message envelope
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FinnhubMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: Option<Vec<FinnhubTrade>>,
}

/// Alpaca trade message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AlpacaTrade {
    #[serde(rename = "S")]
    pub symbol: String,
    #[serde(rename = "p")]
    pub price: f64,
    #[serde(rename = "s")]
    pub size: u64,
    #[serde(rename = "t")]
    pub timestamp: String,
    #[serde(rename = "x")]
    pub exchange: String,
    #[serde(rename = "c")]
    pub conditions: Option<Vec<String>>,
}

/// Alpaca quote message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AlpacaQuote {
    #[serde(rename = "S")]
    pub symbol: String,
    #[serde(rename = "bp")]
    pub bid_price: f64,
    #[serde(rename = "bs")]
    pub bid_size: u64,
    #[serde(rename = "ap")]
    pub ask_price: f64,
    #[serde(rename = "as")]
    pub ask_size: u64,
    #[serde(rename = "t")]
    pub timestamp: String,
}

/// Polygon trade message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PolygonTrade {
    #[serde(rename = "sym")]
    pub symbol: String,
    #[serde(rename = "p")]
    pub price: f64,
    #[serde(rename = "s")]
    pub size: u64,
    #[serde(rename = "t")]
    pub timestamp: i64,
    #[serde(rename = "x")]
    pub exchange: i32,
    #[serde(rename = "c")]
    pub conditions: Option<Vec<i32>>,
}

/// Unified internal trade representation
#[derive(Debug, Clone)]
pub struct ParsedTrade {
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub timestamp_ms: i64,
    pub exchange: String,
    pub source: TradeSource,
}

#[derive(Debug, Clone, Copy)]
pub enum TradeSource {
    Finnhub,
    Alpaca,
    Polygon,
}

// ============================================================================
// SIMD JSON PARSER
// ============================================================================

/// High-performance WebSocket message parser
pub struct FastJsonParser {
    // Buffer for simd-json (requires mutable buffer)
    buffer: Vec<u8>,
}

impl FastJsonParser {
    pub fn new() -> Self {
        FastJsonParser {
            buffer: Vec::with_capacity(4096),
        }
    }

    /// Parse Finnhub WebSocket message
    /// Input: raw JSON bytes
    /// Output: Vec of parsed trades
    pub fn parse_finnhub(&mut self, json: &[u8]) -> Result<Vec<ParsedTrade>, String> {
        // Copy to mutable buffer (simd-json requirement)
        self.buffer.clear();
        self.buffer.extend_from_slice(json);
        
        // Parse with simd-json
        let msg: FinnhubMessage = simd_json::from_slice(&mut self.buffer)
            .map_err(|e| format!("Parse error: {}", e))?;
        
        if msg.msg_type != "trade" {
            return Ok(Vec::new());
        }
        
        let trades = msg.data.unwrap_or_default();
        
        Ok(trades.iter().map(|t| ParsedTrade {
            symbol: t.symbol.clone(),
            price: t.price,
            size: t.volume,
            timestamp_ms: t.timestamp,
            exchange: "FINNHUB".to_string(),
            source: TradeSource::Finnhub,
        }).collect())
    }

    /// Parse Alpaca trade message
    pub fn parse_alpaca_trade(&mut self, json: &[u8]) -> Result<ParsedTrade, String> {
        self.buffer.clear();
        self.buffer.extend_from_slice(json);
        
        let trade: AlpacaTrade = simd_json::from_slice(&mut self.buffer)
            .map_err(|e| format!("Parse error: {}", e))?;
        
        // Parse ISO timestamp to milliseconds
        let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&trade.timestamp)
            .map(|dt| dt.timestamp_millis())
            .unwrap_or(0);
        
        Ok(ParsedTrade {
            symbol: trade.symbol,
            price: trade.price,
            size: trade.size as f64,
            timestamp_ms,
            exchange: trade.exchange,
            source: TradeSource::Alpaca,
        })
    }

    /// Parse Alpaca quote message
    pub fn parse_alpaca_quote(&mut self, json: &[u8]) -> Result<AlpacaQuote, String> {
        self.buffer.clear();
        self.buffer.extend_from_slice(json);
        
        simd_json::from_slice(&mut self.buffer)
            .map_err(|e| format!("Parse error: {}", e))
    }

    /// Parse Polygon trade message
    pub fn parse_polygon(&mut self, json: &[u8]) -> Result<ParsedTrade, String> {
        self.buffer.clear();
        self.buffer.extend_from_slice(json);
        
        let trade: PolygonTrade = simd_json::from_slice(&mut self.buffer)
            .map_err(|e| format!("Parse error: {}", e))?;
        
        // Map exchange ID to name
        let exchange = match trade.exchange {
            1 => "NYSE",
            2 => "NASDAQ",
            4 => "AMEX",
            8 => "ARCA",
            11 => "BATS",
            _ => "UNKNOWN",
        }.to_string();
        
        Ok(ParsedTrade {
            symbol: trade.symbol,
            price: trade.price,
            size: trade.size as f64,
            timestamp_ms: trade.timestamp / 1_000_000, // Nanoseconds to milliseconds
            exchange,
            source: TradeSource::Polygon,
        })
    }

    /// Batch parse multiple messages (optimized for throughput)
    pub fn batch_parse_finnhub(&mut self, messages: &[&[u8]]) -> Vec<ParsedTrade> {
        let mut all_trades = Vec::with_capacity(messages.len() * 10);
        
        for msg in messages {
            if let Ok(trades) = self.parse_finnhub(msg) {
                all_trades.extend(trades);
            }
        }
        
        all_trades
    }
}

impl Default for FastJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PYO3 BINDINGS
// ============================================================================

/// Python wrapper for FastJsonParser
#[pyclass]
pub struct PyFastJsonParser {
    inner: Arc<RwLock<FastJsonParser>>,
}

#[pymethods]
impl PyFastJsonParser {
    #[new]
    fn new() -> Self {
        PyFastJsonParser {
            inner: Arc::new(RwLock::new(FastJsonParser::new())),
        }
    }

    /// Parse Finnhub message and return trades as list of dicts
    fn parse_finnhub(&self, json_str: &str) -> PyResult<Vec<std::collections::HashMap<String, PyObject>>> {
        let mut parser = self.inner.write();
        let trades = parser.parse_finnhub(json_str.as_bytes())
            .map_err(|e| PyValueError::new_err(e))?;
        
        Python::with_gil(|py| {
            Ok(trades.iter().map(|t| {
                let mut map = std::collections::HashMap::new();
                map.insert("symbol".to_string(), t.symbol.clone().into_py(py));
                map.insert("price".to_string(), t.price.into_py(py));
                map.insert("size".to_string(), t.size.into_py(py));
                map.insert("timestamp_ms".to_string(), t.timestamp_ms.into_py(py));
                map.insert("exchange".to_string(), t.exchange.clone().into_py(py));
                map
            }).collect())
        })
    }

    /// Parse Finnhub and return as flat arrays (faster for NumPy)
    /// Returns: (symbols, prices, sizes, timestamps)
    fn parse_finnhub_arrays(
        &self,
        json_str: &str,
    ) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>, Vec<i64>)> {
        let mut parser = self.inner.write();
        let trades = parser.parse_finnhub(json_str.as_bytes())
            .map_err(|e| PyValueError::new_err(e))?;
        
        let symbols: Vec<String> = trades.iter().map(|t| t.symbol.clone()).collect();
        let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
        let sizes: Vec<f64> = trades.iter().map(|t| t.size).collect();
        let timestamps: Vec<i64> = trades.iter().map(|t| t.timestamp_ms).collect();
        
        Ok((symbols, prices, sizes, timestamps))
    }

    /// Parse raw Alpaca trade JSON
    fn parse_alpaca_trade(&self, json_str: &str) -> PyResult<std::collections::HashMap<String, PyObject>> {
        let mut parser = self.inner.write();
        let trade = parser.parse_alpaca_trade(json_str.as_bytes())
            .map_err(|e| PyValueError::new_err(e))?;
        
        Python::with_gil(|py| {
            let mut map = std::collections::HashMap::new();
            map.insert("symbol".to_string(), trade.symbol.into_py(py));
            map.insert("price".to_string(), trade.price.into_py(py));
            map.insert("size".to_string(), trade.size.into_py(py));
            map.insert("timestamp_ms".to_string(), trade.timestamp_ms.into_py(py));
            map.insert("exchange".to_string(), trade.exchange.into_py(py));
            Ok(map)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_finnhub() {
        let mut parser = FastJsonParser::new();
        
        let msg = br#"{"type":"trade","data":[{"s":"AAPL","p":178.50,"v":100,"t":1699876543000}]}"#;
        let trades = parser.parse_finnhub(msg).unwrap();
        
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].symbol, "AAPL");
        assert!((trades[0].price - 178.50).abs() < 0.01);
        assert_eq!(trades[0].size, 100.0);
    }

    #[test]
    fn test_parse_multiple_trades() {
        let mut parser = FastJsonParser::new();
        
        let msg = br#"{"type":"trade","data":[
            {"s":"AAPL","p":178.50,"v":100,"t":1699876543000},
            {"s":"AAPL","p":178.51,"v":200,"t":1699876543001},
            {"s":"MSFT","p":370.25,"v":50,"t":1699876543002}
        ]}"#;
        
        let trades = parser.parse_finnhub(msg).unwrap();
        
        assert_eq!(trades.len(), 3);
        assert_eq!(trades[0].symbol, "AAPL");
        assert_eq!(trades[2].symbol, "MSFT");
    }

    #[test]
    fn test_non_trade_message() {
        let mut parser = FastJsonParser::new();
        
        let msg = br#"{"type":"ping"}"#;
        let trades = parser.parse_finnhub(msg).unwrap();
        
        assert_eq!(trades.len(), 0);
    }
}
