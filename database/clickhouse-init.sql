-- ClickHouse Initialization Script for CIFT Markets Analytics Database
-- 100x faster complex queries compared to traditional time-series databases

-- Create database
CREATE DATABASE IF NOT EXISTS cift_analytics;

-- Use the analytics database
USE cift_analytics;

-- =====================================================
-- MARKET DATA TABLES (Optimized for Analytics)
-- =====================================================

-- Historical tick data (compressed, columnar storage)
CREATE TABLE IF NOT EXISTS ticks_analytics
(
    timestamp DateTime64(6, 'UTC'),
    symbol LowCardinality(String),
    price Decimal64(4),
    volume UInt32,
    bid Decimal64(4),
    ask Decimal64(4),
    bid_size UInt32,
    ask_size UInt32,
    exchange LowCardinality(String),
    conditions Array(String)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;

-- OHLCV bars (pre-aggregated for fast queries)
CREATE TABLE IF NOT EXISTS bars_analytics
(
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    timeframe LowCardinality(String),  -- '1m', '5m', '15m', '1h', '1d'
    open Decimal64(4),
    high Decimal64(4),
    low Decimal64(4),
    close Decimal64(4),
    volume UInt64,
    vwap Decimal64(4),
    trade_count UInt32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timeframe, timestamp)
SETTINGS index_granularity = 8192;

-- Order book snapshots (L2 depth)
CREATE TABLE IF NOT EXISTS order_book_snapshots
(
    timestamp DateTime64(6, 'UTC'),
    symbol LowCardinality(String),
    side Enum8('bid' = 1, 'ask' = 2),
    price Decimal64(4),
    quantity Decimal64(4),
    order_count UInt16,
    level UInt8  -- Price level (1 = best, 2 = second best, etc.)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp, side, level)
SETTINGS index_granularity = 8192;

-- =====================================================
-- TRADING ANALYTICS TABLES
-- =====================================================

-- Trade executions (comprehensive fill data)
CREATE TABLE IF NOT EXISTS trade_executions
(
    execution_id UInt64,
    order_id UInt64,
    user_id UInt64,
    symbol LowCardinality(String),
    side Enum8('buy' = 1, 'sell' = 2),
    quantity Decimal64(4),
    price Decimal64(4),
    value Decimal64(2),
    commission Decimal64(4),
    timestamp DateTime64(6, 'UTC'),
    execution_venue LowCardinality(String),
    liquidity_flag Enum8('maker' = 1, 'taker' = 2),
    slippage Decimal64(4)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, symbol, timestamp)
SETTINGS index_granularity = 8192;

-- Position history (closed positions for analysis)
CREATE TABLE IF NOT EXISTS position_history_analytics
(
    position_id UInt64,
    user_id UInt64,
    symbol LowCardinality(String),
    side Enum8('long' = 1, 'short' = 2),
    entry_price Decimal64(4),
    exit_price Decimal64(4),
    quantity Decimal64(4),
    entry_time DateTime64(3, 'UTC'),
    exit_time DateTime64(3, 'UTC'),
    realized_pnl Decimal64(2),
    commission_total Decimal64(4),
    hold_duration_seconds UInt32,
    max_favorable_excursion Decimal64(4),
    max_adverse_excursion Decimal64(4),
    exit_reason LowCardinality(String)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(exit_time)
ORDER BY (user_id, symbol, exit_time)
SETTINGS index_granularity = 8192;

-- =====================================================
-- FEATURE ENGINEERING TABLES
-- =====================================================

-- Technical indicators (pre-calculated)
CREATE TABLE IF NOT EXISTS technical_indicators
(
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    timeframe LowCardinality(String),
    -- Trend indicators
    sma_20 Nullable(Decimal64(4)),
    sma_50 Nullable(Decimal64(4)),
    ema_12 Nullable(Decimal64(4)),
    ema_26 Nullable(Decimal64(4)),
    macd Nullable(Decimal64(4)),
    macd_signal Nullable(Decimal64(4)),
    -- Momentum indicators
    rsi_14 Nullable(Decimal64(2)),
    stoch_k Nullable(Decimal64(2)),
    stoch_d Nullable(Decimal64(2)),
    -- Volatility indicators
    bb_upper Nullable(Decimal64(4)),
    bb_middle Nullable(Decimal64(4)),
    bb_lower Nullable(Decimal64(4)),
    atr_14 Nullable(Decimal64(4)),
    -- Volume indicators
    obv Nullable(Int64),
    vwap Nullable(Decimal64(4))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timeframe, timestamp)
SETTINGS index_granularity = 8192;

-- Order flow features (for ML models)
CREATE TABLE IF NOT EXISTS order_flow_features
(
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    -- Order flow metrics
    ofi Decimal64(6),           -- Order Flow Imbalance
    weighted_ofi Decimal64(6),  -- Distance-weighted OFI
    microprice Decimal64(4),
    bid_ask_spread Decimal64(4),
    effective_spread Decimal64(4),
    -- Book metrics
    book_pressure Decimal64(4),
    book_slope Decimal64(6),
    depth_imbalance Decimal64(4),
    -- Volume metrics
    trade_intensity UInt32,
    buy_volume UInt64,
    sell_volume UInt64,
    -- Price movement
    price_momentum Decimal64(6),
    volatility Decimal64(6)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;

-- =====================================================
-- PERFORMANCE ANALYTICS TABLES
-- =====================================================

-- Strategy performance metrics
CREATE TABLE IF NOT EXISTS strategy_performance
(
    timestamp DateTime64(3, 'UTC'),
    strategy_id LowCardinality(String),
    user_id UInt64,
    -- Returns
    total_return Decimal64(4),
    daily_return Decimal64(4),
    cumulative_return Decimal64(4),
    -- Risk metrics
    sharpe_ratio Decimal64(4),
    sortino_ratio Decimal64(4),
    max_drawdown Decimal64(4),
    volatility Decimal64(4),
    -- Trading stats
    win_rate Decimal64(4),
    profit_factor Decimal64(4),
    avg_win Decimal64(2),
    avg_loss Decimal64(2),
    total_trades UInt32,
    winning_trades UInt32,
    losing_trades UInt32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (strategy_id, user_id, timestamp)
SETTINGS index_granularity = 8192;

-- Account snapshots (daily)
CREATE TABLE IF NOT EXISTS account_snapshots
(
    snapshot_date Date,
    user_id UInt64,
    account_value Decimal64(2),
    cash_balance Decimal64(2),
    position_value Decimal64(2),
    buying_power Decimal64(2),
    margin_used Decimal64(2),
    unrealized_pnl Decimal64(2),
    realized_pnl_today Decimal64(2),
    total_trades_today UInt32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (user_id, snapshot_date)
SETTINGS index_granularity = 8192;

-- =====================================================
-- MATERIALIZED VIEWS FOR REAL-TIME AGGREGATIONS
-- =====================================================

-- 1-minute OHLCV aggregation from ticks
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1m_mv
TO bars_analytics
AS SELECT
    toStartOfMinute(timestamp) AS timestamp,
    symbol,
    '1m' AS timeframe,
    anyLast(price) AS open,
    max(price) AS high,
    min(price) AS low,
    anyLast(price) AS close,
    sum(volume) AS volume,
    avg(price) AS vwap,
    count() AS trade_count
FROM ticks_analytics
GROUP BY timestamp, symbol;

-- Daily trading statistics by symbol
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stats_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, date)
AS SELECT
    toDate(timestamp) AS date,
    symbol,
    count() AS trade_count,
    sum(quantity) AS total_volume,
    sum(value) AS total_value,
    avg(price) AS avg_price,
    max(price) AS high_price,
    min(price) AS low_price,
    sum(commission) AS total_commission
FROM trade_executions
GROUP BY symbol, date;

-- User daily P&L aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS user_daily_pnl_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (user_id, date)
AS SELECT
    toDate(timestamp) AS date,
    user_id,
    sum(CASE WHEN side = 'buy' THEN -value ELSE value END) AS realized_pnl,
    sum(commission) AS total_commission,
    count() AS trade_count,
    countIf(side = 'buy') AS buy_count,
    countIf(side = 'sell') AS sell_count
FROM trade_executions
GROUP BY user_id, toDate(timestamp);

-- =====================================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- =====================================================

-- Secondary indexes (skipped if already exist from previous runs)
-- ALTER TABLE ticks_analytics ADD INDEX IF NOT EXISTS idx_price price TYPE minmax GRANULARITY 4;
-- ALTER TABLE ticks_analytics ADD INDEX IF NOT EXISTS idx_volume volume TYPE minmax GRANULARITY 4;
-- ALTER TABLE trade_executions ADD INDEX IF NOT EXISTS idx_value value TYPE minmax GRANULARITY 4;
-- ALTER TABLE trade_executions ADD INDEX IF NOT EXISTS idx_commission commission TYPE minmax GRANULARITY 4;

-- =====================================================
-- COMPRESSION CODECS FOR STORAGE OPTIMIZATION
-- =====================================================

-- Compression (skipped if already applied from previous runs)
-- ALTER TABLE ticks_analytics MODIFY COLUMN price CODEC(Delta, ZSTD);
-- ALTER TABLE ticks_analytics MODIFY COLUMN bid CODEC(Delta, ZSTD);
-- ALTER TABLE ticks_analytics MODIFY COLUMN ask CODEC(Delta, ZSTD);
-- ALTER TABLE ticks_analytics MODIFY COLUMN volume CODEC(T64, ZSTD);
-- ALTER TABLE bars_analytics MODIFY COLUMN open CODEC(Delta, ZSTD);
-- ALTER TABLE bars_analytics MODIFY COLUMN high CODEC(Delta, ZSTD);
-- ALTER TABLE bars_analytics MODIFY COLUMN low CODEC(Delta, ZSTD);
-- ALTER TABLE bars_analytics MODIFY COLUMN close CODEC(Delta, ZSTD);

-- =====================================================
-- UTILITY FUNCTIONS (Skipped - CREATE OR REPLACE not idempotent)
-- =====================================================

-- Function to calculate Sharpe ratio
-- CREATE OR REPLACE FUNCTION sharpe_ratio AS (returns, risk_free_rate) -> 
--     (avg(returns) - risk_free_rate) / stddevPop(returns);

-- Function to calculate maximum drawdown
-- CREATE OR REPLACE FUNCTION max_drawdown AS (cumulative_returns) ->
--     max(arrayMax(cumulative_returns) - cumulative_returns);

-- =====================================================
-- GRANTS AND PERMISSIONS
-- =====================================================

-- Grant permissions to cift_user (skipped if already granted)
-- GRANT SELECT, INSERT, ALTER, CREATE ON cift_analytics.* TO cift_user;

-- =====================================================
-- SAMPLE QUERIES FOR REFERENCE
-- =====================================================

-- Query 1: Get 1-minute OHLCV for last hour
-- SELECT * FROM bars_analytics 
-- WHERE symbol = 'AAPL' 
--   AND timeframe = '1m' 
--   AND timestamp >= now() - INTERVAL 1 HOUR 
-- ORDER BY timestamp DESC;

-- Query 2: Calculate win rate by user
-- SELECT 
--     user_id,
--     countIf(realized_pnl > 0) / count() as win_rate,
--     avg(realized_pnl) as avg_pnl,
--     sum(realized_pnl) as total_pnl
-- FROM position_history_analytics
-- WHERE exit_time >= today() - INTERVAL 30 DAY
-- GROUP BY user_id;

-- Query 3: Top traded symbols by volume
-- SELECT 
--     symbol,
--     sum(volume) as total_volume,
--     count() as trade_count,
--     avg(price) as avg_price
-- FROM ticks_analytics
-- WHERE timestamp >= now() - INTERVAL 1 DAY
-- GROUP BY symbol
-- ORDER BY total_volume DESC
-- LIMIT 10;
