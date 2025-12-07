-- CIFT Markets - QuestDB Initialization
-- Time-series database schema for market data (ticks, bars, order flow)
-- QuestDB optimized for financial time-series data

-- ============================================================================
-- TICK DATA TABLE (Raw Market Data)
-- ============================================================================
-- High-performance tick storage with designated timestamp column
-- Uses QuestDB's native time partitioning for optimal performance

CREATE TABLE IF NOT EXISTS ticks (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 500 CACHE,
    price DOUBLE,
    volume LONG,
    bid DOUBLE,
    ask DOUBLE,
    exchange SYMBOL CAPACITY 50 CACHE,
    conditions SYMBOL CAPACITY 100 CACHE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;

-- Add index for fast symbol lookups
CREATE INDEX IF NOT EXISTS idx_ticks_symbol ON ticks (symbol);

-- ============================================================================
-- OHLCV BARS TABLE (Aggregated Candles)
-- ============================================================================
-- Pre-aggregated bars for faster chart loading
-- Generated from ticks using SAMPLE BY queries

CREATE TABLE IF NOT EXISTS ohlcv_bars (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 500 CACHE,
    timeframe SYMBOL CAPACITY 20 CACHE,  -- 1m, 5m, 15m, 1h, 1d
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume LONG,
    trade_count INT,
    vwap DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_bars (symbol, timeframe);

-- ============================================================================
-- ORDER BOOK SNAPSHOTS (Level 2 Data)
-- ============================================================================
-- L2 order book data for market microstructure analysis

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 500 CACHE,
    side SYMBOL CAPACITY 2 CACHE,  -- 'bid' or 'ask'
    level INT,
    price DOUBLE,
    size LONG,
    exchange SYMBOL CAPACITY 50 CACHE
) TIMESTAMP(timestamp) PARTITION BY HOUR WAL;

-- ============================================================================
-- TRADE EXECUTIONS (Order Flow)
-- ============================================================================
-- Individual trade executions for Hawkes process modeling

CREATE TABLE IF NOT EXISTS trade_executions (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 500 CACHE,
    price DOUBLE,
    size LONG,
    side SYMBOL CAPACITY 2 CACHE,  -- 'buy' or 'sell'
    trade_id STRING,
    exchange SYMBOL CAPACITY 50 CACHE,
    aggressive_side SYMBOL CAPACITY 2 CACHE  -- For order flow toxicity
) TIMESTAMP(timestamp) PARTITION BY HOUR WAL;

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_executions (symbol);

-- ============================================================================
-- SAMPLE DATA INSERTION (For Development/Testing)
-- ============================================================================
-- Insert sample market data for testing charts
-- This will be replaced by real market data feed in production

-- Sample ticks for popular symbols
INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions) VALUES
    (now() - INTERVAL '30' DAY, 'AAPL', 150.00, 1000, 149.98, 150.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '29' DAY, 'AAPL', 151.50, 1200, 151.48, 151.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '28' DAY, 'AAPL', 149.75, 900, 149.73, 149.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '27' DAY, 'AAPL', 152.00, 1100, 151.98, 152.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '26' DAY, 'AAPL', 153.25, 1300, 153.23, 153.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '25' DAY, 'AAPL', 152.50, 1050, 152.48, 152.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '24' DAY, 'AAPL', 154.00, 1400, 153.98, 154.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '23' DAY, 'AAPL', 155.50, 1500, 155.48, 155.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '22' DAY, 'AAPL', 154.75, 1250, 154.73, 154.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '21' DAY, 'AAPL', 156.00, 1600, 155.98, 156.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '20' DAY, 'AAPL', 157.25, 1700, 157.23, 157.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '19' DAY, 'AAPL', 156.50, 1450, 156.48, 156.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '18' DAY, 'AAPL', 158.00, 1800, 157.98, 158.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '17' DAY, 'AAPL', 159.50, 1900, 159.48, 159.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '16' DAY, 'AAPL', 158.75, 1650, 158.73, 158.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '15' DAY, 'AAPL', 160.00, 2000, 159.98, 160.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '14' DAY, 'AAPL', 161.25, 2100, 161.23, 161.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '13' DAY, 'AAPL', 160.50, 1850, 160.48, 160.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '12' DAY, 'AAPL', 162.00, 2200, 161.98, 162.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '11' DAY, 'AAPL', 163.50, 2300, 163.48, 163.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '10' DAY, 'AAPL', 162.75, 2050, 162.73, 162.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '9' DAY, 'AAPL', 164.00, 2400, 163.98, 164.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '8' DAY, 'AAPL', 165.50, 2500, 165.48, 165.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '7' DAY, 'AAPL', 164.75, 2250, 164.73, 164.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '6' DAY, 'AAPL', 166.00, 2600, 165.98, 166.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '5' DAY, 'AAPL', 167.25, 2700, 167.23, 167.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '4' DAY, 'AAPL', 166.50, 2450, 166.48, 166.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '3' DAY, 'AAPL', 168.00, 2800, 167.98, 168.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '2' DAY, 'AAPL', 169.50, 2900, 169.48, 169.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '1' DAY, 'AAPL', 168.75, 2650, 168.73, 168.77, 'NASDAQ', 'REGULAR'),
    (now(), 'AAPL', 170.00, 3000, 169.98, 170.02, 'NASDAQ', 'REGULAR');

-- Sample ticks for MSFT
INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions) VALUES
    (now() - INTERVAL '30' DAY, 'MSFT', 300.00, 800, 299.98, 300.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '29' DAY, 'MSFT', 302.50, 850, 302.48, 302.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '28' DAY, 'MSFT', 298.75, 750, 298.73, 298.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '27' DAY, 'MSFT', 305.00, 900, 304.98, 305.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '26' DAY, 'MSFT', 307.25, 950, 307.23, 307.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '25' DAY, 'MSFT', 305.50, 800, 305.48, 305.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '24' DAY, 'MSFT', 310.00, 1000, 309.98, 310.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '23' DAY, 'MSFT', 312.50, 1050, 312.48, 312.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '22' DAY, 'MSFT', 310.75, 900, 310.73, 310.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '21' DAY, 'MSFT', 315.00, 1100, 314.98, 315.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '20' DAY, 'MSFT', 317.25, 1150, 317.23, 317.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '19' DAY, 'MSFT', 315.50, 1000, 315.48, 315.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '18' DAY, 'MSFT', 320.00, 1200, 319.98, 320.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '17' DAY, 'MSFT', 322.50, 1250, 322.48, 322.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '16' DAY, 'MSFT', 320.75, 1100, 320.73, 320.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '15' DAY, 'MSFT', 325.00, 1300, 324.98, 325.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '14' DAY, 'MSFT', 327.25, 1350, 327.23, 327.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '13' DAY, 'MSFT', 325.50, 1200, 325.48, 325.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '12' DAY, 'MSFT', 330.00, 1400, 329.98, 330.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '11' DAY, 'MSFT', 332.50, 1450, 332.48, 332.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '10' DAY, 'MSFT', 330.75, 1300, 330.73, 330.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '9' DAY, 'MSFT', 335.00, 1500, 334.98, 335.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '8' DAY, 'MSFT', 337.50, 1550, 337.48, 337.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '7' DAY, 'MSFT', 335.75, 1400, 335.73, 335.77, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '6' DAY, 'MSFT', 340.00, 1600, 339.98, 340.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '5' DAY, 'MSFT', 342.25, 1650, 342.23, 342.27, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '4' DAY, 'MSFT', 340.50, 1500, 340.48, 340.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '3' DAY, 'MSFT', 345.00, 1700, 344.98, 345.02, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '2' DAY, 'MSFT', 347.50, 1750, 347.48, 347.52, 'NASDAQ', 'REGULAR'),
    (now() - INTERVAL '1' DAY, 'MSFT', 345.75, 1600, 345.73, 345.77, 'NASDAQ', 'REGULAR'),
    (now(), 'MSFT', 350.00, 1800, 349.98, 350.02, 'NASDAQ', 'REGULAR');

-- Additional symbols for testing
INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions) VALUES
    (now(), 'GOOGL', 2800.00, 500, 2799.50, 2800.50, 'NASDAQ', 'REGULAR'),
    (now(), 'AMZN', 3100.00, 600, 3099.50, 3100.50, 'NASDAQ', 'REGULAR'),
    (now(), 'TSLA', 250.00, 2000, 249.95, 250.05, 'NASDAQ', 'REGULAR'),
    (now(), 'META', 320.00, 700, 319.90, 320.10, 'NASDAQ', 'REGULAR'),
    (now(), 'NVDA', 450.00, 1500, 449.90, 450.10, 'NASDAQ', 'REGULAR');

-- ============================================================================
-- SAMPLE AGGREGATED BARS (For immediate testing)
-- ============================================================================
-- Pre-calculate some daily bars using SAMPLE BY
-- In production, these would be generated by a scheduled job

-- This query would normally be run periodically:
-- INSERT INTO ohlcv_bars 
-- SELECT 
--   timestamp,
--   symbol,
--   '1d' as timeframe,
--   first(price) as open,
--   max(price) as high,
--   min(price) as low,
--   last(price) as close,
--   sum(volume) as volume,
--   count(*) as trade_count,
--   sum(price * volume) / sum(volume) as vwap
-- FROM ticks
-- WHERE timestamp >= dateadd('d', -30, now())
-- SAMPLE BY 1d ALIGN TO CALENDAR;

-- ============================================================================
-- PERFORMANCE NOTES
-- ============================================================================
-- 1. SYMBOL type: Optimized for low-cardinality string data (symbols)
-- 2. TIMESTAMP designated column: Enables time-based partitioning
-- 3. WAL (Write-Ahead Log): Ensures data durability
-- 4. PARTITION BY DAY/HOUR: Automatic time-based partitioning for query speed
-- 5. Indexes on symbol columns: Fast lookups by ticker
--
-- Query Performance Targets:
-- - Tick query (1 day): <10ms
-- - OHLCV aggregation (30 days): <50ms
-- - SAMPLE BY query (1 year): <100ms
--
-- ============================================================================
-- PRODUCTION DATA INGESTION
-- ============================================================================
-- In production, data will be ingested via:
-- 1. REST API (HTTP): For historical bulk loads
-- 2. InfluxDB Line Protocol: For real-time streaming (9009)
-- 3. PostgreSQL wire protocol: For application integration (8812)
-- 4. CSV import: For backtesting data
--
-- Example real-time ingestion:
-- curl -F data=@ticks.csv http://localhost:9000/imp?name=ticks
