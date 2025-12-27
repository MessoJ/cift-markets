# Cap'n Proto Schema for CIFT Market Data
# 
# Zero-copy serialization for IPC between services
# 10-100x faster than JSON/Protocol Buffers for read-heavy workloads
#
# Usage:
#   capnp compile -orust:src schema/market_data.capnp

@0xc8f3e7a9d2b14567;

# ============================================================================
# MARKET DATA MESSAGES
# ============================================================================

struct Trade {
  symbol @0 :Text;
  price @1 :Float64;
  size @2 :Float64;
  timestampMs @3 :Int64;
  exchange @4 :Text;
  side @5 :TradeSide;
  tradeId @6 :Text;
}

enum TradeSide {
  unknown @0;
  buy @1;
  sell @2;
}

struct Quote {
  symbol @0 :Text;
  bidPrice @1 :Float64;
  bidSize @2 :Float64;
  askPrice @3 :Float64;
  askSize @4 :Float64;
  timestampMs @5 :Int64;
  exchange @6 :Text;
}

struct Bar {
  symbol @0 :Text;
  open @1 :Float64;
  high @2 :Float64;
  low @3 :Float64;
  close @4 :Float64;
  volume @5 :Float64;
  timestampMs @6 :Int64;
  vwap @7 :Float64;
  tradeCount @8 :UInt32;
}

# ============================================================================
# ORDER BOOK
# ============================================================================

struct PriceLevel {
  price @0 :Float64;
  size @1 :Float64;
  orderCount @2 :UInt32;
}

struct OrderBookSnapshot {
  symbol @0 :Text;
  bids @1 :List(PriceLevel);
  asks @2 :List(PriceLevel);
  timestampMs @3 :Int64;
  sequenceNumber @4 :UInt64;
}

struct OrderBookDelta {
  symbol @0 :Text;
  changes @1 :List(BookChange);
  timestampMs @2 :Int64;
  sequenceNumber @3 :UInt64;
}

struct BookChange {
  side @0 :TradeSide;
  price @1 :Float64;
  size @2 :Float64;  # 0 = remove level
}

# ============================================================================
# FEATURE VECTORS FOR ML
# ============================================================================

struct FeatureVector {
  # Returns
  return1 @0 :Float64;
  return5 @1 :Float64;
  return20 @2 :Float64;
  return60 @3 :Float64;
  
  # Volatility
  volatility20 @4 :Float64;
  volatility60 @5 :Float64;
  volatilityRatio @6 :Float64;
  priceDeviation @7 :Float64;
  
  # Volume
  volume @8 :Float64;
  volumeZscore @9 :Float64;
  volumeMaRatio @10 :Float64;
  
  # Spread
  spread @11 :Float64;
  spreadZscore @12 :Float64;
  spreadMaRatio @13 :Float64;
  
  # Order flow
  ofi @14 :Float64;
  ofiMean @15 :Float64;
  ofiCumulative @16 :Float64;
  imbalance @17 :Float64;
  logPressure @18 :Float64;
  
  # Microprice
  microprice @19 :Float64;
  micropriceDeviation @20 :Float64;
  
  # Technical
  rsi @21 :Float64;
  momentumDivergence @22 :Float64;
  
  # Raw values
  price @23 :Float64;
  mid @24 :Float64;
  bid @25 :Float64;
  ask @26 :Float64;
  tradeIntensity @27 :Float64;
}

struct FeatureBatch {
  symbol @0 :Text;
  features @1 :List(FeatureVector);
  timestampMs @2 :Int64;
}

# ============================================================================
# PREDICTION MESSAGES
# ============================================================================

struct Prediction {
  symbol @0 :Text;
  direction @1 :PredictionDirection;
  probability @2 :Float64;
  expectedReturn @3 :Float64;
  confidence @4 :Float64;
  modelVersion @5 :Text;
  timestampMs @6 :Int64;
}

enum PredictionDirection {
  down @0;
  neutral @1;
  up @2;
}

struct PredictionBatch {
  predictions @0 :List(Prediction);
  batchId @1 :UInt64;
  latencyUs @2 :UInt32;  # Inference latency in microseconds
}

# ============================================================================
# IPC CHANNEL MESSAGES
# ============================================================================

struct MarketDataMessage {
  union {
    trade @0 :Trade;
    quote @1 :Quote;
    bar @2 :Bar;
    bookSnapshot @3 :OrderBookSnapshot;
    bookDelta @4 :OrderBookDelta;
    features @5 :FeatureBatch;
    prediction @6 :PredictionBatch;
    heartbeat @7 :Int64;
  }
}

# ============================================================================
# SERVICE DISCOVERY
# ============================================================================

struct ServiceInfo {
  name @0 :Text;
  host @1 :Text;
  port @2 :UInt16;
  capabilities @3 :List(Text);
  loadFactor @4 :Float32;  # 0.0 - 1.0
}

struct ServiceRegistry {
  services @0 :List(ServiceInfo);
  timestampMs @1 :Int64;
}
