@0x9eb32e19f86ee174;

# Cap'n Proto schemas for market data
# Zero-copy serialization for maximum performance

struct Tick {
  timestamp @0 :Int64;      # Microseconds since epoch
  symbol @1 :Text;
  price @2 :Float64;
  volume @3 :UInt32;
  bid @4 :Float64;
  ask @5 :Float64;
  bidSize @6 :UInt32;
  askSize @7 :UInt32;
  exchange @8 :Text;
}

struct Quote {
  timestamp @0 :Int64;
  symbol @1 :Text;
  bidPrice @2 :Float64;
  askPrice @3 :Float64;
  bidSize @4 :UInt32;
  askSize @5 :UInt32;
  conditions @6 :List(Text);
}

struct Bar {
  timestamp @0 :Int64;
  symbol @1 :Text;
  timeframe @2 :Text;
  open @3 :Float64;
  high @4 :Float64;
  low @5 :Float64;
  close @6 :Float64;
  volume @7 :UInt64;
  vwap @8 :Float64;
  tradeCount @9 :UInt32;
}

struct OrderBookLevel {
  price @0 :Float64;
  quantity @1 :Float64;
  orderCount @2 :UInt16;
}

struct OrderBookSnapshot {
  timestamp @0 :Int64;
  symbol @1 :Text;
  bids @2 :List(OrderBookLevel);
  asks @3 :List(OrderBookLevel);
}

struct MarketDataBatch {
  ticks @0 :List(Tick);
  quotes @1 :List(Quote);
  bars @2 :List(Bar);
}
