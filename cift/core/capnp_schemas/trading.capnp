@0xb312c20f19e5c281;

# Cap'n Proto schemas for trading messages
# Zero-copy serialization for ultra-low latency

enum Side {
  buy @0;
  sell @1;
}

enum OrderType {
  market @0;
  limit @1;
  stop @2;
  stopLimit @3;
}

enum OrderStatus {
  pending @0;
  accepted @1;
  filled @2;
  partiallyFilled @3;
  cancelled @4;
  rejected @5;
}

struct Order {
  orderId @0 :UInt64;
  userId @1 :UInt64;
  symbol @2 :Text;
  side @3 :Side;
  orderType @4 :OrderType;
  quantity @5 :Float64;
  price @6 :Float64;
  status @7 :OrderStatus;
  timestamp @8 :Int64;
}

struct Fill {
  fillId @0 :UInt64;
  orderId @1 :UInt64;
  userId @2 :UInt64;
  symbol @3 :Text;
  side @4 :Side;
  quantity @5 :Float64;
  price @6 :Float64;
  value @7 :Float64;
  commission @8 :Float64;
  timestamp @9 :Int64;
  venue @10 :Text;
}

struct Position {
  positionId @0 :UInt64;
  userId @1 :UInt64;
  symbol @2 :Text;
  quantity @3 :Float64;
  avgCost @4 :Float64;
  currentPrice @5 :Float64;
  unrealizedPnl @6 :Float64;
  realizedPnl @7 :Float64;
}

struct Signal {
  signalId @0 :UInt64;
  symbol @1 :Text;
  timestamp @2 :Int64;
  side @3 :Side;
  confidence @4 :Float64;
  features @5 :List(Float64);
  modelVersion @6 :Text;
}

struct TradingBatch {
  orders @0 :List(Order);
  fills @1 :List(Fill);
  positions @2 :List(Position);
  signals @3 :List(Signal);
}
