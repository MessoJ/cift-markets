# CIFT Markets Architecture - Explained Simply

**Comparing to Your Inventory Management App**

---

## 🎯 Your Inventory App vs CIFT Markets

### **Your Inventory App Structure:**
```
Inventory Management App
├── Auth Service          (handles login/signup)
├── Inventory Service     (core business logic - products)
├── Frontend              (user interface)
├── Database              (stores data)
└── Maybe API Gateway     (routes requests)
```

### **CIFT Markets Structure:**
```
CIFT Markets (Trading Platform)
├── Auth Service          ✅ Built into API (same concept)
├── Trading Service       ✅ Core business logic (like your inventory service)
├── Market Data Service   ✅ Gets real-time stock prices
├── Risk Service          ✅ Prevents bad trades
├── Execution Service     ✅ Sends orders to brokers
├── ML/Analytics Service  ✅ Predicts price movements
├── Frontend              ⏳ Phase 8 (not built yet)
└── Databases (4)         ✅ Each optimized for specific data type
```

**Key Difference:** CIFT has MORE services because trading is more complex than inventory management.

---

## 📦 Breaking Down CIFT's 11 Containers

### **Think of it Like Your Inventory App:**

Your inventory app probably had:
- **1 service** = Inventory logic (add/remove products)
- **1 database** = Store products
- **1 frontend** = User interface

CIFT has:
- **1 API service** = All business logic (like your inventory service but bigger)
- **4 databases** = Each optimized for different data types
- **1 message queue** = Communication between services
- **3 monitoring tools** = Watch for problems
- **1 ML tool** = Track experiments
- **1 frontend** = Not built yet (Phase 8)

---

## 🏗️ Complete Service Breakdown (11 Containers)

### **1. The "Brain" (API Service)**
```
Container: cift-api (port 8000)
```

**What it does:**
- **Authentication** - Login, signup, JWT tokens (like your auth service)
- **Trading Logic** - Place orders, manage positions (like your inventory CRUD)
- **Risk Management** - Prevent bad trades (like inventory stock validation)
- **Market Data** - Get real-time prices (like product price updates)
- **ML Predictions** - AI forecasts (optional feature)

**Compare to your app:**
```
Your Inventory Service:
- Add product ✅
- Remove product ✅
- Update stock ✅
- Check availability ✅

CIFT API Service:
- Place order ✅
- Cancel order ✅
- Check portfolio ✅
- Get market data ✅
- Run risk checks ✅
```

**Technology:**
- FastAPI (Python web framework)
- Rust core (for super-fast order matching)
- Built with multi-stage Docker (compiles Rust automatically)

---

### **2. Databases (4 Containers - Not Just 1!)**

**Why 4 databases?** Each is optimized for different data types.

#### **A. PostgreSQL** (port 5432)
```
Container: cift-postgres
```

**What it stores:**
- User accounts (email, password, profile)
- Orders (buy/sell records)
- Positions (what stocks you own)
- Portfolio (account balance, trades)

**Compare to your app:**
```
Your inventory DB probably stores:
- Products ✅
- Categories ✅
- Stock levels ✅
- Users ✅

CIFT PostgreSQL stores:
- Users ✅
- Orders ✅
- Positions ✅
- Settings ✅
```

**Why:** Traditional relational data (like your inventory DB).

---

#### **B. QuestDB** (port 9000)
```
Container: cift-questdb
```

**What it stores:**
- Real-time stock prices (every second!)
- Historical price data
- Market ticks (1.4 million per second!)

**Compare to your app:**
```
Your inventory app probably doesn't have this, but imagine:
- Product price history (every minute)
- Customer visit timestamps
- Sensor data from warehouse

CIFT QuestDB stores:
- AAPL stock price at 09:30:00.001 = $150.45
- AAPL stock price at 09:30:00.002 = $150.46
- AAPL stock price at 09:30:00.003 = $150.47
... millions of records per day
```

**Why:** PostgreSQL is too slow for time-series data.

---

#### **C. ClickHouse** (port 8123)
```
Container: cift-clickhouse
```

**What it does:**
- Complex analytics queries (100x faster!)
- "Show me all trades in 2024 where profit > $1000"
- "What's my average win rate by hour of day?"
- Aggregations over millions of rows

**Compare to your app:**
```
Your inventory analytics might be:
- "Total sales by category this month"
- "Which products sold most?"
- "Average order value"

CIFT ClickHouse does:
- "Profit/loss by strategy over 6 months"
- "Win rate by time of day"
- "Best performing symbols"
- Analyze 100M+ trades in seconds
```

**Why:** PostgreSQL would take minutes, ClickHouse takes milliseconds.

---

#### **D. Dragonfly** (port 6379)
```
Container: cift-dragonfly
```

**What it stores:**
- Cached data (temporary, fast access)
- Session data
- Real-time calculations
- 2.5 million operations per second!

**Compare to your app:**
```
Your inventory app might cache:
- Product list (so you don't query DB every time)
- User session
- Shopping cart

CIFT Dragonfly caches:
- Latest stock prices (updated every ms)
- User portfolio (instant access)
- Risk calculations
- WebSocket connections
```

**Why:** Redis-compatible but 25x faster. Reduces database load.

---

### **3. Message Queue (1 Container)**

#### **NATS JetStream** (port 4222)
```
Container: cift-nats
```

**What it does:**
- Services talk to each other via messages
- "Order placed" → notify execution service
- "Price updated" → notify all clients
- Sub-millisecond delivery (5-10x faster than Kafka)

**Compare to your app:**
```
Your inventory app probably calls services directly:
Frontend → API → Database

But imagine if you had:
- Order placed → Send email notification
- Stock low → Alert warehouse
- Price changed → Update all users

CIFT NATS does:
- Order placed → Risk check → Execute → Notify user
- Market data → ML prediction → Generate signal
- Trade filled → Update portfolio → Log event
```

**Why:** Decouples services, allows real-time streaming.

---

### **4. Monitoring Stack (3 Containers)**

#### **A. Prometheus** (port 9090)
```
Container: cift-prometheus
```

**What it does:**
- Collects metrics (CPU, memory, request count)
- "How many orders per second?"
- "Is the API slow?"

**Compare to your app:**
```
You might check:
- How many products in DB?
- How many users logged in?
- API response time?

CIFT Prometheus tracks:
- Orders per second
- P99 latency (<10ms target)
- Database connections
- Cache hit rate
```

---

#### **B. Grafana** (port 3001)
```
Container: cift-grafana
```

**What it does:**
- Pretty dashboards for Prometheus data
- Graphs, charts, alerts

**Compare to your app:**
```
You might have:
- Admin dashboard showing sales

CIFT Grafana shows:
- Real-time order flow
- Latency graphs
- System health
- Performance metrics
```

**Access:** http://localhost:3001 (admin/admin)

---

#### **C. Jaeger** (port 16686)
```
Container: cift-jaeger
```

**What it does:**
- Distributed tracing
- "Why is this API call slow?"
- Shows exact path: API → DB → Cache → Return

**Compare to your app:**
```
You might debug:
- "Why is product page slow?"

CIFT Jaeger shows:
- Order request: 10ms total
  - Risk check: 2ms
  - Database query: 5ms
  - Execution: 3ms
```

**Access:** http://localhost:16686

---

### **5. MLOps (1 Container)**

#### **MLflow** (port 5000)
```
Container: cift-mlflow
```

**What it does:**
- Track ML experiments
- Compare models
- Store trained models
- Version control for AI

**Compare to your app:**
```
Your inventory app probably doesn't have ML, but imagine:
- Testing different recommendation algorithms
- Tracking which model predicts sales best

CIFT MLflow tracks:
- LSTM model accuracy: 68%
- Transformer model accuracy: 72%
- Best model → deploy to production
```

**Access:** http://localhost:5000

---

## 🎯 How They Work Together

### **Example Flow: Placing an Order**

**Your inventory app:**
```
1. User clicks "Add Product"
2. Frontend → API
3. API → Database (INSERT)
4. Return success
```

**CIFT trading platform:**
```
1. User clicks "Buy AAPL"
2. Frontend → API (port 8000)
3. API → Risk check (Rust core, <1μs)
4. API → Check cache (Dragonfly) for latest price
5. API → Save order (PostgreSQL)
6. API → Publish message (NATS) "order.new.AAPL"
7. Execution service listens → Sends to broker
8. Broker confirms → Publish "order.filled.AAPL"
9. API receives → Update position (PostgreSQL)
10. API → Save fill (QuestDB)
11. API → Update cache (Dragonfly)
12. WebSocket → Notify user "Order filled!"
13. Prometheus → Log metrics
14. All done in <10ms!
```

**Why so complex?** Trading requires:
- Risk checks (prevent losing money)
- Real-time data (prices change every millisecond)
- Audit trail (regulators require it)
- High performance (<10ms latency)

---

## 📊 Visual Comparison

### **Your Inventory App:**
```
┌─────────────────────────────────┐
│         Frontend                │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│      Auth Service (maybe)       │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│     Inventory Service (API)     │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│        Database (1)             │
└─────────────────────────────────┘
```

**Total: ~3-4 services**

---

### **CIFT Markets:**
```
┌─────────────────────────────────┐
│    Frontend (Phase 8 - TBD)     │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│      API Service (FastAPI)      │
│   - Auth                        │
│   - Trading logic               │
│   - Risk management             │
│   - Market data                 │
│   - ML predictions              │
└─────────────────────────────────┘
         ↓         ↓          ↓
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│PostgreSQL│ │ QuestDB  │ │ClickHouse│ │Dragonfly │
│ Users    │ │  Ticks   │ │Analytics │ │  Cache   │
│ Orders   │ │  Prices  │ │ Complex  │ │  Fast    │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
         ↓
┌─────────────────────────────────┐
│    NATS JetStream (Messages)    │
└─────────────────────────────────┘
         ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│Prometheus│ │ Grafana  │ │  Jaeger  │
│ Metrics  │ │Dashboard │ │ Tracing  │
└──────────┘ └──────────┘ └──────────┘
         ↓
┌─────────────────────────────────┐
│         MLflow (ML Ops)         │
└─────────────────────────────────┘
```

**Total: 11 services**

---

## 💡 Why So Many Services?

### **Your inventory app needs:**
- ✅ Store products
- ✅ Update stock
- ✅ Show to users

**Simple domain = fewer services**

### **CIFT trading platform needs:**
- ✅ Store users, orders, positions (PostgreSQL)
- ✅ Real-time tick data, millions per second (QuestDB)
- ✅ Complex analytics, 100x faster (ClickHouse)
- ✅ Ultra-fast cache, 2.5M ops/sec (Dragonfly)
- ✅ Message streaming, <1ms latency (NATS)
- ✅ Monitor performance (Prometheus, Grafana, Jaeger)
- ✅ Track ML experiments (MLflow)
- ✅ Super-fast order matching, <10μs (Rust core)

**Complex domain = more specialized services**

---

## 🚀 Summary

### **Inventory App Pattern:**
```
Frontend → API → Database
```

**Simple, clean, perfect for CRUD operations.**

### **CIFT Pattern:**
```
Frontend → API (FastAPI + Rust) → 
  - 4 Databases (each specialized)
  - Message Queue (real-time events)
  - Monitoring Stack (observability)
  - ML Platform (experiments)
```

**Complex, but necessary for high-frequency trading.**

---

## 🎯 The Big Picture

| Your Inventory App | CIFT Markets |
|-------------------|-------------|
| **1 API Service** | **1 API Service** (but does more) |
| **1 Database** | **4 Databases** (specialized) |
| **1 Frontend** | **1 Frontend** (Phase 8) |
| **Maybe monitoring** | **3 Monitoring tools** (required) |
| **No message queue** | **1 Message queue** (real-time) |
| **No ML** | **1 ML platform** (predictions) |
| **~3-4 containers** | **11 containers** |

### **Why the difference?**

**Inventory Management:**
- Data changes slowly (products don't change every millisecond)
- Queries are simple ("show all products")
- No real-time requirements
- Simple business logic

**Trading Platform:**
- Data changes extremely fast (prices every millisecond)
- Complex queries ("analyze 100M trades")
- Real-time critical (<10ms latency)
- Complex business logic (risk, execution, ML)

---

## ✅ Is It Properly Structured?

**YES!** Each service has ONE job:

1. ✅ **API** - Business logic (like your inventory service)
2. ✅ **PostgreSQL** - Transactional data (like your product DB)
3. ✅ **QuestDB** - Time-series (new requirement for trading)
4. ✅ **ClickHouse** - Analytics (new requirement for trading)
5. ✅ **Dragonfly** - Cache (performance optimization)
6. ✅ **NATS** - Messages (decouples services)
7. ✅ **Prometheus** - Metrics (monitoring)
8. ✅ **Grafana** - Dashboards (visualization)
9. ✅ **Jaeger** - Tracing (debugging)
10. ✅ **MLflow** - ML tracking (experiments)
11. ✅ **Frontend** - UI (Phase 8)

**Just like your inventory app, but scaled for trading complexity!**

---

## 🎓 Next Steps

1. ✅ Build is running (compiling Rust in Docker)
2. ⏳ Wait ~5-10 minutes for first build
3. ✅ Start all services: `docker-compose up -d`
4. ✅ Access API: http://localhost:8000/docs
5. ⏳ Build frontend (Phase 8) - awaiting your requirements

**Your inventory app experience will help!** The patterns are the same, just more services.
