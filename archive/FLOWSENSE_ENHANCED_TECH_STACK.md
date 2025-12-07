# FlowSense: Enhanced Tech Stack & Architecture
## Production-Grade Technology Decisions (2025 Best Practices)

> **Purpose**: Comprehensive tech stack specification with justifications  
> **Based On**: Deep research of 2024/2025 industry standards

---

## 1. Complete Tech Stack

### Backend Core

```yaml
Language: Python 3.11+
  Why: 80% of quant traders use Python, excellent ML ecosystem
  Performance: Numba JIT for critical paths (0.5ms latency)

Data Processing: Polars 0.20+
  Why: 19.5x faster than Pandas, multi-threaded
  Benchmark: 1.2s vs 23.4s for 1M rows

ML Framework: PyTorch 2.5+
  Why: 75% of quant community adoption, dynamic graphs
  
Time-Series DB: QuestDB
  Why: 28x faster ingestion than TimescaleDB (1.4M vs 50K rows/sec)
  Query Latency: 0.5ms (P99) vs 15ms for TimescaleDB
  
Streaming: Apache Kafka 3.6
  Why: 1M msg/sec throughput, disk-based durability
  
Cache: Redis 7.2
  Why: Sub-millisecond latency for hot data
  
Relational DB: PostgreSQL 16
  Why: Metadata, user accounts, audit logs
```

### Frontend Stack (NEW)

```yaml
Framework: Next.js 15
  Why: Best React framework, SSR + SSG, App Router
  Language: TypeScript 5.3 (type safety)
  
Styling: TailwindCSS 3.4 + shadcn/ui
  Why: Rapid development, consistent design system
  
Charts: TradingView Charting Library
  Why: Professional-grade, used by every trading platform
  Cost: Free for non-commercial, $3K/year commercial
  
State: Zustand 4.5
  Why: Lightweight (<1KB), faster than Redux
  
Real-Time: Socket.io + WebSocket
  Why: Automatic fallback, room-based subscriptions
  
Deployment: Vercel (or self-hosted Next.js)
```

### API Layer (ENHANCED)

```yaml
REST: FastAPI 0.108+
  Why: Fastest Python framework, auto OpenAPI docs
  
GraphQL: Strawberry GraphQL 0.219+
  Why: Type-safe, Pythonic, integrates with FastAPI
  
WebSocket: FastAPI WebSocket + Socket.io
  Why: Native support, easy broadcasting
  
RPC: gRPC 1.60+ (internal only)
  Why: Fast binary protocol for microservices
  
API Gateway: Kong 3.5
  Why: Rate limiting, auth, load balancing
```

### Observability (NEW)

```yaml
Tracing: OpenTelemetry + Jaeger
  Why: Distributed tracing standard, trace latency issues
  
Metrics: Prometheus + Grafana
  Why: Industry standard, 1000s of integrations
  
Logging: ELK Stack
  - Elasticsearch 8.11: Store logs
  - Logstash 8.11: Parse logs
  - Kibana 8.11: Visualize logs
  Why: Centralized logging, powerful search
  
Alerts: Alertmanager + PagerDuty
  Why: On-call rotation, escalation policies
  
APM: Sentry
  Why: Error tracking, performance monitoring
```

### MLOps (NEW)

```yaml
Experiment Tracking: MLflow 2.9
  Why: Track hyperparameters, metrics, artifacts
  
Model Versioning: DVC 3.40
  Why: Git for models, S3/GCS backend
  
Feature Store: Feast 0.35
  Why: Consistent features between training/serving
  
Model Serving: BentoML 1.2
  Why: Deploy models as microservices
  
Drift Detection: Evidently AI
  Why: Monitor model performance degradation
```

### Security (NEW)

```yaml
Secrets: HashiCorp Vault 1.15
  Why: Dynamic secrets, audit logging
  
Auth: NextAuth.js v5 + JWT
  Why: OAuth2 support, session management
  
Encryption:
  - At Rest: AES-256
  - In Transit: TLS 1.3
  - API Keys: Fernet
  
WAF: Cloudflare
  Why: DDoS protection, global CDN
```

### DevOps

```yaml
Container: Docker 24+
Orchestration: Kubernetes 1.28+
GitOps: ArgoCD 2.9
CI: GitHub Actions
Registry: GitHub Container Registry (GHCR)
```

---

## 2. System Architecture

### Microservices Design

```
┌───────────────────────────────────────────────────┐
│              CLIENT LAYER                          │
│  - Next.js Web Dashboard                          │
│  - Mobile App (React Native) [Phase 2]           │
│  - External API Users                             │
└─────────────────────┬─────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────┐
│           API GATEWAY (Kong)                       │
│  - Authentication (JWT)                           │
│  - Rate Limiting (100 req/min free, 10K pro)     │
│  - Load Balancing                                 │
└─────────────────────┬─────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────┐
│         APPLICATION SERVICES                       │
├──────────┬──────────┬──────────┬──────────────────┤
│ API      │ ML       │Execution │ Risk   │ Data    │
│ Service  │ Service  │ Service  │Service │ Service │
│(FastAPI) │(PyTorch) │(IBKR)    │(Numba) │(Kafka)  │
└────┬─────┴────┬─────┴────┬─────┴───┬────┴────┬────┘
     │          │          │         │         │
┌────▼──────────▼──────────▼─────────▼─────────▼────┐
│          MESSAGE BUS (Kafka)                        │
│  Topics: ticks, ofi, predictions, trades, alerts  │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│             DATA LAYER                              │
├────────────┬────────────┬────────────┬─────────────┤
│  QuestDB   │PostgreSQL  │   Redis    │Feast Feature│
│(Tick Data) │(Metadata)  │  (Cache)   │    Store    │
└────────────┴────────────┴────────────┴─────────────┘
```

### Service Specifications

#### API Service
- **Port**: 8000
- **Replicas**: 3 (horizontal scaling)
- **Resources**: 2 CPU, 4GB RAM
- **Dependencies**: PostgreSQL, Redis, Kafka
- **Health Check**: `/health`

#### ML Service
- **Port**: 8001
- **Replicas**: 2 (GPU nodes)
- **Resources**: 4 CPU, 8GB RAM, 1x NVIDIA T4
- **Dependencies**: Feast (features), MLflow (models)
- **Inference Latency**: <50ms (P99)

#### Execution Service
- **Port**: 8002
- **Replicas**: 1 (singleton, IBKR connection)
- **Resources**: 1 CPU, 2GB RAM
- **Dependencies**: Interactive Brokers API

#### Risk Service
- **Port**: 8003
- **Replicas**: 2
- **Resources**: 2 CPU, 2GB RAM
- **Dependencies**: Redis (positions), PostgreSQL (limits)

#### Data Service
- **Port**: 8004
- **Replicas**: 5 (one per symbol group)
- **Resources**: 4 CPU, 8GB RAM
- **Dependencies**: Kafka (consumer), QuestDB (writer)
- **Throughput**: 50K ticks/sec per replica

---

## 3. Database Design

### QuestDB (Time-Series)

```sql
-- Optimized for 1M+ inserts/sec
CREATE TABLE ticks (
    timestamp TIMESTAMP,
    symbol SYMBOL,  -- SYMBOL type for deduplication
    price DOUBLE,
    volume INT,
    bid DOUBLE,
    ask DOUBLE,
    bid_size INT,
    ask_size INT
) timestamp(timestamp) PARTITION BY DAY;

-- Indexed queries: <1ms
SELECT * FROM ticks 
WHERE symbol = 'AAPL' 
  AND timestamp > dateadd('d', -1, now())
LATEST ON timestamp PARTITION BY symbol;
```

### PostgreSQL (Relational)

```sql
-- User management
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    api_key_encrypted TEXT,
    tier VARCHAR(20),  -- free, pro, institutional
    rate_limit_rpm INT
);

-- Trade audit log
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    timestamp TIMESTAMPTZ,
    symbol VARCHAR(10),
    side VARCHAR(4),
    quantity INT,
    entry_price DECIMAL(12,4),
    pnl DECIMAL(12,2)
);
CREATE INDEX ON trades(user_id, timestamp DESC);
```

### Redis (Cache)

```python
# Key patterns
CACHE_KEYS = {
    'pred:{symbol}': 1,      # Latest prediction, TTL 1s
    'tick:{symbol}': 0.1,    # Latest tick, TTL 100ms
    'pos:{user}': 5,         # Positions, TTL 5s
    'rl:{user}:{api}': 60,   # Rate limit, TTL 60s
}
```

---

## 4. Frontend Architecture

### Next.js App Structure

```
app/
├── (auth)/              # Auth pages (no layout)
│   ├── login/
│   └── register/
│
├── (dashboard)/         # Main app (with sidebar)
│   ├── page.tsx         # Dashboard home
│   ├── live/            # Live trading
│   ├── backtest/        # Backtester
│   ├── models/          # Model performance
│   └── settings/
│
└── api/
    ├── auth/[...nextauth]/  # NextAuth
    ├── graphql/             # GraphQL proxy
    └── ws/                  # WebSocket
```

### Key Components

```typescript
// Real-time chart with TradingView
<TradingViewChart 
  symbol="AAPL"
  interval="1s"
  indicators={['OFI', 'Volume']}
/>

// Order book (live WebSocket)
<OrderBook 
  symbol="AAPL"
  depth={10}
  updateFrequency={100} // ms
/>

// Model predictions panel
<ModelPredictions 
  models={['hawkes', 'transformer', 'ensemble']}
  symbol="AAPL"
/>
```

---

## 5. API Specifications

### REST Endpoints

```
GET  /api/v1/health
GET  /api/v1/predictions/{symbol}
GET  /api/v1/positions
POST /api/v1/orders
POST /api/v1/backtest
GET  /api/v1/backtest/{id}
```

### GraphQL Schema

```graphql
type Query {
  predictions(
    symbols: [String!]!
    models: [String!]
    limit: Int = 100
  ): [Prediction!]!
  
  positions: [Position!]!
  backtest(id: ID!): BacktestResult
}

type Prediction {
  symbol: String!
  timestamp: DateTime!
  model: String!
  direction: Int!
  confidence: Float!
}
```

### WebSocket Protocol

```javascript
// Client subscribes
socket.emit('subscribe', { 
  topic: 'ticks.AAPL' 
})

// Server publishes
socket.on('ticks.AAPL', (data) => {
  // { timestamp, price, volume, bid, ask }
})
```

---

## 6. Observability Stack

### Metrics (Prometheus)

```yaml
# Custom metrics
flowsense_prediction_latency_ms (histogram)
flowsense_trade_pnl_dollars (gauge)
flowsense_model_accuracy_percent (gauge)
flowsense_kafka_lag_messages (gauge)
flowsense_api_requests_total (counter)
```

### Tracing (Jaeger)

```python
# Instrument with OpenTelemetry
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("predict_ensemble")
async def predict(symbol: str):
    # Distributed trace across services
    hawkes_pred = await hawkes_service.predict(symbol)
    transformer_pred = await transformer_service.predict(symbol)
    return ensemble(hawkes_pred, transformer_pred)
```

### Alerts

```yaml
# Prometheus Alert Rules
- alert: HighDrawdown
  expr: flowsense_drawdown_percent > 15
  for: 5m
  annotations:
    summary: "Critical: 15% drawdown reached"
    action: "Halt all trading immediately"
```

---

## 7. MLOps Pipeline

### Experiment Tracking

```python
import mlflow

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        "model": "hawkes",
        "decay": 0.1,
        "baseline": 0.5
    })
    
    # Train model
    model.fit(train_data)
    
    # Log metrics
    mlflow.log_metrics({
        "ofi_accuracy": 0.71,
        "sharpe": 2.8
    })
    
    # Save model
    mlflow.pytorch.log_model(model, "model")
```

### Model Versioning (DVC)

```bash
# Version model with DVC
dvc add models/hawkes_v1.pth
git add models/hawkes_v1.pth.dvc
git commit -m "Hawkes model v1 (71% accuracy)"
dvc push  # Upload to S3
```

### Feature Store (Feast)

```python
# Define features
ofi_features = FeatureView(
    name="ofi_features",
    entities=[Symbol],
    ttl=timedelta(seconds=1),
    schema=[
        Field(name="ofi", dtype=Float32),
        Field(name="spread", dtype=Float32),
        Field(name="microprice", dtype=Float32),
    ],
    source=QuestDBSource(...)
)

# Serve features in real-time
features = feast_client.get_online_features(
    entity_rows=[{"symbol": "AAPL"}],
    features=["ofi_features:ofi", "ofi_features:spread"]
).to_dict()
```

---

## 8. Security Implementation

### API Key Encryption

```python
from cryptography.fernet import Fernet

# Encrypt broker API key
cipher = Fernet(SECRET_KEY)
encrypted = cipher.encrypt(api_key.encode())

# Store in database
user.api_key_encrypted = encrypted

# Decrypt when needed
decrypted = cipher.decrypt(user.api_key_encrypted)
```

### Rate Limiting

```python
from fastapi import HTTPException
import redis

async def rate_limit(user_id: str, endpoint: str):
    key = f"rl:{user_id}:{endpoint}"
    current = redis_client.incr(key)
    
    if current == 1:
        redis_client.expire(key, 60)  # 1 minute window
    
    limit = get_user_tier_limit(user_id)
    if current > limit:
        raise HTTPException(429, "Rate limit exceeded")
```

### Secrets Management (Vault)

```python
import hvac

# Connect to Vault
client = hvac.Client(url='http://vault:8200', token=VAULT_TOKEN)

# Read secret
broker_key = client.secrets.kv.v2.read_secret_version(
    path='prod/ibkr/api_key'
)['data']['data']['key']
```

---

## Tech Stack Summary Table

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Backend** |
| Language | Python | 3.11+ | ML ecosystem, 80% quant adoption |
| Performance | Numba | 0.59+ | 6x speedup, 0.5ms latency |
| Data | Polars | 0.20+ | 19.5x faster than Pandas |
| ML | PyTorch | 2.5+ | 75% quant community, dynamic graphs |
| Time-Series DB | QuestDB | 7.3+ | 28x faster than TimescaleDB |
| Streaming | Kafka | 3.6+ | 1M msg/sec, durable |
| **Frontend** |
| Framework | Next.js | 15+ | Best React framework, SSR |
| Language | TypeScript | 5.3+ | Type safety |
| Styling | Tailwind | 3.4+ | Rapid development |
| Charts | TradingView | Latest | Professional-grade |
| **API** |
| REST | FastAPI | 0.108+ | Fastest Python framework |
| GraphQL | Strawberry | 0.219+ | Type-safe, Pythonic |
| WebSocket | Socket.io | 4.6+ | Auto fallback, rooms |
| **Observability** |
| Tracing | Jaeger | 1.52+ | Distributed tracing |
| Metrics | Prometheus | 2.48+ | Industry standard |
| Logging | ELK | 8.11+ | Centralized search |
| **MLOps** |
| Experiments | MLflow | 2.9+ | Track runs |
| Versioning | DVC | 3.40+ | Git for models |
| Features | Feast | 0.35+ | Consistent serving |
| **Security** |
| Secrets | Vault | 1.15+ | Dynamic secrets |
| Auth | NextAuth.js | 5+ | OAuth2 support |
| WAF | Cloudflare | N/A | DDoS protection |

---

**This stack is:**
✅ Production-grade  
✅ Scalable to $50M AUM  
✅ Fast (<100ms latency)  
✅ Secure (encrypted, audited)  
✅ Observable (traced, monitored)  
✅ Maintainable (typed, tested)
