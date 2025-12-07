# FlowSense: Complete Implementation Roadmap
## Step-by-Step Development Guide with File Structure & Tech Stack

> **Author**: Meso Francis  
> **Date**: 2025-01-06  
> **Purpose**: Comprehensive implementation plan from zero to production  
> **Timeline**: 6 months (MVP in 2 months, Production in 6 months)

---

## Table of Contents
1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Phase 0: Project Setup & Environment (Week 1)](#phase-0-project-setup--environment-week-1)
3. [Phase 1: Data Infrastructure (Weeks 2-3)](#phase-1-data-infrastructure-weeks-2-3)
4. [Phase 2: Feature Engineering Pipeline (Weeks 4-5)](#phase-2-feature-engineering-pipeline-weeks-4-5)
5. [Phase 3: Model Development (Weeks 6-11)](#phase-3-model-development-weeks-6-11)
6. [Phase 4: Backtesting Engine (Weeks 12-13)](#phase-4-backtesting-engine-weeks-12-13)
7. [Phase 5: Real-Time Execution (Weeks 14-17)](#phase-5-real-time-execution-weeks-14-17)
8. [Phase 6: Paper Trading (Weeks 18-21)](#phase-6-paper-trading-weeks-18-21)
9. [Phase 7: Production Deployment (Weeks 22-24)](#phase-7-production-deployment-weeks-22-24)

---

## 1. Project Overview & Architecture

### 1.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                  │
├─────────────┬────────────┬────────────┬────────────┬────────────┤
│ Market Data │ Order Book │ Options    │ Social     │ On-Chain   │
│ (NASDAQ)    │ (LOB)      │ Flow       │ Sentiment  │ Whale Data │
└──────┬──────┴──────┬─────┴──────┬─────┴──────┬─────┴──────┬─────┘
       │             │            │            │            │
       ▼             ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    KAFKA STREAMING LAYER                          │
│  Topics: ticks | order_flow | options | sentiment | onchain     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING PIPELINE                     │
│  - Microstructure features (OFI, spread, depth, toxicity)       │
│  - Technical indicators (VWAP, RSI, Bollinger)                  │
│  - Alternative data fusion                                       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MODEL ENSEMBLE (5 MODELS)                      │
├───────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ Hawkes    │Transform │   HMM    │   GNN    │   XGBoost           │
│ Process   │ Patterns │  Regime  │  Cross-  │   Alternative       │
│ (OFI)     │  (1-60s) │ Detection│  Asset   │   Data Fusion       │
└─────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬────────────────┘
      │          │          │          │          │
      └──────────┴────────┬─┴──────────┴──────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE AGGREGATOR                            │
│  Weighted voting based on current regime & confidence           │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT & EXECUTION                    │
│  - Kelly Criterion position sizing                              │
│  - Regime-aware adjustments                                     │
│  - Drawdown protection                                          │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    BROKER API (Interactive Brokers)               │
│  Market orders, limit orders, position management                │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Tech Stack Summary

| Layer | Technology | Purpose | Version |
|-------|-----------|---------|---------|
| **Language** | Python | Primary development | 3.11+ |
| **ML Framework** | PyTorch | Deep learning models | 2.5.0 |
| **Data Processing** | Polars | Fast dataframe operations | 0.20.0 |
| **JIT Compilation** | Numba | Performance optimization | 0.59.0 |
| **Time Series DB** | TimescaleDB | Tick data storage | 2.14.0 |
| **Streaming** | Apache Kafka | Real-time data pipeline | 3.6.0 |
| **Caching** | Redis | Low-latency state storage | 7.2.0 |
| **Model Libraries** | tick-tock (Hawkes) | Specialized models | Latest |
|  | transformers (HF) |  | 4.36.0 |
|  | torch_geometric |  | 2.5.0 |
|  | pomegranate |  | 1.0.0 |
|  | xgboost |  | 2.0.3 |
| **Visualization** | Plotly | Interactive charts | 5.18.0 |
| **API Framework** | FastAPI | REST API endpoints | 0.108.0 |
| **Task Queue** | Celery | Async processing | 5.3.0 |
| **Container** | Docker | Deployment | 24.0.0 |
| **Orchestration** | Kubernetes | Production scaling | 1.28.0 |
| **Monitoring** | Prometheus + Grafana | Metrics & alerts | Latest |

---

## Phase 0: Project Setup & Environment (Week 1)

### Day 1-2: Project Initialization

**Objective**: Set up development environment and project structure

#### Step 1: Create Project Directory Structure
```bash
# Create project root
mkdir flowsense
cd flowsense

# Initialize Git repository
git init
git remote add origin <your-github-repo>

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create directory structure
mkdir -p {data/{raw,processed,features},ml/{models,training,inference},backtest,execution,api,tests,notebooks,docs,config,logs}
```

#### Step 2: Create Core Configuration Files

**File: `requirements.txt`**  
**Language**: Text  
**Purpose**: Python dependencies management  
**Significance**: Ensures reproducible environment
```txt
# Core
python>=3.11.0

# Data Processing
polars==0.20.0
pandas==2.1.0
numpy==1.26.0
numba==0.59.0

# Machine Learning
torch==2.5.0
torchvision==0.20.0
transformers==4.36.0
torch-geometric==2.5.0
xgboost==2.0.3
scikit-learn==1.4.0
pomegranate==1.0.0

# Hawkes Processes
tick==0.7.0.1

# Time Series
statsmodels==0.14.0
arch==6.2.0

# Database & Streaming
psycopg2-binary==2.9.9
kafka-python==2.0.2
redis==5.0.1
sqlalchemy==2.0.23

# API & Web
fastapi==0.108.0
uvicorn[standard]==0.25.0
websockets==12.0
pydantic==2.5.0

# Monitoring
prometheus-client==0.19.0
python-json-logger==2.0.7

# Utilities
python-dotenv==1.0.0
click==8.1.7
tqdm==4.66.0
loguru==0.7.2

# Backtesting
backtrader==1.9.78.123

# Broker APIs
ib-insync==0.9.86

# Dev Tools
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.12.0
flake8==7.0.0
mypy==1.7.0
```

**File: `pyproject.toml`**  
**Language**: TOML  
**Purpose**: Project metadata and build config  
**Significance**: Modern Python packaging
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "flowsense"
version = "0.1.0"
description = "Institutional-grade algorithmic trading system"
authors = [{name = "Meso Francis", email = "mesofrancis@outlook.com"}]
license = {text = "Proprietary"}
requires-python = ">=3.11"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
```

**File: `.env.example`**  
**Language**: Environment  
**Purpose**: Environment variables template  
**Significance**: Configuration management
```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=flowsense
POSTGRES_USER=flowsense
POSTGRES_PASSWORD=your_password_here

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=flowsense-consumer

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Broker (Interactive Brokers)
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# Model Paths
MODEL_DIR=./ml/models
CHECKPOINT_DIR=./ml/checkpoints

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/flowsense.log

# Trading
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
RISK_FREE_RATE=0.045
```

**File: `.gitignore`**  
**Language**: Git Config  
**Purpose**: Exclude sensitive/generated files  
**Significance**: Security and clean repo
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
venv/
.env

# Data
data/raw/
data/processed/
*.h5
*.hdf5
*.parquet

# Models
ml/models/*.pth
ml/models/*.pt
ml/checkpoints/

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Secrets
*.key
*.pem
credentials.json
```

#### Step 3: Create Docker Infrastructure

**File: `docker-compose.yml`**  
**Language**: YAML  
**Purpose**: Local development infrastructure  
**Significance**: Reproducible dev environment
```yaml
version: '3.8'

services:
  # TimescaleDB for tick data
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    container_name: flowsense-timescaledb
    environment:
      POSTGRES_DB: flowsense
      POSTGRES_USER: flowsense
      POSTGRES_PASSWORD: flowsense_dev
    ports:
      - "5432:5432"
    volumes:
      - timescale-data:/var/lib/postgresql/data
    command: postgres -c shared_preload_libraries=timescaledb

  # Apache Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    container_name: flowsense-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    container_name: flowsense-kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  # Redis for caching
  redis:
    image: redis:7.2-alpine
    container_name: flowsense-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes

  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: flowsense-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: flowsense-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  timescale-data:
  prometheus-data:
  grafana-data:
```

### Day 3-4: Core Utilities & Logging

**File: `flowsense/config/config.py`**  
**Language**: Python  
**Purpose**: Centralized configuration management  
**Significance**: Single source of truth for settings
```python
"""Configuration management for FlowSense."""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, PostgresDsn, validator


class Settings(BaseSettings):
    """Application settings."""
    
    # Project
    PROJECT_NAME: str = "FlowSense"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "ml" / "models"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # Database
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URI: Optional[PostgresDsn] = None
    
    @validator("DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST"),
            port=str(values.get("POSTGRES_PORT")),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "flowsense-consumer"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    # Trading
    INITIAL_CAPITAL: float = 100000.0
    MAX_POSITION_SIZE: float = 0.1  # 10% of capital
    RISK_FREE_RATE: float = 0.045  # 4.5%
    
    # Models
    HAWKES_LOOKBACK: int = 1000  # ticks
    TRANSFORMER_SEQUENCE_LENGTH: int = 60  # seconds
    HMM_N_STATES: int = 3  # low_vol, trending, high_vol
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
```

**File: `flowsense/utils/logger.py`**  
**Language**: Python  
**Purpose**: Structured logging utility  
**Significance**: Debug, monitor, and audit system
```python
"""Logging configuration for FlowSense."""
import sys
from pathlib import Path
from loguru import logger
from flowsense.config.config import settings


def setup_logger():
    """Configure loguru logger with file and console outputs."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler (colorized)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL if hasattr(settings, 'LOG_LEVEL') else "INFO",
        colorize=True,
    )
    
    # File handler (JSON for parsing)
    log_path = settings.LOG_DIR / "flowsense.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        serialize=True,  # JSON format
    )
    
    return logger


# Initialize logger
log = setup_logger()
```

### Day 5-7: Database Schema & Initial Setup

**File: `database/schema.sql`**  
**Language**: SQL (PostgreSQL + TimescaleDB)  
**Purpose**: Define database tables for tick data  
**Significance**: Foundation for all historical analysis
```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Tick data table (converted to hypertable)
CREATE TABLE ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(12, 4) NOT NULL,
    volume INTEGER NOT NULL,
    bid DECIMAL(12, 4),
    ask DECIMAL(12, 4),
    bid_size INTEGER,
    ask_size INTEGER,
    CONSTRAINT ticks_timestamp_symbol_key UNIQUE (timestamp, symbol)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('ticks', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for fast queries
CREATE INDEX idx_ticks_symbol_timestamp ON ticks (symbol, timestamp DESC);
CREATE INDEX idx_ticks_timestamp ON ticks (timestamp DESC);

-- Order flow imbalance (OFI) features
CREATE TABLE order_flow (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    ofi DECIMAL(8, 6),
    spread DECIMAL(8, 6),
    mid_price DECIMAL(12, 4),
    toxicity DECIMAL(8, 6),
    microprice DECIMAL(12, 4),
    depth_imbalance DECIMAL(8, 6),
    CONSTRAINT order_flow_timestamp_symbol_key UNIQUE (timestamp, symbol)
);

SELECT create_hypertable('order_flow', 'timestamp', chunk_time_interval => INTERVAL '1 day');
CREATE INDEX idx_order_flow_symbol_timestamp ON order_flow (symbol, timestamp DESC);

-- Model predictions
CREATE TABLE predictions (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    direction SMALLINT,  -- 1 = buy, -1 = sell, 0 = neutral
    confidence DECIMAL(5, 4),
    horizon_ms INTEGER,  -- prediction horizon in milliseconds
    CONSTRAINT predictions_pkey PRIMARY KEY (timestamp, symbol, model_name)
);

SELECT create_hypertable('predictions', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Trades table
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) CHECK (side IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(12, 4) NOT NULL,
    exit_price DECIMAL(12, 4),
    pnl DECIMAL(12, 2),
    fees DECIMAL(10, 2),
    regime VARCHAR(20)
);

CREATE INDEX idx_trades_timestamp ON trades (timestamp DESC);
CREATE INDEX idx_trades_symbol ON trades (symbol);

-- Continuous aggregates for performance (1-second OHLCV)
CREATE MATERIALIZED VIEW ohlcv_1s
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 second', timestamp) AS bucket,
    symbol,
    first(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, timestamp) AS close,
    sum(volume) AS volume,
    count(*) AS tick_count
FROM ticks
GROUP BY bucket, symbol;

-- Refresh policy (auto-update every 10 seconds)
SELECT add_continuous_aggregate_policy('ohlcv_1s',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 second',
    schedule_interval => INTERVAL '10 seconds');
```

**File: `database/init_db.py`**  
**Language**: Python  
**Purpose**: Initialize database with schema  
**Significance**: Automated database setup
```python
"""Initialize FlowSense database schema."""
import psycopg2
from pathlib import Path
from flowsense.config.config import settings
from flowsense.utils.logger import log


def init_database():
    """Create database schema and hypertables."""
    
    log.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Read schema file
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    log.info("Executing schema creation...")
    cursor.execute(schema_sql)
    
    log.info("Database schema created successfully!")
    
    # Verify hypertables
    cursor.execute("""
        SELECT hypertable_schema, hypertable_name 
        FROM timescaledb_information.hypertables;
    """)
    
    hypertables = cursor.fetchall()
    log.info(f"Created {len(hypertables)} hypertables:")
    for schema, table in hypertables:
        log.info(f"  - {schema}.{table}")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    init_database()
```

---

## Summary: Week 1 Deliverables

### Files Created (14 files):
1. `requirements.txt` - Dependencies
2. `pyproject.toml` - Project config
3. `.env.example` - Environment template
4. `.gitignore` - Git exclusions
5. `docker-compose.yml` - Infrastructure
6. `config/config.py` - Settings management
7. `utils/logger.py` - Logging utility
8. `database/schema.sql` - DB schema
9. `database/init_db.py` - DB initialization
10. `README.md` - Project documentation
11. `Makefile` - Common commands
12. `.github/workflows/ci.yml` - CI/CD pipeline
13. `tests/__init__.py` - Test package init
14. `notebooks/01_data_exploration.ipynb` - Initial EDA

### Infrastructure Running:
- ✅ TimescaleDB (PostgreSQL 16 + TimescaleDB)
- ✅ Apache Kafka + Zookeeper
- ✅ Redis
- ✅ Prometheus
- ✅ Grafana

### Next Phase:
**Phase 1: Data Infrastructure (Weeks 2-3)** - Data ingestion, storage, and streaming pipelines

