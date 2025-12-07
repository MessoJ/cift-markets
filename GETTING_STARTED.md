# Getting Started with CIFT Markets

This guide will help you set up your development environment and start building.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed
- **Docker Desktop** installed and running
- **Git** installed
- **API Keys** from data providers (optional for now):
  - Polygon.io (free tier available)
  - Alpaca (free paper trading)

## 🚀 Quick Setup (5 Minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/MessoJ/cift-markets.git
cd cift-markets
```

### 2. Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file (optional for local development)
# You can use the default values for now
```

### 3. Automated Setup

```bash
# This will:
# - Install Python dependencies
# - Start Docker services
# - Initialize the database
make setup
```

Wait for services to start (about 30 seconds). You'll see:

```
✅ Services started. Access:
   - QuestDB Console: http://localhost:9000
   - Grafana: http://localhost:3001 (admin/admin)
   - Prometheus: http://localhost:9090
   - Jaeger UI: http://localhost:16686
   - MLflow UI: http://localhost:5000
```

### 4. Start the API Server

```bash
make run-api
```

The API will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎯 Verify Installation

### Check Services

```bash
# View running containers
make ps

# Expected output:
# NAME                 STATUS              PORTS
# cift-postgres       Up (healthy)       0.0.0.0:5432->5432/tcp
# cift-redis          Up (healthy)       0.0.0.0:6379->6379/tcp
# cift-questdb        Up (healthy)       0.0.0.0:9000->9000/tcp
# cift-kafka          Up (healthy)       0.0.0.0:9092->9092/tcp
# ...
```

### Test API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","environment":"development","version":"0.1.0"}
```

### Open Dashboards

```bash
# Open Grafana
make grafana

# Open Prometheus
make prometheus

# Open QuestDB Console
make questdb-shell
```

## 📚 Development Workflow

### Daily Development

```bash
# Start all services
make up

# Start API with hot reload
make run-api

# In another terminal, run tests
make test

# When done, stop services
make down
```

### Using the CLI

```bash
# View system info
cift info

# Create a user
cift create-user

# Download sample data (requires API keys)
cift download-data --symbols AAPL --start-date 2024-01-01 --end-date 2024-01-31
```

### Running Tests

```bash
# Run all tests
make test

# Run specific tests
pytest tests/unit/test_config.py -v

# Run with coverage
make coverage
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Run all checks (format + lint + test)
make check
```

## 🗂️ Project Structure Overview

```
cift-markets/
├── cift/                  # Main application code
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Configuration, logging, exceptions
│   ├── data/             # Data ingestion (coming soon)
│   ├── ml/               # ML models (coming soon)
│   └── cli.py            # Command-line interface
├── config/               # Configuration files
├── database/             # Database schemas
├── tests/                # Test suite
└── docker-compose.yml    # Infrastructure stack
```

## 🐛 Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker info

# Clean and restart
make clean-all
make up
```

### Port conflicts

If you see "port already in use" errors:

```bash
# Stop conflicting services or change ports in docker-compose.yml
# For example, if port 5432 is taken:
# Edit docker-compose.yml and change postgres port to 5433:5432
```

### Database connection errors

```bash
# Reinitialize database
make down
make up
make migrate
```

### Python dependency issues

```bash
# Reinstall dependencies
pip install --upgrade pip
make dev-install
```

## 📖 Next Steps

Now that your environment is set up:

1. **Read the Architecture** - Understanding the system design
2. **Explore the API** - Visit http://localhost:8000/docs
3. **Review the Roadmap** - Check CIFT_7MONTH_ROADMAP.md (in docs/)
4. **Start Phase 1** - Data infrastructure implementation

## 🆘 Getting Help

- **Documentation**: Check README.md and docs/ folder
- **Issues**: Create a GitHub issue
- **Email**: mesofrancis@outlook.com

## 🎉 You're Ready!

Your CIFT Markets development environment is now ready. Start building! 🚀

**Next**: Begin with Phase 1 - Data Infrastructure (see CIFT_7MONTH_ROADMAP.md)
