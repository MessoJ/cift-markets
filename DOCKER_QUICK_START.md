# 🐳 DOCKER QUICK START GUIDE

## 🚀 One-Command Startup

```bash
docker-compose up -d && echo "✅ All services started!"
```

---

## 📋 Essential Commands

### Start Services
```bash
# Start all services (detached)
docker-compose up -d

# Start specific service
docker-compose up -d api
docker-compose up -d postgres

# Start with live logs
docker-compose up
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f postgres

# Last 100 lines
docker-compose logs --tail=100 api
```

### Check Status
```bash
# List all containers
docker-compose ps

# Check health
docker-compose ps | grep healthy
```

### Stop Services
```bash
# Stop all
docker-compose down

# Stop but keep volumes (data preserved)
docker-compose stop

# Stop and remove volumes (fresh start)
docker-compose down -v
```

### Rebuild Containers
```bash
# Rebuild after code changes
docker-compose build api
docker-compose up -d api

# Force rebuild (no cache)
docker-compose build --no-cache api
```

---

## 🔍 Debugging

### Check API Logs
```bash
docker-compose logs -f api | grep ERROR
```

### Enter Container Shell
```bash
# API container
docker-compose exec api bash

# PostgreSQL container
docker-compose exec postgres psql -U cift_user -d cift_markets
```

### Check Database Tables
```bash
docker-compose exec postgres psql -U cift_user -d cift_markets -c "\dt"
```

### Run SQL Query
```bash
docker-compose exec postgres psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM users;"
```

---

## 🗄️ Database Migrations

### Run Migration
```bash
# Method 1: Direct SQL
docker-compose exec -T postgres psql -U cift_user -d cift_markets < database/migrations/002_critical_features.sql

# Method 2: Inside container
docker-compose exec postgres psql -U cift_user -d cift_markets
# Then paste SQL manually

# Method 3: Use Alembic
docker-compose exec api python -m alembic upgrade head
```

### Check Migration Status
```bash
# List all tables
docker-compose exec postgres psql -U cift_user -d cift_markets -c "\dt"

# Check specific table
docker-compose exec postgres psql -U cift_user -d cift_markets -c "\d payment_methods"
```

---

## 🌐 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **API** | http://localhost:8000 | FastAPI application |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **API Health** | http://localhost:8000/health | Health check |
| **PostgreSQL** | localhost:5432 | Relational database |
| **QuestDB Console** | http://localhost:9000 | Time-series DB UI |
| **Dragonfly** | localhost:6379 | Redis-compatible cache |
| **ClickHouse** | http://localhost:8123 | Analytics database |
| **Prometheus** | http://localhost:9090 | Metrics |
| **Grafana** | http://localhost:3001 | Dashboards (admin/admin) |
| **Jaeger** | http://localhost:16686 | Distributed tracing |
| **MLflow** | http://localhost:5000 | ML experiment tracking |

---

## 🔧 Common Issues

### Issue: Container won't start
```bash
# Check logs
docker-compose logs api

# Check if port is in use
netstat -ano | findstr :8000

# Remove and recreate
docker-compose down
docker-compose up -d
```

### Issue: Database connection refused
```bash
# Ensure postgres is running
docker-compose ps postgres

# Check postgres logs
docker-compose logs postgres

# Restart postgres
docker-compose restart postgres
```

### Issue: API returns 500 errors
```bash
# Check if migrations ran
docker-compose exec postgres psql -U cift_user -d cift_markets -c "\dt"

# Run migrations
docker-compose exec -T postgres psql -U cift_user -d cift_markets < database/migrations/002_critical_features.sql

# Restart API
docker-compose restart api
```

### Issue: "Pool not initialized" error
```bash
# This is fixed! But if you see it:
# 1. Ensure databases are healthy
docker-compose ps

# 2. Check API environment variables
docker-compose exec api env | grep POSTGRES

# 3. Restart API with logs
docker-compose restart api
docker-compose logs -f api
```

---

## 🧹 Cleanup

### Remove Stopped Containers
```bash
docker-compose down
```

### Remove All Data (DESTRUCTIVE)
```bash
# This deletes ALL data!
docker-compose down -v
```

### Remove Images (Free Space)
```bash
docker-compose down --rmi all
```

### Full Clean Start
```bash
# Stop everything
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Rebuild
docker-compose build

# Start fresh
docker-compose up -d
```

---

## 📊 Health Check Script

Save as `check-health.sh`:
```bash
#!/bin/bash

echo "Checking CIFT Markets Services..."
echo ""

# API
curl -s http://localhost:8000/health | grep -q "ok" && echo "✅ API" || echo "❌ API"

# PostgreSQL
docker-compose exec -T postgres pg_isready -U cift_user -d cift_markets > /dev/null && echo "✅ PostgreSQL" || echo "❌ PostgreSQL"

# QuestDB
curl -s http://localhost:9003/status > /dev/null && echo "✅ QuestDB" || echo "❌ QuestDB"

# Dragonfly
docker-compose exec -T dragonfly redis-cli ping > /dev/null && echo "✅ Dragonfly" || echo "❌ Dragonfly"

echo ""
echo "Services status check complete!"
```

Run with: `bash check-health.sh`

---

## 🚀 Production Deployment

### Build Production Image
```bash
# Build optimized image
docker-compose -f docker-compose.prod.yml build

# Tag for registry
docker tag cift-api:latest registry.example.com/cift-api:v1.0.0

# Push to registry
docker push registry.example.com/cift-api:v1.0.0
```

### Environment Variables
```bash
# Create production .env
cat > .env.prod << EOF
APP_ENV=production
APP_DEBUG=false
POSTGRES_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 48)
SECRET_KEY=$(openssl rand -base64 48)
EOF
```

---

## 📚 Learn More

- [Docker Compose Docs](https://docs.docker.com/compose/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)

---

**Quick Reference Printed! Keep this handy.** 📌
