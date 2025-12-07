# CIFT Markets - Docker Deployment (Rust Included)

**Solution**: Build Rust modules **inside Docker** - no local setup needed!

---

## 🚀 Quick Deploy (3 Commands)

### **Step 1: Build with Rust**
```bash
docker-compose build --no-cache api
```

**What happens:**
- ✅ Downloads Rust compiler in Docker
- ✅ Compiles Rust core modules (order matching, risk, market data)
- ✅ Builds Python PyO3 bindings with maturin
- ✅ Creates production-ready image with all dependencies

**Time:** ~5-10 minutes (first build)

---

### **Step 2: Start All Services**
```bash
docker-compose up -d
```

**What starts:**
```
✅ cift-postgres      (port 5432)
✅ cift-questdb       (port 9000)
✅ cift-clickhouse    (port 8123)
✅ cift-dragonfly     (port 6379)
✅ cift-nats          (port 4222)
✅ cift-prometheus    (port 9090)
✅ cift-grafana       (port 3001)
✅ cift-jaeger        (port 16686)
✅ cift-mlflow        (port 5000)
✅ cift-api           (port 8000) ⚡ With Rust core!
```

**Time:** ~30 seconds

---

### **Step 3: Verify**
```bash
# Check all services running
docker-compose ps

# Test API with full database health check
curl http://localhost:8000/ready

# Should return:
# {
#   "postgres": "healthy",
#   "questdb": "healthy",
#   "dragonfly": "healthy",
#   "clickhouse": "healthy",
#   "nats": "healthy"
# }
```

---

## 🎯 Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | - |
| **Grafana** | http://localhost:3001 | admin/admin |
| **Prometheus** | http://localhost:9090 | - |
| **QuestDB Console** | http://localhost:9000 | - |
| **Jaeger UI** | http://localhost:16686 | - |
| **MLflow** | http://localhost:5000 | - |
| **NATS Monitor** | http://localhost:8222 | - |

---

## 🔧 Common Commands

```bash
# View logs
docker-compose logs -f cift-api

# Restart specific service
docker-compose restart cift-api

# Stop all
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Rebuild single service
docker-compose build api
docker-compose up -d api

# Check resource usage
docker stats
```

---

## ⚡ Performance Verification

### **Test Rust Core is Working**

```bash
# Enter API container
docker exec -it cift-api bash

# Test Rust import
python -c "from cift_core import FastOrderBook, FastMarketData, FastRiskEngine; print('✓ Rust core loaded')"

# Run quick benchmark
python -c "
from cift_core import FastOrderBook
import time

book = FastOrderBook('TEST')
start = time.perf_counter()
for i in range(1000):
    book.add_limit_order(i, 'buy', 100.0 + i * 0.01, 10.0, 1)
end = time.perf_counter()

avg_us = (end - start) * 1_000_000 / 1000
print(f'✓ Order matching: {avg_us:.2f}μs per order')
"

# Expected: < 10μs
```

---

## 🐛 Troubleshooting

### **Build Fails**

```bash
# Clean Docker build cache
docker-compose down
docker system prune -a -f
docker volume prune -f

# Rebuild from scratch
docker-compose build --no-cache api
docker-compose up -d
```

### **Service Won't Start**

```bash
# Check logs for specific service
docker-compose logs cift-clickhouse
docker-compose logs cift-api

# Check if port is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Restart single service
docker-compose restart cift-api
```

### **Database Not Ready**

```bash
# Wait for databases to initialize (first start)
docker-compose logs -f cift-postgres
docker-compose logs -f cift-clickhouse

# Re-check health
curl http://localhost:8000/ready
```

---

## 📊 Resource Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- Disk: 50GB+ (for data)
- SSD preferred

---

## 🎓 Why Docker for Rust?

### **Problems Solved:**

✅ **No local Rust installation** - Built in Docker  
✅ **No virtualenv issues** - Maturin runs in container  
✅ **Consistent builds** - Same result everywhere  
✅ **Production-ready** - Exact same image for dev/staging/prod  
✅ **Easy cleanup** - `docker-compose down` removes everything  

### **Multi-Stage Build:**

1. **Rust Builder** - Compiles Rust code
2. **Python Builder** - Builds PyO3 bindings with maturin
3. **Runtime** - Minimal image with Rust modules

**Final image:** ~500MB (vs ~2GB if not optimized)

---

## ✅ Deployment Checklist

Before deploying:

- [ ] Updated `.env` file with secrets
- [ ] Changed default passwords in `docker-compose.yml`
- [ ] Configured resource limits for production
- [ ] Set up SSL/TLS for public endpoints
- [ ] Configured backup strategy for volumes
- [ ] Set up log aggregation
- [ ] Configured alerts in Prometheus

---

## 🚀 Production Deployment

### **Option 1: Docker Swarm**
```bash
docker swarm init
docker stack deploy -c docker-compose.yml cift
```

### **Option 2: Kubernetes**
```bash
# Convert to k8s manifests
kompose convert

# Deploy
kubectl apply -f ./k8s/
```

### **Option 3: Cloud Run**
```bash
# Build and push to registry
docker build -t gcr.io/YOUR_PROJECT/cift-api .
docker push gcr.io/YOUR_PROJECT/cift-api

# Deploy
gcloud run deploy cift-api --image gcr.io/YOUR_PROJECT/cift-api
```

---

## 🎉 Success!

If everything works:

```bash
$ curl http://localhost:8000/ready
{
  "postgres": "healthy",
  "questdb": "healthy",
  "dragonfly": "healthy",
  "clickhouse": "healthy",
  "nats": "healthy"
}

$ curl http://localhost:8000/docs
# Opens Swagger UI
```

**You now have:**
- ✅ 11 services running
- ✅ Rust core compiled (<10μs order matching)
- ✅ Phase 5-7 ultra-low-latency stack
- ✅ Full monitoring and observability
- ✅ Production-ready infrastructure

**Next:** Build frontend or start trading! 🎯
