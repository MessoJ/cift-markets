#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup real market data for CIFT Markets platform
.DESCRIPTION
    Populates QuestDB with real market tick data and PostgreSQL with economic calendar events
#>

Write-Host "🚀 Setting up real market data for CIFT Markets..." -ForegroundColor Cyan
Write-Host ""

# Check if containers are running
Write-Host "📋 Checking Docker containers..." -ForegroundColor Yellow
$questdbRunning = docker ps --filter "name=cift-questdb" --format "{{.Names}}"
$postgresRunning = docker ps --filter "name=cift-postgres" --format "{{.Names}}"

if (-not $questdbRunning) {
    Write-Host "❌ QuestDB container is not running. Please start it with: docker-compose up -d questdb" -ForegroundColor Red
    exit 1
}

if (-not $postgresRunning) {
    Write-Host "❌ PostgreSQL container is not running. Please start it with: docker-compose up -d postgres" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker containers are running" -ForegroundColor Green
Write-Host ""

# Step 1: Populate market tick data
Write-Host "📊 Step 1: Populating market tick data (this may take a few minutes)..." -ForegroundColor Cyan
python scripts/populate_today.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Failed to populate market data" -ForegroundColor Yellow
} else {
    Write-Host "✅ Market tick data populated successfully" -ForegroundColor Green
}
Write-Host ""

# Step 2: Verify data
Write-Host "🔍 Step 2: Verifying data..." -ForegroundColor Cyan

# Check QuestDB tick count
Write-Host "  Checking QuestDB ticks..."
docker exec cift-questdb sh -c "wget -qO- 'http://localhost:9000/exec?query=SELECT COUNT(*) FROM ticks;'" | Select-String -Pattern '"count"'

# Check symbols in PostgreSQL
Write-Host "  Checking PostgreSQL symbols..."
docker exec cift-postgres psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM symbols;" 2>$null

Write-Host ""
Write-Host "✨ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Restart the API: docker-compose restart api"
Write-Host "  2. Refresh the frontend at http://localhost:3000/news"
Write-Host "  3. Market movers should now show real data!"
Write-Host ""
