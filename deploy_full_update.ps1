$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying Backend Updates..."
scp cift/services/market_data_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/market_data_service.py
scp cift/services/polygon_realtime_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/polygon_realtime_service.py
scp cift/services/finnhub_realtime_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/finnhub_realtime_service.py
scp cift/api/routes/market_data.py ${User}@${ServerIP}:${RemotePath}/cift/api/routes/market_data.py
scp cift/api/routes/trading.py ${User}@${ServerIP}:${RemotePath}/cift/api/routes/trading.py
scp cift/core/trading_queries.py ${User}@${ServerIP}:${RemotePath}/cift/core/trading_queries.py

Write-Host "Restarting Backend API..."
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose restart api"

Write-Host "Deploying Frontend Updates..."
# Create directory if it doesn't exist
ssh ${User}@${ServerIP} "mkdir -p ${RemotePath}/frontend/src/components/trading"

# Copy modified/new files
scp frontend/src/lib/api/client.ts ${User}@${ServerIP}:${RemotePath}/frontend/src/lib/api/client.ts
scp frontend/src/components/trading/CompanyProfileWidget.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/components/trading/CompanyProfileWidget.tsx
scp frontend/src/pages/trading/TradingPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/trading/TradingPage.tsx

Write-Host "Rebuilding Frontend on Server..."
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Full Deployment Complete!"