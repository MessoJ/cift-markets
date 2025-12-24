$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying Market Data Services..."

# Copy files
scp cift/services/finnhub_realtime_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/finnhub_realtime_service.py
scp cift/services/alltick_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/alltick_service.py
scp cift/services/polygon_realtime_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/polygon_realtime_service.py
scp cift/services/market_data_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/market_data_service.py

# Update .env on server (Check if key exists first to avoid duplicates, simple append for now)
ssh ${User}@${ServerIP} "grep -qF 'ALLTICK_API_KEY' ${RemotePath}/.env || echo 'ALLTICK_API_KEY=fd881057aae7a4b2045a1fb659f7a670-c-app' >> ${RemotePath}/.env"

# Restart API
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose restart api"

Write-Host "Deployment Complete!"