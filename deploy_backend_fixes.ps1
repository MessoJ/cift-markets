# Deploy Backend Fixes to Azure
# This script copies the modified backend files to the Azure server and restarts the API.

$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying backend fixes to $ServerIP..."

# 1. Copy Polygon Service (Indentation Fix)
scp cift/services/polygon_realtime_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/polygon_realtime_service.py

# 2. Copy Portfolio Analytics (DB Column Fix)
scp cift/services/portfolio_analytics.py ${User}@${ServerIP}:${RemotePath}/cift/services/portfolio_analytics.py

# 3. Copy Price Alerts (DB Column Fix)
scp cift/services/price_alerts.py ${User}@${ServerIP}:${RemotePath}/cift/services/price_alerts.py

# 4. Restart API Container
ssh ${User}@${ServerIP} "sudo docker restart cift-api && sleep 5 && sudo docker logs cift-api --tail 50"

Write-Host "Deployment complete! Check the logs above for successful startup."
