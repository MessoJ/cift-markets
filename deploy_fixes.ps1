# Deploy Fixes to Azure
# This script copies the modified files to the Azure server and restarts the API.

$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying fixes to $ServerIP..."

# 1. Copy Polygon Service (Real-time data fix)
scp cift/services/polygon_realtime_service.py ${User}@${ServerIP}:${RemotePath}/cift/services/polygon_realtime_service.py

# 2. Copy Main API Entrypoint (Market data loop fix + Logging)
scp cift/api/main.py ${User}@${ServerIP}:${RemotePath}/cift/api/main.py

# 3. Restart API Container
ssh ${User}@${ServerIP} "sudo docker restart cift-api && sleep 5 && sudo docker logs cift-api --tail 50"

Write-Host "Deployment complete! Check the logs above for 'Starting real-time data fetch' and URL configuration."
