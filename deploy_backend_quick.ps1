# Deploy Backend Fixes (Quick)
$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying backend fixes..."

# Copy alerts.py
scp cift/api/routes/alerts.py ${User}@${ServerIP}:${RemotePath}/cift/api/routes/alerts.py

# Restart API
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose restart api"

Write-Host "Backend deployment complete!"
