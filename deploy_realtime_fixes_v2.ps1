$ErrorActionPreference = "Stop"

Write-Host "Deploying Real-Time Market Data Fixes..."

# Define variables
$User = "azureuser"
$HostName = "20.250.40.67"
$KeyFile = "temp_key"
$RemotePath = "~/cift-markets"

# Function to copy file
function Copy-File {
    param (
        [string]$LocalPath,
        [string]$RemoteDir
    )
    $FileName = Split-Path $LocalPath -Leaf
    Write-Host "Copying $LocalPath to $RemoteDir..."
    scp -i $KeyFile -o StrictHostKeyChecking=no $LocalPath "${User}@${HostName}:${RemoteDir}/${FileName}"
}

# Deploy Backend Files
Copy-File "cift/services/polygon_realtime_service.py" "$RemotePath/cift/services"

# Deploy Frontend Files
Copy-File "frontend/src/stores/marketData.store.ts" "$RemotePath/frontend/src/stores"
Copy-File "frontend/src/pages/trading/TradingPage.tsx" "$RemotePath/frontend/src/pages/trading"

# Restart Services
Write-Host "Restarting Backend and Rebuilding Frontend..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose restart api && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Deployment Complete!"
