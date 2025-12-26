$ErrorActionPreference = "Stop"

Write-Host "Starting Fixes Deployment..."

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

# Deploy Backend Fixes
Copy-File "cift/api/routes/trading.py" "$RemotePath/cift/api/routes"

# Deploy Frontend Fixes
Copy-File "frontend/src/pages/orders/OrdersPage.tsx" "$RemotePath/frontend/src/pages/orders"

# Restart Backend
Write-Host "Restarting Backend Service..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose restart api"

# Rebuild Frontend
Write-Host "Rebuilding Frontend..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Deployment Complete!"
