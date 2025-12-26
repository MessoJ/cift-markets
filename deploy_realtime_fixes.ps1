$ErrorActionPreference = "Stop"

Write-Host "Starting Real-Time Data Fixes Deployment..."

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

# Deploy Backend Services
Copy-File "cift/services/news_service.py" "$RemotePath/cift/services"

# Deploy Backend Routes
Copy-File "cift/api/routes/market_data.py" "$RemotePath/cift/api/routes"

# Restart Backend
Write-Host "Restarting Backend Service..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose restart api"

Write-Host "Deployment Complete!"
