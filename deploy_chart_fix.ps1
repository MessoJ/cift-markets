$ErrorActionPreference = "Stop"

Write-Host "Deploying Chart Data Fixes..."

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
Copy-File "cift/api/routes/market_data.py" "$RemotePath/cift/api/routes"
Copy-File "cift/core/trading_queries.py" "$RemotePath/cift/core"

# Restart Services
Write-Host "Restarting Backend API..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose restart api"

Write-Host "Deployment Complete! Charts should now auto-fetch missing data."
