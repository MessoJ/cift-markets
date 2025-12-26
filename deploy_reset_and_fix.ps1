$ErrorActionPreference = "Stop"

Write-Host "Deploying Market Data Fixes (Finnhub Primary)..."

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
Write-Host "`n=== Deploying Backend Files ===" -ForegroundColor Cyan
Copy-File "cift/services/market_data_service.py" "$RemotePath/cift/services"
Copy-File "cift/api/routes/market_data.py" "$RemotePath/cift/api/routes"
Copy-File "cift/core/trading_queries.py" "$RemotePath/cift/core"
Copy-File "reset_market_data.py" "$RemotePath"

# Restart API to pick up new code
Write-Host "`n=== Restarting API ===" -ForegroundColor Cyan
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose restart api"

Write-Host "`n=== Waiting for API to start ===" -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Run Reset Script inside docker container
Write-Host "`n=== Resetting Market Data (Truncating old data, fetching fresh from Finnhub) ===" -ForegroundColor Cyan
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose exec -T api python /app/reset_market_data.py"

Write-Host "`n=== Deployment Complete! ===" -ForegroundColor Green
Write-Host "Database cleaned and fresh REAL data fetched from Finnhub (FREE tier)."
Write-Host "Quotes will now update in real-time when requested."
