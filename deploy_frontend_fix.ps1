$ErrorActionPreference = "Stop"

Write-Host "Deploying Frontend Fixes..."

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

# Deploy Fixed Files
Copy-File "frontend/src/lib/api/client.ts" "$RemotePath/frontend/src/lib/api"
Copy-File "frontend/src/pages/funding/tabs/PaymentMethodsTab.tsx" "$RemotePath/frontend/src/pages/funding/tabs"

# Rebuild Frontend
Write-Host "Rebuilding Frontend Container..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Deployment Complete!"
