$ErrorActionPreference = "Stop"

Write-Host "Starting Frontend Fixes Deployment..."

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

# Deploy Files
Copy-File "frontend/src/components/ui/Table.tsx" "$RemotePath/frontend/src/components/ui"
Copy-File "frontend/src/pages/orders/OrdersPage.tsx" "$RemotePath/frontend/src/pages/orders"
Copy-File "frontend/src/pages/dashboard/DashboardPage.tsx" "$RemotePath/frontend/src/pages/dashboard"
Copy-File "frontend/src/pages/news/NewsPage.tsx" "$RemotePath/frontend/src/pages/news"

# Rebuild Frontend
Write-Host "Rebuilding Frontend..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Deployment Complete!"
