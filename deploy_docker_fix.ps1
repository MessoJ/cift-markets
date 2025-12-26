$ErrorActionPreference = "Stop"

Write-Host "Starting Docker Compose Fix Deployment..."

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

# Deploy docker-compose.yml
Copy-File "docker-compose.yml" "$RemotePath"

# Restart Frontend
Write-Host "Restarting Frontend Service..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && sudo docker compose up -d --force-recreate frontend"

Write-Host "Deployment Complete!"
