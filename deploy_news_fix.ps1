$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"
$KeyFile = "temp_key"

# Function to run SSH command
function Run-SSH {
    param($Cmd)
    Write-Host "Running: $Cmd"
    ssh -i $KeyFile -o StrictHostKeyChecking=no ${User}@${ServerIP} $Cmd
}

# Function to run SCP
function Run-SCP {
    param($Source, $Dest)
    Write-Host "Copying $Source to $Dest"
    scp -i $KeyFile -o StrictHostKeyChecking=no -r $Source ${User}@${ServerIP}:${Dest}
}

Write-Host "Starting News Fix Deployment..."

# 1. Backend Updates
Write-Host "Deploying Backend Files..."
Run-SCP "cift/services/news_service.py" "${RemotePath}/cift/services/"
Run-SCP "cift/api/routes/news.py" "${RemotePath}/cift/api/routes/"

# 2. Frontend Updates
Write-Host "Deploying Frontend Files..."
Run-SCP "frontend/src/pages/news/NewsPage.tsx" "${RemotePath}/frontend/src/pages/news/"

# 3. Restart Services
Write-Host "Restarting Backend Service..."
$RestartCmd = "cd ${RemotePath} && sudo docker compose restart api"
Run-SSH $RestartCmd

Write-Host "Rebuilding Frontend..."
$BuildCmd = "cd ${RemotePath} && sudo docker compose build frontend && sudo docker compose up -d frontend"
Run-SSH $BuildCmd

Write-Host "Deployment Complete!"
Write-Host "REMINDER: Ensure FMP_API_KEY is set in .env on the server!"
