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

Write-Host "Starting Payment Integration Deployment..."

# 1. Backend Updates
Write-Host "Deploying Backend Files..."
Run-SCP "cift/api/routes/funding.py" "${RemotePath}/cift/api/routes/"
Run-SCP "cift/services/payment_processors" "${RemotePath}/cift/services/"
Run-SCP "cift/services/plaid_service.py" "${RemotePath}/cift/services/"
Run-SCP "cift/services/fmp_economic_calendar.py" "${RemotePath}/cift/services/"
Run-SCP ".env.example" "${RemotePath}/.env.example"

# 2. Frontend Updates
Write-Host "Deploying Frontend Files..."
Run-SCP "frontend/src/pages/funding" "${RemotePath}/frontend/src/pages/"
Run-SCP "frontend/src/lib/api/client.ts" "${RemotePath}/frontend/src/lib/api/"
Run-SCP "frontend/src/components/market/MarketIndicesBar.tsx" "${RemotePath}/frontend/src/components/market/"
Run-SCP "frontend/src/pages/funding/tabs/PaymentMethodsTab.tsx" "${RemotePath}/frontend/src/pages/funding/tabs/"

# 3. Restart Services
Write-Host "Restarting Backend Service..."
$RestartCmd = "cd ${RemotePath} && sudo docker compose restart api"
Run-SSH $RestartCmd

Write-Host "Rebuilding Frontend..."
$BuildCmd = "cd ${RemotePath} && sudo docker compose build frontend && sudo docker compose up -d frontend"
Run-SSH $BuildCmd

Write-Host "Deployment Complete!"
Write-Host "REMINDER: SSH into the server and update .env with real API keys from .env.example"
