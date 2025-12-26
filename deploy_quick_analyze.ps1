# Deploy Quick Analyze Feature to Azure

$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying Quick Analyze feature to $ServerIP..."

# 1. Copy QuickAnalyzeButton.tsx
# First create the directory if it doesn't exist (it might not on the server if it's new)
ssh ${User}@${ServerIP} "mkdir -p ${RemotePath}/frontend/src/components/analysis"
scp frontend/src/components/analysis/QuickAnalyzeButton.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/components/analysis/QuickAnalyzeButton.tsx

# 2. Copy WatchlistsPage.tsx
scp frontend/src/pages/watchlists/WatchlistsPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/watchlists/WatchlistsPage.tsx

# 3. Copy PortfolioPage.tsx
scp frontend/src/pages/portfolio/PortfolioPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/portfolio/PortfolioPage.tsx

# 4. Rebuild Frontend Container on Server
Write-Host "Rebuilding frontend container..."
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Deployment complete!"
