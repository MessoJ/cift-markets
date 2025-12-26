# Deploy Inline Analyzers to Azure

$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying Inline Analyzers to $ServerIP..."

# 1. Create the analysis directory on server if needed
ssh ${User}@${ServerIP} "mkdir -p ${RemotePath}/frontend/src/components/analysis"

# 2. Copy all analyzer components
Write-Host "Copying InlineAnalyzer.tsx..."
scp frontend/src/components/analysis/InlineAnalyzer.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/components/analysis/InlineAnalyzer.tsx

Write-Host "Copying PortfolioAnalyzer.tsx..."
scp frontend/src/components/analysis/PortfolioAnalyzer.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/components/analysis/PortfolioAnalyzer.tsx

Write-Host "Copying WatchlistAnalyzer.tsx..."
scp frontend/src/components/analysis/WatchlistAnalyzer.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/components/analysis/WatchlistAnalyzer.tsx

# 3. Copy updated pages
Write-Host "Copying TradingPage.tsx..."
scp frontend/src/pages/trading/TradingPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/trading/TradingPage.tsx

Write-Host "Copying PortfolioPage.tsx..."
scp frontend/src/pages/portfolio/PortfolioPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/portfolio/PortfolioPage.tsx

Write-Host "Copying WatchlistsPage.tsx..."
scp frontend/src/pages/watchlists/WatchlistsPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/watchlists/WatchlistsPage.tsx

# 4. Rebuild Frontend Container on Server
Write-Host "Rebuilding frontend container..."
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Deployment complete!"
