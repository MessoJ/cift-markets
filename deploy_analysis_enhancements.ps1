# Deploy Analysis Enhancements to Azure

$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Deploying Analysis Enhancements to $ServerIP..."

# 1. Copy QuickAnalyzeButton.tsx (Fixed navigation and added compact prop)
scp frontend/src/components/analysis/QuickAnalyzeButton.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/components/analysis/QuickAnalyzeButton.tsx

# 2. Copy AnalysisPage.tsx (Added Portfolio Context and News)
scp frontend/src/pages/analysis/AnalysisPage.tsx ${User}@${ServerIP}:${RemotePath}/frontend/src/pages/analysis/AnalysisPage.tsx

# 3. Rebuild Frontend Container on Server
Write-Host "Rebuilding frontend container..."
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose build frontend && sudo docker compose up -d frontend"

Write-Host "Deployment complete!"
