# Helper script to set GitHub Secrets for Azure CI/CD
# Usage: ./set_azure_secrets.ps1

Write-Host "=== CIFT Markets: Azure Secrets Setup ===" -ForegroundColor Cyan
Write-Host "This script will help you set the necessary GitHub Secrets for the Azure training pipeline."
Write-Host "Prerequisites:"
Write-Host "1. GitHub CLI (gh) installed and authenticated (run 'gh auth login')"
Write-Host "2. Azure CLI (az) installed and authenticated (run 'az login')"
Write-Host ""

# Check if gh is installed
if (-not (Get-Command "gh" -ErrorAction SilentlyContinue)) {
    Write-Error "GitHub CLI (gh) is not installed. Please install it first."
    exit 1
}

# 1. Azure Credentials
Write-Host "1. AZURE_CREDENTIALS" -ForegroundColor Yellow
Write-Host "   To generate this, run: az ad sp create-for-rbac --name 'cift-ml-sp' --role contributor --scopes /subscriptions/<SUBSCRIPTION_ID> --sdk-auth"
$creds = Read-Host "   Paste the full JSON output here (or press Enter to skip)"
if (-not [string]::IsNullOrWhiteSpace($creds)) {
    $creds | gh secret set AZURE_CREDENTIALS
    Write-Host "   ✅ AZURE_CREDENTIALS set." -ForegroundColor Green
}

# 2. Subscription ID
Write-Host "2. AZURE_SUBSCRIPTION_ID" -ForegroundColor Yellow
$subId = Read-Host "   Enter your Azure Subscription ID (or press Enter to skip)"
if (-not [string]::IsNullOrWhiteSpace($subId)) {
    $subId | gh secret set AZURE_SUBSCRIPTION_ID
    Write-Host "   ✅ AZURE_SUBSCRIPTION_ID set." -ForegroundColor Green
}

# 3. Resource Group
Write-Host "3. AZURE_RESOURCE_GROUP" -ForegroundColor Yellow
$rg = Read-Host "   Enter Resource Group Name [default: cift-ml-rg]"
if ([string]::IsNullOrWhiteSpace($rg)) { $rg = "cift-ml-rg" }
$rg | gh secret set AZURE_RESOURCE_GROUP
Write-Host "   ✅ AZURE_RESOURCE_GROUP set to '$rg'." -ForegroundColor Green

# 4. Workspace Name
Write-Host "4. AZURE_WORKSPACE_NAME" -ForegroundColor Yellow
$ws = Read-Host "   Enter Azure ML Workspace Name [default: cift-ml-workspace]"
if ([string]::IsNullOrWhiteSpace($ws)) { $ws = "cift-ml-workspace" }
$ws | gh secret set AZURE_WORKSPACE_NAME
Write-Host "   ✅ AZURE_WORKSPACE_NAME set to '$ws'." -ForegroundColor Green

# 5. Data Path
Write-Host "5. AZURE_DATA_PATH" -ForegroundColor Yellow
$dp = Read-Host "   Enter Data Path (e.g., azureml:data@1 or https://.../data.csv) [default: ./data/market_data.csv]"
if ([string]::IsNullOrWhiteSpace($dp)) { $dp = "./data/market_data.csv" }
$dp | gh secret set AZURE_DATA_PATH
Write-Host "   ✅ AZURE_DATA_PATH set to '$dp'." -ForegroundColor Green

Write-Host ""
Write-Host "🎉 All secrets processed!" -ForegroundColor Cyan
Write-Host "You can now push to main to trigger the training pipeline."
