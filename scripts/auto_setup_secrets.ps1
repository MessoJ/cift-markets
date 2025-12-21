# Automated Azure Secrets Setup
# Usage: ./scripts/auto_setup_secrets.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== CIFT Markets: Automated Azure Secrets Setup ===" -ForegroundColor Cyan

# 1. Get Azure Subscription ID
Write-Host "1. Fetching Azure Subscription..." -ForegroundColor Yellow
try {
    $subJson = az account show --output json | ConvertFrom-Json
    $subId = $subJson.id
    Write-Host "   Found Subscription ID: $subId" -ForegroundColor Green
    
    # Set AZURE_SUBSCRIPTION_ID
    $subId | gh secret set AZURE_SUBSCRIPTION_ID
    Write-Host "   ✅ AZURE_SUBSCRIPTION_ID set." -ForegroundColor Green
} catch {
    Write-Error "Failed to get Azure account. Please run 'az login' first."
}

# 2. Generate Service Principal Credentials
Write-Host "2. Generating Service Principal (this may take a moment)..." -ForegroundColor Yellow
try {
    # Create SP and get JSON output
    $spName = "cift-ml-sp-$(Get-Random)"
    $cmd = "az ad sp create-for-rbac --name '$spName' --role contributor --scopes /subscriptions/$subId --sdk-auth --output json"
    Write-Host "   Running: $cmd" -ForegroundColor Gray
    
    $creds = az ad sp create-for-rbac --name $spName --role contributor --scopes /subscriptions/$subId --sdk-auth --output json
    
    if (-not $creds) {
        throw "Failed to generate credentials. Output was empty."
    }

    # Set AZURE_CREDENTIALS
    $creds | gh secret set AZURE_CREDENTIALS
    Write-Host "   ✅ AZURE_CREDENTIALS set." -ForegroundColor Green
} catch {
    Write-Error "Failed to generate Service Principal: $_"
}

# 3. Set other secrets (defaults)
Write-Host "3. Setting configuration secrets..." -ForegroundColor Yellow

"cift-ml-rg" | gh secret set AZURE_RESOURCE_GROUP
Write-Host "   ✅ AZURE_RESOURCE_GROUP set to 'cift-ml-rg'." -ForegroundColor Green

"cift-ml-workspace" | gh secret set AZURE_WORKSPACE_NAME
Write-Host "   ✅ AZURE_WORKSPACE_NAME set to 'cift-ml-workspace'." -ForegroundColor Green

"./data/market_data.csv" | gh secret set AZURE_DATA_PATH
Write-Host "   ✅ AZURE_DATA_PATH set to './data/market_data.csv'." -ForegroundColor Green

Write-Host ""
Write-Host "🎉 All secrets configured successfully!" -ForegroundColor Cyan
Write-Host "Triggering a new workflow run..." -ForegroundColor Cyan

# Trigger the workflow
gh workflow run azure-train.yml --ref main
Write-Host "🚀 Workflow triggered! Check status with 'gh run list'." -ForegroundColor Green
