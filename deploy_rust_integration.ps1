$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"
$KeyFile = "~/.ssh/id_rsa" # Assuming default key location, or we can omit -i if agent is running

Write-Host "🚀 Starting Deployment: Rust Core Integration & Valuation Update"

# 1. Deploy Documentation
Write-Host "📄 Deploying Documentation..."
scp docs/CIFT_MARKETS_VALUATION_V2.md ${User}@${ServerIP}:${RemotePath}/docs/CIFT_MARKETS_VALUATION_V2.md
scp docs/FULL_PLATFORM_AUDIT.md ${User}@${ServerIP}:${RemotePath}/docs/FULL_PLATFORM_AUDIT.md

# 2. Deploy Python ML Engine
Write-Host "🧠 Deploying Institutional Engine..."
scp cift/ml/institutional_production.py ${User}@${ServerIP}:${RemotePath}/cift/ml/institutional_production.py

# 3. Deploy Rust Core (Source Only)
Write-Host "🦀 Deploying Rust Core..."
# Ensure remote directory exists
ssh ${User}@${ServerIP} "mkdir -p ${RemotePath}/rust_core/src"
scp rust_core/Cargo.toml ${User}@${ServerIP}:${RemotePath}/rust_core/Cargo.toml
scp rust_core/pyproject.toml ${User}@${ServerIP}:${RemotePath}/rust_core/pyproject.toml
scp -r rust_core/src/* ${User}@${ServerIP}:${RemotePath}/rust_core/src/

# 4. Trigger Remote Build
Write-Host "🏗️  Building Docker Container on Azure (This compiles the Rust code)..."
$BuildCmd = "cd ${RemotePath} && sudo docker compose build api && sudo docker compose up -d api"
ssh ${User}@${ServerIP} $BuildCmd

Write-Host "✅ Deployment Complete! The API is now running with the Rust Execution Engine."
