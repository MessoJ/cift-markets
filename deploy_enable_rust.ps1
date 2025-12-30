$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "🔧 Switching Azure to Production Mode (Enabling Rust Compilation)..."

# 1. Deploy Production Dockerfile
Write-Host "📄 Deploying Production Dockerfile..."
scp Dockerfile ${User}@${ServerIP}:${RemotePath}/Dockerfile

# 2. Update .env to use Production Dockerfile
Write-Host "⚙️  Updating Server Configuration..."
ssh ${User}@${ServerIP} "grep -qF 'API_DOCKERFILE' ${RemotePath}/.env || echo 'API_DOCKERFILE=Dockerfile' >> ${RemotePath}/.env"
ssh ${User}@${ServerIP} "sed -i 's/API_DOCKERFILE=Dockerfile.dev/API_DOCKERFILE=Dockerfile/g' ${RemotePath}/.env"

# 3. Rebuild with Rust
Write-Host "🏗️  Rebuilding with Rust Core (This may take a few minutes)..."
$BuildCmd = "cd ${RemotePath} && sudo docker compose build api && sudo docker compose up -d api"
ssh ${User}@${ServerIP} $BuildCmd

Write-Host "✅ Production Deployment Complete! Rust Core is now ACTIVE."
