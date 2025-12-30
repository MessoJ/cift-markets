$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Switching Azure to Production Mode..."

# 1. Deploy Production Dockerfile
scp Dockerfile ${User}@${ServerIP}:${RemotePath}/Dockerfile

# 2. Update .env safely
Write-Host "Updating .env..."
# We use a simple approach: Append the variable. If it's duplicated, we hope the last one wins or we fix it later.
# Actually, let's just run the sed command.
ssh ${User}@${ServerIP} "sed -i 's/API_DOCKERFILE=Dockerfile.dev/API_DOCKERFILE=Dockerfile/g' ${RemotePath}/.env"

# 3. Rebuild
Write-Host "Rebuilding container (Compiling Rust)..."
# We use ; to avoid && parsing issues in PowerShell if any
ssh ${User}@${ServerIP} "cd ${RemotePath}; sudo docker compose build api; sudo docker compose up -d api"

Write-Host "Done."
