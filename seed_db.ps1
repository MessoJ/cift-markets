$ErrorActionPreference = "Stop"

Write-Host "Seeding Market Data..."

# Define variables
$User = "azureuser"
$HostName = "20.250.40.67"
$KeyFile = "temp_key"
$RemotePath = "~/cift-markets"

# Copy seed file
Write-Host "Copying seed_market_data.sql..."
scp -i $KeyFile -o StrictHostKeyChecking=no seed_market_data.sql "${User}@${HostName}:${RemotePath}/seed_market_data.sql"

# Run seed command
Write-Host "Running seed command..."
ssh -i $KeyFile -o StrictHostKeyChecking=no "${User}@${HostName}" "cd $RemotePath && cat seed_market_data.sql | sudo docker exec -i cift-postgres psql -U cift_user -d cift_db"

Write-Host "Seeding Complete!"
