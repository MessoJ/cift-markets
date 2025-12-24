# Run DB Fix on Azure Server

$ServerIP = "20.250.40.67"
$User = "azureuser"
$RemotePath = "~/cift-markets"

Write-Host "Copying fix script to server..."
scp fix_notifications_table.py ${User}@${ServerIP}:${RemotePath}/fix_notifications_table.py

Write-Host "Running fix script inside API container..."
ssh ${User}@${ServerIP} "cd ${RemotePath} && sudo docker compose cp fix_notifications_table.py api:/app/fix_notifications_table.py && sudo docker compose exec api python /app/fix_notifications_table.py"

Write-Host "DB Fix Complete!"
