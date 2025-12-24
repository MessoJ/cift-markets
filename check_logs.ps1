# Check API Logs
$ServerIP = "20.250.40.67"
$User = "azureuser"

ssh ${User}@${ServerIP} "sudo docker logs --tail 50 cift-api"
