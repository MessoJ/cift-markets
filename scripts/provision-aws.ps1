<#
.SYNOPSIS
    Provisions AWS Infrastructure for CIFT Markets.
.DESCRIPTION
    This script uses the AWS CLI to:
    1. Create a Key Pair (cift-key) if it doesn't exist.
    2. Create a Security Group (cift-sg) with necessary ports.
    3. Launch an EC2 instance (Ubuntu 22.04 LTS).
    4. Output the connection details.
.NOTES
    Requires AWS CLI to be installed and configured.
#>

$ErrorActionPreference = "Stop"

# Configuration
$Region = "us-east-1"
$InstanceType = "t3.xlarge" # Recommended for full stack
$KeyName = "cift-key"
$GroupName = "cift-sg"
$AmiNamePattern = "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"

Write-Host "🚀 Starting AWS Provisioning in $Region..." -ForegroundColor Cyan

# 1. Check/Create Key Pair
Write-Host "🔑 Checking Key Pair '$KeyName'..."
$KeyExists = aws ec2 describe-key-pairs --key-names $KeyName --region $Region --query "KeyPairs[0].KeyName" --output text 2>$null
if (-not $KeyExists) {
    Write-Host "   Creating new key pair..."
    aws ec2 create-key-pair --key-name $KeyName --query "KeyMaterial" --output text --region $Region > "$KeyName.pem"
    Write-Host "   ✅ Key pair created and saved to $KeyName.pem" -ForegroundColor Green
} else {
    Write-Host "   ✅ Key pair already exists." -ForegroundColor Green
}

# 2. Check/Create Security Group
Write-Host "🛡️  Checking Security Group '$GroupName'..."
$GroupId = aws ec2 describe-security-groups --group-names $GroupName --region $Region --query "SecurityGroups[0].GroupId" --output text 2>$null

if (-not $GroupId) {
    Write-Host "   Creating new security group..."
    $GroupId = aws ec2 create-security-group --group-name $GroupName --description "Security group for CIFT Markets" --region $Region --query "GroupId" --output text
    
    Write-Host "   Authorizing ports..."
    # SSH
    aws ec2 authorize-security-group-ingress --group-id $GroupId --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $Region
    # HTTP/HTTPS
    aws ec2 authorize-security-group-ingress --group-id $GroupId --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $Region
    aws ec2 authorize-security-group-ingress --group-id $GroupId --protocol tcp --port 443 --cidr 0.0.0.0/0 --region $Region
    # App Ports (Frontend, API, QuestDB Console)
    aws ec2 authorize-security-group-ingress --group-id $GroupId --protocol tcp --port 3000 --cidr 0.0.0.0/0 --region $Region
    aws ec2 authorize-security-group-ingress --group-id $GroupId --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $Region
    aws ec2 authorize-security-group-ingress --group-id $GroupId --protocol tcp --port 9000 --cidr 0.0.0.0/0 --region $Region
    
    Write-Host "   ✅ Security Group created ($GroupId)." -ForegroundColor Green
} else {
    Write-Host "   ✅ Security Group already exists ($GroupId)." -ForegroundColor Green
}

# 3. Find AMI
Write-Host "💿 Finding latest Ubuntu 22.04 AMI..."
$AmiId = aws ec2 describe-images --owners 099720109477 --filters "Name=name,Values=$AmiNamePattern" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text --region $Region

if (-not $AmiId -or $AmiId -eq "None") {
    Write-Error "Could not find Ubuntu AMI."
}
Write-Host "   Found AMI: $AmiId"

# 4. Launch Instance
Write-Host "🚀 Launching EC2 Instance ($InstanceType)..."
$InstanceId = aws ec2 run-instances `
    --image-id $AmiId `
    --count 1 `
    --instance-type $InstanceType `
    --key-name $KeyName `
    --security-group-ids $GroupId `
    --block-device-mappings "[{\`"DeviceName\`":\`"/dev/sda1\`",\`"Ebs\`":{\`"VolumeSize\`":50,\`"VolumeType\`":\`"gp3\`"}}]" `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cift-production}]" `
    --query "Instances[0].InstanceId" `
    --output text `
    --region $Region

Write-Host "   Instance launched: $InstanceId"
Write-Host "   Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $InstanceId --region $Region

# 5. Get Public IP
$PublicIp = aws ec2 describe-instances --instance-ids $InstanceId --region $Region --query "Reservations[0].Instances[0].PublicIpAddress" --output text

Write-Host "`n✨ Provisioning Complete!" -ForegroundColor Green
Write-Host "--------------------------------------------------"
Write-Host "Instance ID: $InstanceId"
Write-Host "Public IP:   $PublicIp"
Write-Host "Key File:    $PWD\$KeyName.pem"
Write-Host "--------------------------------------------------"
Write-Host "To connect:"
Write-Host "ssh -i $KeyName.pem ubuntu@$PublicIp"
Write-Host "--------------------------------------------------"
