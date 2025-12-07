<#
.SYNOPSIS
    Provisions Azure Infrastructure for CIFT Markets.
.DESCRIPTION
    This script uses the Azure CLI (az) to:
    1. Create a Resource Group.
    2. Create a Virtual Machine (Ubuntu 22.04 LTS).
    3. Open necessary ports.
    4. Output the connection details.
.NOTES
    Requires Azure CLI to be installed and logged in (az login).
#>

$ErrorActionPreference = "Stop"

# Configuration
$ResourceGroup = "cift-resources"
$Location = "eastus"
$VmName = "cift-production"
$Image = "Ubuntu2204"
$Size = "Standard_D4s_v5" # 4 vCPUs, 16 GiB RAM (Recommended for full stack)
$AdminUser = "azureuser"

Write-Host "🚀 Starting Azure Provisioning in $Location..." -ForegroundColor Cyan

# 1. Create Resource Group
Write-Host "📦 Creating Resource Group '$ResourceGroup'..."
az group create --name $ResourceGroup --location $Location --output none
Write-Host "   ✅ Resource Group created." -ForegroundColor Green

# 2. Create VM
Write-Host "💻 Creating Virtual Machine '$VmName' ($Size)..."
Write-Host "   This may take a few minutes..."

# Create VM and generate SSH keys automatically if missing
$VmJson = az vm create `
    --resource-group $ResourceGroup `
    --name $VmName `
    --image $Image `
    --size $Size `
    --admin-username $AdminUser `
    --generate-ssh-keys `
    --public-ip-sku Standard `
    --output json | ConvertFrom-Json

Write-Host "   ✅ VM Created." -ForegroundColor Green

# 3. Open Ports
Write-Host "🛡️  Opening Ports..."
$Ports = @("80", "443", "3000", "8000", "9000")
$Priority = 1000

foreach ($Port in $Ports) {
    Write-Host "   Opening port $Port..."
    az vm open-port --resource-group $ResourceGroup --name $VmName --port $Port --priority $Priority --output none
    $Priority += 10
}
Write-Host "   ✅ Ports opened." -ForegroundColor Green

# 4. Get Details
$PublicIp = $VmJson.publicIpAddress
$PrivateKeyPath = "~/.ssh/id_rsa" # Default location for az vm create generated keys

Write-Host "`n✨ Provisioning Complete!" -ForegroundColor Green
Write-Host "--------------------------------------------------"
Write-Host "VM Name:     $VmName"
Write-Host "Public IP:   $PublicIp"
Write-Host "User:        $AdminUser"
Write-Host "SSH Key:     $PrivateKeyPath (Default)"
Write-Host "--------------------------------------------------"
Write-Host "To connect:"
Write-Host "ssh $AdminUser@$PublicIp"
Write-Host "--------------------------------------------------"
