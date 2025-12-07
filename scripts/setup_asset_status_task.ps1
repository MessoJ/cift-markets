# PowerShell Script to Setup Windows Scheduled Task for Asset Status Updates
# Run this script as Administrator to setup automatic status updates every 10 minutes

$taskName = "CIFT_AssetStatusUpdate"
$pythonScript = "$PSScriptRoot\update_asset_status.py"
$pythonPath = (Get-Command python).Source

# Create the action
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $pythonScript -WorkingDirectory "$PSScriptRoot"

# Create the trigger (every 10 minutes)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration ([TimeSpan]::MaxValue)

# Create the principal (run even if user is not logged in)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# Create the settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the task
try {
    # Remove existing task if present
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    
    # Register new task
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Update CIFT asset location operational status every 10 minutes"
    
    Write-Host "✅ Scheduled task '$taskName' created successfully!" -ForegroundColor Green
    Write-Host "   Runs every 10 minutes" -ForegroundColor Cyan
    Write-Host "   Script: $pythonScript" -ForegroundColor Cyan
    Write-Host "" 
    Write-Host "To run manually: python $pythonScript" -ForegroundColor Yellow
    Write-Host "To view task: Get-ScheduledTask -TaskName '$taskName'" -ForegroundColor Yellow
    Write-Host "To remove task: Unregister-ScheduledTask -TaskName '$taskName'" -ForegroundColor Yellow
}
catch {
    Write-Host "❌ Error creating scheduled task: $_" -ForegroundColor Red
    exit 1
}
