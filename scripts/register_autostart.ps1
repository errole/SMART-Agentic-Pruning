# Survive reboots, Windows Update restarts and power cuts: registers a Task
# Scheduler task that resumes the experiment matrix every time you log on.
# Duplicate starts are harmless (the matrix lock makes a second copy exit).
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_autostart.ps1 -Config configs\base.yaml
#   # remove later with:  Unregister-ScheduledTask -TaskName "CAP-Prune matrix" -Confirm:$false
param([string]$Config = "configs\base.yaml")
$root = Split-Path $PSScriptRoot -Parent
$script = Join-Path $root "scripts\run_matrix.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -Config `"$Config`"" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 30)
Register-ScheduledTask -TaskName "CAP-Prune matrix" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Resumes the CAP-Prune experiment matrix ($Config) after logon/reboot" -Force | Out-Null
Write-Host "Registered 'CAP-Prune matrix' (runs at logon). Start it now with:"
Write-Host "  Start-ScheduledTask -TaskName 'CAP-Prune matrix'"
