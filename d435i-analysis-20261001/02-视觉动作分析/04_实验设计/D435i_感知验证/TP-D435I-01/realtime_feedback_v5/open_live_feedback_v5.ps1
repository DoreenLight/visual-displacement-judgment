$ErrorActionPreference = 'Stop'
try {
    $taskStart = Join-Path $PSScriptRoot 'start_live_feedback_v5.ps1'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $taskStart
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to start D435i service.'
    }
    $taskStatus = Invoke-RestMethod 'http://127.0.0.1:8090/telemetry' -TimeoutSec 5
    if ($taskStatus.live_version -ne 'feedback/v5') {
        throw 'Port 8090 is not feedback/v5.'
    }
    Start-Process 'http://localhost:8090/'
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'D435i Error') | Out-Null
    exit 1
}
