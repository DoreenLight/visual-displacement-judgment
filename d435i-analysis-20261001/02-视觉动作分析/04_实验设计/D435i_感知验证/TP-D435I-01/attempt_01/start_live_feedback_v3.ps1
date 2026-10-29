$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$taskPython = Join-Path $taskWorkspace '.venv\Scripts\python.exe'
$taskApp = Join-Path $PSScriptRoot 'live_feedback_v3.py'
$taskUrl = 'http://localhost:8090'
$taskReady = $false
$taskStatus = $null
try {
    $taskStatus = Invoke-RestMethod "$taskUrl/telemetry" -TimeoutSec 2
} catch { }
if ($null -ne $taskStatus) {
    if ($null -ne $taskStatus.task_catalog) { $taskReady = $true }
    elseif ($null -ne $taskStatus.result_age_ms -and $null -ne $taskStatus.camera) {
        Invoke-RestMethod "$taskUrl/stop" -Method Post -Body '{}' -ContentType 'application/json' | Out-Null
        for ($taskWait = 0; $taskWait -lt 80; $taskWait++) {
            Start-Sleep -Milliseconds 250
            try { Invoke-RestMethod "$taskUrl/telemetry" -TimeoutSec 1 | Out-Null }
            catch { break }
        }
    } else { throw 'Port 8090 belongs to another service.' }
}
if (-not $taskReady) {
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', ('"' + $taskApp + '"')) -WorkingDirectory $taskWorkspace -WindowStyle Hidden -RedirectStandardError (Join-Path $PSScriptRoot 'live_feedback_v3.stderr.log') -RedirectStandardOutput (Join-Path $PSScriptRoot 'live_feedback_v3.stdout.log') -PassThru
    for ($taskAttempt = 0; $taskAttempt -lt 80; $taskAttempt++) {
        Start-Sleep -Milliseconds 250
        if ($taskProcess.HasExited) { throw 'Process stopped. Check live_feedback_v3.stderr.log.' }
        try {
            $taskStatus = Invoke-RestMethod "$taskUrl/telemetry" -TimeoutSec 1
            $taskReady = $null -ne $taskStatus.task_catalog
            if ($taskReady) { break }
        } catch { }
    }
    if (-not $taskReady) { throw 'Local interface did not become ready.' }
}
Start-Process $taskUrl
Write-Output "Eight tasks running: $taskUrl"
