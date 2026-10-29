$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$taskPython = Join-Path $taskWorkspace '.venv\Scripts\python.exe'
$taskApp = Join-Path $PSScriptRoot 'live_feedback_v5.py'
$taskUrl = 'http://127.0.0.1:8090'
$taskStatus = $null
try { $taskStatus = Invoke-RestMethod "$taskUrl/telemetry" -TimeoutSec 2 } catch { }
if ($taskStatus.live_version -eq 'feedback/v5') { Write-Output 'v5 already running'; exit 0 }
if ($null -ne $taskStatus) {
 if ($null -eq $taskStatus.task_catalog) { throw 'Port 8090 belongs to an unrecognized service.' }
 if ($taskStatus.training -or $taskStatus.recording -or $taskStatus.finalizing) { throw 'Trial active. Finish and save before upgrading.' }
 Invoke-RestMethod "$taskUrl/stop" -Method Post -Body '{}' -ContentType 'application/json' | Out-Null
 for ($taskWait=0;$taskWait -lt 60;$taskWait++) {
  Start-Sleep -Milliseconds 250
  try { Invoke-RestMethod "$taskUrl/telemetry" -TimeoutSec 1 | Out-Null } catch { break }
 }
}
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList @('-u', ('"'+$taskApp+'"')) -WorkingDirectory $taskWorkspace -WindowStyle Hidden -RedirectStandardError (Join-Path $PSScriptRoot 'service.stderr.log') -RedirectStandardOutput (Join-Path $PSScriptRoot 'service.stdout.log') -PassThru
for ($taskAttempt=0;$taskAttempt -lt 80;$taskAttempt++) {
 Start-Sleep -Milliseconds 250
 if ($taskProcess.HasExited) { throw 'v5 stopped. Check service.stderr.log.' }
 try { $taskStatus=Invoke-RestMethod "$taskUrl/telemetry" -TimeoutSec 1; if ($taskStatus.live_version -eq 'feedback/v5') { Write-Output 'v5 ready: http://localhost:8090/'; exit 0 } } catch { }
}
throw 'Service did not become ready.'
