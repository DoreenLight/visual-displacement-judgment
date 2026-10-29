$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) { $ScriptDir = $PSScriptRoot }
$RootDir = (Resolve-Path "$ScriptDir\..\..\..\..").Path
Set-Location $RootDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting RealSense D435i Real-Time Visual Biofeedback..." -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

$PythonExe = "$RootDir\.venv\Scripts\python.exe"
$AppScript = "$ScriptDir\realtime_biofeedback_app.py"

& $PythonExe $AppScript
Write-Host "App exited." -ForegroundColor Green
Read-Host "Press Enter to close window..."
