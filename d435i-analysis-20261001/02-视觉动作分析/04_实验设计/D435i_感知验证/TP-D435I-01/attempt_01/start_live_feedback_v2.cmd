@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_live_feedback_v2.ps1"
if errorlevel 1 pause
