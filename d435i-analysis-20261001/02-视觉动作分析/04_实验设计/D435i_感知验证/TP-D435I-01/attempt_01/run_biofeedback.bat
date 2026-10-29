@echo off
title D435i 实时闭环视觉生物反馈康复系统
chcp 936 >nul
cd /d "%~dp0..\..\..\.."
echo ========================================================
echo 正在启动 RealSense D435i 实时闭环视觉生物反馈系统...
echo ========================================================
".venv\Scripts\python.exe" "04_实验设计\D435i_感知验证\TP-D435I-01\attempt_01\realtime_biofeedback_app.py"
if %errorlevel% neq 0 (
    echo [提示] 程序已退出 (代码: %errorlevel%)
    pause
)
