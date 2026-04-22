@echo off
setlocal
cd /d "%~dp0"
title ChemBlast - Local Launcher

echo ============================================================
echo  ChemBlast
echo  Local Windows Launcher
echo ============================================================
echo.
echo This will open the app at http://localhost:8501
echo Keep this window open while using the app.
echo.

if "%QSAR_SMOKE_TEST%"=="1" (
  if "%QSAR_TEST_PORT%"=="" set "QSAR_TEST_PORT=8510"
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_local.ps1" -SmokeTest -Port %QSAR_TEST_PORT%
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_local.ps1"
)

if errorlevel 1 (
  echo.
  echo The app could not start. Please send the messages above to support.
  pause
  exit /b 1
)

