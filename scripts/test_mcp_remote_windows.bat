@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%test_mcp_remote_windows.ps1"

if not exist "%PS_SCRIPT%" (
  echo Missing script: %PS_SCRIPT%
  exit /b 1
)

set "SERVER_URL=%~1"
if "%SERVER_URL%"=="" set "SERVER_URL=http://kbi-cs2.fbmi.cvut.cz:8000"

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -ServerUrl "%SERVER_URL%"
exit /b %errorlevel%
