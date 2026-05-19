@echo off
setlocal
set REPO=\\tsclient\repo
set LOGDIR=%REPO%\win\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

if /i not "%~1"=="elevated" (
    net session >nul 2>&1
    if errorlevel 1 (
        powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c ""\\tsclient\repo\win\bootstrap-rdp.cmd"" elevated' -Verb RunAs"
        exit /b 0
    )
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO%\win\bootstrap-windows.ps1" > "%LOGDIR%\bootstrap-windows.log" 2>&1
exit /b %ERRORLEVEL%
