@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM SkyNet — Windows Launcher (calls skynet.ps1)
REM Usage: skynet.bat start | skynet.bat setup | skynet.bat help
REM ═══════════════════════════════════════════════════════════════════════════
powershell -ExecutionPolicy Bypass -File "%~dp0skynet.ps1" %*
