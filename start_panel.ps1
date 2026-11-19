$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host 'Balloon Control: http://127.0.0.1:8787'
& .\.venv\Scripts\python.exe web_app.py
