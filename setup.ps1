$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.tools\bin\uv.exe')) {
    python -m pip install --target .tools uv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install uv' }
}
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.runtime'
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & .\.tools\bin\uv.exe venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment' }
}
$taskRequirements = if (Test-Path 'requirements.lock.txt') { 'requirements.lock.txt' } else { 'requirements.txt' }
& .\.tools\bin\uv.exe pip install --python .venv\Scripts\python.exe -r $taskRequirements
if ($LASTEXITCODE -ne 0) { throw 'Failed to install dependencies' }
