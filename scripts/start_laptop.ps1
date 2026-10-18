param([string]$ListenAddress = "127.0.0.1")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python environment missing: .venv" }
Write-Host "ForteVision: keep this terminal and laptop running during jury checks."
Write-Host "Local UI: http://127.0.0.1:8501/"
Write-Host "API: http://127.0.0.1:8000/api/v1/analyze"
& $pythonPath -B -m scripts.serve --host $ListenAddress
exit $LASTEXITCODE
