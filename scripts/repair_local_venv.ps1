# repair_local_venv.ps1 — Nuke and rebuild local .venv with Python 3.12
# Usage: powershell -ExecutionPolicy Bypass -File scripts/repair_local_venv.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Repair Local Venv ===" -ForegroundColor Cyan

# Deactivate if active
if ($env:VIRTUAL_ENV) {
    Write-Host "Deactivating current venv..."
    deactivate 2>$null
}

# Remove old venv
if (Test-Path ".venv") {
    Write-Host "Removing .venv..."
    Remove-Item -Recurse -Force .venv
}

# Create fresh venv with Python 3.12
Write-Host "Creating .venv with Python 3.12..."
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "py -3.12 -m venv failed. Install Python 3.12 from python.org." }

# Activate
& .\.venv\Scripts\Activate.ps1

# Upgrade pip + tools
Write-Host "Upgrading pip, setuptools, wheel..."
python -m pip install -U pip setuptools wheel -q

# Install editable + light deps
Write-Host "Installing marketify (editable) + requirements.txt..."
python -m pip install -e . -q
python -m pip install -r requirements.txt -q

# Verify
Write-Host ""
Write-Host "=== Verify ===" -ForegroundColor Cyan
python -m pytest tests/ -q --tb=short
if ($LASTEXITCODE -ne 0) { Write-Host "PYTEST FAILED" -ForegroundColor Red; exit 1 }

python scripts/colab_check.py
Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Local venv ready. Do NOT run heavy training here. Use Colab."
