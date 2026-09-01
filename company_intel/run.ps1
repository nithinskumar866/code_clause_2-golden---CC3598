# Start company_intel.
#
#   .\run.ps1              start on port 8123
#   .\run.ps1 -Reload      restart automatically when you edit code
#   .\run.ps1 -Port 9000   somewhere else
#
# Checks the things that actually go wrong before starting, so a misconfiguration is a
# sentence on the console rather than a stack trace thirty seconds later.

param(
    [int]$Port = 8123,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No virtualenv found. Run this once:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Host "No .env found. Copy the example and fill in your Qdrant details:" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.example .env"
    exit 1
}

if (-not (Select-String -Path ".env" -Pattern "^QDRANT_URL=\S" -Quiet)) {
    Write-Host "QDRANT_URL is empty in .env — the service will start but every data route will return 503." -ForegroundColor Yellow
}

# Windows reserves ranges of ports for Hyper-V and WinNAT; binding one fails with
# WinError 10013 after startup has already done its work, which reads as a crash.
$excluded = (netsh interface ipv4 show excludedportrange protocol=tcp) -join "`n"
foreach ($line in $excluded -split "`n") {
    if ($line -match "^\s*(\d+)\s+(\d+)\s*$") {
        if ($Port -ge [int]$Matches[1] -and $Port -le [int]$Matches[2]) {
            Write-Host "Port $Port is inside a reserved Windows range ($($Matches[1])-$($Matches[2]))." -ForegroundColor Red
            Write-Host "Pick another: .\run.ps1 -Port 8321"
            exit 1
        }
    }
}

Write-Host ""
Write-Host "  company_intel  ->  http://localhost:$Port/docs" -ForegroundColor Green
Write-Host "  health         ->  http://localhost:$Port/api/v1/health"
Write-Host "  stop with Ctrl+C"
Write-Host ""

$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--port", "$Port")
if ($Reload) { $uvicornArgs += "--reload" }

& $python @uvicornArgs
