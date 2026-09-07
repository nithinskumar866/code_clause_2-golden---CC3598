# Start every service the platform needs, each in its own window.
#
#   .\run-all.ps1
#
# Three processes, three ports. They are separate on purpose - the crawler is a
# standalone service, and the frontend talks to both backends directly - so this
# launches rather than merges them.
#
#   5173  frontend            the UI
#   8000  hiring backend      resumes, JD analysis, Excel company records
#   8123  company_intel       crawled company websites (the "Web" chat mode)
#
# If the Web tab reports an unreachable embedding endpoint, the RunPod pod has been
# restarted and has a new id: run .\update-runpod.ps1 <new-id> first.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$services = @(
    @{
        Name = "company_intel (8123)"
        Dir  = "company_intel"
        Cmd  = ".\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8123"
        Url  = "http://localhost:8123/api/v1/health"
    },
    @{
        Name = "hiring backend (8000)"
        Dir  = "ai_hiring_platform\backend"
        Cmd  = ".\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000"
        Url  = "http://localhost:8000/api/v1/health"
    },
    @{
        Name = "frontend (5173)"
        Dir  = "ai_hiring_platform\frontend"
        Cmd  = "npm run dev -- --port 5173"
        Url  = "http://localhost:5173/"
    }
)

Write-Host ""
foreach ($svc in $services) {
    $path = Join-Path $PSScriptRoot $svc.Dir
    if (-not (Test-Path $path)) {
        Write-Host "  missing: $($svc.Dir)" -ForegroundColor Red
        continue
    }
    Write-Host "  starting $($svc.Name)" -ForegroundColor Cyan
    # A separate window per service, so one crashing is visible and restartable on its
    # own rather than taking the others down with it.
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "Set-Location '$path'; Write-Host '$($svc.Name)' -ForegroundColor Green; $($svc.Cmd)"
    ) | Out-Null
}

Write-Host ""
Write-Host "  waiting for them to answer..." -ForegroundColor DarkGray

foreach ($svc in $services) {
    $ok = $false
    # The crawler's health check probes the remote embedding endpoint, so it is the
    # slow one - allow generously rather than reporting a false failure.
    for ($i = 0; $i -lt 40; $i++) {
        try {
            Invoke-WebRequest -Uri $svc.Url -TimeoutSec 5 -UseBasicParsing | Out-Null
            $ok = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    if ($ok) {
        Write-Host "  up   $($svc.Name)" -ForegroundColor Green
    } else {
        Write-Host "  DOWN $($svc.Name) - check its window" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  Open http://localhost:5173  ->  Recruiter Assistant  ->  Web" -ForegroundColor Cyan
Write-Host ""
