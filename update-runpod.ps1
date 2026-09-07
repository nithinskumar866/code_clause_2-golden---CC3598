# Point every service at a new RunPod pod.
#
#   .\update-runpod.ps1 f9ogfpvbwc4egt
#
# A RunPod pod gets a NEW id every time it is started, and that id appears in four
# places across two projects. Editing them by hand is four chances to miss one and then
# debug a service that is silently pointed at yesterday's pod.
#
# The script verifies the pod is actually answering, and that it still has the
# embedding model, BEFORE writing anything - a pod without nomic-embed-text cannot
# serve the 768-dimensional index that has already been built, and finding that out
# after the edit means editing back.

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$PodId
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Accept a bare id or a full URL, so pasting either from the RunPod console works.
if ($PodId -match '([a-z0-9]+)-11434') { $PodId = $Matches[1] }
$url = "https://$PodId-11434.proxy.runpod.net"

Write-Host ""
Write-Host "Checking $url ..." -ForegroundColor Cyan

try {
    $tags = Invoke-RestMethod -Uri "$url/api/tags" -TimeoutSec 30
} catch {
    Write-Host "  Not reachable. Is the pod started, and is the id correct?" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)"
    exit 1
}

$models = $tags.models | ForEach-Object { $_.name }
$embedders = $tags.models | Where-Object { $_.capabilities -contains "embedding" } | ForEach-Object { $_.name }

Write-Host "  models: $($models -join ', ')"

if (-not ($embedders -match "nomic-embed-text")) {
    # Refuse rather than warn: the content collection is 768-dimensional nomic vectors.
    # Any other model produces vectors that cannot be compared against them, so a
    # config pointing at this pod would fail on the first question instead of at setup.
    Write-Host ""
    Write-Host "  nomic-embed-text is NOT on this pod." -ForegroundColor Red
    Write-Host "  The Web mode index is 768d nomic; nothing else can search it."
    Write-Host "  Pull it on the pod first:  ollama pull nomic-embed-text"
    exit 1
}
Write-Host "  nomic-embed-text present." -ForegroundColor Green

$files = @(
    "ai_hiring_platform\backend\.env",
    "company_intel\.env"
)

$pattern = 'https://[a-z0-9]+-11434\.proxy\.runpod\.net'
$changed = 0

Write-Host ""
foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        Write-Host "  skipped (absent): $file" -ForegroundColor Yellow
        continue
    }
    $content = Get-Content $file -Raw
    if ($content -notmatch $pattern) {
        Write-Host "  no pod URL found in: $file" -ForegroundColor Yellow
        continue
    }
    $updated = $content -replace $pattern, $url
    if ($updated -ne $content) {
        Set-Content -Path $file -Value $updated -NoNewline -Encoding utf8
        $changed++
    }
    Write-Host "  updated: $file" -ForegroundColor Green
    Select-String -Path $file -Pattern 'RUNPOD_BASE_URL|EMBEDDING_GPU_URL|EMBED_OLLAMA_URL|ANSWER_OLLAMA_URL' |
        ForEach-Object { Write-Host "      $($_.Line)" -ForegroundColor DarkGray }
}

Write-Host ""
if ($changed -gt 0) {
    Write-Host "Done. Restart the services so they re-read the config:" -ForegroundColor Cyan
    Write-Host "  .\run-all.ps1"
} else {
    Write-Host "Already pointed at $url - nothing to change." -ForegroundColor Cyan
}
Write-Host ""
