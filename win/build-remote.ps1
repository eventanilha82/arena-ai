[CmdletBinding()]
param(
    [string]$WorkRoot = "C:\arena-ai",
    [ValidateSet("none", "smoke", "validate", "aaa")]
    [string]$Qa = "smoke"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[arena-win-build] $Message" -ForegroundColor Cyan
}

function Convert-ToGitBashPath {
    param([string]$Path)
    $full = (Resolve-Path $Path).Path -replace "\\", "/"
    if ($full -match "^([A-Za-z]):/(.*)$") {
        return "/" + $matches[1].ToLowerInvariant() + "/" + $matches[2]
    }
    return $full
}

function Resolve-Bash {
    $bash = Get-Command bash.exe -ErrorAction SilentlyContinue
    if ($bash) {
        return $bash.Source
    }
    $candidates = @(
        "C:\Program Files\Git\bin\bash.exe",
        "C:\Program Files\Git\usr\bin\bash.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "bash.exe não encontrado. Rode win/bootstrap-windows.ps1 primeiro."
}

$IncomingDir = Join-Path $WorkRoot "incoming"
$SourceArchive = Join-Path $IncomingDir "arena-ai-source.tar.gz"
$SourceProvenance = Join-Path $IncomingDir "release-source-provenance.json"
$SourceDir = Join-Path $WorkRoot "source"
$OutDir = Join-Path $WorkRoot "out"

if (-not (Test-Path $SourceArchive)) {
    throw "Bundle de código não encontrado: $SourceArchive"
}
if (-not (Test-Path $SourceProvenance)) {
    throw "Proveniência do source não encontrada: $SourceProvenance"
}

Write-Step "Preparando diretório de código"
if (Test-Path $SourceDir) {
    Remove-Item -Recurse -Force $SourceDir
}
New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Step "Extraindo código enviado"
tar -xzf $SourceArchive -C $SourceDir
Copy-Item -Path $SourceProvenance -Destination (Join-Path $SourceDir "release-source-provenance.json") -Force

$env:Path = "$env:Path;C:\ProgramData\chocolatey\bin;$env:USERPROFILE\.local\bin;C:\Program Files\Git\cmd;C:\Program Files\Git\bin;C:\Program Files\Git\usr\bin"
$bash = Resolve-Bash
$sourceForBash = Convert-ToGitBashPath $SourceDir

$commands = @(
    "set -euo pipefail",
    "cd '$sourceForBash'",
    "export SDL_VIDEODRIVER=dummy",
    "export SDL_AUDIODRIVER=dummy",
    "export PYTHON=.venv/Scripts/python.exe",
    "uv sync --dev"
)

if ($Qa -eq "smoke") {
    $commands += "make smoke PYTHON=.venv/Scripts/python.exe"
}
elseif ($Qa -eq "validate") {
    $commands += "make validate PYTHON=.venv/Scripts/python.exe"
}
elseif ($Qa -eq "aaa") {
    $commands += "make aaa-runtime-qa PYTHON=.venv/Scripts/python.exe"
}

$commands += "make build-current PYTHON=.venv/Scripts/python.exe RELEASE_SOURCE_PROVENANCE=release-source-provenance.json"
$commandLine = $commands -join " && "

Write-Step "Rodando build com QA=$Qa"
& $bash -lc $commandLine

if ($LASTEXITCODE -ne 0) {
    throw "Comando de build falhou com exit code $LASTEXITCODE"
}

Write-Step "Compactando artefato"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactName = "ArenaAI-windows-$stamp.zip"
$artifactPath = Join-Path $OutDir $artifactName
$latestPath = Join-Path $OutDir "ArenaAI-windows-latest.zip"
$distPath = Join-Path $SourceDir "dist\ArenaAI"

if (-not (Test-Path $distPath)) {
    throw "Saída esperada do build não encontrada: $distPath"
}

if (Test-Path $artifactPath) {
    Remove-Item -Force $artifactPath
}
if (Test-Path $latestPath) {
    Remove-Item -Force $latestPath
}

Compress-Archive -Path (Join-Path $distPath "*") -DestinationPath $artifactPath -Force
Copy-Item -Path $artifactPath -Destination $latestPath -Force

$python = Join-Path $SourceDir ".venv\Scripts\python.exe"
$provenanceScript = Join-Path $SourceDir "scripts\release_provenance.py"
$buildResult = Join-Path $OutDir "build-result.json"
& $python $provenanceScript write-build `
    --platform windows `
    --artifact $latestPath `
    --output $buildResult `
    --source-provenance $SourceProvenance
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao gerar proveniência do build Windows: exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "ARTIFACT=$latestPath" -ForegroundColor Green
Write-Host "PROVENANCE=$buildResult" -ForegroundColor Green
