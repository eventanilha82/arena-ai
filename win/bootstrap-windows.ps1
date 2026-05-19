[CmdletBinding()]
param(
    [string]$WorkRoot = "C:\arena-ai",
    [string]$PublicKey = ""
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[arena-win-bootstrap] $Message" -ForegroundColor Cyan
}

function Assert-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Rode este script em uma sessão elevada do PowerShell."
    }
}

function Add-PathForCurrentSession {
    param([string]$PathItem)
    if ((Test-Path $PathItem) -and (($env:Path -split ";") -notcontains $PathItem)) {
        $env:Path = "$env:Path;$PathItem"
    }
}

function Ensure-Chocolatey {
    if (Get-Command choco.exe -ErrorAction SilentlyContinue) {
        Write-Host "Chocolatey já instalado."
        return
    }

    Write-Step "Instalando Chocolatey"
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $install = (New-Object Net.WebClient).DownloadString("https://community.chocolatey.org/install.ps1")
    Invoke-Expression $install
    Add-PathForCurrentSession "C:\ProgramData\chocolatey\bin"
}

function Ensure-Uv {
    if (Get-Command uv.exe -ErrorAction SilentlyContinue) {
        Write-Host "uv já instalado."
        return
    }

    Write-Step "Instalando uv"
    try {
        choco install -y uv
    }
    catch {
        Write-Host "Instalação do uv via Chocolatey falhou; usando instalador da Astral."
        powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
        Add-PathForCurrentSession "$env:USERPROFILE\.local\bin"
    }
}

function Ensure-OpenSshServer {
    Write-Step "Habilitando OpenSSH Server"
    $capability = Get-WindowsCapability -Online | Where-Object Name -like "OpenSSH.Server*"
    if ($capability.State -ne "Installed") {
        Add-WindowsCapability -Online -Name $capability.Name | Out-Null
    }

    Set-Service -Name sshd -StartupType Automatic
    Start-Service sshd

    if (-not (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule `
            -Name "OpenSSH-Server-In-TCP" `
            -DisplayName "OpenSSH Server (sshd)" `
            -Enabled True `
            -Direction Inbound `
            -Protocol TCP `
            -Action Allow `
            -LocalPort 22 | Out-Null
    }

    $config = "$env:ProgramData\ssh\sshd_config"
    if (Test-Path $config) {
        $content = Get-Content -Raw -Path $config
        if ($content -match "(?m)^#?PasswordAuthentication\s+") {
            $content = $content -replace "(?m)^#?PasswordAuthentication\s+.*$", "PasswordAuthentication yes"
        }
        else {
            $content += "`r`nPasswordAuthentication yes`r`n"
        }
        if ($content -match "(?m)^#?PubkeyAuthentication\s+") {
            $content = $content -replace "(?m)^#?PubkeyAuthentication\s+.*$", "PubkeyAuthentication yes"
        }
        else {
            $content += "PubkeyAuthentication yes`r`n"
        }
        Set-Content -Path $config -Value $content -Encoding ascii
        Restart-Service sshd
    }
}

function Add-PublicKey {
    param([string]$Key)
    if ([string]::IsNullOrWhiteSpace($Key)) {
        return
    }

    Write-Step "Instalando chave pública SSH"
    $userSsh = Join-Path $env:USERPROFILE ".ssh"
    New-Item -ItemType Directory -Force -Path $userSsh | Out-Null
    $userAuthorizedKeys = Join-Path $userSsh "authorized_keys"
    if (-not (Test-Path $userAuthorizedKeys) -or -not ((Get-Content -Raw $userAuthorizedKeys) -match [regex]::Escape($Key))) {
        Add-Content -Path $userAuthorizedKeys -Value $Key
    }

    $adminAuthorizedKeys = "$env:ProgramData\ssh\administrators_authorized_keys"
    if (-not (Test-Path $adminAuthorizedKeys) -or -not ((Get-Content -Raw $adminAuthorizedKeys) -match [regex]::Escape($Key))) {
        Add-Content -Path $adminAuthorizedKeys -Value $Key
    }
    icacls $adminAuthorizedKeys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F" | Out-Null
}

Assert-Admin

Write-Step "Preparando diretórios"
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkRoot "incoming") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkRoot "out") | Out-Null

Ensure-Chocolatey
Write-Step "Instalando ferramentas de build"
choco install -y git make
Add-PathForCurrentSession "C:\ProgramData\chocolatey\bin"
Add-PathForCurrentSession "C:\Program Files\Git\cmd"
Add-PathForCurrentSession "C:\Program Files\Git\bin"
Add-PathForCurrentSession "C:\Program Files\Git\usr\bin"

Ensure-Uv
Add-PathForCurrentSession "$env:USERPROFILE\.local\bin"

Write-Step "Instalando Python 3.12 gerenciado pelo uv"
uv python install 3.12

if ([string]::IsNullOrWhiteSpace($PublicKey)) {
    $repoKey = "\\tsclient\repo\win\authorized_keys.pub"
    if (Test-Path $repoKey) {
        $PublicKey = (Get-Content -Raw -Path $repoKey).Trim()
    }
}

Ensure-OpenSshServer
Add-PublicKey -Key $PublicKey

Write-Step "Versões instaladas"
git --version
uv --version
make --version | Select-Object -First 1
uv python list --only-installed

Write-Host ""
Write-Host "Bootstrap concluído. Na máquina local, rode: bash win/remote-build.sh" -ForegroundColor Green
