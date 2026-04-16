<#
.SYNOPSIS
    SkyNet — Windows Startup Script (PowerShell)
.DESCRIPTION
    Cross-platform management script for SkyNet cognitive architecture.
    For Linux/macOS/WSL, use ./skynet.sh instead.
.EXAMPLE
    .\skynet.ps1 setup    # First-time setup wizard
    .\skynet.ps1 start    # Build and start everything
    .\skynet.ps1 update   # Pull + rebuild + restart (live update)
    .\skynet.ps1 help     # Show all commands
#>
param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    [Parameter(Position = 1, ValueFromRemainingArguments)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── Constants ─────────────────────────────────────────────────────────────────
$GatewayUrl   = "http://localhost:8000"
$AutonomyUrl  = "http://localhost:8018"
$MaxRetries   = 3

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Info  { param([string]$m) Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok    { param([string]$m) Write-Host "[OK]    $m" -ForegroundColor Green }
function Write-Warn  { param([string]$m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err   { param([string]$m) Write-Host "[ERROR] $m" -ForegroundColor Red }
function Write-Die   { param([string]$m) Write-Err $m; exit 1 }

function Test-CommandExists { param([string]$Name) $null -ne (Get-Command $Name -ErrorAction SilentlyContinue) }

function Test-VersionGe {
    param([string]$Actual, [string]$Minimum)
    $a = $Actual -split '\.' | ForEach-Object { [int]$_ }
    $b = $Minimum -split '\.' | ForEach-Object { [int]$_ }
    for ($i = 0; $i -lt [Math]::Max($a.Count, $b.Count); $i++) {
        $av = if ($i -lt $a.Count) { $a[$i] } else { 0 }
        $bv = if ($i -lt $b.Count) { $b[$i] } else { 0 }
        if ($av -gt $bv) { return $true }
        if ($av -lt $bv) { return $false }
    }
    return $true
}

function Show-Banner {
    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════════════╗" -ForegroundColor Blue
    Write-Host "  ║           S K Y N E T  v1.0                   ║" -ForegroundColor Blue
    Write-Host "  ║       Cognitive Architecture System           ║" -ForegroundColor Blue
    Write-Host "  ╚═══════════════════════════════════════════════╝" -ForegroundColor Blue
    Write-Host ""
}

function Invoke-ComposeBuild {
    param([string[]]$BuildArgs)
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        try {
            docker compose build @BuildArgs
            return
        } catch {
            if ($attempt -lt $MaxRetries) {
                $wait = $attempt * 5
                Write-Warn "Build failed (attempt $attempt/$MaxRetries) — retrying in ${wait}s..."
                Start-Sleep -Seconds $wait
            }
        }
    }
    Write-Die "Build failed after $MaxRetries attempts"
}

function Test-UrlReachable {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
        return $response.StatusCode -lt 400
    } catch {
        return $false
    }
}

# ── Dependency Check ──────────────────────────────────────────────────────────
function Test-Dependencies {
    Write-Info "Checking dependencies..."
    Write-Info "  Platform: Windows"
    $fail = $false

    if (-not (Test-CommandExists "docker"))  { Write-Err "Docker is not installed"; $fail = $true }
    if (-not (Test-CommandExists "git"))      { Write-Err "Git is not installed"; $fail = $true }

    if (-not $fail) {
        $dockerVer = (docker --version 2>&1) -replace '.*?(\d+\.\d+).*', '$1'
        if ($dockerVer -and (Test-VersionGe $dockerVer "24.0")) {
            Write-Ok "Docker version $dockerVer (>= 24.0)"
        } else {
            Write-Err "Docker version $dockerVer is below minimum 24.0"; $fail = $true
        }

        $composeVer = (docker compose version 2>&1) -replace '.*?(\d+\.\d+).*', '$1'
        if ($composeVer -and (Test-VersionGe $composeVer "2.20")) {
            Write-Ok "Docker Compose version $composeVer (>= 2.20)"
        } else {
            Write-Err "Docker Compose version $composeVer is below minimum 2.20"; $fail = $true
        }
    }

    # Check Ollama
    if (Test-CommandExists "ollama") {
        if (Test-UrlReachable "http://localhost:11434/api/tags") {
            Write-Ok "Ollama is running"
        } else {
            Write-Warn "Ollama is installed but not running — start it from the system tray or run: ollama serve"
        }
    } else {
        Write-Warn "Ollama not found — download from https://ollama.com/download"
    }

    # Check .env
    if (Test-Path ".env") { Write-Ok ".env file exists" }
    else { Write-Warn ".env file missing — run '.\skynet.ps1 setup' first" }

    if ($fail) { Write-Die "Missing dependencies — install them and retry" }
    Write-Ok "All dependencies satisfied"
}

# ── Setup ─────────────────────────────────────────────────────────────────────
function Invoke-Setup {
    Show-Banner
    Write-Info "First-time setup wizard (platform: Windows)"
    Write-Host ""

    # Docker Desktop check
    if (-not (Test-CommandExists "docker")) {
        Write-Err "Docker Desktop is not installed."
        Write-Info "Download from: https://www.docker.com/products/docker-desktop"
        Write-Info "After installing, launch Docker Desktop and wait for it to start."
        Read-Host "Press Enter once Docker is installed and running"
        if (-not (Test-CommandExists "docker")) { Write-Die "Docker still not found" }
    }
    Write-Ok "Docker detected"

    # Ollama check
    if (-not (Test-CommandExists "ollama")) {
        Write-Info "Ollama is not installed."
        Write-Info "Download from: https://ollama.com/download"
        Read-Host "Press Enter once Ollama is installed"
        if (-not (Test-CommandExists "ollama")) { Write-Warn "Ollama not found — continuing without it" }
    }
    if (Test-CommandExists "ollama") { Write-Ok "Ollama detected" }

    # ── GPU / CPU detection ──────────────────────────────────────────────────
    $gpuMode = "cpu"
    $gpuName = ""
    $gpuVram = ""
    $gpuVendor = ""

    # Method 1: nvidia-smi (NVIDIA GPU)
    if (Test-CommandExists "nvidia-smi") {
        try {
            $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
            $gpuVram = (nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
            if ($gpuName) { $gpuVendor = "nvidia" }
        } catch {}
        # Fallback: parse nvidia-smi table output
        if (-not $gpuName) {
            try {
                $smiOut = nvidia-smi 2>$null | Out-String
                if ($smiOut -match 'NVIDIA\s+([A-Za-z0-9 ]+)') { $gpuName = "NVIDIA $($Matches[1])".Trim(); $gpuVendor = "nvidia" }
            } catch {}
        }
    }

    # Method 2: WMI query (works even without nvidia-smi, detects ALL GPUs)
    if (-not $gpuVendor) {
        try {
            $wmiGpus = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notmatch 'Microsoft|Basic|Remote' }
            foreach ($wmiGpu in $wmiGpus) {
                if ($wmiGpu.Name -match 'NVIDIA') {
                    $gpuName = $wmiGpu.Name
                    $gpuVram = "{0:N0} MiB" -f ($wmiGpu.AdapterRAM / 1MB)
                    $gpuVendor = "nvidia"
                    break
                }
                if ($wmiGpu.Name -match 'AMD|Radeon') {
                    $gpuName = $wmiGpu.Name
                    $gpuVram = "{0:N0} MiB" -f ($wmiGpu.AdapterRAM / 1MB)
                    $gpuVendor = "amd"
                    break
                }
            }
        } catch {}
    }

    if ($gpuVendor -eq "nvidia") {
        Write-Host ""
        Write-Host "  +======================================================+" -ForegroundColor Green
        Write-Host "  |  NVIDIA dGPU DETECTED                                 |" -ForegroundColor Green
        Write-Host "  |  $gpuName" -ForegroundColor Green
        if ($gpuVram) { Write-Host "  |  VRAM: $gpuVram" -ForegroundColor Green }
        Write-Host "  +======================================================+" -ForegroundColor Green
        Write-Host ""
        Write-Host "  GPU mode:  Ollama uses your NVIDIA GPU for inference + embeddings"
        Write-Host "             Learner uses CUDA for local model training/inference"
        Write-Host ""
        Write-Host "  CPU mode:  Everything runs on CPU (slower but no GPU drivers needed)"
        Write-Host ""
        $choice = Read-Host "Use GPU acceleration? [Y/n]"
        if ($choice -match '^[Nn]') {
            $gpuMode = "cpu"
            Write-Info "Selected: CPU mode"
        } else {
            $gpuMode = "gpu"
            Write-Ok "Selected: GPU mode ($gpuName)"
        }
    } elseif ($gpuVendor -eq "amd") {
        Write-Host ""
        Write-Host "  +======================================================+" -ForegroundColor Green
        Write-Host "  |  AMD GPU DETECTED                                     |" -ForegroundColor Green
        Write-Host "  |  $gpuName" -ForegroundColor Green
        Write-Host "  +======================================================+" -ForegroundColor Green
        Write-Host ""
        Write-Host "  GPU mode:  Ollama can use AMD GPU via ROCm"
        Write-Host "             Learner uses ROCm PyTorch"
        Write-Host ""
        Write-Host "  CPU mode:  Everything runs on CPU"
        Write-Host ""
        $choice = Read-Host "Use GPU acceleration? [Y/n]"
        if ($choice -match '^[Nn]') {
            $gpuMode = "cpu"
            Write-Info "Selected: CPU mode"
        } else {
            $gpuMode = "gpu"
            Write-Ok "Selected: GPU mode - AMD ROCm ($gpuName)"
        }
    } else {
        Write-Info "No dedicated GPU detected — using CPU mode"
    }

    # ── Configure Ollama (Windows environment variables) ─────────────────────
    if (Test-CommandExists "ollama") {
        Write-Info "Configuring Ollama (keep_alive=-1, num_parallel=4, mode=$gpuMode)..."

        # Set persistent environment variables (User scope)
        [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
        [Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "-1", "User")
        [Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "4", "User")
        if ($gpuMode -eq "cpu") {
            [Environment]::SetEnvironmentVariable("OLLAMA_NUM_GPU", "0", "User")
        } else {
            [Environment]::SetEnvironmentVariable("OLLAMA_NUM_GPU", $null, "User")
        }

        # Also set for current session
        $env:OLLAMA_HOST = "0.0.0.0"
        $env:OLLAMA_KEEP_ALIVE = "-1"
        $env:OLLAMA_NUM_PARALLEL = "4"
        if ($gpuMode -eq "cpu") { $env:OLLAMA_NUM_GPU = "0" }

        Write-Ok "Ollama configured via Windows environment variables"
        Write-Warn "Restart Ollama (system tray > Quit > relaunch) for changes to take effect"
    }

    # ── PyTorch URL ──────────────────────────────────────────────────────────
    if ($gpuMode -eq "gpu") {
        if ($gpuVendor -eq "amd") {
            $torchUrl = "https://download.pytorch.org/whl/rocm6.0"
        } else {
            $torchUrl = "https://download.pytorch.org/whl/cu121"
        }
    } else {
        $torchUrl = "https://download.pytorch.org/whl/cpu"
    }

    # ── Create .env ──────────────────────────────────────────────────────────
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Ok "Created .env from .env.example"
        } else {
            @"
# LLM Provider — choose one: anthropic, openai, gemini, ollama, deepseek, groq, xai, openrouter, generic
LLM_PROVIDER=anthropic

# API Keys (set the one matching your LLM_PROVIDER)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
GROQ_API_KEY=
XAI_API_KEY=
OPENROUTER_API_KEY=
GENERIC_API_KEY=
GENERIC_BASE_URL=

# Token limits
MAX_TOKENS_MAIN=4096
MAX_TOKENS_FAST=1024

# Autonomy
AUTONOMY_MAX_RISK=0.7

# Security
SUPERGROKIE_WEBHOOK_SECRET=change-this-to-a-random-secret-min-32-chars

# Ollama
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_EMBED_MODEL=nomic-embed-text

# Local Model
LOCAL_MODEL_NAME=microsoft/phi-4-mini-instruct
HF_TOKEN=

# Vision (disabled by default)
VISION_ENABLED=false
CAMERA_INDEX=0
"@ | Set-Content ".env" -Encoding UTF8
            Write-Ok "Created minimal .env — edit it with your API keys"
        }
    } else {
        Write-Ok ".env already exists"
    }

    # Inject TORCH_INDEX_URL
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match 'TORCH_INDEX_URL=') {
        $envContent = $envContent -replace 'TORCH_INDEX_URL=.*', "TORCH_INDEX_URL=$torchUrl"
        Set-Content ".env" $envContent -NoNewline -Encoding UTF8
    } else {
        Add-Content ".env" "`n# GPU/CPU — set by skynet.ps1 setup (gpu_mode=$gpuMode)`nTORCH_INDEX_URL=$torchUrl"
    }
    Write-Ok "Learner PyTorch: $gpuMode ($torchUrl)"

    # Pull Ollama models
    if (Test-CommandExists "ollama") {
        Write-Info "Pulling Ollama embedding model..."
        try { ollama pull nomic-embed-text 2>$null; Write-Ok "nomic-embed-text ready" }
        catch { Write-Warn "Could not pull nomic-embed-text" }

        Write-Info "Pulling Ollama local inference model..."
        try { ollama pull phi4 2>$null; Write-Ok "phi4 ready" }
        catch { Write-Warn "Could not pull phi4 (optional)" }
    }

    Write-Host ""
    Write-Ok "Setup complete! Edit .env with your API keys, then run: .\skynet.ps1 start"
    Write-Host ""
    Write-Info "  Platform: Windows"
    Write-Info "  GPU mode: $gpuMode"
    if ($gpuName) { Write-Info "  GPU:      $gpuName ($gpuVram)" }
}

# ── Start ─────────────────────────────────────────────────────────────────────
function Invoke-Start {
    Show-Banner
    Test-Dependencies

    Write-Info "Building containers..."
    Invoke-ComposeBuild

    # Phase 1: Infrastructure
    Write-Info "Starting infrastructure (Redis, Elasticsearch, EdgeDB)..."
    docker compose up -d redis elasticsearch edgedb

    Write-Info "Waiting for databases to be healthy..."
    for ($i = 0; $i -lt 30; $i++) {
        $ps = docker compose ps 2>&1 | Out-String
        if ($ps -match "healthy") { break }
        Start-Sleep -Seconds 2
    }
    Write-Ok "Databases ready"

    Write-Info "Initializing databases..."
    try { docker compose up db-init 2>$null } catch { Write-Warn "db-init may have already run" }

    # Phase 2: Warm up Ollama
    Write-Info "Warming up Ollama (pre-loading embed model)..."
    $ollamaUrl = "http://localhost:11434"
    $embedModel = "nomic-embed-text"
    if (Test-Path ".env") {
        $envLines = Get-Content ".env"
        foreach ($line in $envLines) {
            if ($line -match '^OLLAMA_EMBED_MODEL=(.+)') { $embedModel = $Matches[1] }
        }
    }
    try {
        $body = @{ model = $embedModel; prompt = "warmup"; keep_alive = -1 } | ConvertTo-Json
        Invoke-RestMethod -Uri "$ollamaUrl/api/embeddings" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30 -ErrorAction SilentlyContinue | Out-Null
        Write-Ok "Ollama embed model loaded ($embedModel, keep_alive=-1)"
    } catch {
        Write-Warn "Could not warm up Ollama — embeddings may be slow on first call"
    }

    # Phase 3: Start all services
    Write-Info "Starting all services..."
    docker compose up -d

    Write-Info "Waiting for system health..."
    for ($i = 0; $i -lt 60; $i++) {
        if (Test-UrlReachable "$GatewayUrl/health") { break }
        Start-Sleep -Seconds 3
    }

    if (Test-UrlReachable "$GatewayUrl/health") {
        Write-Ok "All services are running!"
    } else {
        Write-Warn "Some services may still be starting — check: .\skynet.ps1 status"
    }

    Write-Host ""
    Write-Ok "SkyNet is online — running red-hot"
    Write-Host ""
    Write-Info "  Dashboard:   $GatewayUrl              (main UI)"
    Write-Info "  Health:      $GatewayUrl/health        (system health)"
    Write-Info "  Embed Stats: $GatewayUrl/brain/embed-queue/stats"
    Write-Host ""
    Write-Info "Management:"
    Write-Info "  .\skynet.ps1 stop      Graceful shutdown"
    Write-Info "  .\skynet.ps1 restart   Quick restart (no rebuild, keeps data)"
    Write-Info "  .\skynet.ps1 update    Git pull + rebuild + restart"
    Write-Info "  .\skynet.ps1 status    Service health overview"
    Write-Info "  .\skynet.ps1 logs      Tail all logs"
}

# ── Stop ──────────────────────────────────────────────────────────────────────
function Invoke-Stop {
    Show-Banner
    Write-Info "Gracefully shutting down all services..."
    docker compose down
    Write-Ok "All services stopped"
}

# ── Restart ───────────────────────────────────────────────────────────────────
function Invoke-Restart {
    Show-Banner
    Write-Info "Restarting services (preserving all data)..."
    docker compose stop
    docker compose up -d
    Write-Info "Waiting for system health..."
    for ($i = 0; $i -lt 30; $i++) {
        if (Test-UrlReachable "$GatewayUrl/health") { break }
        Start-Sleep -Seconds 3
    }
    if (Test-UrlReachable "$GatewayUrl/health") {
        Write-Ok "All services restarted — data preserved"
    } else {
        Write-Warn "Some services may still be starting — check: .\skynet.ps1 status"
    }
}

# ── Status ────────────────────────────────────────────────────────────────────
function Invoke-Status {
    Show-Banner
    Write-Info "Service Status:"
    Write-Host ""
    docker compose ps
    Write-Host ""
    Write-Info "Brain Health:"
    try {
        $health = Invoke-RestMethod -Uri "$GatewayUrl/health" -TimeoutSec 5
        $health | ConvertTo-Json -Depth 5
    } catch {
        Write-Warn "Gateway not responding at $GatewayUrl"
    }
}

# ── Logs ──────────────────────────────────────────────────────────────────────
function Invoke-Logs {
    if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
        $svc = $ExtraArgs[0]
        Write-Info "Tailing logs for $svc..."
        docker compose logs -f $svc
    } else {
        Write-Info "Tailing all logs..."
        docker compose logs -f
    }
}

# ── Dashboard ─────────────────────────────────────────────────────────────────
function Invoke-Dashboard {
    Show-Banner
    Write-Info "Dashboard URL: $GatewayUrl"
    Start-Process $GatewayUrl
    Write-Ok "Opening dashboard in browser..."
}

# ── Think ─────────────────────────────────────────────────────────────────────
function Invoke-Think {
    Show-Banner
    Write-Info "Activating autonomy mode..."
    try {
        $body = '{"mode": "autonomous"}'
        $response = Invoke-RestMethod -Uri "$AutonomyUrl/activate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 10
        Write-Ok "Autonomy mode activated"
        $response | ConvertTo-Json -Depth 3
    } catch {
        Write-Warn "Could not reach autonomy service at $AutonomyUrl"
        Write-Info "Make sure services are running: .\skynet.ps1 status"
    }
}

# ── Update ────────────────────────────────────────────────────────────────────
function Invoke-Update {
    Show-Banner
    Write-Info "Pulling latest code..."
    try { git pull --rebase } catch { git pull }

    Write-Info "Rebuilding changed services..."
    Invoke-ComposeBuild

    Write-Info "Restarting updated services..."
    docker compose up -d

    Write-Ok "Update complete"
    Write-Info "Check health: .\skynet.ps1 status"
}

# ── Rebuild ───────────────────────────────────────────────────────────────────
function Invoke-Rebuild {
    Show-Banner
    Write-Warn "REBUILD — This will wipe ALL databases and rebuild everything fresh"
    Write-Host ""
    $confirm = Read-Host "Type 'REBUILD' to confirm"
    if ($confirm -ne "REBUILD") {
        Write-Info "Rebuild cancelled"
        return
    }

    Write-Info "Stopping all services and removing volumes..."
    docker compose down -v --remove-orphans

    Write-Info "Rebuilding all containers..."
    Invoke-ComposeBuild @("--no-cache")

    Write-Info "Starting fresh system..."
    Invoke-Start

    Write-Ok "Rebuild complete — fresh environment, all databases recreated"
}

# ── Reset ─────────────────────────────────────────────────────────────────────
function Invoke-Reset {
    Show-Banner
    Write-Warn "FACTORY RESET — This will destroy ALL data (memories, models, goals, etc.)"
    Write-Host ""
    $confirm = Read-Host "Type 'RESET' to confirm"
    if ($confirm -ne "RESET") {
        Write-Info "Reset cancelled"
        return
    }

    Write-Info "Stopping all services..."
    docker compose down -v --remove-orphans

    Write-Info "Rebuilding from scratch..."
    Invoke-ComposeBuild @("--no-cache")

    Write-Info "Starting fresh system..."
    Invoke-Start

    Write-Ok "Factory reset complete — SkyNet has no memory of the past"
}

# ── Help ──────────────────────────────────────────────────────────────────────
function Show-Help {
    Show-Banner
    Write-Host @"
Usage: .\skynet.ps1 <command> [options]

Platform: Windows (PowerShell)
Linux/macOS/WSL: Use ./skynet.sh instead

Commands:
  setup       First-time wizard: Docker Desktop, Ollama, GPU/CPU config, .env
  start       Full startup: build, init DBs, warm Ollama, start all 24 services
  stop        Graceful shutdown (data preserved)
  restart     Quick restart — no rebuild, keeps all data and conversations
  update      Git pull + rebuild changed services + restart (use after code changes)
  status      Show all service health and brain diagnostics
  logs [svc]  Tail logs for a service (or all). Try: .\skynet.ps1 logs embed-queue
  dashboard   Open the dashboard in your browser
  think       Activate autonomy mode (self-reflective thinking)
  rebuild     Full wipe: destroy databases + rebuild all containers from scratch
  reset       Factory reset (alias for rebuild)
  help        Show this help message

Scenarios:
  First install:            .\skynet.ps1 setup; .\skynet.ps1 start
  After code changes:       .\skynet.ps1 update
  Quick restart (no build): .\skynet.ps1 restart
  Config change (.env):     .\skynet.ps1 restart
  Wipe and reinstall:       .\skynet.ps1 rebuild
  After system reboot:      .\skynet.ps1 start (Ollama auto-starts from system tray)

Infrastructure:
  Ollama runs on the HOST (not in Docker) with keep_alive=-1 (never unloads models).
  Redis, Elasticsearch, EdgeDB run in Docker with restart: unless-stopped.
  All infrastructure is pre-heated on start — zero cold starts.
"@
}

# ── Main ──────────────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    "setup"     { Invoke-Setup }
    "start"     { Invoke-Start }
    "stop"      { Invoke-Stop }
    "restart"   { Invoke-Restart }
    "status"    { Invoke-Status }
    "logs"      { Invoke-Logs }
    "dashboard" { Invoke-Dashboard }
    "think"     { Invoke-Think }
    "update"    { Invoke-Update }
    "rebuild"   { Invoke-Rebuild }
    "reset"     { Invoke-Reset }
    "help"      { Show-Help }
    "--help"    { Show-Help }
    "-h"        { Show-Help }
    default     { Write-Err "Unknown command: $Command"; Show-Help; exit 1 }
}
