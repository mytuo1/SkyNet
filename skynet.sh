#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# SkyNet — Cross-Platform Startup Script (Linux / macOS / WSL)
# For Windows (native), use skynet.ps1 or skynet.bat
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
die()     { error "$*"; exit 1; }

# Portable lowercase (bash 3.2 compat — macOS ships bash 3.2)
to_lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

# Portable sed -i (macOS requires '' argument)
sed_inplace() {
    if [ "$OS" = "macos" ]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── OS Detection ─────────────────────────────────────────────────────────────
OS="unknown"
case "$(uname -s)" in
    Linux*)
        if grep -qi microsoft /proc/version 2>/dev/null; then
            OS="wsl"
        else
            OS="linux"
        fi
        ;;
    Darwin*)  OS="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows-bash" ;;  # Git Bash on Windows
    *)        OS="unknown" ;;
esac

COMPOSE="docker compose"
BUILD_MAX_RETRIES=3
GATEWAY_URL="http://localhost:8000"
AUTONOMY_URL="http://localhost:8018"

# ── Build with Retry ──────────────────────────────────────────────────────────
# Docker image pulls can fail with transient network errors (EOF, timeout).
# This wrapper retries the build up to BUILD_MAX_RETRIES times.
compose_build() {
    local attempt=1
    while [ $attempt -le "$BUILD_MAX_RETRIES" ]; do
        if $COMPOSE build "$@"; then
            return 0
        fi
        if [ $attempt -lt "$BUILD_MAX_RETRIES" ]; then
            local wait_secs=$((attempt * 5))
            warn "Build failed (attempt $attempt/$BUILD_MAX_RETRIES) — retrying in ${wait_secs}s..."
            sleep "$wait_secs"
        fi
        attempt=$((attempt + 1))
    done
    die "Build failed after $BUILD_MAX_RETRIES attempts"
}

# ── Banner ────────────────────────────────────────────────────────────────────
banner() {
    echo -e "${BLUE}"
    cat <<'EOF'
╔═══════════════════════════════════════════════╗
║           S K Y N E T  v1.0                   ║
║       Cognitive Architecture System           ║
╚═══════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# ── Dependency Checks ─────────────────────────────────────────────────────────
check_command() {
    command -v "$1" &>/dev/null || { error "$1 is not installed"; return 1; }
}

# ── Version Comparison (portable — works with macOS sort) ─────────────────────
# Returns 0 if $1 >= $2 (version strings like "24.0", "2.20")
version_ge() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n | head -n1)" = "$2" ]
}

check_version() {
    local cmd="$1" min="$2" actual
    actual=$($cmd 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1) || true
    if [ -z "$actual" ]; then
        warn "Could not determine $cmd version"
        return 0
    fi
    if version_ge "$actual" "$min"; then
        success "$cmd version $actual (>= $min)"
    else
        error "$cmd version $actual is below minimum $min"
        return 1
    fi
}

check_deps() {
    info "Checking dependencies..."
    info "  Platform: $OS"
    local fail=0

    check_command docker   || fail=1
    check_command curl     || fail=1
    check_command git      || fail=1

    if [ $fail -eq 0 ]; then
        check_version "docker --version" "24.0" || fail=1
        # Docker Compose version
        local compose_ver
        compose_ver=$(docker compose version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1) || true
        if [ -n "$compose_ver" ]; then
            if version_ge "$compose_ver" "2.20"; then
                success "Docker Compose version $compose_ver (>= 2.20)"
            else
                error "Docker Compose version $compose_ver is below minimum 2.20"
                fail=1
            fi
        else
            warn "Could not determine Docker Compose version"
        fi
    fi

    # Check Ollama
    if check_command ollama; then
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            success "Ollama is running"
            # On Linux/WSL, warn if Ollama is bound to 127.0.0.1 only
            # On macOS, host.docker.internal works via Docker Desktop VM — no issue
            if [ "$OS" = "linux" ] || [ "$OS" = "wsl" ]; then
                if command -v ss &>/dev/null; then
                    local ss_out
                    ss_out=$(ss -tlnp 2>/dev/null | grep ':11434' || true)
                    if echo "$ss_out" | grep -q '127.0.0.1' && \
                       ! echo "$ss_out" | grep -qE '0\.0\.0\.0|\*|\[::\]'; then
                        warn "Ollama is bound to 127.0.0.1 — Docker containers cannot reach it"
                        warn "Fix: run './skynet.sh setup' or set OLLAMA_HOST=0.0.0.0 and restart Ollama"
                    fi
                elif command -v lsof &>/dev/null; then
                    local lsof_out
                    lsof_out=$(lsof -iTCP:11434 -sTCP:LISTEN -P -n 2>/dev/null || true)
                    if echo "$lsof_out" | grep -q '127.0.0.1' && \
                       ! echo "$lsof_out" | grep -qE '\*:11434|0\.0\.0\.0'; then
                        warn "Ollama is bound to 127.0.0.1 — Docker containers cannot reach it"
                        warn "Fix: run './skynet.sh setup' or set OLLAMA_HOST=0.0.0.0 and restart Ollama"
                    fi
                fi
            fi
        else
            warn "Ollama is installed but not running — start with: ollama serve"
        fi
    else
        warn "Ollama not found — embeddings will not work"
    fi

    # Check .env
    if [ -f .env ]; then
        success ".env file exists"
    else
        warn ".env file missing — run './skynet.sh setup' first"
    fi

    [ $fail -eq 0 ] && success "All dependencies satisfied" || die "Missing dependencies"
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_setup() {
    banner
    info "First-time setup wizard (platform: $OS)"
    echo ""

    # ── Install Docker ───────────────────────────────────────────────────────
    if ! check_command docker 2>/dev/null; then
        case "$OS" in
            linux|wsl)
                info "Installing Docker..."
                curl -fsSL https://get.docker.com | sh
                sudo usermod -aG docker "$USER" 2>/dev/null || true
                success "Docker installed"
                ;;
            macos)
                echo ""
                error "Docker is not installed."
                info "Install Docker Desktop for Mac:"
                info "  brew install --cask docker"
                info "  OR download from: https://www.docker.com/products/docker-desktop"
                info ""
                info "After installing, launch Docker Desktop and wait for it to start."
                read -rp "Press Enter once Docker is installed and running... "
                check_command docker 2>/dev/null || die "Docker still not found"
                success "Docker detected"
                ;;
            *)
                die "Cannot auto-install Docker on this platform. Install Docker manually."
                ;;
        esac
    else
        success "Docker already installed"
    fi

    # ── Install Ollama ───────────────────────────────────────────────────────
    if ! check_command ollama 2>/dev/null; then
        case "$OS" in
            linux|wsl)
                info "Installing Ollama..."
                curl -fsSL https://ollama.com/install.sh | sh
                success "Ollama installed"
                ;;
            macos)
                if command -v brew &>/dev/null; then
                    info "Installing Ollama via Homebrew..."
                    brew install ollama
                    success "Ollama installed"
                else
                    info "Installing Ollama..."
                    curl -fsSL https://ollama.com/install.sh | sh
                    success "Ollama installed"
                fi
                ;;
            *)
                warn "Install Ollama manually from: https://ollama.com/download"
                ;;
        esac
    else
        success "Ollama already installed"
    fi

    # ── GPU / CPU detection and interactive prompt ───────────────────────────
    local gpu_mode="cpu"
    local gpu_name=""
    local gpu_vram=""
    local is_wsl=false
    local is_apple_silicon=false

    # Detect WSL
    if [ "$OS" = "wsl" ]; then
        is_wsl=true
        info "Detected: Windows Subsystem for Linux (WSL2)"
    fi

    # Detect Apple Silicon (Metal — auto-accelerated by Ollama)
    if [ "$OS" = "macos" ]; then
        local cpu_brand
        cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
        if echo "$cpu_brand" | grep -qi "apple"; then
            is_apple_silicon=true
            info "Detected: Apple Silicon ($cpu_brand)"
            info "Ollama uses Metal acceleration automatically — no configuration needed"
        fi
    fi

    # Detect NVIDIA GPU (Linux, WSL, or Git Bash on Windows)
    if command -v nvidia-smi &>/dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
        gpu_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
    fi

    if [ -n "$gpu_name" ]; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  dGPU DETECTED                                      ║${NC}"
        echo -e "${GREEN}║  ${gpu_name}${NC}"
        echo -e "${GREEN}║  VRAM: ${gpu_vram}${NC}"
        if $is_wsl; then
            echo -e "${GREEN}║  Runtime: WSL2 (GPU passthrough)                    ║${NC}"
        fi
        echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "  GPU mode:  Ollama uses your GPU for embeddings (fast)"
        echo "             Learner uses CUDA for local model inference"
        echo "             Requires: nvidia-container-toolkit (for Docker GPU)"
        echo ""
        echo "  CPU mode:  Everything runs on CPU (slower but no GPU drivers needed)"
        echo ""
        read -rp "Use GPU acceleration? [Y/n]: " gpu_choice
        case "$(to_lower "${gpu_choice:-y}")" in
            n|no)
                gpu_mode="cpu"
                info "Selected: CPU mode"
                ;;
            *)
                gpu_mode="gpu"
                success "Selected: GPU mode ($gpu_name)"

                # Check nvidia-container-toolkit
                if ! docker info 2>/dev/null | grep -qi nvidia; then
                    warn "nvidia-container-toolkit not detected in Docker"
                    echo ""
                    if $is_wsl; then
                        info "To enable GPU in Docker on WSL2:"
                        info "  1. Install NVIDIA drivers on Windows (not inside WSL)"
                        info "  2. Inside WSL, run the commands below"
                    else
                        info "Install nvidia-container-toolkit:"
                    fi
                    info "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
                    info "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
                    info "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
                    info "  sudo nvidia-ctk runtime configure --runtime=docker"
                    info "  sudo systemctl restart docker"
                    echo ""
                    read -rp "Continue with GPU mode anyway? [Y/n]: " cont
                    case "$(to_lower "${cont:-y}")" in
                        n|no) gpu_mode="cpu"; warn "Falling back to CPU mode" ;;
                    esac
                fi
                ;;
        esac
    elif $is_apple_silicon; then
        # Apple Silicon: Metal is automatic, no GPU/CPU choice needed for Ollama
        # Learner still uses CPU (no CUDA on Mac)
        gpu_mode="cpu"
        info "Apple Silicon: Ollama uses Metal GPU automatically"
        info "Learner (Phi-4-mini) will use CPU (CUDA not available on macOS)"
    else
        info "No NVIDIA GPU detected — using CPU mode"
        if $is_wsl; then
            info "If you have an NVIDIA GPU, install drivers on Windows first"
            info "Then 'nvidia-smi' should work inside WSL2"
        fi
        gpu_mode="cpu"
    fi

    # ── Ollama configuration ─────────────────────────────────────────────────
    # Goal: OLLAMA_HOST=0.0.0.0 (Docker access), KEEP_ALIVE=-1 (never unload),
    #        NUM_PARALLEL=4, and optionally NUM_GPU=0 for CPU mode.
    _configure_ollama "$gpu_mode"

    # ── Write GPU/CPU choice to .env ─────────────────────────────────────────
    local torch_url
    if [ "$gpu_mode" = "gpu" ]; then
        torch_url="https://download.pytorch.org/whl/cu121"
    else
        torch_url="https://download.pytorch.org/whl/cpu"
    fi

    # Create .env from template
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            success "Created .env from .env.example"
        else
            info "Creating minimal .env file..."
            cat > .env <<'ENVEOF'
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

# Ollama — must listen on 0.0.0.0 for Docker access (see ./skynet.sh setup)
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_EMBED_MODEL=nomic-embed-text

# Local Model
LOCAL_MODEL_NAME=microsoft/phi-4-mini-instruct
HF_TOKEN=

# Vision (disabled by default)
VISION_ENABLED=false
CAMERA_INDEX=0
ENVEOF
            success "Created minimal .env — edit it with your API keys"
        fi
    else
        success ".env already exists"
    fi

    # Inject TORCH_INDEX_URL into .env
    if grep -q '^TORCH_INDEX_URL=' .env 2>/dev/null; then
        sed_inplace "s|^TORCH_INDEX_URL=.*|TORCH_INDEX_URL=${torch_url}|" .env
    else
        echo "" >> .env
        echo "# GPU/CPU — set by ./skynet.sh setup (gpu_mode=$gpu_mode)" >> .env
        echo "TORCH_INDEX_URL=${torch_url}" >> .env
    fi
    success "Learner PyTorch: $gpu_mode ($torch_url)"

    # Pull Ollama models
    info "Pulling Ollama embedding model..."
    ollama pull nomic-embed-text 2>/dev/null && success "nomic-embed-text ready" || warn "Could not pull nomic-embed-text"

    info "Pulling Ollama local inference model..."
    ollama pull phi4 2>/dev/null && success "phi4 ready" || warn "Could not pull phi4 (optional)"

    echo ""
    success "Setup complete! Edit .env with your API keys, then run: ./skynet.sh start"
    echo ""
    info "  Platform: $OS"
    info "  GPU mode: $gpu_mode"
    if [ -n "$gpu_name" ]; then
        info "  GPU:      $gpu_name ($gpu_vram)"
    fi
    if $is_apple_silicon; then
        info "  Apple Silicon: Metal (auto-accelerated)"
    fi
    if $is_wsl; then
        if [ "$gpu_mode" = "gpu" ]; then
            info "  WSL2: GPU passthrough enabled"
        else
            info "  WSL2: CPU mode"
        fi
    fi
}

# ── Ollama Configuration (per-platform) ──────────────────────────────────────
_configure_ollama() {
    local gpu_mode="$1"
    local ollama_gpu_env=""
    if [ "$gpu_mode" = "cpu" ]; then
        ollama_gpu_env='OLLAMA_NUM_GPU=0'
    fi

    case "$OS" in
        linux|wsl)
            # Linux/WSL: systemd drop-in override
            if command -v systemctl &>/dev/null; then
                local override_dir="/etc/systemd/system/ollama.service.d"
                local override_file="$override_dir/override.conf"
                local needs_update=false
                if [ ! -f "$override_file" ]; then
                    needs_update=true
                elif ! grep -q 'OLLAMA_HOST=0.0.0.0' "$override_file" 2>/dev/null; then
                    needs_update=true
                elif ! grep -q 'OLLAMA_KEEP_ALIVE' "$override_file" 2>/dev/null; then
                    needs_update=true
                fi

                if $needs_update || [ -n "$ollama_gpu_env" ]; then
                    info "Configuring Ollama via systemd (0.0.0.0, keep_alive=-1, num_parallel=4, mode=$gpu_mode)..."
                    sudo mkdir -p "$override_dir"
                    {
                        echo "[Service]"
                        echo 'Environment="OLLAMA_HOST=0.0.0.0"'
                        echo 'Environment="OLLAMA_KEEP_ALIVE=-1"'
                        echo 'Environment="OLLAMA_NUM_PARALLEL=4"'
                        if [ -n "$ollama_gpu_env" ]; then
                            echo "Environment=\"$ollama_gpu_env\""
                        fi
                    } | sudo tee "$override_file" > /dev/null
                    sudo systemctl daemon-reload
                    sudo systemctl restart ollama 2>/dev/null || true
                    success "Ollama configured via systemd ($gpu_mode)"
                else
                    success "Ollama already configured for Docker access"
                fi
            else
                warn "systemd not found — configure Ollama manually:"
                info "  export OLLAMA_HOST=0.0.0.0"
                info "  export OLLAMA_KEEP_ALIVE=-1"
                info "  export OLLAMA_NUM_PARALLEL=4"
                if [ -n "$ollama_gpu_env" ]; then
                    info "  export $ollama_gpu_env"
                fi
                info "  Then restart: ollama serve"
            fi
            ;;

        macos)
            # macOS: launchctl setenv for the GUI app + shell profile
            info "Configuring Ollama for Docker access (macOS)..."

            # On macOS with Docker Desktop, host.docker.internal routes through
            # the VM, so Ollama on 127.0.0.1 IS reachable. But setting 0.0.0.0
            # is still good practice for network access.
            launchctl setenv OLLAMA_HOST "0.0.0.0" 2>/dev/null || true
            launchctl setenv OLLAMA_KEEP_ALIVE "-1" 2>/dev/null || true
            launchctl setenv OLLAMA_NUM_PARALLEL "4" 2>/dev/null || true

            # Also set for current shell session
            export OLLAMA_HOST="0.0.0.0"
            export OLLAMA_KEEP_ALIVE="-1"
            export OLLAMA_NUM_PARALLEL="4"

            # Write to shell profile for persistence
            local shell_profile=""
            if [ -f "$HOME/.zshrc" ]; then
                shell_profile="$HOME/.zshrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                shell_profile="$HOME/.bash_profile"
            elif [ -f "$HOME/.bashrc" ]; then
                shell_profile="$HOME/.bashrc"
            fi

            if [ -n "$shell_profile" ]; then
                if ! grep -q 'OLLAMA_HOST' "$shell_profile" 2>/dev/null; then
                    {
                        echo ""
                        echo "# SkyNet: Ollama configuration"
                        echo 'export OLLAMA_HOST="0.0.0.0"'
                        echo 'export OLLAMA_KEEP_ALIVE="-1"'
                        echo 'export OLLAMA_NUM_PARALLEL="4"'
                    } >> "$shell_profile"
                    success "Ollama env vars added to $shell_profile"
                fi
            fi

            # Try to restart Ollama
            if command -v brew &>/dev/null && brew services list 2>/dev/null | grep -q ollama; then
                brew services restart ollama 2>/dev/null && success "Ollama restarted via brew" || true
            else
                warn "Restart Ollama manually (quit and reopen the app) for changes to take effect"
            fi
            ;;

        *)
            warn "Unknown platform — configure Ollama manually (OLLAMA_HOST=0.0.0.0)"
            ;;
    esac
}

cmd_start() {
    banner
    check_deps

    info "Building containers..."
    compose_build

    # ── Phase 1: Infrastructure first (always hot, zero cold starts) ─────────
    info "Starting infrastructure (Redis, Elasticsearch, EdgeDB)..."
    $COMPOSE up -d redis elasticsearch edgedb

    info "Waiting for databases to be healthy..."
    local retries=0
    while [ $retries -lt 30 ]; do
        if $COMPOSE ps --format json 2>/dev/null | grep -q '"healthy"' || \
           $COMPOSE ps 2>/dev/null | grep -q "healthy"; then
            break
        fi
        sleep 2
        retries=$((retries + 1))
    done
    success "Databases ready"

    info "Initializing databases..."
    $COMPOSE up db-init 2>/dev/null || warn "db-init may have already run"

    # ── Phase 2: Warm up Ollama (never cold, never unloading) ────────────────
    info "Warming up Ollama (pre-loading embed model)..."
    local ollama_url
    ollama_url=$(grep '^OLLAMA_URL=' .env 2>/dev/null | cut -d= -f2- || echo "http://localhost:11434")
    ollama_url="${ollama_url:-http://localhost:11434}"
    # Convert Docker internal URL to localhost for host-side curl
    ollama_url=$(echo "$ollama_url" | sed 's|host.docker.internal|localhost|')
    local embed_model
    embed_model=$(grep '^OLLAMA_EMBED_MODEL=' .env 2>/dev/null | cut -d= -f2- || echo "nomic-embed-text")
    embed_model="${embed_model:-nomic-embed-text}"
    if curl -sf "${ollama_url}/api/embeddings" -d "{\"model\":\"${embed_model}\",\"prompt\":\"warmup\",\"keep_alive\":-1}" &>/dev/null; then
        success "Ollama embed model loaded (${embed_model}, keep_alive=-1)"
    else
        warn "Could not warm up Ollama — embeddings may be slow on first call"
    fi
    # Also warm up the local inference model
    local local_model
    local_model=$(grep '^OLLAMA_LOCAL_MODEL=' .env 2>/dev/null | cut -d= -f2- || echo "phi4")
    local_model="${local_model:-phi4}"
    if curl -sf "${ollama_url}/api/generate" -d "{\"model\":\"${local_model}\",\"prompt\":\"hi\",\"keep_alive\":-1}" &>/dev/null; then
        success "Ollama local model loaded (${local_model}, keep_alive=-1)"
    else
        info "Ollama local model warmup skipped (${local_model} — may not be pulled)"
    fi

    # ── Phase 3: Start all services ──────────────────────────────────────────
    info "Starting all services..."
    $COMPOSE up -d

    info "Waiting for system health..."
    local health_retries=0
    while [ $health_retries -lt 60 ]; do
        if curl -sf "$GATEWAY_URL/health" &>/dev/null; then
            break
        fi
        sleep 3
        health_retries=$((health_retries + 1))
    done

    if curl -sf "$GATEWAY_URL/health" &>/dev/null; then
        success "All services are running!"
    else
        warn "Some services may still be starting — check: ./skynet.sh status"
    fi

    echo ""
    success "SkyNet is online — running red-hot"
    echo ""
    info "  Dashboard:  $GATEWAY_URL              (main UI)"
    info "  Health:     $GATEWAY_URL/health        (system health)"
    info "  Embed Stats:$GATEWAY_URL/brain/embed-queue/stats"
    info ""
    info "Management:"
    info "  ./skynet.sh stop      Graceful shutdown"
    info "  ./skynet.sh restart   Quick restart (no rebuild, keeps data)"
    info "  ./skynet.sh update    Git pull + rebuild + restart"
    info "  ./skynet.sh status    Service health overview"
    info "  ./skynet.sh logs      Tail all logs (or ./skynet.sh logs embed-queue)"
    info ""
    info "Try: curl -X POST $GATEWAY_URL/brain/process -H 'Content-Type: application/json' -d '{\"session_id\":\"s1\",\"user_input\":\"Hello\"}'"
}

cmd_stop() {
    banner
    info "Gracefully shutting down all services..."
    $COMPOSE down
    success "All services stopped"
}

cmd_restart() {
    banner
    info "Restarting services (preserving all data)..."
    $COMPOSE stop
    $COMPOSE up -d
    info "Waiting for system health..."
    local health_retries=0
    while [ $health_retries -lt 30 ]; do
        if curl -sf "$GATEWAY_URL/health" &>/dev/null; then
            break
        fi
        sleep 3
        health_retries=$((health_retries + 1))
    done
    if curl -sf "$GATEWAY_URL/health" &>/dev/null; then
        success "All services restarted — data preserved"
    else
        warn "Some services may still be starting — check: ./skynet.sh status"
    fi
}

cmd_status() {
    banner
    info "Service Status:"
    echo ""
    $COMPOSE ps
    echo ""

    info "Brain Health:"
    if curl -sf "$GATEWAY_URL/health" 2>/dev/null; then
        curl -s "$GATEWAY_URL/health" | python3 -m json.tool 2>/dev/null || curl -s "$GATEWAY_URL/health"
    else
        warn "Gateway not responding at $GATEWAY_URL"
    fi
}

cmd_logs() {
    local service="${1:-}"
    if [ -n "$service" ]; then
        info "Tailing logs for $service..."
        $COMPOSE logs -f "$service"
    else
        info "Tailing all logs..."
        $COMPOSE logs -f
    fi
}

cmd_dashboard() {
    banner
    info "Dashboard URL: $GATEWAY_URL"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$GATEWAY_URL" 2>/dev/null &
        success "Opening dashboard in browser..."
    elif command -v open &>/dev/null; then
        open "$GATEWAY_URL" 2>/dev/null &
        success "Opening dashboard in browser..."
    else
        info "Open manually: $GATEWAY_URL"
    fi
}

cmd_think() {
    banner
    info "Activating autonomy mode..."
    local response
    response=$(curl -sf -X POST "$AUTONOMY_URL/activate" \
        -H "Content-Type: application/json" \
        -d '{"mode": "autonomous"}' 2>&1) || true

    if [ -n "$response" ]; then
        success "Autonomy mode activated"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    else
        warn "Could not reach autonomy service at $AUTONOMY_URL"
        info "Make sure the autonomy service is running: docker compose ps autonomy"
    fi
}

cmd_update() {
    banner
    info "Pulling latest code..."
    git pull --rebase || git pull

    info "Rebuilding changed services..."
    compose_build

    info "Restarting updated services..."
    $COMPOSE up -d

    success "Update complete"
    info "Check health: ./skynet.sh status"
}

cmd_reset() {
    banner
    warn "FACTORY RESET — This will destroy ALL data (memories, models, goals, etc.)"
    echo ""
    read -rp "Type 'RESET' to confirm: " confirm
    if [ "$confirm" != "RESET" ]; then
        info "Reset cancelled"
        exit 0
    fi

    info "Stopping all services..."
    $COMPOSE down -v --remove-orphans

    info "Rebuilding from scratch..."
    compose_build --no-cache

    info "Starting fresh system..."
    cmd_start

    success "Factory reset complete — SkyNet has no memory of the past"
}

cmd_rebuild() {
    banner
    warn "REBUILD — This will wipe ALL databases and rebuild everything fresh"
    echo ""
    read -rp "Type 'REBUILD' to confirm: " confirm
    if [ "$confirm" != "REBUILD" ]; then
        info "Rebuild cancelled"
        exit 0
    fi

    info "Stopping all services and removing volumes..."
    $COMPOSE down -v --remove-orphans

    info "Rebuilding all containers..."
    compose_build --no-cache

    info "Starting fresh system..."
    cmd_start

    success "Rebuild complete — fresh environment, all databases recreated"
}

cmd_help() {
    banner
    cat <<EOF
Usage: ./skynet.sh <command> [options]

Platform: $OS  (Linux, macOS, WSL supported)
Windows:  Use skynet.bat or skynet.ps1 instead

Commands:
  setup       First-time wizard: Docker, Ollama, GPU/CPU config, .env, model pull
  start       Full startup: build, init DBs, warm Ollama, start all 24 services
  stop        Graceful shutdown (data preserved)
  restart     Quick restart — no rebuild, keeps all data and conversations
  update      Git pull + rebuild changed services + restart (use after code changes)
  status      Show all service health and brain diagnostics
  logs [svc]  Tail logs for a service (or all). Try: ./skynet.sh logs embed-queue
  dashboard   Open the dashboard in your browser
  think       Activate autonomy mode (self-reflective thinking)
  rebuild     Full wipe: destroy databases + rebuild all containers from scratch
  reset       Factory reset (alias for rebuild)
  help        Show this help message

Scenarios:
  First install:            ./skynet.sh setup && ./skynet.sh start
  After code changes:       ./skynet.sh update
  Quick restart (no build): ./skynet.sh restart
  Config change (.env):     ./skynet.sh restart
  Wipe and reinstall:       ./skynet.sh rebuild
  After system reboot:      ./skynet.sh start

Infrastructure:
  Ollama runs on the HOST (not in Docker) with keep_alive=-1 (never unloads models).
  Redis, Elasticsearch, EdgeDB run in Docker with restart: unless-stopped.
  All infrastructure is pre-heated on start — zero cold starts.
EOF
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    logs)      shift; cmd_logs "$@" ;;
    dashboard) cmd_dashboard ;;
    think)     cmd_think ;;
    setup)     cmd_setup ;;
    update)    cmd_update ;;
    rebuild)   cmd_rebuild ;;
    reset)     cmd_reset ;;
    help|--help|-h) cmd_help ;;
    *)         error "Unknown command: $1"; cmd_help; exit 1 ;;
esac
