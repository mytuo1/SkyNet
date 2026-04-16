#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# SkyNet — Cross-Platform Startup Script (Ubuntu)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

check_version() {
    local cmd="$1" min="$2" actual
    actual=$($cmd 2>&1 | grep -oP '\d+\.\d+' | head -1) || true
    if [ -z "$actual" ]; then
        warn "Could not determine $cmd version"
        return 0
    fi
    if printf '%s\n%s\n' "$min" "$actual" | sort -V -C; then
        success "$cmd version $actual (>= $min)"
    else
        error "$cmd version $actual is below minimum $min"
        return 1
    fi
}

check_deps() {
    info "Checking dependencies..."
    local fail=0

    check_command docker   || fail=1
    check_command curl     || fail=1
    check_command git      || fail=1

    if [ $fail -eq 0 ]; then
        check_version "docker --version" "24.0" || fail=1
        # Docker Compose version
        local compose_ver
        compose_ver=$(docker compose version 2>&1 | grep -oP '\d+\.\d+' | head -1) || true
        if [ -n "$compose_ver" ]; then
            if printf '%s\n%s\n' "2.20" "$compose_ver" | sort -V -C; then
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
            # Warn if Ollama is bound to 127.0.0.1 only (unreachable from Docker)
            if command -v ss &>/dev/null; then
                local ss_out
                ss_out=$(ss -tlnp 2>/dev/null | grep ':11434' || true)
                if echo "$ss_out" | grep -q '127.0.0.1' && \
                   ! echo "$ss_out" | grep -qE '0\.0\.0\.0|\*|\[::\]'; then
                    warn "Ollama is bound to 127.0.0.1 — Docker containers cannot reach it"
                    warn "Fix: run './skynet.sh setup' or set OLLAMA_HOST=0.0.0.0 and restart Ollama"
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
    info "First-time setup wizard"
    echo ""

    # Install Docker if missing
    if ! check_command docker 2>/dev/null; then
        info "Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "$USER" 2>/dev/null || true
        success "Docker installed"
    else
        success "Docker already installed"
    fi

    # Install Ollama if missing
    if ! check_command ollama 2>/dev/null; then
        info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        success "Ollama installed"
    else
        success "Ollama already installed"
    fi

    # ── Ollama binding: ensure it listens on 0.0.0.0 for Docker access ──────
    # By default Ollama binds to 127.0.0.1, which is unreachable from Docker
    # containers. On Linux with systemd we create a drop-in override.
    if [ "$(uname)" = "Linux" ] && command -v systemctl &>/dev/null; then
        local override_dir="/etc/systemd/system/ollama.service.d"
        local override_file="$override_dir/override.conf"
        if [ ! -f "$override_file" ] || ! grep -q 'OLLAMA_HOST=0.0.0.0' "$override_file" 2>/dev/null; then
            info "Configuring Ollama to listen on 0.0.0.0 (required for Docker)..."
            sudo mkdir -p "$override_dir"
            printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n' | sudo tee "$override_file" > /dev/null
            sudo systemctl daemon-reload
            sudo systemctl restart ollama 2>/dev/null || true
            success "Ollama configured to listen on 0.0.0.0"
        else
            success "Ollama already configured for Docker access"
        fi
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

    # Pull Ollama models
    info "Pulling Ollama embedding model..."
    ollama pull nomic-embed-text 2>/dev/null && success "nomic-embed-text ready" || warn "Could not pull nomic-embed-text"

    info "Pulling Ollama local inference model..."
    ollama pull phi4 2>/dev/null && success "phi4 ready" || warn "Could not pull phi4 (optional)"

    echo ""
    success "Setup complete! Edit .env with your API keys, then run: ./skynet.sh start"
}

cmd_start() {
    banner
    check_deps

    info "Building containers..."
    compose_build

    info "Starting databases..."
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
    success "SkyNet is online"
    echo ""
    info "  Dashboard:  $GATEWAY_URL              (main UI)"
    info "  Health:     $GATEWAY_URL/health        (system health)"
    info "  Knobs API:  $GATEWAY_URL/knobs/current (behavioral parameters)"
    info ""
    info "Management:"
    info "  ./skynet.sh stop      Graceful shutdown"
    info "  ./skynet.sh restart   Stop + start"
    info "  ./skynet.sh status    Service health overview"
    info "  ./skynet.sh logs      Tail all logs"
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

Commands:
  start       Full startup: check deps, build, init DBs, start all services
  stop        Graceful shutdown of all services
  restart     Quick restart (preserves all data and conversations)
  status      Show all service health
  logs [svc]  Tail logs for a service (or all)
  dashboard   Open the dashboard in your browser
  think       Activate autonomy mode
  setup       First-time setup wizard (installs Docker, Ollama, creates .env)
  update      Git pull + rebuild changed services
  rebuild     Wipe databases and rebuild fresh (new environment)
  reset       Factory reset (alias for rebuild)
  help        Show this help message

Examples:
  ./skynet.sh setup          # First time? Start here
  ./skynet.sh start          # Launch everything
  ./skynet.sh restart        # Quick restart, keeps conversations
  ./skynet.sh status         # Check system health
  ./skynet.sh logs thalamus  # Watch the orchestrator
  ./skynet.sh think          # Enable autonomous thinking
  ./skynet.sh dashboard      # Open the web dashboard
  ./skynet.sh rebuild        # Fresh environment, wipes all data
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
