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

# ── GPU override ──────────────────────────────────────────────────────────────
# When GPU_MODE=gpu is set in .env (written by setup), compose includes the
# docker-compose.gpu.yml override which adds NVIDIA device reservations.
# This avoids "nvidia-container-cli: initialization error" on machines where
# the NVIDIA container runtime is not available (WSL2 snap Docker, CPU-only).
_compose_files() {
    if [ -f .env ] && grep -q '^GPU_MODE=gpu' .env 2>/dev/null; then
        if [ -f docker-compose.gpu.yml ]; then
            echo "docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
            return
        fi
    fi
    echo "docker compose"
}
COMPOSE="$(_compose_files)"

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

# ── Docker Install Method Detection ──────────────────────────────────────────
# Returns: "snap", "systemd", "desktop", "service", or "unknown"
# Used for restart commands and nvidia-ctk daemon.json path.
_docker_method() {
    local _di
    _di=$(docker info 2>/dev/null || true)
    # Snap Docker: root dir under /var/snap/docker
    if echo "$_di" | grep -q '/var/snap/docker'; then
        echo "snap"; return
    elif [ -d "/var/snap/docker" ] && command -v snap &>/dev/null; then
        if snap list docker &>/dev/null; then
            echo "snap"; return
        fi
    fi
    # Docker Desktop: look for Desktop markers in docker info or context
    if echo "$_di" | grep -qiE 'Docker Desktop|desktop-linux|desktop.linux'; then
        echo "desktop"; return
    elif docker context ls 2>/dev/null | grep -q 'desktop-linux'; then
        echo "desktop"; return
    fi
    # systemd service
    if command -v systemctl &>/dev/null && systemctl list-unit-files docker.service &>/dev/null 2>&1; then
        echo "systemd"; return
    fi
    # SysV init / upstart
    if [ -f /etc/init.d/docker ] || command -v service &>/dev/null; then
        echo "service"; return
    fi
    echo "unknown"
}

# Print the right Docker restart command for the current install method.
_docker_restart_hint() {
    case "$(_docker_method)" in
        snap)    info "  sudo snap restart docker" ;;
        desktop) info "  Restart Docker Desktop from the system tray (right-click → Restart)" ;;
        systemd) info "  sudo systemctl restart docker" ;;
        service) info "  sudo service docker restart" ;;
        *)       info "  Restart Docker (close and reopen, or use your init system)" ;;
    esac
}

# Print nvidia-ctk configure command with the correct daemon.json path.
_nvidia_ctk_hint() {
    if [ "$(_docker_method)" = "snap" ]; then
        # Snap Docker uses a different daemon.json location
        info "  sudo nvidia-ctk runtime configure --runtime=docker --config /var/snap/docker/current/config/daemon.json"
    else
        info "  sudo nvidia-ctk runtime configure --runtime=docker"
    fi
}

# ── nvidia-container-cli honest check ─────────────────────────────────────────
# Runs nvidia-container-cli WITHOUT LD_LIBRARY_PATH to simulate the environment
# Docker's prestart hook uses. This prevents false positives: if the check only
# passes because of LD_LIBRARY_PATH, Docker's hook will still fail.
_nvidia_cli_check() {
    env -u LD_LIBRARY_PATH nvidia-container-cli info &>/dev/null
}

# ── GPU Runtime Validation (called at every start/restart/update) ─────────────
# When GPU_MODE=gpu is set in .env, this function ensures the NVIDIA container
# runtime actually works before Docker tries to start GPU containers.
#
# On WSL2, libnvidia-ml.so.1 lives at /usr/lib/wsl/lib/ but ldconfig can lose
# the registration after a WSL restart. This function re-applies the fix every
# time — it's idempotent and takes <100ms.
#
# CRITICAL: The nvidia-container-cli check must run WITHOUT LD_LIBRARY_PATH set,
# because Docker's prestart hook runs from the Docker daemon's environment —
# which does NOT inherit this script's LD_LIBRARY_PATH. If we check with
# LD_LIBRARY_PATH, it masks the real problem and the container still fails.
#
# If the GPU runtime is broken and unfixable, it falls back to CPU compose
# (no docker-compose.gpu.yml) so the learner still starts on CPU.
_ensure_gpu_runtime() {
    # Only relevant when GPU mode is active
    [ -f .env ] && grep -q '^GPU_MODE=gpu' .env 2>/dev/null || return 0

    local is_wsl=false
    grep -qi microsoft /proc/version 2>/dev/null && is_wsl=true

    # ── WSL2: ALWAYS force ldconfig (unconditional — fast & idempotent) ──
    # On WSL2, NVIDIA libraries live at /usr/lib/wsl/lib/ but ldconfig loses
    # the registration after every `wsl --shutdown` or Windows reboot.
    # We always re-register. This takes <50ms and is the single most common
    # cause of "libnvidia-ml.so.1: cannot open shared object file".
    local wsl_config_changed=false
    if $is_wsl && [ -d /usr/lib/wsl/lib ]; then
        echo "/usr/lib/wsl/lib" | sudo tee /etc/ld.so.conf.d/nvidia-wsl.conf >/dev/null 2>&1
        sudo ldconfig 2>/dev/null

        # ── Copy WSL NVIDIA libs into standard system path ───────────────
        # On WSL2, /usr/lib/wsl/lib/ is a special directory that snap Docker
        # cannot access — its prestart hooks run nvidia-container-cli inside
        # the snap daemon's confined context, which cannot reach that path.
        # Symlinks don't help either since following them requires the same
        # path access. Actual file COPIES into /usr/lib/x86_64-linux-gnu/
        # are needed so the dynamic linker finds real files in a standard
        # path. Copies are refreshed if the source is newer (driver update).
        # Won't clobber native driver installs (skips existing regular files
        # that aren't our copies).
        local _wsl_copy_changed=false
        for _lib in /usr/lib/wsl/lib/lib*.so*; do
            [ -f "$_lib" ] || continue
            local _basename
            _basename="$(basename "$_lib")"
            local _dst="/usr/lib/x86_64-linux-gnu/$_basename"
            if [ -L "$_dst" ]; then
                # Replace stale symlink (from previous fix version) with copy
                sudo cp -f "$_lib" "$_dst" 2>/dev/null && _wsl_copy_changed=true
            elif [ ! -e "$_dst" ]; then
                # Destination doesn't exist — copy
                sudo cp -f "$_lib" "$_dst" 2>/dev/null && _wsl_copy_changed=true
            elif [ "$_lib" -nt "$_dst" ]; then
                # Source is newer (driver updated) — refresh the copy
                sudo cp -f "$_lib" "$_dst" 2>/dev/null && _wsl_copy_changed=true
            fi
        done
        if $_wsl_copy_changed; then
            sudo ldconfig 2>/dev/null
            info "Copied WSL2 NVIDIA libraries into /usr/lib/x86_64-linux-gnu/"
            # Snap Docker runs in a confined AppArmor context that may cache
            # filesystem state. Force a daemon restart so it sees the new libs.
            if [ "$(_docker_method)" = "snap" ]; then
                wsl_config_changed=true
            fi
        fi

        # Verify the ldconfig cache was actually updated
        if ! ldconfig -p 2>/dev/null | grep -q libnvidia-ml; then
            warn "ldconfig cache missing libnvidia-ml — forcing cache rebuild..."
            sudo ldconfig 2>/dev/null || true
        fi

        # WSL2: fix nvidia-container-toolkit config.toml
        # This fixes BOTH issues that break GPU containers on WSL2:
        #   1) ldconfig = "@/sbin/ldconfig.real" → "@/sbin/ldconfig"
        #      ("ldconfig.real" is Ubuntu-specific; "/sbin/ldconfig" is portable.
        #       The "@" prefix is required by toolkit v1.17+ for host-relative paths.
        #       Since we registered /usr/lib/wsl/lib with ldconfig above, the host
        #       ldconfig cache now includes WSL2 NVIDIA libs.)
        #   2) no-cgroups = false → true
        #      (WSL2 lacks standard cgroup device controller)
        if ! _ensure_wsl2_nvidia_config; then
            # Return code 1 = changes were made → Docker must restart
            wsl_config_changed=true
        fi

        # If config.toml was modified, Docker MUST restart to pick up changes.
        # Without this, the running daemon still uses the old "@/sbin/ldconfig.real"
        # and the prestart hook fails even though nvidia-container-cli works on host.
        if $wsl_config_changed; then
            info "WSL2 NVIDIA config was updated — restarting Docker daemon..."
            local _dm
            _dm="$(_docker_method)"
            case "$_dm" in
                snap)    sudo snap restart docker 2>/dev/null ;;
                systemd) sudo systemctl restart docker 2>/dev/null ;;
                service) sudo service docker restart 2>/dev/null ;;
                desktop) info "Please restart Docker Desktop from the system tray" ;;
            esac
            sleep 3
        fi
    fi

    # ── Native Linux: find libnvidia-ml.so.1 in non-standard locations ───
    # Some distros/drivers install to /usr/lib/x86_64-linux-gnu/ or other
    # paths that ldconfig doesn't know about by default.
    if ! $is_wsl; then
        local nvidia_lib_path=""
        nvidia_lib_path=$(find /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu \
            /usr/local/lib -name "libnvidia-ml.so.1" 2>/dev/null | head -1 || true)
        if [ -n "$nvidia_lib_path" ]; then
            local nvidia_lib_dir
            nvidia_lib_dir=$(dirname "$nvidia_lib_path")
            if ! ldconfig -p 2>/dev/null | grep -q libnvidia-ml; then
                info "Registering NVIDIA libraries from $nvidia_lib_dir..."
                echo "$nvidia_lib_dir" | sudo tee /etc/ld.so.conf.d/nvidia-skynet.conf >/dev/null 2>&1
                sudo ldconfig 2>/dev/null
            fi
        fi
    fi

    # ── Verify nvidia-container-cli (the binary Docker actually uses) ────
    # IMPORTANT: Check WITHOUT LD_LIBRARY_PATH — Docker's prestart hook runs
    # from the daemon's environment, not ours. If we let LD_LIBRARY_PATH mask
    # the problem, the check passes but the container still fails with:
    #   "libnvidia-ml.so.1: cannot open shared object file"
    if command -v nvidia-container-cli &>/dev/null; then
        # Strip LD_LIBRARY_PATH for an honest check (simulates Docker's hook env)
        if _nvidia_cli_check; then
            # GPU runtime is healthy — set LD_LIBRARY_PATH as belt+suspenders
            if $is_wsl; then
                export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
            return 0
        fi

        # nvidia-container-cli failed — Docker daemon may need restart to see
        # the updated ldconfig cache (daemon started before ldconfig ran)
        warn "nvidia-container-cli cannot find NVIDIA libs — restarting Docker daemon..."
        local _dm2
        _dm2="$(_docker_method)"
        case "$_dm2" in
            snap)    sudo snap restart docker 2>/dev/null ;;
            systemd) sudo systemctl restart docker 2>/dev/null ;;
            service) sudo service docker restart 2>/dev/null ;;
            desktop) info "Please restart Docker Desktop from the system tray" ;;
        esac
        sleep 3

        if _nvidia_cli_check; then
            success "GPU runtime recovered after Docker restart"
            if $is_wsl; then
                export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
            return 0
        fi

        # Still broken — fall back to CPU
        warn "NVIDIA GPU runtime is not working — falling back to CPU for this session"
        warn "  The learner will start on CPU (slower but functional)"
        warn ""
        warn "  To retry with GPU after fixing the issue:"
        warn "    ./skynet.sh restart   (re-validates GPU runtime before starting)"
        warn ""
        warn "  Common fixes:"
        if $is_wsl; then
            warn "    • Ensure nvidia-container-toolkit is installed in WSL"
            warn "    • Run: ./skynet.sh setup  (auto-fixes config.toml + copies WSL libs)"
        else
            warn "    • Install nvidia-container-toolkit and restart Docker"
        fi
        _docker_restart_hint
        COMPOSE="docker compose"
        return 0
    fi

    # No nvidia-container-cli — check if Docker at least knows about NVIDIA runtime
    if docker info 2>/dev/null | grep -qi nvidia; then
        if $is_wsl; then
            export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        return 0
    fi

    # No NVIDIA runtime at all — fall back to CPU
    warn "NVIDIA container runtime not detected — falling back to CPU for this session"
    warn "  Install nvidia-container-toolkit to enable GPU containers"
    COMPOSE="docker compose"
    return 0
}

# ── WSL2: fix nvidia-container-toolkit config.toml for WSL2 ──────────────────
# On WSL2, the default config.toml has two settings that break GPU containers:
#
# 1) ldconfig = "@/sbin/ldconfig.real"
#    "ldconfig.real" is Ubuntu-specific and may not exist on all distros.
#    Fix: ldconfig = "@/sbin/ldconfig" (portable path, "@" = host-relative)
#    The "@" prefix is REQUIRED by nvidia-container-toolkit v1.17+ and tells
#    nvidia-container-cli to use the host's ldconfig binary. This works because
#    we register /usr/lib/wsl/lib with ldconfig BEFORE this fix runs, so the
#    host cache already includes the WSL2 NVIDIA libraries.
#
# 2) no-cgroups = false (or commented out)
#    WSL2 lacks the standard cgroup device controller. Without no-cgroups=true,
#    nvidia-container-cli fails with cgroup setup errors.
#    Fix: no-cgroups = true
#
# Returns 0 if config was already correct, 1 if changes were made (caller
# must restart Docker for changes to take effect).
_ensure_wsl2_nvidia_config() {
    # Find the config file — check both toolkit paths
    local config_file=""
    for candidate in /etc/nvidia-container-runtime/config.toml \
                     /etc/nvidia-container-toolkit/config.toml; do
        if [ -f "$candidate" ]; then
            config_file="$candidate"
            break
        fi
    done

    # No config file found — try to create one if the toolkit is installed
    if [ -z "$config_file" ]; then
        if [ -d /etc/nvidia-container-runtime ] || command -v nvidia-container-runtime &>/dev/null; then
            config_file="/etc/nvidia-container-runtime/config.toml"
            sudo mkdir -p "$(dirname "$config_file")" 2>/dev/null || return 0
            info "Creating nvidia-container-toolkit config for WSL2..."
            sudo tee "$config_file" >/dev/null 2>&1 <<'NCREOF'
[nvidia-container-cli]
no-cgroups = true
ldconfig = "@/sbin/ldconfig"
NCREOF
            return 1  # config was created — Docker restart needed
        fi
        return 0  # no toolkit installed
    fi

    local changed=0

    # ── Fix 1: no-cgroups = true ──────────────────────────────────────────
    if grep -q '^no-cgroups[[:space:]]*=[[:space:]]*true' "$config_file" 2>/dev/null; then
        : # already correct
    elif grep -q '^no-cgroups' "$config_file" 2>/dev/null; then
        # Uncommented but wrong value — replace
        sudo sed -i 's|^no-cgroups.*|no-cgroups = true|' "$config_file" 2>/dev/null
        changed=1
    elif grep -q '^#.*no-cgroups' "$config_file" 2>/dev/null; then
        # Commented out — replace the commented line with the correct value
        sudo sed -i 's|^#.*no-cgroups.*|no-cgroups = true|' "$config_file" 2>/dev/null
        changed=1
    elif grep -q '\[nvidia-container-cli\]' "$config_file" 2>/dev/null; then
        # Section exists but no no-cgroups line at all — insert after header
        sudo sed -i '/\[nvidia-container-cli\]/a no-cgroups = true' "$config_file" 2>/dev/null
        changed=1
    else
        # No section header — append
        printf '\n[nvidia-container-cli]\nno-cgroups = true\n' | sudo tee -a "$config_file" >/dev/null 2>&1
        changed=1
    fi

    # ── Fix 2: ldconfig = "@/sbin/ldconfig" (host-relative, portable) ─────
    # The "@" prefix is REQUIRED by nvidia-container-toolkit v1.17+ and means
    # "use this binary from the host". Without "@", the toolkit rejects the
    # config with: 'ldconfig value is not host-relative (does not start with @)'.
    # We use "/sbin/ldconfig" instead of the Ubuntu-specific "/sbin/ldconfig.real"
    # for portability. The host ldconfig cache includes WSL2 NVIDIA libs because
    # we register /usr/lib/wsl/lib with ldconfig before reaching this point.
    if grep -q '^ldconfig[[:space:]]*=[[:space:]]*"@/sbin/ldconfig"' "$config_file" 2>/dev/null; then
        : # already correct ("@/sbin/ldconfig")
    elif grep -q "^ldconfig[[:space:]]*=[[:space:]]*'@/sbin/ldconfig'" "$config_file" 2>/dev/null; then
        : # already correct (single quotes)
    elif grep -q '^ldconfig' "$config_file" 2>/dev/null; then
        # Has ldconfig but wrong value (e.g. "@/sbin/ldconfig.real" or "/sbin/ldconfig") — replace
        sudo sed -i 's|^ldconfig.*|ldconfig = "@/sbin/ldconfig"|' "$config_file" 2>/dev/null
        changed=1
    elif grep -q '^#.*ldconfig' "$config_file" 2>/dev/null; then
        # Commented out — replace with correct value
        sudo sed -i 's|^#.*ldconfig.*|ldconfig = "@/sbin/ldconfig"|' "$config_file" 2>/dev/null
        changed=1
    elif grep -q '\[nvidia-container-cli\]' "$config_file" 2>/dev/null; then
        # Section exists but no ldconfig line — insert after header
        sudo sed -i '/\[nvidia-container-cli\]/a ldconfig = "@/sbin/ldconfig"' "$config_file" 2>/dev/null
        changed=1
    fi

    if [ "$changed" -eq 1 ]; then
        info "Fixed nvidia-container-toolkit config.toml for WSL2 ($config_file)"
    fi
    return "$changed"
}

# ── Centralized compose-up wrapper ────────────────────────────────────────────
# EVERY code path that starts containers MUST use this wrapper.
# It guarantees GPU runtime validation before any `docker compose up`.
_compose_up() {
    _ensure_gpu_runtime
    # Re-evaluate COMPOSE in case _ensure_gpu_runtime fell back to CPU
    COMPOSE="$(_compose_files)"
    $COMPOSE up "$@"
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
    local gpu_vendor=""   # "nvidia" | "amd" | "apple" | ""
    local gpu_list=""     # newline-separated CSV: index,name,vram (for multi-GPU)
    local gpu_count=0
    local gpu_select="all"  # which GPUs to use: "all", "0", "1", "0,1", etc.
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
            gpu_vendor="apple"
            gpu_name="$cpu_brand"
            info "Detected: Apple Silicon ($cpu_brand)"
            info "Ollama uses Metal acceleration automatically — no configuration needed"
        fi
    fi

    # ── NVIDIA GPU Detection (multi-method, ironclad) ────────────────────────
    # Method 1: Find nvidia-smi binary
    #   - PATH lookup (works on native Linux, non-sudo WSL)
    #   - /usr/lib/wsl/lib/nvidia-smi (WSL2 driver shim — the REAL location on WSL2)
    #   - /usr/local/cuda/bin/nvidia-smi (CUDA toolkit install)
    #   - Standard FHS paths (covers edge cases)
    # Method 2: If running under sudo on WSL2, nvidia-smi may fail because WSL
    #   interop doesn't work under root. Drop to the real user ($SUDO_USER).
    # Method 3: /proc/driver/nvidia/version (kernel driver loaded, no nvidia-smi needed)
    # Method 4: lspci hardware scan (native Linux only — not available on WSL2)
    if [ -z "$gpu_vendor" ]; then
        local nvidia_smi_cmd=""

        # Scan all known nvidia-smi paths
        local nv_paths="nvidia-smi /usr/lib/wsl/lib/nvidia-smi /usr/local/cuda/bin/nvidia-smi /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi /opt/cuda/bin/nvidia-smi"
        for nv_path in $nv_paths; do
            if [ "$nv_path" = "nvidia-smi" ]; then
                # PATH-based lookup
                if command -v nvidia-smi &>/dev/null; then
                    nvidia_smi_cmd="nvidia-smi"
                    break
                fi
            elif [ -x "$nv_path" ]; then
                nvidia_smi_cmd="$nv_path"
                break
            fi
        done

        # Build environment for nvidia-smi (WSL2 needs LD_LIBRARY_PATH for libnvidia-ml.so)
        local nv_env=""
        if [ -d "/usr/lib/wsl/lib" ]; then
            nv_env="LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
        fi

        # Try running nvidia-smi — enumerate ALL GPUs
        if [ -n "$nvidia_smi_cmd" ]; then
            # Attempt 1: Query structured CSV for all GPUs (index, name, VRAM)
            gpu_list=$(env $nv_env $nvidia_smi_cmd --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)

            # Attempt 2: On WSL2 under sudo, nvidia-smi may fail (WSL interop broken under root).
            #            Drop to the real user who invoked sudo.
            if [ -z "$gpu_list" ] && [ -n "${SUDO_USER:-}" ] && $is_wsl; then
                info "nvidia-smi failed as root — trying as $SUDO_USER (WSL2 interop)..."
                gpu_list=$(su - "$SUDO_USER" -c "env $nv_env $nvidia_smi_cmd --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null" 2>/dev/null || true)
            fi

            # Parse the GPU list
            if [ -n "$gpu_list" ]; then
                gpu_count=$(echo "$gpu_list" | wc -l | tr -d ' ')
                # Use first GPU for display defaults
                gpu_name=$(echo "$gpu_list" | head -1 | cut -d',' -f2 | sed 's/^ *//')
                gpu_vram=$(echo "$gpu_list" | head -1 | cut -d',' -f3 | sed 's/^ *//')
            fi

            # Attempt 3: Parse human-readable nvidia-smi table output (older drivers)
            if [ -z "$gpu_name" ]; then
                local smi_output=""
                smi_output=$(env $nv_env $nvidia_smi_cmd 2>/dev/null || true)
                if [ -z "$smi_output" ] && [ -n "${SUDO_USER:-}" ] && $is_wsl; then
                    smi_output=$(su - "$SUDO_USER" -c "env $nv_env $nvidia_smi_cmd 2>/dev/null" 2>/dev/null || true)
                fi
                if [ -n "$smi_output" ]; then
                    gpu_name=$(echo "$smi_output" | grep -oE 'NVIDIA [A-Za-z0-9 ]+' | head -1 || true)
                    gpu_vram=$(echo "$smi_output" | grep -oE '[0-9]+MiB\s*/\s*[0-9]+MiB' | head -1 | sed 's|.*/ *||' || true)
                    gpu_count=1
                fi
            fi
        fi

        # Method 3: /proc/driver/nvidia/version — kernel driver is loaded even without nvidia-smi
        if [ -z "$gpu_name" ] && [ -f /proc/driver/nvidia/version ]; then
            info "NVIDIA kernel driver detected via /proc/driver/nvidia/version"
            local nv_driver_ver
            nv_driver_ver=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
            gpu_name="NVIDIA GPU (driver ${nv_driver_ver:-unknown})"
            gpu_vram="unknown"
        fi

        # Method 4: lspci hardware scan (native Linux — not available on WSL2)
        if [ -z "$gpu_name" ] && ! $is_wsl && command -v lspci &>/dev/null; then
            local lspci_nvidia
            lspci_nvidia=$(lspci 2>/dev/null | grep -i 'nvidia' | grep -iE 'vga|3d|display' | head -1 || true)
            if [ -n "$lspci_nvidia" ]; then
                gpu_name=$(echo "$lspci_nvidia" | sed 's/.*: //' || true)
                gpu_vram="unknown (install nvidia-smi for details)"
                info "NVIDIA GPU found via lspci (nvidia-smi not available)"
            fi
        fi

        # Method 5: Check for /dev/nvidia* device nodes (GPU exists even if tools are missing)
        if [ -z "$gpu_name" ] && [ -e /dev/nvidia0 ]; then
            gpu_name="NVIDIA GPU (device /dev/nvidia0 — install nvidia-smi for details)"
            gpu_vram="unknown"
            info "NVIDIA GPU device node found at /dev/nvidia0"
        fi

        if [ -n "$gpu_name" ]; then
            gpu_vendor="nvidia"
        fi
    fi

    # ── AMD ROCm GPU Detection ───────────────────────────────────────────────
    if [ -z "$gpu_vendor" ] && [ "$OS" != "macos" ]; then
        local amd_gpu_name=""
        local amd_gpu_list=""

        # Method 1: rocm-smi — enumerate all AMD GPUs
        if command -v rocm-smi &>/dev/null; then
            # Try CSV output for all GPUs
            amd_gpu_list=$(rocm-smi --showproductname 2>/dev/null | grep -i 'card' | sed 's/.*: *//' || true)
            if [ -n "$amd_gpu_list" ]; then
                amd_gpu_name=$(echo "$amd_gpu_list" | head -1)
                gpu_count=$(echo "$amd_gpu_list" | wc -l | tr -d ' ')
            else
                amd_gpu_name=$(rocm-smi --showproductname 2>/dev/null | grep -i 'card' | head -1 | sed 's/.*: *//' || true)
            fi
        fi

        # Method 2: rocminfo
        if [ -z "$amd_gpu_name" ] && command -v rocminfo &>/dev/null; then
            amd_gpu_name=$(rocminfo 2>/dev/null | grep -A2 'Marketing Name' | grep -v 'Marketing' | head -1 | sed 's/^ *//' || true)
            if [ -z "$amd_gpu_name" ]; then
                amd_gpu_name=$(rocminfo 2>/dev/null | grep 'Name:' | grep -vi 'cpu' | head -1 | sed 's/.*Name: *//' || true)
            fi
        fi

        # Method 3: /dev/kfd exists (ROCm kernel driver loaded)
        if [ -z "$amd_gpu_name" ] && [ -c /dev/kfd ]; then
            amd_gpu_name="AMD GPU (/dev/kfd present)"
            # Try lspci for a better name
            if command -v lspci &>/dev/null; then
                local lspci_amd
                lspci_amd=$(lspci 2>/dev/null | grep -i 'amd\|radeon' | grep -iE 'vga|3d|display' | head -1 || true)
                if [ -n "$lspci_amd" ]; then
                    amd_gpu_name=$(echo "$lspci_amd" | sed 's/.*: //')
                fi
            fi
        fi

        # Method 4: lspci fallback for AMD discrete GPUs (multi-GPU aware)
        if [ -z "$amd_gpu_name" ] && ! $is_wsl && command -v lspci &>/dev/null; then
            local lspci_amd_all
            lspci_amd_all=$(lspci 2>/dev/null | grep -iE 'radeon|amd.*navi|amd.*vega|amd.*rx' | grep -iE 'vga|3d|display' || true)
            if [ -n "$lspci_amd_all" ]; then
                amd_gpu_name=$(echo "$lspci_amd_all" | head -1 | sed 's/.*: //')
                local _amd_count
                _amd_count=$(echo "$lspci_amd_all" | wc -l | tr -d ' ')
                if [ "$_amd_count" -gt "$gpu_count" ] 2>/dev/null; then
                    gpu_count="$_amd_count"
                fi
            fi
        fi

        if [ -n "$amd_gpu_name" ]; then
            gpu_vendor="amd"
            gpu_name="$amd_gpu_name"
            gpu_vram="unknown (use rocm-smi for details)"
            if [ "$gpu_count" -lt 1 ] 2>/dev/null; then gpu_count=1; fi
        fi
    fi

    # ── Intel iGPU Detection (informational) ─────────────────────────────────
    # Intel iGPUs are NOT supported by Ollama (no SYCL backend) or CUDA/ROCm.
    # We detect them only to inform the user so they don't wonder why it's not used.
    if [ "$OS" != "macos" ]; then
        local intel_gpu_name=""
        if ! $is_wsl && command -v lspci &>/dev/null; then
            intel_gpu_name=$(lspci 2>/dev/null | grep -iE 'intel.*vga|intel.*display|intel.*graphics' | head -1 | sed 's/.*: //' || true)
        fi
        # WSL2: check /sys vendor files
        if [ -z "$intel_gpu_name" ]; then
            local drm_card
            for drm_card in /sys/class/drm/card*/device/vendor; do
                if [ -f "$drm_card" ] && grep -q '0x8086' "$drm_card" 2>/dev/null; then
                    intel_gpu_name="Intel Integrated Graphics"
                    break
                fi
            done
        fi
        if [ -n "$intel_gpu_name" ]; then
            info "Intel GPU detected: $intel_gpu_name"
            info "Note: Intel iGPUs are not supported by Ollama or CUDA/ROCm."
            info "      Ollama will use your NVIDIA/AMD dGPU if available, otherwise CPU."
        fi
    fi

    # ── GPU Mode Selection (interactive prompt) ──────────────────────────────
    if [ "$gpu_vendor" = "nvidia" ]; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
        if [ "$gpu_count" -gt 1 ] 2>/dev/null; then
            echo -e "${GREEN}║  ${gpu_count} NVIDIA GPUs DETECTED                              ║${NC}"
        else
            echo -e "${GREEN}║  NVIDIA dGPU DETECTED                               ║${NC}"
        fi
        # List all GPUs if we have structured data
        if [ -n "$gpu_list" ] && [ "$gpu_count" -gt 1 ] 2>/dev/null; then
            echo "$gpu_list" | while IFS=',' read -r _idx _gname _gvram; do
                _idx=$(echo "$_idx" | sed 's/^ *//')
                _gname=$(echo "$_gname" | sed 's/^ *//')
                _gvram=$(echo "$_gvram" | sed 's/^ *//')
                echo -e "${GREEN}║  [${_idx}] ${_gname} — ${_gvram}${NC}"
            done
        else
            echo -e "${GREEN}║  ${gpu_name}${NC}"
            if [ "$gpu_vram" != "unknown" ] && [ -n "$gpu_vram" ]; then
                echo -e "${GREEN}║  VRAM: ${gpu_vram}${NC}"
            fi
        fi
        if $is_wsl; then
            echo -e "${GREEN}║  Runtime: WSL2 (GPU passthrough)                    ║${NC}"
        fi
        echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "  GPU mode:  Ollama uses your NVIDIA GPU for inference + embeddings"
        echo "             Learner uses CUDA for local model training/inference"
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

                # ── Multi-GPU selection (NVIDIA) ────────────────────────
                if [ -n "$gpu_list" ] && [ "$gpu_count" -gt 1 ] 2>/dev/null; then
                    echo ""
                    info "Which GPU(s) should Ollama and the Learner use?"
                    echo ""
                    echo "$gpu_list" | while IFS=',' read -r _idx _gname _gvram; do
                        _idx=$(echo "$_idx" | tr -d ' ')
                        _gname=$(echo "$_gname" | sed 's/^ *//')
                        _gvram=$(echo "$_gvram" | sed 's/^ *//')
                        echo "  [$_idx] $_gname — $_gvram"
                    done
                    echo "  [all] Use all $gpu_count GPUs"
                    echo ""
                    read -rp "GPU selection (0, 1, 0,1, or all) [all]: " gpu_select
                    gpu_select=$(echo "${gpu_select:-all}" | tr -d ' ')
                    if [ "$gpu_select" = "all" ]; then
                        success "Using all $gpu_count NVIDIA GPUs"
                    else
                        success "Using GPU(s): $gpu_select"
                    fi
                fi

                # Check nvidia-container-toolkit / Docker GPU support
                # Detects Docker install method (snap, systemd, Desktop, service)
                # and gives correct nvidia-ctk path + restart command.
                local _dmethod
                _dmethod="$(_docker_method)"

                # ── WSL2 library path fix ─────────────────────────────────
                # On WSL2, NVIDIA libraries live at /usr/lib/wsl/lib/ but
                # nvidia-container-cli won't find them unless ldconfig knows
                # AND the libs are in a path accessible from all contexts
                # (snap Docker's prestart hooks can't reach /usr/lib/wsl/lib/).
                # ALWAYS run ldconfig (don't check ldconfig -p — it can be stale)
                if $is_wsl && [ -d /usr/lib/wsl/lib ]; then
                    info "Registering WSL2 NVIDIA libraries with ldconfig..."
                    echo "/usr/lib/wsl/lib" | sudo tee /etc/ld.so.conf.d/nvidia-wsl.conf >/dev/null 2>&1

                    # Copy WSL NVIDIA libs into standard system path so they
                    # are discoverable from any execution context (including snap).
                    # Symlinks don't work — snap can't follow links to /usr/lib/wsl/lib/.
                    for _lib in /usr/lib/wsl/lib/lib*.so*; do
                        [ -f "$_lib" ] || continue
                        _bn="$(basename "$_lib")"
                        _dst="/usr/lib/x86_64-linux-gnu/$_bn"
                        if [ -L "$_dst" ]; then
                            sudo cp -f "$_lib" "$_dst" 2>/dev/null || true
                        elif [ ! -e "$_dst" ]; then
                            sudo cp -f "$_lib" "$_dst" 2>/dev/null || true
                        elif [ "$_lib" -nt "$_dst" ]; then
                            sudo cp -f "$_lib" "$_dst" 2>/dev/null || true
                        fi
                    done

                    sudo ldconfig 2>/dev/null
                    if ldconfig -p 2>/dev/null | grep -q libnvidia-ml; then
                        success "WSL2 NVIDIA libraries registered"
                    else
                        warn "Could not register WSL2 NVIDIA libs — GPU containers may fail"
                        info "  Manual fix: echo '/usr/lib/wsl/lib' | sudo tee /etc/ld.so.conf.d/nvidia-wsl.conf && sudo ldconfig"
                    fi
                    # Fix nvidia-container-toolkit config.toml for WSL2
                    # (no-cgroups=true + ldconfig="@/sbin/ldconfig")
                    local _wsl_setup_needs_restart=false
                    _ensure_wsl2_nvidia_config || _wsl_setup_needs_restart=true
                    # Snap Docker: also restart if libs were copied (snap caches fs state)
                    if [ "$_dmethod" = "snap" ]; then
                        _wsl_setup_needs_restart=true
                    fi
                    if $_wsl_setup_needs_restart; then
                        info "Restarting Docker daemon to apply WSL2 NVIDIA fixes..."
                        case "$_dmethod" in
                            snap)    sudo snap restart docker 2>/dev/null ;;
                            systemd) sudo systemctl restart docker 2>/dev/null ;;
                            service) sudo service docker restart 2>/dev/null ;;
                            desktop) info "Please restart Docker Desktop from the system tray" ;;
                        esac
                        sleep 3
                    fi
                fi

                if ! docker info 2>/dev/null | grep -qi nvidia; then
                    if [ "$_dmethod" = "desktop" ]; then
                        # Docker Desktop detected — it has built-in GPU support
                        warn "Docker GPU runtime not yet active"
                        echo ""
                        info "You're using Docker Desktop, which has built-in NVIDIA GPU support."
                        info "To enable it:"
                        info "  1. Open Docker Desktop → Settings → Resources → WSL Integration"
                        info "     (ensure your WSL distro is enabled)"
                        info "  2. Settings → Docker Engine → ensure the config includes:"
                        info '     "runtimes": { "nvidia": { "path": "nvidia-container-runtime" } }'
                        info "  3. Restart Docker Desktop from the system tray (right-click → Restart)"
                        info ""
                        info "If GPU still isn't detected, install nvidia-container-toolkit inside WSL:"
                        info "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
                        info "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
                        info "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
                        _nvidia_ctk_hint
                        info "  Then restart Docker Desktop from the system tray"
                    else
                        # Native Docker (snap, systemd, or service) — needs nvidia-container-toolkit
                        warn "nvidia-container-toolkit not detected in Docker"
                        echo ""
                        if $is_wsl; then
                            info "To enable GPU in Docker on WSL2:"
                            info "  1. Ensure NVIDIA drivers are installed on Windows (not inside WSL)"
                            info "  2. Install nvidia-container-toolkit inside WSL:"
                        else
                            info "Install nvidia-container-toolkit:"
                        fi
                        info "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
                        info "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
                        info "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
                        _nvidia_ctk_hint
                        _docker_restart_hint
                        if $is_wsl; then
                            echo ""
                            info "If you get 'libnvidia-ml.so.1: cannot open shared object file':"
                            info "  echo '/usr/lib/wsl/lib' | sudo tee /etc/ld.so.conf.d/nvidia-wsl.conf && sudo ldconfig"
                        fi
                    fi
                    echo ""
                    read -rp "Continue with GPU mode anyway? [Y/n]: " cont
                    case "$(to_lower "${cont:-y}")" in
                        n|no) gpu_mode="cpu"; warn "Falling back to CPU mode" ;;
                    esac
                fi
                ;;
        esac
    elif [ "$gpu_vendor" = "amd" ]; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
        if [ "$gpu_count" -gt 1 ] 2>/dev/null; then
            echo -e "${GREEN}║  ${gpu_count} AMD GPUs DETECTED (ROCm)                          ║${NC}"
        else
            echo -e "${GREEN}║  AMD GPU DETECTED (ROCm)                            ║${NC}"
        fi
        echo -e "${GREEN}║  ${gpu_name}${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "  GPU mode:  Ollama can use AMD GPU via ROCm"
        echo "             Learner uses ROCm PyTorch for local model inference"
        echo "             Requires: ROCm drivers + amdgpu-dkms"
        echo ""
        echo "  CPU mode:  Everything runs on CPU"
        echo ""
        read -rp "Use GPU acceleration? [Y/n]: " gpu_choice
        case "$(to_lower "${gpu_choice:-y}")" in
            n|no)
                gpu_mode="cpu"
                info "Selected: CPU mode"
                ;;
            *)
                gpu_mode="gpu"
                success "Selected: GPU mode — AMD ROCm ($gpu_name)"
                info "Note: Learner will be built with ROCm PyTorch (rocm6.0)"

                # ── Multi-GPU selection (AMD) ────────────────────────────
                if [ "$gpu_count" -gt 1 ] 2>/dev/null; then
                    echo ""
                    info "Which GPU(s) should Ollama and the Learner use?"
                    echo ""
                    local _amd_i=0
                    echo "$amd_gpu_list" | while read -r _agname; do
                        echo "  [$_amd_i] $_agname"
                        _amd_i=$((_amd_i + 1))
                    done
                    echo "  [all] Use all $gpu_count GPUs"
                    echo ""
                    read -rp "GPU selection (0, 1, 0,1, or all) [all]: " gpu_select
                    gpu_select=$(echo "${gpu_select:-all}" | tr -d ' ')
                    if [ "$gpu_select" = "all" ]; then
                        success "Using all $gpu_count AMD GPUs"
                    else
                        success "Using GPU(s): $gpu_select"
                    fi
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
        echo ""
        info "No dedicated GPU detected — using CPU mode"
        if $is_wsl; then
            warn "WSL2 GPU troubleshooting:"
            info "  1. Install NVIDIA Game Ready / Studio drivers on Windows (NOT inside WSL)"
            info "  2. Restart WSL: wsl --shutdown (from PowerShell), then reopen"
            info "  3. Verify: nvidia-smi (should show your GPU)"
            info "  4. If using sudo: try running without sudo, or use: sudo -E ./skynet.sh setup"
            info "  5. NVIDIA driver path on WSL2: /usr/lib/wsl/lib/nvidia-smi"
        elif [ "$OS" = "linux" ]; then
            info "If you have an NVIDIA GPU: install nvidia-driver-XXX (or nvidia-utils)"
            info "If you have an AMD GPU: install ROCm (rocm-smi, amdgpu-dkms)"
        fi
        gpu_mode="cpu"
    fi

    # ── Ollama configuration ─────────────────────────────────────────────────
    # Goal: OLLAMA_HOST=0.0.0.0 (Docker access), KEEP_ALIVE=-1 (never unload),
    #        NUM_PARALLEL=4, and optionally NUM_GPU=0 for CPU mode.
    _configure_ollama "$gpu_mode" "$gpu_vendor" "$gpu_select"

    # ── Write GPU/CPU choice to .env ─────────────────────────────────────────
    local torch_url
    if [ "$gpu_mode" = "gpu" ]; then
        if [ "$gpu_vendor" = "amd" ]; then
            torch_url="https://download.pytorch.org/whl/rocm6.0"
        else
            torch_url="https://download.pytorch.org/whl/cu121"
        fi
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

# Model selection (defaults are fine — override only if needed)
OPENROUTER_MAIN_MODEL=
OPENROUTER_FAST_MODEL=
GENERIC_MAIN_MODEL=
GENERIC_FAST_MODEL=

# Token limits
MAX_TOKENS_MAIN=4096
MAX_TOKENS_FAST=1024

# Web search — model decides when to search (memory first, then web)
WEB_SEARCH_ENABLED=true

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

    # Persist GPU_MODE so subsequent start/restart/rebuild pick up the override
    if grep -q '^GPU_MODE=' .env 2>/dev/null; then
        sed_inplace "s|^GPU_MODE=.*|GPU_MODE=${gpu_mode}|" .env
    else
        echo "GPU_MODE=${gpu_mode}" >> .env
    fi
    success "Learner PyTorch: $gpu_mode ($torch_url)"

    # Write GPU selection to .env for Docker (NVIDIA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)
    if [ "$gpu_mode" = "gpu" ] && [ "$gpu_select" != "all" ] && [ -n "$gpu_select" ]; then
        if [ "$gpu_vendor" = "amd" ]; then
            if grep -q '^HIP_VISIBLE_DEVICES=' .env 2>/dev/null; then
                sed_inplace "s|^HIP_VISIBLE_DEVICES=.*|HIP_VISIBLE_DEVICES=$gpu_select|" .env
            else
                echo "HIP_VISIBLE_DEVICES=$gpu_select" >> .env
            fi
            info "AMD GPU selection: HIP_VISIBLE_DEVICES=$gpu_select"
        else
            if grep -q '^NVIDIA_VISIBLE_DEVICES=' .env 2>/dev/null; then
                sed_inplace "s|^NVIDIA_VISIBLE_DEVICES=.*|NVIDIA_VISIBLE_DEVICES=$gpu_select|" .env
            else
                echo "NVIDIA_VISIBLE_DEVICES=$gpu_select" >> .env
            fi
            if grep -q '^CUDA_VISIBLE_DEVICES=' .env 2>/dev/null; then
                sed_inplace "s|^CUDA_VISIBLE_DEVICES=.*|CUDA_VISIBLE_DEVICES=$gpu_select|" .env
            else
                echo "CUDA_VISIBLE_DEVICES=$gpu_select" >> .env
            fi
            info "NVIDIA GPU selection: CUDA_VISIBLE_DEVICES=$gpu_select"
        fi
    fi

    # Pull Ollama models — wait for Ollama to be ready first
    # (systemd restart above may take a moment)
    info "Waiting for Ollama to be ready..."
    local ollama_ready=false
    for i in $(seq 1 15); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            ollama_ready=true
            break
        fi
        sleep 2
    done
    if $ollama_ready; then
        success "Ollama is ready"
    else
        warn "Ollama not responding on port 11434 — trying to start it..."
        if command -v systemctl &>/dev/null; then
            sudo systemctl start ollama 2>/dev/null || true
            sleep 3
        fi
    fi

    info "Pulling Ollama embedding model..."
    if ollama pull nomic-embed-text; then
        success "nomic-embed-text ready"
    else
        warn "Could not pull nomic-embed-text — you can retry later with: ollama pull nomic-embed-text"
    fi

    info "Pulling Ollama local inference model..."
    if ollama pull phi4; then
        success "phi4 ready"
    else
        warn "Could not pull phi4 (optional) — retry with: ollama pull phi4"
    fi

    echo ""
    success "Setup complete! Edit .env with your API keys, then run: ./skynet.sh start"
    echo ""
    info "  Platform: $OS"
    info "  GPU mode: $gpu_mode"
    if [ -n "$gpu_name" ] && [ "$gpu_vendor" != "apple" ]; then
        info "  GPU:      $gpu_name${gpu_vram:+ ($gpu_vram)}"
        info "  Vendor:   $gpu_vendor"
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
    if [ "$gpu_vendor" = "amd" ] && [ "$gpu_mode" = "gpu" ]; then
        info "  PyTorch:  ROCm 6.0 ($torch_url)"
    fi
}

# ── Ollama Configuration (per-platform) ──────────────────────────────────────
_configure_ollama() {
    local gpu_mode="$1"
    local gpu_vendor="${2:-}"
    local gpu_select="${3:-all}"
    local ollama_gpu_env=""
    local ollama_visible_env=""
    if [ "$gpu_mode" = "cpu" ]; then
        ollama_gpu_env='OLLAMA_NUM_GPU=0'
    elif [ "$gpu_select" != "all" ] && [ -n "$gpu_select" ]; then
        # User selected specific GPU(s) — set CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES
        if [ "$gpu_vendor" = "amd" ]; then
            ollama_visible_env="HIP_VISIBLE_DEVICES=$gpu_select"
        else
            ollama_visible_env="CUDA_VISIBLE_DEVICES=$gpu_select"
        fi
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

                if $needs_update || [ -n "$ollama_gpu_env" ] || [ -n "$ollama_visible_env" ]; then
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
                        if [ -n "$ollama_visible_env" ]; then
                            echo "Environment=\"$ollama_visible_env\""
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
                if [ -n "$ollama_visible_env" ]; then
                    info "  export $ollama_visible_env"
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

    # Validate GPU runtime + re-evaluate COMPOSE before any compose operations
    _ensure_gpu_runtime
    COMPOSE="$(_compose_files)"

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
    _compose_up -d

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

    # Validate GPU runtime + re-evaluate COMPOSE before restarting
    _ensure_gpu_runtime
    COMPOSE="$(_compose_files)"

    info "Restarting services (preserving all data)..."
    $COMPOSE stop
    _compose_up -d
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

    # Validate GPU runtime + re-evaluate COMPOSE before any compose operations.
    # Without this, WSL2 ldconfig can be stale after wsl --shutdown, causing:
    #   "libnvidia-ml.so.1: cannot open shared object file: no such file or directory"
    _ensure_gpu_runtime
    COMPOSE="$(_compose_files)"

    info "Pulling latest code..."
    git pull --rebase || git pull

    info "Rebuilding changed services..."
    compose_build

    info "Restarting updated services..."
    _compose_up -d

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

    # Ensure GPU runtime is valid so COMPOSE points at the right files
    _ensure_gpu_runtime
    COMPOSE="$(_compose_files)"

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

    # Ensure GPU runtime is valid so COMPOSE points at the right files
    _ensure_gpu_runtime
    COMPOSE="$(_compose_files)"

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
