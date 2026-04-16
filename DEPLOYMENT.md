# Brain System — Deployment Guide

## What You're Running

24 containerized services + infrastructure. Total system at rest: ~6-8GB RAM minimum.
With learner model loaded: 12-16GB. With vision: +4GB.

```
SERVICE          PORT    RAM       CPU      PURPOSE
────────────────────────────────────────────────────────────────────
gateway          8000    100MB     minimal  Nginx routing
redis            6379    1GB       minimal  Working memory, knobs, metrics
elasticsearch    9200    2-4GB     medium   Neocortex semantic storage
edgedb           5656    512MB     minimal  Episodic memory, skills
thalamus         8000    256MB     high     Orchestrator
amygdala         8001    128MB     minimal  Urgency + emotion detection
hippocampus      8002    256MB     medium   Episode encoding
neocortex        8004    256MB     medium   Knowledge search
cerebellum       8003    256MB     medium   Skill matching
working-memory   8005    128MB     minimal  Session state
perception       8006    128MB     minimal  Experience detection (21 types)
verification     8007    256MB     medium   Fact checking
cross-domain     8008    256MB     medium   Analogy reasoning
compass          8009    256MB     medium   7-dimension ethical compass
framework-router 8010    256MB     high     Multi-agent frameworks
knobs            8090    128MB     minimal  Behavioral parameters (21 knobs)
self-update      8011    256MB     medium   Code modification via Supergrokie
causal-graph     8012    256MB     medium   Cause-effect graph
simulator        8013    128MB     minimal  Code execution sandbox
metacognition    8014    256MB     medium   Uncertainty calibration
goals            8015    256MB     medium   Long-horizon planning
learner          8016    4-8GB     high     Local model (Phi-4-mini)
vision           8017    2-4GB     medium   Camera + CLIP
dashboard        8080    128MB     minimal  Web UI — terminal, monitoring, logs
autonomy         8018    256MB     medium   Autonomous self-improvement loop
hypothalamus     8019    128MB     minimal  Hormonal system — 16 hormones, 7 glands
insight-engine   8020    128MB     minimal  Epiphany detection via graph analysis
embed-queue      8021    128MB     minimal  Centralized embedding — Redis Streams MQ + cache
────────────────────────────────────────────────────────────────────
TOTAL (no learner/vision):  ~6GB RAM
TOTAL (full system):        ~16GB RAM
```

---

## Quick Start

### Linux / WSL
```bash
git clone https://github.com/mytuo1/SkyNet.git && cd SkyNet
./skynet.sh setup    # installs deps, detects GPU, creates .env, pulls models
./skynet.sh start    # builds, init DBs, warms Ollama, starts all 24 services
./skynet.sh dashboard  # opens http://localhost:8080
```

### macOS
```bash
git clone https://github.com/mytuo1/SkyNet.git && cd SkyNet
./skynet.sh setup    # installs via Homebrew, detects Apple Silicon, creates .env
./skynet.sh start    # builds, init DBs, warms Ollama, starts all 24 services
./skynet.sh dashboard  # opens http://localhost:8080
```

### Windows (PowerShell)
```powershell
git clone https://github.com/mytuo1/SkyNet.git; cd SkyNet
.\skynet.ps1 setup    # checks Docker Desktop, Ollama, detects GPU, creates .env
.\skynet.ps1 start    # builds, init DBs, warms Ollama, starts all 24 services
.\skynet.ps1 dashboard # opens http://localhost:8080
```

Or use the batch launcher: `skynet.bat start`

### Available commands

All commands work identically across platforms (`./skynet.sh` on Linux/macOS/WSL, `.\skynet.ps1` on Windows):

| Command | Description |
|---------|-------------|
| `setup` | First-time wizard: Docker, Ollama, **GPU/CPU selection**, .env, model pull |
| `start` | Full startup: build, init DBs, warm Ollama, start all 24 services |
| `stop` | Graceful shutdown (data preserved) |
| `restart` | Quick restart — no rebuild, keeps all data and conversations |
| `update` | Git pull + rebuild changed services + restart |
| `status` | Show all service health and brain diagnostics |
| `logs [svc]` | Tail logs (all or specific). Try: `logs embed-queue` |
| `dashboard` | Open the web dashboard |
| `think` | Activate autonomous thinking |
| `rebuild` | **DESTRUCTIVE** — wipe databases, rebuild all containers fresh |
| `reset` | Factory reset (alias for rebuild) |

### Common scenarios

| Scenario | Linux/macOS/WSL | Windows |
|----------|----------------|---------|
| **First install** | `./skynet.sh setup && ./skynet.sh start` | `.\skynet.ps1 setup; .\skynet.ps1 start` |
| **After code changes** | `./skynet.sh update` | `.\skynet.ps1 update` |
| **Quick restart** (no rebuild) | `./skynet.sh restart` | `.\skynet.ps1 restart` |
| **Config change** (.env edit) | `./skynet.sh restart` | `.\skynet.ps1 restart` |
| **After system reboot** | `./skynet.sh start` | `.\skynet.ps1 start` |
| **Wipe and reinstall** | `./skynet.sh rebuild` | `.\skynet.ps1 rebuild` |
| **Switch GPU ↔ CPU** | `./skynet.sh setup` then `rebuild` | `.\skynet.ps1 setup` then `rebuild` |
| **Check embed queue** | `./skynet.sh logs embed-queue` | `.\skynet.ps1 logs embed-queue` |

---

## Platform Notes

### Linux (native)
- `./skynet.sh setup` auto-installs Docker and Ollama via system package managers
- Ollama configured via systemd drop-in override (automatic)
- GPU: `nvidia-smi` detection + nvidia-container-toolkit for Docker GPU

### macOS
- Requires **Docker Desktop** (not native Docker) — install via `brew install --cask docker`
- Ollama: `brew install ollama` or download from ollama.com
- Apple Silicon: Metal GPU acceleration is **automatic** (Ollama uses it without configuration)
- Intel Macs: CPU-only for Ollama (no CUDA)
- `host.docker.internal` works out of the box via Docker Desktop VM
- Ollama configured via `launchctl setenv` + shell profile

### Windows (native PowerShell)
- Requires **Docker Desktop for Windows** — download from docker.com
- Ollama: download from ollama.com (runs in system tray)
- GPU: NVIDIA GPU auto-detected via `nvidia-smi`; Ollama uses it automatically
- Ollama configured via Windows environment variables (`[Environment]::SetEnvironmentVariable`)
- Use `skynet.bat` for cmd.exe or `.\skynet.ps1` for PowerShell

### WSL2 (Windows Subsystem for Linux)
- Same as Linux — use `./skynet.sh`
- GPU passthrough: install NVIDIA drivers on **Windows** (not inside WSL)
- `nvidia-smi` should work inside WSL2 if Windows drivers are installed
- The script checks `/usr/lib/wsl/lib/nvidia-smi` automatically (the WSL2 GPU driver path)
- If running with `sudo`, the script handles PATH differences for nvidia-smi detection
- Docker: use Docker Desktop with WSL2 backend, or install Docker inside WSL2
- **Troubleshooting GPU not detected in WSL2**:
  1. Ensure NVIDIA Game Ready / Studio drivers are installed on **Windows** (not inside WSL)
  2. Run `nvidia-smi` (without sudo) — it should show your GPU
  3. If it works without sudo but not with sudo: the script now checks the WSL path automatically
  4. If it doesn't work at all: update your Windows NVIDIA drivers and restart WSL (`wsl --shutdown`)

---

## Prerequisites

### Supported platforms:
- **Linux** (Ubuntu 22.04+, Debian, Fedora — tested)
- **macOS** (Intel or Apple Silicon — requires Docker Desktop)
- **Windows** (10/11 — requires Docker Desktop)
- **WSL2** (recommended for Windows users who prefer bash)

### Required on every deployment target:
- Docker 24+ and Docker Compose 2.20+
- Ollama running on host (for embeddings)
- At least one LLM API key (see supported providers below)

### Supported LLM Providers (set LLM_PROVIDER in .env):
```
anthropic   — Claude (needs ANTHROPIC_API_KEY)
openai      — GPT-5.4-mini/nano (needs OPENAI_API_KEY)
gemini      — Google Gemini (needs GEMINI_API_KEY)
ollama      — Local models via Ollama (no key needed)
deepseek    — DeepSeek R1 (needs DEEPSEEK_API_KEY)
groq        — Groq (needs GROQ_API_KEY)
xai         — xAI Grok 4.20 (needs XAI_API_KEY)
openrouter  — OpenRouter (needs OPENROUTER_API_KEY)
generic     — Any OpenAI-compatible endpoint (needs GENERIC_API_KEY + GENERIC_BASE_URL)
```

### Install Ollama and pull models:
```bash
# Mac/Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text   # 274MB — embeddings (required)
ollama pull phi4                # 9GB — local inference (optional)
```

---

## GPU / CPU Configuration

`./skynet.sh setup` auto-detects your GPU and asks interactively:

```
╔══════════════════════════════════════════════════════╗
║  dGPU DETECTED                                      ║
║  NVIDIA GeForce RTX 3080 Ti Laptop GPU              ║
║  VRAM: 16384 MiB                                    ║
║  Runtime: WSL2 (GPU passthrough)                    ║
╚══════════════════════════════════════════════════════╝

Use GPU acceleration? [Y/n]:
```

### What GPU mode changes

| Component | GPU mode (NVIDIA) | GPU mode (AMD) | CPU mode |
|-----------|----------|---------|----------|
| **Ollama** | Uses NVIDIA GPU (fast) | Uses ROCm (fast) | `OLLAMA_NUM_GPU=0` forces CPU |
| **Learner** (Phi-4-mini) | CUDA 12.1, float16, ~4GB VRAM | ROCm 6.0, float16 | CPU, float32, ~8GB RAM |
| **Docker** | Needs `nvidia-container-toolkit` | Needs ROCm drivers | No extra drivers |
| **TORCH_INDEX_URL** | `whl/cu121` | `whl/rocm6.0` | `whl/cpu` |

### GPU Auto-Detection Methods

The setup script uses multiple detection methods in order of priority:

**NVIDIA GPU detection (`skynet.sh`):**
1. `nvidia-smi` via system PATH
2. `/usr/lib/wsl/lib/nvidia-smi` — WSL2 driver shim (not in sudo PATH)
3. `/usr/local/cuda/bin/nvidia-smi` — CUDA toolkit install
4. `/usr/bin/nvidia-smi`, `/usr/local/bin/nvidia-smi` — standard FHS paths
5. If running under `sudo` on WSL2: drops to real user (`$SUDO_USER`) since WSL interop may fail under root
6. `/proc/driver/nvidia/version` — kernel driver loaded (detects GPU even without nvidia-smi)
7. `lspci` hardware scan (native Linux only)
8. `/dev/nvidia0` device node (GPU exists but tools missing)

**AMD GPU detection (`skynet.sh`):**
1. `rocm-smi --showproductname`
2. `rocminfo` agent name
3. `/dev/kfd` device exists (ROCm kernel driver)
4. `lspci` hardware scan for Radeon/AMD discrete GPUs

**Windows GPU detection (`skynet.ps1`):**
1. `nvidia-smi` CLI
2. `Win32_VideoController` WMI query (detects NVIDIA and AMD without CLI tools)

**macOS GPU detection:**
- Apple Silicon: `sysctl -n machdep.cpu.brand_string` (Metal auto-accelerated)
- Intel Macs: CPU only (no CUDA/ROCm)

### WSL2 Setup (Windows + GPU)

WSL2 passes through the host GPU to Linux containers. Requirements:

1. **Windows**: Install NVIDIA GPU drivers (Game Ready or Studio) — NOT inside WSL
2. **WSL2**: GPU is automatically visible (`nvidia-smi` should work)
3. **Docker GPU**: Install nvidia-container-toolkit inside WSL:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. **Verify**: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Native Linux Setup (GPU)

Same as WSL2 step 3 above. The `./skynet.sh setup` wizard will guide you.

### Switching between GPU and CPU

```bash
./skynet.sh setup    # re-run — select different mode
./skynet.sh rebuild  # rebuild containers with new PyTorch (CUDA vs CPU)
```

---

## Running Red-Hot — Zero Cold Starts

The system is designed so all infrastructure is always hot and ready:

### Ollama (host-side, never restarts, never unloads)

- Runs as a systemd service on the host (not in Docker)
- `OLLAMA_KEEP_ALIVE=-1` — models stay loaded in VRAM/RAM permanently
- `OLLAMA_NUM_PARALLEL=4` — handles 4 concurrent embed requests
- `restart: always` via systemd — survives reboots
- `./skynet.sh start` warms up the embed model before starting services
- Every API call includes `keep_alive: -1` to prevent unloading

### Databases (always available)

All database containers use `restart: unless-stopped`:
- **Redis** — sub-ms latency, no cold start possible
- **Elasticsearch** — JVM warm-up takes ~15s on first boot, then stays hot
- **EdgeDB** — connection pool stays warm via health checks every 15s

### Embed Queue (Redis Streams MQ)

The `embed-queue` service (port 8021) centralizes all embedding calls:
- **Cache**: Redis cache with 6h TTL — repeated texts return in <1ms
- **Dedup**: Concurrent requests for identical text share one Ollama call
- **Async queue**: Write-path embeds (store/encode) can be fire-and-forget via Redis Streams
- **Background workers**: 2 stream consumers drain the queue continuously
- **Stale job recovery**: XCLAIM reclaims stuck jobs after 30s
- The queue drains fast enough that by the time the user reads a response (~5-30s), all write-path embeddings are indexed and searchable for the next interaction

### Monitoring embed performance

```bash
# Docker logs show every Ollama call with timing and source service
./skynet.sh logs embed-queue

# Example output:
# [embed-queue] SYNC src=neocortex cache=HIT 2ms dims=768 "What is machine learning?"
# [embed-queue] SYNC src=hippocampus cache=MISS ollama=145ms dims=768 "User asked about ML"
# [embed-queue] OLLAMA src=async:neocortex OK 142ms dims=768 "New knowledge to store"
# [embed-queue] ASYNC src=neocortex job=a3f2b1 OK 148ms cache=MISS dims=768 callback=200 12ms

# Stats API
curl http://localhost:8000/brain/embed-queue/stats
# Returns: cache_hit_rate, avg_ollama_ms, queue_depth, async completed/failed counts
```

---

## Step 1: Configure Environment

```bash
cp .env.example .env
nano .env   # or your editor of choice
```

**Minimum required fields:**
```
ANTHROPIC_API_KEY=sk-ant-...   # OR
OPENAI_API_KEY=sk-...          # set LLM_PROVIDER=openai if using OpenAI
LLM_PROVIDER=anthropic
SUPERGROKIE_WEBHOOK_SECRET=<32+ random chars>
```

**To switch to OpenAI (any time, no code changes):**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```
That's all. Every service reads from llm_provider.py which handles the switch.

---

## Step 2: First-Time Setup

```bash
# Build all images (takes 10-20 minutes first time)
docker compose build

# Start databases first
docker compose up -d redis elasticsearch edgedb

# Wait for databases to be healthy
docker compose ps

# Run database initialization (creates indices, schema, seeds data)
docker compose up db-init
# Wait for: "All databases initialized"

# Start all remaining services
docker compose up -d

# Watch startup (thalamus starts last, takes ~2 minutes)
docker compose logs -f thalamus
```

---

## Step 3: Verify Everything Is Running

```bash
# Full brain health check
curl http://localhost:8000/health | python3 -m json.tool

# Expected: all services show "alive"
# Acceptable degraded: learner may show "model_not_loaded" on first start
#   (model downloads in background, takes 10-30 min)

# Check learner model status
curl http://localhost:8000/brain/learner/status | python3 -m json.tool

# Check knobs
curl http://localhost:8090/current

# Test the brain
curl -X POST http://localhost:8000/brain/process \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_001", "user_input": "Hello, are you working?"}'
```

---

## Step 4: Using the System

### Talk to the brain:
```bash
curl -X POST http://localhost:8000/brain/process \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my_session",
    "user_input": "Explain the causal relationship between sleep and memory consolidation",
    "include_vision": false
  }'
```

### Adjust personality knobs:
```bash
# Make it more empathetic
curl -X PATCH "http://localhost:8090/adjust?knob_name=empathy&value=0.95"

# Apply a preset personality
curl -X POST http://localhost:8090/preset/researcher

# List available presets
curl http://localhost:8090/presets
```

### Create a persistent goal:
```bash
curl -X POST http://localhost:8000/brain/goals/goals \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Build a better understanding of distributed systems",
    "description": "Learn the core concepts, trade-offs, and patterns in distributed computing",
    "priority": 0.8,
    "horizon": "week",
    "domain": "technology",
    "auto_plan": true
  }'
```

### Enable vision (if camera present):
```bash
# Edit .env: VISION_ENABLED=true
docker compose restart vision

# Capture a frame
curl -X POST http://localhost:8000/brain/vision/capture?session_id=my_session

# Search visual memories
curl -X POST http://localhost:8000/brain/vision/search \
  -H "Content-Type: application/json" \
  -d '{"query": "whiteboard with diagrams", "top_k": 5}'
```

### Connect Supergrokie:
```bash
# In Supergrokie, set webhook URL to:
http://your-host:8000/update/webhook/supergrokie

# Supergrokie secret must match SUPERGROKIE_WEBHOOK_SECRET in .env
# Supergrokie can then say:
# "Make the brain warmer and more empathetic"
# "Create a service that monitors the weather API"
# "The verification service is too slow, optimize it"
```

---

## Local Model — What You Need to Know

**The local model (Phi-4-mini) is NOT Claude.**
It handles:
- Routine classification tasks
- Domain-specific pattern matching learned from your interactions
- Emotion detection (after fine-tuning)
- Low-complexity queries that would waste Claude API calls

It does NOT handle:
- Complex multi-step reasoning (Claude does this)
- Cross-domain synthesis
- Novel problem solving

**Resource requirements for Phi-4-mini:**
```
RAM:  3.8B params × 4 bytes (float32) = ~15GB  ← float32
      3.8B params × 0.5 bytes (4-bit)  = ~2GB   ← quantized (recommended)
CPU:  Any modern x86 or ARM
Time: ~2-5 tokens/second on CPU (fast enough for background tasks)
```

**First download (happens automatically on first start):**
```bash
# Monitor download progress
docker compose logs -f learner
# "Downloading from HuggingFace: microsoft/phi-4-mini-instruct"
# This takes 10-30 minutes depending on connection
```

**Requirements:** The learner requires `transformers>=4.46.3` for Phi-4-mini-instruct
support (SlidingWindowCache). If you see import errors, rebuild the learner:
```bash
docker compose build --no-cache learner
```

**To use a smaller/faster model instead:**
```bash
# In .env:
LOCAL_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct   # 1.5B params, ~800MB quantized
# Or:
LOCAL_MODEL_NAME=microsoft/phi-2               # 2.7B params, fast on CPU
```

---

## Resource Requirements by Deployment Tier

### Tier 1: Minimal (development, testing)
```
RAM:    16GB
CPU:    4 cores
Disk:   50GB
Cost:   A laptop you already own

Limits: Learner model disabled, Vision disabled
Config: Comment out learner and vision in docker-compose.yml
```

### Tier 2: Standard (personal assistant, small team)
```
RAM:    32GB
CPU:    8 cores
Disk:   100GB SSD
Cost:   $800-1200 (mini PC or used workstation)
        OR ~$0.15-0.40/hour cloud

Enables: Full system including learner (Phi-4-mini quantized)
Vision:  Optional (needs USB camera)
```

### Tier 3: Production (team, business)
```
RAM:    64GB+
CPU:    16+ cores
Disk:   500GB NVMe
Cost:   $2000-5000 (server) OR $0.50-1.50/hour cloud

Enables: Full system + large local model (Qwen3-14B)
Vision:  Multiple cameras supported
Scale:   Multiple concurrent sessions
```

---

## Cloud Deployment Options

### Option A: AWS EC2 (Best for production)

**Recommended instances:**
```
Tier 1 (no local model):  t3.xlarge    ($0.17/hr)  16GB RAM, 4 vCPU
Tier 2 (Phi-4-mini):      m6i.2xlarge  ($0.38/hr)  32GB RAM, 8 vCPU
Tier 3 (Qwen3-14B):       m6i.4xlarge  ($0.77/hr)  64GB RAM, 16 vCPU
```

**Deploy to EC2:**
```bash
# 1. Launch EC2 instance (Amazon Linux 2023 or Ubuntu 22.04)
# 2. Install Docker
sudo dnf install docker -y && sudo systemctl start docker
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text

# 4. Clone/upload your brain directory
scp -r ./brain ec2-user@your-instance:/home/ec2-user/

# 5. Configure .env and deploy
cd brain && cp .env.example .env && nano .env
docker-compose up -d

# 6. (Optional) Use EBS for persistent storage
# Mount /dev/xvdb to /var/lib/docker for database persistence
```

**Use Spot instances** to cut costs by 60-70%:
```bash
# Spot instances work well for the stateless services (amygdala, perception, etc.)
# Use On-Demand for databases (redis, elasticsearch, edgedb)
# Or use managed services: ElastiCache (Redis), OpenSearch (ES)
```

**AWS managed databases (recommended for production):**
```yaml
# Replace self-hosted with:
# redis      → AWS ElastiCache Redis
# elasticsearch → Amazon OpenSearch Service
# edgedb     → Keep self-hosted (no managed equivalent) or use PostgreSQL
```

### Option B: DigitalOcean Droplets (Easiest setup)

```bash
# Create a Droplet: Memory-Optimized m-16vcpu-128gb ($0.893/hr)
# Or regular: s-8vcpu-16gb ($0.119/hr) for minimal system

# DigitalOcean one-click Docker setup available
# Then: same docker-compose up steps as above
```

**DigitalOcean Volumes** for persistent data:
```bash
doctl compute volume create brain-data --size 100GiB --region nyc1
doctl compute volume-action attach <volume-id> <droplet-id>
```

### Option C: GCP Cloud Run (Serverless, pay per use)

Good for: low-traffic, cost-sensitive deployments.
Not ideal for: learner service (needs persistent model in memory).

```bash
# Build and push images to GCR
gcloud auth configure-docker
docker compose build
docker tag brain_thalamus:latest gcr.io/YOUR_PROJECT/thalamus:latest
docker push gcr.io/YOUR_PROJECT/thalamus:latest

# Deploy to Cloud Run
gcloud run deploy thalamus \
  --image gcr.io/YOUR_PROJECT/thalamus:latest \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "$(cat .env | tr '\n' ',')"
```

### Option D: Fly.io (Simplest global deployment)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# For each service:
fly launch --name brain-thalamus --dockerfile services/thalamus/Dockerfile
fly secrets import < .env
fly deploy
```

### Option E: Self-hosted Kubernetes (k3s on VPS)

For teams wanting production-grade orchestration on a budget:
```bash
# Install k3s (single-node Kubernetes)
curl -sfL https://get.k3s.io | sh

# Convert docker-compose to k8s manifests
pip install kompose
kompose convert -f docker-compose.yml

# Deploy
kubectl apply -f .
```

---

## Security Hardening for Production

```bash
# 1. Set strong passwords in .env
REDIS_PASSWORD=$(openssl rand -hex 32)
ELASTICSEARCH_PASSWORD=$(openssl rand -hex 32)
SUPERGROKIE_WEBHOOK_SECRET=$(openssl rand -hex 32)

# 2. Enable Redis password auth
# In databases/redis/redis.conf, uncomment:
# requirepass ${REDIS_PASSWORD}

# 3. Enable Elasticsearch security
# In docker-compose.yml elasticsearch env:
# - xpack.security.enabled=true

# 4. Run services as non-root
# All service Dockerfiles use python:3.12-slim which defaults to root
# Add to Dockerfiles:
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# 5. Restrict Docker socket access
# self-update uses /var/run/docker.sock — dangerous
# In production, use Docker SDK with TLS or restrict to read-only

# 6. Use HTTPS
# Add certbot + nginx SSL config for production domains

# 7. Rate limiting
# Add nginx rate limiting in nginx.conf:
# limit_req_zone $binary_remote_addr zone=brain:10m rate=10r/s;
```

---

## Monitoring and Maintenance

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f thalamus

# Memory usage per container
docker stats

# Full brain health
curl http://localhost:8000/health

# Knob history (who changed what)
curl http://localhost:8090/history

# Pending code changes (from self-update)
curl http://localhost:8000/update/pending

# Training history (local model learning)
curl http://localhost:8000/brain/learner/training-history

# Calibration status (metacognition accuracy)
curl http://localhost:8000/brain/metacognition/calibration

# Causal graph statistics
curl http://localhost:8000/brain/causal-graph/stats

# Visual memory count
curl http://localhost:8000/brain/vision/stats

# Active goals
curl http://localhost:8000/brain/goals/goals

# Hormonal state (per session)
curl http://localhost:8000/brain/hypothalamus/state/<session_id>

# Inject a hormone manually (e.g. boost dopamine by 0.15)
curl -X POST http://localhost:8000/brain/hypothalamus/inject \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my_session", "hormone": "dopamine", "amount": 0.15}'

# Inject with gland bypass (exact amount, ignores gland capacity)
curl -X POST http://localhost:8000/brain/hypothalamus/inject \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my_session", "hormone": "cortisol", "amount": -0.10, "bypass_gland": true}'

# Docker container logs (via dashboard API)
curl http://localhost:8080/api/docker-logs/thalamus?tail=100
curl http://localhost:8080/api/docker-logs/hypothalamus?tail=50

# Insight Engine — pending epiphanies
curl http://localhost:8000/brain/insight-engine/pending/<session_id>

# Restart a single service (hot swap)
docker compose restart compass

# Update a single service
docker compose build compass && docker compose up -d --no-deps compass
```

---

## Backup and Recovery

```bash
# Backup databases
docker exec brain_redis_1 redis-cli SAVE
docker cp brain_redis_1:/data/dump.rdb ./backups/redis_$(date +%Y%m%d).rdb

# Elasticsearch snapshot
curl -X PUT "localhost:9200/_snapshot/brain_backup" \
  -H "Content-Type: application/json" \
  -d '{"type":"fs","settings":{"location":"/usr/share/elasticsearch/data/snapshots"}}'

# EdgeDB backup
docker exec brain_edgedb_1 edgedb dump --dsn edgedb://edgedb/main > ./backups/edgedb_$(date +%Y%m%d).dump

# Restore knobs from history
curl -X POST http://localhost:8090/history/rollback?steps=1

# Restore a pending code change was rejected
curl http://localhost:8000/update/change-log
```

---

## Switching LLM Provider

9 providers supported. Switch at any time — no code changes, no rebuild:

```bash
# 1. Edit .env — pick one:
LLM_PROVIDER=anthropic      # Claude (default)
LLM_PROVIDER=openai         # GPT-5.4-mini / GPT-5.4-nano
LLM_PROVIDER=gemini         # Google Gemini
LLM_PROVIDER=ollama         # Local Ollama models (free, private)
LLM_PROVIDER=deepseek       # DeepSeek R1
LLM_PROVIDER=groq           # Groq (fast inference)
LLM_PROVIDER=xai            # xAI Grok 4.20
LLM_PROVIDER=openrouter     # OpenRouter (multi-model)
LLM_PROVIDER=generic        # Any OpenAI-compatible endpoint

# 2. Set the corresponding API key:
ANTHROPIC_API_KEY=sk-ant-...
# or OPENAI_API_KEY=sk-...
# or GEMINI_API_KEY=...
# or DEEPSEEK_API_KEY=...
# or GROQ_API_KEY=...
# or XAI_API_KEY=...
# or OPENROUTER_API_KEY=...
# or GENERIC_API_KEY=... + GENERIC_BASE_URL=https://your-endpoint/v1

# 3. Optional: override default models
ANTHROPIC_MAIN_MODEL=claude-sonnet-4-20250514
ANTHROPIC_FAST_MODEL=claude-haiku-4-5-20251001
# or OPENAI_MAIN_MODEL=gpt-5.4-mini / OPENAI_FAST_MODEL=gpt-5.4-nano
# or XAI_MAIN_MODEL=grok-4-20 / XAI_FAST_MODEL=grok-3-mini-fast
# etc.

# 4. Optional: override token limits (defaults: main=16384, fast=8192)
MAX_TOKENS_MAIN=16384
MAX_TOKENS_FAST=8192

# 5. Restart services (no rebuild needed)
./skynet.sh restart

# 6. Verify
curl http://localhost:8000/health | grep llm_provider
```

## Dashboard

Access the web dashboard at **http://localhost:8080** after startup.

Features:
- **Terminal**: Chat with the brain, voice input (STT), webcam emotional analysis
- **Chat History**: Conversations are persisted in browser localStorage and survive page refresh. Trace/debug data is stripped before saving to prevent localStorage overflow. Switch between sessions in the sidebar — each session's full history is restored.
- **Session Recall**: The brain remembers the entire conversation within a session. Full messages are stored in Redis working memory (200 turns, 7-day TTL, no truncation) and fed to the LLM as context, so it can reference previous exchanges naturally.
- **3D Brain**: Three.js rendered brain with 14 anatomical regions (including hypothalamus + insightEngine), auto-rotation, click-to-expand, drag to rotate, emissive glow on activation during cognitive processing
- **Pixelated Face**: Pixel-art avatar (80x80) showing current emotion — 6 distinct expressions (neutral, joy, curiosity, alarm, regret, belonging)
- **Emotion Tracker**: Glassmorphism panel (upper-left) with current feeling, overall mood bar, expandable history of last 20 interactions
- **Hormonal State**: 16-hormone panel below emotion tracker showing real-time hormone levels per gland, with overstimulation/depletion warnings
- **Hormone Injection**: Manual hormone injection panel (below hormone bars, above sessions) — pick any of the 16 hormones, set amount ±0.40, optional gland bypass
- **Monitoring**: Per-service metrics (all 24 services), latency, error rates, sparkline charts (7-day history)
- **Logging**: App logs per service + Docker container logs toggle, live WebSocket event stream
- **Response Trace**: Every pipeline step with wall-clock ms timing, ordered `pipeline_order` summary, full hormone activity (levels, deltas, gland energy, modified knobs)
- **Controls**: Rotary dial knobs (drag/scroll to adjust), personality presets, perception feed
- **Autonomy**: Activate/deactivate autonomous thinking, view self-generated insights
- **Premium Theme**: Glassmorphism, gradient backgrounds, glowing borders, smooth transitions

## Autonomous Thinking

The autonomy service (port 8018) enables the brain to think on its own:

```bash
# Activate via skynet.sh
./skynet.sh think

# Or via API
curl -X POST http://localhost:8018/activate \
  -H "Content-Type: application/json" \
  -d '{"mode": "full", "interval_seconds": 300}'

# Modes: observe, analyze, improve, full
# Risk levels: AUTONOMY_MAX_RISK=low|medium|high (in .env)
```

That's all. The llm_provider.py abstraction handles everything else.
Model quality differences between providers will be visible in response quality.
Anthropic Claude Sonnet 4 and OpenAI GPT-4o are broadly comparable.