# INSTRUCTIONS.md
# Brain System — Complete Setup, Configuration, and Redeployment Guide

This document covers every variable, key, secret, code path, and command
you need to deploy the brain system from zero to running, and redeploy it
cleanly any number of times without errors.

Read this once fully before touching anything.

---

## Table of Contents

1. [What You Need Before You Start](#1-what-you-need-before-you-start)
2. [Directory Structure](#2-directory-structure)
3. [Every Environment Variable](#3-every-environment-variable)
4. [Every API Key — Where to Get Them](#4-every-api-key--where-to-get-them)
5. [First-Time Setup — Step by Step](#5-first-time-setup--step-by-step)
6. [Verifying Everything Works](#6-verifying-everything-works)
7. [Redeployment Procedures](#7-redeployment-procedures)
8. [Updating Individual Services](#8-updating-individual-services)
9. [Code Changes That Require Extra Steps](#9-code-changes-that-require-extra-steps)
10. [supergrokie Integration — Build and Maintain](#10-supergrokie-integration--build-and-maintain)
11. [Local Model — First Download and Updates](#11-local-model--first-download-and-updates)
12. [Vision Service — Camera Setup](#12-vision-service--camera-setup)
13. [Common Errors and Fixes](#13-common-errors-and-fixes)
14. [Production Hardening Checklist](#14-production-hardening-checklist)

---

## 1. What You Need Before You Start

### Required on every machine that runs this:

```
Docker          24.0+     docker --version
Docker Compose  2.20+     docker compose version
Git             any       git --version
curl            any       curl --version
Ollama          any       ollama --version       (must be on HOST, not in Docker)
```

### Hardware minimums:

```
TIER        RAM     CPU CORES   DISK      USE CASE
──────────────────────────────────────────────────────────────
Minimal     16 GB   4           50 GB     Dev, no learner model
Standard    32 GB   8           100 GB    Personal assistant
Full        64 GB   16          300 GB    Production, all services
```

### Ollama must be installed on the HOST machine (not in Docker):

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify it's running
ollama list

# Pull required models BEFORE starting the brain
ollama pull nomic-embed-text    # 274 MB  — REQUIRED for all embedding
ollama pull phi4                # 9 GB    — optional, local inference
```

Ollama must be running as a service on the host. The brain containers reach
it via `host.docker.internal:11434` (Mac/Linux with Docker Desktop) or
`172.17.0.1:11434` (Linux with native Docker). See Section 3 for how to
configure this.

---

## 2. Directory Structure

Your brain directory must look exactly like this before deployment:

```
brain/
├── .env                          ← you create this from .env.example
├── docker-compose.yml
├── SUPERGROKIE.md                  ← supergrokie architecture context (do not delete)
├── nginx/
│   └── nginx.conf
├── shared/
│   ├── __init__.py
│   ├── models.py
│   ├── knobs.py
│   ├── events.py
│   └── llm_provider.py
├── services/
│   ├── thalamus/     main.py  Dockerfile  requirements.txt
│   ├── amygdala/     main.py  Dockerfile  requirements.txt
│   ├── hippocampus/  main.py  Dockerfile  requirements.txt
│   ├── neocortex/    main.py  Dockerfile  requirements.txt
│   ├── cerebellum/   main.py  Dockerfile  requirements.txt
│   ├── working-memory/
│   ├── perception/
│   ├── verification/
│   ├── cross-domain/
│   ├── compass/
│   ├── framework-router/
│   ├── self-update/  main.py  Dockerfile  requirements.txt
│   ├── knobs/
│   ├── causal-graph/
│   ├── simulator/
│   ├── metacognition/
│   ├── goals/
│   ├── learner/
│   └── vision/
└── databases/
    ├── redis/        Dockerfile  redis.conf  init.py  init.sh  requirements.txt
    ├── elasticsearch/ Dockerfile  init.py  requirements.txt
    └── edgedb/       Dockerfile  init.py  requirements.txt  schema/schema.esdl
```

If any of these files are missing the build will fail at that service.

---

## 3. Every Environment Variable

Copy `.env.example` to `.env`. Every variable below must exist in `.env`.
Variables marked **REQUIRED** have no working default and will break the
system if left empty.

### LLM Provider

```
LLM_PROVIDER=anthropic
```
Controls which AI provider is used for reasoning.
- `anthropic` — uses Claude (Sonnet for reasoning, Haiku for fast calls)
- `openai`    — uses GPT-4o and GPT-4o-mini
- To switch: change this value, restart thalamus and compass only

```
ANTHROPIC_API_KEY=sk-ant-...          REQUIRED if LLM_PROVIDER=anthropic
OPENAI_API_KEY=sk-...                 REQUIRED if LLM_PROVIDER=openai
```

```
ANTHROPIC_MAIN_MODEL=claude-sonnet-4-20250514
ANTHROPIC_FAST_MODEL=claude-haiku-4-5-20251001
OPENAI_MAIN_MODEL=gpt-4o
OPENAI_FAST_MODEL=gpt-4o-mini
DEFAULT_CLAUDE_MODEL=claude-sonnet-4-20250514
FAST_CLAUDE_MODEL=claude-haiku-4-5-20251001
```
These are the model names sent in API calls. Update if Anthropic or OpenAI
releases new models. The DEFAULT_CLAUDE_MODEL and FAST_CLAUDE_MODEL values
are legacy aliases read by older service code — keep them in sync with the
Anthropic model names.

### Databases

```
REDIS_URL=redis://redis:6379
```
Connection string for Redis. Inside Docker the hostname is `redis` (the
container name). If you use an external Redis (e.g. AWS ElastiCache), change
this to the external endpoint and remove the redis service from compose.

```
REDIS_PASSWORD=
```
Leave empty for development. For production, set a strong password and
uncomment `requirepass` in `databases/redis/redis.conf`.

```
ELASTICSEARCH_URL=http://elasticsearch:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=
```
If using Amazon OpenSearch or Elastic Cloud, replace the URL and credentials.
If left with empty password, xpack security must be disabled (already done
in the compose file with `xpack.security.enabled=false`).

```
EDGEDB_DSN=edgedb://edgedb/main
EDGEDB_HOST=edgedb
EDGEDB_PORT=5656
EDGEDB_DATABASE=main
```
EdgeDB connection. If using an external EdgeDB instance, update EDGEDB_DSN
to the full connection string including auth credentials.

### Local Model (Learner Service)

```
LOCAL_MODEL_NAME=microsoft/phi-4-mini-instruct
```
The HuggingFace model ID to download. This is pulled automatically on first
start. Options:
- `microsoft/phi-4-mini-instruct` — 3.8B, runs on 4GB RAM (4-bit quantized)
- `Qwen/Qwen2.5-1.5B-Instruct`   — 1.5B, very fast on CPU, 2GB RAM
- `microsoft/phi-2`               — 2.7B, good balance

```
LOCAL_MODEL_PATH=/models/local_llm
ADAPTER_PATH=/models/adapters
EMOTION_MODEL_PATH=/models/emotion
```
Paths inside the container where models are stored. These map to the
`models-volume` Docker volume. Do not change unless you also update the
volume mount in docker-compose.yml.

```
HF_TOKEN=hf_...
```
HuggingFace access token. Required for gated models (Llama 3, Mistral, etc.).
Not required for Phi-4-mini or Qwen. Get from: https://huggingface.co/settings/tokens

```
HF_HOME=/models/hf_cache
```
HuggingFace cache directory inside the container. Leave as-is.

### Ollama

```
OLLAMA_URL=http://host.docker.internal:11434
```
How containers reach Ollama on the host machine.
- **Mac (Docker Desktop)**: `http://host.docker.internal:11434` ← default, works
- **Linux (Docker Desktop)**: `http://host.docker.internal:11434` ← works
- **Linux (native Docker)**: `http://172.17.0.1:11434` ← use this instead
- **Ollama in Docker**: `http://ollama:11434` ← only if you run Ollama in a container

```
OLLAMA_EMBED_MODEL=nomic-embed-text
```
The embedding model. Must be pulled on the host (`ollama pull nomic-embed-text`)
before starting. This is used by neocortex, hippocampus, cross-domain,
causal-graph, and metacognition.

```
OLLAMA_LOCAL_MODEL=phi4
```
The Ollama inference model used as a fallback. Pull with `ollama pull phi4`.
Optional — the main reasoning is done by the LLM provider, not Ollama.

### supergrokie Code Modification Engine

```
SUPERGROKIE_PATH=/usr/local/bin/supergrokie
```
Path to supergrokie inside the self-update container. Installed during
Docker image build. Do not change unless you are mounting the binary
from the host instead of building it.

```
SUPERGROKIE_GIT_REF=main
```
Which git ref to install supergrokie from. Options:
- `main`          — latest (may include breaking changes)
- `<commit-sha>`  — pinned to a specific commit (recommended for production)

To pin: run `git log --oneline` in the supergrokie repo, copy the hash,
set `SUPERGROKIE_GIT_REF=a3f91b2` (or whatever the hash is).

### Self-Update / Supergrokie

```
SUPERGROKIE_WEBHOOK_SECRET=change-this-to-a-random-secret-min-32-chars
```
**REQUIRED.** The HMAC secret that Supergrokie uses to sign webhook payloads.
Must be at least 32 characters. Generate with:
```bash
openssl rand -hex 32
```
This value must match what you configure in Supergrokie's webhook settings.

```
GITHUB_TOKEN=ghp_...
GITHUB_REPO=your-username/brain-system
```
Optional. Used if self-update commits generated code to a GitHub repo for
review. Not required for local-only operation.

```
AUTO_APPLY_CODE_CHANGES=false
```
- `false` — all code changes go to a pending queue. You approve them manually
  with `POST /update/approve/{id}`. Recommended.
- `true`  — supergrokie-generated code is applied immediately without review.
  Only use this if you trust the system completely.

```
CODE_REVIEW_REQUIRED=true
```
- `true`  — generated code is always queued for review regardless of confidence
- `false` — high-confidence, low-risk code changes may be auto-applied

### Vision

```
VISION_ENABLED=false
```
Set to `true` to enable camera capture. Also requires the vision profile:
`docker compose --profile vision up -d`

```
VISION_MODEL=openai/clip-vit-base-patch32
```
CLIP model for visual embeddings. Downloaded from HuggingFace on first start.
Alternative: `openai/clip-vit-large-patch14` (better quality, larger, slower)

```
VISION_CAPTURE_INTERVAL=0
```
Seconds between automatic captures. `0` means manual-only (no auto-capture).
Set to `5` for a frame every 5 seconds.

```
CAMERA_INDEX=0
```
USB camera device index. `0` is the first camera. If you have multiple
cameras, try `1`, `2` etc. Only used when VISION_ENABLED=true.

```
VISION_STORE_FRAMES=false
```
If `true`, JPEG frames are saved to disk under the frames-volume.
If `false`, only the embedding and caption are stored (Elasticsearch only).

### Service URLs (Internal Docker Network)

These are the HTTP addresses services use to talk to each other.
Do not change these unless you are moving services to different hosts.

```
AMYGDALA_URL=http://amygdala:8001
HIPPOCAMPUS_URL=http://hippocampus:8002
CEREBELLUM_URL=http://cerebellum:8003
NEOCORTEX_URL=http://neocortex:8004
WORKING_MEMORY_URL=http://working-memory:8005
PERCEPTION_URL=http://perception:8006
VERIFICATION_URL=http://verification:8007
CROSS_DOMAIN_URL=http://cross-domain:8008
COMPASS_URL=http://compass:8009
FRAMEWORK_ROUTER_URL=http://framework-router:8010
SELF_UPDATE_URL=http://self-update:8011
KNOBS_URL=http://knobs:8090
CAUSAL_GRAPH_URL=http://causal-graph:8012
SIMULATOR_URL=http://simulator:8013
METACOGNITION_URL=http://metacognition:8014
GOALS_URL=http://goals:8015
LEARNER_URL=http://learner:8016
VISION_URL=http://vision:8017
THALAMUS_URL=http://thalamus:8000
```

### Resource Limits

```
ES_JAVA_OPTS=-Xms2g -Xmx2g
```
Elasticsearch JVM heap. Rule: set to half your available RAM, max 32GB.
On a 16GB machine: `-Xms4g -Xmx4g`
On a 32GB machine: `-Xms8g -Xmx8g`

```
REDIS_MAX_MEMORY=1gb
```
Maximum Redis memory. If Redis hits this, it evicts least-recently-used keys.
Increase on machines with more RAM.

```
LEARNER_MAX_MEMORY=8g
VISION_MAX_MEMORY=4g
```
Container memory limits. Increase if learner OOM-kills during model loading.

### Ports (exposed to host)

```
GATEWAY_PORT=8000
KNOBS_PORT=8090
```
The ports the gateway and knobs service expose to the host machine.
All other services are internal to the Docker network only.

### Paths

```
WORKSPACE_PATH=/workspace
MODELS_PATH=/models
SERVICES_DIR=/services
```
Internal container paths. The workspace and models map to named Docker volumes.
The services directory is mounted from `./services` on the host.

---

## 4. Every API Key — Where to Get Them

### Anthropic API Key (`ANTHROPIC_API_KEY`)
**Required if `LLM_PROVIDER=anthropic`**
1. Go to https://console.anthropic.com
2. Account → API Keys → Create Key
3. Copy the key — it is shown only once
4. Format: `sk-ant-api03-...`

### OpenAI API Key (`OPENAI_API_KEY`)
**Required if `LLM_PROVIDER=openai`**
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Format: `sk-...`

### HuggingFace Token (`HF_TOKEN`)
**Required only for gated models (Llama 3, Mistral)**
**Not required for Phi-4-mini or Qwen**
1. Go to https://huggingface.co/settings/tokens
2. New Token → Read access
3. Format: `hf_...`

### Supergrokie Webhook Secret (`SUPERGROKIE_WEBHOOK_SECRET`)
**Required for Supergrokie integration**
**Not required if you don't use Supergrokie**
Generate locally — this is not obtained from a third party:
```bash
openssl rand -hex 32
```
This value must be set identically in:
1. Your `.env` file (`SUPERGROKIE_WEBHOOK_SECRET=...`)
2. Supergrokie's webhook configuration panel

### GitHub Token (`GITHUB_TOKEN`)
**Optional — only needed if self-update commits code to GitHub**
1. GitHub → Settings → Developer Settings → Personal Access Tokens → Fine-grained
2. Select repository access for your brain repo
3. Permissions: Contents (read/write)

---

## 5. First-Time Setup — Step by Step

Follow these steps in exact order. Do not skip any step.

### Step 1: Clone and configure

```bash
# Clone your brain repository
git clone https://github.com/your-username/brain-system.git
cd brain-system

# Create your .env file from the example
cp .env.example .env
```

Open `.env` and fill in every **REQUIRED** value:
```
ANTHROPIC_API_KEY=sk-ant-...        ← your real key
SUPERGROKIE_WEBHOOK_SECRET=<32+ chars> ← generate with: openssl rand -hex 32
```

Set your Ollama URL correctly for your OS (see Section 3).

### Step 2: Pull Ollama models on the host

```bash
# Verify Ollama is running
ollama list

# Pull the embedding model (required — without this neocortex, hippocampus,
# and cross-domain will fail on every request)
ollama pull nomic-embed-text

# Optional: pull local inference model
ollama pull phi4
```

Do not skip this. The embedding model is not downloaded by Docker — it must
exist in Ollama on the host before any brain services start.

### Step 3: Build all images

```bash
docker compose build
```

This builds every service image. Expected duration:
- Without supergrokie (if self-update is excluded): 5-10 minutes
- With supergrokie: 5-15 minutes on first build

Supergrokie installation only runs once — the
Docker layer cache preserves it on all subsequent builds.

Watch for errors. If any service fails to build, fix it before proceeding.
A failed build does not leave broken images — Docker rolls back automatically.

### Step 4: Start the databases

```bash
docker compose up -d redis elasticsearch edgedb
```

Wait for all three to be healthy:
```bash
# Check every 5 seconds until all three show "healthy"
watch docker compose ps redis elasticsearch edgedb
```

Expected wait times:
- Redis: 5-10 seconds
- EdgeDB: 20-40 seconds
- Elasticsearch: 30-60 seconds

Do not proceed until all three show `healthy`. If any stays `starting` for
more than 2 minutes, check logs:
```bash
docker compose logs elasticsearch
docker compose logs edgedb
```

### Step 5: Initialize the databases

```bash
docker compose up db-init
```

Watch the output. You should see:
```
--- Redis init ---
Redis brain namespace initialized
--- Elasticsearch init ---
  Created index: neocortex_knowledge
  Created index: neocortex_patterns
  Created index: visual_memories
  Created index: causal_events
--- EdgeDB init ---
Migrations applied successfully
Seeded 8 base concepts
=== All databases initialized ===
```

If initialization fails, check the error and fix it before proceeding.
Common causes: database not healthy yet (wait longer), wrong DSN.

### Step 6: Start all brain services

```bash
docker compose up -d
```

Thalamus starts last because it depends on every other service being healthy.
The full startup sequence takes 2-4 minutes.

Watch the startup:
```bash
docker compose logs -f thalamus
```

You are looking for the thalamus health check to pass. Once it does,
the full brain is running.

### Step 7: Verify (see Section 6)

---

## 6. Verifying Everything Works

Run these checks in order. Each one verifies a different layer.

### Check 1: All containers running
```bash
docker compose ps
```
Every service should show `running` or `healthy`.
`db-init` shows `exited (0)` — that is correct, it is a one-shot container.
No service should show `restarting` or `exited` with a non-zero code.

### Check 2: Full brain health
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```
Expected: every brain region shows `"alive"`. A `"degraded"` status means
fewer than 5 services are unreachable — the brain still works.

### Check 3: Knobs responding
```bash
curl -s http://localhost:8090/current | python3 -m json.tool
```
Expected: a JSON object with 19 knob values (humor, honesty, empathy, etc.)

### Check 4: LLM provider connected
```bash
curl -s http://localhost:8000/health | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('LLM provider:', data.get('llm_provider', {}).get('provider'))
print('Has key:', data.get('llm_provider', {}).get('has_key'))
"
```
Expected: `LLM provider: anthropic` (or openai) and `Has key: True`

### Check 5: First real interaction
```bash
curl -s -X POST http://localhost:8000/brain/process \
  -H "Content-Type: application/json" \
  -d '{"session_id": "verify_001", "user_input": "Hello. Are you working?"}' \
  | python3 -m json.tool
```
Expected: a `response` field with actual text, `cognitive_path` showing
how the brain processed it, `execution_ms` showing total time.

### Check 6: supergrokie available
```bash
curl -s http://localhost:8000/update/supergrokie-status
```
Expected: `"available": true` with a version string.
If `"available": false`: supergrokie was not installed. Check self-update logs:
```bash
docker compose logs self-update | grep -i supergrokie
```

### Check 7: Ollama embedding working
```bash
curl -s http://localhost:8000/brain/neocortex/search \
  -H "Content-Type: application/json" \
  -d '{"request_id": "t", "session_id": "t", "user_input": "test"}' \
  | python3 -m json.tool
```
Expected: a response (even empty results). If it errors with embedding
failure, Ollama is not reachable — check OLLAMA_URL in .env.

### Check 8: Learner status
```bash
curl -s http://localhost:8000/brain/learner/status | python3 -m json.tool
```
If `model_loaded: false` and no `load_error` — model is still downloading
(normal on first start, takes 10-30 minutes). Check progress:
```bash
docker compose logs -f learner
```

---

## 7. Redeployment Procedures

There are four types of redeployment. Use the right one for your situation.

---

### Type 1: Rolling restart (code changes, no data wipe)

Use when: you have changed service code, updated .env, or pulled new code.
Data is fully preserved. Services restart one at a time.

```bash
# Pull latest code (if applicable)
git pull

# Rebuild changed images
# If you only changed one service:
docker compose build <service-name>
# If you changed shared/ or multiple services:
docker compose build

# Rolling restart — Docker replaces containers one at a time
docker compose up -d

# Verify
curl -s http://localhost:8000/health | python3 -m json.tool
```

This is idempotent. Run it any number of times. Services that have not
changed are not restarted. Database init runs again (idempotent — no effect
if already initialized).

**Important**: The database init container (`db-init`) runs again every time
you do `docker compose up`. All init scripts are written to be safe:
- Redis: `SET` is idempotent
- Elasticsearch: checks `indices.exists()` before creating
- EdgeDB: uses `UNLESS CONFLICT` on all inserts, migrations detect current state

---

### Type 2: Full rebuild (dependency changes, Dockerfile changes)

Use when: you changed requirements.txt, Dockerfile, or the supergrokie Dockerfile.

```bash
docker compose build --no-cache
docker compose up -d
```

`--no-cache` forces Docker to re-download base images and re-run all build
steps. Slower, but guarantees a clean state. Use this when you suspect a
stale layer is causing problems.

**Warning**: `--no-cache` will re-install supergrokie.
Only use it when you actually changed the Dockerfile, not for routine deploys.

---

### Type 3: Data-preserving full restart

Use when: containers are in a bad state, you want to restart everything
cleanly without losing any stored data.

```bash
# Stop all containers (data volumes are NOT removed)
docker compose down

# Start fresh
docker compose up -d
```

`docker compose down` without the `-v` flag preserves all named volumes:
- redis-data (knobs, working memory, goals, causal graph)
- elasticsearch-data (all semantic knowledge, visual memories)
- edgedb-data (all episodic memories, skills)
- models-volume (downloaded local models)
- workspace-volume (replay buffer, training logs, pending changes)

Your data is exactly where you left it. The brain resumes with full memory.

---

### Type 4: Complete factory reset (wipes everything)

Use when: you want to start completely fresh with no memory, no models,
no stored data. This is irreversible.

```bash
# Stop all containers AND remove all volumes
docker compose down -v

# Remove built images (optional — forces full rebuild)
docker compose down --rmi all -v

# Start fresh from zero
docker compose up -d
```

After this, the learner service will re-download the local model (10-30 min),
all episodic and semantic memory is gone, all goals are gone, all learned
skills are gone. The brain has no past.

---

### Redeployment decision tree

```
Did you change .env?
  └── YES → docker compose up -d        (rolling restart, no rebuild needed)
             Then restart affected services

Did you change a service's main.py?
  └── YES → docker compose build <service>
             docker compose up -d --no-deps <service>

Did you change shared/ (models.py, knobs.py, llm_provider.py)?
  └── YES → docker compose build
             docker compose up -d        (rebuilds all services that use shared/)

Did you change requirements.txt or Dockerfile?
  └── YES → docker compose build --no-cache <service>
             docker compose up -d --no-deps <service>

Did you change databases/*/init.py or schema?
  └── YES → docker compose up db-init   (safe to rerun, idempotent)

Is everything broken and you don't know why?
  └── → docker compose down
        docker compose up -d
        docker compose logs -f

Is it totally corrupted and you want to start over?
  └── → docker compose down -v
        docker compose up -d
```

---

## 8. Updating Individual Services

### Update a single service (most common operation):

```bash
# Make your code change to services/<name>/main.py

# Rebuild only that service
docker compose build <name>

# Restart only that service (--no-deps = don't restart dependencies)
docker compose up -d --no-deps <name>

# Verify it came back healthy
docker compose ps <name>
curl -s http://localhost:8000/brain/<path>/health
```

### Services and their restart impact:

```
SERVICE         RESTART IMPACT               DATA LOSS ON RESTART?
─────────────────────────────────────────────────────────────────────────
thalamus        Interrupts active sessions   None (state in Redis)
amygdala        Zero — stateless             None
hippocampus     Zero — state in EdgeDB       None
neocortex       Zero — state in ES           None
cerebellum      Reloads skill cache (5-10s)  None
working-memory  Interrupts active sessions   None (state in Redis)
perception      Zero — stateless             None
verification    Clears in-memory cache       None (cache regenerates)
cross-domain    Zero — stateless             None
compass         Clears session care states   Care states in Redis survive
framework-router Zero — stateless            None
knobs           Brief knob read gap          None (Redis persists)
self-update     Brief webhook unavailability None
causal-graph    Zero — state in Redis        None
simulator       Zero — stateless             None
metacognition   Clears calibration cache     Calibration in Redis survives
goals           Zero — state in Redis        None
learner         Model unloads, reloads (2-5 min) None (adapter saved)
vision          Camera disconnects briefly   None (ES persists)
```

### Changing the LLM provider (Anthropic ↔ OpenAI):

```bash
# 1. Edit .env
LLM_PROVIDER=openai            # or anthropic

# 2. Restart only the services that use the LLM provider directly
docker compose up -d --no-deps \
  thalamus compass verification cross-domain \
  metacognition framework-router self-update goals

# 3. Verify
curl -s http://localhost:8000/health | grep -A3 llm_provider
```

No rebuild needed. `llm_provider.py` reads `LLM_PROVIDER` from environment
on every call. Switching takes about 30 seconds.

---

## 9. Code Changes That Require Extra Steps

Some code changes need more than just a rebuild. These will break silently
if you skip the extra steps.

### Changing shared/models.py (BrainRequest, BrainResponse, KnobSet)

Every service imports from models.py. If you add a field, change a type,
or rename anything:

```bash
# Rebuild ALL services (not just one)
docker compose build

# Restart everything
docker compose up -d

# If you changed KnobSet (added/removed a knob):
# Also update the Redis default in databases/redis/init.py
# AND re-run db-init:
docker compose up db-init
```

If you remove a field from KnobSet, any service that reads that field from
Redis will get a validation error until the Redis key is updated.

### Changing shared/llm_provider.py

Any service that calls `complete()` or `complete_json()` is affected.
That is currently: thalamus, compass, verification, cross-domain,
metacognition, framework-router, self-update, goals, causal-graph, vision.

```bash
docker compose build thalamus compass verification cross-domain \
                     metacognition framework-router self-update \
                     goals causal-graph vision
docker compose up -d
```

### Changing shared/knobs.py (KnobsClient)

All 19 services use this. Rebuild everything.

```bash
docker compose build
docker compose up -d
```

### Adding a new knob (new field in KnobSet)

1. Add the field to `shared/models.py` in the `KnobSet` class with a default
2. Add a description to `services/knobs/main.py` in the `descriptions` dict
3. Add the knob to the default value in `databases/redis/init.py`
4. Rebuild all services and re-run db-init:

```bash
docker compose build
docker compose up db-init    # seeds the new knob default into Redis
docker compose up -d
```

### Adding a new brain service

1. Create `services/<name>/main.py`, `Dockerfile`, `requirements.txt`
2. Add the service to `docker-compose.yml`
3. Add the URL to `.env`: `NEW_SERVICE_URL=http://<name>:<port>`
4. Add the URL to `services/thalamus/main.py` SERVICES dict
5. Add a depends_on entry for thalamus in docker-compose.yml

```bash
docker compose build <name> thalamus
docker compose up -d
```

### Changing the EdgeDB schema (schema.esdl)

EdgeDB uses migrations. Never edit the schema file and restart hoping it
applies — you must create a migration:

```bash
# Connect to EdgeDB container
docker exec -it brain_edgedb bash

# Create migration (prompts you to confirm schema changes)
edgedb migration create --dsn edgedb://edgedb/main

# Apply migration
edgedb migrate --dsn edgedb://edgedb/main

exit
```

Or run via db-init (init.py calls edgedb migrate, which is idempotent):
```bash
docker compose up db-init
```

### Changing Elasticsearch index mappings

Elasticsearch does not allow changing the mapping of an existing index.
You must delete and recreate the index:

```bash
# Delete the index (DATA LOSS for that index)
curl -X DELETE http://localhost:9200/neocortex_knowledge

# Recreate via init (which will create it with new mapping)
docker compose up db-init
```

Or for a full mapping reset without data loss (using reindex):
```bash
# 1. Create a new index with the new mapping (via init.py with a temp name)
# 2. Reindex from old to new
curl -X POST http://localhost:9200/_reindex -H 'Content-Type: application/json' -d '{
  "source": {"index": "neocortex_knowledge"},
  "dest":   {"index": "neocortex_knowledge_v2"}
}'
# 3. Delete old, create alias, rename
```

### Changing nginx.conf (gateway routing)

```bash
# nginx reloads config without restarting
docker exec brain_gateway nginx -s reload

# Verify
docker exec brain_gateway nginx -t
```

---

## 10. supergrokie Integration — Build and Maintain

### What supergrokie is and why it's in the build

supergrokie (github.com/mytuo1/supergrokie) is a claude-code equivalent coding agent that
the self-update service uses to read existing service files, understand their
context, and make precise edits. It is compiled from source during the
Docker build of the self-update service.

### The build process

The self-update Dockerfile has two stages:

```
Stage 1 (rust:1.78-slim):
  git clone https://github.com/mytuo1/supergrokie
  cd rust && cargo build --release
  → produces: supergrokie (installed)

Stage 2 (python:3.12-slim):
  installs supergrokie into /usr/local/bin/supergrokie
  pip install requirements.txt
  → produces: final image with both supergrokie and Python runtime
```

The Rust compilation only runs once and is cached by Docker. Subsequent
builds skip re-installation unless you change SUPERGROKIE_GIT_REF.

### First build (slow):
```bash
docker compose build self-update
# Expected time: 15-25 minutes
# Watch progress: docker compose build self-update 2>&1 | tail -20
```

### Subsequent builds (fast):
```bash
docker compose build self-update
# Expected time: 30-60 seconds (cached layers)
```

### Pinning supergrokie to a specific version

The supergrokie repo is new and active. `main` may have breaking changes.
Once the project cuts a stable release, pin to it:

```bash
# Find a stable commit
git clone https://github.com/mytuo1/supergrokie supergrokie-check
cd supergrokie-check && git log --oneline | head -10
# Find the last known-good commit hash, e.g. a3f91b2

# Set in .env
SUPERGROKIE_GIT_REF=a3f91b2

# Rebuild self-update with the pinned version
docker compose build --no-cache self-update
docker compose up -d --no-deps self-update
```

### If the supergrokie repo moves, renames, or disappears

The Dockerfile clones from:
```
https://github.com/mytuo1/supergrokie
```

If this URL changes (the repo is mid-ownership-transfer as of this writing):
1. Open `services/self-update/Dockerfile`
2. Find the `git clone` line
3. Update to the new URL
4. Rebuild: `docker compose build --no-cache self-update`

If supergrokie becomes unavailable entirely, the self-update service falls back to
direct API code generation automatically. The brain continues to function.

### Verify supergrokie is working after any rebuild

```bash
curl -s http://localhost:8000/update/supergrokie-status | python3 -m json.tool
```

Expected:
```json
{
  "available": true,
  "path": "/usr/local/bin/supergrokie",
  "version": "supergrokie ..."
}
```

---

## 11. Local Model — First Download and Updates

### What happens on first start

When the learner service starts for the first time:
1. It checks if `LOCAL_MODEL_PATH` (`/models/local_llm`) has a `config.json`
2. If not, it downloads the model from HuggingFace
3. Phi-4-mini is ~8GB download (7.7GB in float16)
4. First download: 10-30 minutes depending on connection

```bash
# Watch the download progress
docker compose logs -f learner
```

You will see:
```
Loading local model...
  Downloading from HuggingFace: microsoft/phi-4-mini-instruct
  This may take 10-30 minutes on first run...
```

Once complete:
```
  Local model ready: microsoft/phi-4-mini-instruct
Learner service starting...
```

### Changing the model

To use a different model:

```bash
# 1. Update .env
LOCAL_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct

# 2. Delete the old model from the volume (optional — keeps disk usage down)
docker run --rm -v brain_models-volume:/models alpine \
  rm -rf /models/local_llm

# 3. Restart learner — it will download the new model
docker compose up -d --no-deps learner
docker compose logs -f learner
```

### Updating the emotion classifier

The emotion model is a fine-tuned DistilBERT trained on GoEmotions.
To use a pre-trained version:

```bash
# Train the base model (run on host with GPU or CPU — outside Docker)
pip install transformers datasets torch scikit-learn
python scripts/train_emotion_classifier.py

# Copy trained model into the volume
docker cp ./emotion_model/final brain_learner:/models/emotion
```

Or let the learner train it from corrections automatically — every 30
emotion corrections trigger an incremental fine-tune.

---

## 12. Vision Service — Camera Setup

### Without a physical camera

The vision service works for image uploads even without a camera:

```bash
# Enable vision (no camera needed for uploads)
# In .env:
VISION_ENABLED=true
# Comment out or leave the devices section in docker-compose.yml

# Start with vision profile
docker compose --profile vision up -d vision

# Test with an image upload
curl -X POST http://localhost:8000/brain/vision/upload \
  -F "file=@/path/to/any/image.jpg" \
  -F "session_id=test"
```

### With a USB camera

```bash
# 1. Verify camera exists on host
ls /dev/video*
# Expected: /dev/video0 (or video1, video2...)

# 2. Uncomment in docker-compose.yml:
#   devices:
#     - /dev/video0:/dev/video0

# 3. Set in .env:
VISION_ENABLED=true
CAMERA_INDEX=0

# 4. Start with vision profile
docker compose --profile vision up -d

# 5. Test capture
curl -X POST "http://localhost:8000/brain/vision/capture?session_id=test"
```

### On a server with no camera or display

Remove the `devices` section from docker-compose.yml entirely.
Vision will work for image uploads but not for live camera capture.
Set `VISION_ENABLED=false` if you do not need vision at all — it saves
4GB of container memory.

---

## 13. Common Errors and Fixes

### "Connection refused" on any service

The service is not running or not yet healthy.
```bash
docker compose ps                    # check state
docker compose logs <service>        # read the error
docker compose up -d --no-deps <service>  # restart it
```

### Elasticsearch keeps restarting with "max virtual memory areas"

```bash
# On Linux host only
sudo sysctl -w vm.max_map_count=262144

# Make permanent
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
```

### EdgeDB migration fails with "already exists"

```bash
docker compose exec edgedb \
  edgedb migration status --dsn edgedb://edgedb/main
```
If the migration is already applied, this is safe to ignore. If it shows
a genuine conflict, check `schema.esdl` for the change that caused it.

### "No module named 'shared'" in any service

The shared directory is not being copied into the container correctly.
Check the Dockerfile for that service — it must have:
```dockerfile
COPY shared /shared
```
And the working directory must be `/app`.

### Learner OOM-killed (exits with code 137)

```bash
# Increase memory limit in .env
LEARNER_MAX_MEMORY=12g

# Or use a smaller model
LOCAL_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct

# Restart
docker compose up -d --no-deps learner
```

### supergrokie not found after build

```bash
# Check if the binary was built
docker compose exec self-update ls -la /usr/local/bin/supergrokie

# If missing, force rebuild of self-update
docker compose build --no-cache self-update
docker compose up -d --no-deps self-update
```

### Knobs returning stale/wrong values

Redis may have old data. Reset to defaults:
```bash
curl -X POST http://localhost:8090/reset
```

Or re-run the init (reseeds only if key is missing):
```bash
docker compose up db-init
```

### Supergrokie webhook returns 401

The signature check is failing. Verify:
1. `SUPERGROKIE_WEBHOOK_SECRET` in `.env` matches Supergrokie's webhook secret exactly
2. No trailing spaces or newlines in the secret value
3. The webhook URL is correct: `http://your-host:8000/update/webhook/supergrokie`

### "embedding dimension mismatch" in Elasticsearch

This happens when the embedding model changes (different dimension output).
Fix by deleting and recreating the affected index:
```bash
curl -X DELETE http://localhost:9200/neocortex_knowledge
docker compose up db-init
```

### Thalamus health check failing but all other services healthy

Thalamus requires ALL services to be healthy before it starts. Check which
service is the bottleneck:
```bash
docker compose ps
```
Fix the unhealthy service first. Thalamus will become healthy once all
dependencies pass their health checks.

### db-init exits with non-zero code

Read the output carefully:
```bash
docker compose logs db-init
```
Common causes:
- Database not ready yet (wait longer before running init)
- Wrong DSN or URL in .env
- Python package version conflict in the init container

Re-run init after fixing the cause:
```bash
docker compose up db-init
```

---

## 14. Production Hardening Checklist

Before going to production, complete every item on this list.

### Secrets

```
[ ] ANTHROPIC_API_KEY set to real key (not example value)
[ ] SUPERGROKIE_WEBHOOK_SECRET is 32+ random characters (openssl rand -hex 32)
[ ] REDIS_PASSWORD set to strong password
[ ] ELASTICSEARCH_PASSWORD set to strong password
[ ] .env file is NOT committed to git (add .env to .gitignore)
[ ] .env.example committed instead, with no real values
```

### Redis password setup (if REDIS_PASSWORD is set)

```bash
# 1. Set password in .env:
REDIS_PASSWORD=your-strong-password

# 2. Update REDIS_URL to include password:
REDIS_URL=redis://:your-strong-password@redis:6379

# 3. Uncomment in databases/redis/redis.conf:
requirepass your-strong-password

# 4. Rebuild Redis:
docker compose build redis
docker compose up -d --no-deps redis
```

### Code change safety

```
[ ] AUTO_APPLY_CODE_CHANGES=false
[ ] CODE_REVIEW_REQUIRED=true
[ ] SUPERGROKIE_GIT_REF pinned to a specific commit hash (not "main")
```

### Resource limits

```
[ ] ES_JAVA_OPTS set to appropriate heap for your RAM
[ ] LEARNER_MAX_MEMORY set appropriately
[ ] VISION_ENABLED=false if no camera (saves 4GB)
```

### Networking

```
[ ] GATEWAY_PORT changed from 8000 if port conflicts
[ ] KNOBS_PORT (8090) not exposed publicly — it has no auth
    Add nginx auth or firewall rule for 8090
[ ] Docker socket mount in self-update is a security risk —
    Consider using Docker TCP with TLS instead for production
```

### Data backup

```bash
# Schedule these as cron jobs

# Redis backup (runs inside container)
docker exec brain_redis redis-cli BGSAVE
docker cp brain_redis:/data/dump.rdb ./backups/redis_$(date +%Y%m%d).rdb

# Elasticsearch snapshot
curl -X PUT "http://localhost:9200/_snapshot/backup" \
  -H 'Content-Type: application/json' \
  -d '{"type":"fs","settings":{"location":"/usr/share/elasticsearch/data/snapshots"}}'
curl -X PUT "http://localhost:9200/_snapshot/backup/$(date +%Y%m%d)?wait_for_completion=true"

# EdgeDB dump
docker exec brain_edgedb \
  edgedb dump --dsn edgedb://edgedb/main > ./backups/edgedb_$(date +%Y%m%d).dump
```

### Monitoring

```bash
# Add these to cron or a monitoring tool:

# Health check every minute
curl -sf http://localhost:8000/health > /dev/null || alert "brain unhealthy"

# Memory usage
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}"

# Learner training log
curl -s http://localhost:8000/brain/learner/training-history | tail -5
```

---

## Quick Reference

### Start the brain
```bash
docker compose up -d
```

### Stop the brain (preserve data)
```bash
docker compose down
```

### Full reset (wipe all data)
```bash
docker compose down -v && docker compose up -d
```

### Update one service
```bash
docker compose build <name> && docker compose up -d --no-deps <name>
```

### Switch LLM provider
```bash
# Edit .env: LLM_PROVIDER=openai
docker compose up -d --no-deps thalamus compass verification \
  cross-domain metacognition framework-router self-update goals
```

### View all logs
```bash
docker compose logs -f
```

### View one service's logs
```bash
docker compose logs -f <name>
```

### Full health check
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Apply a personality preset
```bash
curl -X POST http://localhost:8090/preset/researcher
```

### Tune a knob
```bash
curl -X PATCH "http://localhost:8090/adjust?knob_name=empathy&value=0.95"
```

### Talk to the brain
```bash
curl -X POST http://localhost:8000/brain/process \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "user_input": "Your question here"}'
```

### Review pending code changes (from supergrokie/self-update)
```bash
curl http://localhost:8000/update/pending
curl -X POST http://localhost:8000/update/approve/<change_id>
```

### Check active goals
```bash
curl http://localhost:8000/brain/goals/goals
```

### Rollback knob changes
```bash
curl -X POST "http://localhost:8090/history/rollback?steps=1"
```
