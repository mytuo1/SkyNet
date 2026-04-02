# services/self-update/main.py
# ── Self-Update: Autonomous Code Modification via claw ────────────────────────
#
# claw (from github.com/ultraworkers/claw-code) is the primary code engine.
# It replaces the "generate string → append to file" approach with a full
# agentic tool loop: read existing code → understand context → edit precisely
# → verify → done. Handles multi-file changes in a single session.
#
# MODIFICATION LEVELS:
#   Level 1  Knob adjustment     — Redis write, instant, no code
#   Level 2  Code modification   — claw edits existing service files
#   Level 3  New service         — claw generates full service from description
#
# FALLBACK:
#   If claw binary is not found or fails, every level falls back to the
#   direct API approach via llm_provider.complete(). The brain keeps working.
#
# FLOW (OpenClaw → Level 2 example):
#   OpenClaw: "Add a /stats endpoint to thalamus"
#       ↓ webhook → _interpret_instruction()
#       ↓ → _modify_with_claw("thalamus", "Add a /stats endpoint...")
#       ↓ → claw reads services/thalamus/main.py via CLAW.md context
#       ↓ → claw writes the endpoint directly into the file
#       ↓ → result queued for human review (CODE_REVIEW_REQUIRED=true)
#       ↓ → POST /approve/{id} → docker restart thalamus

import asyncio
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/shared")

import docker
import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus
from llm_provider import complete, complete_json

app = FastAPI(
    title     = "SelfUpdate",
    version   = "3.0.0",
    description = "claw-powered autonomous code modification"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

events = EventBus()
http   = httpx.AsyncClient(timeout=120.0)

# ── Configuration ──────────────────────────────────────────────────────────────
CLAW_BINARY      = os.getenv("CLAW_BINARY_PATH",       "/usr/local/bin/claw")
OPENCLAW_SECRET  = os.getenv("OPENCLAW_WEBHOOK_SECRET", "")
KNOBS_URL        = os.getenv("KNOBS_URL",               "http://knobs:8090")
CEREBELLUM_URL   = os.getenv("CEREBELLUM_URL",          "http://cerebellum:8003")
AUTO_APPLY       = os.getenv("AUTO_APPLY_CODE_CHANGES", "false").lower() == "true"
CODE_REVIEW_REQD = os.getenv("CODE_REVIEW_REQUIRED",    "true").lower() == "true"
WORKSPACE        = os.getenv("WORKSPACE_PATH",          "/workspace")
SERVICES_DIR     = os.getenv("SERVICES_DIR",            "/services")
GITHUB_REPO      = os.getenv("GITHUB_REPO",             "")

CHANGE_LOG     = f"{WORKSPACE}/code_changes.jsonl"
PENDING_FILE   = f"{WORKSPACE}/pending_changes.jsonl"

Path(WORKSPACE).mkdir(parents=True, exist_ok=True)

try:
    docker_client    = docker.from_env()
    DOCKER_AVAILABLE = True
except Exception:
    docker_client    = None
    DOCKER_AVAILABLE = False

# ── Verify claw is available ──────────────────────────────────────────────────
CLAW_AVAILABLE = os.path.exists(CLAW_BINARY) and os.access(CLAW_BINARY, os.X_OK)


# ── Models ────────────────────────────────────────────────────────────────────

class UpdateRequest(BaseModel):
    target:      str    # knobs | code | service | config | skill | restart | prompt
    description: str
    payload:     dict = {}
    source:      str  = "api"
    auto_apply:  bool = False


class SelfAnalysisRequest(BaseModel):
    session_id:  str
    experiences: list = []
    performance: dict = {}
    auto_apply:  bool = True


class CodeChangeRequest(BaseModel):
    service:      str
    description:  str
    change_type:  str  = "modify"   # modify | add_endpoint | add_function | rewrite
    auto_apply:   bool = False
    use_claw:     bool = True        # prefer claw over direct API call


class GenerateServiceRequest(BaseModel):
    service_name: str
    description:  str
    port:         int
    depends_on:   list = []
    auto_deploy:  bool = False


# ── CLAW.md Templates ─────────────────────────────────────────────────────────

BRAIN_ARCHITECTURE_CONTEXT = """# Brain System — Architecture Context

This is one microservice in a 19-service cognitive AI architecture.
Each service is a FastAPI app in a Docker container.

## Core Rules

All services MUST follow these patterns exactly:

### Imports
    import sys
    sys.path.insert(0, "/shared")
    from models import BrainRequest, BrainResponse
    from knobs import KnobsClient
    from events import EventBus
    from llm_provider import complete, complete_json  # NEVER call Anthropic/OpenAI directly

### App setup
    app = FastAPI(title="ServiceName", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    knobs  = KnobsClient()
    events = EventBus()
    http   = httpx.AsyncClient(timeout=30.0)

### Required endpoints (every service must have both)
    @app.get("/health")
    async def health():
        return {"status": "alive", "service": "service-name"}

    @app.get("/knobs")
    async def get_knobs():
        return knobs.get().dict()

### Standard response pattern
    @app.post("/endpoint", response_model=BrainResponse)
    async def endpoint(req: BrainRequest):
        start = time.time()
        k = knobs.get()   # read knobs on every request
        # ... logic ...
        return BrainResponse(
            request_id = req.request_id,
            service    = "service-name",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {...}
        )

### Inter-service calls (HTTP only, never import other services)
    resp = await http.post(f"{OTHER_SERVICE_URL}/endpoint", json=data)

### Error handling
    # Never let errors crash the service — return BrainResponse with success=False
    try:
        result = await do_something()
    except Exception as e:
        return BrainResponse(request_id=req.request_id, service="name",
                             success=False, error=str(e), data={})

### Background tasks
    asyncio.create_task(background_function())  # fire and forget

## What NOT to do
- Do NOT import from other services directly
- Do NOT call Anthropic/OpenAI API directly (use llm_provider.complete())
- Do NOT add unnecessary dependencies
- Do NOT use synchronous blocking calls in async endpoints
- Do NOT store state in global variables (use Redis via working_memory service)
"""


def write_claw_md(target_dir: str, service_name: str,
                   extra_context: str = "") -> str:
    """
    Write a CLAW.md instruction file to a directory.
    claw reads this before every session to understand the workspace.
    """
    content = f"""# {service_name} — Brain Service

{BRAIN_ARCHITECTURE_CONTEXT}

## This Service

Service name: {service_name}
Located at:   services/{service_name}/main.py

{extra_context}

## Verification

After making any change:
1. Check that /health and /knobs endpoints still exist
2. Check that all imports from /shared/ are correct
3. Check that the response uses BrainResponse model
4. Do NOT run cargo, npm, or any build tool — this is Python only

## Style

- Keep changes small and focused
- Match the existing code style exactly
- Add a brief comment above new functions explaining what they do
- Do not add __main__ blocks
"""
    claw_md_path = os.path.join(target_dir, "CLAW.md")
    with open(claw_md_path, "w") as f:
        f.write(content)
    return claw_md_path


# ── claw Integration ──────────────────────────────────────────────────────────

async def run_claw(prompt: str, working_dir: str,
                   timeout: int = 120) -> dict:
    """
    Invoke the claw binary in one-shot prompt mode.
    claw reads the workspace, understands context via CLAW.md,
    uses its own tool loop (file read → edit → verify), and exits.

    Returns dict with success, stdout, stderr, and any file changes.
    """
    if not CLAW_AVAILABLE:
        return {
            "success":  False,
            "error":    f"claw binary not found at {CLAW_BINARY}",
            "fallback": True
        }

    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "OPENAI_API_KEY":    os.getenv("OPENAI_API_KEY", ""),
        # Tell claw which provider to use based on our LLM_PROVIDER setting
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
    }

    # Remove empty keys — claw may behave differently with empty API keys
    env = {k: v for k, v in env.items() if v}

    try:
        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [CLAW_BINARY, "prompt", prompt],
                    cwd            = working_dir,
                    env            = env,
                    capture_output = True,
                    text           = True,
                    timeout        = timeout
                )
            ),
            timeout = timeout + 5
        )

        return {
            "success":   result.returncode == 0,
            "stdout":    result.stdout[:3000],
            "stderr":    result.stderr[:500] if result.returncode != 0 else "",
            "exit_code": result.returncode,
            "tool":      "claw",
        }

    except asyncio.TimeoutError:
        return {"success": False, "error": f"claw timed out after {timeout}s",
                "tool": "claw"}
    except Exception as e:
        return {"success": False, "error": str(e), "tool": "claw"}


async def modify_with_claw(service: str, description: str,
                            change_type: str = "modify") -> dict:
    """
    Use claw to modify an existing service file.

    Process:
    1. Copy current service file to workspace staging area
    2. Write CLAW.md with architecture context
    3. Run claw prompt — it reads file, edits, verifies
    4. Return result (queued for review if CODE_REVIEW_REQUIRED)
    """
    # Resolve service file path (mounted from docker-compose)
    service_paths = [
        Path(f"{SERVICES_DIR}/{service}/main.py"),
        Path(f"/app/../services/{service}/main.py"),
        Path(f"../services/{service}/main.py"),
    ]

    source_path = None
    for p in service_paths:
        if p.exists():
            source_path = p
            break

    if not source_path:
        # Fall back to generating without reading existing file
        return await _generate_code_direct(service, description)

    # Stage: copy service to workspace for claw to work on
    stage_dir = Path(f"{WORKSPACE}/claw-staging/{service}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_file = stage_dir / "main.py"
    shutil.copy2(source_path, stage_file)

    # Write CLAW.md
    write_claw_md(str(stage_dir), service, f"Change requested: {description}")

    # Build the claw prompt — precise instruction for one-shot mode
    prompt_map = {
        "add_endpoint":  f"Add a new FastAPI endpoint to main.py: {description}",
        "add_function":  f"Add a new function to main.py: {description}",
        "modify":        f"Modify main.py as follows: {description}",
        "rewrite":       f"Rewrite main.py to: {description}. Keep all existing endpoints.",
    }
    claw_prompt = prompt_map.get(change_type, f"Modify main.py: {description}")

    # Run claw
    result = await run_claw(claw_prompt, str(stage_dir))

    if result.get("success") and stage_file.exists():
        # Read the modified file
        modified_code = stage_file.read_text()

        change = {
            "id":           uuid.uuid4().hex[:8],
            "type":         change_type,
            "service":      service,
            "description":  description,
            "source_file":  str(source_path),
            "staged_file":  str(stage_file),
            "tool":         "claw",
            "claw_output":  result.get("stdout", "")[:500],
            "timestamp":    time.time(),
            "preview":      modified_code[:600]
        }

        if not CODE_REVIEW_REQD or AUTO_APPLY:
            # Apply directly — copy staged file back
            backup = source_path.with_suffix(f".{int(time.time())}.bak")
            shutil.copy2(source_path, backup)
            shutil.copy2(stage_file, source_path)
            _log_change({**change, "applied": True, "backup": str(backup)})
            return {
                "applied":     True,
                "tool":        "claw",
                "service":     service,
                "backup":      str(backup),
                "claw_output": result.get("stdout", "")[:500]
            }
        else:
            # Queue for human review
            _log_pending(change)
            return {
                "applied":    False,
                "change_id":  change["id"],
                "review_url": f"POST /approve/{change['id']}",
                "tool":       "claw",
                "preview":    modified_code[:600],
                "message":    "Queued for review. claw has written the code — inspect and approve."
            }
    else:
        # claw failed or binary not available — fall back to direct API
        print(f"[self-update] claw failed: {result.get('error', result.get('stderr', ''))}")
        print("[self-update] Falling back to direct API code generation")
        return await _generate_code_direct(service, description)


async def generate_service_with_claw(req: GenerateServiceRequest) -> dict:
    """
    Use claw to generate a complete new service.

    Process:
    1. Create a scaffold in workspace with the port and service name
    2. Write a minimal main.py stub and CLAW.md with full spec
    3. claw reads the stub + CLAW.md, writes the complete service
    4. Result queued for review before deployment
    """
    stage_dir = Path(f"{WORKSPACE}/claw-new-services/{req.service_name}")
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Write a minimal stub so claw has something to expand
    stub = f'''# services/{req.service_name}/main.py
# SCAFFOLD — claw will complete this

import sys
import time
sys.path.insert(0, "/shared")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus

app    = FastAPI(title="{req.service_name}", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
knobs  = KnobsClient()
events = EventBus()

@app.get("/health")
async def health():
    return {{"status": "alive", "service": "{req.service_name}"}}

@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()

# TODO: claw will add the service-specific endpoints below
'''
    (stage_dir / "main.py").write_text(stub)

    # Write CLAW.md with full service specification
    depends_str = ", ".join(req.depends_on) if req.depends_on else "none"
    extra = f"""## Service Specification

Port:       {req.port}
Depends on: {depends_str}

## What to implement

{req.description}

## Instructions

Expand main.py from the scaffold above into a complete, working service.
All endpoints must follow the BrainRequest/BrainResponse pattern.
Import environment variables with os.getenv().
Use httpx.AsyncClient for HTTP calls to other services.
The service URLs for dependencies come from environment variables:
  {chr(10).join(f"  {dep.upper().replace('-','_')}_URL = os.getenv('{dep.upper().replace('-','_')}_URL')" for dep in req.depends_on)}
"""
    write_claw_md(str(stage_dir), req.service_name, extra)

    # Write Dockerfile
    dockerfile = f"""FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential curl && rm -rf /var/lib/apt/lists/*
COPY shared /shared
COPY services/{req.service_name}/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/{req.service_name}/main.py .
EXPOSE {req.port}
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{req.port}", "--workers", "2"]
"""
    (stage_dir / "Dockerfile").write_text(dockerfile)

    # Write minimal requirements
    (stage_dir / "requirements.txt").write_text(
        "fastapi==0.111.0\nuvicorn[standard]==0.30.0\n"
        "httpx==0.27.0\npydantic==2.7.0\nredis==5.0.4\n"
    )

    # Run claw to complete the main.py
    result = await run_claw(
        f"Complete main.py to implement: {req.description}. "
        f"Expand from the scaffold. Make it production-ready. "
        f"Follow CLAW.md exactly.",
        str(stage_dir),
        timeout=180
    )

    main_py   = (stage_dir / "main.py").read_text()
    change_id = uuid.uuid4().hex[:8]

    change = {
        "id":           change_id,
        "type":         "new_service",
        "service":      req.service_name,
        "port":         req.port,
        "description":  req.description,
        "stage_dir":    str(stage_dir),
        "claw_output":  result.get("stdout", "")[:500],
        "claw_success": result.get("success", False),
        "tool":         "claw",
        "timestamp":    time.time(),
        "files": {
            "main.py":          len(main_py),
            "Dockerfile":       len(dockerfile),
            "requirements.txt": 100
        }
    }

    _log_pending(change)

    if not result.get("success"):
        print("[self-update] claw did not fully complete service — "
              "stub is in stage_dir for manual completion")

    return {
        "generated":  True,
        "deployed":   False,
        "change_id":  change_id,
        "stage_dir":  str(stage_dir),
        "tool":       "claw",
        "claw_ok":    result.get("success", False),
        "files":      change["files"],
        "review_cmd": f"POST /approve/{change_id}",
        "preview":    main_py[:800],
        "message": (
            f"Service '{req.service_name}' generated by claw. "
            f"Review at {stage_dir} then approve to deploy."
        )
    }


# ── Direct API fallback (when claw unavailable or fails) ──────────────────────

async def _generate_code_direct(service: str, description: str) -> dict:
    """
    Direct API code generation — fallback when claw is not available.
    Same as original approach: generate string, queue for review.
    """
    generated = await complete(
        prompt=(
            f"Write Python code for a FastAPI brain service called '{service}'.\n"
            f"Task: {description}\n\n"
            f"Rules:\n"
            f"- Follow the BrainRequest/BrainResponse pattern from /shared/models.py\n"
            f"- Import KnobsClient from /shared/knobs.py\n"
            f"- Never call Anthropic/OpenAI directly — use llm_provider.complete()\n"
            f"- Handle exceptions, return BrainResponse with success=False on error\n"
            f"- Write only the new function(s) or endpoint(s), not the full file\n"
            f"Python code only, no markdown:"
        ),
        system    = "Write clean FastAPI Python. No markdown. No explanations.",
        model_tier = "main",
        max_tokens = 2000
    )

    if generated.startswith("```"):
        lines = generated.split("\n")
        generated = "\n".join(l for l in lines if not l.strip().startswith("```"))

    change_id = uuid.uuid4().hex[:8]
    change = {
        "id":          change_id,
        "type":        "code_direct",
        "service":     service,
        "description": description,
        "code":        generated,
        "tool":        "direct_api",
        "timestamp":   time.time()
    }
    _log_pending(change)

    return {
        "applied":    False,
        "change_id":  change_id,
        "tool":       "direct_api",
        "preview":    generated[:600],
        "message":    "Generated via direct API (claw fallback). Review and approve."
    }


# ── Pending change management ─────────────────────────────────────────────────

def _log_change(change: dict):
    with open(CHANGE_LOG, "a") as f:
        f.write(json.dumps({**change, "logged_at": time.time()}) + "\n")


def _log_pending(change: dict):
    with open(PENDING_FILE, "a") as f:
        f.write(json.dumps(change) + "\n")


def _get_pending() -> list:
    if not os.path.exists(PENDING_FILE):
        return []
    items = []
    with open(PENDING_FILE) as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def _remove_pending(change_id: str):
    if not os.path.exists(PENDING_FILE):
        return
    with open(PENDING_FILE) as f:
        lines = f.readlines()
    with open(PENDING_FILE, "w") as f:
        for line in lines:
            try:
                c = json.loads(line)
                if c.get("id") != change_id:
                    f.write(line)
            except Exception:
                f.write(line)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":           "alive",
        "service":          "self-update",
        "version":          "3.0.0",
        "claw_available":   CLAW_AVAILABLE,
        "claw_path":        CLAW_BINARY,
        "docker_available": DOCKER_AVAILABLE,
        "auto_apply":       AUTO_APPLY,
        "code_review":      CODE_REVIEW_REQD,
        "pending_changes":  len(_get_pending())
    }


@app.get("/knobs")
async def get_knobs():
    import redis as redis_lib
    r = redis_lib.from_url(os.getenv("REDIS_URL", "redis://redis:6379"),
                            decode_responses=True)
    raw = r.get("brain:knobs:current")
    if raw:
        return json.loads(raw)
    return {}


@app.get("/claw-status")
async def claw_status():
    """Check claw binary version and availability."""
    if not CLAW_AVAILABLE:
        return {
            "available": False,
            "path":      CLAW_BINARY,
            "message":   "Binary not found. Build: cd claw-code/rust && cargo build --release"
        }
    try:
        result = subprocess.run(
            [CLAW_BINARY, "--version"],
            capture_output = True,
            text           = True,
            timeout        = 5
        )
        return {
            "available": True,
            "path":      CLAW_BINARY,
            "version":   result.stdout.strip() or result.stderr.strip()
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/webhook/openclaw")
async def openclaw_webhook(request: Request,
                            background: BackgroundTasks):
    """
    OpenClaw sends natural language instructions here.
    claw handles the code changes, not just regex interpretation.

    Examples OpenClaw can send:
      "Add endpoint /stats to thalamus showing session count and avg latency"
      "The verification service needs a cache clear endpoint"
      "Create a service that monitors Redis memory usage on port 8018"
      "Make the brain less verbose — lower verbosity knob"
    """
    body = await request.body()

    if OPENCLAW_SECRET:
        sig      = request.headers.get("X-OpenClaw-Signature", "")
        expected = "sha256=" + hmac.new(
            OPENCLAW_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(401, "Invalid signature")

    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    message = (
        data.get("message") or data.get("text") or
        data.get("content") or data.get("input") or ""
    )
    if not message:
        raise HTTPException(400, "No message in payload")

    interpretation = await _interpret_instruction(message)
    background.add_task(_execute_change, interpretation, "openclaw")

    return {
        "received":       message,
        "interpretation": interpretation,
        "tool":           "claw" if CLAW_AVAILABLE else "direct_api",
        "status":         "queued"
    }


@app.post("/update", response_model=BrainResponse)
async def update(req: UpdateRequest, background: BackgroundTasks):
    """
    Structured update — Claude Code or any HTTP client calls this directly.
    """
    start = time.time()

    if req.target == "knobs":
        result = await _update_knobs(req.payload, req.source)

    elif req.target == "code":
        result = await modify_with_claw(
            service     = req.payload.get("service", ""),
            description = req.description,
            change_type = req.payload.get("change_type", "modify")
        )

    elif req.target == "service":
        result = await generate_service_with_claw(GenerateServiceRequest(
            service_name = req.payload.get("service_name", ""),
            description  = req.description,
            port         = req.payload.get("port", 8100),
            depends_on   = req.payload.get("depends_on", []),
            auto_deploy  = req.auto_apply
        ))

    elif req.target == "skill":
        result = await _update_skill(req.payload)

    elif req.target == "restart":
        result = _restart_container(req.payload.get("service_name", ""))

    elif req.target == "prompt":
        result = await _update_system_prompt(req.payload, req.source)

    else:
        raise HTTPException(400, f"Unknown target: {req.target}")

    _log_change({"target": req.target, "source": req.source,
                 "description": req.description, "result": str(result)[:200]})
    events.publish("self.update.triggered", {
        "target": req.target, "source": req.source, "tool": "claw"
    })

    return BrainResponse(
        request_id = "update",
        service    = "self-update",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result
    )


@app.post("/modify-code")
async def modify_code(req: CodeChangeRequest):
    """
    Modify code in an existing service using claw.
    claw reads the file, understands context, makes targeted edits.
    """
    if req.use_claw and CLAW_AVAILABLE:
        return await modify_with_claw(req.service, req.description, req.change_type)
    else:
        return await _generate_code_direct(req.service, req.description)


@app.post("/generate-service")
async def generate_service(req: GenerateServiceRequest):
    """
    Generate a complete new microservice using claw.
    claw writes main.py from a scaffold guided by CLAW.md spec.
    """
    return await generate_service_with_claw(req)


@app.post("/self-analyze", response_model=BrainResponse)
async def self_analyze(req: SelfAnalysisRequest):
    """
    Analyze experiences and propose improvements.
    Auto-applies low-risk knob changes.
    Queues code changes for claw to implement after human review.
    """
    start = time.time()

    exp_summary = "\n".join([
        f"- [{e.get('experience_type','?')}] "
        f"intensity={e.get('intensity',0):.2f}: "
        f"{e.get('content','')[:100]}"
        for e in req.experiences[:25]
    ]) or "No recent experiences."

    proposal = await complete_json(
        prompt=f"""You are the self-improvement system of a cognitive AI brain.
Analyze recent experiences and propose improvements.

EXPERIENCES:
{exp_summary}

PERFORMANCE:
{json.dumps(req.performance, indent=2) if req.performance else "{}"}

Propose specific, concrete changes.

JSON response:
{{
    "analysis": "what you observe",
    "summary": "one sentence of what most needs to change",
    "proposed_changes": [
        {{
            "type": "knob_adjust|code_modify|new_service",
            "target": "knob name OR service name",
            "description": "what to change and why",
            "risk": "low|medium|high",
            "confidence": 0.0 to 1.0
        }}
    ],
    "priority": "high|medium|low"
}}""",
        system     = "Analyze AI system data. Propose improvements. JSON only.",
        model_tier = "main"
    )

    if not proposal:
        return BrainResponse(
            request_id = "analyze",
            service    = "self-update",
            success    = False,
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"error": "Analysis call failed"}
        )

    applied = []
    queued  = []

    for change in proposal.get("proposed_changes", []):
        change_type = change.get("type", "")
        risk        = change.get("risk", "medium")
        confidence  = float(change.get("confidence", 0))

        if (change_type == "knob_adjust"
                and risk == "low"
                and confidence > 0.72
                and req.auto_apply):
            try:
                resp = await http.patch(
                    f"{KNOBS_URL}/adjust",
                    params={
                        "knob_name": change["target"],
                        "value":     float(change.get("value", 0.5))
                    }
                )
                if resp.status_code == 200:
                    applied.append(change)
                    continue
            except Exception as e:
                print(f"Knob auto-apply error: {e}")

        elif (change_type == "code_modify"
              and risk == "low"
              and confidence > 0.80
              and req.auto_apply
              and CLAW_AVAILABLE):
            # Queue for claw — human still approves but claw does the writing
            service  = change.get("target", "")
            desc     = change.get("description", "")
            if service and desc:
                result = await modify_with_claw(service, desc, "modify")
                queued.append({**change, "claw_result": result})
                continue

        queued.append(change)

    events.publish("self.update.triggered", {
        "analysis":   proposal.get("analysis", ""),
        "applied":    len(applied),
        "queued":     len(queued),
        "session_id": req.session_id
    })

    return BrainResponse(
        request_id = "analyze",
        service    = "self-update",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "analysis":          proposal.get("analysis", ""),
            "summary":           proposal.get("summary", ""),
            "priority":          proposal.get("priority", "low"),
            "auto_applied":      applied,
            "queued_for_review": queued,
            "tool":              "claw" if CLAW_AVAILABLE else "direct_api"
        }
    )


@app.get("/pending")
async def get_pending():
    return {"pending": _get_pending(), "count": len(_get_pending())}


@app.post("/approve/{change_id}")
async def approve_change(change_id: str):
    """
    Approve a pending change.
    For claw-generated code: the staged file is copied to the service.
    For direct-API code: the code string is appended to the service file.
    """
    pending = _get_pending()
    change  = next((c for c in pending if c.get("id") == change_id), None)

    if not change:
        raise HTTPException(404, f"Change {change_id} not found")

    result = await _apply_approved_change(change)
    _remove_pending(change_id)
    _log_change({**change, "approved_at": time.time(), "apply_result": result})

    return {"approved": change_id, "result": result}


async def _apply_approved_change(change: dict) -> dict:
    """Apply an approved change to the live service files."""
    change_type = change.get("type", "")

    if change_type == "new_service":
        # Deploy the new service from stage_dir
        stage_dir    = Path(change.get("stage_dir", ""))
        service_name = change.get("service", "")

        if not stage_dir.exists():
            return {"error": f"Stage directory {stage_dir} no longer exists"}

        # Copy to actual services directory
        dest = Path(f"{SERVICES_DIR}/{service_name}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(stage_dir, dest)

        if change.get("auto_deploy") and DOCKER_AVAILABLE:
            return _build_and_start_container(service_name, str(dest),
                                               change.get("port", 8100))
        return {
            "deployed":  False,
            "copied_to": str(dest),
            "message":   f"Files copied. Add to docker-compose.yml and run: "
                         f"docker compose build {service_name} && "
                         f"docker compose up -d {service_name}"
        }

    elif change_type in ("modify", "add_endpoint", "add_function",
                          "code_direct", "rewrite"):
        staged    = change.get("staged_file")
        source    = change.get("source_file")
        code      = change.get("code")     # direct API fallback

        if staged and source and Path(staged).exists():
            # claw wrote the file — copy staged back to source
            backup = Path(source).with_suffix(f".{int(time.time())}.bak")
            shutil.copy2(source, backup)
            shutil.copy2(staged, source)
            service = change.get("service", "")
            _restart_container(service)
            return {
                "applied": True,
                "file":    source,
                "backup":  str(backup),
                "restarted": service
            }

        elif code and source:
            # Direct API fallback — append the generated code
            backup = Path(source).with_suffix(f".{int(time.time())}.bak")
            shutil.copy2(source, backup)
            with open(source, "a") as f:
                f.write(f"\n\n# Auto-generated {time.strftime('%Y-%m-%d %H:%M')}\n")
                f.write(code)
            service = change.get("service", "")
            _restart_container(service)
            return {
                "applied": True,
                "file":    source,
                "backup":  str(backup),
                "restarted": service
            }

        return {"error": "No staged file or code found in change record"}

    return {"error": f"Unknown change type: {change_type}"}


@app.delete("/pending/{change_id}")
async def reject_change(change_id: str):
    _remove_pending(change_id)
    _log_change({"type": "rejection", "change_id": change_id,
                 "timestamp": time.time()})
    return {"rejected": change_id}


@app.get("/change-log")
async def change_log(limit: int = 20):
    if not os.path.exists(CHANGE_LOG):
        return {"changes": []}
    with open(CHANGE_LOG) as f:
        lines = f.readlines()
    changes = []
    for line in lines[-limit:]:
        try:
            changes.append(json.loads(line))
        except Exception:
            pass
    return {"changes": list(reversed(changes))}


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _interpret_instruction(message: str) -> dict:
    """Interpret a natural language instruction into a structured action."""
    return await complete_json(
        prompt=f"""Interpret this instruction for an AI system that can:
1. Adjust behavioral knobs (0.0–1.0 values: humor, empathy, formality etc.)
2. Modify code in existing services (using claw coding agent)
3. Create entirely new microservices (claw generates complete files)
4. Restart services
5. Update system prompts

INSTRUCTION: "{message}"

Valid services: thalamus, amygdala, hippocampus, neocortex, cerebellum,
working-memory, perception, verification, cross-domain, compass,
framework-router, knobs, self-update, causal-graph, simulator,
metacognition, goals, learner, vision

JSON response:
{{
    "type": "knob_adjust|code_modify|new_service|restart|prompt_update",
    "target": "knob name OR service name",
    "description": "detailed description of the change",
    "value": "new knob value if knob_adjust",
    "port": 8100,
    "confidence": 0.0 to 1.0,
    "risk": "low|medium|high"
}}""",
        system = "Interpret AI system instructions precisely. JSON only."
    )


async def _execute_change(interpretation: dict, source: str = "api") -> dict:
    """Route an interpreted instruction to the right handler."""
    change_type = interpretation.get("type", "")
    confidence  = float(interpretation.get("confidence", 0))
    target      = interpretation.get("target", "")
    description = interpretation.get("description", "")

    if confidence < 0.45:
        _log_pending({**interpretation, "source": source,
                      "id": uuid.uuid4().hex[:8], "timestamp": time.time()})
        return {"executed": False,
                "reason": f"Confidence too low ({confidence:.2f}) — queued for review"}

    if change_type == "knob_adjust":
        return await _update_knobs(
            {"changes": {target: interpretation.get("value", 0.5)}},
            source
        )

    elif change_type == "code_modify":
        return await modify_with_claw(target, description)

    elif change_type == "new_service":
        return await generate_service_with_claw(GenerateServiceRequest(
            service_name = target,
            description  = description,
            port         = int(interpretation.get("port", 8100)),
            auto_deploy  = False
        ))

    elif change_type == "restart":
        return _restart_container(target)

    elif change_type == "prompt_update":
        return await _update_system_prompt(
            {"prompt_addition": description}, source
        )

    return {"executed": False, "reason": f"Unknown type: {change_type}"}


async def _update_knobs(payload: dict, source: str) -> dict:
    results = []
    for name, value in payload.get("changes", {}).items():
        if name is None or value is None:
            continue
        try:
            resp = await http.patch(
                f"{KNOBS_URL}/adjust",
                params={"knob_name": name, "value": float(value)}
            )
            results.append({"knob": name, "value": value,
                            "applied": resp.status_code == 200})
        except Exception as e:
            results.append({"knob": name, "error": str(e)})
    return {"target": "knobs", "results": results, "source": source}


async def _update_skill(payload: dict) -> dict:
    try:
        resp = await http.post(f"{CEREBELLUM_URL}/skills", json={
            "name":            payload.get("name", "auto_skill"),
            "trigger_pattern": payload.get("trigger_pattern", ""),
            "procedure":       payload.get("procedure", []),
            "source":          "self_update"
        })
        return {"target": "skill", "result": resp.json()}
    except Exception as e:
        return {"target": "skill", "error": str(e)}


async def _update_system_prompt(payload: dict, source: str) -> dict:
    import redis as redis_lib
    r   = redis_lib.from_url(os.getenv("REDIS_URL", "redis://redis:6379"),
                              decode_responses=True)
    add = payload.get("prompt_addition", "")
    if not add:
        return {"error": "No prompt_addition provided"}
    existing = r.get("brain:system_prompt_additions") or ""
    r.set("brain:system_prompt_additions",
          f"{existing}\n{add}" if existing else add)
    return {"updated": True, "addition": add, "source": source}


def _restart_container(service_name: str) -> dict:
    if not DOCKER_AVAILABLE:
        return {"restarted": False, "error": "Docker not available"}
    if not service_name:
        return {"restarted": False, "error": "No service name"}
    for fmt in [f"brain_{service_name}_1", f"brain-{service_name}-1", service_name]:
        try:
            c = docker_client.containers.get(fmt)
            c.restart()
            return {"restarted": True, "service": service_name, "container": fmt}
        except Exception:
            continue
    return {"restarted": False, "error": f"Container for {service_name} not found"}


def _build_and_start_container(name: str, path: str, port: int) -> dict:
    if not DOCKER_AVAILABLE:
        return {"deployed": False, "error": "Docker not available"}
    try:
        image, _ = docker_client.images.build(
            path=path, tag=f"brain/{name}:latest", rm=True
        )
        container = docker_client.containers.run(
            f"brain/{name}:latest",
            name   = f"brain_{name}_1",
            detach = True,
            ports  = {f"{port}/tcp": port},
            network = "brain_brain-net",
            restart_policy = {"Name": "unless-stopped"}
        )
        return {"deployed": True, "service": name,
                "container": container.name, "port": port}
    except Exception as e:
        return {"deployed": False, "error": str(e)}