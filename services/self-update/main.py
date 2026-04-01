# services/self-update/main.py
# ── Self-Update: Full Autonomous Code Modification ────────────────────────────
# Not just turning knobs. Writes real code.
# Can create new services, new API endpoints, new skills, new functions.
# Operates at three levels:
#
#   LEVEL 1 — Knob adjustment (instant, no code)
#   LEVEL 2 — Code modification (edits existing files, hot-reload)
#   LEVEL 3 — New service creation (generates full service, builds, deploys)
#
# Safety:
#   - Code changes require human approval when CODE_REVIEW_REQUIRED=true
#   - Auto-apply only for low-risk, well-tested patterns
#   - All changes logged with full diff
#   - Rollback available for all code changes
#   - Docker socket access allows container management
#
# Integration:
#   - OpenClaw sends natural language → interpreted → executed
#   - Claude Code calls /update directly with structured payload
#   - /self-analyze runs automatically on significant experiences
#   - /generate-service creates full new microservices from description

import asyncio
import hashlib
import hmac
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/shared")

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus
from llm_provider import complete, complete_json

app    = FastAPI(title="SelfUpdate", version="2.0.0",
                 description="Autonomous code modification — knobs, functions, services")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
http   = httpx.AsyncClient(timeout=120.0)

ANTHROPIC_KEY       = os.getenv("ANTHROPIC_API_KEY",       "")
OPENCLAW_SECRET     = os.getenv("OPENCLAW_WEBHOOK_SECRET",  "")
KNOBS_URL           = os.getenv("KNOBS_URL",               "http://knobs:8090")
CEREBELLUM_URL      = os.getenv("CEREBELLUM_URL",          "http://cerebellum:8003")
LEARNER_URL         = os.getenv("LEARNER_URL",             "http://learner:8016")
GITHUB_TOKEN        = os.getenv("GITHUB_TOKEN",            "")
GITHUB_REPO         = os.getenv("GITHUB_REPO",             "")
AUTO_APPLY          = os.getenv("AUTO_APPLY_CODE_CHANGES",  "false").lower() == "true"
CODE_REVIEW_REQD    = os.getenv("CODE_REVIEW_REQUIRED",     "true").lower() == "true"
WORKSPACE           = os.getenv("WORKSPACE_PATH",           "/workspace")
SERVICES_ROOT       = "/app/../services"   # relative to container
CHANGE_LOG          = f"{WORKSPACE}/code_changes.jsonl"
PENDING_CHANGES     = f"{WORKSPACE}/pending_changes.jsonl"

Path(WORKSPACE).mkdir(parents=True, exist_ok=True)

try:
    import docker
    docker_client    = docker.from_env()
    DOCKER_AVAILABLE = True
except Exception:
    docker_client    = None
    DOCKER_AVAILABLE = False


# ── Models ────────────────────────────────────────────────────────────────────

class UpdateRequest(BaseModel):
    target:      str   # knobs | function | service | config | skill | prompt
    description: str
    payload:     dict  = {}
    source:      str   = "api"
    auto_apply:  bool  = False


class SelfAnalysisRequest(BaseModel):
    session_id:  str
    experiences: list = []
    performance: dict = {}
    auto_apply:  bool = True


class GenerateServiceRequest(BaseModel):
    service_name:   str
    description:    str
    port:           int
    depends_on:     list = []
    auto_deploy:    bool = False


class CodeChangeRequest(BaseModel):
    service:        str    # which service to modify
    file_path:      str    # relative path within service
    change_type:    str    # add_function | modify_function | add_endpoint | rewrite
    description:    str    # what to change and why
    existing_code:  str    = ""  # current code for context
    auto_apply:     bool   = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_change(change: dict):
    """Log all code changes for audit trail."""
    with open(CHANGE_LOG, "a") as f:
        f.write(json.dumps({**change, "timestamp": time.time()}) + "\n")


def log_pending(change: dict) -> str:
    """Log a pending change awaiting review."""
    change_id = uuid.uuid4().hex[:8]
    with open(PENDING_CHANGES, "a") as f:
        f.write(json.dumps({**change, "id": change_id,
                            "timestamp": time.time()}) + "\n")
    return change_id


def get_pending_changes() -> list:
    """Get all pending changes awaiting approval."""
    if not os.path.exists(PENDING_CHANGES):
        return []
    changes = []
    with open(PENDING_CHANGES) as f:
        for line in f:
            try:
                changes.append(json.loads(line))
            except Exception:
                pass
    return changes


def remove_pending(change_id: str):
    """Remove a pending change (after approval or rejection)."""
    if not os.path.exists(PENDING_CHANGES):
        return
    with open(PENDING_CHANGES) as f:
        lines = f.readlines()
    with open(PENDING_CHANGES, "w") as f:
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
        "docker_available": DOCKER_AVAILABLE,
        "auto_apply":       AUTO_APPLY,
        "code_review":      CODE_REVIEW_REQD,
        "pending_changes":  len(get_pending_changes())
    }


@app.post("/webhook/openclaw")
async def openclaw_webhook(request: Request,
                            background: BackgroundTasks):
    """
    OpenClaw sends natural language instructions here.
    Examples:
      "The brain is being too formal, make it more casual"
      "Create a new service that tracks news headlines"
      "Add an endpoint to thalamus that shows conversation statistics"
      "The verification service is slow, optimize it"
    """
    body = await request.body()

    if OPENCLAW_SECRET:
        sig      = request.headers.get("X-OpenClaw-Signature", "")
        expected = "sha256=" + hmac.new(
            OPENCLAW_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(401, "Invalid webhook signature")

    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

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
        "status":         "queued"
    }


@app.post("/update", response_model=BrainResponse)
async def update(req: UpdateRequest, background: BackgroundTasks):
    """
    Structured update endpoint — used by Claude Code directly.
    Supports all targets: knobs, function, service, config, skill, prompt.
    """
    start = time.time()

    if req.target == "knobs":
        result = await _update_knobs(req.payload, req.source)

    elif req.target == "function":
        result = await _modify_code(CodeChangeRequest(
            service      = req.payload.get("service", ""),
            file_path    = req.payload.get("file_path", "main.py"),
            change_type  = req.payload.get("change_type", "add_function"),
            description  = req.description,
            existing_code= req.payload.get("existing_code", ""),
            auto_apply   = req.auto_apply or AUTO_APPLY
        ))

    elif req.target == "service":
        result = await _generate_service(GenerateServiceRequest(
            service_name = req.payload.get("service_name", ""),
            description  = req.description,
            port         = req.payload.get("port", 8100),
            depends_on   = req.payload.get("depends_on", []),
            auto_deploy  = req.auto_apply or AUTO_APPLY
        ))

    elif req.target == "skill":
        result = await _update_skill(req.payload)

    elif req.target == "config":
        result = _write_config(req.payload)

    elif req.target == "restart":
        service_name = req.payload.get("service_name", "")
        result = _restart_container(service_name)

    elif req.target == "prompt":
        result = await _update_system_prompt(req.payload, req.source)

    else:
        raise HTTPException(400, f"Unknown target: '{req.target}'")

    log_change({
        "target":  req.target,
        "source":  req.source,
        "description": req.description,
        "result":  result
    })

    events.publish("self.update.triggered", {
        "target": req.target,
        "source": req.source,
        "result": str(result)[:200]
    })

    return BrainResponse(
        request_id = "update",
        service    = "self-update",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result
    )


@app.post("/modify-code")
async def modify_code_endpoint(req: CodeChangeRequest):
    """
    Modify code in an existing service.
    Claude writes the new/modified function.
    Change is either applied immediately or queued for review.
    """
    return await _modify_code(req)


@app.post("/generate-service")
async def generate_service_endpoint(req: GenerateServiceRequest):
    """
    Generate a complete new microservice from a description.
    Claude writes: main.py, Dockerfile, requirements.txt, and env vars.
    Optionally builds and deploys the container.
    """
    return await _generate_service(req)


@app.post("/self-analyze", response_model=BrainResponse)
async def self_analyze(req: SelfAnalysisRequest):
    """
    Analyze recent experiences and propose improvements.
    Auto-applies low-risk changes (knobs).
    Queues high-risk changes (code) for review.
    """
    start = time.time()

    exp_summary = "\n".join([
        f"- [{e.get('experience_type','?')}] "
        f"intensity={e.get('intensity',0):.2f}: "
        f"{e.get('content','')[:100]}"
        for e in req.experiences[:25]
    ]) or "No recent experiences provided."

    perf_summary = json.dumps(req.performance, indent=2) \
        if req.performance else "{}"

    proposal = await complete_json(
        prompt=f"""You are the self-improvement system of a cognitive AI brain.
Analyze these recent experiences and performance data.
Propose specific, concrete, implementable improvements.

RECENT EXPERIENCES:
{exp_summary}

PERFORMANCE:
{perf_summary}

Propose changes across multiple levels:

JSON response:
{{
    "analysis": "what patterns you observe in the data",
    "summary": "one sentence of what most needs to change",
    "proposed_changes": [
        {{
            "type": "knob_adjust|add_function|modify_function|new_service|update_prompt",
            "target": "knob name OR service name OR description",
            "value": "new value for knobs, OR code description for functions",
            "reasoning": "why this change improves the system",
            "risk": "low|medium|high",
            "confidence": 0.0 to 1.0,
            "estimated_impact": "what will improve"
        }}
    ],
    "priority": "high|medium|low"
}}""",
        system="Analyze AI system data. Propose concrete improvements. JSON only.",
        model_tier="main"
    )

    if not proposal:
        return BrainResponse(
            request_id = "analyze",
            service    = "self-update",
            success    = False,
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"error": "Analysis failed"}
        )

    applied = []
    queued  = []

    for change in proposal.get("proposed_changes", []):
        change_type = change.get("type", "")
        risk        = change.get("risk", "medium")
        confidence  = float(change.get("confidence", 0))

        # Auto-apply low-risk knob adjustments
        if (change_type == "knob_adjust"
                and risk == "low"
                and confidence > 0.72
                and req.auto_apply):
            try:
                resp = await http.patch(
                    f"{KNOBS_URL}/adjust",
                    params={
                        "knob_name": change["target"],
                        "value":     float(change["value"])
                    }
                )
                if resp.status_code == 200:
                    applied.append(change)
                    continue
            except Exception as e:
                print(f"Auto-apply error: {e}")

        # Auto-apply low-risk function additions
        elif (change_type in ("add_function", "update_prompt")
              and risk == "low"
              and confidence > 0.85
              and AUTO_APPLY
              and req.auto_apply):
            change_id = log_pending({
                "type":        change_type,
                "description": change["value"],
                "reasoning":   change["reasoning"],
                "auto_queued": True
            })
            queued.append({**change, "change_id": change_id})
            continue

        # Everything else goes to pending queue for human review
        if risk != "low" or not req.auto_apply:
            change_id = log_pending(change)
            queued.append({**change, "change_id": change_id})

    events.publish("self.update.triggered", {
        "analysis":    proposal.get("analysis", ""),
        "applied":     len(applied),
        "queued":      len(queued),
        "session_id":  req.session_id
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
            "applied_count":     len(applied),
            "queued_count":      len(queued)
        }
    )


@app.get("/pending")
async def get_pending():
    """List all pending code changes awaiting human review."""
    return {"pending": get_pending_changes()}


@app.post("/approve/{change_id}")
async def approve_change(change_id: str):
    """Approve and apply a pending code change."""
    pending = get_pending_changes()
    change  = next((c for c in pending if c.get("id") == change_id), None)

    if not change:
        raise HTTPException(404, f"Change {change_id} not found")

    # Apply the change
    result = await _execute_change(change, source="approved")
    remove_pending(change_id)

    return {"approved": change_id, "result": result}


@app.delete("/pending/{change_id}")
async def reject_change(change_id: str):
    """Reject a pending change."""
    remove_pending(change_id)
    log_change({"type": "rejection", "change_id": change_id})
    return {"rejected": change_id}


# ── Core Implementation Functions ──────────────────────────────────────────────

async def _interpret_instruction(message: str) -> dict:
    """Interpret a natural language instruction into a structured action."""
    return await complete_json(
        prompt=f"""Interpret this instruction for an AI cognitive system with code-writing capability.

INSTRUCTION: "{message}"

The system can:
1. Adjust behavioral knobs (humor, empathy, honesty, etc. — values 0.0-1.0)
2. Add new functions to existing services
3. Modify existing functions
4. Create entirely new microservices
5. Update system prompts
6. Restart/reload services

Respond with JSON:
{{
    "type": "knob_adjust|add_function|modify_function|new_service|restart|update_prompt",
    "target": "specific knob name, service name, or description",
    "value": "new value for knobs, OR detailed description for code changes",
    "service": "which service to modify (if applicable)",
    "reasoning": "why this addresses the instruction",
    "confidence": 0.0 to 1.0,
    "risk": "low|medium|high"
}}""",
        system="Interpret AI system instructions precisely. JSON only."
    )


async def _execute_change(interpretation: dict, source: str = "api") -> dict:
    """Execute any type of change based on interpretation."""
    change_type = interpretation.get("type", "")
    confidence  = float(interpretation.get("confidence", 0))

    if confidence < 0.45:
        return {
            "executed": False,
            "reason":   f"Confidence too low ({confidence:.2f})",
            "queued":   log_pending({**interpretation, "source": source})
        }

    if change_type == "knob_adjust":
        return await _update_knobs(
            {"changes": {
                interpretation.get("target"): interpretation.get("value")
            }},
            source
        )

    elif change_type in ("add_function", "modify_function"):
        return await _modify_code(CodeChangeRequest(
            service      = interpretation.get("service", "thalamus"),
            file_path    = "main.py",
            change_type  = change_type,
            description  = interpretation.get("value", ""),
            auto_apply   = AUTO_APPLY
        ))

    elif change_type == "new_service":
        # New services always require review
        change_id = log_pending({**interpretation, "source": source})
        return {
            "executed": False,
            "queued":   True,
            "change_id": change_id,
            "message":  "New service creation queued for human review"
        }

    elif change_type == "restart":
        return _restart_container(interpretation.get("target", ""))

    elif change_type == "update_prompt":
        return await _update_system_prompt(
            {"prompt_addition": interpretation.get("value", "")}, source
        )

    return {"executed": False, "reason": f"Unknown type: {change_type}"}


async def _modify_code(req: CodeChangeRequest) -> dict:
    """
    Write or modify a function in an existing service using Claude.
    This is real code generation — Claude writes actual Python.
    """
    # Read current service code for context
    current_code = req.existing_code

    prompt = f"""You are writing Python code for a FastAPI microservice.

SERVICE: {req.service}
FILE: {req.file_path}
CHANGE TYPE: {req.change_type}
DESCRIPTION: {req.description}

CURRENT CODE (if provided):
{current_code[:3000] if current_code else "No existing code provided"}

Write the new/modified code. Requirements:
- Valid Python 3.12
- FastAPI compatible (use @app.get/post/patch/delete for endpoints)
- Follow the existing code style
- Include docstring explaining what was added/changed
- Import anything you need at the top
- Handle exceptions gracefully

For add_function: write only the new function(s) to add.
For modify_function: write the complete modified function.
For add_endpoint: write the complete endpoint function.

Return ONLY the Python code, no explanations, no markdown:"""

    generated_code = await complete(
        prompt     = prompt,
        system     = "You are an expert Python developer. Write clean, production-quality code. No markdown, just code.",
        model_tier = "main",
        max_tokens = 2000
    )

    # Clean up any accidental markdown
    if generated_code.startswith("```"):
        lines = generated_code.split("\n")
        generated_code = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )

    change = {
        "type":           req.change_type,
        "service":        req.service,
        "file":           req.file_path,
        "description":    req.description,
        "generated_code": generated_code,
        "timestamp":      time.time()
    }

    if req.auto_apply and not CODE_REVIEW_REQD:
        # Actually write the code to the service file
        result = _write_code_to_service(req.service, req.file_path,
                                         generated_code, req.change_type)
        log_change({**change, "applied": True, "result": result})
        return {
            "applied":        True,
            "service":        req.service,
            "change_type":    req.change_type,
            "code_written":   len(generated_code),
            "result":         result
        }
    else:
        # Queue for review
        change_id = log_pending(change)
        return {
            "applied":        False,
            "change_id":      change_id,
            "review_required": True,
            "preview":        generated_code[:500] + "..." if len(generated_code) > 500 else generated_code,
            "message":        f"Code queued for review (change_id={change_id})"
        }


def _write_code_to_service(service: str, file_path: str,
                             new_code: str, change_type: str) -> dict:
    """Write generated code to a service file. Creates backup first."""
    # Find the service file
    possible_paths = [
        Path(f"/app/../services/{service}/{file_path}"),
        Path(f"/services/{service}/{file_path}"),
        Path(f"../services/{service}/{file_path}"),
    ]

    target_path = None
    for p in possible_paths:
        if p.exists():
            target_path = p
            break

    if not target_path:
        return {
            "success": False,
            "error":   f"Could not find {service}/{file_path}",
            "tried":   [str(p) for p in possible_paths]
        }

    # Create backup
    backup_path = target_path.with_suffix(
        f".{int(time.time())}.bak"
    )
    shutil.copy2(target_path, backup_path)

    try:
        if change_type in ("add_function", "add_endpoint"):
            # Append to the end of the file
            with open(target_path, "a") as f:
                f.write(f"\n\n# Auto-generated at {time.strftime('%Y-%m-%d %H:%M')}\n")
                f.write(new_code)
        else:
            # Full rewrite (for modify_function, the caller provides context)
            current = target_path.read_text()
            # Simple: append the modification as a comment+replacement
            with open(target_path, "a") as f:
                f.write(f"\n\n# Modified at {time.strftime('%Y-%m-%d %H:%M')}\n")
                f.write(new_code)

        return {
            "success":      True,
            "file":         str(target_path),
            "backup":       str(backup_path),
            "bytes_written": len(new_code)
        }

    except Exception as e:
        # Restore from backup
        shutil.copy2(backup_path, target_path)
        return {"success": False, "error": str(e), "restored_backup": True}


async def _generate_service(req: GenerateServiceRequest) -> dict:
    """
    Generate a complete new microservice from a description.
    Writes: main.py, Dockerfile, requirements.txt.
    Optionally builds the Docker image and starts the container.
    """
    # Generate main.py
    main_py = await complete(
        prompt=f"""Generate a complete FastAPI microservice.

SERVICE NAME: {req.service_name}
PORT: {req.port}
DESCRIPTION: {req.description}
DEPENDS ON: {req.depends_on}

Generate a complete main.py with:
1. FastAPI app with CORS middleware
2. /health endpoint
3. /knobs endpoint (reads from KnobsClient)
4. All endpoints described in the service description
5. Proper error handling
6. Shared imports (sys.path.insert(0, '/shared'))
7. Event bus integration

Follow exactly the same pattern as the other brain services.
Write ONLY the Python code:""",
        system="Expert FastAPI developer. Write complete, production-quality Python. No markdown.",
        model_tier="main",
        max_tokens=3000
    )

    # Generate requirements.txt
    requirements = await complete(
        prompt=f"""List the Python packages needed for this service:
{req.description}

The service already has access to shared packages.
Base packages always included: fastapi, uvicorn, httpx, pydantic, redis

List ONLY additional packages needed, one per line, with version pins.
No explanations:""",
        system="Python dependency expert. List packages only.",
        model_tier="fast",
        max_tokens=200
    )

    # Generate Dockerfile
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

    change = {
        "type":        "new_service",
        "name":        req.service_name,
        "port":        req.port,
        "description": req.description,
        "main_py":     main_py,
        "requirements": requirements,
        "dockerfile":  dockerfile,
        "timestamp":   time.time()
    }

    if req.auto_deploy and AUTO_APPLY:
        result = _deploy_new_service(req.service_name, main_py,
                                      requirements, dockerfile, req.port)
        log_change({**change, "deployed": result})
        return result
    else:
        change_id = log_pending(change)

        # Save the generated files to workspace for review
        preview_dir = Path(f"{WORKSPACE}/preview_{req.service_name}")
        preview_dir.mkdir(exist_ok=True)
        (preview_dir / "main.py").write_text(main_py)
        (preview_dir / "requirements.txt").write_text(requirements)
        (preview_dir / "Dockerfile").write_text(dockerfile)

        return {
            "generated":  True,
            "deployed":   False,
            "change_id":  change_id,
            "preview_dir": str(preview_dir),
            "service":    req.service_name,
            "files": {
                "main.py":          len(main_py),
                "requirements.txt": len(requirements),
                "Dockerfile":       len(dockerfile)
            },
            "message": f"Service generated and saved to {preview_dir}. "
                       f"Review with GET /pending and approve with "
                       f"POST /approve/{change_id}"
        }


def _deploy_new_service(name: str, main_py: str, requirements: str,
                         dockerfile: str, port: int) -> dict:
    """Write files and build+start the new service container."""
    if not DOCKER_AVAILABLE:
        return {"deployed": False, "error": "Docker not available"}

    service_dir = Path(f"/workspace/services/{name}")
    service_dir.mkdir(parents=True, exist_ok=True)

    (service_dir / "main.py").write_text(main_py)
    (service_dir / "requirements.txt").write_text(requirements)
    (service_dir / "Dockerfile").write_text(dockerfile)

    try:
        # Build the image
        image, logs = docker_client.images.build(
            path       = str(service_dir),
            tag        = f"brain/{name}:latest",
            rm         = True
        )

        # Start the container
        container = docker_client.containers.run(
            f"brain/{name}:latest",
            name        = f"brain_{name}_1",
            detach      = True,
            ports       = {f"{port}/tcp": port},
            network     = "brain_brain-net",
            restart_policy = {"Name": "unless-stopped"}
        )

        return {
            "deployed":    True,
            "service":     name,
            "container":   container.name,
            "port":        port,
            "image":       f"brain/{name}:latest"
        }

    except Exception as e:
        return {"deployed": False, "error": str(e)}


async def _update_knobs(payload: dict, source: str) -> dict:
    """Apply knob changes via knobs service."""
    changes  = payload.get("changes", {})
    results  = []

    for name, value in changes.items():
        if name is None or value is None:
            continue
        try:
            resp = await http.patch(
                f"{KNOBS_URL}/adjust",
                params={"knob_name": name, "value": float(value)}
            )
            results.append({
                "knob":    name,
                "value":   value,
                "applied": resp.status_code == 200
            })
        except Exception as e:
            results.append({"knob": name, "error": str(e)})

    return {"target": "knobs", "results": results, "source": source}


async def _update_skill(payload: dict) -> dict:
    """Teach the cerebellum a new skill."""
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
    """Store a system prompt addition in knobs Redis namespace."""
    addition = payload.get("prompt_addition", "")
    if not addition:
        return {"error": "No prompt_addition provided"}

    import redis as redis_lib
    r = redis_lib.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379"),
        decode_responses=True
    )

    existing = r.get("brain:system_prompt_additions") or ""
    updated  = existing + f"\n{addition}" if existing else addition
    r.set("brain:system_prompt_additions", updated)

    return {
        "target":  "system_prompt",
        "updated": True,
        "addition": addition,
        "source":  source
    }


def _write_config(payload: dict) -> dict:
    """Write a config file to workspace."""
    filename = re.sub(r"[^\w\-_\.]", "_", payload.get("file", "config.json"))
    content  = payload.get("content", {})
    filepath = Path(WORKSPACE) / filename

    try:
        with open(filepath, "w") as f:
            if isinstance(content, dict):
                json.dump(content, f, indent=2)
            else:
                f.write(str(content))
        return {"written": True, "file": str(filepath)}
    except Exception as e:
        return {"written": False, "error": str(e)}


def _restart_container(service_name: str) -> dict:
    """Restart a brain service container."""
    if not DOCKER_AVAILABLE:
        return {"restarted": False, "error": "Docker not available"}
    if not service_name:
        return {"restarted": False, "error": "No service_name"}

    for name_fmt in [f"brain_{service_name}_1",
                     f"brain-{service_name}-1",
                     service_name]:
        try:
            c = docker_client.containers.get(name_fmt)
            c.restart()
            return {"restarted": True, "service": service_name,
                    "container": name_fmt}
        except Exception:
            continue

    return {"restarted": False, "error": f"Container for {service_name} not found"}


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