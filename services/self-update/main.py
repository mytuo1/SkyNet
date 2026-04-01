# services/self-update/main.py
# ── Self-Update: Code Brain Autonomic Modification ───────────────────────────
# Receives instructions from OpenClaw, Claude Code, or internal self-analysis.
# Interprets natural language → structured action → applies change.
# Can adjust knobs, teach skills, reload services, update config.
# Runs self-analysis to propose its own improvements.
# Has access to Docker socket to restart individual containers.

import hashlib
import hmac
import json
import os
import re
import sys
import time

sys.path.insert(0, "/shared")

import docker
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import BrainResponse
from events import EventBus

app    = FastAPI(title="SelfUpdate", version="1.0.0",
                 description="Autonomic self-modification — knobs, skills, services")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

events = EventBus()
http   = httpx.AsyncClient(timeout=90.0)

ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY",       "")
OPENCLAW_SECRET  = os.getenv("OPENCLAW_WEBHOOK_SECRET",  "")
KNOBS_URL        = os.getenv("KNOBS_URL",               "http://knobs:8090")
CEREBELLUM_URL   = os.getenv("CEREBELLUM_URL",          "http://cerebellum:8003")
CLAUDE_MODEL     = os.getenv("FAST_CLAUDE_MODEL",       "claude-haiku-4-5-20251001")
MAIN_MODEL       = os.getenv("DEFAULT_CLAUDE_MODEL",    "claude-sonnet-4-20250514")

try:
    docker_client = docker.from_env()
    DOCKER_AVAILABLE = True
except Exception:
    docker_client    = None
    DOCKER_AVAILABLE = False

WORKSPACE = "/workspace"
os.makedirs(WORKSPACE, exist_ok=True)


# ── Models ────────────────────────────────────────────────────────────────────

class UpdateRequest(BaseModel):
    target:      str          # knobs | skill | config | service
    description: str
    payload:     dict = {}
    source:      str  = "api"


class SelfAnalysisRequest(BaseModel):
    session_id:  str
    experiences: list = []
    performance: dict = {}
    auto_apply:  bool = True


# ── Claude helper ─────────────────────────────────────────────────────────────

async def claude_json(prompt: str, system: str = "", model: str = None) -> dict:
    """Call Claude and parse JSON response."""
    if not ANTHROPIC_KEY:
        return {}
    try:
        resp = await http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json"
            },
            json={
                "model":      model or CLAUDE_MODEL,
                "max_tokens": 800,
                "system":     system or "Respond only with valid JSON. No markdown.",
                "messages":   [{"role": "user", "content": prompt}]
            }
        )
        text = resp.json()["content"][0]["text"].strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except Exception as e:
        print(f"claude_json error: {e}")
        return {}


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":           "alive",
        "service":          "self-update",
        "docker_available": DOCKER_AVAILABLE,
        "workspace":        WORKSPACE
    }


@app.post("/webhook/openclaw")
async def openclaw_webhook(request: Request):
    """
    OpenClaw sends natural language instructions here via webhook.
    Examples:
      "Make the brain more empathetic and warmer"
      "The brain seems too formal lately, dial it back"
      "Restart the verification service"
      "Set the researcher preset"
    Claude interprets the instruction → structured action → applied.
    """
    body = await request.body()

    # Verify signature if configured
    if OPENCLAW_SECRET:
        sig      = request.headers.get("X-OpenClaw-Signature", "")
        expected = hmac.new(
            OPENCLAW_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, f"sha256={expected}"):
            raise HTTPException(401, "Invalid signature")

    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    message = (
        data.get("message")
        or data.get("text")
        or data.get("content")
        or ""
    )
    if not message:
        raise HTTPException(400, "No message found in payload")

    interpretation = await _interpret_instruction(message)
    result         = await _execute_action(interpretation, source="openclaw")

    return {"received": message, "action": interpretation, "result": result}


@app.post("/update", response_model=BrainResponse)
async def update(req: UpdateRequest):
    """
    Direct structured update — used by Claude Code.
    Targets: knobs | skill | config | service
    """
    start = time.time()

    if req.target == "knobs":
        result = await _update_knobs(req.payload, req.source)

    elif req.target == "skill":
        result = await _update_skill(req.payload)

    elif req.target == "config":
        result = _update_config(req.payload)

    elif req.target == "service":
        result = _restart_service(req.payload.get("service_name", ""))

    else:
        raise HTTPException(400, f"Unknown target: '{req.target}'. "
                                  f"Valid: knobs, skill, config, service")

    events.publish("self.update.triggered", {
        "target":  req.target,
        "source":  req.source,
        "result":  result,
        "ts":      time.time()
    })

    return BrainResponse(
        request_id = "update",
        service    = "self-update",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result
    )


@app.post("/self-analyze", response_model=BrainResponse)
async def self_analyze(req: SelfAnalysisRequest):
    """
    The brain analyzes its own recent experiences and performance.
    Proposes concrete improvements. Optionally auto-applies safe ones (knobs).

    This is the autonomic self-improvement loop:
    1. Analyze experience events (what happened?)
    2. Analyze performance metrics (how did it do?)
    3. Propose specific changes
    4. Auto-apply low-risk changes (knob adjustments)
    5. Surface high-risk changes for human review
    """
    start = time.time()

    exp_summary = "\n".join([
        f"- [{e.get('experience_type', 'UNKNOWN')}] "
        f"intensity={e.get('intensity', 0):.2f}: "
        f"{e.get('content', '')[:100]}"
        for e in req.experiences[:25]
    ]) or "No recent experiences."

    perf_summary = json.dumps(req.performance, indent=2) \
        if req.performance else "No performance data."

    prompt = f"""You are the self-improvement module of a cognitive AI brain.
Analyze recent experiences and performance. Propose concrete improvements.

RECENT EXPERIENCES:
{exp_summary}

PERFORMANCE METRICS:
{perf_summary}

Based on this, propose specific changes. Be concrete and measurable.

Respond with JSON:
{{
    "analysis": "what patterns you observe",
    "proposed_changes": [
        {{
            "type": "knob_adjust | skill_learn | knowledge_fetch | architecture",
            "target": "specific knob name or thing to change",
            "value": "new value or description",
            "reasoning": "why this change would help",
            "risk": "low | medium | high",
            "confidence": 0.0 to 1.0
        }}
    ],
    "priority": "high | medium | low",
    "summary": "one sentence summary of what needs to change"
}}"""

    proposal = await claude_json(prompt, model=MAIN_MODEL)
    if not proposal:
        return BrainResponse(
            request_id = "analyze",
            service    = "self-update",
            success    = False,
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"error": "Analysis failed — no API key or call error"}
        )

    applied = []
    pending = []

    for change in proposal.get("proposed_changes", []):
        change_type = change.get("type", "")
        risk        = change.get("risk", "medium")
        confidence  = float(change.get("confidence", 0))

        # Auto-apply low-risk knob adjustments with high confidence
        if (change_type == "knob_adjust"
                and risk == "low"
                and confidence > 0.72
                and req.auto_apply):
            try:
                knob_name = change.get("target", "")
                value     = float(change.get("value", 0.5))
                resp      = await http.patch(
                    f"{KNOBS_URL}/adjust",
                    params={"knob_name": knob_name, "value": value}
                )
                if resp.status_code == 200:
                    applied.append(change)
                    continue
            except Exception as e:
                print(f"Auto-apply error: {e}")

        pending.append(change)

    events.publish("self.update.triggered", {
        "analysis": proposal.get("analysis", ""),
        "applied":  len(applied),
        "pending":  len(pending),
        "session":  req.session_id
    })

    return BrainResponse(
        request_id = "analyze",
        service    = "self-update",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "analysis":        proposal.get("analysis", ""),
            "summary":         proposal.get("summary", ""),
            "priority":        proposal.get("priority", "low"),
            "auto_applied":    applied,
            "requires_review": pending,
            "applied_count":   len(applied),
            "pending_count":   len(pending)
        }
    )


# ── Internal action handlers ──────────────────────────────────────────────────

async def _interpret_instruction(message: str) -> dict:
    """Interpret a natural language instruction into a structured action."""
    prompt = f"""Interpret this instruction for an AI cognitive system.
Convert it to a specific, structured action.

INSTRUCTION: {message}

Valid targets: knobs, skill, config, service, preset
Valid knob names: humor, honesty, empathy, formality, verbosity, creativity,
  skepticism, assertiveness, curiosity, warmth, reasoning_depth,
  analogy_seeking, verification_threshold, memory_weight, novelty_sensitivity,
  swarm_preference, ethical_weight, care_explicitness

Respond with JSON:
{{
    "target": "knobs | skill | config | service | preset",
    "action": "adjust | set | add | remove | reload | apply",
    "parameters": {{
        "knob_name": "if adjusting a knob",
        "value": 0.0,
        "preset_name": "if applying a preset",
        "service_name": "if restarting a service",
        "description": "other details"
    }},
    "confidence": 0.0 to 1.0,
    "reasoning": "why this action"
}}"""

    result = await claude_json(prompt)
    return result or {
        "target":     "unknown",
        "action":     "unknown",
        "parameters": {},
        "confidence": 0.0,
        "reasoning":  "interpretation failed"
    }


async def _execute_action(interpretation: dict, source: str = "api") -> dict:
    """Execute a structured action."""
    target     = interpretation.get("target", "")
    params     = interpretation.get("parameters", {})
    confidence = float(interpretation.get("confidence", 0))

    if confidence < 0.5:
        return {
            "executed":  False,
            "reason":    f"Confidence too low ({confidence:.2f}) to auto-execute",
            "proposed":  interpretation
        }

    if target == "knobs":
        return await _update_knobs(
            {"changes": {params.get("knob_name"): params.get("value")}},
            source
        )

    elif target == "preset":
        preset_name = params.get("preset_name", "default")
        try:
            resp = await http.post(f"{KNOBS_URL}/preset/{preset_name}")
            return {"applied_preset": preset_name, "status": resp.status_code}
        except Exception as e:
            return {"error": str(e)}

    elif target == "service":
        return _restart_service(params.get("service_name", ""))

    elif target == "skill":
        return await _update_skill(params)

    return {
        "executed": False,
        "reason":   f"Unknown target: {target}",
        "interpretation": interpretation
    }


async def _update_knobs(payload: dict, source: str) -> dict:
    """Apply knob changes via the knobs service."""
    changes = payload.get("changes", {})
    if not changes:
        return {"error": "No changes specified"}

    results = []
    for knob_name, value in changes.items():
        if knob_name is None or value is None:
            continue
        try:
            resp = await http.patch(
                f"{KNOBS_URL}/adjust",
                params={"knob_name": knob_name, "value": float(value)}
            )
            results.append({
                "knob":    knob_name,
                "value":   value,
                "applied": resp.status_code == 200,
                "source":  source
            })
        except Exception as e:
            results.append({"knob": knob_name, "error": str(e)})

    return {"target": "knobs", "results": results}


async def _update_skill(payload: dict) -> dict:
    """Teach the cerebellum a new skill."""
    try:
        resp = await http.post(
            f"{CEREBELLUM_URL}/skills",
            json={
                "name":            payload.get("name", "auto_skill"),
                "trigger_pattern": payload.get("trigger_pattern", ""),
                "procedure":       payload.get("procedure", []),
                "source":          "self_update"
            }
        )
        return {"target": "skill", "result": resp.json()}
    except Exception as e:
        return {"target": "skill", "error": str(e)}


def _update_config(payload: dict) -> dict:
    """Write a config file to the workspace."""
    filename = payload.get("file", "config.json")
    content  = payload.get("content", {})

    # Sanitize filename
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    filepath = os.path.join(WORKSPACE, filename)

    try:
        with open(filepath, "w") as f:
            if isinstance(content, dict):
                json.dump(content, f, indent=2)
            else:
                f.write(str(content))
        return {
            "target":  "config",
            "file":    filepath,
            "written": True
        }
    except Exception as e:
        return {"target": "config", "error": str(e)}


def _restart_service(service_name: str) -> dict:
    """Restart a Docker container by name."""
    if not DOCKER_AVAILABLE:
        return {
            "restarted": False,
            "reason":    "Docker not available in this environment"
        }
    if not service_name:
        return {"restarted": False, "reason": "No service name provided"}

    # Try both naming conventions
    container_names = [
        f"brain_{service_name}_1",
        f"brain-{service_name}-1",
        service_name
    ]

    for name in container_names:
        try:
            container = docker_client.containers.get(name)
            container.restart()
            return {
                "restarted":  True,
                "service":    service_name,
                "container":  name
            }
        except docker.errors.NotFound:
            continue
        except Exception as e:
            return {"restarted": False, "error": str(e)}

    return {
        "restarted": False,
        "reason":    f"Container for '{service_name}' not found"
    }
