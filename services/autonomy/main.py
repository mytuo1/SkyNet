# services/autonomy/main.py
# ── Autonomy: Self-Reflective Thinking and Continuous Improvement ────────────
#
# An autonomous thinking loop that periodically:
#   1. Observes system metrics (latencies, errors, confidence scores)
#   2. Reflects on patterns and identifies weaknesses
#   3. Proposes improvements (knob tweaks, code changes, new patterns)
#   4. Submits proposals to self-update for review/approval
#   5. Learns from outcomes to improve future proposals
#
# MODES:
#   observe  — only collect metrics and log observations
#   analyze  — observe + generate analysis of what could be better
#   improve  — analyze + generate concrete improvement proposals
#   full     — improve + auto-submit proposals to self-update
#
# RISK LEVELS (AUTONOMY_MAX_RISK):
#   low    — only knob adjustments
#   medium — knobs + small code improvements (add endpoints, fix bugs)
#   high   — can propose new services and major refactors

import asyncio
import json
import os
import sys
import time
import uuid
from enum import Enum
from typing import Optional

sys.path.insert(0, "/shared")

import redis
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_provider import complete, complete_json, has_api_key

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomy",
    version="1.0.0",
    description="Autonomous thinking loop — self-reflection and continuous improvement",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ───────────────────────────────────────────────────────────────────

r = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=True
)
http = httpx.AsyncClient(timeout=60.0)

# ── Configuration ─────────────────────────────────────────────────────────────

THALAMUS_URL    = os.getenv("THALAMUS_URL",    "http://thalamus:8000")
KNOBS_URL       = os.getenv("KNOBS_URL",       "http://knobs:8090")
SELF_UPDATE_URL = os.getenv("SELF_UPDATE_URL",  "http://self-update:8011")
MAX_RISK        = os.getenv("AUTONOMY_MAX_RISK", "low").lower()
DEFAULT_INTERVAL = 300  # 5 minutes
MIN_INTERVAL     = 60   # Rate limit: never think more often than once per minute

# Redis key prefixes
THOUGHTS_PREFIX  = "autonomy:thoughts:"
STATE_KEY        = "autonomy:state"
OUTCOMES_KEY     = "autonomy:outcomes"
METRICS_KEY      = "autonomy:metrics"
LAST_CYCLE_KEY   = "autonomy:last_cycle_ts"


# ── Enums & Models ────────────────────────────────────────────────────────────

class ThinkingMode(str, Enum):
    OBSERVE = "observe"
    ANALYZE = "analyze"
    IMPROVE = "improve"
    FULL    = "full"


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class ProposalType(str, Enum):
    KNOB_ADJUSTMENT  = "knob_adjustment"
    CODE_IMPROVEMENT = "code_improvement"
    NEW_SKILL        = "new_skill"
    NEW_PATTERN      = "new_pattern"


class ActivateRequest(BaseModel):
    mode:             ThinkingMode = ThinkingMode.OBSERVE
    interval_seconds: int          = DEFAULT_INTERVAL
    session_id:       str          = ""


class Thought(BaseModel):
    id:         str
    timestamp:  float
    phase:      str    # observe, reflect, propose, act, learn
    content:    str
    metadata:   dict   = {}


class Proposal(BaseModel):
    id:                  str
    type:                ProposalType
    description:         str
    rationale:           str
    risk_level:          RiskLevel
    implementation_plan: str
    payload:             dict = {}


# ── In-process state ──────────────────────────────────────────────────────────

_loop_task: Optional[asyncio.Task] = None
_active    = False
_mode      = ThinkingMode.OBSERVE
_interval  = DEFAULT_INTERVAL
_session   = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed_proposal_types() -> list[ProposalType]:
    """Return which proposal types are permitted under current risk ceiling."""
    if MAX_RISK == "high":
        return list(ProposalType)
    if MAX_RISK == "medium":
        return [ProposalType.KNOB_ADJUSTMENT, ProposalType.CODE_IMPROVEMENT]
    return [ProposalType.KNOB_ADJUSTMENT]


def _save_thought(thought: Thought) -> None:
    """Persist a thought to Redis."""
    key = f"{THOUGHTS_PREFIX}{thought.id}"
    r.set(key, thought.model_dump_json(), ex=86400 * 7)  # TTL 7 days
    r.lpush("autonomy:thought_ids", thought.id)
    r.ltrim("autonomy:thought_ids", 0, 499)  # keep last 500


def _make_thought(phase: str, content: str, **metadata) -> Thought:
    thought = Thought(
        id=str(uuid.uuid4()),
        timestamp=time.time(),
        phase=phase,
        content=content,
        metadata=metadata,
    )
    _save_thought(thought)
    return thought


def _get_state() -> dict:
    return {
        "active":          _active,
        "mode":            _mode.value if _active else None,
        "interval_seconds": _interval,
        "session_id":      _session,
        "max_risk":        MAX_RISK,
        "has_api_key":     has_api_key(),
    }


async def _safe_get(url: str) -> dict:
    """HTTP GET with graceful failure."""
    try:
        resp = await http.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


async def _safe_post(url: str, payload: dict) -> dict:
    """HTTP POST with graceful failure."""
    try:
        resp = await http.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# ── Phase 1: OBSERVE ─────────────────────────────────────────────────────────

async def _observe() -> dict:
    """Gather system health, knobs, and recent interaction metrics."""
    health_data, knobs_data = await asyncio.gather(
        _safe_get(f"{THALAMUS_URL}/health"),
        _safe_get(f"{KNOBS_URL}/current"),
    )

    # Pull recent metrics from Redis (latency, errors, confidence)
    recent_latencies = r.lrange("autonomy:metrics:latencies", 0, 49) or []
    recent_errors    = r.lrange("autonomy:metrics:errors", 0, 49)    or []
    recent_confs     = r.lrange("autonomy:metrics:confidences", 0, 49) or []

    observations = {
        "health":       health_data,
        "knobs":        knobs_data,
        "recent_latencies": recent_latencies[:20],
        "recent_errors":    recent_errors[:20],
        "recent_confidences": recent_confs[:20],
        "timestamp":    time.time(),
    }

    _make_thought(
        "observe",
        f"Collected system snapshot — health={bool(health_data)}, "
        f"knobs={bool(knobs_data)}, "
        f"latency_samples={len(recent_latencies)}, "
        f"error_samples={len(recent_errors)}",
        observations=observations,
    )

    return observations


# ── Phase 2: REFLECT ──────────────────────────────────────────────────────────

async def _reflect(observations: dict) -> dict:
    """Use LLM to analyze observations and identify patterns."""
    if not has_api_key():
        summary = {
            "patterns": [],
            "strengths": ["System is running"],
            "weaknesses": [],
            "overall": "No LLM key configured — reflection skipped.",
        }
        _make_thought("reflect", summary["overall"])
        return summary

    obs_text = json.dumps(observations, indent=2, default=str)
    prompt = f"""You are the self-reflection module of an AI brain system.

Analyze these system observations and identify patterns:

{obs_text}

Respond in JSON with:
{{
  "patterns":   ["pattern1", "pattern2"],
  "strengths":  ["what is working well"],
  "weaknesses": ["what needs improvement"],
  "overall":    "one-paragraph summary of system health"
}}"""

    try:
        reflection = await complete_json(
            prompt,
            system="You are an introspective AI analyzing its own performance metrics. "
                   "Be honest and specific about what you observe.",
            model_tier="fast",
        )
    except Exception:
        reflection = {
            "patterns": [],
            "strengths": [],
            "weaknesses": [],
            "overall": "Reflection failed due to LLM error.",
        }

    _make_thought(
        "reflect",
        reflection.get("overall", "Reflection complete."),
        reflection=reflection,
    )

    return reflection


# ── Phase 3: PROPOSE ──────────────────────────────────────────────────────────

async def _propose(reflection: dict) -> list[dict]:
    """Generate improvement proposals based on reflection."""
    weaknesses = reflection.get("weaknesses", [])
    if not weaknesses:
        _make_thought("propose", "No weaknesses identified — nothing to propose.")
        return []

    if not has_api_key():
        _make_thought("propose", "No LLM key — cannot generate proposals.")
        return []

    allowed = [t.value for t in _allowed_proposal_types()]

    prompt = f"""You are the self-improvement module of an AI brain system.

Based on this analysis, generate concrete improvement proposals.

Weaknesses identified:
{json.dumps(weaknesses, indent=2)}

Full reflection:
{json.dumps(reflection, indent=2)}

CONSTRAINTS:
- Allowed proposal types: {allowed}
- Max risk level: {MAX_RISK}
- Proposals must be specific, actionable, and safe
- Each proposal must include a clear implementation plan

Respond in JSON:
{{
  "proposals": [
    {{
      "type":               "knob_adjustment | code_improvement | new_skill | new_pattern",
      "description":        "what to change",
      "rationale":          "why this helps",
      "risk_level":         "low | medium | high",
      "implementation_plan": "step-by-step how to implement",
      "payload":            {{}}
    }}
  ]
}}

For knob_adjustment payloads, use: {{"knob_name": "...", "new_value": 0.XX}}
For code_improvement payloads, use: {{"service": "...", "change_type": "modify", "description": "..."}}

Only propose changes whose type is in the allowed list and whose risk_level
does not exceed "{MAX_RISK}"."""

    try:
        result = await complete_json(
            prompt,
            system="You are a careful AI engineer proposing safe, incremental improvements. "
                   "Never propose anything reckless.",
            model_tier="fast",
        )
        raw_proposals = result.get("proposals", [])
    except Exception:
        _make_thought("propose", "Proposal generation failed due to LLM error.")
        return []

    # Filter proposals that exceed risk or type constraints
    valid = []
    risk_order = {"low": 0, "medium": 1, "high": 2}
    max_risk_val = risk_order.get(MAX_RISK, 0)

    for p in raw_proposals:
        p_risk = risk_order.get(p.get("risk_level", "high"), 2)
        p_type = p.get("type", "")
        if p_risk <= max_risk_val and p_type in allowed:
            valid.append(p)

    _make_thought(
        "propose",
        f"Generated {len(valid)} valid proposals (filtered from {len(raw_proposals)} raw).",
        proposals=valid,
    )

    return valid


# ── Phase 4: ACT ─────────────────────────────────────────────────────────────

async def _act(proposals: list[dict]) -> list[dict]:
    """Submit proposals to the appropriate services."""
    results = []

    for p in proposals:
        p_type  = p.get("type", "")
        payload = p.get("payload", {})

        if p_type == "knob_adjustment":
            knob_name = payload.get("knob_name", "")
            new_value = payload.get("new_value")
            if knob_name and new_value is not None:
                resp = await _safe_post(
                    f"{KNOBS_URL}/adjust",
                    {"name": knob_name, "value": float(new_value),
                     "reason": f"[autonomy] {p.get('rationale', 'auto-tuning')}"},
                )
                outcome = {
                    "proposal": p.get("description", ""),
                    "action":   "knob_adjustment",
                    "success":  bool(resp),
                    "response": resp,
                }
            else:
                outcome = {
                    "proposal": p.get("description", ""),
                    "action":   "knob_adjustment",
                    "success":  False,
                    "error":    "Missing knob_name or new_value in payload",
                }

        elif p_type == "code_improvement":
            resp = await _safe_post(
                f"{SELF_UPDATE_URL}/update",
                {
                    "target":      "code",
                    "description": p.get("description", ""),
                    "payload":     payload,
                    "source":      "autonomy",
                    "auto_apply":  False,
                },
            )
            outcome = {
                "proposal": p.get("description", ""),
                "action":   "code_improvement",
                "success":  bool(resp),
                "response": resp,
            }

        else:
            # new_skill and new_pattern go through self-update as code changes
            resp = await _safe_post(
                f"{SELF_UPDATE_URL}/update",
                {
                    "target":      "code",
                    "description": p.get("implementation_plan", p.get("description", "")),
                    "payload":     payload,
                    "source":      "autonomy",
                    "auto_apply":  False,
                },
            )
            outcome = {
                "proposal": p.get("description", ""),
                "action":   p_type,
                "success":  bool(resp),
                "response": resp,
            }

        results.append(outcome)

    acted = sum(1 for o in results if o.get("success"))
    _make_thought(
        "act",
        f"Submitted {acted}/{len(results)} proposals successfully.",
        outcomes=results,
    )

    return results


# ── Phase 5: LEARN ────────────────────────────────────────────────────────────

async def _learn(outcomes: list[dict]) -> None:
    """Record outcomes so future cycles can learn from them."""
    for o in outcomes:
        entry = json.dumps({
            "timestamp":  time.time(),
            "proposal":   o.get("proposal", ""),
            "action":     o.get("action", ""),
            "success":    o.get("success", False),
        })
        r.lpush(OUTCOMES_KEY, entry)
        r.ltrim(OUTCOMES_KEY, 0, 199)

    _make_thought(
        "learn",
        f"Recorded {len(outcomes)} outcomes for future reference.",
    )


# ── Full Thinking Cycle ──────────────────────────────────────────────────────

async def _think_once(mode: ThinkingMode) -> dict:
    """Execute one complete thinking cycle, respecting mode and rate limit."""

    # Rate limiting
    last_ts = r.get(LAST_CYCLE_KEY)
    if last_ts is not None:
        elapsed = time.time() - float(last_ts)
        if elapsed < MIN_INTERVAL:
            return {
                "skipped":      True,
                "reason":       f"Rate limited — {MIN_INTERVAL - elapsed:.0f}s remaining",
                "last_cycle":   float(last_ts),
            }

    r.set(LAST_CYCLE_KEY, str(time.time()))

    cycle_id = str(uuid.uuid4())[:8]
    result: dict = {"cycle_id": cycle_id, "mode": mode.value, "phases": {}}

    # Phase 1: Observe (always)
    observations = await _observe()
    result["phases"]["observe"] = {"done": True}

    if mode == ThinkingMode.OBSERVE:
        return result

    # Phase 2: Reflect (analyze and above)
    reflection = await _reflect(observations)
    result["phases"]["reflect"] = {
        "done":       True,
        "weaknesses": len(reflection.get("weaknesses", [])),
        "strengths":  len(reflection.get("strengths", [])),
    }

    if mode == ThinkingMode.ANALYZE:
        return result

    # Phase 3: Propose (improve and above)
    proposals = await _propose(reflection)
    result["phases"]["propose"] = {
        "done":  True,
        "count": len(proposals),
    }

    if mode == ThinkingMode.IMPROVE:
        return result

    # Phase 4: Act (full only)
    outcomes = await _act(proposals)
    result["phases"]["act"] = {
        "done":      True,
        "submitted": sum(1 for o in outcomes if o.get("success")),
        "total":     len(outcomes),
    }

    # Phase 5: Learn (full only)
    await _learn(outcomes)
    result["phases"]["learn"] = {"done": True}

    return result


# ── Background Loop ──────────────────────────────────────────────────────────

async def _thinking_loop() -> None:
    """Continuous thinking loop running in the background."""
    global _active

    while _active:
        try:
            await _think_once(_mode)
        except Exception as exc:
            _make_thought("error", f"Thinking cycle failed: {exc}")

        # Sleep in small increments so deactivation is responsive
        for _ in range(max(1, _interval)):
            if not _active:
                break
            await asyncio.sleep(1)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "alive",
        "service": "autonomy",
        "active":  _active,
        "mode":    _mode.value if _active else None,
    }


@app.post("/activate")
async def activate(req: ActivateRequest):
    global _loop_task, _active, _mode, _interval, _session

    if _active:
        raise HTTPException(409, "Already active. Deactivate first.")

    _mode     = req.mode
    _interval = max(MIN_INTERVAL, req.interval_seconds)
    _session  = req.session_id or str(uuid.uuid4())
    _active   = True

    _loop_task = asyncio.create_task(_thinking_loop())

    state = _get_state()
    r.set(STATE_KEY, json.dumps(state))

    _make_thought("activate", f"Autonomous thinking activated — mode={_mode.value}, "
                  f"interval={_interval}s, risk_ceiling={MAX_RISK}")

    return state


@app.post("/deactivate")
async def deactivate():
    global _loop_task, _active

    if not _active:
        raise HTTPException(409, "Not active.")

    _active = False

    if _loop_task and not _loop_task.done():
        _loop_task.cancel()
        try:
            await _loop_task
        except asyncio.CancelledError:
            pass
    _loop_task = None

    state = _get_state()
    r.set(STATE_KEY, json.dumps(state))

    _make_thought("deactivate", "Autonomous thinking deactivated.")

    return state


@app.get("/status")
async def status():
    # Fetch latest thought
    last_id = r.lindex("autonomy:thought_ids", 0)
    last_thought = None
    if last_id:
        raw = r.get(f"{THOUGHTS_PREFIX}{last_id}")
        if raw:
            last_thought = json.loads(raw)

    outcome_count = r.llen(OUTCOMES_KEY) or 0

    return {
        **_get_state(),
        "last_thought":  last_thought,
        "total_outcomes": outcome_count,
    }


@app.get("/thoughts")
async def list_thoughts(limit: int = 50):
    ids = r.lrange("autonomy:thought_ids", 0, min(limit, 500) - 1) or []
    thoughts = []
    for tid in ids:
        raw = r.get(f"{THOUGHTS_PREFIX}{tid}")
        if raw:
            thoughts.append(json.loads(raw))
    return {"thoughts": thoughts, "count": len(thoughts)}


@app.post("/think-once")
async def think_once(mode: Optional[ThinkingMode] = None):
    use_mode = mode or (_mode if _active else ThinkingMode.OBSERVE)
    result = await _think_once(use_mode)
    return result
