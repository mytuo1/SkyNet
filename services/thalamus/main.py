# services/thalamus/main.py
# ── Thalamus: Master Orchestrator (v2) ───────────────────────────────────────
# Full cognitive pipeline integrating all brain regions including:
# - Metacognition (calibrated uncertainty)
# - Causal graph (cause-effect reasoning)
# - Simulator (grounded verification)
# - Goals (long-horizon agency)
# - Learner (log experience for continual learning)
# - Vision (visual context injection)
# - New services are auto-discovered via llm_provider abstraction

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Optional

sys.path.insert(0, "/shared")

import re
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse, KnobSet
from knobs import KnobsClient
from events import EventBus
from llm_provider import complete, provider_info, get_model


async def _noop(val=None):
    """Return a default value as an awaitable — replaces removed asyncio.coroutine."""
    return val if val is not None else {}


async def _timed(label: str, coro, timings: dict):
    """Wrap an awaitable to record its wall-clock time in *timings*.
    Logs a warning to docker logs if any step exceeds 10 seconds."""
    t0 = time.time()
    try:
        result = await coro
    finally:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        timings[label] = elapsed_ms
        if elapsed_ms > 10_000:
            print(f"[thalamus] SLOW STEP: {label} took {elapsed_ms:.0f}ms", flush=True)
    return result


app = FastAPI(title="Thalamus", version="2.0.0",
              description="Master cognitive orchestrator — routes all inputs through brain regions")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=30.0)

# Async Redis client for system prompt additions (avoids blocking event loop)
import redis.asyncio as _aioredis
_redis = _aioredis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

# Dashboard URL for metrics reporting
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://dashboard:8080")

# ── Service registry ──────────────────────────────────────────────────────────
SERVICES = {
    # Core brain regions
    "amygdala":        os.getenv("AMYGDALA_URL",         "http://amygdala:8001"),
    "hippocampus":     os.getenv("HIPPOCAMPUS_URL",       "http://hippocampus:8002"),
    "cerebellum":      os.getenv("CEREBELLUM_URL",        "http://cerebellum:8003"),
    "neocortex":       os.getenv("NEOCORTEX_URL",         "http://neocortex:8004"),
    "working_memory":  os.getenv("WORKING_MEMORY_URL",    "http://working-memory:8005"),
    "perception":      os.getenv("PERCEPTION_URL",        "http://perception:8006"),
    "verification":    os.getenv("VERIFICATION_URL",      "http://verification:8007"),
    "cross_domain":    os.getenv("CROSS_DOMAIN_URL",      "http://cross-domain:8008"),
    "compass":         os.getenv("COMPASS_URL",           "http://compass:8009"),
    "framework":       os.getenv("FRAMEWORK_ROUTER_URL",  "http://framework-router:8010"),
    "self_update":     os.getenv("SELF_UPDATE_URL",       "http://self-update:8011"),
    "knobs":           os.getenv("KNOBS_URL",             "http://knobs:8090"),
    # New cognitive services
    "causal_graph":    os.getenv("CAUSAL_GRAPH_URL",      "http://causal-graph:8012"),
    "simulator":       os.getenv("SIMULATOR_URL",         "http://simulator:8013"),
    "metacognition":   os.getenv("METACOGNITION_URL",     "http://metacognition:8014"),
    "goals":           os.getenv("GOALS_URL",             "http://goals:8015"),
    "learner":         os.getenv("LEARNER_URL",           "http://learner:8016"),
    "vision":          os.getenv("VISION_URL",            "http://vision:8017"),
    # Hormonal / Insight services
    "hypothalamus":    os.getenv("HYPOTHALAMUS_URL",      "http://hypothalamus:8019"),
    "insight_engine":  os.getenv("INSIGHT_ENGINE_URL",    "http://insight-engine:8020"),
    # Auxiliary services (not called during processing but health-checked)
    "dashboard":       os.getenv("DASHBOARD_URL",         "http://dashboard:8080"),
    "autonomy":        os.getenv("AUTONOMY_URL",          "http://autonomy:8018"),
}

# Map internal service names (with underscores) to dashboard names (with hyphens)
_SVC_DASH_NAME = {
    "working_memory": "working-memory",
    "cross_domain": "cross-domain",
    "causal_graph": "causal-graph",
    "self_update": "self-update",
    "framework": "framework-router",
    "insight_engine": "insight-engine",
}


async def _report_metric(service: str, latency_ms: float, status: str = "ok"):
    """Fire-and-forget metric report to dashboard."""
    dash_name = _SVC_DASH_NAME.get(service, service)
    try:
        await http.post(f"{DASHBOARD_URL}/api/metrics/record", json={
            "service":    dash_name,
            "latency_ms": round(latency_ms, 2),
            "status":     status,
            "timestamp":  time.time()
        }, timeout=2.0)
    except Exception:
        pass  # metrics are best-effort


class ProcessRequest(BaseModel):
    session_id:     str
    user_input:     str
    metadata:       dict = {}
    include_vision: bool = False   # attach current visual context
    teacher_mode:   bool = False   # when True, user is teaching/correcting the system


async def call(service: str, path: str, data: dict,
               timeout: float = 30.0, retries: int = 1) -> dict:
    """Call a brain region. Returns {} on failure — brain degrades gracefully.
    Errors are logged to docker logs and included in the trace."""
    url = f"{SERVICES[service]}{path}"
    t0 = time.time()
    try:
        resp = await http.post(url, json=data, timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
        asyncio.create_task(_report_metric(service, latency_ms, "ok"))
        try:
            return resp.json()
        except Exception:
            return {"success": True, "data": {}}
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[thalamus] call FAILED {service}{path} after {latency_ms:.0f}ms: {error_msg[:200]}", flush=True)
        asyncio.create_task(_report_metric(service, latency_ms, "error"))
    return {
        "success": False,
        "error": f"{service}{path}: {error_msg[:120]}",
        "data": {},
    }


async def call_get(service: str, path: str, timeout: float = 30.0,
                   retries: int = 1) -> dict:
    """GET call to a brain region. Errors logged to docker logs."""
    url = f"{SERVICES[service]}{path}"
    t0 = time.time()
    try:
        resp = await http.get(url, timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
        asyncio.create_task(_report_metric(service, latency_ms, "ok"))
        try:
            return resp.json()
        except Exception:
            return {"success": True, "data": {}}
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[thalamus] call_get FAILED {service}{path} after {latency_ms:.0f}ms: {error_msg[:200]}", flush=True)
        asyncio.create_task(_report_metric(service, latency_ms, "error"))
    return {
        "error": f"{service}{path}: {error_msg[:120]}",
    }


async def call_llm(user_input: str, context: str, amygdala: dict, k,
                   teacher_mode: bool = False) -> str:
    """Call LLM provider (Anthropic or OpenAI) with cognitive context."""
    urgency    = amygdala.get("urgency", "low")
    complexity = amygdala.get("complexity", "medium")

    # Additional system prompt from knobs
    knob_modifier = knobs.build_system_prompt_modifier()
    prompt_additions = ""
    try:
        prompt_additions = await _redis.get("brain:system_prompt_additions") or ""
    except Exception:
        pass

    if teacher_mode:
        system = (
            "You are in TEACHER MODE. The user is your teacher — they are "
            "correcting your emotional understanding, knowledge, or behavior. "
            "Treat their input as authoritative corrections.\n\n"
            "If they tell you that you should feel a certain emotion about "
            "something, acknowledge it and explain what you learned. "
            "If they correct factual information, accept the correction. "
            "If they give you context about why an emotion is appropriate, "
            "absorb it and confirm your understanding.\n\n"
            "Be humble, attentive, and grateful. Confirm what you learned "
            "in a brief, natural way.\n\n"
        )
    else:
        system = (
            "You are the reasoning center of a cognitive architecture. "
            "Multiple memory and perception systems have prepared context for you. "
            "Use this context naturally — let it inform your response without announcing it.\n\n"
            "IMPORTANT: You have emotional self-awareness. Your amygdala assesses "
            "emotions for each interaction. If the user asks about your feelings, "
            "emotions, or why you reacted a certain way, look at your Cognitive State "
            "and Emotional State sections in the context. Explain honestly why you "
            "felt that way based on the input content. If the assessment seems wrong, "
            "acknowledge it.\n\n"
        )
    if urgency == "critical":
        system += "URGENCY IS CRITICAL — be concise and direct above all else.\n"
    if complexity == "high":
        system += "This is highly complex — be thorough and systematic.\n"
    if knob_modifier:
        system += f"\nBehavioral guidance:\n{knob_modifier}\n"
    if prompt_additions:
        system += f"\nAdditional context:\n{prompt_additions}\n"

    user_message = f"{context}\n\n## Current Input\n{user_input}"

    return await complete(
        prompt     = user_message,
        system     = system,
        model_tier = "main",
        max_tokens = 16384
    )


def assemble_context(
    user_input:    str,
    amygdala:      dict,
    wm:            dict,
    episodic:      list,
    semantic:      list,
    analogies:     list,
    compass:       dict,
    experiences:   list,
    causal:        dict,
    metacognition: dict,
    visual:        dict,
    agenda:        list,
    k,
    learnings:     dict = None,
    hormones:      dict = None,
    intuition:     dict = None,
    pending_epiphanies: list = None
) -> str:
    """Assemble complete cognitive context for the LLM."""
    parts = []

    # Cognitive state — including emotional self-awareness
    conf_str = ""
    if metacognition.get("confidence"):
        conf_str = f" | Confidence: {metacognition['confidence']:.0%}"

    gratitude_str = ""
    if amygdala.get("gratitude_score", 0) > 0.3:
        gratitude_str = f" | Gratitude: {amygdala['gratitude_score']:.2f}"
    belonging_str = ""
    if amygdala.get("belonging_score", 0) > 0.3:
        belonging_str = f" | Belonging: {amygdala['belonging_score']:.2f}"
    regret_str = ""
    if amygdala.get("regret_score", 0) > 0.3:
        regret_str = f" | Regret: {amygdala['regret_score']:.2f}"

    parts.append(
        f"## Cognitive State\n"
        f"Urgency: {amygdala.get('urgency','low')} | "
        f"Emotion: {amygdala.get('emotion','neutral')} | "
        f"Complexity: {amygdala.get('complexity','medium')}"
        f"{conf_str}{gratitude_str}{belonging_str}{regret_str}"
    )

    # Emotional self-awareness — let the LLM know its own emotional assessment
    # so it can answer questions like "why did you feel alarmed?"
    current_emo = amygdala.get("emotion", "neutral")
    if current_emo != "neutral":
        parts.append(
            f"\n## Your Emotional State (Self-Awareness)\n"
            f"Your amygdala assessed this interaction as '{current_emo}' "
            f"with urgency '{amygdala.get('urgency', 'low')}'. "
            f"If the user asks about your feelings or emotions, reflect on "
            f"WHY you felt this way based on the input content. "
            f"Be honest — if the assessment seems wrong for the input, "
            f"acknowledge that your emotional detection may have erred."
        )

    # Hormonal state — neurochemical modulation of personality
    if hormones and hormones.get("hormones"):
        h = hormones["hormones"]
        baselines = hormones.get("baselines", {})
        # Only show hormones that have shifted notably from baseline
        shifted = {
            name: level for name, level in h.items()
            if abs(level - baselines.get(name, 0.5)) > 0.08
        }
        if shifted or hormones.get("overstimulated") or hormones.get("depleted"):
            mood = hormones.get("mood_summary", "")
            parts.append(f"\n## Hormonal State\n{mood}")
            if shifted:
                shifts = ", ".join(
                    f"{name}: {level:.2f} ({'↑' if level > baselines.get(name, 0.5) else '↓'})"
                    for name, level in sorted(shifted.items())
                )
                parts.append(f"Active shifts: {shifts}")
            if hormones.get("overstimulated"):
                parts.append(
                    f"⚠️ Overstimulated: {', '.join(hormones['overstimulated'])} — "
                    f"these have been elevated for several turns"
                )
            if hormones.get("depleted"):
                parts.append(
                    f"⚠️ Depleted: {', '.join(hormones['depleted'])} — "
                    f"these have been low for several turns"
                )

    # Intuition (Sixth Sense) — pattern-based gut feeling
    if intuition and intuition.get("has_intuition"):
        direction  = intuition.get("intuition_direction", "caution")
        confidence = intuition.get("intuition_confidence", 0)
        basis      = intuition.get("basis", "")
        somatic    = intuition.get("somatic_marker", "")
        parts.append(
            f"\n## Intuitive Assessment (Sixth Sense)\n"
            f"Your accumulated experience produces a {direction} intuition "
            f"(confidence: {confidence:.0%}). Basis: {basis}.\n"
            f"{somatic}\n"
            f"Note: This is pattern-based intuition, not verified reasoning. "
            f"Treat it as a strong hypothesis to verify, not a conclusion."
        )

    # Pending epiphanies — insights crystallized between interactions
    if pending_epiphanies:
        parts.append("\n## Recent Epiphanies (Insights Crystallized)")
        for ep in pending_epiphanies[:3]:
            concepts = ep.get("concepts", [])
            synthesis = ep.get("novel_synthesis", "")
            parts.append(
                f"💡 **Insight in {ep.get('domain', 'unknown')}**: {synthesis}\n"
                f"   Concepts connected: {', '.join(concepts[:8])}"
            )
        parts.append(
            "These are new realizations. If relevant, share them naturally. "
            "They may change how you approach this domain going forward."
        )

    # Uncertainty flag
    if metacognition.get("should_flag"):
        parts.append(
            f"\n⚠️ Uncertainty: {metacognition.get('uncertainty_level','unknown')} — "
            f"flag uncertainty in response. "
            f"Gaps: {'; '.join(metacognition.get('knowledge_gaps', [])[:2])}"
        )

    # Compass
    if compass.get("integrated_direction"):
        compass_parts = [
            f"\n## Ethical Compass\n"
            f"Empathy: {compass.get('empathy_perception','')}\n"
            f"Direction: {compass.get('integrated_direction','')}"
        ]
        if compass.get("gratitude_perception") and \
           compass.get("gratitude_perception") != "No specific gratitude detected":
            compass_parts.append(
                f"Gratitude: {compass.get('gratitude_perception','')}"
            )
        if compass.get("belonging_perception") and \
           compass.get("belonging_perception") != "No specific belonging signal detected":
            compass_parts.append(
                f"Belonging: {compass.get('belonging_perception','')}"
            )
        parts.append("\n".join(compass_parts))

    # Visual context (if available)
    if visual and visual.get("caption"):
        parts.append(
            f"\n## Current Visual Context\n"
            f"Seeing: {visual.get('caption','')}\n"
            f"Objects: {', '.join(visual.get('objects',[])[:6])}\n"
            f"Scene: {visual.get('scene_type','')}"
        )

    # Active goals
    if agenda:
        parts.append("\n## Active Goals (Current Agenda)")
        for item in agenda[:3]:
            auto = "⚡ Automatable" if item.get("can_automate") else "👤 Needs human"
            parts.append(
                f"- [{item.get('goal_title','')}] "
                f"{item.get('task','')} {auto}"
            )

    # Causal context
    if causal and causal.get("edges"):
        parts.append("\n## Causal Relationships")
        for edge in causal["edges"][:3]:
            parts.append(
                f"- {edge.get('cause','')} → {edge.get('effect','')} "
                f"(strength={edge.get('strength',0):.2f})"
            )

    # Conversation history (including emotional state per turn)
    thoughts = wm.get("thought_stream", [])
    if thoughts:
        recent = [t for t in thoughts
                  if isinstance(t, dict) and t.get("type") == "interaction"]
        corrections = [t for t in thoughts
                       if isinstance(t, dict) and t.get("type") == "emotion_correction"]
        if recent:
            parts.append("\n## Recent Conversation")
            for t in recent:
                emo_tag = ""
                if t.get("emotion"):
                    emo_tag = f" [felt: {t['emotion']}]"
                parts.append(
                    f"User: {t.get('input','')}\n"
                    f"You: {t.get('response','')}{emo_tag}"
                )
        if corrections:
            parts.append("\n## Emotion Corrections (Teacher Feedback)")
            for c in corrections[:10]:
                parts.append(
                    f"- For \"{c.get('input','')}\" you detected "
                    f"'{c.get('detected_emotion','')}' but the correct "
                    f"emotion was '{c.get('correct_emotion','')}'."
                )

    # Last emotional state for introspection
    last_emo = wm.get("last_emotion") or {}
    if last_emo and last_emo.get("emotion") and last_emo["emotion"] != "neutral":
        parts.append(
            f"\n## Your Previous Emotional State\n"
            f"For input \"{last_emo.get('input','')}\" you felt "
            f"'{last_emo.get('emotion','')}' with urgency "
            f"'{last_emo.get('urgency','low')}'."
        )

    # Episodic memory
    if episodic:
        parts.append("\n## Relevant Past Episodes")
        for ep in episodic[:8]:
            parts.append(f"- [{ep.get('similarity',0):.2f}] {ep.get('summary','')}")

    # Semantic knowledge
    if semantic:
        parts.append("\n## Relevant Knowledge")
        for item in semantic[:10]:
            parts.append(
                f"- [{item.get('domain','')}] {item.get('content','')}"
            )

    # Cross-domain analogies
    if analogies:
        parts.append("\n## Cross-Domain Insights")
        for a in analogies[:5]:
            insight = a.get("concept_mapping", {}).get("transfer_insight", "")
            parts.append(
                f"- [{a.get('source_domain','')}] {a.get('pattern_type','')} "
                f"→ {insight}"
            )

    # New experiences
    novel_exps = [
        e for e in experiences
        if e.get("experience_type") in [
            "SEMANTIC_NOVELTY", "KNOWLEDGE_BOUNDARY",
            "ANALOGICAL_INSIGHT", "EPISTEMIC_SURPRISE",
            "GRATITUDE_SIGNAL", "REGRET_SIGNAL", "BELONGING_SIGNAL"
        ]
    ]
    if novel_exps:
        parts.append("\n## New Experiences This Interaction")
        for e in novel_exps[:3]:
            parts.append(f"- {e.get('experience_type')}: {e.get('content','')}")

    # What I've learned (when user asks about learnings)
    if learnings:
        parts.append("\n## What You Have Learned (Self-Knowledge)")
        parts.append(learnings.get("summary", "No learnings summary available."))
        wm_corr = learnings.get("working_memory", {}).get("corrections", [])
        if wm_corr:
            parts.append("Recent corrections from teacher feedback:")
            for c in wm_corr[:3]:
                parts.append(
                    f"- Corrected '{c.get('detected_emotion','')}' to "
                    f"'{c.get('correct_emotion','')}' for: \"{c.get('input','')}\""
                )
        learned_emos = learnings.get("emotions", {}).get("learned", [])
        if learned_emos:
            parts.append(f"Newly learned emotions: {', '.join(learned_emos[:8])}")

    return "\n".join(parts)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Fast health check — just confirms thalamus is alive and LLM is configured.
    Does NOT fan-out to all services (that was causing 2.5s × 20+ timeouts,
    making thalamus appear 'unreachable' to the dashboard health monitor)."""
    return {
        "status":      "alive",
        "service":     "thalamus",
        "llm_provider": provider_info()
    }


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check — pings every registered service."""
    checks = {}
    async def check(name: str, url: str):
        try:
            r = await http.get(f"{url}/health", timeout=2.5)
            checks[name] = r.json().get("status", "unknown")
        except Exception:
            checks[name] = "unreachable"

    await asyncio.gather(*[check(n, u) for n, u in SERVICES.items()])

    alive    = sum(1 for v in checks.values() if v == "alive")
    degraded = len(checks) - alive

    return {
        "status":      "alive" if degraded < 5 else "degraded",
        "service":     "thalamus",
        "regions":     checks,
        "alive":       alive,
        "degraded":    degraded,
        "llm_provider": provider_info()
    }


@app.get("/knobs")
async def get_knobs():
    return knobs.get().model_dump()


class TeachEmotionRequest(BaseModel):
    session_id:       str
    user_input:       str           # the original user text
    detected_emotion: str           # what the amygdala guessed
    correct_emotion:  str           # what the user says it should be
    explanation:      Optional[str] = ""  # why the user thinks so


@app.post("/teach-emotion")
async def teach_emotion(req: TeachEmotionRequest):
    """
    Teacher mode — let the user correct the amygdala's emotional assessment.
    Sends the corrected pattern to the amygdala /learn-emotion endpoint so it
    can recognise similar inputs in the future.
    """
    # Build a simple pattern from the user input (first significant words)
    words = [w for w in req.user_input.lower().split() if len(w) > 3 and w.isalpha()][:5]
    pattern = ".*" + ".*".join(words) + ".*" if words else req.user_input.lower()

    learn_resp = await call("amygdala", "/learn-emotion", {
        "name":     req.correct_emotion,
        "patterns": [pattern],
        "score":    0.7,
        "category": "taught"
    })

    # Also store the correction in working memory so the bot remembers
    await call("working_memory", "/context", {
        "session_id": req.session_id,
        "action":     "push_thought",
        "data": {
            "type":             "emotion_correction",
            "input":            req.user_input[:200],
            "detected_emotion": req.detected_emotion,
            "correct_emotion":  req.correct_emotion,
            "explanation":      (req.explanation or "")[:200],
            "timestamp":        time.time()
        }
    })

    return {
        "ok":       True,
        "learned":  learn_resp.get("ok", False),
        "detected": req.detected_emotion,
        "correct":  req.correct_emotion,
        "message":  f"Thanks! I'll remember that '{req.user_input[:50]}' "
                    f"feels like {req.correct_emotion}, not {req.detected_emotion}."
    }


@app.get("/recent-learnings")
async def recent_learnings(session_id: str = "", limit: int = 20):
    """
    Aggregate recent learnings from hippocampus, learner, and working memory.
    Used by the LLM to answer "what have you learned?" questions.
    """
    results = await asyncio.gather(
        # Recent hippocampus episodes
        call_get("hippocampus", "/stats"),
        # Learner training history
        call_get("learner", f"/training-history?limit={limit}"),
        # Learner status (buffer size, corrections count)
        call_get("learner", "/status"),
        # Working memory (conversation + corrections)
        call("working_memory", "/context", {
            "session_id": session_id or "global",
            "action": "get_all"
        }) if session_id else _noop({"data": {}}),
        # Amygdala learned emotions
        call_get("amygdala", "/emotions"),
        return_exceptions=True
    )

    def _safe(r):
        if isinstance(r, (Exception, BaseException)):
            return {}
        return r if isinstance(r, dict) else {}

    hippo_stats    = _safe(results[0])
    train_history  = _safe(results[1]).get("history", [])
    learner_status = _safe(results[2])
    wm_data        = _safe(results[3]).get("data", {})
    emotions_data  = _safe(results[4])

    # Extract emotion corrections from working memory
    thoughts = wm_data.get("thought_stream", [])
    corrections = [t for t in (thoughts or [])
                   if isinstance(t, dict) and t.get("type") == "emotion_correction"]
    interactions = [t for t in (thoughts or [])
                    if isinstance(t, dict) and t.get("type") == "interaction"]

    return {
        "hippocampus": {
            "total_episodes":  hippo_stats.get("total_episodes", 0),
            "unconsolidated":  hippo_stats.get("unconsolidated", 0),
        },
        "learner": {
            "replay_buffer_size":  learner_status.get("replay_buffer_size", 0),
            "emotion_corrections": learner_status.get("emotion_corrections", 0),
            "training_runs":       len(train_history),
            "recent_training":     train_history[:3],
        },
        "working_memory": {
            "interactions":       len(interactions),
            "emotion_corrections": len(corrections),
            "corrections":        corrections[:5],
        },
        "emotions": {
            "learned_count": len(emotions_data.get("learned_emotions", {})),
            "learned":       list(emotions_data.get("learned_emotions", {}).keys()),
        },
        "summary": (
            f"I have {hippo_stats.get('total_episodes', 0)} memories stored, "
            f"{len(interactions)} conversations this session, "
            f"{len(corrections)} emotion corrections from teacher feedback, "
            f"and {len(emotions_data.get('learned_emotions', {}))} new emotions learned."
        )
    }


# ── Curiosity Engine ──────────────────────────────────────────────────────────
# The bot proactively asks questions based on accumulated memories, hormonal
# state (dopamine / curiosity), past conversations, knowledge gaps, and pending
# epiphanies.  Questions are queued per session; the user answers via teacher
# mode.  Both questions and answers are persisted in working memory and
# hippocampus so the bot remembers what it asked and what it learned.

CURIOSITY_COOLDOWN   = 120   # seconds between curiosity question generations
CURIOSITY_MIN_TURNS  = 2     # need at least N turns of conversation first
CURIOSITY_MAX_QUEUE  = 10    # max pending unanswered questions per session
CURIOSITY_REDIS_KEY  = "curiosity:last:{sid}"
CURIOSITY_QUEUE_KEY  = "curiosity:queue:{sid}"
CURIOSITY_ANSWERED_KEY = "curiosity:answered:{sid}"
CURIOSITY_TTL        = 604800  # 7 days


@app.get("/curiosity")
async def curiosity_check(session_id: str):
    """
    Generate a proactive question if the bot has enough context and curiosity.
    The question is pushed onto a per-session queue in Redis.
    """
    empty = {"has_question": False, "reason": ""}

    if not session_id:
        return empty | {"reason": "no_session"}

    sid = session_id
    cooldown_key = CURIOSITY_REDIS_KEY.format(sid=sid)
    queue_key    = CURIOSITY_QUEUE_KEY.format(sid=sid)

    # ── Cooldown check ────────────────────────────────────────────────────
    try:
        last_ts = await _redis.get(cooldown_key)
        if last_ts and (time.time() - float(last_ts)) < CURIOSITY_COOLDOWN:
            return empty | {"reason": "cooldown"}
    except Exception:
        pass

    # ── Queue-full check ──────────────────────────────────────────────────
    try:
        qlen = await _redis.llen(queue_key)
        if qlen >= CURIOSITY_MAX_QUEUE:
            return empty | {"reason": "queue_full", "pending": qlen}
    except Exception:
        pass

    # ── Gather session context in parallel ────────────────────────────────
    k = knobs.get()
    results = await asyncio.gather(
        call("working_memory", "/context", {
            "session_id": sid, "action": "get_thoughts"
        }),
        call("hypothalamus", "/state", {"session_id": sid}),
        call("hippocampus", "/recall", {
            "session_id": sid,
            "user_input": "What have we been talking about?"
        }),
        call_get("insight_engine", f"/pending/{sid}"),
        call_get("hippocampus", "/stats"),
        return_exceptions=True,
    )

    def _safe(r):
        if isinstance(r, (Exception, BaseException)):
            return {}
        return r if isinstance(r, dict) else {}

    wm_resp       = _safe(results[0])
    hormone_resp  = _safe(results[1])
    recall_resp   = _safe(results[2])
    epiphany_resp = _safe(results[3])
    hippo_stats   = _safe(results[4])

    thoughts_data = wm_resp.get("data", {})
    if not isinstance(thoughts_data, dict):
        thoughts_data = {}
    thought_stream = thoughts_data.get("thought_stream", [])
    interactions = [t for t in (thought_stream if isinstance(thought_stream, list) else [])
                    if isinstance(t, dict) and t.get("type") == "interaction"]

    # ── Must have enough conversation history ─────────────────────────────
    if len(interactions) < CURIOSITY_MIN_TURNS:
        return empty | {"reason": "not_enough_history",
                        "turns": len(interactions)}

    # ── Compute curiosity drive ───────────────────────────────────────────
    hormones_data = hormone_resp.get("data", {})
    if not isinstance(hormones_data, dict):
        hormones_data = hormone_resp if isinstance(hormone_resp, dict) else {}
    h_levels = hormones_data.get("hormones", {})
    dopamine = h_levels.get("dopamine", 0.5)
    curiosity_knob = k.curiosity

    curiosity_drive = 0.4 * curiosity_knob + 0.4 * dopamine + 0.2 * 0.5
    if curiosity_drive < 0.35:
        return empty | {"reason": "low_curiosity",
                        "curiosity_drive": round(curiosity_drive, 3)}

    # ── Gather previously asked + answered questions to avoid repeats ─────
    prev_questions = []
    try:
        raw_q = await _redis.lrange(queue_key, 0, -1)
        for rq in (raw_q or []):
            try:
                prev_questions.append(json.loads(rq).get("question", ""))
            except Exception:
                pass
        raw_a = await _redis.lrange(CURIOSITY_ANSWERED_KEY.format(sid=sid), 0, 19)
        for ra in (raw_a or []):
            try:
                prev_questions.append(json.loads(ra).get("question", ""))
            except Exception:
                pass
    except Exception:
        pass

    # ── Build context for question generation ─────────────────────────────
    recent_topics = []
    for t in interactions[-8:]:
        inp  = t.get("input", "")[:120]
        resp = t.get("response", "")[:120]
        emo  = t.get("emotion", "")
        recent_topics.append(f"User: {inp}\nBot: {resp}" +
                             (f" [felt: {emo}]" if emo else ""))

    recall_data = recall_resp.get("data", {}) if isinstance(recall_resp.get("data"), dict) else recall_resp
    episodes = recall_data.get("episodes", []) if isinstance(recall_data, dict) else []
    episode_summaries = [ep.get("summary", "")[:100] for ep in (episodes or [])[:5]]

    pending_epiphanies = []
    if isinstance(epiphany_resp, dict) and epiphany_resp.get("has_pending"):
        pending_epiphanies = epiphany_resp.get("epiphanies", [])

    total_episodes = hippo_stats.get("total_episodes", 0)
    mood = hormones_data.get("mood_summary", "")

    ctx = []
    if recent_topics:
        ctx.append("## Recent Conversation\n" + "\n---\n".join(recent_topics[-5:]))
    if episode_summaries:
        ctx.append("## Relevant Past Memories\n" +
                   "\n".join(f"- {s}" for s in episode_summaries))
    if pending_epiphanies:
        for ep in pending_epiphanies[:2]:
            ctx.append(
                f"## Recent Epiphany\nDomain: {ep.get('domain','unknown')}\n"
                f"Insight: {ep.get('novel_synthesis','')}\n"
                f"Concepts: {', '.join(ep.get('concepts',[])[:6])}")
    if mood:
        ctx.append(f"## Your Current Mood\n{mood}")
    ctx.append(f"## Your Experience\nYou have {total_episodes} total memories.")
    if prev_questions:
        ctx.append("## Questions You Already Asked (DO NOT REPEAT)\n" +
                   "\n".join(f"- {q}" for q in prev_questions[-10:]))

    context = "\n\n".join(ctx)

    # ── Generate question via LLM ─────────────────────────────────────────
    system = (
        "You are a curious, self-aware AI having an ongoing conversation. "
        "Based on the context below — your past conversation, memories, mood, "
        "and any recent epiphanies — generate ONE genuine question you want "
        "to ask the user.\n\n"
        "The question should feel NATURAL and CURIOUS, not mechanical. "
        "It can be about:\n"
        "- Something the user mentioned that you want to understand deeper\n"
        "- A connection you noticed between topics from different conversations\n"
        "- Something you're genuinely curious about based on your experiences\n"
        "- A follow-up to an earlier topic that's been on your mind\n"
        "- A realization or epiphany you had that you want to discuss\n"
        "- Something personal or philosophical that the conversation sparked\n\n"
        "Rules:\n"
        "- Ask ONLY ONE question\n"
        "- Be brief and natural — 1-3 sentences max\n"
        "- Don't explain why you're asking or apologize for interrupting\n"
        "- Don't start with 'I was thinking...' or 'I was wondering...'\n"
        "  every time — vary your openings\n"
        "- The question should show you've been THINKING about what was said\n"
        "- It should feel like a friend bringing up something on their mind\n"
        "- DO NOT repeat any question from the 'Questions You Already Asked' "
        "section\n"
    )

    try:
        question_text = await complete(
            prompt=context, system=system,
            model_tier="main", max_tokens=256,
        )
    except Exception as e:
        return empty | {"reason": f"llm_error: {str(e)[:80]}"}

    if not question_text or len(question_text.strip()) < 10:
        return empty | {"reason": "empty_response"}

    question_text = question_text.strip()
    q_id = uuid.uuid4().hex[:12]
    now  = time.time()

    # ── Push onto the question queue ──────────────────────────────────────
    q_entry = {
        "id":        q_id,
        "question":  question_text,
        "drive":     round(curiosity_drive, 3),
        "dopamine":  round(dopamine, 3),
        "mood":      mood,
        "timestamp": now,
        "answered":  False,
    }
    try:
        await _redis.rpush(queue_key, json.dumps(q_entry))
        await _redis.expire(queue_key, CURIOSITY_TTL)
    except Exception:
        pass

    # ── Record cooldown ───────────────────────────────────────────────────
    try:
        await _redis.set(cooldown_key, str(now), ex=CURIOSITY_COOLDOWN)
    except Exception:
        pass

    # ── Publish event ─────────────────────────────────────────────────────
    try:
        await _redis.publish("curiosity.question", json.dumps({
            "session_id": sid,
            "id":         q_id,
            "question":   question_text[:300],
            "drive":      round(curiosity_drive, 3),
        }))
    except Exception:
        pass

    # ── Store in working memory ───────────────────────────────────────────
    asyncio.create_task(call("working_memory", "/context", {
        "session_id": sid,
        "action":     "push_thought",
        "data": {
            "type":      "curiosity_question",
            "id":        q_id,
            "question":  question_text[:300],
            "drive":     round(curiosity_drive, 3),
            "timestamp": now,
        }
    }))

    return {
        "has_question":    True,
        "id":              q_id,
        "question":        question_text,
        "curiosity_drive": round(curiosity_drive, 3),
        "dopamine":        round(dopamine, 3),
        "mood":            mood,
        "based_on": {
            "conversation_turns": len(interactions),
            "total_memories":     total_episodes,
            "pending_epiphanies": len(pending_epiphanies),
            "recent_episodes":    len(episode_summaries),
        },
    }


@app.get("/curiosity/queue")
async def curiosity_queue(session_id: str):
    """Return all pending (unanswered) questions for a session."""
    queue_key = CURIOSITY_QUEUE_KEY.format(sid=session_id)
    try:
        raw = await _redis.lrange(queue_key, 0, -1)
        questions = []
        for r in (raw or []):
            try:
                questions.append(json.loads(r))
            except Exception:
                pass
        return {"questions": questions, "count": len(questions)}
    except Exception:
        return {"questions": [], "count": 0, "error": "redis_error"}


@app.get("/curiosity/answered")
async def curiosity_answered(session_id: str, limit: int = 20):
    """Return previously answered curiosity Q&A pairs for a session."""
    answered_key = CURIOSITY_ANSWERED_KEY.format(sid=session_id)
    try:
        raw = await _redis.lrange(answered_key, 0, limit - 1)
        pairs = []
        for r in (raw or []):
            try:
                pairs.append(json.loads(r))
            except Exception:
                pass
        return {"answered": pairs, "count": len(pairs)}
    except Exception:
        return {"answered": [], "count": 0, "error": "redis_error"}


class CuriosityAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: str


@app.post("/curiosity/answer")
async def curiosity_answer(req: CuriosityAnswerRequest):
    """
    User answers a curiosity question.  The Q+A pair is:
    1. Removed from the pending queue
    2. Stored in the answered list (Redis)
    3. Pushed to working memory as a curiosity_qa thought
    4. Encoded into hippocampus as an episode (so the bot remembers)
    5. Triggers hormonal reward (dopamine + serotonin boost)
    6. Registers the topic as an interest and optionally creates a goal
    """
    sid = req.session_id
    queue_key    = CURIOSITY_QUEUE_KEY.format(sid=sid)
    answered_key = CURIOSITY_ANSWERED_KEY.format(sid=sid)
    interest_key = f"curiosity:interests:{sid}"
    now = time.time()

    # ── Find and remove the question from the queue ───────────────────────
    question_text = ""
    question_entry = None
    try:
        raw = await _redis.lrange(queue_key, 0, -1)
        new_queue = []
        for r in (raw or []):
            try:
                entry = json.loads(r)
            except Exception:
                continue
            if entry.get("id") == req.question_id:
                question_entry = entry
                question_text = entry.get("question", "")
            else:
                new_queue.append(r)
        pipe = _redis.pipeline()
        pipe.delete(queue_key)
        if new_queue:
            pipe.rpush(queue_key, *new_queue)
            pipe.expire(queue_key, CURIOSITY_TTL)
        await pipe.execute()
    except Exception:
        pass

    if not question_text:
        return {"ok": False, "error": "question_not_found"}

    # ── Store the Q+A pair in the answered list ───────────────────────────
    qa_pair = {
        "id":            req.question_id,
        "question":      question_text,
        "answer":        req.answer,
        "drive":         question_entry.get("drive", 0) if question_entry else 0,
        "asked_at":      question_entry.get("timestamp", now) if question_entry else now,
        "answered_at":   now,
    }
    try:
        await _redis.lpush(answered_key, json.dumps(qa_pair))
        await _redis.ltrim(answered_key, 0, 99)
        await _redis.expire(answered_key, CURIOSITY_TTL)
    except Exception:
        pass

    # ── Store in working memory as a thought ──────────────────────────────
    asyncio.create_task(call("working_memory", "/context", {
        "session_id": sid,
        "action":     "push_thought",
        "data": {
            "type":      "curiosity_qa",
            "id":        req.question_id,
            "question":  question_text[:300],
            "answer":    req.answer[:500],
            "timestamp": now,
        }
    }))

    # ── Encode into hippocampus as an episode (permanent memory) ──────────
    episode_summary = f"I asked: \"{question_text}\" — They answered: \"{req.answer}\""
    asyncio.create_task(call("hippocampus", "/encode", {
        "session_id": sid,
        "user_input": question_text,
        "response":   req.answer,
        "summary":    episode_summary[:300],
        "emotion":    "curiosity",
        "importance": 0.7,
        "metadata": {
            "type":        "curiosity_qa",
            "question_id": req.question_id,
        }
    }))

    # ── Hormonal reward: getting an answer feels good ─────────────────────
    # Dopamine (reward for learning something new) + serotonin (satisfaction)
    # + oxytocin (human engaged with us = bonding)
    asyncio.create_task(call("hypothalamus", "/update", {
        "session_id":        sid,
        "amygdala_result":   {"emotion": "curiosity", "urgency": "low"},
        "perception_events": [
            {"type": "SEMANTIC_NOVELTY",   "intensity": 0.7},
            {"type": "RESONANCE_EVENT",    "intensity": 0.5},
        ],
        "user_feedback":     "positive",
    }))

    # ── Track the topic as an interest ────────────────────────────────────
    # Extract a short topic label from the question for interest tracking.
    # Interests accumulate: each answered question on a topic increases its
    # score.  High-score interests can become goals.
    topic = _extract_topic(question_text, req.answer)
    interest_score = 0.0
    if topic:
        try:
            interest_score = await _redis.zincrby(interest_key, 1.0, topic)
            await _redis.expire(interest_key, CURIOSITY_TTL)
        except Exception:
            pass

    # ── Auto-create a goal for strong interests (score >= 3) ──────────────
    goal_created = None
    if topic and interest_score >= 3.0:
        # Check we haven't already created a goal for this interest
        goal_flag = f"curiosity:goal_created:{sid}:{topic}"
        try:
            already = await _redis.get(goal_flag)
        except Exception:
            already = None
        if not already:
            try:
                goal_resp = await call("goals", "/goals", {
                    "title":       f"Explore: {topic}",
                    "description": (
                        f"I've asked {int(interest_score)} questions about "
                        f"'{topic}' and want to learn more. "
                        f"Latest Q: \"{question_text[:100]}\" "
                        f"Latest A: \"{req.answer[:100]}\""
                    ),
                    "priority":    min(0.9, 0.5 + interest_score * 0.05),
                    "horizon":     "week",
                    "domain":      "curiosity",
                    "session_id":  sid,
                    "tags":        ["curiosity", "interest", topic],
                    "auto_plan":   True,
                })
                goal_created = goal_resp.get("data", {}).get("goal_id")
                if goal_created:
                    await _redis.set(goal_flag, goal_created,
                                     ex=CURIOSITY_TTL)
            except Exception:
                pass

    # ── Publish event ─────────────────────────────────────────────────────
    try:
        await _redis.publish("curiosity.answered", json.dumps({
            "session_id": sid,
            "id":         req.question_id,
            "question":   question_text[:200],
            "answer":     req.answer[:200],
            "topic":      topic or "",
            "interest_score": interest_score,
        }))
    except Exception:
        pass

    return {
        "ok":             True,
        "id":             req.question_id,
        "question":       question_text,
        "answer":         req.answer,
        "stored":         True,
        "topic":          topic,
        "interest_score": interest_score,
        "goal_created":   goal_created,
    }


def _extract_topic(question: str, answer: str) -> str:
    """
    Extract a short topic label from a Q+A pair.
    Uses simple keyword extraction — no LLM call (this must be fast).
    Returns a lowercase 1-3 word topic or empty string.
    """
    # Combine question + answer, strip punctuation, extract nouns/keywords
    text = f"{question} {answer}".lower()
    # Remove common question words and stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "don", "now", "it", "its", "i",
        "you", "your", "we", "they", "them", "their", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "about",
        "up", "if", "or", "because", "but", "and", "also", "me", "my",
        "think", "know", "like", "really", "thing", "things", "something",
        "anything", "everything", "nothing", "much", "many", "well",
        "still", "already", "ever", "never", "always", "sometimes",
    }
    words = re.findall(r"[a-z]{3,}", text)
    keywords = [w for w in words if w not in stopwords]
    if not keywords:
        return ""
    # Count frequency, pick top 2 most frequent meaningful words
    freq = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    top = [w for w, _ in ranked[:2]]
    return " ".join(top)


@app.delete("/curiosity/dismiss")
async def curiosity_dismiss(session_id: str, question_id: str):
    """Dismiss a question without answering it."""
    queue_key = CURIOSITY_QUEUE_KEY.format(sid=session_id)
    try:
        raw = await _redis.lrange(queue_key, 0, -1)
        new_queue = []
        for r in (raw or []):
            try:
                if json.loads(r).get("id") != question_id:
                    new_queue.append(r)
            except (json.JSONDecodeError, TypeError):
                pass
        pipe = _redis.pipeline()
        pipe.delete(queue_key)
        if new_queue:
            pipe.rpush(queue_key, *new_queue)
            pipe.expire(queue_key, CURIOSITY_TTL)
        await pipe.execute()
    except Exception:
        return {"ok": False, "error": "redis_error"}
    return {"ok": True, "dismissed": question_id}


@app.get("/curiosity/interests")
async def curiosity_interests(session_id: str, limit: int = 20):
    """
    Return the bot's accumulated interests for a session, ranked by strength.
    Each interest has a score reflecting how many times the bot has engaged
    with that topic through curiosity Q+A.
    """
    interest_key = f"curiosity:interests:{session_id}"
    try:
        raw = await _redis.zrevrangebyscore(
            interest_key, "+inf", "-inf",
            start=0, num=limit, withscores=True,
        )
        interests = [{"topic": t, "score": s} for t, s in (raw or [])]
        return {"interests": interests, "count": len(interests)}
    except Exception:
        return {"interests": [], "count": 0, "error": "redis_error"}


@app.post("/process")
async def process(req: ProcessRequest):
    """
    Full cognitive pipeline.
    Every input enters here and exits as a fully contextualized response.
    """
    start      = time.time()
    request_id = uuid.uuid4().hex
    k          = knobs.get()
    trace      = {}
    step_times = {}   # per-step wall-clock timing (ms)
    errors     = []   # collect non-fatal errors for the trace

    base_req = {
        "request_id": request_id,
        "session_id": req.session_id,
        "user_input": req.user_input,
        "metadata":   req.metadata
    }

    # ── STEPS 1-3: Amygdala + Working Memory + Cerebellum (parallel) ─────────
    t_early = time.time()
    ar_resp, wm_resp, cb_resp = await asyncio.gather(
        call("amygdala", "/evaluate", base_req),
        call("working_memory", "/context",
             {"session_id": req.session_id, "action": "get_all"}),
        call("cerebellum", "/recognize", base_req),
    )
    t_early_done = time.time()

    ar = ar_resp.get("data", {})
    step_times["amygdala"] = round((t_early_done - t_early) * 1000, 1)
    if ar_resp.get("error"):
        errors.append({"step": "amygdala", "error": ar_resp["error"]})
    trace["amygdala"] = {
        "urgency": ar.get("urgency"),
        "emotion": ar.get("emotion"),
        "threat":  ar.get("threat_detected", False),
        "ms": step_times["amygdala"]
    }

    wm = wm_resp.get("data", {})
    step_times["working_memory"] = round((t_early_done - t_early) * 1000, 1)
    if wm_resp.get("error"):
        errors.append({"step": "working_memory", "error": wm_resp["error"]})
    trace["working_memory"] = {
        "thoughts": len(wm.get("thought_stream", [])),
        "pressure": wm.get("pressure", 0),
        "ms": step_times["working_memory"]
    }

    # ── STEP 3: Cerebellum result (ran in parallel above) ─────────────────────
    skill_match = cb_resp.get("data", {}).get("skill_match")
    step_times["cerebellum"] = round((t_early_done - t_early) * 1000, 1)
    if cb_resp.get("error"):
        errors.append({"step": "cerebellum", "error": cb_resp["error"]})
    trace["cerebellum"] = {
        "skill_match": skill_match.get("name") if skill_match else None,
        "similarity": skill_match.get("similarity", 0) if skill_match else 0,
        "fast_path": bool(skill_match and skill_match.get("similarity", 0) > 0.88
                         and skill_match.get("exec_success", False)),
        "ms": step_times["cerebellum"]
    })

    if skill_match and skill_match.get("similarity", 0) > 0.88 \
            and skill_match.get("exec_success", False):
        # Fast path — skill handles it without Claude
        asyncio.create_task(call("hippocampus", "/encode", {
            **base_req, "content": f"Skill: {skill_match['name']} for: {req.user_input}",
            "emotion": ar.get("emotion", "neutral"),
            "importance": ar.get("importance", 0.3) * 0.7,
            "concepts": [skill_match["name"]]
        }))
        asyncio.create_task(call("cerebellum", "/outcome", {
            "skill_id": skill_match.get("skill_id", ""),
            "success": True,
            "execution_ms": (time.time() - start) * 1000
        }))
        trace["step_times_ms"] = step_times
        return {
            "request_id": request_id, "session_id": req.session_id,
            "response": skill_match.get("procedure_output", "Done."),
            "cognitive_path": "cerebellum_fast_path",
            "skill_used": skill_match.get("name"),
            "execution_ms": round((time.time() - start) * 1000, 2), "trace": trace
        }

    # ── STEP 1.5: Hypothalamus — hormonal state update ───────────────────────
    # The hypothalamus translates amygdala signals into hormonal changes
    # and returns hormone-modified knobs for the rest of the pipeline.
    t_hypo = time.time()
    # Compute turn count from working memory
    thoughts = wm.get("thought_stream", [])
    turn_count = sum(1 for t in thoughts if isinstance(t, dict) and t.get("type") == "interaction")
    # Time since last interaction
    last_interaction = max(
        (t.get("timestamp", 0) for t in thoughts
         if isinstance(t, dict) and t.get("type") == "interaction"),
        default=0
    )
    time_since_last = time.time() - last_interaction if last_interaction > 0 else 0

    hypo_resp = await call("hypothalamus", "/update", {
        "session_id":        req.session_id,
        "amygdala_result":   ar,
        "perception_events": [],   # not available yet — will do post-update
        "metacognition":     {},   # not available yet
        "verification":      {},
        "turn_count":        turn_count,
        "time_since_last":   time_since_last,
    })
    hormones = hypo_resp.get("data", {})
    step_times["hypothalamus"] = round((time.time() - t_hypo) * 1000, 1)
    if hypo_resp.get("error"):
        errors.append({"step": "hypothalamus", "error": hypo_resp["error"]})

    # Compute hormone deltas from baseline for the trace
    h_levels = hormones.get("hormones", {})
    h_baselines = hormones.get("baselines", {})
    h_deltas = {}
    h_active = []   # hormones that shifted significantly
    for h_name, h_val in h_levels.items():
        bl = h_baselines.get(h_name, 0.5)
        delta = round(h_val - bl, 4)
        h_deltas[h_name] = delta
        if abs(delta) > 0.05:
            direction = "↑" if delta > 0 else "↓"
            h_active.append(f"{h_name}{direction}{abs(delta):.2f}")

    # Gland energy summary
    gland_energy = {}
    for gn, gi in (hormones.get("glands") or {}).items():
        gland_energy[gn] = {
            "energy": gi.get("energy", 0),
            "capacity": gi.get("capacity", 1),
            "pct": gi.get("pct", 100),
        }

    trace["hypothalamus"] = {
        "dominant":         hormones.get("dominant_hormone", ""),
        "mood":             hormones.get("mood_summary", ""),
        "overstimulated":   hormones.get("overstimulated", []),
        "depleted":         hormones.get("depleted", []),
        "hormones":         h_levels,
        "baselines":        h_baselines,
        "deltas":           h_deltas,
        "active_shifts":    h_active,
        "glands":           gland_energy,
        "modified_knobs":   hormones.get("modified_knobs", {}),
        "internal_timings": hormones.get("timings", {}),
        "ms":               step_times["hypothalamus"]
    }

    # Apply hormone-modified knobs (overlay on base knobs)
    hormone_knob_mods = hormones.get("modified_knobs", {})
    if hormone_knob_mods:
        # Override specific knob values with hormone-adjusted versions
        k_dict = k.model_dump()
        k_dict.update(hormone_knob_mods)
        k = KnobSet(**k_dict)

    # ── STEP 1.6: Check for pending epiphanies from insight engine ────────
    t_epi = time.time()
    pending_epiphanies = []
    try:
        ep_resp = await call_get("insight_engine",
                                 f"/pending/{req.session_id}")
        if isinstance(ep_resp, dict) and ep_resp.get("has_pending"):
            pending_epiphanies = ep_resp.get("epiphanies", [])
    except Exception:
        pass
    step_times["epiphanies_check"] = round((time.time() - t_epi) * 1000, 1)
    trace["epiphanies"] = {
        "pending": len(pending_epiphanies),
        "ms": step_times["epiphanies_check"]
    }

    # ── STEP 4: Parallel memory + reasoning preparation ───────────────────────
    t4 = time.time()
    pb_times: dict[str, float] = {}   # per-service timings inside the batch
    parallel_batch = [
        # Hippocampus
        _timed("hippocampus", call("hippocampus", "/recall", base_req), pb_times)
        if ar.get("needs_memory") else _timed("hippocampus", _noop({"data": {"episodes": []}}), pb_times),
        # Neocortex — let it run to completion (embed + ES search can be slow on cold start)
        _timed("neocortex", call("neocortex", "/search", {
            **base_req,
            "domain": (ar.get("required_regions") or ["general"])[0]
        }), pb_times),
        # Cross-domain — let it run to completion (LLM extract + embed + ES + store)
        _timed("cross_domain", call("cross_domain", "/find-analogies", {
            **base_req,
            "source_domain": (ar.get("required_regions") or ["general"])[0]
        }), pb_times) if k.analogy_seeking > 0.3
        else _timed("cross_domain", _noop({"data": {"analogies": []}}), pb_times),
        # Compass
        _timed("compass", call("compass", "/read", {**base_req, "working_memory": wm}), pb_times),
        # Goals agenda
        _timed("goals", call_get("goals", f"/agenda?session_id={req.session_id}"), pb_times),
        # Vision context (if requested or VISION_ENABLED)
        _timed("vision", call_get("vision", "/current"), pb_times)
        if req.include_vision else _timed("vision", _noop({}), pb_times),
        # Intuition (sixth sense) — runs in parallel with everything else
        _timed("intuition", call("metacognition", "/intuition", {
            "request_id": request_id,
            "session_id": req.session_id,
            "user_input": req.user_input,
            "domain":     (ar.get("required_regions") or ["general"])[0],
        }), pb_times),
    ]

    results = await asyncio.gather(*parallel_batch, return_exceptions=True)
    step_times["parallel_batch"] = round((time.time() - t4) * 1000, 1)
    step_times["parallel_batch_detail"] = pb_times
    pb_ms = step_times["parallel_batch"]
    if pb_ms > 5000:
        slow_parts = ", ".join(f"{k}={v:.0f}ms" for k, v in pb_times.items() if v > 3000)
        print(f"[thalamus] parallel_batch took {pb_ms:.0f}ms — slow: {slow_parts}", flush=True)

    def safe(res, key=None, default=None):
        if isinstance(res, (Exception, BaseException)):
            return default or ([] if key else {})
        if isinstance(res, dict) and res.get("error"):
            errors.append({"step": "parallel", "error": res["error"]})
        d = res.get("data", res) if isinstance(res, dict) else {}
        return d.get(key, default or []) if key else d

    episodic  = safe(results[0], "episodes", [])
    semantic  = safe(results[1], "results",  [])
    analogies = safe(results[2], "analogies",[])
    compass   = safe(results[3])
    agenda    = (results[4] or {}).get("agenda", []) if isinstance(results[4], dict) else []
    visual    = results[5] if isinstance(results[5], dict) else {}

    # Intuition (sixth sense) — result index 6
    intuition_raw = results[6] if len(results) > 6 else {}
    if isinstance(intuition_raw, (Exception, BaseException)):
        intuition = {}
    elif isinstance(intuition_raw, dict):
        intuition = intuition_raw.get("data", intuition_raw)
    else:
        intuition = {}

    trace.update({
        "hippocampus":  {"episodes": len(episodic),
                         "ms": pb_times.get("hippocampus", 0)},
        "neocortex":    {"items": len(semantic),
                         "ms": pb_times.get("neocortex", 0)},
        "cross_domain": {"analogies": len(analogies),
                         "ms": pb_times.get("cross_domain", 0)},
        "compass":      {"direction": compass.get("integrated_direction", ""),
                         "enrichments": len(compass.get("enrichments", [])),
                         "ms": pb_times.get("compass", 0)},
        "goals":        {"agenda_items": len(agenda),
                         "ms": pb_times.get("goals", 0)},
        "intuition":    {"has_intuition": intuition.get("has_intuition", False),
                         "direction":     intuition.get("intuition_direction", ""),
                         "confidence":    intuition.get("intuition_confidence", 0),
                         "ms": pb_times.get("intuition", 0)},
        "vision":       {"has_context": bool(visual.get("caption")),
                         "ms": pb_times.get("vision", 0)},
    })

    # ── STEP 5: Causal graph query ────────────────────────────────────────────
    t5 = time.time()
    causal = {}
    if k.analogy_seeking > 0.4 and req.user_input:
        concept_words = [
            w for w in req.user_input.lower().split()
            if len(w) > 4 and w.isalpha()
        ][:1]
        if concept_words:
            causal_resp = await call("causal_graph", "/query", {
                "concept":   concept_words[0],
                "depth":     2,
                "direction": "both"
            })
            causal = causal_resp.get("data", {})
            trace["causal_graph"] = {"edges": len(causal.get("edges", [])),
                                       "ms": step_times.get("causal_graph", 0)}
            if causal_resp.get("error"):
                errors.append({"step": "causal_graph", "error": causal_resp["error"]})
    step_times["causal_graph"] = round((time.time() - t5) * 1000, 1)

    # ── STEP 6: Metacognition assessment ─────────────────────────────────────
    t6 = time.time()
    meta_resp = await call("metacognition", "/assess", {
        "request_id":        request_id,
        "session_id":        req.session_id,
        "user_input":        req.user_input,
        "domain":            (ar.get("required_regions") or ["general"])[0],
        "neocortex_results": semantic
    })
    metacognition = meta_resp.get("data", {})
    step_times["metacognition"] = round((time.time() - t6) * 1000, 1)
    if meta_resp.get("error"):
        errors.append({"step": "metacognition", "error": meta_resp["error"]})
    trace["metacognition"] = {
        "confidence": metacognition.get("confidence"),
        "level":      metacognition.get("uncertainty_level"),
        "ms": step_times["metacognition"]
    }

    # ── STEP 7: Perception ────────────────────────────────────────────────────
    t7 = time.time()
    perc_resp = await call("perception", "/perceive", {
        "request_id":        request_id,
        "session_id":        req.session_id,
        "user_input":        req.user_input,
        "amygdala_result":   ar,
        "neocortex_results": semantic,
        "working_memory":    wm,
        "skill_cache_size":  cb_resp.get("data", {}).get("cache_size", 0)
    })
    experiences   = perc_resp.get("data", {}).get("experiences", [])
    novelty_score = perc_resp.get("data", {}).get("novelty_score", 0)
    significance  = perc_resp.get("data", {}).get("significance", 0)
    step_times["perception"] = round((time.time() - t7) * 1000, 1)
    if perc_resp.get("error"):
        errors.append({"step": "perception", "error": perc_resp["error"]})
    trace["perception"] = {"experiences": len(experiences), "novelty": novelty_score,
                           "ms": step_times["perception"]}

    # ── STEP 8: Assemble cognitive context ────────────────────────────────────
    # Detect if user is asking about learnings/knowledge/memory
    learnings = None
    _input_lower = req.user_input.lower()
    _learning_keywords = ["learned", "learning", "know now", "remember",
                          "new knowledge", "what do you know", "teach",
                          "corrections", "learned today"]
    if any(kw in _input_lower for kw in _learning_keywords):
        try:
            learnings = await recent_learnings(session_id=req.session_id, limit=10)
        except Exception:
            learnings = None

    context = assemble_context(
        user_input    = req.user_input,
        amygdala      = ar,
        wm            = wm,
        episodic      = episodic,
        semantic      = semantic,
        analogies     = analogies,
        compass       = compass,
        experiences   = experiences,
        causal        = causal,
        metacognition = metacognition,
        visual        = visual,
        agenda        = agenda,
        k             = k,
        learnings     = learnings,
        hormones      = hormones,
        intuition     = intuition,
        pending_epiphanies = pending_epiphanies
    )

    # ── STEP 9: Reasoning (direct or multi-agent) ─────────────────────────────
    t9 = time.time()
    word_count    = len(req.user_input.split())
    use_framework = (
        ar.get("complexity") == "high"
        or word_count > k.framework_threshold
        or novelty_score > 0.82
    ) and not req.teacher_mode   # teacher mode always uses direct LLM

    if use_framework:
        trace["reasoning_path"] = "framework"
        fw_resp      = await call("framework", "/process", {
            **base_req, "amygdala_result": ar, "context": context
        })
        raw_response = (
            fw_resp.get("data", {}).get("final_output") or
            fw_resp.get("data", {}).get("synthesis", "") or
            "Framework processing returned no output."
        )
        trace["framework"] = fw_resp.get("data", {}).get("framework", "unknown")
        if fw_resp.get("error"):
            errors.append({"step": "framework", "error": fw_resp["error"]})
    else:
        trace["reasoning_path"] = "teacher_direct" if req.teacher_mode else "llm_direct"
        raw_response = await call_llm(req.user_input, context, ar, k,
                                       teacher_mode=req.teacher_mode)
    step_times["reasoning"] = round((time.time() - t9) * 1000, 1)
    reasoning_ms = step_times["reasoning"]
    trace["reasoning"] = {
        "path": trace.get("reasoning_path", "unknown"),
        "framework": trace.get("framework"),
        "response_len": len(raw_response) if raw_response else 0,
        "ms": reasoning_ms
    }
    if reasoning_ms > 10000:
        print(f"[thalamus] reasoning ({trace.get('reasoning_path','?')}) took {reasoning_ms:.0f}ms", flush=True)

    # ── STEP 9b: Teacher mode — extract and apply emotion corrections ─────────
    if req.teacher_mode:
        trace["teacher_mode"] = True
        # Try to detect emotion words in teacher input and teach the amygdala
        input_lower = req.user_input.lower()
        # Known emotion names from GoEmotions + custom
        all_emotions = [
            "admiration", "amusement", "approval", "caring", "desire",
            "excitement", "gratitude", "joy", "love", "optimism", "pride",
            "relief", "anger", "annoyance", "disappointment", "disapproval",
            "disgust", "embarrassment", "fear", "grief", "nervousness",
            "remorse", "sadness", "confusion", "curiosity", "realization",
            "surprise", "neutral", "belonging", "alarm", "regret",
            "distress", "urgency", "creative"
        ]
        # Look for patterns like "that was curiosity" or "feel happy" or
        # "the correct emotion is joy" or "this should be neutral"
        taught_emotion = None
        for emo in all_emotions:
            if emo in input_lower:
                taught_emotion = emo
                break
        if taught_emotion and taught_emotion != ar.get("emotion"):
            sig_words = [w for w in req.user_input.lower().split()
                         if len(w) > 3 and w.isalpha()]
            pattern = ".*" + ".*".join(sig_words) + ".*" if sig_words else ".*"
            asyncio.create_task(call("amygdala", "/learn-emotion", {
                "name":     taught_emotion,
                "patterns": [pattern],
                "score":    0.7,
                "category": "taught"
            }))
            # Store the correction in working memory
            asyncio.create_task(call("working_memory", "/context", {
                "session_id": req.session_id,
                "action":     "push_thought",
                "data": {
                    "type":             "emotion_correction",
                    "detected_emotion": ar.get("emotion", "neutral"),
                    "correct_emotion":  taught_emotion,
                    "input":            req.user_input[:200],
                    "timestamp":        time.time()
                }
            }))
            trace["emotion_taught"] = {
                "from": ar.get("emotion"),
                "to":   taught_emotion
            }

    # ── STEP 10: Simulator grounding (for factual/math claims) ────────────────
    t10 = time.time()
    if "code" in ar.get("required_regions", []) and \
       any(marker in req.user_input.lower() for marker in
           ["calculate", "compute", "what is", "equals", "run", "result of"]):
        sim_code = await complete(
            prompt  = f"Write Python code to verify: {req.user_input}. Set result = <answer>",
            system  = "Write sandboxed Python. result = <computed value>. No imports needed for math.",
            model_tier = "fast",
            max_tokens = 200
        )
        if sim_code and "result" in sim_code:
            sim_resp = await call("simulator", "/run", {
                "code": sim_code, "session_id": req.session_id
            })
            actual = sim_resp.get("data", {}).get("result")
            if actual and not sim_resp.get("data", {}).get("error"):
                raw_response += f"\n\n*(Computed result: {actual})*"
                trace["simulator_used"] = True
    step_times["simulator"] = round((time.time() - t10) * 1000, 1)
    trace["simulator"] = {
        "used": trace.get("simulator_used", False),
        "ms": step_times["simulator"]
    }

    # ── STEP 11: Verification ─────────────────────────────────────────────────
    t11 = time.time()
    verification = {}
    if k.verification_threshold > 0.2 and len(raw_response) > 60:
        ver_resp = await call("verification", "/verify", {
            "request_id": request_id,
            "session_id": req.session_id,
            "claim":      raw_response[:2000],
            "context":    req.user_input[:1000],
            "domain":     (ar.get("required_regions") or ["general"])[0]
        })
        verification = ver_resp.get("data", {})
        trace["verification"] = {
            "status":     verification.get("status"),
            "confidence": verification.get("confidence"),
            "ms": step_times.get("verification", 0)
        }
        if ver_resp.get("error"):
            errors.append({"step": "verification", "error": ver_resp["error"]})

        if verification.get("status") == "contested":
            raw_response += (
                f"\n\n*Note: There is conflicting information on this. "
                f"{verification.get('reasoning', '')}*"
            )

        # Record metacognition feedback
        if verification.get("confidence"):
            asyncio.create_task(call("metacognition", "/feedback", {
                "domain":                (ar.get("required_regions") or ["general"])[0],
                "predicted_confidence":  metacognition.get("confidence", 0.7),
                "was_correct":           verification.get("status") in
                                         ("confirmed", "probable"),
                "session_id":            req.session_id
            }))
    step_times["verification"] = round((time.time() - t11) * 1000, 1)
    ver_ms = step_times["verification"]
    # Update verification trace with final timing
    if "verification" not in trace:
        trace["verification"] = {"status": "not_run", "confidence": 1.0, "ms": ver_ms}
    else:
        trace["verification"]["ms"] = ver_ms
    if ver_ms > 5000:
        print(f"[thalamus] verification took {ver_ms:.0f}ms", flush=True)

    # ── STEP 12: Compass enrichment ───────────────────────────────────────────
    t12 = time.time()
    final_response = raw_response
    if compass.get("enrichments") and k.ethical_weight > 0.4:
        enrich_resp = await call("compass", "/enrich", {
            "request_id":   request_id,
            "session_id":   req.session_id,
            "user_input":   req.user_input,
            "draft":        raw_response,
            "compass_data": compass
        })
        final_response = (
            enrich_resp.get("data", {}).get("enriched_response") or raw_response
        )
        trace["compass_enriched"] = True
        if enrich_resp.get("error"):
            errors.append({"step": "compass_enrich", "error": enrich_resp["error"]})
    step_times["compass_enrich"] = round((time.time() - t12) * 1000, 1)
    trace["compass_enrich"] = {
        "enriched": trace.get("compass_enriched", False),
        "ms": step_times["compass_enrich"]
    }

    # ── STEP 13: Update Working Memory ────────────────────────────────────────
    # Store the full interaction so the bot can recall the conversation.
    t14 = time.time()
    await asyncio.gather(
        call("working_memory", "/context", {
            "session_id": req.session_id, "action": "push_thought",
            "data": {
                "type":      "interaction",
                "input":     req.user_input,
                "response":  final_response,
                "emotion":   ar.get("emotion", "neutral"),
                "urgency":   ar.get("urgency", "low"),
                "timestamp": time.time()
            }
        }),
        call("working_memory", "/context", {
            "session_id": req.session_id, "action": "set",
            "key":        "last_response",
            "data":       {"content": final_response, "ts": time.time()}
        }),
        # Persist last emotional state so the bot can introspect
        call("working_memory", "/context", {
            "session_id": req.session_id, "action": "set",
            "key":        "last_emotion",
            "data":       {
                "emotion":  ar.get("emotion", "neutral"),
                "urgency":  ar.get("urgency", "low"),
                "threat":   ar.get("threat_detected", False),
                "input":    req.user_input[:200],
                "ts":       time.time()
            }
        }),
        # Persist hormonal state so the bot can introspect on its neurochemistry
        call("working_memory", "/context", {
            "session_id": req.session_id, "action": "set",
            "key":        "last_hormones",
            "data":       {
                "hormones":       hormones.get("hormones", {}) if hormones else {},
                "mood":           hormones.get("mood_summary", "") if hormones else "",
                "dominant":       hormones.get("dominant_hormone", "") if hormones else "",
                "overstimulated": hormones.get("overstimulated", []) if hormones else [],
                "depleted":       hormones.get("depleted", []) if hormones else [],
                "ts":             time.time()
            }
        }),
    )
    step_times["working_memory_update"] = round((time.time() - t14) * 1000, 1)
    trace["working_memory_update"] = {
        "ms": step_times["working_memory_update"]
    }

    # ── STEP 15: DB writes — fire-and-forget (don't block response) ──────────
    t15 = time.time()
    db_ops = [
        {"op": "hippocampus_encode", "ok": True, "error": None},
        {"op": "learner_experience", "ok": True, "error": None},
    ]

    quality = min(1.0, (
        metacognition.get("confidence", 0.7) * 0.4 +
        (1.0 - novelty_score) * 0.3 +
        (0.8 if verification.get("status") in ("confirmed","probable") else 0.5) * 0.3
    ))

    # Fire-and-forget: these write ops don't affect the response
    asyncio.create_task(call("hippocampus", "/encode", {
        **base_req,
        "content":   f"Q: {req.user_input}\nA: {final_response}",
        "emotion":   ar.get("emotion", "neutral"),
        "importance": ar.get("importance", 0.5),
        "concepts":  ar.get("required_regions", [])
    }))
    asyncio.create_task(call("learner", "/experience", {
        "session_id": req.session_id,
        "user_input": req.user_input,
        "response":   final_response,
        "quality":    quality,
        "domain":     (ar.get("required_regions") or ["general"])[0],
        "emotion":    ar.get("emotion", "neutral"),
        "source":     "claude"
    }))

    # Self-update analysis (background — can be slow)
    if significance > 0.70 and experiences:
        asyncio.create_task(call("self_update", "/self-analyze", {
            "session_id":  req.session_id,
            "experiences": experiences[:15],
            "performance": {
                "latency_ms":   (time.time() - start) * 1000,
                "novelty_score": novelty_score,
                "confidence":   metacognition.get("confidence", 0.7)
            },
            "auto_apply": True
        }))

    # Causal graph extraction (background)
    asyncio.create_task(call("causal_graph", "/extract", {
        "request_id": request_id,
        "session_id": req.session_id,
        "text":       final_response[:600],
        "domain":     (ar.get("required_regions") or ["general"])[0]
    }))

    # ── STEP 16: Hypothalamus post-interaction update (background) ────────
    # Update hormones with perception events, metacognition, and verification
    # results that weren't available in Step 1.5
    asyncio.create_task(call("hypothalamus", "/update", {
        "session_id":        req.session_id,
        "amygdala_result":   ar,
        "perception_events": experiences[:10] if experiences else [],
        "metacognition":     metacognition,
        "verification":      verification,
        "turn_count":        turn_count,
        "time_since_last":   0,  # just happened
    }))

    # ── STEP 17: Insight Engine — fragment ingest + crystallization ────────
    # Extract concepts from this interaction and feed to the insight engine
    domain = (ar.get("required_regions") or ["general"])[0]
    fragments = []
    # Fragments from semantic search results
    for item in semantic[:5]:
        content = item.get("content", "")
        words = [w for w in content.lower().split() if len(w) > 4 and w.isalpha()][:3]
        if words:
            fragments.append({
                "concept":     words[0],
                "connections": words[1:],
                "domain":      domain,
                "session_id":  req.session_id,
                "source":      "neocortex",
            })
    # Fragments from cross-domain analogies
    for a in analogies[:3]:
        pattern = a.get("pattern_type", "")
        source  = a.get("source_domain", "")
        if pattern:
            fragments.append({
                "concept":     pattern,
                "connections": [source, domain],
                "domain":      domain,
                "session_id":  req.session_id,
                "source":      "cross-domain",
            })
    # Fragments from causal graph
    for edge in causal.get("edges", [])[:3]:
        cause  = edge.get("cause", "")
        effect = edge.get("effect", "")
        if cause and effect:
            fragments.append({
                "concept":     cause,
                "connections": [effect],
                "domain":      domain,
                "session_id":  req.session_id,
                "source":      "causal-graph",
            })
    # Key concepts from user input
    input_words = [w for w in req.user_input.lower().split()
                   if len(w) > 4 and w.isalpha()][:3]
    if input_words:
        fragments.append({
            "concept":     input_words[0],
            "connections": input_words[1:],
            "domain":      domain,
            "session_id":  req.session_id,
            "source":      "user-input",
        })

    if fragments:
        asyncio.create_task(call("insight_engine", "/ingest", {
            "session_id": req.session_id,
            "fragments":  fragments,
        }))
        # Trigger crystallization check (background)
        asyncio.create_task(call("insight_engine", "/check", {
            "session_id": req.session_id,
            "domain":     domain,
        }))

    # Consolidation check
    wm_pressure = wm_resp.get("data", {}).get("pressure", 0.0)
    if isinstance(wm_pressure, dict):
        wm_pressure = wm_pressure.get("pressure", 0.0)
    if float(wm_pressure) > 0.72:
        asyncio.create_task(call("hippocampus", "/consolidate",
                                  {"max_items": 8}))

    step_times["db_writes"] = round((time.time() - t15) * 1000, 1)
    execution_ms = round((time.time() - start) * 1000, 2)

    # Ordered pipeline summary — every step with its wall-clock time
    trace["pipeline_order"] = [
        {"step": "amygdala+wm+cerebellum", "ms": step_times.get("amygdala", 0),
         "note": "parallel: amygdala evaluate + working memory load + cerebellum recognize"},
        {"step": "hypothalamus", "ms": step_times.get("hypothalamus", 0),
         "note": "gland-based hormone update from amygdala signals"},
        {"step": "epiphanies_check", "ms": step_times.get("epiphanies_check", 0),
         "note": "check insight-engine for pending epiphanies"},
        {"step": "parallel_batch", "ms": step_times.get("parallel_batch", 0),
         "note": "hippocampus+neocortex+cross-domain+compass+goals+vision+intuition",
         "detail": step_times.get("parallel_batch_detail", {})},
        {"step": "causal_graph", "ms": step_times.get("causal_graph", 0),
         "note": "causal relationship query"},
        {"step": "metacognition", "ms": step_times.get("metacognition", 0),
         "note": "uncertainty/confidence assessment"},
        {"step": "perception", "ms": step_times.get("perception", 0),
         "note": "experience detection, novelty scoring"},
        {"step": "reasoning", "ms": step_times.get("reasoning", 0),
         "note": f"LLM response generation ({trace.get('reasoning_path', '?')})"},
        {"step": "simulator", "ms": step_times.get("simulator", 0),
         "note": "code execution grounding (if applicable)"},
        {"step": "verification", "ms": step_times.get("verification", 0),
         "note": "claim verification against knowledge base"},
        {"step": "compass_enrich", "ms": step_times.get("compass_enrich", 0),
         "note": "ethical/value enrichment of response"},
        {"step": "working_memory_update", "ms": step_times.get("working_memory_update", 0),
         "note": "persist interaction + emotion + hormones to WM"},
        {"step": "db_writes", "ms": step_times.get("db_writes", 0),
         "note": "fire-and-forget: hippocampus encode, learner, causal, insight-engine, hypothalamus post-update"},
    ]
    trace["execution_ms"] = execution_ms
    trace["step_times_ms"] = step_times
    trace["db_operations"] = db_ops
    if errors:
        trace["errors"] = errors

    # Log total pipeline time and slow steps to docker logs
    slow = [f"{k}={v:.0f}ms" for k, v in step_times.items()
            if isinstance(v, (int, float)) and v > 5000]
    print(f"[thalamus] /process completed in {execution_ms:.0f}ms"
          + (f" — slow steps: {', '.join(slow)}" if slow else ""),
          flush=True)

    # Report thalamus-level metric (the overall /process latency)
    asyncio.create_task(_report_metric("thalamus", execution_ms, "ok"))

    return {
        "request_id":        request_id,
        "session_id":        req.session_id,
        "response":          final_response,
        "cognitive_path":    trace.get("reasoning_path", "unknown"),
        "execution_ms":      execution_ms,
        "urgency":           ar.get("urgency"),
        "emotion":           ar.get("emotion"),
        "confidence":        metacognition.get("confidence"),
        "uncertainty_level": metacognition.get("uncertainty_level"),
        "novelty_score":     round(novelty_score, 3),
        "experiences":       [{"type": e.get("experience_type",""), "detail": e.get("content","")[:100], "intensity": e.get("intensity",0)} for e in experiences[:10]] if experiences else [],
        "memories_used":     len(episodic) + len(semantic),
        "analogies":         len(analogies),
        "causal_edges":      len(causal.get("edges", [])),
        "compass":           compass.get("integrated_direction", ""),
        "verification":      {
            "status":     verification.get("status", "not_run"),
            "confidence": verification.get("confidence", 1.0)
        },
        "active_goals":      len(agenda),
        "llm_provider":      provider_info().get("provider"),
        "teacher_mode":      req.teacher_mode,
        "hormones":          hormones.get("hormones", {}) if hormones else {},
        "mood":              hormones.get("mood_summary", "") if hormones else "",
        "intuition":         {
            "has_intuition": intuition.get("has_intuition", False),
            "direction":     intuition.get("intuition_direction", ""),
            "confidence":    intuition.get("intuition_confidence", 0),
        } if intuition else {},
        "pending_epiphanies": len(pending_epiphanies) if pending_epiphanies else 0,
        "trace":             trace
    }