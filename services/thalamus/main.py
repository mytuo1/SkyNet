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

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from knobs import KnobsClient
from events import EventBus
from llm_provider import complete, provider_info, get_model


async def _noop(val=None):
    """Return a default value as an awaitable — replaces removed asyncio.coroutine."""
    return val if val is not None else {}


app = FastAPI(title="Thalamus", version="2.0.0",
              description="Master cognitive orchestrator — routes all inputs through brain regions")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=120.0)

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
}

# Map internal service names (with underscores) to dashboard names (with hyphens)
_SVC_DASH_NAME = {
    "working_memory": "working-memory",
    "cross_domain": "cross-domain",
    "causal_graph": "causal-graph",
    "self_update": "self-update",
    "framework": "framework-router",
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
               timeout: float = 30.0) -> dict:
    """Call a brain region. Returns {} on failure — brain degrades gracefully."""
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
        asyncio.create_task(_report_metric(service, latency_ms, "error"))
        return {"success": False, "error": str(e)[:60], "data": {}}


async def call_get(service: str, path: str, timeout: float = 10.0) -> dict:
    """GET call to a brain region."""
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
        asyncio.create_task(_report_metric(service, latency_ms, "error"))
        return {"error": str(e)[:60]}


async def call_llm(user_input: str, context: str, amygdala: dict, k,
                   teacher_mode: bool = False) -> str:
    """Call LLM provider (Anthropic or OpenAI) with cognitive context."""
    urgency    = amygdala.get("urgency", "low")
    complexity = amygdala.get("complexity", "medium")

    # Additional system prompt from knobs
    knob_modifier = knobs.build_system_prompt_modifier()
    prompt_additions = ""
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"),
                           decode_responses=True)
        prompt_additions = r.get("brain:system_prompt_additions") or ""
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
        max_tokens = 2048
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
    learnings:     dict = None
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
            f"Empathy: {compass.get('empathy_perception','')[:80]}\n"
            f"Direction: {compass.get('integrated_direction','')[:100]}"
        ]
        if compass.get("gratitude_perception") and \
           compass.get("gratitude_perception") != "No specific gratitude detected":
            compass_parts.append(
                f"Gratitude: {compass.get('gratitude_perception','')[:80]}"
            )
        if compass.get("belonging_perception") and \
           compass.get("belonging_perception") != "No specific belonging signal detected":
            compass_parts.append(
                f"Belonging: {compass.get('belonging_perception','')[:80]}"
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
        recent = [t for t in thoughts[:8]
                  if isinstance(t, dict) and t.get("type") == "interaction"]
        corrections = [t for t in thoughts[:8]
                       if isinstance(t, dict) and t.get("type") == "emotion_correction"]
        if recent:
            parts.append("\n## Recent Conversation")
            for t in recent[:5]:
                emo_tag = ""
                if t.get("emotion"):
                    emo_tag = f" [felt: {t['emotion']}]"
                parts.append(
                    f"User: {t.get('input','')[:100]}\n"
                    f"You: {t.get('response','')[:100]}{emo_tag}"
                )
        if corrections:
            parts.append("\n## Emotion Corrections (Teacher Feedback)")
            for c in corrections[:3]:
                parts.append(
                    f"- For \"{c.get('input','')[:60]}\" you detected "
                    f"'{c.get('detected_emotion','')}' but the correct "
                    f"emotion was '{c.get('correct_emotion','')}'."
                )

    # Last emotional state for introspection
    last_emo = wm.get("last_emotion") or {}
    if last_emo and last_emo.get("emotion") and last_emo["emotion"] != "neutral":
        parts.append(
            f"\n## Your Previous Emotional State\n"
            f"For input \"{last_emo.get('input','')[:60]}\" you felt "
            f"'{last_emo.get('emotion','')}' with urgency "
            f"'{last_emo.get('urgency','low')}'."
        )

    # Episodic memory
    if episodic:
        parts.append("\n## Relevant Past Episodes")
        for ep in episodic[:4]:
            parts.append(f"- [{ep.get('similarity',0):.2f}] {ep.get('summary','')[:100]}")

    # Semantic knowledge
    if semantic:
        parts.append("\n## Relevant Knowledge")
        for item in semantic[:5]:
            parts.append(
                f"- [{item.get('domain','')}] {item.get('content','')[:120]}"
            )

    # Cross-domain analogies
    if analogies:
        parts.append("\n## Cross-Domain Insights")
        for a in analogies[:2]:
            insight = a.get("concept_mapping", {}).get("transfer_insight", "")
            parts.append(
                f"- [{a.get('source_domain','')}] {a.get('pattern_type','')} "
                f"→ {insight[:100]}"
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
            parts.append(f"- {e.get('experience_type')}: {e.get('content','')[:80]}")

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
                    f"'{c.get('correct_emotion','')}' for: \"{c.get('input','')[:50]}\""
                )
        learned_emos = learnings.get("emotions", {}).get("learned", [])
        if learned_emos:
            parts.append(f"Newly learned emotions: {', '.join(learned_emos[:8])}")

    return "\n".join(parts)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
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
    return knobs.get().dict()


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
        call("amygdala", "/evaluate", base_req, timeout=10.0),
        call("working_memory", "/context",
             {"session_id": req.session_id, "action": "get_all"}, timeout=10.0),
        call("cerebellum", "/recognize", base_req, timeout=10.0),
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

    # ── STEP 3: Cerebellum result (ran in parallel above) ─────────────────────
    skill_match = cb_resp.get("data", {}).get("skill_match")
    step_times["cerebellum"] = round((t_early_done - t_early) * 1000, 1)

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

    # ── STEP 4: Parallel memory + reasoning preparation ───────────────────────
    t4 = time.time()
    parallel_batch = [
        # Hippocampus
        call("hippocampus", "/recall", base_req, timeout=8.0)
        if ar.get("needs_memory") else _noop({"data": {"episodes": []}}),
        # Neocortex
        call("neocortex", "/search", {
            **base_req,
            "domain": (ar.get("required_regions") or ["general"])[0]
        }, timeout=8.0),
        # Cross-domain
        call("cross_domain", "/find-analogies", {
            **base_req,
            "source_domain": (ar.get("required_regions") or ["general"])[0]
        }, timeout=8.0) if k.analogy_seeking > 0.3 else _noop({"data": {"analogies": []}}),
        # Compass
        call("compass", "/read", {**base_req, "working_memory": wm}, timeout=8.0),
        # Goals agenda
        call_get("goals", f"/agenda?session_id={req.session_id}"),
        # Vision context (if requested or VISION_ENABLED)
        call_get("vision", "/current") if req.include_vision else _noop({}),
    ]

    results = await asyncio.gather(*parallel_batch, return_exceptions=True)
    step_times["parallel_batch"] = round((time.time() - t4) * 1000, 1)

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

    trace.update({
        "hippocampus":  {"episodes": len(episodic)},
        "neocortex":    {"items": len(semantic)},
        "cross_domain": {"analogies": len(analogies)},
        "vision":       {"has_context": bool(visual.get("caption"))}
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
            trace["causal_graph"] = {"edges": len(causal.get("edges", []))}
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
        learnings     = learnings
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
        }, timeout=180.0)
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

    # ── STEP 11: Verification ─────────────────────────────────────────────────
    t11 = time.time()
    verification = {}
    if k.verification_threshold > 0.2 and len(raw_response) > 60:
        ver_resp = await call("verification", "/verify", {
            "request_id": request_id,
            "session_id": req.session_id,
            "claim":      raw_response[:400],
            "context":    req.user_input[:200],
            "domain":     (ar.get("required_regions") or ["general"])[0]
        })
        verification = ver_resp.get("data", {})
        trace["verification"] = {
            "status":     verification.get("status"),
            "confidence": verification.get("confidence")
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
    step_times["compass_enrich"] = round((time.time() - t12) * 1000, 1)

    # ── STEP 13: Update Working Memory ────────────────────────────────────────
    # Store the emotional assessment alongside the interaction so the bot
    # can recall its own feelings when asked about them later.
    t14 = time.time()
    await asyncio.gather(
        call("working_memory", "/context", {
            "session_id": req.session_id, "action": "push_thought",
            "data": {
                "type":      "interaction",
                "input":     req.user_input[:200],
                "response":  final_response[:200],
                "emotion":   ar.get("emotion", "neutral"),
                "urgency":   ar.get("urgency", "low"),
                "timestamp": time.time()
            }
        }),
        call("working_memory", "/context", {
            "session_id": req.session_id, "action": "set",
            "key":        "last_response",
            "data":       {"content": final_response[:300], "ts": time.time()}
        }),
        # Persist last emotional state so the bot can introspect
        call("working_memory", "/context", {
            "session_id": req.session_id, "action": "set",
            "key":        "last_emotion",
            "data":       {
                "emotion":  ar.get("emotion", "neutral"),
                "urgency":  ar.get("urgency", "low"),
                "threat":   ar.get("threat_detected", False),
                "input":    req.user_input[:100],
                "ts":       time.time()
            }
        }),
    )
    step_times["working_memory_update"] = round((time.time() - t14) * 1000, 1)

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
        "content":   f"Q: {req.user_input}\nA: {final_response[:400]}",
        "emotion":   ar.get("emotion", "neutral"),
        "importance": ar.get("importance", 0.5),
        "concepts":  ar.get("required_regions", [])
    }))
    asyncio.create_task(call("learner", "/experience", {
        "session_id": req.session_id,
        "user_input": req.user_input,
        "response":   final_response[:600],
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
        }, timeout=90.0))

    # Causal graph extraction (background)
    asyncio.create_task(call("causal_graph", "/extract", {
        "request_id": request_id,
        "session_id": req.session_id,
        "text":       final_response[:600],
        "domain":     (ar.get("required_regions") or ["general"])[0]
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
    trace["step_times_ms"] = step_times
    trace["db_operations"] = db_ops
    if errors:
        trace["errors"] = errors

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
        "trace":             trace
    }