# services/thalamus/main.py
# ── Thalamus: Master Orchestrator ────────────────────────────────────────────
# Every input enters here. The thalamus decides which brain regions
# activate, in what order, and assembles their outputs into a coherent
# cognitive process before handing off to Claude (prefrontal cortex).
#
# Processing order:
#   1.  Amygdala     — urgency, emotion, threat (always first, fastest)
#   2.  Working Mem  — load active context
#   3.  Cerebellum   — check for known skill (fast path)
#   4.  Hippocampus  — recall past episodes (parallel)
#   5.  Neocortex    — semantic knowledge search (parallel)
#   6.  Cross-Domain — structural analogies (parallel)
#   7.  Compass      — ethical orientation (parallel)
#   8.  Perception   — detect experiences from input characteristics
#   9.  Assemble     — build cognitive context for Claude
#   10. Claude       — reason with full context, or multi-agent framework
#   11. Verification — check response claims
#   12. Compass Enrich — enrich response with genuine care
#   13. Update WM    — push thought to stream
#   14. Encode Ep    — hippocampus encodes new episode (background)
#   15. Self-Analyze — if significant experiences, trigger self-improvement

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

app = FastAPI(title="Thalamus", version="1.0.0",
              description="Master orchestrator — routes inputs through all brain regions")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=120.0)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SERVICES = {
    "amygdala":       os.getenv("AMYGDALA_URL",        "http://amygdala:8001"),
    "hippocampus":    os.getenv("HIPPOCAMPUS_URL",      "http://hippocampus:8002"),
    "cerebellum":     os.getenv("CEREBELLUM_URL",       "http://cerebellum:8003"),
    "neocortex":      os.getenv("NEOCORTEX_URL",        "http://neocortex:8004"),
    "working_memory": os.getenv("WORKING_MEMORY_URL",   "http://working-memory:8005"),
    "perception":     os.getenv("PERCEPTION_URL",       "http://perception:8006"),
    "verification":   os.getenv("VERIFICATION_URL",     "http://verification:8007"),
    "cross_domain":   os.getenv("CROSS_DOMAIN_URL",     "http://cross-domain:8008"),
    "compass":        os.getenv("COMPASS_URL",          "http://compass:8009"),
    "framework":      os.getenv("FRAMEWORK_ROUTER_URL", "http://framework-router:8010"),
    "self_update":    os.getenv("SELF_UPDATE_URL",      "http://self-update:8011"),
    "knobs":          os.getenv("KNOBS_URL",            "http://knobs:8090"),
}

MAIN_MODEL = os.getenv("DEFAULT_CLAUDE_MODEL", "claude-sonnet-4-20250514")


class ProcessRequest(BaseModel):
    session_id: str
    user_input: str
    metadata:   dict = {}


# ── Service call helper ───────────────────────────────────────────────────────

async def call(service: str, path: str, data: dict,
               timeout: float = 30.0) -> dict:
    """
    Call a brain region service. Returns empty dict on failure.
    Failures are non-fatal — the brain degrades gracefully.
    """
    url = f"{SERVICES[service]}{path}"
    try:
        resp = await http.post(url, json=data, timeout=timeout)
        return resp.json()
    except Exception as e:
        print(f"[Thalamus] {service}{path} failed: {str(e)[:60]}")
        return {"success": False, "error": str(e), "data": {}}


async def call_claude(user_input: str, context: str,
                      amygdala: dict, k) -> str:
    """Call Claude with assembled cognitive context."""
    if not ANTHROPIC_KEY:
        return (
            f"[No API key configured] Would respond to: {user_input[:100]}"
        )

    urgency    = amygdala.get("urgency", "low")
    complexity = amygdala.get("complexity", "medium")
    knob_modifier = k.build_system_prompt_modifier()

    system = (
        "You are the reasoning center of a cognitive architecture. "
        "You have been given context from multiple memory and perception systems. "
        "Use this context naturally — do not announce it, let it inform your response.\n\n"
    )
    if urgency == "critical":
        system += "Urgency is CRITICAL — be concise and direct.\n"
    if complexity == "high":
        system += "This is a complex task — be thorough.\n"
    if knob_modifier:
        system += f"\nBehavioral guidance:\n{knob_modifier}"

    user_message = f"{context}\n\n## Current Input\n{user_input}"

    try:
        resp = await http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json"
            },
            json={
                "model":      MAIN_MODEL,
                "max_tokens": 2048,
                "system":     system,
                "messages":   [{"role": "user", "content": user_message}]
            },
            timeout=90.0
        )
        return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"[Claude call failed: {str(e)[:80]}]"


def assemble_context(user_input: str, amygdala: dict, wm: dict,
                      episodic: list, semantic: list, analogies: list,
                      compass: dict, experiences: list, k) -> str:
    """
    Package all retrieved memory and perceptual data into a structured
    context block that orients Claude before responding.
    """
    parts = []

    # Cognitive state
    parts.append(
        f"## Cognitive State\n"
        f"Urgency: {amygdala.get('urgency', 'low')} | "
        f"Emotion: {amygdala.get('emotion', 'neutral')} | "
        f"Complexity: {amygdala.get('complexity', 'medium')} | "
        f"Resonance: {amygdala.get('resonance_score', 0):.2f}"
    )

    # Compass orientation
    if compass.get("integrated_direction"):
        parts.append(
            f"\n## Ethical Compass\n"
            f"Empathy: {compass.get('empathy_perception', '')[:80]}\n"
            f"Direction: {compass.get('integrated_direction', '')[:100]}"
        )
        if compass.get("tensions"):
            parts.append(
                f"Tensions: {'; '.join(compass['tensions'][:2])}"
            )

    # Recent conversation (working memory thought stream)
    thoughts = wm.get("thought_stream", [])
    if thoughts:
        recent = [
            t for t in thoughts[:6]
            if isinstance(t, dict) and t.get("type") == "interaction"
        ]
        if recent:
            parts.append("\n## Recent Conversation")
            for t in recent[:4]:
                parts.append(
                    f"User: {t.get('input', '')[:100]}\n"
                    f"System: {t.get('response', '')[:100]}"
                )

    # Episodic memories
    if episodic:
        parts.append("\n## Relevant Past Episodes")
        for ep in episodic[:4]:
            sim = ep.get("similarity", 0)
            parts.append(
                f"- [{sim:.2f}] {ep.get('summary', '')[:100]}"
            )

    # Semantic knowledge
    if semantic:
        parts.append("\n## Relevant Knowledge")
        for item in semantic[:5]:
            domain = item.get("domain", "")
            score  = item.get("score", 0)
            parts.append(
                f"- [{domain}|{score:.2f}] {item.get('content', '')[:120]}"
            )

    # Cross-domain analogies
    if analogies:
        parts.append("\n## Cross-Domain Insights")
        for a in analogies[:2]:
            mapping = a.get("concept_mapping", {})
            insight = mapping.get("transfer_insight", "")
            parts.append(
                f"- [{a.get('source_domain', '')}] "
                f"Pattern: {a.get('pattern_type', '')} "
                f"(sim={a.get('similarity', 0):.2f})\n"
                f"  → {insight[:100]}"
            )

    # Active working memory (non-stream items)
    active_ctx = {
        k: v for k, v in wm.items()
        if k not in ("thought_stream",) and isinstance(v, (str, int, float, bool))
    }
    if active_ctx:
        parts.append("\n## Active Context")
        for key, val in list(active_ctx.items())[:5]:
            parts.append(f"- {key}: {str(val)[:80]}")

    # Novel experiences detected
    novel_types = [
        "SEMANTIC_NOVELTY", "KNOWLEDGE_BOUNDARY",
        "ANALOGICAL_INSIGHT", "DOMAIN_COLLISION"
    ]
    novel_exps = [
        e for e in experiences
        if e.get("experience_type") in novel_types
    ]
    if novel_exps:
        parts.append("\n## New Experiences This Interaction")
        for e in novel_exps[:3]:
            parts.append(
                f"- {e.get('experience_type')}: "
                f"{e.get('content', '')[:80]}"
            )

    return "\n".join(parts)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Full brain health check — queries every service."""
    checks = {}
    async def check_service(name: str, url: str):
        try:
            resp = await http.get(f"{url}/health", timeout=3.0)
            checks[name] = resp.json().get("status", "unknown")
        except Exception:
            checks[name] = "unreachable"

    await asyncio.gather(*[
        check_service(name, url)
        for name, url in SERVICES.items()
    ])

    all_alive = all(v == "alive" for v in checks.values())
    return {
        "status":        "alive" if all_alive else "degraded",
        "service":       "thalamus",
        "brain_regions": checks,
        "all_regions_up": all_alive
    }


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


@app.post("/process")
async def process(req: ProcessRequest):
    """
    Main brain entry point.
    Orchestrates the full cognitive pipeline for every input.
    """
    start      = time.time()
    request_id = uuid.uuid4().hex
    k          = knobs.get()
    trace      = {}   # full audit trail

    base_req = {
        "request_id": request_id,
        "session_id": req.session_id,
        "user_input": req.user_input,
        "metadata":   req.metadata
    }

    # ── STEP 1: Amygdala (always first, fastest, no awaiting) ─────────────────
    amygdala_resp = await call("amygdala", "/evaluate", base_req)
    ar = amygdala_resp.get("data", {})
    trace["amygdala"] = {
        "urgency":    ar.get("urgency"),
        "emotion":    ar.get("emotion"),
        "importance": ar.get("importance"),
        "threat":     ar.get("threat_detected", False)
    }

    # ── STEP 2: Working Memory (load current session state) ───────────────────
    wm_resp = await call("working_memory", "/context", {
        "session_id": req.session_id,
        "action":     "get_all"
    })
    wm = wm_resp.get("data", {})
    trace["working_memory"] = {"items": len(wm)}

    # ── STEP 3: Cerebellum (check for known skill — fast path) ────────────────
    cerebellum_resp = await call("cerebellum", "/recognize", base_req)
    skill_match = cerebellum_resp.get("data", {}).get("skill_match")
    trace["cerebellum"] = {"skill_found": skill_match is not None}

    if skill_match and skill_match.get("similarity", 0) > 0.88 \
            and skill_match.get("exec_success", False):

        # Cerebellum handled it — encode episode in background and return
        asyncio.create_task(call("hippocampus", "/encode", {
            **base_req,
            "content":   f"Skill used: {skill_match['name']} for: {req.user_input}",
            "emotion":   ar.get("emotion", "neutral"),
            "importance": ar.get("importance", 0.3) * 0.7,
            "concepts":  [skill_match["name"]]
        }))

        # Record success
        asyncio.create_task(call("cerebellum", "/outcome", {
            "skill_id":     skill_match.get("skill_id", ""),
            "success":      True,
            "execution_ms": (time.time() - start) * 1000
        }))

        return {
            "request_id":     request_id,
            "session_id":     req.session_id,
            "response":       skill_match.get("procedure_output", "Done."),
            "cognitive_path": "cerebellum_fast_path",
            "skill_used":     skill_match.get("name"),
            "execution_ms":   round((time.time() - start) * 1000, 2),
            "trace":          trace
        }

    # ── STEP 4-7: Parallel memory and perception tasks ─────────────────────────
    parallel_tasks = []

    # Hippocampus recall (if memory needed)
    if ar.get("needs_memory", False):
        parallel_tasks.append(call("hippocampus", "/recall", base_req))
    else:
        parallel_tasks.append(asyncio.coroutine(lambda: {"data": {"episodes": []}})())

    # Neocortex search
    parallel_tasks.append(call("neocortex", "/search", {
        **base_req,
        "domain": (ar.get("required_regions") or ["general"])[0]
    }))

    # Cross-domain analogies (only if analogy_seeking knob is high enough)
    if k.analogy_seeking > 0.3:
        parallel_tasks.append(call("cross_domain", "/find-analogies", {
            **base_req,
            "source_domain": (ar.get("required_regions") or ["general"])[0]
        }))
    else:
        parallel_tasks.append(asyncio.coroutine(lambda: {"data": {"analogies": []}})())

    # Compass reading
    parallel_tasks.append(call("compass", "/read", {
        **base_req,
        "working_memory": wm
    }))

    results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

    # Unpack parallel results safely
    def safe_data(result, key=None, default=None):
        if isinstance(result, Exception) or not isinstance(result, dict):
            return default or ([] if key else {})
        data = result.get("data", {})
        return data.get(key, default or []) if key else data

    episodic  = safe_data(results[0], "episodes", [])
    semantic  = safe_data(results[1], "results",  [])
    analogies = safe_data(results[2], "analogies", [])
    compass   = safe_data(results[3])

    trace["hippocampus"]  = {"episodes": len(episodic)}
    trace["neocortex"]    = {"knowledge_items": len(semantic)}
    trace["cross_domain"] = {"analogies": len(analogies)}
    trace["compass"]      = {
        "direction": compass.get("integrated_direction", "")[:50]
    }

    # ── STEP 8: Perception (detect experiences) ───────────────────────────────
    perception_resp = await call("perception", "/perceive", {
        "request_id":        request_id,
        "session_id":        req.session_id,
        "user_input":        req.user_input,
        "amygdala_result":   ar,
        "neocortex_results": semantic,
        "working_memory":    wm,
        "skill_cache_size":  cerebellum_resp.get("data", {}).get("cache_size", 0)
    })

    experiences   = perception_resp.get("data", {}).get("experiences", [])
    novelty_score = perception_resp.get("data", {}).get("novelty_score", 0)
    significance  = perception_resp.get("data", {}).get("significance", 0)

    trace["perception"] = {
        "experiences":  len(experiences),
        "novelty":      novelty_score,
        "significance": significance
    }

    # ── STEP 9: Assemble cognitive context ────────────────────────────────────
    context = assemble_context(
        user_input  = req.user_input,
        amygdala    = ar,
        wm          = wm,
        episodic    = episodic,
        semantic    = semantic,
        analogies   = analogies,
        compass     = compass,
        experiences = experiences,
        k           = k
    )

    # ── STEP 10: Reasoning (Claude direct or multi-agent framework) ───────────
    word_count    = len(req.user_input.split())
    use_framework = (
        ar.get("complexity") == "high"
        or word_count > k.framework_threshold
        or novelty_score > 0.82
    )

    if use_framework:
        trace["reasoning_path"] = "framework"
        fw_resp    = await call(
            "framework", "/process",
            {
                **base_req,
                "amygdala_result": ar,
                "context":         context
            },
            timeout=180.0
        )
        raw_response = (
            fw_resp.get("data", {}).get("final_output")
            or fw_resp.get("data", {}).get("synthesis", "")
            or "Framework processing did not return output."
        )
        trace["framework"] = fw_resp.get("data", {}).get("framework", "unknown")
    else:
        trace["reasoning_path"] = "claude_direct"
        raw_response = await call_claude(
            user_input = req.user_input,
            context    = context,
            amygdala   = ar,
            k          = k
        )

    # ── STEP 11: Verification ─────────────────────────────────────────────────
    verification = {}
    if k.verification_threshold > 0.2 and len(raw_response) > 50:
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

        # Append uncertainty note for contested claims
        if verification.get("status") == "contested":
            raw_response += (
                f"\n\n*Note: There is conflicting information on this topic. "
                f"{verification.get('reasoning', '')}*"
            )

    # ── STEP 12: Compass Enrichment ───────────────────────────────────────────
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
            enrich_resp.get("data", {}).get("enriched_response")
            or raw_response
        )
        trace["compass_enriched"] = True

    # ── STEP 13: Update Working Memory ────────────────────────────────────────
    await call("working_memory", "/context", {
        "session_id": req.session_id,
        "action":     "push_thought",
        "data": {
            "type":      "interaction",
            "input":     req.user_input[:200],
            "response":  final_response[:200],
            "timestamp": time.time()
        }
    })

    # Also store last response for self-contradiction detection
    await call("working_memory", "/context", {
        "session_id": req.session_id,
        "action":     "set",
        "key":        "last_response",
        "data":       {"content": final_response[:300], "ts": time.time()}
    })

    # ── STEP 14: Encode Episode (background — non-blocking) ───────────────────
    asyncio.create_task(call("hippocampus", "/encode", {
        **base_req,
        "content":   f"Q: {req.user_input}\nA: {final_response[:400]}",
        "emotion":   ar.get("emotion", "neutral"),
        "importance": ar.get("importance", 0.5),
        "concepts":  ar.get("required_regions", [])
    }))

    # ── STEP 15: Self-Analysis (background — only if significant) ─────────────
    if significance > 0.72 and experiences:
        asyncio.create_task(call(
            "self_update", "/self-analyze",
            {
                "session_id":  req.session_id,
                "experiences": experiences[:15],
                "performance": {
                    "latency_ms":   (time.time() - start) * 1000,
                    "novelty_score": novelty_score,
                    "used_framework": use_framework
                },
                "auto_apply": True
            },
            timeout=90.0
        ))

    # ── Memory consolidation check ────────────────────────────────────────────
    wm_pressure_resp = await call("working_memory", "/context", {
        "session_id": req.session_id,
        "action":     "pressure"
    })
    pressure = wm_pressure_resp.get("data", {}).get("pressure", 0)
    if pressure > 0.75:
        asyncio.create_task(call("hippocampus", "/consolidate",
                                  {"max_items": 8}))
        trace["consolidation_triggered"] = True

    execution_ms = round((time.time() - start) * 1000, 2)

    return {
        "request_id":        request_id,
        "session_id":        req.session_id,
        "response":          final_response,
        "cognitive_path":    trace.get("reasoning_path", "unknown"),
        "execution_ms":      execution_ms,
        "urgency":           ar.get("urgency"),
        "emotion":           ar.get("emotion"),
        "novelty_score":     round(novelty_score, 3),
        "experiences_count": len(experiences),
        "memories_used":     len(episodic) + len(semantic),
        "analogies_found":   len(analogies),
        "compass_direction": compass.get("integrated_direction", ""),
        "verification":      {
            "status":     verification.get("status", "not_run"),
            "confidence": verification.get("confidence", 1.0)
        },
        "trace": trace
    }
