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

app = FastAPI(title="Thalamus", version="2.0.0",
              description="Master cognitive orchestrator — routes all inputs through brain regions")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=120.0)

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


class ProcessRequest(BaseModel):
    session_id:     str
    user_input:     str
    metadata:       dict = {}
    include_vision: bool = False   # attach current visual context


async def call(service: str, path: str, data: dict,
               timeout: float = 30.0) -> dict:
    """Call a brain region. Returns {} on failure — brain degrades gracefully."""
    url = f"{SERVICES[service]}{path}"
    try:
        resp = await http.post(url, json=data, timeout=timeout)
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)[:60], "data": {}}


async def call_get(service: str, path: str, timeout: float = 10.0) -> dict:
    """GET call to a brain region."""
    url = f"{SERVICES[service]}{path}"
    try:
        resp = await http.get(url, timeout=timeout)
        return resp.json()
    except Exception as e:
        return {"error": str(e)[:60]}


async def call_llm(user_input: str, context: str, amygdala: dict, k) -> str:
    """Call LLM provider (Anthropic or OpenAI) with cognitive context."""
    urgency    = amygdala.get("urgency", "low")
    complexity = amygdala.get("complexity", "medium")

    # Additional system prompt from knobs
    knob_modifier = k.build_system_prompt_modifier()
    prompt_additions = ""
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"),
                           decode_responses=True)
        prompt_additions = r.get("brain:system_prompt_additions") or ""
    except Exception:
        pass

    system = (
        "You are the reasoning center of a cognitive architecture. "
        "Multiple memory and perception systems have prepared context for you. "
        "Use this context naturally — let it inform your response without announcing it.\n\n"
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
    k
) -> str:
    """Assemble complete cognitive context for the LLM."""
    parts = []

    # Cognitive state
    conf_str = ""
    if metacognition.get("confidence"):
        conf_str = f" | Confidence: {metacognition['confidence']:.0%}"
    parts.append(
        f"## Cognitive State\n"
        f"Urgency: {amygdala.get('urgency','low')} | "
        f"Emotion: {amygdala.get('emotion','neutral')} | "
        f"Complexity: {amygdala.get('complexity','medium')}"
        f"{conf_str}"
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
        parts.append(
            f"\n## Ethical Compass\n"
            f"Empathy: {compass.get('empathy_perception','')[:80]}\n"
            f"Direction: {compass.get('integrated_direction','')[:100]}"
        )

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

    # Conversation history
    thoughts = wm.get("thought_stream", [])
    if thoughts:
        recent = [t for t in thoughts[:6]
                  if isinstance(t, dict) and t.get("type") == "interaction"]
        if recent:
            parts.append("\n## Recent Conversation")
            for t in recent[:4]:
                parts.append(
                    f"User: {t.get('input','')[:100]}\n"
                    f"System: {t.get('response','')[:100]}"
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
            "ANALOGICAL_INSIGHT", "EPISTEMIC_SURPRISE"
        ]
    ]
    if novel_exps:
        parts.append("\n## New Experiences This Interaction")
        for e in novel_exps[:3]:
            parts.append(f"- {e.get('experience_type')}: {e.get('content','')[:80]}")

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

    base_req = {
        "request_id": request_id,
        "session_id": req.session_id,
        "user_input": req.user_input,
        "metadata":   req.metadata
    }

    # ── STEP 1: Amygdala ──────────────────────────────────────────────────────
    ar_resp = await call("amygdala", "/evaluate", base_req)
    ar      = ar_resp.get("data", {})
    trace["amygdala"] = {
        "urgency": ar.get("urgency"),
        "emotion": ar.get("emotion"),
        "threat":  ar.get("threat_detected", False)
    }

    # ── STEP 2: Working Memory ────────────────────────────────────────────────
    wm_resp = await call("working_memory", "/context",
                          {"session_id": req.session_id, "action": "get_all"})
    wm      = wm_resp.get("data", {})

    # ── STEP 3: Cerebellum (fast path check) ─────────────────────────────────
    cb_resp     = await call("cerebellum", "/recognize", base_req)
    skill_match = cb_resp.get("data", {}).get("skill_match")

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
        return {
            "request_id": request_id, "session_id": req.session_id,
            "response": skill_match.get("procedure_output", "Done."),
            "cognitive_path": "cerebellum_fast_path",
            "skill_used": skill_match.get("name"),
            "execution_ms": round((time.time() - start) * 1000, 2), "trace": trace
        }

    # ── STEP 4: Parallel memory + reasoning preparation ───────────────────────
    parallel_batch = [
        # Hippocampus
        call("hippocampus", "/recall", base_req)
        if ar.get("needs_memory") else asyncio.coroutine(lambda: {"data": {"episodes": []}})(),
        # Neocortex
        call("neocortex", "/search", {
            **base_req,
            "domain": (ar.get("required_regions") or ["general"])[0]
        }),
        # Cross-domain
        call("cross_domain", "/find-analogies", {
            **base_req,
            "source_domain": (ar.get("required_regions") or ["general"])[0]
        }) if k.analogy_seeking > 0.3 else asyncio.coroutine(lambda: {"data": {"analogies": []}})(),
        # Compass
        call("compass", "/read", {**base_req, "working_memory": wm}),
        # Goals agenda
        call_get("goals", f"/agenda?session_id={req.session_id}"),
        # Vision context (if requested or VISION_ENABLED)
        call_get("vision", "/current") if req.include_vision else asyncio.coroutine(lambda: {})(),
    ]

    results = await asyncio.gather(*parallel_batch, return_exceptions=True)

    def safe(res, key=None, default=None):
        if isinstance(res, (Exception, BaseException)):
            return default or ([] if key else {})
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
    causal = {}
    if k.analogy_seeking > 0.4 and req.user_input:
        # Extract key concept from input for causal query
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

    # ── STEP 6: Metacognition assessment ─────────────────────────────────────
    meta_resp = await call("metacognition", "/assess", {
        "request_id":        request_id,
        "session_id":        req.session_id,
        "user_input":        req.user_input,
        "domain":            (ar.get("required_regions") or ["general"])[0],
        "neocortex_results": semantic
    })
    metacognition = meta_resp.get("data", {})
    trace["metacognition"] = {
        "confidence": metacognition.get("confidence"),
        "level":      metacognition.get("uncertainty_level")
    }

    # ── STEP 7: Perception ────────────────────────────────────────────────────
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
    trace["perception"] = {"experiences": len(experiences), "novelty": novelty_score}

    # ── STEP 8: Assemble cognitive context ────────────────────────────────────
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
        k             = k
    )

    # ── STEP 9: Reasoning (direct or multi-agent) ─────────────────────────────
    word_count    = len(req.user_input.split())
    use_framework = (
        ar.get("complexity") == "high"
        or word_count > k.framework_threshold
        or novelty_score > 0.82
    )

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
    else:
        trace["reasoning_path"] = "llm_direct"
        raw_response = await call_llm(req.user_input, context, ar, k)

    # ── STEP 10: Simulator grounding (for factual/math claims) ────────────────
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

    # ── STEP 11: Verification ─────────────────────────────────────────────────
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

    # ── STEP 12: Compass enrichment ───────────────────────────────────────────
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

    # ── STEP 13: Extract causal relations from response ───────────────────────
    asyncio.create_task(call("causal_graph", "/extract", {
        "request_id": request_id,
        "session_id": req.session_id,
        "text":       final_response[:600],
        "domain":     (ar.get("required_regions") or ["general"])[0]
    }))

    # ── STEP 14: Update Working Memory ────────────────────────────────────────
    await call("working_memory", "/context", {
        "session_id": req.session_id, "action": "push_thought",
        "data": {
            "type":      "interaction",
            "input":     req.user_input[:200],
            "response":  final_response[:200],
            "timestamp": time.time()
        }
    })
    await call("working_memory", "/context", {
        "session_id": req.session_id, "action": "set",
        "key":        "last_response",
        "data":       {"content": final_response[:300], "ts": time.time()}
    })

    # ── STEP 15: Background tasks ─────────────────────────────────────────────
    quality = min(1.0, (
        metacognition.get("confidence", 0.7) * 0.4 +
        (1.0 - novelty_score) * 0.3 +
        (0.8 if verification.get("status") in ("confirmed","probable") else 0.5) * 0.3
    ))

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

    # Consolidation check
    wm_pressure = wm_resp.get("data", {}).get("pressure", 0.0)
    if isinstance(wm_pressure, dict):
        wm_pressure = wm_pressure.get("pressure", 0.0)
    if float(wm_pressure) > 0.72:
        asyncio.create_task(call("hippocampus", "/consolidate",
                                  {"max_items": 8}))

    execution_ms = round((time.time() - start) * 1000, 2)

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
        "experiences":       len(experiences),
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
        "trace":             trace
    }