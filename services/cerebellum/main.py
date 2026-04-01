# services/cerebellum/main.py
# ── Cerebellum: Procedural Memory and Skill Execution ────────────────────────
# Owns the Skill and Habit tables in EdgeDB.
# Recognizes repeated patterns, executes known procedures automatically,
# learns new skills from successful Claude interactions.
# The more it's used, the fewer Claude API calls are needed.

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, "/shared")

import edgedb
import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus

app = FastAPI(title="Cerebellum", version="1.0.0",
              description="Procedural memory, skill recognition and execution")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=30.0)

OLLAMA_URL  = os.getenv("OLLAMA_URL",           "http://host.docker.internal:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL",   "nomic-embed-text")
EDGEDB_DSN  = os.getenv("EDGEDB_DSN",           "edgedb://edgedb/main")

db: Optional[edgedb.AsyncIOClient] = None

# ── In-memory skill cache ─────────────────────────────────────────────────────
# Fast lookup without hitting EdgeDB on every request.
# Only proven skills (success_count > 2) are cached.
_skill_cache: dict  = {}   # skill_id → {name, trigger, procedure, embedding, ...}
_cache_ready: bool  = False


@app.on_event("startup")
async def startup():
    global db
    db = edgedb.create_async_client(dsn=EDGEDB_DSN)
    await load_skill_cache()


@app.on_event("shutdown")
async def shutdown():
    await http.aclose()
    if db:
        await db.aclose()


# ── Utilities ─────────────────────────────────────────────────────────────────

async def embed(text: str) -> list:
    try:
        resp = await http.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:2000]}
        )
        return resp.json()["embedding"]
    except Exception:
        return [0.0] * 768


def cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / norm) if norm > 1e-8 else 0.0


async def load_skill_cache():
    """Load proven skills into memory. Called at startup and on reload."""
    global _skill_cache, _cache_ready
    _skill_cache = {}
    try:
        skills = await db.query("""
            SELECT Skill {
                id, name, trigger_pattern, procedure,
                success_count, failure_count, avg_execution_ms
            }
            FILTER .success_count > 2
        """)

        for skill in skills:
            embedding = await embed(skill.trigger_pattern)
            _skill_cache[str(skill.id)] = {
                "name":        skill.name,
                "trigger":     skill.trigger_pattern,
                "procedure":   json.loads(skill.procedure),
                "embedding":   embedding,
                "successes":   int(skill.success_count),
                "failures":    int(skill.failure_count),
                "avg_ms":      float(skill.avg_execution_ms or 0)
            }

        _cache_ready = True
        print(f"Cerebellum: cached {len(_skill_cache)} proven skills")

    except Exception as e:
        print(f"Cerebellum cache load error: {e}")
        _cache_ready = True  # don't block startup


def execute_procedure(procedure: list, user_input: str) -> dict:
    """
    Execute a stored procedural skill locally.
    Procedures are JSON step arrays — no AI call needed.
    """
    results = []
    try:
        for step in procedure:
            step_type = step.get("type", "")

            if step_type == "template_response":
                template = step.get("template", "")
                results.append(
                    template.replace("{input}", user_input)
                             .replace("{INPUT}", user_input.upper())
                )

            elif step_type == "static_response":
                results.append(step.get("content", ""))

            elif step_type == "transform":
                transform = step.get("transform", "upper")
                if transform == "upper":
                    results.append(user_input.upper())
                elif transform == "lower":
                    results.append(user_input.lower())
                elif transform == "title":
                    results.append(user_input.title())

        return {
            "success": True,
            "output":  "\n".join(results) if results else "Skill executed."
        }
    except Exception as e:
        return {"success": False, "error": str(e), "output": ""}


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_ok = False
    if db:
        try:
            await db.query_single("SELECT 1")
            db_ok = True
        except Exception:
            pass
    return {
        "status":         "alive",
        "service":        "cerebellum",
        "skills_cached":  len(_skill_cache),
        "cache_ready":    _cache_ready,
        "edgedb":         "connected" if db_ok else "error"
    }


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


@app.post("/recognize", response_model=BrainResponse)
async def recognize_skill(req: BrainRequest):
    """
    Check if this query matches a known procedural skill.
    Returns the skill if similarity exceeds threshold.
    High threshold (0.88) means only very confident matches execute automatically.
    """
    start     = time.time()
    threshold = 0.88

    if not _skill_cache:
        return BrainResponse(
            request_id = req.request_id,
            service    = "cerebellum",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"skill_match": None, "cache_size": 0}
        )

    query_embedding = await embed(req.user_input)
    query_vec       = np.array(query_embedding, dtype=np.float32)

    best_match = None
    best_sim   = 0.0

    for skill_id, skill in _skill_cache.items():
        skill_vec = np.array(skill["embedding"], dtype=np.float32)
        sim       = cosine_similarity(query_vec, skill_vec)
        if sim > best_sim:
            best_sim   = sim
            best_match = skill_id

    match = None
    if best_sim >= threshold and best_match:
        skill    = _skill_cache[best_match]
        exec_res = execute_procedure(skill["procedure"], req.user_input)
        match    = {
            "skill_id":        best_match,
            "name":            skill["name"],
            "similarity":      round(best_sim, 4),
            "successes":       skill["successes"],
            "procedure_output": exec_res.get("output", ""),
            "exec_success":    exec_res.get("success", False)
        }

    return BrainResponse(
        request_id = req.request_id,
        service    = "cerebellum",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "skill_match": match,
            "cache_size":  len(_skill_cache)
        }
    )


class LearnSkillRequest(BaseModel):
    name:            str
    trigger_pattern: str
    procedure:       list
    source:          str = "inferred"


@app.post("/skills", response_model=BrainResponse)
async def learn_skill(req: LearnSkillRequest):
    """
    Teach the cerebellum a new skill.
    Called by self-update after Claude solves a low-complexity repeatable task.
    """
    start = time.time()
    try:
        skill = await db.query_single("""
            INSERT Skill {
                name            := <str>$name,
                trigger_pattern := <str>$trigger,
                procedure       := <str>$procedure,
                success_count   := 0,
                failure_count   := 0,
                last_used       := <datetime>$now
            }
            UNLESS CONFLICT ON .name
            ELSE (SELECT Skill FILTER .name = <str>$name)
        """,
            name      = req.name,
            trigger   = req.trigger_pattern,
            procedure = json.dumps(req.procedure),
            now       = datetime.now(timezone.utc)
        )

        skill_id  = str(skill.id)
        embedding = await embed(req.trigger_pattern)

        # Add to cache immediately
        _skill_cache[skill_id] = {
            "name":      req.name,
            "trigger":   req.trigger_pattern,
            "procedure": req.procedure,
            "embedding": embedding,
            "successes": 0,
            "failures":  0,
            "avg_ms":    0.0
        }

        events.publish("skill.learned", {
            "skill_id":   skill_id,
            "skill_name": req.name,
            "source":     req.source
        })

        return BrainResponse(
            request_id = "learn",
            service    = "cerebellum",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"skill_id": skill_id, "learned": True}
        )

    except Exception as e:
        return BrainResponse(
            request_id = "learn",
            service    = "cerebellum",
            success    = False,
            error      = str(e),
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {}
        )


class OutcomeRequest(BaseModel):
    skill_id:     str
    success:      bool
    execution_ms: float = 0.0


@app.post("/outcome")
async def record_outcome(req: OutcomeRequest):
    """Record skill execution outcome. Bad skills get evicted from cache."""
    try:
        if req.success:
            await db.execute("""
                UPDATE Skill FILTER .id = <uuid>$id
                SET {
                    success_count    := .success_count + 1,
                    last_used        := <datetime>$now,
                    avg_execution_ms := <float64>$ms
                }
            """, id=req.skill_id,
                 now=datetime.now(timezone.utc),
                 ms=req.execution_ms)
            if req.skill_id in _skill_cache:
                _skill_cache[req.skill_id]["successes"] += 1
        else:
            await db.execute("""
                UPDATE Skill FILTER .id = <uuid>$id
                SET { failure_count := .failure_count + 1 }
            """, id=req.skill_id)

            if req.skill_id in _skill_cache:
                _skill_cache[req.skill_id]["failures"] += 1
                # Evict unreliable skills from cache
                if _skill_cache[req.skill_id]["failures"] > 3:
                    del _skill_cache[req.skill_id]
                    print(f"Cerebellum: evicted skill {req.skill_id} "
                          f"(too many failures)")

        return {"recorded": True, "success": req.success}

    except Exception as e:
        return {"recorded": False, "error": str(e)}


@app.post("/reload-cache")
async def reload_cache():
    """Force cache reload from EdgeDB. Call after bulk skill updates."""
    await load_skill_cache()
    return {"skills_loaded": len(_skill_cache)}


@app.get("/skills")
async def list_skills():
    """List all cached skills with stats."""
    return {
        "skills": [
            {
                "skill_id":  sid,
                "name":      s["name"],
                "trigger":   s["trigger"],
                "successes": s["successes"],
                "failures":  s["failures"],
                "avg_ms":    s["avg_ms"]
            }
            for sid, s in _skill_cache.items()
        ],
        "total": len(_skill_cache)
    }
