# services/causal-graph/main.py
# ── Causal Graph: Explicit Cause-Effect Knowledge Base ───────────────────────
# Tracks causal relationships as a directed graph in Redis.
# Extracts cause→effect pairs from text using Claude.
# Enables predictive reasoning: "if X then Y" queries.
# Integrates with neocortex and verification services.

import asyncio
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/shared")

import redis
import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus
from llm_provider import complete_json, complete

app    = FastAPI(title="CausalGraph", version="1.0.0",
                 description="Explicit cause-effect knowledge and causal reasoning")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
http   = httpx.AsyncClient(timeout=30.0)
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


async def embed(text: str) -> list:
    try:
        resp = await http.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:1500]}
        )
        return resp.json()["embedding"]
    except Exception:
        return [0.0] * 768


def cosine_sim(a: list, b: list) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    n = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / n) if n > 1e-8 else 0.0


class ExtractRequest(BaseModel):
    request_id: str = "causal"
    session_id: str = ""
    text:       str
    domain:     str = "general"


class QueryRequest(BaseModel):
    concept:   str
    depth:     int = 2
    direction: str = "both"   # causes | effects | both


class AddEdgeRequest(BaseModel):
    cause:     str
    effect:    str
    mechanism: str   = ""
    strength:  float = 0.5
    domain:    str   = "general"
    evidence:  int   = 1


@app.get("/health")
async def health():
    node_count = r.scard("cg:nodes")
    return {"status": "alive", "service": "causal-graph",
            "nodes": node_count}


@app.post("/extract", response_model=BrainResponse)
async def extract_causal_relations(req: ExtractRequest):
    """
    Parse text and extract causal relationships using Claude/OpenAI.
    Stores extracted relations in Redis causal graph.
    """
    start = time.time()

    result = await complete_json(
        prompt=f"""Extract all causal relationships from this text.
Look for: causes, leads to, results in, because, therefore, enables,
prevents, increases, decreases, triggers, produces, inhibits.

TEXT: {req.text[:800]}
DOMAIN: {req.domain}

JSON response:
{{
    "causal_relations": [
        {{
            "cause": "the causing entity/event",
            "effect": "the resulting entity/event",
            "mechanism": "how or why this causal link works",
            "strength": 0.0 to 1.0,
            "direction": "forward | bidirectional | probabilistic"
        }}
    ]
}}""",
        system="Extract causal relationships precisely. Only include explicit causal links."
    )

    relations = result.get("causal_relations", [])
    stored    = []

    for rel in relations:
        if not rel.get("cause") or not rel.get("effect"):
            continue
        await _store_edge(
            cause     = str(rel["cause"])[:200],
            effect    = str(rel["effect"])[:200],
            mechanism = str(rel.get("mechanism", ""))[:400],
            strength  = float(rel.get("strength", 0.5)),
            domain    = req.domain,
            evidence  = 1
        )
        stored.append(rel)

    return BrainResponse(
        request_id = req.request_id,
        service    = "causal-graph",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"extracted": stored, "count": len(stored)}
    )


async def _store_edge(cause: str, effect: str, mechanism: str,
                       strength: float, domain: str, evidence: int = 1):
    """Store a causal edge in Redis graph."""
    edge_key = f"cg:edge:{cause}::{effect}"
    existing  = r.get(edge_key)

    if existing:
        try:
            data = json.loads(existing)
        except (json.JSONDecodeError, TypeError):
            data = {"cause": cause, "effect": effect, "strength": strength,
                    "domain": domain, "evidence": 0, "updated": time.time()}
        ev           = data.get("evidence", 1) + 1
        new_strength = (data["strength"] * (ev - 1) + strength) / ev
        data.update({
            "strength": round(new_strength, 3),
            "evidence": ev,
            "updated":  time.time()
        })
    else:
        emb  = await embed(f"{cause} causes {effect}: {mechanism}")
        data = {
            "cause": cause, "effect": effect, "mechanism": mechanism,
            "strength": strength, "domain": domain, "evidence": evidence,
            "embedding": emb, "created": time.time(), "updated": time.time()
        }

    loop = asyncio.get_event_loop()
    def _redis_write():
        r.set(edge_key, json.dumps(data))
        r.sadd(f"cg:causes:{cause}", effect)
        r.sadd(f"cg:effects:{effect}", cause)
        r.sadd("cg:nodes", cause)
        r.sadd("cg:nodes", effect)
    await loop.run_in_executor(None, _redis_write)


@app.post("/add-edge")
async def add_edge(req: AddEdgeRequest):
    """Manually add a causal edge (from user or other services)."""
    await _store_edge(req.cause, req.effect, req.mechanism,
                      req.strength, req.domain, req.evidence)
    return {"added": True, "cause": req.cause, "effect": req.effect}


@app.post("/query", response_model=BrainResponse)
async def query_causal_graph(req: QueryRequest):
    """
    Traverse the causal graph from a starting concept.
    Returns causes and effects up to `depth` hops away.
    """
    start   = time.time()
    visited = set()
    edges   = []

    def traverse(concept: str, direction: str, hops: int):
        if hops == 0 or concept in visited:
            return
        visited.add(concept)

        if direction in ("effects", "both"):
            for effect in r.smembers(f"cg:causes:{concept}"):
                raw = r.get(f"cg:edge:{concept}::{effect}")
                if raw:
                    e = json.loads(raw)
                    e.pop("embedding", None)   # don't return vectors
                    edges.append(e)
                    traverse(effect, "effects", hops - 1)

        if direction in ("causes", "both"):
            for cause in r.smembers(f"cg:effects:{concept}"):
                raw = r.get(f"cg:edge:{cause}::{concept}")
                if raw:
                    e = json.loads(raw)
                    e.pop("embedding", None)
                    edges.append(e)
                    traverse(cause, "causes", hops - 1)

    traverse(req.concept, req.direction, req.depth)

    # Deduplicate
    seen  = set()
    dedup = []
    for e in edges:
        k = f"{e['cause']}::{e['effect']}"
        if k not in seen:
            seen.add(k)
            dedup.append(e)

    dedup.sort(key=lambda x: x.get("strength", 0) * x.get("evidence", 1),
               reverse=True)

    return BrainResponse(
        request_id = "causal",
        service    = "causal-graph",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "concept":    req.concept,
            "edges":      dedup[:20],
            "edge_count": len(dedup),
            "node_count": r.scard("cg:nodes")
        }
    )


@app.post("/predict")
async def predict_effects(body: dict):
    """Given a cause, predict most likely effects."""
    cause    = body.get("cause", "")
    effects  = r.smembers(f"cg:causes:{cause}")
    result   = []

    for effect in effects:
        raw = r.get(f"cg:edge:{cause}::{effect}")
        if raw:
            e = json.loads(raw)
            e.pop("embedding", None)
            result.append(e)

    result.sort(
        key=lambda x: x.get("strength", 0) * min(10, x.get("evidence", 1)),
        reverse=True
    )

    return {"cause": cause, "predicted_effects": result[:10]}


@app.get("/stats")
async def stats():
    return {
        "nodes":     r.scard("cg:nodes"),
        "domains":   list(r.smembers("cg:domains") or []),
    }
