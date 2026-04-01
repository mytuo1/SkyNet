# ── Causal Graph: Explicit Cause-Effect Knowledge ─────────────────────────────
# Tracks causal relationships discovered during interactions.
# When the system learns "X causes Y" it stores this explicitly
# in a directed graph, not just as a text embedding.
# This is the backbone of predictive world modeling.

import asyncio
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/shared")

import httpx
import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus

app    = FastAPI(title="CausalGraph", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
http   = httpx.AsyncClient(timeout=30.0)
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = os.getenv("FAST_CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Causal graph stored in Redis as adjacency structure
# Key: "cg:node:{concept}" → JSON with outgoing edges
# Key: "cg:edge:{cause}::{effect}" → JSON with relationship details
# Key: "cg:evidence:{cause}::{effect}" → count of supporting observations


class CausalEdge(BaseModel):
    cause:       str
    effect:      str
    mechanism:   str   = ""    # how/why this causes that
    strength:    float = 0.5   # 0=weak, 1=strong
    domain:      str   = "general"
    evidence:    int   = 1     # times this relationship was observed
    direction:   str   = "forward"  # forward | bidirectional | probabilistic


class ExtractRequest(BaseModel):
    text:       str
    session_id: str = ""
    domain:     str = "general"


class QueryRequest(BaseModel):
    concept:    str
    depth:      int   = 2      # how many hops to follow
    direction:  str   = "both" # causes | effects | both


@app.get("/health")
async def health():
    return {"status": "alive", "service": "causal-graph"}


@app.post("/extract")
async def extract_causal_relations(req: ExtractRequest):
    """
    Parse text and extract explicit causal relationships.
    Uses Claude to identify cause→effect pairs.
    """
    if not ANTHROPIC_KEY:
        return {"extracted": [], "reason": "no API key"}
    
    prompt = f"""Extract all explicit causal relationships from this text.
Look for: causes, leads to, results in, because, therefore, enables, prevents, increases, decreases.

TEXT: {req.text[:800]}

Respond with JSON:
{{
    "causal_relations": [
        {{
            "cause": "the thing that causes",
            "effect": "the thing caused",
            "mechanism": "how or why this works",
            "strength": 0.0 to 1.0,
            "direction": "forward | bidirectional | probabilistic"
        }}
    ]
}}"""
    
    try:
        resp = await http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json"
            },
            json={
                "model":      CLAUDE_MODEL,
                "max_tokens": 500,
                "system":     "Extract causal relationships. Respond only with valid JSON.",
                "messages":   [{"role": "user", "content": prompt}]
            }
        )
        
        text = resp.json()["content"][0]["text"].strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        data     = json.loads(text)
        relations = data.get("causal_relations", [])
        
        # Store each relation in the graph
        stored = []
        for rel in relations:
            if rel.get("cause") and rel.get("effect"):
                await store_edge(CausalEdge(
                    cause     = rel["cause"],
                    effect    = rel["effect"],
                    mechanism = rel.get("mechanism", ""),
                    strength  = float(rel.get("strength", 0.5)),
                    domain    = req.domain,
                    direction = rel.get("direction", "forward")
                ))
                stored.append(rel)
        
        return {"extracted": stored, "count": len(stored)}
    
    except Exception as e:
        return {"extracted": [], "error": str(e)}


async def store_edge(edge: CausalEdge):
    """Store a causal edge in the Redis graph."""
    edge_key = f"cg:edge:{edge.cause}::{edge.effect}"
    
    existing = r.get(edge_key)
    if existing:
        data     = json.loads(existing)
        evidence = data.get("evidence", 1) + 1
        # Update strength as running average
        new_strength = (data["strength"] * (evidence - 1) + edge.strength) / evidence
        data.update({
            "strength": round(new_strength, 3),
            "evidence": evidence,
            "updated":  time.time()
        })
    else:
        data = {
            **edge.dict(),
            "created": time.time(),
            "updated": time.time()
        }
        evidence = 1
    
    r.set(edge_key, json.dumps(data))
    
    # Add to adjacency lists
    r.sadd(f"cg:causes:{edge.cause}", edge.effect)
    r.sadd(f"cg:effects:{edge.effect}", edge.cause)
    
    # Add node index
    r.sadd("cg:nodes", edge.cause)
    r.sadd("cg:nodes", edge.effect)


@app.post("/query", response_model=BrainResponse)
async def query_causal_graph(req: QueryRequest):
    """
    Query the causal graph for what causes or is caused by a concept.
    Traverses up to `depth` hops.
    """
    start  = time.time()
    result = {"causes": [], "effects": [], "graph": {}}
    
    visited = set()
    
    def get_edges_from(concept: str, direction: str, hops_left: int):
        """Recursively traverse the causal graph."""
        if hops_left == 0 or concept in visited:
            return []
        visited.add(concept)
        
        edges = []
        
        if direction in ("effects", "both"):
            effects = r.smembers(f"cg:causes:{concept}")
            for effect in effects:
                edge_key = f"cg:edge:{concept}::{effect}"
                edge_data = r.get(edge_key)
                if edge_data:
                    e = json.loads(edge_data)
                    edges.append(e)
                    # Recurse
                    edges.extend(
                        get_edges_from(effect, "effects", hops_left - 1)
                    )
        
        if direction in ("causes", "both"):
            causes = r.smembers(f"cg:effects:{concept}")
            for cause in causes:
                edge_key = f"cg:edge:{cause}::{concept}"
                edge_data = r.get(edge_key)
                if edge_data:
                    e = json.loads(edge_data)
                    edges.append(e)
                    edges.extend(
                        get_edges_from(cause, "causes", hops_left - 1)
                    )
        
        return edges
    
    edges = get_edges_from(req.concept, req.direction, req.depth)
    
    # Organize into causes and effects
    for e in edges:
        if e.get("effect") == req.concept or \
           any(e.get("effect") == req.concept for e in edges):
            result["causes"].append(e)
        if e.get("cause") == req.concept:
            result["effects"].append(e)
    
    result["graph"] = {
        "node":       req.concept,
        "all_edges":  edges,
        "edge_count": len(edges)
    }
    
    return BrainResponse(
        request_id = "causal",
        service    = "causal-graph",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result
    )


@app.post("/predict")
async def predict_effects(cause: str, context: str = ""):
    """
    Given a cause, predict likely effects using the stored causal graph
    combined with Claude reasoning.
    """
    # Get direct effects from graph
    known_effects = []
    effects_set   = r.smembers(f"cg:causes:{cause}")
    
    for effect in effects_set:
        edge_key = f"cg:edge:{cause}::{effect}"
        e = r.get(edge_key)
        if e:
            known_effects.append(json.loads(e))
    
    known_effects.sort(key=lambda x: x.get("strength", 0) * x.get("evidence", 1),
                       reverse=True)
    
    return {
        "cause":         cause,
        "known_effects": known_effects[:10],
        "graph_size":    r.scard("cg:nodes"),
        "context":       context
    }
