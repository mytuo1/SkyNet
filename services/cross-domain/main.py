# services/cross-domain/main.py
# ── Cross-Domain Reasoning: Structural Analogy Engine ────────────────────────
# Finds problems in other domains with the same logical shape.
# Extracts abstract patterns from knowledge.
# Maps solutions from one domain onto problems in another.
# This is how insight happens — recognizing that two different
# problems are structurally identical.

import asyncio
import json
import os
import sys
import time
from enum import Enum
from typing import Optional

sys.path.insert(0, "/shared")

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus
from llm_provider import complete_json, has_api_key

app = FastAPI(title="CrossDomain", version="1.0.0",
              description="Structural analogy and cross-domain reasoning")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=60.0)

NEOCORTEX_URL = os.getenv("NEOCORTEX_URL",    "http://neocortex:8004")
OLLAMA_URL    = os.getenv("OLLAMA_URL",        "http://host.docker.internal:11434")
EMBED_MODEL   = os.getenv("OLLAMA_EMBED_MODEL","nomic-embed-text")

# ── Embed client (uses embed-queue with Ollama fallback) ──────────────────────
from embed_client import embed as _embed_fn

async def embed(text: str) -> list:
    return await _embed_fn(text, service="cross-domain")

# ── Pattern types ─────────────────────────────────────────────────────────────

class PatternType(str, Enum):
    FLOW_CONSTRAINT   = "flow_constraint"    # resource hits bottleneck
    FEEDBACK_LOOP     = "feedback_loop"      # output affects input
    EMERGENCE         = "emergence"          # parts create whole
    PHASE_TRANSITION  = "phase_transition"   # gradual → sudden change
    HIERARCHY         = "hierarchy"          # nested levels
    COMPETITION       = "competition"        # limited resource, multiple agents
    COOPERATION       = "cooperation"        # coordination yields benefit
    THRESHOLD         = "threshold"          # nothing until limit, then everything
    OSCILLATION       = "oscillation"        # cycles and pendulum behavior
    DECAY             = "decay"              # gradual loss over time
    CATALYSIS         = "catalysis"          # enabler without being consumed
    SYMMETRY_BREAKING = "symmetry_breaking"  # uniform → differentiated

ADJACENT_PATTERNS = {
    PatternType.FEEDBACK_LOOP:   [PatternType.OSCILLATION, PatternType.DECAY],
    PatternType.COMPETITION:     [PatternType.THRESHOLD, PatternType.FLOW_CONSTRAINT],
    PatternType.COOPERATION:     [PatternType.EMERGENCE, PatternType.CATALYSIS],
    PatternType.PHASE_TRANSITION:[PatternType.THRESHOLD, PatternType.SYMMETRY_BREAKING],
    PatternType.EMERGENCE:       [PatternType.HIERARCHY, PatternType.COOPERATION],
    PatternType.THRESHOLD:       [PatternType.PHASE_TRANSITION, PatternType.COMPETITION],
}

EXTRACTION_PROMPT = """Analyze the logical STRUCTURE of this text.
Extract the abstract pattern — ignore domain-specific words.
Focus only on structural relationships.

TEXT: {content}

Domain: {domain}

Respond with JSON only (no markdown):
{{
    "pattern_type": one of [{pattern_types}],
    "entities": ["abstract role names, not domain terms"],
    "relationships": ["entity A does X to entity B"],
    "constraints": ["what limits or bounds the system"],
    "dynamics": ["how it changes over time"],
    "analogy_domains": ["2-3 other domains where this pattern appears"],
    "confidence": 0.0 to 1.0
}}

Example for "immune cells patrol the body and attack foreign proteins":
{{
    "pattern_type": "competition",
    "entities": ["defender", "intruder", "territory"],
    "relationships": ["defender patrols territory", "defender neutralizes intruder"],
    "constraints": ["defender resources finite", "intruder can replicate"],
    "dynamics": ["detection triggers response", "response scales with threat"],
    "analogy_domains": ["cybersecurity", "military strategy", "economics"],
    "confidence": 0.91
}}"""


class FindAnalogiesRequest(BaseModel):
    request_id:    str = "crossdomain"
    session_id:    str = ""
    user_input:    str
    source_domain: str = "general"
    top_k:         int = 5


class RegisterPatternRequest(BaseModel):
    content: str
    domain:  str


@app.get("/health")
async def health():
    return {"status": "alive", "service": "cross-domain", "version": "1.0.0"}


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


def cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / norm) if norm > 1e-8 else 0.0


async def extract_pattern(content: str, domain: str) -> Optional[dict]:
    """
    Use LLM to extract the abstract logical pattern from content.
    Returns None if extraction fails or no API key.
    """
    if not has_api_key():
        return None

    pattern_types = ", ".join([pt.value for pt in PatternType])
    prompt = EXTRACTION_PROMPT.format(
        content       = content[:600],
        domain        = domain,
        pattern_types = pattern_types
    )

    try:
        data = await complete_json(prompt=prompt, model_tier="fast", max_tokens=400)

        if not data:
            return None

        # Build abstract text representation for embedding
        abstract = " ".join([
            data.get("pattern_type", ""),
            " ".join(data.get("entities", [])),
            " ".join(data.get("relationships", [])),
            " ".join(data.get("dynamics", []))
        ])

        data["embedding"]      = await embed(abstract)
        data["source_content"] = content[:300]
        data["domain"]         = domain

        return data

    except Exception as e:
        print(f"Pattern extraction error: {e}")
        return None


async def search_similar_patterns(pattern: dict, top_k: int = 20) -> list:
    """Search neocortex for structurally similar patterns."""
    try:
        resp = await http.post(
            f"{NEOCORTEX_URL}/search-patterns",
            json={
                "embedding":    pattern.get("embedding", []),
                "pattern_type": pattern.get("pattern_type"),
                "top_k":        top_k
            }
        )
        return resp.json().get("patterns", [])
    except Exception:
        return []


def generate_mapping(source: dict, target: dict) -> dict:
    """Map concepts between two analogous domains."""
    source_entities = source.get("entities", [])
    target_entities = target.get("entities", [])

    mapping = {}
    for i, se in enumerate(source_entities):
        if i < len(target_entities):
            mapping[se] = target_entities[i]

    shared_dynamics = list(
        set(source.get("dynamics", [])) & set(target.get("dynamics", []))
    )

    return {
        "entity_mapping":   mapping,
        "shared_dynamics":  shared_dynamics,
        "transfer_insight": (
            f"In {target.get('domain', 'other domain')}, "
            f"the solution involved: "
            f"{'; '.join(target.get('relationships', [])[:2])}"
        )
    }


@app.post("/find-analogies", response_model=BrainResponse)
async def find_analogies(req: FindAnalogiesRequest):
    """
    Given a problem in source_domain, find structurally analogous
    problems and solutions from other domains.
    The core function of cross-domain reasoning.
    """
    start = time.time()
    k     = knobs.get()

    # If analogy seeking is turned down, skip
    if k.analogy_seeking < 0.2:
        return BrainResponse(
            request_id = req.request_id,
            service    = "cross-domain",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"analogies": [], "reason": "analogy_seeking knob low"}
        )

    # Extract pattern from the query
    t_extract = time.time()
    query_pattern = await extract_pattern(req.user_input, req.source_domain)
    extract_ms = round((time.time() - t_extract) * 1000, 1)
    if extract_ms > 3000:
        print(f"[cross-domain] extract_pattern took {extract_ms:.0f}ms (includes LLM + embed)", flush=True)

    if not query_pattern:
        return BrainResponse(
            request_id = req.request_id,
            service    = "cross-domain",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"analogies": [], "reason": "pattern extraction failed"}
        )

    # Search for similar patterns in neocortex
    candidates = await search_similar_patterns(query_pattern, top_k=30)

    query_vec = np.array(query_pattern.get("embedding") or [0.0]*768,
                         dtype=np.float32)

    analogies = []
    for candidate in candidates:
        # Skip same domain
        if candidate.get("domain") == req.source_domain:
            continue

        # Compute structural similarity
        cand_emb = candidate.get("embedding")
        if not cand_emb:
            continue

        sim = cosine_similarity(query_vec, cand_emb)

        if sim > 0.55:
            mapping = generate_mapping(query_pattern, candidate)
            analogies.append({
                "source_domain":   candidate.get("domain", "unknown"),
                "source_content":  candidate.get("source_content", ""),
                "pattern_type":    candidate.get("pattern_type", ""),
                "similarity":      round(sim, 4),
                "confidence":      round(
                    float(candidate.get("confidence", 0.8)) * sim, 3
                ),
                "concept_mapping": mapping,
                "entities":        candidate.get("entities", []),
                "relationships":   candidate.get("relationships", [])
            })

    analogies.sort(key=lambda x: x["similarity"], reverse=True)
    top_analogies = analogies[:req.top_k]

    # Publish analogical insight if strong analogy found
    if top_analogies and top_analogies[0]["similarity"] > 0.80:
        events.publish("analogical.insight", {
            "query":       req.user_input[:100],
            "domain":      top_analogies[0]["source_domain"],
            "pattern":     top_analogies[0]["pattern_type"],
            "similarity":  top_analogies[0]["similarity"],
            "session_id":  req.session_id
        })

    # Also store this query's pattern for future use
    asyncio.create_task(
        store_pattern(query_pattern, req.source_domain)
    )

    return BrainResponse(
        request_id = req.request_id,
        service    = "cross-domain",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "analogies":     top_analogies,
            "pattern_found": query_pattern.get("pattern_type"),
            "query_entities": query_pattern.get("entities", []),
            "total_candidates": len(candidates)
        }
    )


async def store_pattern(pattern: dict, domain: str):
    """Store extracted pattern in neocortex for future retrieval."""
    # Skip if embedding is empty/zero — neocortex cosine index rejects zero vectors
    emb = pattern.get("embedding", [])
    if not emb:
        return
    try:
        await http.post(
            f"{NEOCORTEX_URL}/store-pattern",
            json={
                "pattern_type":  pattern.get("pattern_type"),
                "domain":        domain,
                "entities":      pattern.get("entities", []),
                "relationships": pattern.get("relationships", []),
                "constraints":   pattern.get("constraints", []),
                "dynamics":      pattern.get("dynamics", []),
                "source_content":pattern.get("source_content", ""),
                "confidence":    pattern.get("confidence", 0.7),
                "embedding":     pattern.get("embedding", [])
            }
        )
        events.publish("pattern.registered", {
            "pattern_type": pattern.get("pattern_type"),
            "domain":       domain
        })
    except Exception as e:
        print(f"Store pattern error: {e}")


@app.post("/register-pattern", response_model=BrainResponse)
async def register_pattern(req: RegisterPatternRequest):
    """
    Manually register a piece of knowledge as a searchable pattern.
    Called when new knowledge enters the neocortex.
    """
    start   = time.time()
    pattern = await extract_pattern(req.content, req.domain)

    if pattern:
        await store_pattern(pattern, req.domain)
        return BrainResponse(
            request_id = "register",
            service    = "cross-domain",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {
                "registered":   True,
                "pattern_type": pattern.get("pattern_type")
            }
        )

    return BrainResponse(
        request_id = "register",
        service    = "cross-domain",
        success    = False,
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"registered": False}
    )
