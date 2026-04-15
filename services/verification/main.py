# services/verification/main.py
# ── Verification: Epistemic Immune System ─────────────────────────────────────
# Three-stage verification pipeline:
#   1. Internal consistency — does this contradict known facts?
#   2. Multi-source corroboration — do independent sources agree?
#   3. Logical coherence — does this violate any known principles?
# Runs on every response before it reaches the user.

import asyncio
import hashlib
import json
import os
import sys
import time
from enum import Enum
from typing import Optional

sys.path.insert(0, "/shared")

import httpx
import numpy as np
import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus
from llm_provider import complete, complete_json, has_api_key

app = FastAPI(title="Verification", version="1.0.0",
              description="Epistemic immune system — fact checking and coherence")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=60.0)

NEOCORTEX_URL = os.getenv("NEOCORTEX_URL",    "http://neocortex:8004")
OLLAMA_URL    = os.getenv("OLLAMA_URL",        "http://host.docker.internal:11434")
EMBED_MODEL   = os.getenv("OLLAMA_EMBED_MODEL","nomic-embed-text")

# ── Verification result types ─────────────────────────────────────────────────

class VerificationStatus(str, Enum):
    CONFIRMED  = "confirmed"   # multiple independent sources agree
    PROBABLE   = "probable"    # one reliable source, no contradictions
    UNCERTAIN  = "uncertain"   # conflicting or low-confidence
    CONTESTED  = "contested"   # active disagreement between sources
    REFUTED    = "refuted"     # contradicted by higher-confidence source
    UNVERIFIED = "unverified"  # never checked

# ── Redis verification cache ──────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
CACHE_TTL = 3600    # 1 hour

try:
    _redis = redis.from_url(REDIS_URL, decode_responses=True)
    _redis.ping()
except Exception:
    _redis = None


def _cache_get(key: str) -> Optional[dict]:
    """Retrieve a cached verification result from Redis."""
    if _redis is None:
        return None
    try:
        raw = _redis.get(f"verify:{key}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _cache_set(key: str, result: dict) -> None:
    """Store a verification result in Redis with TTL."""
    if _redis is None:
        return
    try:
        _redis.setex(f"verify:{key}", CACHE_TTL, json.dumps(result))
    except Exception:
        pass


class VerifyRequest(BaseModel):
    request_id: str = "verify"
    session_id: str = ""
    claim:      str
    context:    str = ""
    domain:     str = "general"


@app.get("/health")
async def health():
    return {"status": "alive", "service": "verification", "version": "1.0.0"}


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


async def embed(text: str) -> list:
    """Generate embedding via local Ollama — 15s timeout per attempt."""
    for attempt in range(2):
        t0 = time.time()
        try:
            resp = await http.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text[:2000]},
                timeout=15.0
            )
            elapsed = (time.time() - t0) * 1000
            vec = resp.json()["embedding"]
            # Guard against zero-magnitude vectors (cosine similarity rejects them)
            if all(v == 0.0 for v in vec[:10]):
                print(f"[verification] embed returned zero vector after {elapsed:.0f}ms", flush=True)
                return []
            if elapsed > 3000:
                print(f"[verification] embed slow: {elapsed:.0f}ms", flush=True)
            return vec
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            print(f"[verification] embed attempt {attempt+1} failed after {elapsed:.0f}ms ({type(e).__name__}): {e}", flush=True)
            if attempt == 0:
                await asyncio.sleep(0.3)
    return []


async def search_neocortex(embedding: list, text: str = "",
                             domain: str = None, top_k: int = 12) -> list:
    """Search neocortex for relevant knowledge — fail fast on timeout."""
    for attempt in range(2):
        try:
            resp = await http.post(
                f"{NEOCORTEX_URL}/search",
                json={
                    "request_id": "verify",
                    "session_id": "verify",
                    "user_input": text
                },
                params={"domain": domain, "top_k": top_k},
                timeout=10.0
            )
            return resp.json().get("data", {}).get("results", [])
        except Exception as e:
            if attempt == 0:
                print(f"[verification] search_neocortex attempt 1 failed ({type(e).__name__}): {e}", flush=True)
                await asyncio.sleep(0.3)
            else:
                print(f"[verification] search_neocortex failed after 2 attempts", flush=True)
    return []


def generate_negation(claim: str) -> str:
    """Generate the logical negation of a claim."""
    pairs = [
        ("is not ", "is "), ("are not ", "are "),
        (" is ", " is not "), (" are ", " are not "),
        ("can ", "cannot "), ("cannot ", "can "),
        ("will ", "will not "), ("does ", "does not "),
        ("always", "never"), ("never", "always"),
    ]
    lower = claim.lower()
    for pos, neg in pairs:
        if pos in lower:
            return claim.lower().replace(pos, neg, 1)
    return f"not true that: {claim}"


async def check_internal_consistency(claim: str, domain: str) -> dict:
    """
    Stage 1: Does this claim contradict anything already in the neocortex?
    Searches for both the claim and its logical negation.
    """
    t0 = time.time()
    embedding        = await embed(claim)
    if not embedding:
        elapsed = (time.time() - t0) * 1000
        print(f"[verification] check_internal_consistency early-return (no embedding) after {elapsed:.0f}ms", flush=True)
        return {"supporting": [], "contradicting": [], "consistency_score": 0.5}

    negation         = generate_negation(claim)
    negation_embedding = await embed(negation)

    # Search for supporting evidence
    supporting = await search_neocortex(embedding, claim, domain, top_k=8)

    # Search for contradicting evidence (negation search)
    contradicting_raw = []
    if negation_embedding:
        contradicting_raw = await search_neocortex(
            negation_embedding, negation, domain, top_k=6
        )
    contradicting = [r for r in contradicting_raw
                     if float(r.get("score", 0)) > 0.72]

    consistency_score = max(
        0.0, 1.0 - len(contradicting) * 0.3
    )

    return {
        "supporting":       [s for s in supporting if float(s.get("score",0)) > 0.65],
        "contradicting":    contradicting,
        "consistency_score": round(consistency_score, 3)
    }


async def check_corroboration(claim: str, domain: str) -> dict:
    """
    Stage 2: Do independent sources agree with this claim?
    Independence across domains is stronger than same-domain agreement.
    """
    t0 = time.time()
    embedding = await embed(claim)
    if not embedding:
        elapsed = (time.time() - t0) * 1000
        print(f"[verification] check_corroboration early-return (no embedding) after {elapsed:.0f}ms", flush=True)
        return {"independent_domains": [], "corroborating_count": 0,
                "corroboration_score": 0.2, "top_sources": []}

    all_results = await search_neocortex(embedding, claim, top_k=20)

    # Group by domain
    domain_groups: dict = {}
    for item in all_results:
        d = item.get("domain", "unknown")
        domain_groups.setdefault(d, []).append(item)

    # Count domains with corroborating evidence
    corroborating_domains = [
        d for d, items in domain_groups.items()
        if any(float(i.get("score", 0)) > 0.65 for i in items)
        and d != domain
    ]

    count = len(corroborating_domains)
    score = min(1.0, count * 0.25 + 0.2)

    return {
        "independent_domains":  corroborating_domains,
        "corroborating_count":  count,
        "corroboration_score":  round(score, 3),
        "top_sources":          all_results[:4]
    }


async def check_logical_coherence(claim: str, context: str) -> dict:
    """
    Stage 3: Use LLM to check logical coherence.
    Is this claim internally consistent? Does it violate known principles?
    No external search — pure reasoning.
    """
    if not has_api_key():
        return {
            "is_coherent":       True,
            "coherence_score":   0.6,
            "logical_issues":    [],
            "hidden_assumptions": [],
            "is_falsifiable":    True,
            "reasoning":         "No API key — coherence check skipped"
        }

    prompt = f"""Evaluate the logical coherence of this claim briefly.

CLAIM: {claim[:400]}
CONTEXT: {context[:200] if context else "No additional context."}

Check:
1. Is this claim internally self-consistent?
2. Does it violate any known logical or physical principles?
3. Are there hidden assumptions that may be false?

Respond with JSON only:
{{
    "is_coherent": true or false,
    "coherence_score": 0.0 to 1.0,
    "logical_issues": ["list issues if any"],
    "hidden_assumptions": ["key assumptions"],
    "is_falsifiable": true or false,
    "reasoning": "one sentence explanation"
}}"""

    try:
        result = await complete_json(
            prompt=prompt,
            system="You are a logical coherence checker. Evaluate claims for internal consistency and logical validity.",
            model_tier="fast",
            max_tokens=300,
        )
        if result:
            return result
        return {
            "is_coherent":       True,
            "coherence_score":   0.55,
            "logical_issues":    [],
            "hidden_assumptions": [],
            "is_falsifiable":    True,
            "reasoning":         "Coherence check returned empty result"
        }
    except Exception as e:
        return {
            "is_coherent":       True,
            "coherence_score":   0.55,
            "logical_issues":    [],
            "hidden_assumptions": [],
            "is_falsifiable":    True,
            "reasoning":         f"Coherence check inconclusive: {str(e)[:50]}"
        }


@app.post("/verify", response_model=BrainResponse)
async def verify(req: VerifyRequest):
    """
    Full three-stage verification of a claim.
    Returns a VerificationStatus with confidence score and reasoning.
    Results are cached by claim hash for 1 hour.
    """
    start = time.time()
    k     = knobs.get()

    # Check cache first
    cache_key = hashlib.md5(req.claim.encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        return BrainResponse(
            request_id = req.request_id,
            service    = "verification",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {**cached, "from_cache": True}
        )

    # Run all three checks in parallel
    print(f"[verification] /verify starting 3 parallel checks...", flush=True)
    consistency_task    = asyncio.create_task(
        check_internal_consistency(req.claim, req.domain)
    )
    corroboration_task  = asyncio.create_task(
        check_corroboration(req.claim, req.domain)
    )
    coherence_task      = asyncio.create_task(
        check_logical_coherence(req.claim, req.context)
    )

    consistency, corroboration, coherence = await asyncio.gather(
        consistency_task, corroboration_task, coherence_task
    )
    gather_ms = round((time.time() - start) * 1000, 1)
    print(f"[verification] /verify 3 checks completed in {gather_ms:.0f}ms", flush=True)

    # ── Synthesize ────────────────────────────────────────────────────────────
    # Weighting: coherence 40% | consistency 35% | corroboration 25%
    coherence_score     = float(coherence.get("coherence_score", 0.5))
    consistency_score   = float(consistency.get("consistency_score", 0.5))
    corroboration_score = float(corroboration.get("corroboration_score", 0.25))

    combined = (
        coherence_score     * 0.40 +
        consistency_score   * 0.35 +
        corroboration_score * 0.25
    )

    contradictions = consistency.get("contradicting", [])
    logical_issues = coherence.get("logical_issues", [])

    # Determine status
    if not coherence.get("is_coherent", True):
        status   = VerificationStatus.REFUTED
        combined = min(combined, 0.2)

    elif len(contradictions) > 1:
        status = VerificationStatus.CONTESTED

    elif len(contradictions) == 1:
        status = VerificationStatus.UNCERTAIN

    elif combined > 0.78:
        status = VerificationStatus.CONFIRMED

    elif combined > 0.50:
        status = VerificationStatus.PROBABLE

    else:
        status = VerificationStatus.UNVERIFIED

    # Build reasoning string
    reasoning_parts = []
    if logical_issues:
        reasoning_parts.append(f"Logic: {'; '.join(logical_issues[:2])}")
    if contradictions:
        reasoning_parts.append(f"{len(contradictions)} contradiction(s) found")
    corr_count = corroboration.get("corroborating_count", 0)
    if corr_count > 0:
        reasoning_parts.append(
            f"Corroborated by {corr_count} independent domain(s)"
        )
    if not reasoning_parts:
        reasoning_parts.append(coherence.get("reasoning", "No issues detected"))

    result = {
        "claim":               req.claim[:200],
        "status":              status.value,
        "confidence":          round(combined, 3),
        "supporting_count":    len(consistency.get("supporting", [])),
        "contradicting_count": len(contradictions),
        "corroborating_domains": corroboration.get("independent_domains", []),
        "reasoning":           " | ".join(reasoning_parts),
        "hidden_assumptions":  coherence.get("hidden_assumptions", []),
        "requires_recheck":    status in [
            VerificationStatus.UNCERTAIN,
            VerificationStatus.CONTESTED
        ]
    }

    # Cache result
    _cache_set(cache_key, result)

    # Publish contested claims to event bus
    if status == VerificationStatus.CONTESTED:
        events.publish("verification.contested", {
            "claim":   req.claim[:200],
            "session": req.session_id,
            "domains": corroboration.get("independent_domains", [])
        })

    return BrainResponse(
        request_id = req.request_id,
        service    = "verification",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result
    )


@app.delete("/cache")
async def clear_cache():
    """Clear the verification cache."""
    if _redis is None:
        return {"cleared": 0}
    try:
        keys = _redis.keys("verify:*")
        count = len(keys)
        if keys:
            _redis.delete(*keys)
        return {"cleared": count}
    except Exception:
        return {"cleared": 0}


@app.get("/cache/stats")
async def cache_stats():
    count = 0
    if _redis is not None:
        try:
            count = len(_redis.keys("verify:*"))
        except Exception:
            pass
    return {
        "cached_items": count,
        "ttl_seconds":  CACHE_TTL,
        "backend":      "redis" if _redis is not None else "none"
    }
