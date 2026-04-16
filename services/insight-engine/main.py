# services/insight-engine/main.py
# ── Insight Engine: Epiphany & Insight Crystallization ────────────────────────
# Background process that detects when accumulated knowledge fragments
# form a complete picture — the "aha moment" / epiphany.
#
# Neuroscience basis (Kounios & Beeman 2014):
#   1. Unconscious recombination of previously disconnected information
#   2. Gamma wave burst in right anterior superior temporal gyrus
#   3. Occurs 0.3-0.5s before conscious awareness
#   4. Requires a critical mass of puzzle pieces — phase transition
#
# Architecture:
#   - Fragments are collected continuously from every interaction
#   - Crystallization check runs on /check or periodically
#   - When a near-complete subgraph (clique) is found that wasn't there before → EPIPHANY
#   - Epiphanies permanently alter knowledge (neocortex) and can shift knob baselines
#
# This is the only mechanism that permanently alters baseline personality knobs
# without manual intervention — the analog of a "life-changing realization."

import json
import os
import sys
import time
import hashlib
from typing import Optional
from collections import defaultdict

sys.path.insert(0, "/shared")

import redis
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import BrainResponse, ExperienceType
from events import EventBus

app = FastAPI(title="InsightEngine", version="1.0.0",
              description="Epiphany detection — background insight crystallization from accumulated fragments")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

events = EventBus()
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)
http   = httpx.AsyncClient(timeout=30.0)

NEOCORTEX_URL = os.getenv("NEOCORTEX_URL", "http://neocortex:8004")
KNOBS_URL     = os.getenv("KNOBS_URL",     "http://knobs:8090")

FRAGMENT_KEY     = "insight:fragments:{domain}"
KNOWN_CLIQUES    = "insight:known_cliques"
EPIPHANY_LOG     = "insight:epiphanies"
PENDING_NOTIFY   = "insight:pending_notify:{session_id}"
MAX_FRAGMENTS    = 500          # per domain
COMPLETENESS_TH  = 0.70        # clique completeness threshold
MIN_CLIQUE_SIZE  = 3            # minimum concepts to form an epiphany


# ── Data Models ───────────────────────────────────────────────────────────────

class Fragment(BaseModel):
    concept:     str
    connections: list[str] = []
    domain:      str       = "general"
    session_id:  str       = ""
    source:      str       = ""       # neocortex, cross-domain, causal, verification, etc.
    timestamp:   float     = Field(default_factory=time.time)


class IngestRequest(BaseModel):
    session_id:  str
    fragments:   list[Fragment] = []


class CheckRequest(BaseModel):
    session_id:  str       = ""
    domain:      str       = "general"
    max_checks:  int       = 5         # check up to N domains


class Epiphany(BaseModel):
    concepts:         list[str]
    connections:      list[tuple[str, str]] = []
    domain:           str
    novel_synthesis:  str
    trigger_fragment: str = ""
    confidence:       float = 0.0
    timestamp:        float = Field(default_factory=time.time)


# ── Graph Analysis ────────────────────────────────────────────────────────────

def _build_graph(fragments: list[dict]) -> dict[str, set[str]]:
    """Build adjacency list from fragments."""
    graph: dict[str, set[str]] = defaultdict(set)
    for frag in fragments:
        concept = frag.get("concept", "").lower().strip()
        if not concept:
            continue
        for conn in frag.get("connections", []):
            conn = conn.lower().strip()
            if conn and conn != concept:
                graph[concept].add(conn)
                graph[conn].add(concept)
    return graph


def _find_near_cliques(graph: dict[str, set[str]],
                       completeness_threshold: float = COMPLETENESS_TH,
                       min_size: int = MIN_CLIQUE_SIZE) -> list[set[str]]:
    """
    Find densely connected subgraphs (near-cliques).
    A near-clique is a set of nodes where the ratio of existing edges
    to possible edges exceeds the completeness threshold.
    Uses a greedy expansion from each node.
    """
    cliques = []
    visited_sets = set()
    nodes = list(graph.keys())

    for seed in nodes:
        # Start with seed + its neighbors
        candidates = {seed} | graph[seed]
        if len(candidates) < min_size:
            continue

        # Greedily build the densest subgraph
        best_clique = {seed}
        for candidate in sorted(candidates - {seed},
                                key=lambda n: len(graph[n] & candidates), reverse=True):
            test_set = best_clique | {candidate}
            # Count edges in the subgraph
            n = len(test_set)
            max_edges = n * (n - 1) / 2
            actual_edges = sum(
                1 for a in test_set for b in test_set
                if a < b and b in graph.get(a, set())
            )
            completeness = actual_edges / max_edges if max_edges > 0 else 0

            if completeness >= completeness_threshold:
                best_clique = test_set

        if len(best_clique) >= min_size:
            frozen = frozenset(best_clique)
            if frozen not in visited_sets:
                visited_sets.add(frozen)
                cliques.append(best_clique)

    return cliques


def _clique_hash(clique: set[str]) -> str:
    """Deterministic hash for a clique."""
    key = "|".join(sorted(clique))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _was_incomplete_before(clique: set[str], fragments: list[dict],
                           lookback: int = 5) -> Optional[str]:
    """
    Check if the clique was incomplete before the last `lookback` fragments.
    Returns the trigger fragment concept if the clique was just completed, else None.
    """
    if len(fragments) < lookback:
        return None

    # Get the concepts added in the last N fragments
    recent_concepts = set()
    for frag in fragments[-lookback:]:
        recent_concepts.add(frag.get("concept", "").lower().strip())
        for conn in frag.get("connections", []):
            recent_concepts.add(conn.lower().strip())

    # If any concept in the clique was recently added, the clique was just completed
    overlap = clique & recent_concepts
    if overlap:
        # The most recent one is the trigger
        for frag in reversed(fragments[-lookback:]):
            concept = frag.get("concept", "").lower().strip()
            if concept in overlap:
                return concept
        return overlap.pop()

    return None


def _describe_synthesis(clique: set[str], graph: dict[str, set[str]], domain: str) -> str:
    """Describe the novel synthesis this clique represents."""
    concepts = sorted(clique)
    connections = []
    for a in concepts:
        for b in concepts:
            if a < b and b in graph.get(a, set()):
                connections.append(f"{a} ↔ {b}")

    return (
        f"Insight crystallized in domain '{domain}': "
        f"concepts [{', '.join(concepts)}] form a densely connected cluster. "
        f"Key connections: {'; '.join(connections[:8])}. "
        f"This pattern was not previously recognized as a unified insight."
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    domains = r.keys("insight:fragments:*")
    known   = r.scard(KNOWN_CLIQUES)
    total   = r.llen(EPIPHANY_LOG)
    return {
        "status":           "alive",
        "service":          "insight-engine",
        "tracked_domains":  len(domains),
        "known_insights":   known,
        "total_epiphanies": total,
    }


@app.post("/ingest", response_model=BrainResponse)
async def ingest_fragments(req: IngestRequest):
    """
    Ingest new knowledge fragments from an interaction.
    Called by thalamus after each interaction (fire-and-forget).
    """
    start = time.time()
    stored = 0

    for frag in req.fragments:
        if not frag.concept.strip():
            continue
        key = FRAGMENT_KEY.format(domain=frag.domain)
        r.rpush(key, json.dumps({
            "concept":     frag.concept.lower().strip(),
            "connections": [c.lower().strip() for c in frag.connections if c.strip()],
            "session_id":  req.session_id,
            "source":      frag.source,
            "timestamp":   frag.timestamp,
        }))
        # Trim to max
        r.ltrim(key, -MAX_FRAGMENTS, -1)
        stored += 1

    return BrainResponse(
        request_id = "insight_ingest",
        service    = "insight-engine",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"fragments_stored": stored}
    )


@app.post("/check", response_model=BrainResponse)
async def check_crystallization(req: CheckRequest):
    """
    Run crystallization check on one or more domains.
    Called periodically by thalamus (fire-and-forget) or on demand.
    Detects if recent fragments completed any insight subgraphs.
    """
    start = time.time()
    epiphanies = []

    # Determine domains to check
    if req.domain and req.domain != "general":
        domains = [req.domain]
    else:
        keys = r.keys("insight:fragments:*")
        domains = [k.replace("insight:fragments:", "") for k in keys]
    domains = domains[:req.max_checks]

    for domain in domains:
        key = FRAGMENT_KEY.format(domain=domain)
        raw_fragments = r.lrange(key, 0, -1)
        if len(raw_fragments) < MIN_CLIQUE_SIZE:
            continue

        fragments = [json.loads(f) for f in raw_fragments]
        graph = _build_graph(fragments)

        if len(graph) < MIN_CLIQUE_SIZE:
            continue

        cliques = _find_near_cliques(graph)

        for clique in cliques:
            chash = _clique_hash(clique)
            # Skip if already known
            if r.sismember(KNOWN_CLIQUES, chash):
                continue

            # Check if recently completed
            trigger = _was_incomplete_before(clique, fragments)
            if trigger is None:
                # Not recently completed — might be old, just record as known
                r.sadd(KNOWN_CLIQUES, chash)
                continue

            # EPIPHANY!
            synthesis = _describe_synthesis(clique, graph, domain)
            confidence = min(1.0, len(clique) / 8.0 + 0.3)

            epiphany = Epiphany(
                concepts         = sorted(clique),
                connections      = [(a, b) for a in clique for b in clique
                                    if a < b and b in graph.get(a, set())],
                domain           = domain,
                novel_synthesis  = synthesis,
                trigger_fragment = trigger,
                confidence       = round(confidence, 3),
            )
            epiphanies.append(epiphany)

            # Mark as known
            r.sadd(KNOWN_CLIQUES, chash)

            # Log the epiphany
            r.lpush(EPIPHANY_LOG, json.dumps(epiphany.dict()))
            r.ltrim(EPIPHANY_LOG, 0, 199)  # keep last 200

            # Queue notification for the session
            if req.session_id:
                r.lpush(
                    PENDING_NOTIFY.format(session_id=req.session_id),
                    json.dumps(epiphany.dict())
                )
                r.expire(
                    PENDING_NOTIFY.format(session_id=req.session_id),
                    86400  # 24h TTL
                )

            # Publish event
            events.publish("epiphany.crystallized", {
                "experience_type": ExperienceType.EPIPHANY_EVENT,
                "content":         synthesis,
                "intensity":       0.95,
                "session_id":      req.session_id,
                "timestamp":       time.time(),
                "metadata": {
                    "concepts":  sorted(clique),
                    "domain":    domain,
                    "trigger":   trigger,
                    "confidence": confidence,
                }
            })

            # Also publish as an experience
            events.publish("experience.new", {
                "experience_type": ExperienceType.EPIPHANY_EVENT,
                "content":         synthesis,
                "intensity":       0.95,
                "session_id":      req.session_id,
                "timestamp":       time.time(),
                "metadata": {
                    "concepts":  sorted(clique),
                    "domain":    domain,
                    "trigger":   trigger,
                    "confidence": confidence,
                }
            })

            # Try to store as first-class knowledge in neocortex
            try:
                await http.post(f"{NEOCORTEX_URL}/store", json={
                    "request_id": "epiphany",
                    "session_id": req.session_id,
                    "content":    synthesis,
                    "domain":     domain,
                    "source":     "epiphany",
                    "importance": 0.95,
                    "concepts":   sorted(clique),
                }, timeout=10.0)
            except Exception as e:
                print(f"[insight-engine] Failed to store epiphany in neocortex: {e}",
                      flush=True)

    return BrainResponse(
        request_id = "insight_check",
        service    = "insight-engine",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "domains_checked":  len(domains),
            "epiphanies_found": len(epiphanies),
            "epiphanies":       [ep.dict() for ep in epiphanies],
        }
    )


@app.get("/pending/{session_id}")
async def get_pending_epiphanies(session_id: str):
    """
    Check if there are pending epiphany notifications for a session.
    Called by thalamus at the start of each interaction to inject
    "I've had a realization..." into the context.
    Returns and clears the queue.
    """
    key = PENDING_NOTIFY.format(session_id=session_id)
    raw = r.lrange(key, 0, -1)
    r.delete(key)

    epiphanies = [json.loads(item) for item in raw] if raw else []
    return {
        "session_id":  session_id,
        "has_pending":  len(epiphanies) > 0,
        "epiphanies":  epiphanies,
    }


@app.get("/history")
async def epiphany_history(limit: int = 20):
    """Return recent epiphanies across all sessions."""
    raw = r.lrange(EPIPHANY_LOG, 0, limit - 1)
    return {
        "epiphanies": [json.loads(item) for item in raw] if raw else [],
        "total":      r.llen(EPIPHANY_LOG),
    }
