# services/hippocampus/main.py
# ── Hippocampus: Episodic Memory Formation and Consolidation ──────────────────
# Owns the EdgeDB connection. Encodes experiences, recalls episodes,
# consolidates to neocortex (Elasticsearch) during idle / high-pressure states.

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, "/shared")

import edgedb
import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus

app = FastAPI(title="Hippocampus", version="1.0.0",
              description="Episodic memory formation and consolidation")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=30.0)

OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://host.docker.internal:11434")
NEOCORTEX_URL = os.getenv("NEOCORTEX_URL", "http://neocortex:8004")
OLLAMA_MODEL  = os.getenv("OLLAMA_LOCAL_MODEL",  "phi4")
EMBED_MODEL   = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── Embed client (uses embed-queue with Ollama fallback) ──────────────────────
from embed_client import embed as _embed_fn

async def embed(text: str) -> list:
    return await _embed_fn(text, service="hippocampus")

db: Optional[edgedb.AsyncIOClient] = None


@app.on_event("startup")
async def startup():
    global db
    dsn = os.getenv("EDGEDB_DSN", "edgedb://edgedb/main")
    tls = os.getenv("EDGEDB_CLIENT_TLS_SECURITY", "insecure")
    db  = edgedb.create_async_client(dsn=dsn, tls_security=tls)
    # Run migrations if needed
    try:
        await db.query("SELECT 1")
    except Exception as e:
        print(f"EdgeDB startup check: {e}")


@app.on_event("shutdown")
async def shutdown():
    await http.aclose()
    if db:
        await db.aclose()


# ── Utilities ─────────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / norm)


async def quick_summarize(content: str) -> str:
    """Summarize content locally without using Claude API."""
    if len(content.split()) <= 35:
        return content
    for attempt in range(2):
        try:
            resp = await http.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": f"Summarize in one concise sentence:\n{content[:600]}",
                    "stream": False,
                    "options": {"num_predict": 80}
                },
                timeout=10.0
            )
            return resp.json().get("response", content[:200]).strip()
        except (httpx.ConnectTimeout, httpx.ConnectError) as e:
            print(f"[hippocampus] ERROR: Ollama unreachable at {OLLAMA_URL} ({type(e).__name__}) — skipping summarize", flush=True)
            break
        except Exception:
            if attempt == 0:
                await asyncio.sleep(0.5)
    # Final fallback: mechanical truncation
    sentences = content.replace("! ", ". ").replace("? ", ". ").split(". ")
    return ". ".join(sentences[:2]) + "."


def classify_domain(concepts: list) -> str:
    tech    = {"python","code","api","database","algorithm","function","class",
               "neural","model","software","system","docker","server","deploy"}
    science = {"biology","physics","chemistry","math","science","research","data"}
    concept_set = {c.lower() for c in concepts}
    if concept_set & tech:    return "technology"
    if concept_set & science: return "science"
    return "general"


# ── Core functions ────────────────────────────────────────────────────────────

async def find_related_episodes(embedding: list,
                                  threshold: float = 0.72,
                                  limit: int = 6) -> list:
    """Find past episodes similar to a given embedding."""
    if not embedding:
        return []
    try:
        episodes = await db.query("""
            SELECT Episode {
                id, summary, embedding, importance, emotion_tag, timestamp
            }
            ORDER BY .timestamp DESC
            LIMIT 200
        """)
    except Exception as e:
        print(f"find_related_episodes error: {e}")
        return []

    query_vec = np.array(embedding, dtype=np.float32)
    scored    = []

    for ep in episodes:
        if ep.embedding:
            sim = cosine_similarity(query_vec, list(ep.embedding))
            if sim > threshold:
                scored.append({
                    "id":         str(ep.id),
                    "summary":    ep.summary or "",
                    "similarity": round(sim, 4),
                    "importance": float(ep.importance),
                    "emotion":    ep.emotion_tag
                })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]


async def link_episodes_bg(episode_id: str, related_ids: list):
    """Background task: create associative links between episodes."""
    for rid in related_ids:
        try:
            await db.execute("""
                UPDATE Episode FILTER .id = <uuid>$id
                SET { related_episodes += (SELECT Episode FILTER .id = <uuid>$rid) }
            """, id=episode_id, rid=rid)
        except Exception:
            pass


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_status = "unknown"
    if db:
        try:
            await db.query_single("SELECT 1")
            db_status = "connected"
        except Exception:
            db_status = "error"
    return {"status": "alive", "service": "hippocampus", "edgedb": db_status}


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


class EncodeRequest(BaseModel):
    request_id: str = "encode"
    session_id: str
    user_input: str = ""
    content:    str
    emotion:    str   = "neutral"
    importance: float = 0.5
    concepts:   list  = []


@app.post("/encode", response_model=BrainResponse)
async def encode_episode(req: EncodeRequest,
                          background: BackgroundTasks):
    """
    Encode a new experience as an episode in EdgeDB.
    Core hippocampus function: converts a moment into a memory.
    """
    start   = time.time()
    content = req.content or req.user_input

    embedding = await embed(content)
    summary   = await quick_summarize(content)
    related   = await find_related_episodes(embedding, limit=5) if embedding else []

    try:
        episode = await db.query_single("""
            INSERT Episode {
                content     := <str>$content,
                timestamp   := <datetime>$ts,
                importance  := <float64>$importance,
                emotion_tag := <str>$emotion,
                summary     := <str>$summary,
                embedding   := <array<float32>>$embedding,
                concepts    := (
                    FOR name IN array_unpack(<array<str>>$concepts)
                    UNION (
                        INSERT Concept { name := name }
                        UNLESS CONFLICT ON .name
                        ELSE (SELECT Concept FILTER .name = name)
                    )
                )
            }
        """,
            content   = content[:5000],
            ts        = datetime.now(timezone.utc),
            importance= req.importance,
            emotion   = req.emotion,
            summary   = summary,
            embedding = [float(x) for x in embedding],
            concepts  = req.concepts[:20]
        )
        episode_id = str(episode.id)

        # Link related episodes in the background
        if related:
            background.add_task(
                link_episodes_bg,
                episode_id,
                [r["id"] for r in related]
            )

        # Queue for consolidation if important enough
        if req.importance > 0.6:
            events.publish("memory.consolidate", {
                "episode_id": episode_id,
                "summary":    summary,
                "concepts":   req.concepts,
                "importance": req.importance,
                "emotion":    req.emotion,
                "session_id": req.session_id
            })

        return BrainResponse(
            request_id = req.request_id,
            service    = "hippocampus",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {
                "episode_id":      episode_id,
                "summary":         summary,
                "related_found":   len(related)
            }
        )

    except Exception as e:
        return BrainResponse(
            request_id = req.request_id,
            service    = "hippocampus",
            success    = False,
            latency_ms = round((time.time() - start) * 1000, 2),
            error      = str(e),
            data       = {}
        )


@app.post("/recall", response_model=BrainResponse)
async def recall(req: BrainRequest):
    """Recall relevant past episodes for a given query."""
    start     = time.time()
    embedding = await embed(req.user_input)
    episodes  = await find_related_episodes(embedding, limit=6) if embedding else []

    # Boost importance of accessed episodes (reconsolidation effect)
    async def boost_importance():
        for ep in episodes[:3]:
            try:
                await db.execute("""
                    UPDATE Episode FILTER .id = <uuid>$id
                    SET { importance := min(1.0, .importance * 1.04) }
                """, id=ep["id"])
            except Exception:
                pass

    asyncio.create_task(boost_importance())

    return BrainResponse(
        request_id = req.request_id,
        service    = "hippocampus",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"episodes": episodes}
    )


@app.post("/consolidate")
async def consolidate(max_items: int = 10):
    """
    Consolidation: extract semantic knowledge from important episodes,
    push to neocortex (Elasticsearch), mark episodes as consolidated.
    Mimics hippocampal replay during sleep.
    """
    start = time.time()
    consolidated = 0

    try:
        episodes = await db.query("""
            SELECT Episode {
                id, content, summary, importance, emotion_tag,
                concepts: { name }
            }
            FILTER .consolidated = false AND .importance > 0.58
            ORDER BY .importance DESC
            LIMIT <int64>$limit
        """, limit=max_items)
    except Exception as e:
        return {"error": str(e), "consolidated": 0}

    for ep in episodes:
        try:
            concepts = [c.name for c in (ep.concepts or [])]
            text     = ep.summary or ep.content[:400]
            embedding = await embed(text)
            if not embedding:
                continue  # skip if embeddings unavailable

            knowledge = {
                "content":          text,
                "domain":           classify_domain(concepts),
                "confidence":       min(1.0, float(ep.importance) + 0.15),
                "source_type":      "episode",
                "related_concepts": concepts,
                "embedding":        embedding,
                "summary":          text[:200]
            }

            # Push to neocortex
            resp = await http.post(f"{NEOCORTEX_URL}/store", json=knowledge)
            if resp.status_code == 200:
                await db.execute("""
                    UPDATE Episode FILTER .id = <uuid>$id
                    SET {
                        consolidated       := true,
                        consolidation_time := <datetime>$now
                    }
                """, id=str(ep.id), now=datetime.now(timezone.utc))
                consolidated += 1

        except Exception as e:
            print(f"Consolidation error for episode {ep.id}: {e}")

    return {
        "consolidated": consolidated,
        "latency_ms":   round((time.time() - start) * 1000, 2)
    }


@app.get("/stats")
async def stats():
    try:
        total         = await db.query_single("SELECT count(Episode)")
        unconsolidated = await db.query_single(
            "SELECT count(Episode FILTER .consolidated = false)"
        )
        skills = await db.query_single("SELECT count(Skill)")
        return {
            "total_episodes":  int(total),
            "unconsolidated":  int(unconsolidated),
            "total_skills":    int(skills)
        }
    except Exception as e:
        return {"error": str(e)}
