# services/embed-queue/main.py
# ── Embed Queue: Centralized Embedding Service ─────────────────────────────────
# All brain services call this instead of Ollama directly.
#
# Architecture:
#   SYNC path  (POST /embed)    — cache-first, then Ollama.  Used for search.
#   ASYNC path (POST /enqueue)  — Redis Streams MQ.  Used for writes (store/encode).
#                                  Background workers drain the stream, embed via
#                                  Ollama, and POST the result back to a callback URL.
#   BATCH path (POST /embed/batch) — parallel sync embeds in one call.
#
# Redis Streams give us:
#   - Consumer groups with ACK (at-least-once delivery)
#   - Backpressure via XLEN monitoring
#   - Persistence (survives Redis restart via AOF/RDB)
#   - Dead-letter handling via XPENDING + XCLAIM
#   - Bounded memory via XTRIM
#
# Cache layer (regular Redis keys):
#   - Key: emb:{sha256_prefix}  TTL: 6 hours
#   - In-flight dedup via per-key asyncio locks

import asyncio
import hashlib
import json
import os
import time
import uuid
from typing import Optional

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="EmbedQueue", version="1.0.0",
              description="Centralized embedding — Redis Streams MQ + cache + dedup")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_URL   = os.getenv("OLLAMA_URL",          "http://host.docker.internal:11434")
EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")
REDIS_URL    = os.getenv("REDIS_URL",           "redis://redis:6379")
CACHE_TTL    = int(os.getenv("EMBED_CACHE_TTL", "21600"))   # 6 hours
MAX_TEXT_LEN = 2000
STREAM_NAME  = "embed_stream"           # Redis Stream key
GROUP_NAME   = "embed_workers"           # Consumer group
CONSUMER_ID  = f"worker-{os.getpid()}"   # Unique per process
MAX_STREAM   = 50_000                    # XTRIM MAXLEN — bounds memory
RESULT_TTL   = 3600                      # Async job results kept 1 hour
WORKERS      = int(os.getenv("EMBED_QUEUE_WORKERS", "2"))  # concurrent stream consumers

# ── Globals ───────────────────────────────────────────────────────────────────

http_client:  Optional[httpx.AsyncClient]  = None
redis_client: Optional[aioredis.Redis]     = None

_key_locks: dict[str, asyncio.Lock] = {}
_key_locks_guard = asyncio.Lock()
_shutdown = False

_stats = {
    "cache_hits":       0,
    "cache_misses":     0,
    "ollama_calls":     0,
    "ollama_errors":    0,
    "sync_requests":    0,
    "async_enqueued":   0,
    "async_completed":  0,
    "async_failed":     0,
    "callbacks_ok":     0,
    "callbacks_failed": 0,
    "avg_ollama_ms":    0.0,
    "_ollama_total_ms": 0.0,
}


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global http_client, redis_client
    http_client  = httpx.AsyncClient(timeout=30.0)
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=False)

    # Create consumer group (idempotent — MKSTREAM creates the stream if absent)
    try:
        await redis_client.xgroup_create(STREAM_NAME, GROUP_NAME,
                                          id="0", mkstream=True)
        print(f"[embed-queue] Created consumer group '{GROUP_NAME}'", flush=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            pass   # already exists
        else:
            raise

    # Launch background tasks
    asyncio.create_task(_warmup())
    asyncio.create_task(_cleanup_locks())
    for i in range(WORKERS):
        asyncio.create_task(_stream_worker(f"{CONSUMER_ID}-{i}"))
    asyncio.create_task(_reclaim_stale_jobs())
    print(f"[embed-queue] Started — model={EMBED_MODEL} cache_ttl={CACHE_TTL}s "
          f"workers={WORKERS} max_stream={MAX_STREAM}", flush=True)


@app.on_event("shutdown")
async def shutdown():
    global _shutdown
    _shutdown = True
    if http_client:
        await http_client.aclose()
    if redis_client:
        await redis_client.aclose()


async def _warmup():
    """Ping Ollama to pre-load the embedding model into memory."""
    for attempt in range(5):
        try:
            t0 = time.time()
            resp = await http_client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": "warmup", "keep_alive": -1},
                timeout=60.0
            )
            elapsed = round((time.time() - t0) * 1000)
            if resp.status_code == 200:
                print(f"[embed-queue] Ollama warmup OK — {EMBED_MODEL} loaded ({elapsed}ms)", flush=True)
                return
        except Exception as e:
            print(f"[embed-queue] Ollama warmup attempt {attempt+1}/5: {e}", flush=True)
        await asyncio.sleep(3)
    print("[embed-queue] Ollama warmup failed — will retry on first request", flush=True)


async def _cleanup_locks():
    """Periodically clean up idle per-key locks to prevent memory leaks."""
    while not _shutdown:
        await asyncio.sleep(300)
        async with _key_locks_guard:
            idle = [k for k, lock in _key_locks.items() if not lock.locked()]
            for k in idle:
                del _key_locks[k]
            if idle:
                print(f"[embed-queue] Cleaned {len(idle)} idle locks", flush=True)


# ── Core Embedding Logic ──────────────────────────────────────────────────────

def _cache_key(text: str) -> str:
    h = hashlib.sha256(text[:MAX_TEXT_LEN].encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"emb:{h}"


async def _embed_via_ollama(text: str, source: str = "") -> list:
    """Call Ollama for a single embedding. Returns [] on failure.
    Every call is logged with timing so docker logs show the full picture."""
    t0 = time.time()
    preview = text[:60].replace("\n", " ")
    try:
        resp = await http_client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:MAX_TEXT_LEN], "keep_alive": -1},
            timeout=15.0
        )
        elapsed = (time.time() - t0) * 1000
        vec = resp.json().get("embedding", [])
        if not vec or all(v == 0.0 for v in vec[:10]):
            print(f"[embed-queue] OLLAMA src={source} ZERO_VEC {elapsed:.0f}ms "
                  f"\"{preview}\"", flush=True)
            return []
        _stats["ollama_calls"] += 1
        _stats["_ollama_total_ms"] += elapsed
        _stats["avg_ollama_ms"] = round(
            _stats["_ollama_total_ms"] / max(_stats["ollama_calls"], 1), 1)
        tag = "SLOW" if elapsed > 3000 else "OK"
        print(f"[embed-queue] OLLAMA src={source} {tag} {elapsed:.0f}ms "
              f"dims={len(vec)} \"{preview}\"", flush=True)
        return vec
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        _stats["ollama_errors"] += 1
        print(f"[embed-queue] OLLAMA src={source} ERROR {elapsed:.0f}ms "
              f"{type(e).__name__}: {e} \"{preview}\"", flush=True)
        return []


async def get_embedding(text: str, source: str = "") -> tuple[list, bool]:
    """Sync embedding with Redis cache + in-flight dedup."""
    _stats["sync_requests"] += 1
    key = _cache_key(text)

    # Check cache
    try:
        cached = await redis_client.get(key)
        if cached:
            _stats["cache_hits"] += 1
            return json.loads(cached), True
    except Exception:
        pass

    _stats["cache_misses"] += 1

    # Per-key lock for dedup
    async with _key_locks_guard:
        if key not in _key_locks:
            _key_locks[key] = asyncio.Lock()
        lock = _key_locks[key]

    async with lock:
        # Double-check after acquiring lock
        try:
            cached = await redis_client.get(key)
            if cached:
                _stats["cache_hits"] += 1
                return json.loads(cached), True
        except Exception:
            pass

        vec = await _embed_via_ollama(text, source=source)
        if vec:
            try:
                await redis_client.setex(key, CACHE_TTL, json.dumps(vec))
            except Exception:
                pass
        return vec, False


# ── Redis Streams — Async Queue Workers ───────────────────────────────────────

async def _stream_worker(consumer: str):
    """
    Background worker: reads from Redis Stream, embeds, delivers via callback.
    Uses XREADGROUP for consumer-group semantics (at-least-once, ACK-based).
    """
    print(f"[embed-queue] Stream worker '{consumer}' started", flush=True)
    while not _shutdown:
        try:
            # Block up to 2s waiting for new messages
            messages = await redis_client.xreadgroup(
                GROUP_NAME, consumer, {STREAM_NAME: ">"}, count=5, block=2000
            )
            if not messages:
                continue

            for stream_key, entries in messages:
                for msg_id, fields in entries:
                    await _process_stream_job(msg_id, fields, consumer)

        except aioredis.ConnectionError:
            await asyncio.sleep(1)
        except Exception as e:
            print(f"[embed-queue] Worker '{consumer}' error: {e}", flush=True)
            await asyncio.sleep(1)

    print(f"[embed-queue] Stream worker '{consumer}' stopped", flush=True)


async def _process_stream_job(msg_id: bytes, fields: dict, consumer: str):
    """Process one async embed job from the stream. Logs every call with timing."""
    t0            = time.time()
    text          = (fields.get(b"text") or b"").decode("utf-8", errors="replace")
    job_id        = (fields.get(b"job_id") or b"").decode()
    source        = (fields.get(b"source") or b"unknown").decode()
    callback_url  = (fields.get(b"callback_url") or b"").decode()
    callback_data = (fields.get(b"callback_payload") or b"{}").decode()
    preview       = text[:60].replace("\n", " ")

    if not text:
        await redis_client.xack(STREAM_NAME, GROUP_NAME, msg_id)
        print(f"[embed-queue] ASYNC src={source} job={job_id} SKIP empty text", flush=True)
        return

    # Embed (cache-aware)
    key = _cache_key(text)
    vec = []
    was_cached = False
    try:
        cached = await redis_client.get(key)
        if cached:
            vec = json.loads(cached)
            was_cached = True
            _stats["cache_hits"] += 1
        else:
            vec = await _embed_via_ollama(text, source=f"async:{source}")
            if vec:
                try:
                    await redis_client.setex(key, CACHE_TTL, json.dumps(vec))
                except Exception:
                    pass
    except Exception as e:
        print(f"[embed-queue] ASYNC src={source} job={job_id} EMBED_ERROR "
              f"{type(e).__name__}: {e}", flush=True)

    # Store result
    if job_id:
        try:
            result = json.dumps({"embedding": vec, "status": "done" if vec else "failed"})
            await redis_client.setex(f"embed_result:{job_id}", RESULT_TTL, result)
        except Exception:
            pass

    # Deliver callback
    callback_status = ""
    if callback_url and vec:
        cb_t0 = time.time()
        try:
            payload = json.loads(callback_data)
            payload["embedding"] = vec
            resp = await http_client.post(callback_url, json=payload, timeout=10.0)
            cb_ms = round((time.time() - cb_t0) * 1000, 1)
            if resp.status_code < 300:
                _stats["callbacks_ok"] += 1
                callback_status = f" callback={resp.status_code} {cb_ms}ms"
            else:
                _stats["callbacks_failed"] += 1
                callback_status = f" callback=FAIL:{resp.status_code} {cb_ms}ms"
                print(f"[embed-queue] ASYNC src={source} job={job_id} "
                      f"CALLBACK_ERROR {callback_url} → {resp.status_code}", flush=True)
        except Exception as e:
            cb_ms = round((time.time() - cb_t0) * 1000, 1)
            _stats["callbacks_failed"] += 1
            callback_status = f" callback=ERROR {cb_ms}ms"
            print(f"[embed-queue] ASYNC src={source} job={job_id} "
                  f"CALLBACK_ERROR {callback_url} → {type(e).__name__}: {e}", flush=True)

    # ACK + stats
    await redis_client.xack(STREAM_NAME, GROUP_NAME, msg_id)
    total_ms = round((time.time() - t0) * 1000, 1)
    if vec:
        _stats["async_completed"] += 1
        cache_tag = "cache=HIT" if was_cached else "cache=MISS"
        print(f"[embed-queue] ASYNC src={source} job={job_id} OK {total_ms}ms "
              f"{cache_tag} dims={len(vec)}{callback_status} \"{preview}\"", flush=True)
    else:
        _stats["async_failed"] += 1
        print(f"[embed-queue] ASYNC src={source} job={job_id} FAILED {total_ms}ms "
              f"\"{preview}\"", flush=True)

    # Trim stream to bounded size
    try:
        await redis_client.xtrim(STREAM_NAME, maxlen=MAX_STREAM, approximate=True)
    except Exception:
        pass


async def _reclaim_stale_jobs():
    """
    Periodically reclaim jobs stuck in PEL (pending entries list).
    If a consumer crashes mid-processing, its jobs stay pending.
    XCLAIM transfers them to this worker for retry.
    """
    while not _shutdown:
        await asyncio.sleep(60)
        try:
            # Find entries pending > 30s with no ACK
            pending = await redis_client.xpending_range(
                STREAM_NAME, GROUP_NAME, "-", "+", count=20
            )
            now_ms = int(time.time() * 1000)
            for entry in pending:
                idle_ms = entry.get("time_since_delivered", 0)
                if idle_ms > 30_000:
                    msg_id = entry["message_id"]
                    claimed = await redis_client.xclaim(
                        STREAM_NAME, GROUP_NAME, f"{CONSUMER_ID}-0",
                        min_idle_time=30_000, message_ids=[msg_id]
                    )
                    for claim_id, fields in claimed:
                        print(f"[embed-queue] Reclaimed stale job {claim_id}", flush=True)
                        await _process_stream_job(claim_id, fields, f"{CONSUMER_ID}-0")
        except Exception:
            pass


# ── API Endpoints ─────────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text:   str
    source: str = ""    # calling service name (for logging)


class BatchEmbedRequest(BaseModel):
    texts:  list[str]
    source: str = ""


class EnqueueRequest(BaseModel):
    text:             str
    source:           str  = ""        # calling service name (for logging)
    callback_url:     str  = ""        # POST embedding here when done
    callback_payload: dict = {}        # extra fields sent with callback


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Synchronous embedding with cache + dedup. For search (read path)."""
    t0 = time.time()
    vec, cached = await get_embedding(req.text, source=req.source or "unknown")
    elapsed = round((time.time() - t0) * 1000, 1)
    src = req.source or "unknown"
    dims = len(vec)
    preview = req.text[:60].replace("\n", " ")
    if cached:
        print(f"[embed-queue] SYNC src={src} cache=HIT {elapsed}ms dims={dims} \"{preview}\"", flush=True)
    else:
        print(f"[embed-queue] SYNC src={src} cache=MISS ollama={elapsed}ms dims={dims} \"{preview}\"", flush=True)
    return {"embedding": vec, "cached": cached, "model": EMBED_MODEL,
            "elapsed_ms": elapsed}


@app.post("/embed/batch")
async def embed_batch(req: BatchEmbedRequest):
    """Batch sync embedding — parallel cache lookups, deduped Ollama calls."""
    t0 = time.time()
    results = await asyncio.gather(*[get_embedding(t) for t in req.texts[:50]])
    elapsed = round((time.time() - t0) * 1000, 1)
    hits = sum(1 for _, c in results if c)
    src = req.source or "unknown"
    print(f"[embed-queue] BATCH src={src} count={len(results)} "
          f"cache_hits={hits} {elapsed}ms", flush=True)
    return {
        "embeddings": [{"embedding": vec, "cached": cached} for vec, cached in results],
        "model": EMBED_MODEL,
        "count": len(results),
        "elapsed_ms": elapsed,
    }


@app.post("/enqueue")
async def enqueue(req: EnqueueRequest):
    """
    Async embedding via Redis Streams.
    Returns immediately with a job_id. The embedding is processed in background
    by stream workers and (optionally) delivered to callback_url.
    Poll GET /job/{job_id} for the result, or provide a callback_url.
    """
    job_id = uuid.uuid4().hex[:12]
    fields = {
        "text":             req.text[:MAX_TEXT_LEN],
        "job_id":           job_id,
        "source":           req.source or "unknown",
        "callback_url":     req.callback_url,
        "callback_payload": json.dumps(req.callback_payload),
    }
    await redis_client.xadd(STREAM_NAME, fields, maxlen=MAX_STREAM,
                             approximate=True)
    _stats["async_enqueued"] += 1
    preview = req.text[:60].replace("\n", " ")
    print(f"[embed-queue] ENQUEUE src={req.source or 'unknown'} "
          f"job={job_id} \"{preview}\"", flush=True)
    return {"job_id": job_id, "status": "queued", "stream": STREAM_NAME}


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """Check status of an async embed job."""
    try:
        result = await redis_client.get(f"embed_result:{job_id}")
        if result:
            return json.loads(result)
    except Exception:
        pass
    return {"status": "pending", "job_id": job_id}


@app.get("/health")
async def health():
    redis_ok = False
    try:
        redis_ok = await redis_client.ping()
    except Exception:
        pass

    # Stream depth
    stream_len = 0
    try:
        stream_len = await redis_client.xlen(STREAM_NAME)
    except Exception:
        pass

    return {
        "status":       "alive",
        "service":      "embed-queue",
        "model":        EMBED_MODEL,
        "redis":        "connected" if redis_ok else "disconnected",
        "queue_depth":  stream_len,
    }


@app.get("/stats")
async def stats():
    """Cache hit rate, Ollama latency, queue depth, async job counts."""
    total = (_stats["sync_requests"] + _stats["async_enqueued"]) or 1
    stream_len = 0
    try:
        stream_len = await redis_client.xlen(STREAM_NAME)
    except Exception:
        pass

    return {
        "sync_requests":    _stats["sync_requests"],
        "async_enqueued":   _stats["async_enqueued"],
        "async_completed":  _stats["async_completed"],
        "async_failed":     _stats["async_failed"],
        "cache_hits":       _stats["cache_hits"],
        "cache_misses":     _stats["cache_misses"],
        "cache_hit_rate":   round(_stats["cache_hits"] / total * 100, 1),
        "ollama_calls":     _stats["ollama_calls"],
        "ollama_errors":    _stats["ollama_errors"],
        "avg_ollama_ms":    _stats["avg_ollama_ms"],
        "callbacks_ok":     _stats["callbacks_ok"],
        "callbacks_failed": _stats["callbacks_failed"],
        "queue_depth":      stream_len,
        "model":            EMBED_MODEL,
        "cache_ttl_s":      CACHE_TTL,
    }
