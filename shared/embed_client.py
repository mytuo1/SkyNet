# shared/embed_client.py
# ── Centralized Embed Client ──────────────────────────────────────────────────
# Used by neocortex, hippocampus, cross-domain, verification.
# Tries embed-queue service first (cached + deduped), falls back to direct Ollama.
# Supports async enqueue for write-path operations (store/encode).

import asyncio
import os
import time

import httpx

EMBED_QUEUE_URL = os.getenv("EMBED_QUEUE_URL", "http://embed-queue:8021")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://host.docker.internal:11434")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(timeout=30.0)
    return _http


async def embed(text: str, service: str = "unknown") -> list:
    """
    Get embedding via embed-queue (cached, deduped) with direct Ollama fallback.
    Returns [] on failure — callers must handle empty vectors.
    Used for READ path (search) — blocks until embedding is ready.
    """
    client = _client()

    # ── Try embed-queue first (centralized cache + dedup) ─────────────────
    try:
        t0 = time.time()
        resp = await client.post(
            f"{EMBED_QUEUE_URL}/embed",
            json={"text": text[:2000], "source": service},
            timeout=15.0
        )
        data    = resp.json()
        vec     = data.get("embedding", [])
        elapsed = (time.time() - t0) * 1000
        if vec and not all(v == 0.0 for v in vec[:10]):
            if elapsed > 3000:
                print(f"[{service}] embed via queue: {elapsed:.0f}ms "
                      f"cached={data.get('cached')}", flush=True)
            return vec
        # Empty/zero vector from queue — fall through to direct
    except Exception as e:
        print(f"[{service}] embed-queue unavailable ({type(e).__name__}), "
              f"falling back to Ollama", flush=True)

    # ── Fallback: direct Ollama call ──────────────────────────────────────
    for attempt in range(2):
        t0 = time.time()
        try:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text[:2000], "keep_alive": -1},
                timeout=15.0
            )
            elapsed = (time.time() - t0) * 1000
            vec = resp.json()["embedding"]
            if all(v == 0.0 for v in vec[:10]):
                print(f"[{service}] embed returned zero vector after "
                      f"{elapsed:.0f}ms", flush=True)
                return []
            if elapsed > 3000:
                print(f"[{service}] embed slow (direct): {elapsed:.0f}ms", flush=True)
            return vec
        except (httpx.ConnectTimeout, httpx.ConnectError) as e:
            elapsed = (time.time() - t0) * 1000
            print(f"[{service}] ERROR: Ollama unreachable at {OLLAMA_URL} after "
                  f"{elapsed:.0f}ms ({type(e).__name__}) — skipping embed "
                  f"(no retry)", flush=True)
            return []
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            print(f"[{service}] embed attempt {attempt+1} failed after "
                  f"{elapsed:.0f}ms ({type(e).__name__}): {e}", flush=True)
            if attempt == 0:
                await asyncio.sleep(0.3)
    return []


async def enqueue(text: str, service: str = "unknown",
                  callback_url: str = "", callback_payload: dict = None) -> str:
    """
    Async embedding via Redis Streams (fire-and-forget for write path).
    Returns job_id immediately. The embedding is processed in background
    by embed-queue workers and delivered to callback_url when done.
    Falls back to sync embed() if embed-queue is unreachable.
    """
    client = _client()
    try:
        resp = await client.post(
            f"{EMBED_QUEUE_URL}/enqueue",
            json={
                "text":             text[:2000],
                "source":           service,
                "callback_url":     callback_url,
                "callback_payload": callback_payload or {},
            },
            timeout=5.0
        )
        data = resp.json()
        return data.get("job_id", "")
    except Exception as e:
        print(f"[{service}] embed enqueue failed ({type(e).__name__}), "
              f"falling back to sync embed", flush=True)
        # Fallback: do it synchronously so the embedding isn't lost
        await embed(text, service=service)
        return ""
