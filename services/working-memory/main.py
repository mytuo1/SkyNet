# services/working-memory/main.py
# ── Working Memory: Active Context and Session State ─────────────────────────
# Backed by Redis. Volatile — data expires. Fast.
# Holds: thought stream, focus, task state, consolidation queue.
# Mirrors the brain's working memory: ~7±2 items, fast decay.

import json
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(0, "/shared")

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from knobs import KnobsClient

app = FastAPI(title="WorkingMemory", version="1.0.0",
              description="Active context, session state, consolidation queue")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs = KnobsClient()
r     = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_TTL  = 86400    # 24 hours (was 1 hour — let sessions persist)
SHORT_TTL    = 3600     # 1 hour  (was 5 minutes)
LONG_TTL     = 604800   # 7 days  (was 24 hours — full conversation retention)
MAX_THOUGHTS = 200      # conversation turns to keep (was 24 — deep recall)


class ContextRequest(BaseModel):
    session_id: str
    action:     str          # see handler below for valid actions
    key:        Optional[str] = None
    data:       Optional[Any] = None
    ttl:        Optional[int] = None


@app.get("/health")
async def health():
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status":  "alive" if redis_ok else "degraded",
        "service": "working-memory",
        "redis":   "connected" if redis_ok else "error"
    }


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


@app.post("/context", response_model=BrainResponse)
async def handle_context(req: ContextRequest):
    """
    Unified working memory endpoint.
    All operations go through here — no separate endpoints per operation.
    action values:
      get_all | get | set | delete
      push_thought | get_thoughts | clear_thoughts
      set_focus | get_focus
      set_task | get_task | update_task
      queue_consolidation | get_consolidation_queue | dequeue_consolidation
      pressure
    """
    start  = time.time()
    sid    = req.session_id
    result = {}

    # ── GET ALL (full working memory dump) ─────────────────────────────────────
    if req.action == "get_all":
        pattern = f"wm:{sid}:*"
        keys    = r.keys(pattern)
        context = {}
        for key in keys:
            short = key.replace(f"wm:{sid}:", "")
            # Check key type — only GET string keys; lists/sets are handled below
            key_type = r.type(key)
            if key_type != "string":
                continue
            val   = r.get(key)
            if val:
                try:    context[short] = json.loads(val)
                except (json.JSONDecodeError, TypeError, ValueError): context[short] = val

        # Include thought stream (Redis list — not fetchable via GET)
        raw_thoughts = r.lrange(f"wm:{sid}:thought_stream", 0, -1)
        thoughts = []
        for t in raw_thoughts:
            try:
                thoughts.append(json.loads(t))
            except (json.JSONDecodeError, TypeError):
                thoughts.append({"raw": t, "parse_error": True})
        context["thought_stream"] = thoughts
        result = context

    # ── GET ────────────────────────────────────────────────────────────────────
    elif req.action == "get":
        full_key = f"wm:{sid}:{req.key}"
        # Check key type — only GET works on string keys
        key_type = r.type(full_key)
        if key_type == "string":
            val = r.get(full_key)
            if val:
                r.expire(full_key, DEFAULT_TTL)  # refresh TTL on access
            if val:
                try:
                    parsed = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    parsed = {"raw": val, "parse_error": True}
            else:
                parsed = None
        elif key_type == "list":
            raw = r.lrange(full_key, 0, -1)
            parsed = []
            for item in raw:
                try:
                    parsed.append(json.loads(item))
                except (json.JSONDecodeError, TypeError):
                    parsed.append(item)
        elif key_type == "none":
            parsed = None
        else:
            # zset, hash, set — return type info instead of crashing
            parsed = {"type": key_type, "note": "non-string key — use specific action"}
        result = {"value": parsed}

    # ── SET ────────────────────────────────────────────────────────────────────
    elif req.action == "set":
        full_key = f"wm:{sid}:{req.key}"
        ttl      = req.ttl or DEFAULT_TTL
        r.setex(full_key, ttl, json.dumps(req.data))
        result = {"set": req.key}

    # ── DELETE ─────────────────────────────────────────────────────────────────
    elif req.action == "delete":
        full_key = f"wm:{sid}:{req.key}"
        r.delete(full_key)
        result = {"deleted": req.key}

    # ── PUSH THOUGHT (conversation stream) ────────────────────────────────────
    elif req.action == "push_thought":
        key = f"wm:{sid}:thought_stream"
        r.lpush(key, json.dumps(req.data))
        r.expire(key, LONG_TTL)
        r.ltrim(key, 0, MAX_THOUGHTS - 1)  # enforce capacity
        result = {"pushed": True, "length": r.llen(key)}

    # ── GET THOUGHTS ──────────────────────────────────────────────────────────
    elif req.action == "get_thoughts":
        key     = f"wm:{sid}:thought_stream"
        raw     = r.lrange(key, 0, -1)
        parsed_thoughts = []
        for t in raw:
            try:
                parsed_thoughts.append(json.loads(t))
            except (json.JSONDecodeError, TypeError):
                parsed_thoughts.append({"raw": t, "parse_error": True})
        result  = {"thoughts": parsed_thoughts}

    # ── CLEAR THOUGHTS ────────────────────────────────────────────────────────
    elif req.action == "clear_thoughts":
        r.delete(f"wm:{sid}:thought_stream")
        result = {"cleared": True}

    # ── SET FOCUS ─────────────────────────────────────────────────────────────
    elif req.action == "set_focus":
        focus_data = req.data or {}
        urgency    = focus_data.get("urgency", 0.5)
        current_raw = r.get(f"wm:{sid}:focus")
        if current_raw:
            curr = json.loads(current_raw)
            if urgency > curr.get("urgency", 0):
                r.setex(f"wm:{sid}:focus", SHORT_TTL,
                        json.dumps({**focus_data, "set_at": time.time()}))
                result = {"focus_updated": True, "urgency": urgency}
            else:
                result = {"focus_unchanged": True}
        else:
            r.setex(f"wm:{sid}:focus", SHORT_TTL,
                    json.dumps({**focus_data, "set_at": time.time()}))
            result = {"focus_set": True}

    # ── GET FOCUS ─────────────────────────────────────────────────────────────
    elif req.action == "get_focus":
        val    = r.get(f"wm:{sid}:focus")
        result = {"focus": json.loads(val) if val else None}

    # ── SET TASK STATE ────────────────────────────────────────────────────────
    elif req.action == "set_task":
        task_id = (req.data or {}).get("task_id", req.key or "default")
        r.setex(f"wm:task:{task_id}", LONG_TTL, json.dumps(req.data))
        result = {"task_set": task_id}

    # ── GET TASK STATE ────────────────────────────────────────────────────────
    elif req.action == "get_task":
        task_id = req.key or "default"
        val     = r.get(f"wm:task:{task_id}")
        result  = {"task": json.loads(val) if val else None}

    # ── UPDATE TASK STATE (partial update) ────────────────────────────────────
    elif req.action == "update_task":
        task_id     = (req.data or {}).get("task_id", req.key or "default")
        current_raw = r.get(f"wm:task:{task_id}")
        current     = json.loads(current_raw) if current_raw else {}
        current.update(req.data or {})
        r.setex(f"wm:task:{task_id}", LONG_TTL, json.dumps(current))
        result = {"task_updated": task_id}

    # ── QUEUE FOR CONSOLIDATION ───────────────────────────────────────────────
    elif req.action == "queue_consolidation":
        item     = dict(req.data or {})
        priority = item.pop("priority", 0.5)
        item["queued_at"] = time.time()
        r.zadd("wm:consolidation_queue",
               {json.dumps(item): priority})
        result = {"queued": True, "priority": priority}

    # ── GET CONSOLIDATION QUEUE ───────────────────────────────────────────────
    elif req.action == "get_consolidation_queue":
        limit = (req.data or {}).get("limit", 20)
        items = r.zrevrange("wm:consolidation_queue",
                            0, limit - 1, withscores=True)
        result = {
            "queue": [
                {"item": json.loads(item), "priority": score}
                for item, score in items
            ]
        }

    # ── DEQUEUE (remove after processing) ────────────────────────────────────
    elif req.action == "dequeue_consolidation":
        item_json = json.dumps(req.data)
        r.zrem("wm:consolidation_queue", item_json)
        result = {"dequeued": True}

    # ── MEMORY PRESSURE ───────────────────────────────────────────────────────
    elif req.action == "pressure":
        count    = sum(1 for _ in r.scan_iter(f"wm:{sid}:*"))
        pressure = min(1.0, count / (MAX_THOUGHTS * 3))
        result   = {"pressure": round(pressure, 3), "item_count": count}

    else:
        result = {"error": f"Unknown action: {req.action}"}

    return BrainResponse(
        request_id = "wm",
        service    = "working-memory",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Wipe all working memory for a session."""
    keys = r.keys(f"wm:{session_id}:*")
    if keys:
        r.delete(*keys)
    return {"cleared_keys": len(keys), "session_id": session_id}


@app.get("/session/{session_id}/stats")
async def session_stats(session_id: str):
    keys      = r.keys(f"wm:{session_id}:*")
    thoughts  = r.llen(f"wm:{session_id}:thought_stream")
    queue_len = r.zcard("wm:consolidation_queue")
    pressure  = min(1.0, len(keys) / (MAX_THOUGHTS * 3))
    return {
        "session_id":         session_id,
        "active_keys":        len(keys),
        "thought_stream_len": thoughts,
        "consolidation_queue": queue_len,
        "memory_pressure":    round(pressure, 3)
    }
