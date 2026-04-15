# services/knobs/main.py
# ── Knobs: Dynamic Behavioral Parameter Control ───────────────────────────────
# The only service that WRITES knobs. All others read only.
# Backed by Redis. Sub-millisecond reads across all services.
# Presets let you switch entire personality profiles instantly.
# History lets you track and roll back changes.
# Self-update service can call this to tune the brain automatically.

import json
import os
import sys
import time

sys.path.insert(0, "/shared")

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import KnobSet
from events import EventBus

app    = FastAPI(title="Knobs", version="1.0.0",
                 description="Dynamic behavioral parameter control")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)
events = EventBus()

KNOBS_KEY         = "brain:knobs:current"
HISTORY_KEY       = "brain:knobs:history"
PRESETS_KEY       = "brain:knobs:presets"
HISTORY_MAX       = 100

# ── Built-in presets ──────────────────────────────────────────────────────────

PRESETS: dict[str, KnobSet] = {
    "default": KnobSet(),

    "therapist": KnobSet(
        empathy=0.96, warmth=0.95, honesty=0.88,
        humor=0.08, formality=0.30, care_explicitness=0.85,
        reasoning_depth=0.75, skepticism=0.25, assertiveness=0.25,
        ethical_weight=0.95, verbosity=0.55
    ),

    "researcher": KnobSet(
        honesty=1.00, skepticism=0.92, reasoning_depth=0.95,
        humor=0.08, formality=0.72, creativity=0.50,
        verification_threshold=0.88, novelty_sensitivity=0.92,
        assertiveness=0.70, verbosity=0.80, warmth=0.40
    ),

    "creative_partner": KnobSet(
        creativity=0.96, humor=0.72, warmth=0.82,
        formality=0.10, verbosity=0.65, assertiveness=0.30,
        analogy_seeking=0.92, skepticism=0.18, curiosity=0.95,
        swarm_preference=0.75
    ),

    "technical_expert": KnobSet(
        honesty=1.00, formality=0.65, reasoning_depth=0.92,
        verbosity=0.80, humor=0.18, skepticism=0.82,
        verification_threshold=0.82, assertiveness=0.70,
        warmth=0.35, empathy=0.45
    ),

    "friend": KnobSet(
        warmth=0.96, humor=0.72, formality=0.08,
        empathy=0.88, verbosity=0.40, assertiveness=0.50,
        care_explicitness=0.55, honesty=0.85
    ),

    "socratic": KnobSet(
        curiosity=0.99, skepticism=0.92, humor=0.28,
        assertiveness=0.18, reasoning_depth=0.96,
        honesty=0.96, verbosity=0.58, formality=0.42
    ),

    "concise_professional": KnobSet(
        formality=0.75, verbosity=0.18, honesty=0.90,
        humor=0.08, reasoning_depth=0.72, empathy=0.45,
        assertiveness=0.68, warmth=0.40
    ),

    "philosopher": KnobSet(
        curiosity=0.96, reasoning_depth=0.98, analogy_seeking=0.92,
        creativity=0.85, honesty=0.96, skepticism=0.88,
        formality=0.55, verbosity=0.72, humor=0.25,
        ethical_weight=0.95
    ),

    "coach": KnobSet(
        warmth=0.88, assertiveness=0.78, empathy=0.82,
        honesty=0.90, humor=0.42, formality=0.30,
        care_explicitness=0.70, creativity=0.65,
        reasoning_depth=0.65, verbosity=0.55
    ),
}


def _read_current() -> KnobSet:
    raw = r.get(KNOBS_KEY)
    if raw:
        try:
            return KnobSet(**json.loads(raw))
        except Exception:
            pass
    return KnobSet()


def _write_knobs(new_knobs: KnobSet, source: str = "api"):
    current_raw = r.get(KNOBS_KEY)
    if current_raw:
        r.lpush(HISTORY_KEY, json.dumps({
            "timestamp": time.time(),
            "source":    source,
            "previous":  json.loads(current_raw),
            "new":       new_knobs.dict()
        }))
        r.ltrim(HISTORY_KEY, 0, HISTORY_MAX - 1)

    r.set(KNOBS_KEY, json.dumps(new_knobs.dict()))
    events.publish("knobs.updated", {
        "timestamp": time.time(),
        "source":    source,
        "knobs":     new_knobs.dict()
    })


@app.get("/")
async def root():
    """Root endpoint — redirect to health or show service info."""
    return {
        "service": "knobs",
        "description": "Dynamic behavioral parameter control",
        "endpoints": [
            "GET  /health",
            "GET  /current",
            "GET  /current/{name}",
            "PUT  /set",
            "PATCH /adjust?knob_name=...&value=...",
            "PATCH /adjust-many",
            "POST /preset/{name}",
            "GET  /presets",
            "GET  /history",
            "POST /reset",
        ],
    }


@app.get("/health")
async def health():
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status":        "alive" if redis_ok else "degraded",
        "service":       "knobs",
        "redis":         "connected" if redis_ok else "error",
        "presets_count": len(PRESETS)
    }


@app.get("/current", response_model=KnobSet)
async def get_current():
    """Get all current knob values."""
    return _read_current()


@app.get("/current/{knob_name}")
async def get_knob(knob_name: str):
    """Get a single knob value."""
    k = _read_current()
    if not hasattr(k, knob_name):
        raise HTTPException(404, f"Knob '{knob_name}' not found")
    return {"knob": knob_name, "value": getattr(k, knob_name)}


@app.put("/set", response_model=KnobSet)
async def set_all_knobs(new_knobs: KnobSet):
    """Replace all knobs at once."""
    _write_knobs(new_knobs, source="api:set")
    return new_knobs


@app.patch("/adjust")
async def adjust_knob(knob_name: str, value: float):
    """
    Adjust a single knob without touching others.
    Primary interface for fine-tuning and self-update service.
    """
    default_k = KnobSet()
    if not hasattr(default_k, knob_name):
        raise HTTPException(400, f"Unknown knob: '{knob_name}'. "
                                 f"Valid knobs: {list(default_k.dict().keys())}")

    # Type check — framework_threshold is int
    if knob_name == "framework_threshold":
        if not isinstance(value, (int, float)):
            raise HTTPException(400, "framework_threshold must be numeric")
        value = int(value)
    else:
        if not (0.0 <= float(value) <= 1.0):
            raise HTTPException(400, f"Value must be 0.0–1.0, got {value}")

    current      = _read_current()
    current_dict = current.dict()
    current_dict[knob_name] = value
    new_knobs    = KnobSet(**current_dict)

    _write_knobs(new_knobs, source=f"api:adjust:{knob_name}")

    return {
        "adjusted": knob_name,
        "old_value": getattr(current, knob_name),
        "new_value": value
    }


@app.patch("/adjust-many")
async def adjust_many(changes: dict):
    """
    Adjust multiple knobs in one atomic operation.
    Body: {"humor": 0.7, "formality": 0.2, ...}
    """
    current      = _read_current()
    current_dict = current.dict()
    errors       = []

    for name, value in changes.items():
        if not hasattr(current, name):
            errors.append(f"Unknown: {name}")
            continue
        if name != "framework_threshold" and not (0.0 <= float(value) <= 1.0):
            errors.append(f"Out of range: {name}={value}")
            continue
        current_dict[name] = int(value) if name == "framework_threshold" else float(value)

    if errors:
        raise HTTPException(400, f"Validation errors: {'; '.join(errors)}")

    new_knobs = KnobSet(**current_dict)
    _write_knobs(new_knobs, source="api:adjust-many")

    return {
        "adjusted": list(changes.keys()),
        "current":  new_knobs.dict()
    }


@app.post("/preset/{name}")
async def apply_preset(name: str):
    """Apply a named preset (built-in or custom)."""
    if name in PRESETS:
        _write_knobs(PRESETS[name], source=f"preset:{name}")
        return {"preset": name, "applied": True, "knobs": PRESETS[name].dict()}

    # Check custom presets
    raw = r.hget(PRESETS_KEY, name)
    if raw:
        custom_knobs = KnobSet(**json.loads(raw))
        _write_knobs(custom_knobs, source=f"preset:custom:{name}")
        return {"preset": name, "applied": True, "knobs": custom_knobs.dict()}

    available = list(PRESETS.keys()) + list(r.hgetall(PRESETS_KEY).keys())
    raise HTTPException(404, f"Preset '{name}' not found. Available: {available}")


@app.post("/preset/save/{name}")
async def save_preset(name: str):
    """Save current knobs as a custom preset."""
    if name in PRESETS:
        raise HTTPException(400, f"Cannot overwrite built-in preset '{name}'")
    current = _read_current()
    r.hset(PRESETS_KEY, name, json.dumps(current.dict()))
    return {"saved": name, "knobs": current.dict()}


@app.delete("/preset/{name}")
async def delete_preset(name: str):
    """Delete a custom preset."""
    if name in PRESETS:
        raise HTTPException(400, f"Cannot delete built-in preset '{name}'")
    r.hdel(PRESETS_KEY, name)
    return {"deleted": name}


@app.get("/presets")
async def list_presets():
    """List all available presets with descriptions."""
    custom = list(r.hgetall(PRESETS_KEY).keys())
    return {
        "built_in": {
            name: {
                "empathy":       k.empathy,
                "humor":         k.humor,
                "honesty":       k.honesty,
                "formality":     k.formality,
                "creativity":    k.creativity
            }
            for name, k in PRESETS.items()
        },
        "custom": custom
    }


@app.get("/history")
async def get_history(limit: int = 20):
    """Recent knob change history."""
    raw = r.lrange(HISTORY_KEY, 0, limit - 1)
    return [json.loads(h) for h in raw]


@app.post("/history/rollback")
async def rollback(steps: int = 1):
    """Roll back knobs to a previous state."""
    raw = r.lrange(HISTORY_KEY, 0, steps - 1)
    if not raw:
        raise HTTPException(404, "No history to roll back to")

    target = json.loads(raw[-1])
    previous_knobs = KnobSet(**target["previous"])
    _write_knobs(previous_knobs, source="rollback")
    return {"rolled_back": steps, "restored": previous_knobs.dict()}


@app.post("/reset")
async def reset_to_defaults():
    """Reset all knobs to factory defaults."""
    _write_knobs(KnobSet(), source="reset")
    return {"reset": True, "knobs": KnobSet().dict()}


@app.get("/describe/{knob_name}")
async def describe_knob(knob_name: str):
    """Get a description of what a knob controls."""
    descriptions = {
        "humor":                  "0=serious/dry, 1=witty/playful",
        "honesty":                "0=diplomatic, 1=radically honest",
        "empathy":                "0=task-focused, 1=deeply attuned to feelings",
        "formality":              "0=casual friend, 1=formal professional",
        "verbosity":              "0=terse, 1=comprehensive",
        "creativity":             "0=conventional, 1=highly original",
        "skepticism":             "0=trusting, 1=questioning",
        "assertiveness":          "0=deferential, 1=confident",
        "curiosity":              "0=focused, 1=exploratory",
        "warmth":                 "0=clinical, 1=warm",
        "reasoning_depth":        "0=quick, 1=deep reasoning",
        "analogy_seeking":        "0=direct, 1=cross-domain search intensity",
        "verification_threshold": "0=trust content, 1=verify everything",
        "memory_weight":          "0=ignore past, 1=heavily influenced by past",
        "novelty_sensitivity":    "0=ignore novelty, 1=flag any new experience",
        "swarm_preference":       "0=single agent, 1=swarm/multi-agent",
        "framework_threshold":    "Word count before engaging multi-agent framework",
        "ethical_weight":         "0=compass off, 1=fully compass-driven",
        "care_explicitness":      "0=implicit care, 1=explicit caring language"
    }
    if knob_name not in descriptions:
        raise HTTPException(404, f"Unknown knob: {knob_name}")
    current = _read_current()
    return {
        "knob":        knob_name,
        "description": descriptions[knob_name],
        "current":     getattr(current, knob_name)
    }
