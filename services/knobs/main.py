from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis, json, time, os, sys
sys.path.append("/shared")
from models import KnobSet
from events import EventBus

app    = FastAPI(title="Knobs", version="1.0.0")
r      = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"),
                         decode_responses=True)
events = EventBus(os.getenv("REDIS_URL"))

KNOBS_KEY         = "brain:knobs:current"
KNOBS_HISTORY_KEY = "brain:knobs:history"
PRESETS_KEY       = "brain:knobs:presets"

# ─── Built-in presets ─────────────────────────────────────────────────────
BUILT_IN_PRESETS = {
    "default": KnobSet(),
    "therapist": KnobSet(
        empathy=0.95, warmth=0.95, honesty=0.9,
        humor=0.1, formality=0.3, care_explicitness=0.8,
        reasoning_depth=0.8, skepticism=0.3
    ),
    "researcher": KnobSet(
        honesty=1.0, skepticism=0.9, reasoning_depth=0.95,
        humor=0.1, formality=0.7, creativity=0.5,
        verification_threshold=0.85, novelty_sensitivity=0.9
    ),
    "creative_partner": KnobSet(
        creativity=0.95, humor=0.7, warmth=0.8,
        formality=0.1, verbosity=0.7, assertiveness=0.3,
        analogy_seeking=0.9, skepticism=0.2
    ),
    "technical_expert": KnobSet(
        honesty=1.0, formality=0.6, reasoning_depth=0.9,
        verbosity=0.8, humor=0.2, skepticism=0.8,
        verification_threshold=0.8
    ),
    "friend": KnobSet(
        warmth=0.95, humor=0.7, formality=0.1,
        empathy=0.85, verbosity=0.4, assertiveness=0.5
    ),
    "socratic": KnobSet(
        curiosity=0.99, skepticism=0.9, humor=0.3,
        assertiveness=0.2, reasoning_depth=0.95,
        honesty=0.95, verbosity=0.6
    )
}

@app.get("/health")
async def health():
    return {"status": "alive", "service": "knobs"}

@app.get("/current", response_model=KnobSet)
async def get_current():
    """Get current knob values."""
    raw = r.get(KNOBS_KEY)
    if raw:
        return KnobSet(**json.loads(raw))
    return KnobSet()

@app.put("/set", response_model=KnobSet)
async def set_knobs(knobs: KnobSet):
    """
    Set all knobs at once.
    Saves history. Publishes event to all services.
    """
    # Save history
    current_raw = r.get(KNOBS_KEY)
    if current_raw:
        history_entry = {
            "timestamp": time.time(),
            "previous":  json.loads(current_raw),
            "new":       knobs.dict()
        }
        r.lpush(KNOBS_HISTORY_KEY, json.dumps(history_entry))
        r.ltrim(KNOBS_HISTORY_KEY, 0, 99)  # keep last 100
    
    # Save new knobs
    r.set(KNOBS_KEY, json.dumps(knobs.dict()))
    
    # Notify all services
    events.publish("knobs.updated", {
        "timestamp": time.time(),
        "knobs":     knobs.dict()
    })
    
    return knobs

@app.patch("/adjust")
async def adjust_knob(knob_name: str, value: float):
    """
    Adjust a single knob without touching others.
    The primary interface for dynamic adjustment.
    """
    if not hasattr(KnobSet(), knob_name):
        raise HTTPException(400, f"Unknown knob: {knob_name}")
    
    if not (0.0 <= value <= 1.0) and knob_name != "framework_threshold":
        raise HTTPException(400, f"Value must be 0.0-1.0, got {value}")
    
    current = await get_current()
    current_dict = current.dict()
    current_dict[knob_name] = value
    new_knobs = KnobSet(**current_dict)
    
    return await set_knobs(new_knobs)

@app.post("/preset/{name}")
async def apply_preset(name: str):
    """Apply a named preset."""
    if name in BUILT_IN_PRESETS:
        return await set_knobs(BUILT_IN_PRESETS[name])
    
    # Check custom presets in Redis
    custom_raw = r.hget(PRESETS_KEY, name)
    if custom_raw:
        return await set_knobs(KnobSet(**json.loads(custom_raw)))
    
    raise HTTPException(404, f"Preset '{name}' not found. "
                            f"Built-in: {list(BUILT_IN_PRESETS.keys())}")

@app.post("/preset/save/{name}")
async def save_preset(name: str):
    """Save current knobs as a named preset."""
    current = await get_current()
    r.hset(PRESETS_KEY, name, json.dumps(current.dict()))
    return {"saved": name, "knobs": current.dict()}

@app.get("/presets")
async def list_presets():
    custom = {k: json.loads(v) 
              for k, v in (r.hgetall(PRESETS_KEY) or {}).items()}
    return {
        "built_in": list(BUILT_IN_PRESETS.keys()),
        "custom":   list(custom.keys())
    }

@app.get("/history")
async def get_history(limit: int = 10):
    history = r.lrange(KNOBS_HISTORY_KEY, 0, limit - 1)
    return [json.loads(h) for h in history]

@app.post("/reset")
async def reset_to_defaults():
    return await set_knobs(KnobSet())