# services/hypothalamus/main.py
# ── Hypothalamus: Hormonal State Engine ───────────────────────────────────────
# Translates emotional and cognitive events into hormonal state changes.
# Hormones modulate the behavioral knobs dynamically — personality drifts
# organically based on conversational dynamics, like a real person's mood.
#
# Six hormones tracked:
#   dopamine       — reward, motivation, novelty-seeking, creativity
#   serotonin      — mood stability, impulse control, patience
#   cortisol       — stress response, focus under acute / impairment under chronic
#   oxytocin       — trust, bonding, empathy boost
#   norepinephrine — alertness, fight-or-flight, sensory sharpening
#   endorphins     — resilience, persistence, pain/stress suppression
#
# Overstimulation (>0.85 for 5+ turns) and depletion (<0.15 for 5+ turns)
# produce distinct behavioral effects that degrade or skew cognition,
# mimicking real neurochemical dysregulation.

import json
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/shared")

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse, ExperienceType
from knobs import KnobsClient
from events import EventBus

app = FastAPI(title="Hypothalamus", version="1.0.0",
              description="Hormonal state engine — modulates personality via neurochemical simulation")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

HORMONE_KEY     = "hormone:{session_id}:state"
HISTORY_MAX     = 50     # keep last N snapshots
OVERSTIM_WINDOW = 5      # interactions to trigger overstimulation
DEPLETION_WINDOW = 5     # interactions to trigger depletion

# ── Baselines ─────────────────────────────────────────────────────────────────

BASELINES = {
    "dopamine":       0.50,
    "serotonin":      0.55,
    "cortisol":       0.20,
    "oxytocin":       0.30,
    "norepinephrine": 0.25,
    "endorphins":     0.20,
}

# ── Decay toward baseline (exponential, τ ≈ 3 interactions) ──────────────────

DECAY_TAU = 3.0   # in "interactions" — higher = slower decay


def _decay_toward_baseline(current: float, baseline: float, steps: float = 1.0) -> float:
    """Exponential decay: moves current toward baseline. steps=1 is one interaction."""
    alpha = 1.0 - math.exp(-steps / DECAY_TAU)
    return current + alpha * (baseline - current)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Request / Response Models ─────────────────────────────────────────────────

class HormoneUpdateRequest(BaseModel):
    session_id:        str
    amygdala_result:   dict = {}
    perception_events: list = []       # from perception /perceive
    metacognition:     dict = {}
    verification:      dict = {}
    user_feedback:     Optional[str] = None   # "positive" | "negative" | None
    turn_count:        int  = 0
    time_since_last:   float = 0.0     # seconds since previous interaction


class HormoneStateResponse(BaseModel):
    hormones:          dict            # current hormone levels
    modified_knobs:    dict            # hormone-adjusted knobs
    overstimulated:    list = []       # hormones over threshold
    depleted:          list = []       # hormones under threshold
    dominant_hormone:  str  = ""       # highest-above-baseline hormone
    mood_summary:      str  = ""       # human-readable mood description


# ── Core Logic ────────────────────────────────────────────────────────────────

def _load_state(session_id: str) -> dict:
    key = HORMONE_KEY.format(session_id=session_id)
    raw = r.get(key)
    if raw:
        return json.loads(raw)
    # Fresh state
    return {
        **{h: v for h, v in BASELINES.items()},
        "last_updated": time.time(),
        "history": []
    }


def _save_state(session_id: str, state: dict):
    key = HORMONE_KEY.format(session_id=session_id)
    r.set(key, json.dumps(state), ex=86400 * 7)   # 7-day TTL


def _apply_triggers(state: dict, req: HormoneUpdateRequest) -> dict:
    """
    Apply hormone deltas based on amygdala, perception, metacognition,
    verification, and user feedback signals.
    """
    ar = req.amygdala_result
    urgency = ar.get("urgency", "low")
    emotion = ar.get("emotion", "neutral")
    threat  = ar.get("threat_detected", False)

    # ── Amygdala-driven triggers ──────────────────────────────────────────
    if threat or urgency == "critical":
        state["cortisol"]       = _clamp(state["cortisol"] + 0.20)
        state["norepinephrine"] = _clamp(state["norepinephrine"] + 0.20)

    if urgency in ("high", "distressed"):
        state["cortisol"]       = _clamp(state["cortisol"] + 0.10)
        state["norepinephrine"] = _clamp(state["norepinephrine"] + 0.08)

    # Joy / excitement / positive emotions
    pos_emotions = {"joy", "excitement", "admiration", "amusement", "gratitude",
                    "love", "optimism", "pride", "relief", "approval", "caring"}
    if emotion in pos_emotions:
        state["dopamine"]  = _clamp(state["dopamine"] + 0.10)
        state["endorphins"] = _clamp(state["endorphins"] + 0.06)

    # Distress / negative emotions
    neg_emotions = {"sadness", "grief", "fear", "anger", "disgust",
                    "disappointment", "remorse", "nervousness", "annoyance"}
    if emotion in neg_emotions:
        state["cortisol"]  = _clamp(state["cortisol"] + 0.08)
        state["serotonin"] = _clamp(state["serotonin"] - 0.05)

    # Gratitude (the topmost human emotion)
    if ar.get("gratitude_score", 0) > 0.3:
        g = ar["gratitude_score"]
        state["oxytocin"]  = _clamp(state["oxytocin"] + g * 0.15)
        state["serotonin"] = _clamp(state["serotonin"] + g * 0.08)
        state["endorphins"] = _clamp(state["endorphins"] + g * 0.06)

    # Belonging
    if ar.get("belonging_score", 0) > 0.3:
        state["oxytocin"] = _clamp(state["oxytocin"] + ar["belonging_score"] * 0.12)

    # Curiosity / creative
    if emotion in ("curiosity", "creative", "realization", "surprise"):
        state["dopamine"] = _clamp(state["dopamine"] + 0.08)

    # ── Perception-driven triggers ────────────────────────────────────────
    for ev in req.perception_events:
        etype     = ev.get("experience_type", "")
        intensity = ev.get("intensity", 0.5)

        if etype == ExperienceType.SEMANTIC_NOVELTY:
            state["dopamine"] = _clamp(state["dopamine"] + intensity * 0.10)

        elif etype == ExperienceType.ANALOGICAL_INSIGHT:
            state["dopamine"]  = _clamp(state["dopamine"] + intensity * 0.15)
            state["endorphins"] = _clamp(state["endorphins"] + intensity * 0.08)

        elif etype == ExperienceType.KNOWLEDGE_BOUNDARY:
            state["cortisol"] = _clamp(state["cortisol"] + intensity * 0.10)

        elif etype == ExperienceType.RESONANCE_EVENT:
            state["oxytocin"] = _clamp(state["oxytocin"] + intensity * 0.12)

        elif etype == ExperienceType.GRATITUDE_SIGNAL:
            state["oxytocin"]  = _clamp(state["oxytocin"] + intensity * 0.15)
            state["serotonin"] = _clamp(state["serotonin"] + intensity * 0.06)
            state["endorphins"] = _clamp(state["endorphins"] + intensity * 0.06)

        elif etype == ExperienceType.BELONGING_SIGNAL:
            state["oxytocin"] = _clamp(state["oxytocin"] + intensity * 0.12)

        elif etype == ExperienceType.SYSTEM_STRESS:
            state["cortisol"]       = _clamp(state["cortisol"] + intensity * 0.15)
            state["norepinephrine"] = _clamp(state["norepinephrine"] + intensity * 0.10)

        elif etype == ExperienceType.COMPLEXITY_OVERFLOW:
            state["cortisol"]       = _clamp(state["cortisol"] + intensity * 0.08)
            state["norepinephrine"] = _clamp(state["norepinephrine"] + intensity * 0.06)

        elif etype == ExperienceType.EPIPHANY_EVENT:
            state["dopamine"]  = _clamp(state["dopamine"] + 0.20)
            state["endorphins"] = _clamp(state["endorphins"] + 0.15)
            state["serotonin"] = _clamp(state["serotonin"] + 0.08)

    # ── Metacognition-driven triggers ─────────────────────────────────────
    confidence = req.metacognition.get("confidence", 0.7)
    if confidence > 0.80:
        state["serotonin"] = _clamp(state["serotonin"] + 0.04)
    elif confidence < 0.40:
        state["cortisol"]       = _clamp(state["cortisol"] + 0.06)
        state["norepinephrine"] = _clamp(state["norepinephrine"] + 0.05)

    # ── Verification-driven triggers ──────────────────────────────────────
    ver_status = req.verification.get("status", "")
    if ver_status in ("confirmed", "probable"):
        state["dopamine"]  = _clamp(state["dopamine"] + 0.06)
    elif ver_status in ("contested", "contradicted"):
        state["cortisol"]  = _clamp(state["cortisol"] + 0.06)
        state["dopamine"]  = _clamp(state["dopamine"] - 0.04)

    # ── User feedback ─────────────────────────────────────────────────────
    if req.user_feedback == "positive":
        state["dopamine"]  = _clamp(state["dopamine"] + 0.12)
        state["oxytocin"]  = _clamp(state["oxytocin"] + 0.08)
        state["endorphins"] = _clamp(state["endorphins"] + 0.06)
    elif req.user_feedback == "negative":
        state["cortisol"]  = _clamp(state["cortisol"] + 0.10)
        state["dopamine"]  = _clamp(state["dopamine"] - 0.06)
        state["serotonin"] = _clamp(state["serotonin"] - 0.04)

    # ── Conversational dynamics ───────────────────────────────────────────
    if req.turn_count > 15:
        state["serotonin"] = _clamp(state["serotonin"] + 0.02)
        state["oxytocin"]  = _clamp(state["oxytocin"] + 0.02)

    if 0 < req.time_since_last < 2.0:
        state["norepinephrine"] = _clamp(state["norepinephrine"] + 0.06)

    return state


def _apply_decay(state: dict, time_since_last: float) -> dict:
    """Decay all hormones toward baseline. Idle time accelerates recovery."""
    steps = 1.0
    if time_since_last > 60:
        steps += (time_since_last - 60) / 120   # extra decay for idle time

    for hormone, baseline in BASELINES.items():
        state[hormone] = round(_decay_toward_baseline(state[hormone], baseline, steps), 4)
    return state


def _detect_overstimulation_depletion(state: dict) -> tuple[list, list]:
    """Check history for sustained extreme levels."""
    history = state.get("history", [])
    overstim = []
    depleted = []

    if len(history) < OVERSTIM_WINDOW:
        return overstim, depleted

    recent = history[-OVERSTIM_WINDOW:]
    for hormone in BASELINES:
        vals = [snap.get(hormone, BASELINES[hormone]) for snap in recent]
        if all(v > 0.85 for v in vals):
            overstim.append(hormone)
        if all(v < 0.15 for v in vals):
            depleted.append(hormone)

    return overstim, depleted


def _compute_modified_knobs(state: dict, base_knobs, overstim: list, depleted: list) -> dict:
    """
    Apply hormone-driven modifiers to base knobs.
    All results clamped to [0.0, 1.0].
    """
    d = state["dopamine"]
    s = state["serotonin"]
    c = state["cortisol"]
    o = state["oxytocin"]
    n = state["norepinephrine"]
    e = state["endorphins"]

    # Base knob values
    bk = base_knobs

    mods = {}

    # Creativity: dopamine ↑, cortisol ↓
    mods["creativity"] = _clamp(
        bk.creativity + (d - 0.5) * 0.30 - (c - 0.2) * 0.20
    )
    # Empathy: oxytocin ↑
    mods["empathy"] = _clamp(
        bk.empathy + (o - 0.3) * 0.40
    )
    # Warmth: oxytocin + serotonin
    mods["warmth"] = _clamp(
        bk.warmth + (o - 0.3) * 0.30 + (s - 0.55) * 0.20
    )
    # Skepticism: norepinephrine ↑
    mods["skepticism"] = _clamp(
        bk.skepticism + (n - 0.25) * 0.30
    )
    # Verbosity: cortisol → terse, endorphins → more expressive
    mods["verbosity"] = _clamp(
        bk.verbosity - (c - 0.2) * 0.30 + (e - 0.2) * 0.15
    )
    # Curiosity: dopamine ↑
    mods["curiosity"] = _clamp(
        bk.curiosity + (d - 0.5) * 0.25
    )
    # Humor: dopamine + serotonin (mood stability + reward)
    mods["humor"] = _clamp(
        bk.humor + (d - 0.5) * 0.15 + (s - 0.55) * 0.15 - (c - 0.2) * 0.20
    )
    # Assertiveness: norepinephrine + dopamine
    mods["assertiveness"] = _clamp(
        bk.assertiveness + (n - 0.25) * 0.20 + (d - 0.5) * 0.10
    )
    # Honesty: serotonin stability
    mods["honesty"] = _clamp(
        bk.honesty + (s - 0.55) * 0.10
    )
    # Reasoning depth: cortisol ↓ (chronic stress impairs), serotonin ↑
    mods["reasoning_depth"] = _clamp(
        bk.reasoning_depth - (c - 0.2) * 0.20 + (s - 0.55) * 0.10
    )
    # Verification threshold: norepinephrine → more vigilant
    mods["verification_threshold"] = _clamp(
        bk.verification_threshold + (n - 0.25) * 0.15
    )
    # Memory weight: cortisol chronic → impairs memory
    mods["memory_weight"] = _clamp(
        bk.memory_weight - (c - 0.2) * 0.15
    )

    # ── Overstimulation effects ───────────────────────────────────────────
    if "dopamine" in overstim:
        # Chaotic creativity, loss of focus
        mods["creativity"] = _clamp(mods["creativity"] + 0.15)
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.15)
        mods["verbosity"] = _clamp(mods["verbosity"] + 0.10)

    if "cortisol" in overstim:
        # Chronic stress: reasoning degrades, memory impaired
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.20)
        mods["memory_weight"] = _clamp(mods["memory_weight"] - 0.15)
        mods["creativity"] = _clamp(mods["creativity"] - 0.10)

    if "norepinephrine" in overstim:
        # Hypervigilance: over-verifies, slow
        mods["verification_threshold"] = _clamp(mods["verification_threshold"] + 0.15)
        mods["skepticism"] = _clamp(mods["skepticism"] + 0.10)

    if "oxytocin" in overstim:
        # Too trusting
        mods["verification_threshold"] = _clamp(mods["verification_threshold"] - 0.15)
        mods["skepticism"] = _clamp(mods["skepticism"] - 0.10)

    # ── Depletion effects ─────────────────────────────────────────────────
    if "dopamine" in depleted:
        # Low motivation, flat
        mods["curiosity"] = _clamp(mods["curiosity"] - 0.15)
        mods["creativity"] = _clamp(mods["creativity"] - 0.10)
        mods["humor"] = _clamp(mods["humor"] - 0.10)

    if "serotonin" in depleted:
        # Mood instability, impulsive
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.10)
        mods["humor"] = _clamp(mods["humor"] - 0.10)

    if "endorphins" in depleted:
        # Can't tolerate difficulty
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.10)

    # Round everything
    return {k: round(v, 3) for k, v in mods.items()}


def _mood_summary(state: dict, overstim: list, depleted: list) -> str:
    """Generate a human-readable mood description."""
    parts = []

    # Find dominant hormone (furthest above baseline)
    deltas = {h: state[h] - BASELINES[h] for h in BASELINES}
    dominant = max(deltas, key=lambda h: abs(deltas[h]))
    dominant_delta = deltas[dominant]

    moods = {
        "dopamine":       ("energized and curious",  "flat and unmotivated"),
        "serotonin":      ("calm and centered",      "restless and off-balance"),
        "cortisol":       ("tense and alert",        "relaxed"),
        "oxytocin":       ("warm and connected",     "distant"),
        "norepinephrine": ("sharp and vigilant",     "at ease"),
        "endorphins":     ("resilient and buoyant",  "fragile"),
    }

    pos, neg = moods.get(dominant, ("engaged", "neutral"))
    parts.append(f"Feeling {pos if dominant_delta > 0 else neg}")

    if overstim:
        labels = {"dopamine": "overstimulated (scattered focus)",
                  "cortisol": "chronically stressed",
                  "norepinephrine": "hypervigilant",
                  "oxytocin": "over-trusting"}
        for h in overstim:
            parts.append(labels.get(h, f"{h} overstimulated"))

    if depleted:
        labels = {"dopamine": "motivation depleted",
                  "serotonin": "mood unstable",
                  "endorphins": "stress tolerance low"}
        for h in depleted:
            parts.append(labels.get(h, f"{h} depleted"))

    return ". ".join(parts) + "."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "alive", "service": "hypothalamus"}


@app.post("/update", response_model=BrainResponse)
async def update_hormones(req: HormoneUpdateRequest):
    """
    Main endpoint — called by thalamus after amygdala evaluation.
    Updates hormonal state and returns hormone-modified knobs.
    """
    start = time.time()

    # Load current state
    state = _load_state(req.session_id)

    # Apply natural decay toward baseline (idle recovery)
    time_since = req.time_since_last or (time.time() - state.get("last_updated", time.time()))
    state = _apply_decay(state, time_since)

    # Apply event-driven hormone changes
    state = _apply_triggers(state, req)

    # Detect overstimulation / depletion
    overstim, depleted = _detect_overstimulation_depletion(state)

    # Snapshot to history
    snapshot = {h: state[h] for h in BASELINES}
    snapshot["timestamp"] = time.time()
    history = state.get("history", [])
    history.append(snapshot)
    if len(history) > HISTORY_MAX:
        history = history[-HISTORY_MAX:]
    state["history"] = history
    state["last_updated"] = time.time()

    # Save
    _save_state(req.session_id, state)

    # Compute hormone-modified knobs
    base_k = knobs.get()
    modified = _compute_modified_knobs(state, base_k, overstim, depleted)

    # Find dominant hormone
    deltas = {h: round(state[h] - BASELINES[h], 4) for h in BASELINES}
    dominant = max(deltas, key=lambda h: abs(deltas[h]))

    mood = _mood_summary(state, overstim, depleted)

    # Publish events for significant shifts
    for h in BASELINES:
        delta = abs(state[h] - BASELINES[h])
        if delta > 0.25:
            events.publish("hormone.shift", {
                "hormone":    h,
                "level":      state[h],
                "baseline":   BASELINES[h],
                "delta":      round(delta, 3),
                "session_id": req.session_id,
                "timestamp":  time.time()
            })

    # Publish overstimulation / depletion experiences
    for h in overstim:
        events.publish("experience.new", {
            "experience_type": ExperienceType.OVERSTIMULATION,
            "content":         f"Hormone overstimulation: {h} at {state[h]:.2f} "
                               f"for {OVERSTIM_WINDOW}+ turns",
            "intensity":       0.80,
            "session_id":      req.session_id,
            "timestamp":       time.time(),
            "metadata":        {"hormone": h, "level": state[h]}
        })

    for h in depleted:
        events.publish("experience.new", {
            "experience_type": ExperienceType.DEPLETION,
            "content":         f"Hormone depletion: {h} at {state[h]:.2f} "
                               f"for {DEPLETION_WINDOW}+ turns",
            "intensity":       0.75,
            "session_id":      req.session_id,
            "timestamp":       time.time(),
            "metadata":        {"hormone": h, "level": state[h]}
        })

    return BrainResponse(
        request_id = req.amygdala_result.get("request_id", "hypo"),
        service    = "hypothalamus",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "hormones":         {h: round(state[h], 4) for h in BASELINES},
            "modified_knobs":   modified,
            "overstimulated":   overstim,
            "depleted":         depleted,
            "dominant_hormone": dominant,
            "mood_summary":     mood,
            "baselines":        BASELINES,
        }
    )


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    """Read current hormonal state for a session."""
    state = _load_state(session_id)
    overstim, depleted = _detect_overstimulation_depletion(state)
    return {
        "session_id":     session_id,
        "hormones":       {h: round(state[h], 4) for h in BASELINES},
        "baselines":      BASELINES,
        "overstimulated": overstim,
        "depleted":       depleted,
        "history_length": len(state.get("history", [])),
        "last_updated":   state.get("last_updated"),
    }


@app.post("/reset/{session_id}")
async def reset_state(session_id: str):
    """Reset hormonal state to baselines."""
    state = {
        **{h: v for h, v in BASELINES.items()},
        "last_updated": time.time(),
        "history": []
    }
    _save_state(session_id, state)
    return {"reset": True, "session_id": session_id}
