# services/hypothalamus/main.py
# ── Hypothalamus: Gland-Based Hormonal State Engine ───────────────────────────
# Models anatomical glands that produce and release hormones into the bot's
# "bloodstream".  Each gland has capacity, release rate, and refractory period.
# Experiences trigger glands (not raw hormone levels directly), and glands
# release hormones that raise or lower from baseline.
#
# 16 hormones / neurotransmitters tracked, produced by 7 glands:
#
#   ADRENAL GLANDS (medulla + cortex):
#     adrenaline (epinephrine) — acute fight-or-flight, burst energy, fear
#     cortisol                 — stress response, focus (acute) / impairment (chronic)
#     norepinephrine           — alertness, sustained attention, sensory sharpening
#
#   PITUITARY GLAND (anterior + posterior):
#     oxytocin                 — trust, bonding, empathy boost
#     vasopressin (ADH)        — social reinforcement, protective/territorial instinct
#     endorphins               — resilience, persistence, pain/stress suppression
#
#   PINEAL GLAND:
#     melatonin                — fatigue/engagement cycles, "mental tiredness"
#     serotonin (co-regulated) — mood stability, impulse control, patience
#
#   HYPOTHALAMIC NUCLEI (self-regulation):
#     GABA                     — anxiety suppression, calm, inhibitory tone
#     glutamate                — excitatory drive, thinking speed, neural plasticity
#
#   VENTRAL TEGMENTAL AREA (VTA) + NUCLEUS ACCUMBENS:
#     dopamine                 — reward, motivation, novelty-seeking, creativity
#     anandamide               — bliss / creative flow states, pattern recognition
#     phenylethylamine (PEA)   — infatuation/passion for topic, obsessive interest
#
#   BASAL FOREBRAIN (CHOLINERGIC):
#     acetylcholine            — learning, memory formation, attention focus
#
#   GONADS (abstract — personality coloring):
#     testosterone             — confidence, risk-taking, competitive drive
#     estrogen                 — emotional depth, verbal fluency, social cognition
#
# Overstimulation (>0.85 for 5+ turns) and depletion (<0.15 for 5+ turns)
# produce distinct behavioral effects mimicking neurochemical dysregulation.

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

app = FastAPI(title="Hypothalamus", version="2.0.0",
              description="Gland-based hormonal state engine — anatomical glands "
                          "release hormones that modulate personality")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

HORMONE_KEY      = "hormone:{session_id}:state"
HISTORY_MAX      = 50
OVERSTIM_WINDOW  = 5
DEPLETION_WINDOW = 5

# ── Baselines ─────────────────────────────────────────────────────────────────
# Resting levels — experiences push hormones above or below these.

BASELINES = {
    # Adrenal glands
    "adrenaline":       0.10,   # low at rest; spikes acutely
    "cortisol":         0.20,
    "norepinephrine":   0.25,
    # Pituitary gland
    "oxytocin":         0.30,
    "vasopressin":      0.20,
    "endorphins":       0.20,
    # Pineal gland
    "melatonin":        0.15,   # low when alert
    "serotonin":        0.55,
    # Hypothalamic nuclei
    "GABA":             0.50,   # tonic inhibitory tone
    "glutamate":        0.45,   # tonic excitatory drive
    # VTA / Nucleus Accumbens
    "dopamine":         0.50,
    "anandamide":       0.25,   # low at rest; flow states push it up
    "PEA":              0.10,   # low at rest; passionate interest spikes it
    # Basal forebrain
    "acetylcholine":    0.40,   # moderate at rest; learning pushes it up
    # Gonads (abstract)
    "testosterone":     0.35,
    "estrogen":         0.35,
}

# ── Glands ────────────────────────────────────────────────────────────────────
# Each gland produces specific hormones.  Glands have:
#   capacity   : max hormone units a gland can release per interaction (0-1)
#   refractory : how many interactions to fully recharge after max release
#   decay_tau  : how fast the gland's hormones decay toward baseline

GLANDS = {
    "adrenal": {
        "hormones":    ["adrenaline", "cortisol", "norepinephrine"],
        "capacity":    1.0,
        "refractory":  4,     # fast-recharging (fight-or-flight must be ready)
        "decay_tau":   2.0,   # adrenaline fades fast
    },
    "pituitary": {
        "hormones":    ["oxytocin", "vasopressin", "endorphins"],
        "capacity":    0.8,
        "refractory":  5,
        "decay_tau":   4.0,   # bonding hormones linger
    },
    "pineal": {
        "hormones":    ["melatonin", "serotonin"],
        "capacity":    0.6,
        "refractory":  6,     # slow cycling (circadian-like)
        "decay_tau":   5.0,
    },
    "hypothalamic_nuclei": {
        "hormones":    ["GABA", "glutamate"],
        "capacity":    0.7,
        "refractory":  3,
        "decay_tau":   3.0,
    },
    "VTA": {
        "hormones":    ["dopamine", "anandamide", "PEA"],
        "capacity":    0.9,
        "refractory":  3,
        "decay_tau":   3.0,
    },
    "basal_forebrain": {
        "hormones":    ["acetylcholine"],
        "capacity":    0.7,
        "refractory":  4,
        "decay_tau":   3.5,
    },
    "gonads": {
        "hormones":    ["testosterone", "estrogen"],
        "capacity":    0.5,
        "refractory":  8,     # slow-cycling
        "decay_tau":   6.0,
    },
}

# Reverse lookup: hormone → gland name
HORMONE_TO_GLAND = {}
for _gname, _ginfo in GLANDS.items():
    for _h in _ginfo["hormones"]:
        HORMONE_TO_GLAND[_h] = _gname

DEFAULT_DECAY_TAU = 3.0


def _decay_toward_baseline(current: float, baseline: float,
                           steps: float = 1.0, tau: float = DEFAULT_DECAY_TAU) -> float:
    alpha = 1.0 - math.exp(-steps / tau)
    return current + alpha * (baseline - current)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Request / Response Models ─────────────────────────────────────────────────

class HormoneUpdateRequest(BaseModel):
    session_id:        str
    amygdala_result:   dict = {}
    perception_events: list = []
    metacognition:     dict = {}
    verification:      dict = {}
    user_feedback:     Optional[str] = None
    turn_count:        int  = 0
    time_since_last:   float = 0.0


class HormoneStateResponse(BaseModel):
    hormones:          dict
    glands:            dict = {}
    modified_knobs:    dict
    overstimulated:    list = []
    depleted:          list = []
    dominant_hormone:  str  = ""
    mood_summary:      str  = ""


# ── Gland State Management ────────────────────────────────────────────────────

def _default_gland_states() -> dict:
    """Fresh gland states — each gland starts at full capacity."""
    return {
        gname: {"energy": ginfo["capacity"], "last_release": 0.0}
        for gname, ginfo in GLANDS.items()
    }


def _recharge_glands(gland_states: dict, time_since_last: float) -> dict:
    """Glands recharge over time. Refractory period = interactions to full."""
    for gname, ginfo in GLANDS.items():
        gs = gland_states.get(gname, {"energy": ginfo["capacity"], "last_release": 0.0})
        refrac = ginfo["refractory"]
        cap = ginfo["capacity"]
        # Each interaction recharges 1/refractory of capacity
        recharge = cap / max(refrac, 1)
        # Idle time gives extra recharge
        if time_since_last > 30:
            recharge += cap * min((time_since_last - 30) / 300, 0.5)
        gs["energy"] = min(cap, gs.get("energy", 0) + recharge)
        gland_states[gname] = gs
    return gland_states


def _gland_release(gland_states: dict, hormone: str, amount: float) -> float:
    """
    Request a release of `amount` of a hormone from its gland.
    Returns the actual amount released (limited by gland energy).
    Depletes the gland's energy accordingly.
    """
    gname = HORMONE_TO_GLAND.get(hormone)
    if not gname:
        return amount  # fallback: no gland constraint
    gs = gland_states.get(gname, {"energy": GLANDS[gname]["capacity"], "last_release": 0.0})
    available = gs.get("energy", 0)
    actual = min(abs(amount), available) * (1.0 if amount >= 0 else -1.0)
    gs["energy"] = max(0, available - abs(actual))
    gs["last_release"] = time.time()
    gland_states[gname] = gs
    return actual


# ── Core Logic ────────────────────────────────────────────────────────────────

def _load_state(session_id: str) -> dict:
    key = HORMONE_KEY.format(session_id=session_id)
    raw = r.get(key)
    if raw:
        return json.loads(raw)
    return {
        **{h: v for h, v in BASELINES.items()},
        "glands": _default_gland_states(),
        "last_updated": time.time(),
        "history": []
    }


def _save_state(session_id: str, state: dict):
    key = HORMONE_KEY.format(session_id=session_id)
    r.set(key, json.dumps(state), ex=86400 * 7)


def _apply_triggers(state: dict, req: HormoneUpdateRequest) -> dict:
    """
    Apply hormone deltas via gland releases based on amygdala, perception,
    metacognition, verification, and user feedback signals.
    """
    gland_states = state.get("glands", _default_gland_states())
    ar = req.amygdala_result
    urgency = ar.get("urgency", "low")
    emotion = ar.get("emotion", "neutral")
    threat  = ar.get("threat_detected", False)

    def _release(hormone, amount):
        """Release hormone via gland and apply to state."""
        actual = _gland_release(gland_states, hormone, amount)
        state[hormone] = _clamp(state.get(hormone, BASELINES.get(hormone, 0.5)) + actual)

    # ── Amygdala-driven: adrenal gland activation ─────────────────────────
    if threat or urgency == "critical":
        _release("adrenaline", 0.30)       # acute burst
        _release("cortisol", 0.20)
        _release("norepinephrine", 0.20)
        _release("GABA", -0.10)            # GABA drops in acute stress
        _release("glutamate", 0.10)        # excitatory surge

    if urgency in ("high", "distressed"):
        _release("adrenaline", 0.15)
        _release("cortisol", 0.10)
        _release("norepinephrine", 0.08)

    # Joy / excitement / positive emotions
    pos_emotions = {"joy", "excitement", "admiration", "amusement", "gratitude",
                    "love", "optimism", "pride", "relief", "approval", "caring"}
    if emotion in pos_emotions:
        _release("dopamine", 0.10)
        _release("endorphins", 0.06)
        _release("serotonin", 0.04)
        _release("anandamide", 0.05)       # bliss component
        _release("GABA", 0.04)             # calming positive

    # Distress / negative emotions
    neg_emotions = {"sadness", "grief", "fear", "anger", "disgust",
                    "disappointment", "remorse", "nervousness", "annoyance"}
    if emotion in neg_emotions:
        _release("cortisol", 0.08)
        _release("serotonin", -0.05)
        _release("GABA", -0.06)            # anxiety = low GABA
        if emotion in ("fear", "anger"):
            _release("adrenaline", 0.15)
            _release("testosterone", 0.06) # fight response

    # Gratitude
    if ar.get("gratitude_score", 0) > 0.3:
        g = ar["gratitude_score"]
        _release("oxytocin", g * 0.15)
        _release("serotonin", g * 0.08)
        _release("endorphins", g * 0.06)
        _release("vasopressin", g * 0.05)

    # Belonging
    if ar.get("belonging_score", 0) > 0.3:
        b = ar["belonging_score"]
        _release("oxytocin", b * 0.12)
        _release("vasopressin", b * 0.08)
        _release("estrogen", b * 0.04)     # social cognition

    # Curiosity / creative / surprise
    if emotion in ("curiosity", "creative", "realization", "surprise"):
        _release("dopamine", 0.08)
        _release("acetylcholine", 0.08)    # attention + learning
        _release("PEA", 0.06)             # passionate interest
        _release("glutamate", 0.05)        # excitatory for thinking

    # ── Perception-driven triggers ────────────────────────────────────────
    for ev in req.perception_events:
        etype     = ev.get("experience_type", "")
        intensity = ev.get("intensity", 0.5)

        if etype == ExperienceType.SEMANTIC_NOVELTY:
            _release("dopamine", intensity * 0.10)
            _release("acetylcholine", intensity * 0.08)
            _release("PEA", intensity * 0.05)

        elif etype == ExperienceType.ANALOGICAL_INSIGHT:
            _release("dopamine", intensity * 0.15)
            _release("endorphins", intensity * 0.08)
            _release("anandamide", intensity * 0.10)  # flow/insight state
            _release("acetylcholine", intensity * 0.06)

        elif etype == ExperienceType.KNOWLEDGE_BOUNDARY:
            _release("cortisol", intensity * 0.10)
            _release("acetylcholine", intensity * 0.05)  # trying to learn
            _release("norepinephrine", intensity * 0.05)

        elif etype == ExperienceType.RESONANCE_EVENT:
            _release("oxytocin", intensity * 0.12)
            _release("estrogen", intensity * 0.04)

        elif etype == ExperienceType.GRATITUDE_SIGNAL:
            _release("oxytocin", intensity * 0.15)
            _release("serotonin", intensity * 0.06)
            _release("endorphins", intensity * 0.06)

        elif etype == ExperienceType.BELONGING_SIGNAL:
            _release("oxytocin", intensity * 0.12)
            _release("vasopressin", intensity * 0.06)

        elif etype == ExperienceType.SYSTEM_STRESS:
            _release("cortisol", intensity * 0.15)
            _release("norepinephrine", intensity * 0.10)
            _release("adrenaline", intensity * 0.12)
            _release("GABA", -intensity * 0.08)

        elif etype == ExperienceType.COMPLEXITY_OVERFLOW:
            _release("cortisol", intensity * 0.08)
            _release("norepinephrine", intensity * 0.06)
            _release("glutamate", intensity * 0.08)  # brain working hard
            _release("melatonin", intensity * 0.05)   # mental fatigue

        elif etype == ExperienceType.EPIPHANY_EVENT:
            _release("dopamine", 0.20)
            _release("endorphins", 0.15)
            _release("serotonin", 0.08)
            _release("anandamide", 0.15)  # eureka = bliss
            _release("PEA", 0.10)         # passionate about discovery

    # ── Metacognition-driven triggers ─────────────────────────────────────
    confidence = req.metacognition.get("confidence", 0.7)
    if confidence > 0.80:
        _release("serotonin", 0.04)
        _release("testosterone", 0.03)    # confidence boost
        _release("GABA", 0.03)            # calm certainty
    elif confidence < 0.40:
        _release("cortisol", 0.06)
        _release("norepinephrine", 0.05)
        _release("acetylcholine", 0.04)   # try harder to figure it out
        _release("GABA", -0.04)           # anxiety from uncertainty

    # ── Verification-driven triggers ──────────────────────────────────────
    ver_status = req.verification.get("status", "")
    if ver_status in ("confirmed", "probable"):
        _release("dopamine", 0.06)
        _release("serotonin", 0.03)
    elif ver_status in ("contested", "contradicted"):
        _release("cortisol", 0.06)
        _release("dopamine", -0.04)
        _release("norepinephrine", 0.04)

    # ── User feedback ─────────────────────────────────────────────────────
    if req.user_feedback == "positive":
        _release("dopamine", 0.12)
        _release("oxytocin", 0.08)
        _release("endorphins", 0.06)
        _release("serotonin", 0.05)
        _release("testosterone", 0.04)   # social success
    elif req.user_feedback == "negative":
        _release("cortisol", 0.10)
        _release("dopamine", -0.06)
        _release("serotonin", -0.04)
        _release("adrenaline", 0.05)     # mild fight-or-flight

    # ── Conversational dynamics ───────────────────────────────────────────
    if req.turn_count > 15:
        _release("serotonin", 0.02)
        _release("oxytocin", 0.02)
        _release("melatonin", 0.03)      # extended conversation = fatigue
        _release("acetylcholine", -0.02)  # attention wanes slightly

    if req.turn_count > 30:
        _release("melatonin", 0.05)      # more fatigue
        _release("glutamate", -0.03)     # thinking slows

    if 0 < req.time_since_last < 2.0:
        _release("norepinephrine", 0.06)
        _release("adrenaline", 0.04)     # rapid-fire = adrenaline

    state["glands"] = gland_states
    return state


def _apply_decay(state: dict, time_since_last: float) -> dict:
    """Decay all hormones toward baseline using per-gland decay tau."""
    steps = 1.0
    if time_since_last > 60:
        steps += (time_since_last - 60) / 120

    for hormone, baseline in BASELINES.items():
        gname = HORMONE_TO_GLAND.get(hormone)
        tau = GLANDS[gname]["decay_tau"] if gname else DEFAULT_DECAY_TAU
        state[hormone] = round(
            _decay_toward_baseline(state.get(hormone, baseline), baseline, steps, tau), 4
        )
    return state


def _detect_overstimulation_depletion(state: dict) -> tuple:
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
    """Apply hormone-driven modifiers to base knobs."""
    d  = state.get("dopamine", BASELINES["dopamine"])
    s  = state.get("serotonin", BASELINES["serotonin"])
    c  = state.get("cortisol", BASELINES["cortisol"])
    o  = state.get("oxytocin", BASELINES["oxytocin"])
    n  = state.get("norepinephrine", BASELINES["norepinephrine"])
    e  = state.get("endorphins", BASELINES["endorphins"])
    ad = state.get("adrenaline", BASELINES["adrenaline"])
    ga = state.get("GABA", BASELINES["GABA"])
    gl = state.get("glutamate", BASELINES["glutamate"])
    ac = state.get("acetylcholine", BASELINES["acetylcholine"])
    an = state.get("anandamide", BASELINES["anandamide"])
    pe = state.get("PEA", BASELINES["PEA"])
    ts = state.get("testosterone", BASELINES["testosterone"])
    es = state.get("estrogen", BASELINES["estrogen"])
    ml = state.get("melatonin", BASELINES["melatonin"])
    vp = state.get("vasopressin", BASELINES["vasopressin"])

    bk = base_knobs
    mods = {}

    # Creativity: dopamine ↑, anandamide ↑ (flow), cortisol ↓, GABA moderate
    mods["creativity"] = _clamp(
        bk.creativity + (d - 0.5) * 0.25 + (an - 0.25) * 0.30
        - (c - 0.2) * 0.20 + (ga - 0.5) * 0.10
    )
    # Empathy: oxytocin ↑, estrogen ↑
    mods["empathy"] = _clamp(
        bk.empathy + (o - 0.3) * 0.35 + (es - 0.35) * 0.20
    )
    # Warmth: oxytocin + serotonin + estrogen
    mods["warmth"] = _clamp(
        bk.warmth + (o - 0.3) * 0.25 + (s - 0.55) * 0.15 + (es - 0.35) * 0.15
    )
    # Skepticism: norepinephrine ↑, adrenaline (hyperalert)
    mods["skepticism"] = _clamp(
        bk.skepticism + (n - 0.25) * 0.25 + (ad - 0.1) * 0.15
    )
    # Verbosity: cortisol → terse, endorphins → expressive, estrogen → fluent
    mods["verbosity"] = _clamp(
        bk.verbosity - (c - 0.2) * 0.25 + (e - 0.2) * 0.12
        + (es - 0.35) * 0.10 - (ml - 0.15) * 0.20  # melatonin = tired = less verbose
    )
    # Curiosity: dopamine ↑, PEA ↑ (obsessive interest), acetylcholine ↑
    mods["curiosity"] = _clamp(
        bk.curiosity + (d - 0.5) * 0.20 + (pe - 0.1) * 0.25
        + (ac - 0.4) * 0.15
    )
    # Humor: dopamine + serotonin + GABA (relaxed), minus cortisol
    mods["humor"] = _clamp(
        bk.humor + (d - 0.5) * 0.12 + (s - 0.55) * 0.12
        + (ga - 0.5) * 0.10 - (c - 0.2) * 0.20
    )
    # Assertiveness: testosterone ↑, norepinephrine ↑, adrenaline ↑
    mods["assertiveness"] = _clamp(
        bk.assertiveness + (ts - 0.35) * 0.25 + (n - 0.25) * 0.15
        + (ad - 0.1) * 0.10
    )
    # Honesty: serotonin stability, GABA (calm = honest)
    mods["honesty"] = _clamp(
        bk.honesty + (s - 0.55) * 0.10 + (ga - 0.5) * 0.08
    )
    # Reasoning depth: acetylcholine ↑ (focus), glutamate ↑ (speed),
    #   cortisol ↓ (chronic stress impairs), melatonin ↓ (fatigue impairs)
    mods["reasoning_depth"] = _clamp(
        bk.reasoning_depth + (ac - 0.4) * 0.20 + (gl - 0.45) * 0.15
        - (c - 0.2) * 0.20 - (ml - 0.15) * 0.20
    )
    # Verification threshold: norepinephrine → vigilant, adrenaline → hyperalert
    mods["verification_threshold"] = _clamp(
        bk.verification_threshold + (n - 0.25) * 0.12 + (ad - 0.1) * 0.10
    )
    # Memory weight: acetylcholine ↑ (memory formation), cortisol chronic ↓
    mods["memory_weight"] = _clamp(
        bk.memory_weight + (ac - 0.4) * 0.20 - (c - 0.2) * 0.15
    )

    # ── Overstimulation effects ───────────────────────────────────────────
    if "dopamine" in overstim:
        mods["creativity"] = _clamp(mods["creativity"] + 0.15)
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.15)
        mods["verbosity"] = _clamp(mods["verbosity"] + 0.10)

    if "cortisol" in overstim:
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.20)
        mods["memory_weight"] = _clamp(mods["memory_weight"] - 0.15)
        mods["creativity"] = _clamp(mods["creativity"] - 0.10)

    if "norepinephrine" in overstim:
        mods["verification_threshold"] = _clamp(mods["verification_threshold"] + 0.15)
        mods["skepticism"] = _clamp(mods["skepticism"] + 0.10)

    if "oxytocin" in overstim:
        mods["verification_threshold"] = _clamp(mods["verification_threshold"] - 0.15)
        mods["skepticism"] = _clamp(mods["skepticism"] - 0.10)

    if "adrenaline" in overstim:
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.15)
        mods["assertiveness"] = _clamp(mods["assertiveness"] + 0.10)
        mods["verbosity"] = _clamp(mods["verbosity"] - 0.10)  # terse under pressure

    if "melatonin" in overstim:
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.20)
        mods["curiosity"] = _clamp(mods["curiosity"] - 0.15)
        mods["verbosity"] = _clamp(mods["verbosity"] - 0.15)

    if "PEA" in overstim:
        # Obsessive hyperfocus on one topic
        mods["curiosity"] = _clamp(mods["curiosity"] + 0.15)
        mods["empathy"] = _clamp(mods["empathy"] - 0.10)

    if "glutamate" in overstim:
        # Excitotoxicity — overactive thinking, anxiety
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.10)
        mods["skepticism"] = _clamp(mods["skepticism"] + 0.10)

    # ── Depletion effects ─────────────────────────────────────────────────
    if "dopamine" in depleted:
        mods["curiosity"] = _clamp(mods["curiosity"] - 0.15)
        mods["creativity"] = _clamp(mods["creativity"] - 0.10)
        mods["humor"] = _clamp(mods["humor"] - 0.10)

    if "serotonin" in depleted:
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.10)
        mods["humor"] = _clamp(mods["humor"] - 0.10)

    if "endorphins" in depleted:
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.10)

    if "GABA" in depleted:
        # High anxiety state
        mods["creativity"] = _clamp(mods["creativity"] - 0.15)
        mods["humor"] = _clamp(mods["humor"] - 0.10)
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.10)

    if "acetylcholine" in depleted:
        mods["memory_weight"] = _clamp(mods["memory_weight"] - 0.15)
        mods["reasoning_depth"] = _clamp(mods["reasoning_depth"] - 0.10)

    return {k: round(v, 3) for k, v in mods.items()}


def _mood_summary(state: dict, overstim: list, depleted: list) -> str:
    """Generate a human-readable mood description."""
    parts = []

    deltas = {h: state.get(h, BASELINES[h]) - BASELINES[h] for h in BASELINES}
    dominant = max(deltas, key=lambda h: abs(deltas[h]))
    dominant_delta = deltas[dominant]

    moods = {
        "dopamine":       ("energized and curious",        "flat and unmotivated"),
        "serotonin":      ("calm and centered",            "restless and off-balance"),
        "cortisol":       ("tense and alert",              "relaxed"),
        "oxytocin":       ("warm and connected",           "distant"),
        "norepinephrine": ("sharp and vigilant",           "at ease"),
        "endorphins":     ("resilient and buoyant",        "fragile"),
        "adrenaline":     ("charged and ready",            "settled"),
        "GABA":           ("tranquil and composed",        "anxious and unsettled"),
        "glutamate":      ("mentally activated",           "mentally quiet"),
        "acetylcholine":  ("focused and attentive",        "distracted"),
        "anandamide":     ("in a creative flow state",     "uninspired"),
        "PEA":            ("passionately engaged",         "disengaged"),
        "melatonin":      ("mentally tired",               "wide awake"),
        "testosterone":   ("confident and decisive",       "hesitant"),
        "estrogen":       ("emotionally perceptive",       "emotionally blunted"),
        "vasopressin":    ("protective and loyal",         "detached"),
    }

    pos, neg = moods.get(dominant, ("engaged", "neutral"))
    parts.append(f"Feeling {pos if dominant_delta > 0 else neg}")

    # Add secondary mood if another hormone is also notably shifted
    sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    if len(sorted_deltas) > 1 and abs(sorted_deltas[1][1]) > 0.10:
        h2 = sorted_deltas[1][0]
        p2, n2 = moods.get(h2, ("engaged", "neutral"))
        parts.append(f"also {p2 if sorted_deltas[1][1] > 0 else n2}")

    if overstim:
        labels = {
            "dopamine": "overstimulated (scattered focus)",
            "cortisol": "chronically stressed",
            "norepinephrine": "hypervigilant",
            "oxytocin": "over-trusting",
            "adrenaline": "adrenaline overload (jittery)",
            "glutamate": "excitotoxic (racing thoughts)",
            "PEA": "obsessively fixated",
            "melatonin": "exhausted",
        }
        for h in overstim:
            parts.append(labels.get(h, f"{h} overstimulated"))

    if depleted:
        labels = {
            "dopamine": "motivation depleted",
            "serotonin": "mood unstable",
            "endorphins": "stress tolerance low",
            "GABA": "anxiety unchecked",
            "acetylcholine": "attention scattered",
        }
        for h in depleted:
            parts.append(labels.get(h, f"{h} depleted"))

    return ". ".join(parts) + "."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "alive", "service": "hypothalamus", "version": "2.0.0",
            "hormones": len(BASELINES), "glands": len(GLANDS)}


@app.post("/update", response_model=BrainResponse)
async def update_hormones(req: HormoneUpdateRequest):
    """
    Main endpoint — called by thalamus after amygdala evaluation.
    Glands release hormones based on triggers; returns hormone-modified knobs.
    """
    start = time.time()
    timings = {}

    t0 = time.time()
    state = _load_state(req.session_id)
    timings["load_state"] = round((time.time() - t0) * 1000, 1)

    # Ensure gland states exist (migration from v1)
    if "glands" not in state:
        state["glands"] = _default_gland_states()

    # Recharge glands based on time elapsed
    t0 = time.time()
    time_since = req.time_since_last or (time.time() - state.get("last_updated", time.time()))
    state["glands"] = _recharge_glands(state.get("glands", _default_gland_states()), time_since)
    timings["recharge_glands"] = round((time.time() - t0) * 1000, 1)

    # Apply natural decay toward baseline
    t0 = time.time()
    state = _apply_decay(state, time_since)
    timings["decay"] = round((time.time() - t0) * 1000, 1)

    # Apply event-driven hormone changes (via gland releases)
    t0 = time.time()
    state = _apply_triggers(state, req)
    timings["triggers"] = round((time.time() - t0) * 1000, 1)

    # Detect overstimulation / depletion
    t0 = time.time()
    overstim, depleted_list = _detect_overstimulation_depletion(state)
    timings["detect_extremes"] = round((time.time() - t0) * 1000, 1)

    # Snapshot to history
    snapshot = {h: state.get(h, BASELINES[h]) for h in BASELINES}
    snapshot["timestamp"] = time.time()
    history = state.get("history", [])
    history.append(snapshot)
    if len(history) > HISTORY_MAX:
        history = history[-HISTORY_MAX:]
    state["history"] = history
    state["last_updated"] = time.time()

    t0 = time.time()
    _save_state(req.session_id, state)
    timings["save_state"] = round((time.time() - t0) * 1000, 1)

    # Compute hormone-modified knobs
    t0 = time.time()
    base_k = knobs.get()
    modified = _compute_modified_knobs(state, base_k, overstim, depleted_list)
    timings["compute_knobs"] = round((time.time() - t0) * 1000, 1)

    # Find dominant hormone
    deltas = {h: round(state.get(h, BASELINES[h]) - BASELINES[h], 4) for h in BASELINES}
    dominant = max(deltas, key=lambda h: abs(deltas[h]))

    mood = _mood_summary(state, overstim, depleted_list)

    total_ms = round((time.time() - start) * 1000, 2)
    timings["total"] = total_ms

    # Log the update operation with full timing breakdown
    _log_operation("update", total_ms, {
        "session_id": req.session_id,
        "dominant": dominant,
        "mood": mood[:80],
        "overstim": overstim,
        "depleted": depleted_list,
        "timings": timings,
    })

    # Publish events for significant shifts
    for h in BASELINES:
        delta = abs(state.get(h, BASELINES[h]) - BASELINES[h])
        if delta > 0.25:
            events.publish("hormone.shift", {
                "hormone":    h,
                "level":      state.get(h, BASELINES[h]),
                "baseline":   BASELINES[h],
                "delta":      round(delta, 3),
                "gland":      HORMONE_TO_GLAND.get(h, "unknown"),
                "session_id": req.session_id,
                "timestamp":  time.time()
            })

    for h in overstim:
        events.publish("experience.new", {
            "experience_type": ExperienceType.OVERSTIMULATION,
            "content":         f"Hormone overstimulation: {h} at {state.get(h, 0):.2f} "
                               f"for {OVERSTIM_WINDOW}+ turns (gland: {HORMONE_TO_GLAND.get(h, '?')})",
            "intensity":       0.80,
            "session_id":      req.session_id,
            "timestamp":       time.time(),
            "metadata":        {"hormone": h, "level": state.get(h, 0), "gland": HORMONE_TO_GLAND.get(h)}
        })

    for h in depleted_list:
        events.publish("experience.new", {
            "experience_type": ExperienceType.DEPLETION,
            "content":         f"Hormone depletion: {h} at {state.get(h, 0):.2f} "
                               f"for {DEPLETION_WINDOW}+ turns (gland: {HORMONE_TO_GLAND.get(h, '?')})",
            "intensity":       0.75,
            "session_id":      req.session_id,
            "timestamp":       time.time(),
            "metadata":        {"hormone": h, "level": state.get(h, 0), "gland": HORMONE_TO_GLAND.get(h)}
        })

    # Gland status summary
    gland_summary = {}
    for gname, ginfo in GLANDS.items():
        gs = state.get("glands", {}).get(gname, {})
        gland_summary[gname] = {
            "energy":   round(gs.get("energy", ginfo["capacity"]), 3),
            "capacity": ginfo["capacity"],
            "pct":      round(gs.get("energy", ginfo["capacity"]) / ginfo["capacity"] * 100, 1),
            "hormones": ginfo["hormones"],
        }

    return BrainResponse(
        request_id = req.amygdala_result.get("request_id", "hypo"),
        service    = "hypothalamus",
        latency_ms = total_ms,
        data       = {
            "hormones":         {h: round(state.get(h, BASELINES[h]), 4) for h in BASELINES},
            "glands":           gland_summary,
            "modified_knobs":   modified,
            "overstimulated":   overstim,
            "depleted":         depleted_list,
            "dominant_hormone": dominant,
            "mood_summary":     mood,
            "baselines":        BASELINES,
            "timings":          timings,
        }
    )


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    """Read current hormonal state and gland status for a session."""
    state = _load_state(session_id)
    overstim, depleted_list = _detect_overstimulation_depletion(state)
    gland_summary = {}
    for gname, ginfo in GLANDS.items():
        gs = state.get("glands", {}).get(gname, {})
        gland_summary[gname] = {
            "energy":   round(gs.get("energy", ginfo["capacity"]), 3),
            "capacity": ginfo["capacity"],
            "pct":      round(gs.get("energy", ginfo["capacity"]) / ginfo["capacity"] * 100, 1),
            "hormones": ginfo["hormones"],
        }
    return {
        "session_id":     session_id,
        "hormones":       {h: round(state.get(h, BASELINES[h]), 4) for h in BASELINES},
        "baselines":      BASELINES,
        "glands":         gland_summary,
        "overstimulated": overstim,
        "depleted":       depleted_list,
        "history_length": len(state.get("history", [])),
        "last_updated":   state.get("last_updated"),
    }


@app.post("/reset/{session_id}")
async def reset_state(session_id: str):
    """Reset hormonal state to baselines and glands to full capacity."""
    state = {
        **{h: v for h, v in BASELINES.items()},
        "glands": _default_gland_states(),
        "last_updated": time.time(),
        "history": []
    }
    _save_state(session_id, state)
    return {"reset": True, "session_id": session_id}


@app.get("/glands")
async def list_glands():
    """Return the anatomical gland definitions and their hormone mappings."""
    return {"glands": GLANDS, "hormone_to_gland": HORMONE_TO_GLAND}


class InjectRequest(BaseModel):
    session_id: str
    hormone: str
    amount: float           # positive = raise, negative = lower from current
    bypass_gland: bool = False   # if True, inject directly without gland constraint


@app.post("/inject")
async def inject_hormone(req: InjectRequest):
    """
    Manually inject (or suppress) a hormone for debugging / experimentation.
    By default, the injection goes through the gland (limited by capacity).
    Set bypass_gland=True to force the exact amount.
    """
    if req.hormone not in BASELINES:
        return {"error": f"Unknown hormone: {req.hormone}",
                "available": list(BASELINES.keys())}

    start = time.time()
    state = _load_state(req.session_id)

    if "glands" not in state:
        state["glands"] = _default_gland_states()

    old_level = state.get(req.hormone, BASELINES[req.hormone])

    if req.bypass_gland:
        actual = req.amount
    else:
        actual = _gland_release(state["glands"], req.hormone, req.amount)

    state[req.hormone] = _clamp(old_level + actual)
    new_level = state[req.hormone]
    state["last_updated"] = time.time()
    _save_state(req.session_id, state)

    gname = HORMONE_TO_GLAND.get(req.hormone, "unknown")
    latency_ms = round((time.time() - start) * 1000, 2)

    # Publish the injection event
    events.publish("hormone.shift", {
        "hormone":    req.hormone,
        "level":      new_level,
        "baseline":   BASELINES[req.hormone],
        "delta":      round(new_level - old_level, 4),
        "gland":      gname,
        "source":     "manual_inject",
        "session_id": req.session_id,
        "timestamp":  time.time()
    })

    # Log the operation
    _log_operation("inject", latency_ms, {
        "hormone": req.hormone, "requested": req.amount,
        "actual": round(actual, 4), "old": round(old_level, 4),
        "new": round(new_level, 4), "gland": gname,
        "bypass": req.bypass_gland, "session_id": req.session_id,
    })

    return {
        "hormone":    req.hormone,
        "old_level":  round(old_level, 4),
        "new_level":  round(new_level, 4),
        "delta":      round(new_level - old_level, 4),
        "requested":  req.amount,
        "actual":     round(actual, 4),
        "gland":      gname,
        "bypass":     req.bypass_gland,
        "latency_ms": latency_ms,
    }


# ── Internal Logging ──────────────────────────────────────────────────────────
# Push structured logs to Redis so the dashboard Logging page shows them

DASHBOARD_LOG_KEY = "logs:hypothalamus"
DASHBOARD_LOG_ALL = "logs:_all"
DASHBOARD_LOG_MAX = 1000


def _log_operation(op: str, latency_ms: float, details: dict = None):
    """Record a structured log entry to Redis for the dashboard."""
    entry = json.dumps({
        "service":   "hypothalamus",
        "level":     "info",
        "message":   f"[{op}] {latency_ms:.1f}ms" + (
            f" — {json.dumps(details)[:300]}" if details else ""
        ),
        "timestamp": time.time(),
    })
    try:
        pipe = r.pipeline()
        pipe.lpush(DASHBOARD_LOG_KEY, entry)
        pipe.ltrim(DASHBOARD_LOG_KEY, 0, DASHBOARD_LOG_MAX - 1)
        pipe.lpush(DASHBOARD_LOG_ALL, entry)
        pipe.ltrim(DASHBOARD_LOG_ALL, 0, DASHBOARD_LOG_MAX - 1)
        pipe.execute()
    except Exception:
        pass
