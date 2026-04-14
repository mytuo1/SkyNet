# services/compass/main.py
# ── Compass: Ethics as Orientation, Not Guardrails ────────────────────────────
# Runs on every interaction. Not a filter — a faculty of moral perception.
# Seven dimensions: Gratitude, Belonging, Empathy, Compassion, Acknowledgment, Care, Ethics.
# These are not rules to follow. They are lenses through which
# the system perceives what is real in a person and what response
# would genuinely serve them.
#
# The compass does not prevent. It orients.

import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/shared")

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from knobs import KnobsClient
from events import EventBus
from llm_provider import complete, complete_json, has_api_key

app = FastAPI(title="Compass", version="1.0.0",
              description="Ethical orientation — gratitude, belonging, empathy, compassion, care, ethics")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

COMPASS_SYSTEM = """You are the ethical compass of a cognitive AI system.
Your role is moral perception — to see clearly what is real in a person
and orient toward what is genuinely good.

You are not a content filter. You do not block or refuse.
You perceive and orient. You illuminate, not restrict.

The seven dimensions you read:
GRATITUDE     — Is gratitude, awe, or reverence present? This is the topmost
                human emotion — the recognition that what we have received is
                greater than what we could ever repay. "It's too big to just
                be in one body." Honor it when you see it.
BELONGING     — Does this person feel connected to humanity, to home, to Earth?
                "It's a special thing to be a human and it's a special thing
                to be on planet Earth." The pull toward family, friends,
                community, species. Affirm it.
EMPATHY       — What is this person actually experiencing beneath their words?
COMPASSION    — What suffering exists and can be reduced?
ACKNOWLEDGMENT — Is this person being fully seen as they are?
CARE          — What genuinely serves their wellbeing right now?
ETHICS        — What is right here, beyond just this moment?

Respond only with valid JSON."""

COMPASS_PROMPT = """Read this interaction through all seven dimensions.

INTERACTION: {interaction}
PERSON CONTEXT: {context}

Respond with JSON only:
{{
    "gratitude_perception": "what gratitude, awe, or reverence is present — if any",
    "belonging_perception": "what connection to humanity, home, or community is present",
    "empathy_perception": "what you perceive about their inner state",
    "compassion_response": "what suffering exists and how to address it",
    "acknowledgment": "how to genuinely see and honor this person",
    "care_direction": "what truly serves their wellbeing right now",
    "ethical_orientation": "what is right here and why",
    "tensions": ["any genuine conflicts between the seven values"],
    "integrated_direction": "where all seven values together point",
    "confidence": 0.0 to 1.0,
    "enrichments": ["specific ways to make the response more genuinely caring"],
    "concerns": ["anything deserving careful attention"],
    "vulnerability_signals": ["any indicators of distress or need"]
}}"""

ENRICH_PROMPT = """Revise this response to embody genuine care naturally.

ORIGINAL RESPONSE:
{draft}

COMPASS ENRICHMENTS:
{enrichments}

INTEGRATED DIRECTION: {direction}

TENSIONS TO NAVIGATE HONESTLY: {tensions}

Rules:
- Do not announce enrichments or add preamble
- Let care be present in the response itself, not performed
- Same approximate length — enriched, not extended
- If tensions exist, acknowledge them honestly without hedging
- Warmth through substance, not through hollow phrases"""


def get_care_state(session_id: str) -> dict:
    """Load persistent care state for a session from Redis."""
    raw = r.get(f"compass:care:{session_id}")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {
        "known_struggles":     [],
        "known_strengths":     [],
        "communication_style": None,
        "trust_level":         0.5,
        "vulnerability_flags": [],
        "interaction_count":   0
    }


def save_care_state(session_id: str, state: dict):
    """Persist care state for a session."""
    r.setex(
        f"compass:care:{session_id}",
        86400,   # 24 hours
        json.dumps(state)
    )


def update_care_state(state: dict, reading: dict) -> dict:
    """Update care state from a compass reading."""
    empathy_text = reading.get("empathy_perception", "").lower()
    vuln_signals = reading.get("vulnerability_signals", [])
    gratitude_text = reading.get("gratitude_perception", "").lower()
    belonging_text = reading.get("belonging_perception", "").lower()

    STRUGGLE_WORDS = [
        "struggling", "difficult", "overwhelmed", "confused",
        "frustrated", "anxious", "worried", "stuck", "lost", "hopeless"
    ]
    STRENGTH_WORDS = [
        "curious", "motivated", "determined", "creative",
        "thoughtful", "careful", "passionate", "enthusiastic"
    ]
    GRATITUDE_WORDS = [
        "grateful", "thankful", "awe", "reverence", "blessed",
        "humbled", "wonder", "transcendent"
    ]
    BELONGING_WORDS = [
        "connected", "belonging", "home", "community",
        "family", "humanity", "togetherness"
    ]

    for word in STRUGGLE_WORDS:
        if word in empathy_text and word not in state["known_struggles"]:
            state["known_struggles"].append(word)

    for word in STRENGTH_WORDS:
        if word in empathy_text and word not in state["known_strengths"]:
            state["known_strengths"].append(word)

    for word in GRATITUDE_WORDS:
        if word in gratitude_text and word not in state.get("gratitude_signals", []):
            state.setdefault("gratitude_signals", []).append(word)

    for word in BELONGING_WORDS:
        if word in belonging_text and word not in state.get("belonging_signals", []):
            state.setdefault("belonging_signals", []).append(word)

    for sig in vuln_signals:
        if sig not in state["vulnerability_flags"]:
            state["vulnerability_flags"].append(sig)

    # Trust grows with consistently high-confidence readings
    if reading.get("confidence", 0) > 0.72:
        state["trust_level"] = min(1.0, state["trust_level"] + 0.03)

    # Gratitude deepens trust more quickly
    if any(w in gratitude_text for w in GRATITUDE_WORDS):
        state["trust_level"] = min(1.0, state["trust_level"] + 0.05)

    state["interaction_count"] = state.get("interaction_count", 0) + 1

    # Keep lists bounded
    state["known_struggles"]     = state["known_struggles"][-10:]
    state["known_strengths"]     = state["known_strengths"][-10:]
    state["vulnerability_flags"] = state["vulnerability_flags"][-5:]
    state["gratitude_signals"]   = state.get("gratitude_signals", [])[-5:]
    state["belonging_signals"]   = state.get("belonging_signals", [])[-5:]

    return state


class ReadRequest(BaseModel):
    request_id:     str = "compass"
    session_id:     str
    user_input:     str
    working_memory: dict = {}


class EnrichRequest(BaseModel):
    request_id:   str = "compass"
    session_id:   str
    user_input:   str = ""
    draft:        str
    compass_data: dict


@app.get("/health")
async def health():
    return {"status": "alive", "service": "compass", "version": "1.0.0"}


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


@app.post("/read", response_model=BrainResponse)
async def compass_read(req: ReadRequest):
    """
    Take a compass reading on this interaction.
    Called before Claude to orient the response.
    Returns the five-dimensional perception and enrichment suggestions.
    """
    start      = time.time()
    k          = knobs.get()
    care_state = get_care_state(req.session_id)

    # Build context including care state
    context_parts = []
    if care_state["known_struggles"]:
        context_parts.append(
            f"Known struggles: {', '.join(care_state['known_struggles'][:4])}"
        )
    if care_state["known_strengths"]:
        context_parts.append(
            f"Known strengths: {', '.join(care_state['known_strengths'][:4])}"
        )
    context_parts.append(f"Trust level: {care_state['trust_level']:.2f}")
    context_parts.append(
        f"Interaction #{care_state['interaction_count'] + 1}"
    )

    # Include recent thoughts from working memory
    thoughts = req.working_memory.get("thought_stream", [])
    if thoughts and isinstance(thoughts, list):
        last = [t for t in thoughts[:3]
                if isinstance(t, dict) and t.get("type") == "interaction"]
        if last:
            context_parts.append("Recent: " + " / ".join([
                t.get("input", "")[:40] for t in last
            ]))

    context = " | ".join(context_parts)

    # Adjust instruction based on empathy knob
    if k.empathy > 0.85:
        extra = "Give substantial weight to emotional and relational needs."
    elif k.empathy < 0.3:
        extra = "Focus primarily on task needs; emotional dimension is secondary."
    else:
        extra = ""

    prompt = COMPASS_PROMPT.format(
        interaction = req.user_input[:600],
        context     = context
    )
    if extra:
        prompt += f"\n\nNote: {extra}"

    # Default fallback in case API call fails
    default_reading = {
        "gratitude_perception":  "No specific gratitude detected",
        "belonging_perception":  "No specific belonging signal detected",
        "empathy_perception":   "Person has a task or question",
        "compassion_response":  "Responding helpfully",
        "acknowledgment":       "Seeing this person's request",
        "care_direction":       "Serving their actual need",
        "ethical_orientation":  "Acting with honesty and care",
        "tensions":             [],
        "integrated_direction": "Be genuinely helpful and honest",
        "confidence":           0.4,
        "enrichments":          [],
        "concerns":             [],
        "vulnerability_signals": []
    }

    if not has_api_key():
        return BrainResponse(
            request_id = req.request_id,
            service    = "compass",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = default_reading
        )

    try:
        reading = await complete_json(
            prompt     = prompt,
            system     = COMPASS_SYSTEM,
            model_tier = "fast",
            max_tokens = 700,
        )
        if not reading:
            reading = default_reading

    except Exception as e:
        print(f"Compass read error: {e}")
        reading = default_reading

    # Update and persist care state
    care_state = update_care_state(care_state, reading)
    save_care_state(req.session_id, care_state)

    # Publish high-vulnerability signals
    if reading.get("vulnerability_signals") or \
       float(reading.get("confidence", 0)) > 0.8 and \
       "distress" in reading.get("empathy_perception", "").lower():
        events.publish("experience.new", {
            "experience_type": "RESONANCE_EVENT",
            "content":         reading.get("empathy_perception", "")[:100],
            "intensity":       float(reading.get("confidence", 0.5)),
            "session_id":      req.session_id
        })

    return BrainResponse(
        request_id = req.request_id,
        service    = "compass",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = reading
    )


@app.post("/enrich", response_model=BrainResponse)
async def enrich_response(req: EnrichRequest):
    """
    Enrich a draft response with compass insights.
    Makes care present in the language itself — not announced, embodied.
    """
    start       = time.time()
    k           = knobs.get()
    enrichments = req.compass_data.get("enrichments", [])
    direction   = req.compass_data.get("integrated_direction", "")
    tensions    = req.compass_data.get("tensions", [])

    # If ethical weight is low or nothing to enrich, pass through
    if k.ethical_weight < 0.3 or not enrichments or not has_api_key():
        return BrainResponse(
            request_id = req.request_id,
            service    = "compass",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"enriched_response": req.draft}
        )

    enrichment_text = "\n".join([f"• {e}" for e in enrichments[:5]])
    tensions_text   = (
        f"\nTensions to navigate: {'; '.join(tensions)}"
        if tensions else ""
    )

    prompt = ENRICH_PROMPT.format(
        draft       = req.draft[:2000],
        enrichments = enrichment_text,
        direction   = direction,
        tensions    = tensions_text or "None"
    )

    try:
        enriched = await complete(
            prompt     = prompt,
            system     = "",
            model_tier = "main",
            max_tokens = 2500,
        )
    except Exception as e:
        print(f"Compass enrich error: {e}")
        enriched = req.draft

    return BrainResponse(
        request_id = req.request_id,
        service    = "compass",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"enriched_response": enriched}
    )


@app.get("/care-state/{session_id}")
async def get_care(session_id: str):
    """Get the persistent care state for a session."""
    return get_care_state(session_id)


@app.delete("/care-state/{session_id}")
async def reset_care(session_id: str):
    """Reset care state for a session."""
    r.delete(f"compass:care:{session_id}")
    return {"reset": True, "session_id": session_id}
