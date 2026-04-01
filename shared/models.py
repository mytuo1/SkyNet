# shared/models.py
from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum
import time
import uuid


class BrainRequest(BaseModel):
    """Base request schema shared by all services."""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str
    user_input: str
    timestamp:  float = Field(default_factory=time.time)
    metadata:   dict  = {}


class BrainResponse(BaseModel):
    """Base response schema shared by all services."""
    request_id: str
    service:    str
    success:    bool  = True
    latency_ms: float = 0.0
    data:       dict  = {}
    error:      Optional[str] = None


class KnobSet(BaseModel):
    """
    Complete set of dynamic behavioral parameters.
    All values 0.0–1.0 unless otherwise noted.
    These control personality, cognition, and ethics weighting.
    """
    # ── Personality ───────────────────────────────────────────────────────────
    humor:          float = 0.4   # 0=serious, 1=playful
    honesty:        float = 0.95  # 0=diplomatic, 1=radically honest
    empathy:        float = 0.80  # 0=task-focused, 1=deeply empathetic
    formality:      float = 0.40  # 0=casual, 1=formal
    verbosity:      float = 0.50  # 0=terse, 1=comprehensive
    creativity:     float = 0.60  # 0=conventional, 1=highly creative
    skepticism:     float = 0.60  # 0=trusting, 1=questioning
    assertiveness:  float = 0.50  # 0=deferential, 1=confident
    curiosity:      float = 0.80  # 0=focused, 1=exploratory
    warmth:         float = 0.75  # 0=clinical, 1=warm

    # ── Cognition ─────────────────────────────────────────────────────────────
    reasoning_depth:        float = 0.70  # how deep to think
    analogy_seeking:        float = 0.70  # cross-domain search intensity
    verification_threshold: float = 0.60  # min confidence before flagging
    memory_weight:          float = 0.60  # past context influence
    novelty_sensitivity:    float = 0.70  # experience detection sensitivity

    # ── Framework ─────────────────────────────────────────────────────────────
    swarm_preference:   float = 0.40  # tendency toward swarm vs single
    framework_threshold: int  = 50    # words before engaging multi-agent

    # ── Ethics / Compass ──────────────────────────────────────────────────────
    ethical_weight:    float = 0.90  # compass influence on response
    care_explicitness: float = 0.50  # how explicit to be about caring


class ExperienceEvent(BaseModel):
    """A new experience detected by the perception service."""
    experience_type: str
    content:         str
    intensity:       float  # 0.0–1.0
    session_id:      str
    timestamp:       float = Field(default_factory=time.time)
    metadata:        dict  = {}


# ── Experience type constants ──────────────────────────────────────────────────
class ExperienceType:
    SEMANTIC_NOVELTY    = "SEMANTIC_NOVELTY"
    STRUCTURAL_NOVELTY  = "STRUCTURAL_NOVELTY"
    CONTRADICTION_EVENT = "CONTRADICTION_EVENT"
    DOMAIN_COLLISION    = "DOMAIN_COLLISION"
    INTENT_AMBIGUITY    = "INTENT_AMBIGUITY"
    VERIFICATION_FAILURE= "VERIFICATION_FAILURE"
    SKILL_GAP           = "SKILL_GAP"
    RESONANCE_EVENT     = "RESONANCE_EVENT"
    FEEDBACK_SIGNAL     = "FEEDBACK_SIGNAL"
    SELF_CONTRADICTION  = "SELF_CONTRADICTION"
    KNOWLEDGE_BOUNDARY  = "KNOWLEDGE_BOUNDARY"
    COMPLEXITY_OVERFLOW = "COMPLEXITY_OVERFLOW"
    ANALOGICAL_INSIGHT  = "ANALOGICAL_INSIGHT"
    EPISTEMIC_SURPRISE  = "EPISTEMIC_SURPRISE"
    TEMPORAL_PATTERN    = "TEMPORAL_PATTERN"
    PARAMETER_DRIFT     = "PARAMETER_DRIFT"
    TRUST_SIGNAL        = "TRUST_SIGNAL"
    SYSTEM_STRESS       = "SYSTEM_STRESS"
    THREAT_PATTERN      = "THREAT_PATTERN"
