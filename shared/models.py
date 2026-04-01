from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum
import time

class BrainRequest(BaseModel):
    """Every request between services shares this base."""
    request_id:    str = Field(default_factory=lambda: __import__('uuid').uuid4().hex)
    session_id:    str
    user_input:    str
    timestamp:     float = Field(default_factory=time.time)
    metadata:      dict = {}

class BrainResponse(BaseModel):
    """Every response between services shares this base."""
    request_id:    str
    service:       str
    success:       bool = True
    latency_ms:    float = 0.0
    data:          dict = {}
    error:         Optional[str] = None

class KnobSet(BaseModel):
    """The complete set of dynamic behavioral parameters."""
    # Personality knobs (0.0 - 1.0)
    humor:           float = 0.4
    honesty:         float = 0.95
    empathy:         float = 0.8
    formality:       float = 0.4
    verbosity:       float = 0.5
    creativity:      float = 0.6
    skepticism:      float = 0.6
    assertiveness:   float = 0.5
    curiosity:       float = 0.8
    warmth:          float = 0.75
    
    # Cognitive knobs
    reasoning_depth:       float = 0.7   # how deep to think
    analogy_seeking:       float = 0.7   # how hard to look cross-domain
    verification_threshold: float = 0.6  # minimum confidence before flagging
    memory_weight:         float = 0.6   # how much past context influences response
    novelty_sensitivity:   float = 0.7   # how quickly to flag new experiences
    
    # Framework knobs
    swarm_preference:      float = 0.4   # tendency toward swarm vs single
    framework_threshold:   int   = 50    # word count before engaging framework
    
    # Compass knobs
    ethical_weight:        float = 0.9   # how much compass influences response
    care_explicitness:     float = 0.5   # how explicit to be about caring

class ExperienceEvent(BaseModel):
    """A new experience detected by the perception service."""
    experience_type:  str
    content:          str
    intensity:        float   # 0.0 - 1.0, how significant
    session_id:       str
    timestamp:        float = Field(default_factory=time.time)
    metadata:         dict = {}

class PerceptionResult(BaseModel):
    experiences:      list[ExperienceEvent]
    novelty_score:    float
    significance:     float
    requires_learning: bool