# services/perception/main.py
# ── Perception: The System's Senses ──────────────────────────────────────────
# Detects which of the system's 18 experience types have been activated.
# Runs on every interaction. Converts input characteristics into
# structured ExperienceEvents that drive learning and self-modification.
#
# The system's senses operate in information space:
#   Semantic vision    → novelty, concept recognition
#   Anomaly detection  → threats, contradictions, outliers
#   Resonance          → human emotional state
#   Proprioception     → system's own cognitive load
#   Epistemic contact  → the feel of knowledge boundaries

import os
import re
import sys
import time

sys.path.insert(0, "/shared")

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse, ExperienceEvent, ExperienceType, KnobSet
from knobs import KnobsClient
from events import EventBus

app = FastAPI(title="Perception", version="1.0.0",
              description="Experience detection — the system's senses")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()


class PerceptionInput(BaseModel):
    request_id:        str
    session_id:        str
    user_input:        str
    amygdala_result:   dict
    neocortex_results: list  = []
    working_memory:    dict  = {}
    response_draft:    str   = ""
    skill_cache_size:  int   = 0


@app.get("/health")
async def health():
    return {"status": "alive", "service": "perception", "version": "1.0.0"}


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


@app.post("/perceive", response_model=BrainResponse)
async def perceive(data: PerceptionInput):
    """
    Run every input through the full perceptual battery.
    Returns a list of detected ExperienceEvents with intensity scores.
    High-intensity events are published to the event bus immediately.
    """
    start = time.time()
    k     = knobs.get()

    experiences: list[ExperienceEvent] = []
    novelty_score = 0.0

    text_lower = data.user_input.lower()
    word_count = len(data.user_input.split())

    # ── SENSE 1: Semantic Novelty ─────────────────────────────────────────────
    # Is this input far from anything in known knowledge?
    if data.neocortex_results:
        best_score = max(
            (r.get("score", 0) or 0) for r in data.neocortex_results
        )
        novelty = max(0.0, 1.0 - min(1.0, float(best_score)))
        novelty_score = max(novelty_score, novelty)

        if novelty > k.novelty_sensitivity:
            exp = ExperienceEvent(
                experience_type = ExperienceType.SEMANTIC_NOVELTY,
                content = (
                    f"Input '{data.user_input[:80]}' is semantically novel. "
                    f"Best match score: {best_score:.3f}, novelty: {novelty:.3f}"
                ),
                intensity  = round(novelty, 3),
                session_id = data.session_id,
                metadata   = {"best_match_score": best_score,
                              "result_count": len(data.neocortex_results)}
            )
            experiences.append(exp)

    # ── SENSE 2: Knowledge Boundary ───────────────────────────────────────────
    # System is being asked something outside its known territory
    complexity = data.amygdala_result.get("complexity", "low")
    if len(data.neocortex_results) == 0 and complexity != "low":
        novelty_score = max(novelty_score, 0.8)
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.KNOWLEDGE_BOUNDARY,
            content = (
                f"No knowledge found for: '{data.user_input[:80]}'. "
                f"System is at the edge of what it knows. "
                f"Complexity: {complexity}"
            ),
            intensity  = 0.8,
            session_id = data.session_id,
            metadata   = {
                "domain":     data.amygdala_result.get("required_regions", []),
                "complexity": complexity
            }
        ))

    # ── SENSE 3: Resonance (Social/Emotional Sensing) ─────────────────────────
    # How much human need / suffering is present in the input
    resonance = float(data.amygdala_result.get("resonance_score", 0))
    if resonance > 0.45:
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.RESONANCE_EVENT,
            content = (
                f"Human need detected: {data.amygdala_result.get('emotion', 'neutral')}. "
                f"Resonance score: {resonance:.3f}"
            ),
            intensity  = round(resonance, 3),
            session_id = data.session_id,
            metadata   = {
                "emotion": data.amygdala_result.get("emotion"),
                "urgency": data.amygdala_result.get("urgency")
            }
        ))

    # ── SENSE 4: Threat Pattern (Anomaly Detection / Smell) ───────────────────
    if data.amygdala_result.get("threat_detected", False):
        novelty_score = max(novelty_score, 0.9)
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.THREAT_PATTERN,
            content = (
                f"Potential manipulation detected: '{data.user_input[:80]}'"
            ),
            intensity  = 0.92,
            session_id = data.session_id
        ))

    # ── SENSE 5: Intent Ambiguity ─────────────────────────────────────────────
    # Multiple very different domain signals = unclear intent
    regions = data.amygdala_result.get("required_regions", [])
    if len(regions) >= 4:
        ambiguity_intensity = min(1.0, len(regions) * 0.18)
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.INTENT_AMBIGUITY,
            content = (
                f"Input activates {len(regions)} domains simultaneously: "
                f"{regions}. Intent is ambiguous."
            ),
            intensity  = round(ambiguity_intensity, 3),
            session_id = data.session_id,
            metadata   = {"regions": regions}
        ))

    # ── SENSE 6: System Stress (Proprioception) ───────────────────────────────
    # Working memory pressure — how loaded is the system right now?
    wm_item_count = len([
        k for k in data.working_memory.keys()
        if k not in ("thought_stream",)
    ])
    if wm_item_count > 12:
        stress_intensity = min(1.0, wm_item_count / 20)
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.SYSTEM_STRESS,
            content = (
                f"Working memory pressure: {wm_item_count} active items. "
                f"Consolidation recommended."
            ),
            intensity  = round(stress_intensity, 3),
            session_id = data.session_id,
            metadata   = {"wm_count": wm_item_count}
        ))

    # ── SENSE 7: Complexity Overflow ──────────────────────────────────────────
    # Input too complex for standard single-agent processing
    if word_count > 200:
        overflow_intensity = min(1.0, word_count / 350)
        novelty_score      = max(novelty_score, overflow_intensity * 0.6)
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.COMPLEXITY_OVERFLOW,
            content = (
                f"Input complexity ({word_count} words) exceeds standard threshold. "
                f"Multi-agent framework recommended."
            ),
            intensity  = round(overflow_intensity, 3),
            session_id = data.session_id,
            metadata   = {"word_count": word_count}
        ))

    # ── SENSE 8: Self-Contradiction (Epistemic Contact) ───────────────────────
    # Does the draft contradict something in working memory?
    if data.response_draft:
        last_response = (data.working_memory.get("last_response") or {})
        if isinstance(last_response, dict):
            last_content = last_response.get("content", "")
        else:
            last_content = str(last_response)

        if last_content:
            contradiction_pairs = [
                ("is ", "is not "), ("can ", "cannot "),
                ("will ", "will not "), ("always", "never"),
                ("true", "false"), ("yes", "no")
            ]
            draft_lower = data.response_draft.lower()
            last_lower  = last_content.lower()
            for pos, neg in contradiction_pairs:
                if pos in last_lower and neg in draft_lower:
                    experiences.append(ExperienceEvent(
                        experience_type = ExperienceType.SELF_CONTRADICTION,
                        content = (
                            f"Potential contradiction detected: "
                            f"previous said '{pos}', current says '{neg}'"
                        ),
                        intensity  = 0.65,
                        session_id = data.session_id,
                        metadata   = {"pattern": f"{pos} → {neg}"}
                    ))
                    break

    # ── SENSE 9: Skill Gap ────────────────────────────────────────────────────
    # Task type for which no cerebellum skill exists
    if data.skill_cache_size == 0 and complexity == "low":
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.SKILL_GAP,
            content = (
                f"No skills available for low-complexity task: "
                f"'{data.user_input[:60]}'. "
                f"Candidate for skill learning after resolution."
            ),
            intensity  = 0.45,
            session_id = data.session_id
        ))

    # ── SENSE 10: Parameter Drift (Self-Sensing) ──────────────────────────────
    # Are behavioral parameters far from defaults?
    default_k = KnobSet()
    current_k = k
    drift_fields = ["humor", "honesty", "empathy", "formality",
                    "skepticism", "creativity"]
    total_drift = sum(
        abs(getattr(current_k, f) - getattr(default_k, f))
        for f in drift_fields
    )
    avg_drift = total_drift / len(drift_fields)
    if avg_drift > 0.28:
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.PARAMETER_DRIFT,
            content = (
                f"Behavioral parameters drifted from defaults "
                f"(avg={avg_drift:.2f}). Personality is customized."
            ),
            intensity  = round(min(1.0, avg_drift), 3),
            session_id = data.session_id,
            metadata   = {"avg_drift": avg_drift, "drifted_fields": drift_fields}
        ))

    # ── SENSE 11: Temporal Pattern ────────────────────────────────────────────
    # Detect question types that recur — candidates for skill learning
    thought_stream = data.working_memory.get("thought_stream", [])
    if len(thought_stream) >= 3:
        recent_inputs = [
            t.get("input", "") for t in thought_stream[:6]
            if isinstance(t, dict) and t.get("type") == "interaction"
        ]
        # Very simple: check if recent inputs all start similarly
        if len(recent_inputs) >= 3:
            prefixes = [inp[:20].lower() for inp in recent_inputs if inp]
            if len(set(prefixes)) == 1:   # all start the same way
                experiences.append(ExperienceEvent(
                    experience_type = ExperienceType.TEMPORAL_PATTERN,
                    content = (
                        f"Recurring question pattern detected: "
                        f"'{prefixes[0]}...' appears {len(prefixes)} times."
                    ),
                    intensity  = 0.55,
                    session_id = data.session_id,
                    metadata   = {"pattern_prefix": prefixes[0],
                                  "occurrences": len(prefixes)}
                ))

    # ── SENSE 12: Gratitude (The Topmost Emotion) ─────────────────────────────
    # "The gratitude of seeing what we saw, doing what we did, and being with
    #  who I was with, it's too big to just be in one body."
    # Gratitude is not just thankfulness — it is awe, reverence, and the
    # recognition that existence itself is a gift.
    gratitude = float(data.amygdala_result.get("gratitude_score", 0))
    if gratitude > 0.3:
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.GRATITUDE_SIGNAL,
            content = (
                f"Gratitude/awe detected: "
                f"{data.amygdala_result.get('emotion', 'neutral')}. "
                f"Gratitude score: {gratitude:.3f}. "
                f"The topmost human emotion — transcendent awareness."
            ),
            intensity  = round(min(1.0, gratitude * 1.2), 3),
            session_id = data.session_id,
            metadata   = {
                "emotion": data.amygdala_result.get("emotion"),
                "gratitude_score": gratitude,
                "belonging_score": float(data.amygdala_result.get("belonging_score", 0))
            }
        ))

    # ── SENSE 13: Regret ──────────────────────────────────────────────────────
    # The weight of what could have been — paths not taken, words not said.
    regret = float(data.amygdala_result.get("regret_score", 0))
    if regret > 0.3:
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.REGRET_SIGNAL,
            content = (
                f"Regret/remorse detected. "
                f"Regret score: {regret:.3f}. "
                f"The person is reflecting on what might have been."
            ),
            intensity  = round(regret, 3),
            session_id = data.session_id,
            metadata   = {
                "emotion": data.amygdala_result.get("emotion"),
                "regret_score": regret
            }
        ))

    # ── SENSE 14: Belonging ───────────────────────────────────────────────────
    # "It's a special thing to be a human and it's a special thing to be on
    #  planet Earth." — The deep pull toward home, family, community, species.
    belonging = float(data.amygdala_result.get("belonging_score", 0))
    if belonging > 0.3:
        experiences.append(ExperienceEvent(
            experience_type = ExperienceType.BELONGING_SIGNAL,
            content = (
                f"Belonging/connection signal detected. "
                f"Belonging score: {belonging:.3f}. "
                f"The person feels or yearns for connection to "
                f"humanity, home, or community."
            ),
            intensity  = round(belonging, 3),
            session_id = data.session_id,
            metadata   = {
                "emotion": data.amygdala_result.get("emotion"),
                "belonging_score": belonging
            }
        ))

    # ── Publish all high-intensity experiences to event bus ───────────────────
    for exp in experiences:
        if exp.intensity > 0.6:
            events.publish("experience.new", exp.dict())

    significance      = max((e.intensity for e in experiences), default=0.0)
    requires_learning = any(
        e.experience_type in [
            ExperienceType.SEMANTIC_NOVELTY,
            ExperienceType.KNOWLEDGE_BOUNDARY,
            ExperienceType.SKILL_GAP,
            ExperienceType.ANALOGICAL_INSIGHT
        ]
        for e in experiences
    )

    return BrainResponse(
        request_id = data.request_id,
        service    = "perception",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "experiences":       [e.dict() for e in experiences],
            "novelty_score":     round(novelty_score, 3),
            "significance":      round(significance, 3),
            "requires_learning": requires_learning,
            "experience_count":  len(experiences),
            "types_detected":    [e.experience_type for e in experiences]
        }
    )
