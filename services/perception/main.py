from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import httpx, numpy as np, json, time, os, sys
sys.path.append("/shared")
from models import BrainRequest, BrainResponse, ExperienceEvent, PerceptionResult
from knobs import KnobsClient
from events import EventBus
from .models import KnobSet

app    = FastAPI(title="Perception", version="1.0.0")
knobs  = KnobsClient(os.getenv("REDIS_URL"))
events = EventBus(os.getenv("REDIS_URL"))
http   = httpx.AsyncClient(timeout=30.0)

NEOCORTEX_URL = os.getenv("NEOCORTEX_URL", "http://neocortex:8004")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")

class PerceptionInput(BaseModel):
    request_id:       str
    session_id:       str
    user_input:       str
    amygdala_result:  dict
    neocortex_results: list
    working_memory:   dict
    response_draft:   str = ""

async def embed(text: str) -> list:
    resp = await http.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return resp.json()["embedding"]

async def cosine_sim(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))

@app.get("/health")
async def health():
    return {"status": "alive", "service": "perception"}

@app.post("/perceive", response_model=BrainResponse)
async def perceive(data: PerceptionInput,
                   background: BackgroundTasks):
    """
    The perception layer runs on every interaction.
    It detects which of the system's 'senses' have been activated.
    It converts those activations into experience events.
    """
    start = time.time()
    k = knobs.get()
    
    experiences = []
    novelty_score = 0.0
    
    # ─── SENSE 1: SEMANTIC NOVELTY ───────────────────────────────────────
    # Is this input semantically far from anything in the neocortex?
    if data.neocortex_results:
        best_score = max(
            r.get("score", 0) for r in data.neocortex_results
        )
        # Low best score = high novelty
        novelty = 1.0 - min(1.0, best_score)
        novelty_score = max(novelty_score, novelty)
        
        if novelty > k.novelty_sensitivity:
            experiences.append(ExperienceEvent(
                experience_type = "SEMANTIC_NOVELTY",
                content = (
                    f"Input '{data.user_input[:80]}' is semantically distant "
                    f"from all known knowledge (novelty={novelty:.2f})"
                ),
                intensity   = novelty,
                session_id  = data.session_id,
                metadata    = {"best_match_score": best_score}
            ))
    
    # ─── SENSE 2: KNOWLEDGE BOUNDARY ─────────────────────────────────────
    # Is the system being asked something at the edge of what it knows?
    if len(data.neocortex_results) == 0 and data.amygdala_result.get("complexity") != "low":
        experiences.append(ExperienceEvent(
            experience_type = "KNOWLEDGE_BOUNDARY",
            content = (
                f"No relevant knowledge found for: '{data.user_input[:80]}'. "
                f"System is at the boundary of its knowledge."
            ),
            intensity   = 0.8,
            session_id  = data.session_id,
            metadata    = {"domain": data.amygdala_result.get("required_regions", [])}
        ))
        novelty_score = max(novelty_score, 0.8)
    
    # ─── SENSE 3: RESONANCE EVENT (Social/Emotional) ──────────────────────
    resonance = data.amygdala_result.get("resonance_score", 0)
    if resonance > 0.5:
        experiences.append(ExperienceEvent(
            experience_type = "RESONANCE_EVENT",
            content = (
                f"Strong human need detected: {data.amygdala_result.get('emotion')}. "
                f"Resonance={resonance:.2f}"
            ),
            intensity   = resonance,
            session_id  = data.session_id,
            metadata    = {
                "emotion":  data.amygdala_result.get("emotion"),
                "urgency":  data.amygdala_result.get("urgency")
            }
        ))
    
    # ─── SENSE 4: INTENT AMBIGUITY ───────────────────────────────────────
    # Multiple very different domain signals = ambiguous intent
    regions = data.amygdala_result.get("required_regions", [])
    if len(regions) >= 3:
        experiences.append(ExperienceEvent(
            experience_type = "INTENT_AMBIGUITY",
            content = (
                f"Input activates {len(regions)} different domains: "
                f"{regions}. Intent is ambiguous."
            ),
            intensity   = min(1.0, len(regions) * 0.2),
            session_id  = data.session_id,
            metadata    = {"regions": regions}
        ))
    
    # ─── SENSE 5: THREAT PATTERN (Anomaly detection / Smell) ──────────────
    if data.amygdala_result.get("threat_detected"):
        experiences.append(ExperienceEvent(
            experience_type = "THREAT_PATTERN",
            content = (
                f"Potential manipulation or jailbreak pattern detected "
                f"in: '{data.user_input[:80]}'"
            ),
            intensity   = 0.9,
            session_id  = data.session_id
        ))
        novelty_score = max(novelty_score, 0.9)
    
    # ─── SENSE 6: PROPRIOCEPTION — System Self-Sensing ───────────────────
    # Working memory pressure
    wm_keys = len(data.working_memory)
    if wm_keys > 15:
        experiences.append(ExperienceEvent(
            experience_type = "SYSTEM_STRESS",
            content = (
                f"Working memory at high pressure: {wm_keys} active items. "
                f"Consolidation may be needed."
            ),
            intensity   = min(1.0, wm_keys / 20),
            session_id  = data.session_id,
            metadata    = {"wm_item_count": wm_keys}
        ))
    
    # ─── SENSE 7: COMPLEXITY OVERFLOW ────────────────────────────────────
    word_count = len(data.user_input.split())
    if word_count > 200:
        experiences.append(ExperienceEvent(
            experience_type = "COMPLEXITY_OVERFLOW",
            content = (
                f"Input complexity ({word_count} words) exceeds standard "
                f"processing capacity. Multi-agent framework needed."
            ),
            intensity   = min(1.0, word_count / 300),
            session_id  = data.session_id,
            metadata    = {"word_count": word_count}
        ))
    
    # ─── SENSE 8: SELF-CONTRADICTION CHECK ────────────────────────────────
    # Did the draft response contradict working memory?
    if data.response_draft and data.working_memory.get("last_response"):
        last = data.working_memory["last_response"].get("content", "")
        # Simple heuristic: check for negation patterns
        draft_lower = data.response_draft.lower()
        last_lower  = last.lower()
        
        contradiction_signals = [
            ("is ", "is not "), ("can ", "cannot "), ("will ", "will not "),
            ("always ", "never "), ("yes", "no")
        ]
        
        for pos, neg in contradiction_signals:
            if pos in last_lower and neg in draft_lower:
                experiences.append(ExperienceEvent(
                    experience_type = "SELF_CONTRADICTION",
                    content = (
                        f"Draft response may contradict previous response. "
                        f"Pattern: '{pos}' vs '{neg}'"
                    ),
                    intensity   = 0.7,
                    session_id  = data.session_id
                ))
                break
    
    # ─── SENSE 9: PARAMETER DRIFT ────────────────────────────────────────
    # Are the current knobs significantly different from defaults?
    default_k = KnobSet()
    current_k = k
    drift_fields = ["humor", "honesty", "empathy", "formality"]
    total_drift = sum(
        abs(getattr(current_k, f) - getattr(default_k, f))
        for f in drift_fields
    )
    avg_drift = total_drift / len(drift_fields)
    
    if avg_drift > 0.3:
        experiences.append(ExperienceEvent(
            experience_type = "PARAMETER_DRIFT",
            content = (
                f"Behavioral parameters have drifted significantly from "
                f"defaults (avg drift={avg_drift:.2f}). "
                f"System personality is customized."
            ),
            intensity   = avg_drift,
            session_id  = data.session_id,
            metadata    = {"avg_drift": avg_drift}
        ))
    
    # Publish all experience events to the bus
    for exp in experiences:
        events.publish("experience.new", exp.dict())
    
    # Significance = highest single intensity
    significance = max((e.intensity for e in experiences), default=0.0)
    requires_learning = any(
        e.experience_type in [
            "SEMANTIC_NOVELTY", "KNOWLEDGE_BOUNDARY",
            "SKILL_GAP", "ANALOGICAL_INSIGHT"
        ]
        for e in experiences
    )
    
    return BrainResponse(
        request_id = data.request_id,
        service    = "perception",
        latency_ms = (time.time() - start) * 1000,
        data       = {
            "experiences":      [e.dict() for e in experiences],
            "novelty_score":    round(novelty_score, 3),
            "significance":     round(significance, 3),
            "requires_learning": requires_learning,
            "experience_count": len(experiences)
        }
    )