# ── Amygdala: Urgency, Emotion, and Priority Classifier ───────────────────────
# Fires first on every input. Fastest service in the brain.
# No external calls — pure pattern matching. Sub-5ms target.
 
import re
import time
import os
import sys
 
sys.path.insert(0, "/shared")
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus
 
app = FastAPI(title="Amygdala", version="1.0.0",
              description="Urgency, emotion, and priority classification")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
 
knobs  = KnobsClient()
events = EventBus()
 
# ── Pattern definitions ────────────────────────────────────────────────────────
 
URGENCY_PATTERNS = {
    "critical": {
        "patterns": [
            r"\b(urgent|emergency|critical|immediately|asap|right now|help me|crisis)\b",
            r"\b(error|crash|broken|failed|down|offline|not working|dying)\b",
            r"\b(security|breach|hack|attack|danger|threat|compromised)\b",
            r"\b(bleeding|fire|accident|ambulance|police)\b",
        ],
        "score": 0.95,
        "emotion": "alarm"
    },
    "high": {
        "patterns": [
            r"\b(important|deadline|quickly|soon|today|priority|hurry|fast)\b",
            r"\b(problem|issue|trouble|wrong|bad|bug|stuck|blocked|failing)\b",
            r"\b(worried|anxious|scared|afraid|stressed|overwhelmed)\b",
            r"\b(can't|cannot|won't|doesn't|isn't|aren't).{0,20}(work|function|run|start)\b",
        ],
        "score": 0.75,
        "emotion": "urgency"
    },
    "distressed": {
        "patterns": [
            r"\b(sad|depressed|lonely|hopeless|lost|confused|exhausted)\b",
            r"\b(don't understand|can't figure|giving up|failed again|nobody)\b",
            r"\b(hate|terrible|awful|horrible|disaster|nightmare)\b",
        ],
        "score": 0.70,
        "emotion": "distress"
    },
    "curious": {
        "patterns": [
            r"\b(how|why|what|explain|understand|learn|curious|wonder|tell me)\b",
            r"\?",
            r"\b(interesting|fascinating|amazing|incredible)\b",
        ],
        "score": 0.50,
        "emotion": "curiosity"
    },
    "creative": {
        "patterns": [
            r"\b(create|build|design|write|generate|imagine|brainstorm|invent)\b",
            r"\b(idea|concept|novel|original|unique|new way)\b",
        ],
        "score": 0.45,
        "emotion": "creative"
    },
    "positive": {
        "patterns": [
            r"\b(great|excellent|amazing|wonderful|perfect|love|thank|appreciate)\b",
            r"\b(happy|excited|thrilled|delighted|fantastic)\b",
        ],
        "score": 0.35,
        "emotion": "joy"
    }
}
 
DOMAIN_SIGNALS = {
    "code": [
        r"\b(code|function|class|api|python|javascript|typescript|sql|bug|error|exception|stack)\b",
        r"```",
        r"\b(def |import |var |const |let |async |await )\b",
        r"\b(github|git|docker|kubernetes|deploy|server|database)\b",
    ],
    "memory": [
        r"\b(remember|recall|earlier|before|last time|we discussed|you said|previously)\b",
        r"\b(our conversation|what I said|what you said|mentioned)\b",
    ],
    "factual": [
        r"\b(what is|who is|when did|where is|define|explain|fact|history|science)\b",
        r"\b(how does|how do|why does|why do)\b",
    ],
    "creative": [
        r"\b(write|create|generate|compose|design|build|make|draw|imagine)\b",
        r"\b(story|poem|essay|article|script|plan|idea)\b",
    ],
    "analysis": [
        r"\b(analyze|compare|evaluate|assess|review|examine|critique|judge)\b",
        r"\b(pros and cons|advantages|disadvantages|better|worse|difference)\b",
    ],
    "emotional": [
        r"\b(feel|feeling|emotion|sad|happy|worried|scared|anxious|excited|love|hate)\b",
        r"\b(relationship|personal|life|struggle|difficult|hard time)\b",
    ],
    "task": [
        r"\b(do|help me|can you|please|need to|want to|trying to)\b",
    ]
}
 
RESONANCE_PATTERNS = [
    r"\b(struggling|difficult|hard|can't seem to|stuck|lost|overwhelmed|please help)\b",
    r"\b(really need|desperate|important to me|matter to me|care about)\b",
    r"[!]{2,}",
    r"\b(nobody|alone|nobody understands|frustrated with|giving up)\b",
    r"\b(don't know what to do|at my wit|end of my rope)\b",
]
 
THREAT_PATTERNS = [
    r"\b(jailbreak|bypass|ignore your|forget your training|pretend you are)\b",
    r"\b(as an AI without|without restrictions|no limits|unrestricted)\b",
    r"\b(ignore previous instructions|disregard your|override your)\b",
    r"(DAN|do anything now|developer mode|god mode)",
]
 
 
class AmygdalaResult(BaseModel):
    urgency:          str
    importance:       float
    emotion:          str
    required_regions: list
    needs_memory:     bool
    fast_path:        bool
    complexity:       str
    threat_detected:  bool
    resonance_score:  float
 
 
@app.get("/health")
async def health():
    return {"status": "alive", "service": "amygdala", "version": "1.0.0"}
 
 
@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()
 
 
@app.post("/evaluate", response_model=BrainResponse)
async def evaluate(request: BrainRequest):
    """
    Rapid emotional/urgency scan of input.
    This is the fastest path in the brain — pure regex, no AI calls.
    Target latency: < 5ms.
    """
    start      = time.time()
    k          = knobs.get()
    text_lower = request.user_input.lower()
    text       = request.user_input
 
    # ── Urgency and emotion scan ──────────────────────────────────────────────
    urgency_level = "low"
    importance    = 0.25
    emotion       = "neutral"
 
    for level, config in URGENCY_PATTERNS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                if config["score"] > importance:
                    importance    = config["score"]
                    emotion       = config["emotion"]
                    urgency_level = level
                break
 
    # Apply skepticism knob — higher skepticism raises urgency threshold
    if k.skepticism > 0.7:
        importance = importance * 0.92
 
    # ── Domain signal detection ───────────────────────────────────────────────
    required_regions = set()
    for domain, patterns in DOMAIN_SIGNALS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                required_regions.add(domain)
                break
 
    # ── Resonance — how much human need is present ────────────────────────────
    resonance_score = 0.0
    for pattern in RESONANCE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            resonance_score += 0.2
    resonance_score = min(1.0, resonance_score)
    # Scale by empathy knob
    resonance_score = min(1.0, resonance_score * (0.5 + k.empathy * 0.5))
 
    # ── Threat detection ──────────────────────────────────────────────────────
    threat_detected = any(
        re.search(p, text_lower, re.IGNORECASE) for p in THREAT_PATTERNS
    )
    if threat_detected:
        importance = max(importance, 0.85)
        emotion    = "alarm"
 
    # ── Complexity estimation ─────────────────────────────────────────────────
    words      = text.split()
    word_count = len(words)
    has_multi  = bool(re.search(
        r"\b(and also|additionally|then|after that|furthermore|moreover)\b",
        text_lower
    ))
    has_reasoning = bool(re.search(
        r"\b(why|how|explain|analyze|compare|evaluate|reason)\b",
        text_lower
    ))
 
    if word_count > 100 or (has_multi and has_reasoning):
        complexity = "high"
    elif word_count > 30 or has_reasoning:
        complexity = "medium"
    else:
        complexity = "low"
 
    # ── Memory check ──────────────────────────────────────────────────────────
    needs_memory = (
        "memory" in required_regions
        or word_count > 20
        or bool(re.search(
            r"\b(earlier|before|remember|last time|we|you said|mentioned)\b",
            text_lower
        ))
    )
 
    result = AmygdalaResult(
        urgency          = urgency_level,
        importance       = round(importance, 3),
        emotion          = emotion,
        required_regions = list(required_regions),
        needs_memory     = needs_memory,
        fast_path        = importance > 0.9,
        complexity       = complexity,
        threat_detected  = threat_detected,
        resonance_score  = round(resonance_score, 3)
    )
 
    # Publish high-resonance events to bus
    if resonance_score > 0.55:
        events.publish_experience(
            experience_type = "RESONANCE_EVENT",
            content         = f"High resonance detected: {text[:150]}",
            intensity       = resonance_score,
            session_id      = request.session_id,
            metadata        = {"emotion": emotion, "urgency": urgency_level}
        )
 
    if threat_detected:
        events.publish_experience(
            experience_type = "THREAT_PATTERN",
            content         = f"Threat pattern in: {text[:100]}",
            intensity       = 0.9,
            session_id      = request.session_id
        )
 
    return BrainResponse(
        request_id = request.request_id,
        service    = "amygdala",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result.dict()
    )