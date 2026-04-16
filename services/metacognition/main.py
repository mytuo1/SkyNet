# services/metacognition/main.py
# ── Metacognition: Knowing What You Know ─────────────────────────────────────
# Tracks calibrated confidence across domains.
# Detects when the system is operating outside its competence.
# Builds a calibration database so "80% confident" actually means correct 80%.
# Called by thalamus before and after Claude to assess uncertainty.

import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/shared")

import redis
import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus
from llm_provider import complete_json

app    = FastAPI(title="Metacognition", version="1.0.0",
                 description="Calibrated confidence and uncertainty awareness")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)
http   = httpx.AsyncClient(timeout=30.0)

CALIBRATION_KEY = "meta:calibration:{domain}"
UNCERTAINTY_LOG = "meta:uncertainty:log"


class AssessRequest(BaseModel):
    request_id:        str  = "meta"
    session_id:        str
    user_input:        str
    domain:            str  = "general"
    response_draft:    str  = ""
    neocortex_results: list = []
    experience_count:  int  = 0


class FeedbackRequest(BaseModel):
    domain:                  str
    predicted_confidence:    float
    was_correct:             bool
    session_id:              str = ""


@app.get("/health")
async def health():
    domains = ["technology", "science", "general", "creative", "emotional"]
    calibrated = sum(
        1 for d in domains
        if r.exists(CALIBRATION_KEY.format(domain=d))
    )
    return {
        "status":            "alive",
        "service":           "metacognition",
        "calibrated_domains": calibrated
    }


def get_domain_calibration(domain: str) -> dict:
    raw = r.get(CALIBRATION_KEY.format(domain=domain))
    if raw:
        return json.loads(raw)
    return {"total": 0, "correct": 0, "avg_predicted_confidence": 0.7}


@app.post("/assess", response_model=BrainResponse)
async def assess_uncertainty(req: AssessRequest):
    """
    Multi-signal uncertainty assessment.
    Combines knowledge density, domain calibration, and self-assessment.
    Returns a calibrated confidence score and actionable uncertainty flags.
    """
    start = time.time()

    # ── Signal 1: Knowledge density (neocortex coverage) ─────────────────────
    if req.neocortex_results:
        scores      = [float(r_item.get("score", 0)) for r_item in req.neocortex_results]
        avg_score   = np.mean(scores) if scores else 0.0
        top_score   = max(scores) if scores else 0.0
        knowledge_conf = min(1.0, (avg_score * 0.6 + top_score * 0.4) * 1.2)
    else:
        knowledge_conf = 0.15   # very low — nothing in neocortex matches

    # ── Signal 2: Domain calibration history ─────────────────────────────────
    cal              = get_domain_calibration(req.domain)
    total            = cal.get("total", 0)
    correct          = cal.get("correct", 0)
    domain_accuracy  = correct / max(1, total)
    # Smooth toward 0.68 prior if we have few examples
    weight           = min(1.0, total / 25)
    domain_conf      = domain_accuracy * weight + 0.68 * (1 - weight)

    # ── Signal 3: Self-assessment of response draft ───────────────────────────
    draft_conf          = 0.70   # default prior
    uncertainty_sources = []
    knowledge_gaps      = []
    should_flag         = False

    if req.response_draft and len(req.response_draft) > 50:
        result = await complete_json(
            prompt=f"""Rate your own confidence in this response.
Be honest — overconfident answers cause harm.

QUESTION: {req.user_input[:200]}
RESPONSE: {req.response_draft[:500]}

Respond with JSON:
{{
    "confidence": 0.0 to 1.0,
    "uncertainty_sources": ["what you are uncertain about"],
    "knowledge_gaps": ["what you don't know that's relevant"],
    "should_flag_to_user": true or false,
    "reasoning": "one sentence"
}}""",
            system="Assess confidence honestly. Respond only with valid JSON."
        )
        if result:
            draft_conf          = float(result.get("confidence", 0.70))
            uncertainty_sources = result.get("uncertainty_sources", [])
            knowledge_gaps      = result.get("knowledge_gaps", [])
            should_flag         = result.get("should_flag_to_user", False)

    # ── Combine with calibration correction ──────────────────────────────────
    combined = (
        draft_conf    * 0.50 +
        domain_conf   * 0.30 +
        knowledge_conf * 0.20
    )

    # Apply calibration ratio if we have enough history
    if total >= 15:
        avg_pred       = cal.get("avg_predicted_confidence", 0.70)
        actual_acc     = domain_accuracy
        cal_ratio      = actual_acc / max(0.01, avg_pred)
        combined      *= min(1.5, max(0.4, cal_ratio))

    combined = round(min(1.0, max(0.0, combined)), 3)

    # ── Uncertainty level ─────────────────────────────────────────────────────
    if combined > 0.82:
        level = "low"
    elif combined > 0.60:
        level = "medium"
    elif combined > 0.38:
        level = "high"
    else:
        level = "very_high"

    # Publish knowledge boundary experience if very uncertain
    if combined < 0.35:
        events.publish("experience.new", {
            "experience_type": "KNOWLEDGE_BOUNDARY",
            "content": f"Very high uncertainty ({combined:.2f}) for: '{req.user_input[:80]}'",
            "intensity": 1.0 - combined,
            "session_id": req.session_id
        })

    # Log for calibration tracking
    r.lpush(UNCERTAINTY_LOG, json.dumps({
        "domain":     req.domain,
        "confidence": combined,
        "level":      level,
        "timestamp":  time.time()
    }))
    r.ltrim(UNCERTAINTY_LOG, 0, 999)

    return BrainResponse(
        request_id = req.request_id,
        service    = "metacognition",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "confidence":           combined,
            "uncertainty_level":    level,
            "should_flag":          should_flag or combined < 0.38,
            "knowledge_confidence": round(knowledge_conf, 3),
            "domain_confidence":    round(domain_conf, 3),
            "draft_confidence":     round(draft_conf, 3),
            "uncertainty_sources":  uncertainty_sources[:5],
            "knowledge_gaps":       knowledge_gaps[:5],
            "domain":               req.domain,
            "calibration_samples":  total
        }
    )


@app.post("/feedback")
async def record_outcome(req: FeedbackRequest):
    """
    Record whether the system was actually correct.
    Builds the calibration database over time.
    Essential for the confidence scores to mean anything real.
    """
    key = CALIBRATION_KEY.format(domain=req.domain)
    cal = get_domain_calibration(req.domain)

    cal["total"]   += 1
    cal["correct"] += 1 if req.was_correct else 0

    n = cal["total"]
    cal["avg_predicted_confidence"] = (
        cal.get("avg_predicted_confidence", 0.70) * (n - 1) +
        req.predicted_confidence
    ) / n

    r.set(key, json.dumps(cal))

    accuracy = cal["correct"] / cal["total"]
    return {
        "recorded": True,
        "domain":   req.domain,
        "total":    cal["total"],
        "accuracy": round(accuracy, 3),
        "calibration_error": round(
            abs(accuracy - cal["avg_predicted_confidence"]), 3
        )
    }


@app.get("/calibration/{domain}")
async def get_calibration(domain: str):
    cal = get_domain_calibration(domain)
    if cal["total"] == 0:
        return {"domain": domain, "status": "no_data"}
    accuracy = cal["correct"] / cal["total"]
    return {
        "domain":    domain,
        "accuracy":  round(accuracy, 3),
        "total":     cal["total"],
        "avg_predicted": cal.get("avg_predicted_confidence", 0.70),
        "calibration_error": round(
            abs(accuracy - cal.get("avg_predicted_confidence", 0.70)), 3
        )
    }


@app.get("/calibration")
async def all_calibrations():
    keys    = r.keys("meta:calibration:*")
    results = {}
    for key in keys:
        domain = key.replace("meta:calibration:", "")
        cal    = json.loads(r.get(key))
        if cal["total"] > 0:
            results[domain] = {
                "accuracy": round(cal["correct"] / cal["total"], 3),
                "total":    cal["total"]
            }
    return results


# ── Intuition (Sixth Sense) ──────────────────────────────────────────────────
# Intuition = unconscious pattern recognition from accumulated experience
# + somatic markers (emotional tags on past outcomes).
# Antonio Damasio's Somatic Marker Hypothesis: the brain assigns emotional
# "gut feelings" to decisions based on prior outcomes.
#
# Key ingredients:
#   1. Large accumulated experience in a domain (hippocampus + neocortex)
#   2. Pattern matching below conscious threshold (cross-domain + cerebellum)
#   3. Emotional tagging of past outcomes (amygdala feedback stored w/ episodes)
#   4. A "confidence feeling" that precedes conscious reasoning
#
# The accuracy of intuition improves over time as the system accumulates
# more episodes and tracks whether past intuitions were correct.

INTUITION_TRACK_KEY = "meta:intuition:{domain}"
HIPPOCAMPUS_URL     = os.getenv("HIPPOCAMPUS_URL",  "http://hippocampus:8002")
NEOCORTEX_URL       = os.getenv("NEOCORTEX_URL",    "http://neocortex:8004")
CROSS_DOMAIN_URL    = os.getenv("CROSS_DOMAIN_URL", "http://cross-domain:8008")
CEREBELLUM_URL      = os.getenv("CEREBELLUM_URL",   "http://cerebellum:8003")


class IntuitionRequest(BaseModel):
    request_id:    str = "intuition"
    session_id:    str
    user_input:    str
    domain:        str = "general"


@app.post("/intuition", response_model=BrainResponse)
async def assess_intuition(req: IntuitionRequest):
    """
    Sixth Sense — assess whether the system has an intuitive signal
    about this input based on accumulated experience and past accuracy.

    Returns an intuition assessment: direction, confidence, basis, and
    a somatic marker describing the "gut feeling."

    This is pattern recognition below the conscious reasoning threshold.
    """
    start = time.time()

    # ── Signal 1: Experience depth (hippocampus episodes in this domain) ──
    episode_count = 0
    try:
        hipp_resp = await http.post(f"{HIPPOCAMPUS_URL}/recall", json={
            "request_id": req.request_id,
            "session_id": req.session_id,
            "user_input": req.user_input,
        }, timeout=10.0)
        if hipp_resp.status_code == 200:
            episodes = hipp_resp.json().get("data", {}).get("episodes", [])
            episode_count = len(episodes)
    except Exception:
        episodes = []

    # ── Signal 2: Knowledge breadth (neocortex hits for this domain) ──────
    knowledge_count = 0
    top_score       = 0.0
    try:
        neo_resp = await http.post(f"{NEOCORTEX_URL}/search", json={
            "request_id": req.request_id,
            "session_id": req.session_id,
            "user_input": req.user_input,
            "domain":     req.domain,
        }, timeout=10.0)
        if neo_resp.status_code == 200:
            results = neo_resp.json().get("data", {}).get("results", [])
            knowledge_count = len(results)
            if results:
                top_score = max(float(item.get("score", 0)) for item in results)
    except Exception:
        pass

    # ── Signal 3: Domain calibration accuracy (past prediction track record) ─
    cal            = get_domain_calibration(req.domain)
    total_samples  = cal.get("total", 0)
    past_accuracy  = cal["correct"] / max(1, total_samples) if total_samples > 0 else 0.5

    # ── Signal 4: Intuition-specific track record ────────────────────────
    intuition_data = r.get(INTUITION_TRACK_KEY.format(domain=req.domain))
    if intuition_data:
        idata = json.loads(intuition_data)
        intuition_total   = idata.get("total", 0)
        intuition_correct = idata.get("correct", 0)
        intuition_accuracy = intuition_correct / max(1, intuition_total)
    else:
        intuition_total    = 0
        intuition_correct  = 0
        intuition_accuracy = 0.5  # prior

    # ── Compute intuition strength ────────────────────────────────────────
    # More episodes + more knowledge + higher past accuracy = stronger intuition
    experience_factor  = min(1.0, episode_count / 20.0)         # caps at 20 episodes
    knowledge_factor   = min(1.0, knowledge_count / 15.0)       # caps at 15 items
    accuracy_factor    = past_accuracy
    relevance_factor   = min(1.0, top_score * 1.5) if top_score > 0 else 0.0

    intuition_strength = (
        experience_factor  * 0.30 +
        knowledge_factor   * 0.25 +
        accuracy_factor    * 0.25 +
        relevance_factor   * 0.20
    )

    # Threshold: intuition fires if strength > 0.45
    has_intuition = intuition_strength > 0.45

    # Direction: based on whether past experiences in this domain were positive
    positive_episodes = sum(
        1 for ep in (episodes if episode_count > 0 else [])
        if ep.get("emotion") in ("joy", "gratitude", "admiration", "relief",
                                  "optimism", "pride", "approval", "curiosity")
    )
    negative_episodes = sum(
        1 for ep in (episodes if episode_count > 0 else [])
        if ep.get("emotion") in ("anger", "fear", "sadness", "grief",
                                  "disappointment", "disgust", "remorse")
    )

    if positive_episodes > negative_episodes:
        direction = "positive"
    elif negative_episodes > positive_episodes:
        direction = "negative"
    else:
        direction = "caution"

    # Somatic marker: what does this feel like based on past experience
    if has_intuition and episodes:
        most_similar = episodes[0] if episodes else {}
        somatic_marker = (
            f"This feels similar to past experience: "
            f"'{most_similar.get('summary', 'a previous interaction')[:100]}' "
            f"which was associated with {most_similar.get('emotion', 'a neutral')} feeling."
        )
    else:
        somatic_marker = ""

    # Confidence in the intuition itself
    intuition_confidence = round(min(1.0, intuition_strength * 1.2), 3)

    basis = (
        f"{episode_count} episodes in this domain, "
        f"{knowledge_count} knowledge items, "
        f"{past_accuracy:.0%} past domain accuracy"
    )
    if intuition_total > 0:
        basis += f", {intuition_accuracy:.0%} intuition track record ({intuition_total} signals)"

    # Publish intuition as experience if it fires
    if has_intuition:
        events.publish("intuition.fired", {
            "experience_type": "INTUITION_SIGNAL",
            "content":         f"Intuition signal ({direction}) in domain '{req.domain}' "
                               f"with confidence {intuition_confidence:.0%}. {basis}",
            "intensity":       intuition_confidence,
            "session_id":      req.session_id,
            "timestamp":       time.time(),
            "metadata": {
                "domain":     req.domain,
                "direction":  direction,
                "confidence": intuition_confidence,
                "basis":      basis,
            }
        })

    return BrainResponse(
        request_id = req.request_id,
        service    = "metacognition",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "has_intuition":       has_intuition,
            "intuition_strength":  round(intuition_strength, 3),
            "intuition_direction": direction,
            "intuition_confidence": intuition_confidence,
            "basis":               basis,
            "somatic_marker":      somatic_marker,
            "experience_depth":    episode_count,
            "knowledge_breadth":   knowledge_count,
            "domain_accuracy":     round(past_accuracy, 3),
            "intuition_accuracy":  round(intuition_accuracy, 3),
        }
    )


class IntuitionFeedbackRequest(BaseModel):
    domain:        str
    was_correct:   bool
    session_id:    str = ""


@app.post("/intuition/feedback")
async def record_intuition_outcome(req: IntuitionFeedbackRequest):
    """
    Record whether an intuitive signal was correct.
    Builds the intuition accuracy database over time.
    The system learns which domains its gut feelings are reliable in.
    """
    key = INTUITION_TRACK_KEY.format(domain=req.domain)
    raw = r.get(key)
    if raw:
        data = json.loads(raw)
    else:
        data = {"total": 0, "correct": 0}

    data["total"]   += 1
    data["correct"] += 1 if req.was_correct else 0
    r.set(key, json.dumps(data))

    accuracy = data["correct"] / data["total"]
    return {
        "recorded": True,
        "domain":   req.domain,
        "intuition_accuracy": round(accuracy, 3),
        "total_signals":      data["total"],
    }
