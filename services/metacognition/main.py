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
