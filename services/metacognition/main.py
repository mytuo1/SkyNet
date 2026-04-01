# ── Metacognition: Knowing What You Know ──────────────────────────────────────
# Tracks confidence across domains over time.
# Detects when the system is in unfamiliar territory.
# Enables calibrated uncertainty: "I'm 80% confident" actually means
# the system is right 80% of the time when it says that.

import json
import os
import sys
import time

sys.path.insert(0, "/shared")

import redis
import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus

app    = FastAPI(title="Metacognition", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
r      = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
http   = httpx.AsyncClient(timeout=30.0)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = os.getenv("FAST_CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Calibration tracking
# For each domain, track: predicted_confidence vs actual_accuracy
# This lets us detect overconfidence and underconfidence
CALIBRATION_KEY = "meta:calibration:{domain}"
UNCERTAINTY_KEY = "meta:uncertainty:current"


class AssessRequest(BaseModel):
    request_id:  str = "meta"
    session_id:  str
    user_input:  str
    domain:      str = "general"
    response:    str = ""    # if we have a draft, assess its confidence
    neocortex_results: list = []


class FeedbackRequest(BaseModel):
    domain:           str
    predicted_confidence: float
    was_correct:      bool


@app.get("/health")
async def health():
    return {"status": "alive", "service": "metacognition"}


@app.post("/assess", response_model=BrainResponse)
async def assess_uncertainty(req: AssessRequest):
    """
    Assess how confident the system should be about this query.
    
    Three signals:
    1. Knowledge density — how much do we know about this? (neocortex results)
    2. Domain calibration — how accurate have we been in this domain before?
    3. Response self-assessment — does Claude flag uncertainty in its own draft?
    """
    start = time.time()
    
    # ── Signal 1: Knowledge density ───────────────────────────────────────────
    if req.neocortex_results:
        scores = [float(r.get("score", 0)) for r in req.neocortex_results]
        avg_relevance = np.mean(scores) if scores else 0.0
        knowledge_confidence = min(1.0, avg_relevance * 1.2)
    else:
        knowledge_confidence = 0.2   # very low — no relevant knowledge
    
    # ── Signal 2: Domain calibration ──────────────────────────────────────────
    cal_raw = r.get(CALIBRATION_KEY.format(domain=req.domain))
    if cal_raw:
        cal = json.loads(cal_raw)
        total     = cal.get("total", 0)
        correct   = cal.get("correct", 0)
        domain_accuracy = correct / max(1, total)
        # Smooth toward 0.7 if we have few examples
        weight  = min(1.0, total / 20)
        domain_confidence = domain_accuracy * weight + 0.7 * (1 - weight)
    else:
        domain_confidence = 0.65   # prior: moderately confident in unknown domain
    
    # ── Signal 3: Self-assessment of draft ────────────────────────────────────
    draft_confidence = 0.7   # default
    
    if req.response and ANTHROPIC_KEY:
        prompt = f"""Rate your own confidence in this response on a scale of 0.0 to 1.0.
Consider: factual accuracy, completeness, potential for hallucination.

QUESTION: {req.user_input[:200]}
RESPONSE: {req.response[:400]}

Respond with JSON only:
{{
    "confidence": 0.0 to 1.0,
    "uncertainty_sources": ["list what you're uncertain about"],
    "knowledge_gaps": ["what you don't know"],
    "should_flag": true/false
}}"""
        
        try:
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json"
                },
                json={
                    "model":      CLAUDE_MODEL,
                    "max_tokens": 250,
                    "messages":   [{"role": "user", "content": prompt}]
                }
            )
            text = resp.json()["content"][0]["text"].strip()
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            assessment     = json.loads(text)
            draft_confidence = float(assessment.get("confidence", 0.7))
            uncertainty_sources = assessment.get("uncertainty_sources", [])
            knowledge_gaps      = assessment.get("knowledge_gaps", [])
            should_flag         = assessment.get("should_flag", False)
        except Exception:
            uncertainty_sources = []
            knowledge_gaps      = []
            should_flag         = False
    else:
        uncertainty_sources = []
        knowledge_gaps      = []
        should_flag         = False
    
    # ── Combine signals ───────────────────────────────────────────────────────
    # Weights: draft self-assessment most informative, then domain, then knowledge
    combined_confidence = (
        draft_confidence    * 0.50 +
        domain_confidence   * 0.30 +
        knowledge_confidence * 0.20
    )
    
    # Calibration correction: if we know we're overconfident in this domain,
    # scale down
    if cal_raw:
        cal = json.loads(cal_raw)
        if cal.get("total", 0) > 10:
            avg_predicted = cal.get("avg_predicted_confidence", 0.7)
            actual_acc    = cal.get("correct", 0) / max(1, cal.get("total", 1))
            calibration_ratio = actual_acc / max(0.01, avg_predicted)
            combined_confidence *= min(1.5, max(0.5, calibration_ratio))
    
    combined_confidence = round(min(1.0, max(0.0, combined_confidence)), 3)
    
    # Determine uncertainty level
    if combined_confidence > 0.80:
        uncertainty_level = "low"
    elif combined_confidence > 0.55:
        uncertainty_level = "medium"
    elif combined_confidence > 0.35:
        uncertainty_level = "high"
    else:
        uncertainty_level = "very_high"
    
    # Publish experience if very uncertain
    if combined_confidence < 0.35:
        events.publish("experience.new", {
            "experience_type": "KNOWLEDGE_BOUNDARY",
            "content": f"Very high uncertainty ({combined_confidence:.2f}) on: {req.user_input[:80]}",
            "intensity": 1.0 - combined_confidence,
            "session_id": req.session_id
        })
    
    return BrainResponse(
        request_id = req.request_id,
        service    = "metacognition",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "confidence":           combined_confidence,
            "uncertainty_level":    uncertainty_level,
            "should_flag":          should_flag or combined_confidence < 0.4,
            "knowledge_confidence": knowledge_confidence,
            "domain_confidence":    domain_confidence,
            "draft_confidence":     draft_confidence,
            "uncertainty_sources":  uncertainty_sources,
            "knowledge_gaps":       knowledge_gaps,
            "domain":               req.domain
        }
    )


@app.post("/feedback")
async def record_outcome(req: FeedbackRequest):
    """
    Record whether the system was actually right.
    This builds the calibration database over time.
    As this grows, confidence estimates become more accurate.
    """
    key    = CALIBRATION_KEY.format(domain=req.domain)
    cal_raw = r.get(key)
    
    if cal_raw:
        cal = json.loads(cal_raw)
    else:
        cal = {"total": 0, "correct": 0, "avg_predicted_confidence": 0.0}
    
    cal["total"]   += 1
    cal["correct"] += 1 if req.was_correct else 0
    
    # Running average of predicted confidence
    n = cal["total"]
    cal["avg_predicted_confidence"] = (
        cal["avg_predicted_confidence"] * (n - 1) + req.predicted_confidence
    ) / n
    
    r.set(key, json.dumps(cal))
    
    accuracy = cal["correct"] / cal["total"]
    return {
        "recorded": True,
        "domain":   req.domain,
        "accuracy": round(accuracy, 3),
        "total":    cal["total"]
    }
