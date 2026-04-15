# services/goals/main.py
# ── Goals: Persistent Agency and Long-Horizon Planning ───────────────────────
# The system can pursue goals across sessions, across days, across weeks.
# Goals decompose into sub-goals, sub-goals into tasks.
# Autonomous agenda generation enables unprompted work toward objectives.
# This is what separates "assistant" from "agent."

import asyncio
import json
import os
import sys
import time
import uuid
from enum import Enum
from typing import Optional

sys.path.insert(0, "/shared")

import redis
import httpx
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus
from llm_provider import complete_json

app    = FastAPI(title="Goals", version="1.0.0",
                 description="Persistent goals, hierarchical planning, autonomous agenda")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)
http   = httpx.AsyncClient(timeout=60.0)

THALAMUS_URL = os.getenv("THALAMUS_URL", "http://thalamus:8000")

GOALS_KEY    = "goals:active"        # sorted set: goal_id → priority
GOAL_PREFIX  = "goals:data:"
PLAN_PREFIX  = "goals:plan:"
HISTORY_KEY  = "goals:completed"
PROGRESS_KEY = "goals:progress:"


class GoalStatus(str, Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"
    ABANDONED = "abandoned"


class CreateGoalRequest(BaseModel):
    title:       str
    description: str
    priority:    float = 0.5
    horizon:     str   = "week"    # hour | day | week | month | year
    domain:      str   = "general"
    session_id:  str   = ""
    deadline:    Optional[float] = None
    auto_plan:   bool  = True
    tags:        list  = []


class UpdateProgressRequest(BaseModel):
    goal_id:        str
    progress:       float
    note:           str = ""
    completed_task: str = ""


@app.get("/health")
async def health():
    return {
        "status":       "alive",
        "service":      "goals",
        "active_goals": r.zcard(GOALS_KEY)
    }


@app.post("/goals", response_model=BrainResponse)
async def create_goal(req: CreateGoalRequest,
                      background: BackgroundTasks):
    """
    Create a new persistent goal.
    Auto-decomposes into hierarchical sub-goals and tasks.
    The goal persists until completed, failed, or abandoned.
    """
    start   = time.time()
    goal_id = uuid.uuid4().hex[:12]

    goal = {
        "id":          goal_id,
        "title":       req.title,
        "description": req.description,
        "priority":    req.priority,
        "horizon":     req.horizon,
        "domain":      req.domain,
        "status":      GoalStatus.PENDING,
        "progress":    0.0,
        "sub_goals":   [],
        "created":     time.time(),
        "deadline":    req.deadline,
        "session_id":  req.session_id,
        "tags":        req.tags,
        "notes":       []
    }

    r.set(f"{GOAL_PREFIX}{goal_id}", json.dumps(goal))
    r.zadd(GOALS_KEY, {goal_id: req.priority})

    if req.auto_plan:
        background.add_task(_decompose_goal, goal_id, goal)

    events.publish("experience.new", {
        "experience_type": "TEMPORAL_PATTERN",
        "content": f"New goal created: '{req.title}' (horizon={req.horizon}, priority={req.priority})",
        "intensity": req.priority,
        "session_id": req.session_id
    })

    return BrainResponse(
        request_id = "goals",
        service    = "goals",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"goal_id": goal_id, "goal": goal}
    )


async def _decompose_goal(goal_id: str, goal: dict):
    """
    Use Claude to decompose a goal into a hierarchical action plan.
    Runs in background — doesn't block goal creation.
    """
    horizon_guidance = {
        "hour":  "Break into 3-5 tasks each taking 5-15 minutes",
        "day":   "Break into 4-6 sub-goals each taking 1-3 hours",
        "week":  "Break into 3-5 milestones each taking 1-2 days",
        "month": "Break into 4-6 phases each taking 1-2 weeks",
        "year":  "Break into 4-6 quarters each with clear deliverables"
    }

    guidance = horizon_guidance.get(goal["horizon"], horizon_guidance["week"])

    result = await complete_json(
        prompt=f"""Decompose this goal into a concrete, executable hierarchical plan.

GOAL: {goal['title']}
DESCRIPTION: {goal['description']}
HORIZON: {goal['horizon']}
DOMAIN: {goal['domain']}

STRUCTURE GUIDANCE: {guidance}

Requirements:
- Each task should be concrete and verifiable
- Mark tasks as can_automate=true if the system can do them without human input
- Sequence matters — order sub-goals by dependency
- Be realistic about time estimates

JSON response:
{{
    "sub_goals": [
        {{
            "id": "sg_1",
            "title": "milestone title",
            "description": "what this achieves",
            "sequence": 1,
            "estimated_hours": 2,
            "tasks": [
                {{
                    "id": "t_1",
                    "description": "specific concrete action",
                    "estimated_minutes": 30,
                    "can_automate": false,
                    "depends_on": []
                }}
            ]
        }}
    ],
    "critical_path": ["sg_1", "sg_2"],
    "estimated_total_hours": 10,
    "success_criteria": "how to know the goal is complete",
    "risks": ["potential blockers"]
}}""",
        system="Create actionable, realistic plans. Respond only with valid JSON.",
        model_tier="main"
    )

    if result and result.get("sub_goals"):
        r.set(f"{PLAN_PREFIX}{goal_id}", json.dumps(result))

        # Update goal status to active
        goal_raw = r.get(f"{GOAL_PREFIX}{goal_id}")
        if goal_raw:
            g = json.loads(goal_raw)
            g["status"]    = GoalStatus.ACTIVE
            g["sub_goals"] = [sg["id"] for sg in result["sub_goals"]]
            r.set(f"{GOAL_PREFIX}{goal_id}", json.dumps(g))

        print(f"Goal {goal_id} decomposed: "
              f"{len(result['sub_goals'])} sub-goals, "
              f"~{result.get('estimated_total_hours', '?')}h total")
    else:
        # Mark as active even without plan
        goal_raw = r.get(f"{GOAL_PREFIX}{goal_id}")
        if goal_raw:
            g = json.loads(goal_raw)
            g["status"] = GoalStatus.ACTIVE
            r.set(f"{GOAL_PREFIX}{goal_id}", json.dumps(g))


@app.get("/goals")
async def list_goals(status: str = None, limit: int = 20):
    """List all goals, optionally filtered by status."""
    goal_ids = r.zrevrange(GOALS_KEY, 0, -1)
    goals    = []

    for gid in goal_ids[:limit]:
        raw = r.get(f"{GOAL_PREFIX}{gid}")
        if raw:
            g = json.loads(raw)
            if status is None or g.get("status") == status:
                goals.append(g)

    return {"goals": goals, "total": len(goals)}


@app.get("/goals/{goal_id}")
async def get_goal(goal_id: str):
    raw = r.get(f"{GOAL_PREFIX}{goal_id}")
    if not raw:
        return {"error": "Goal not found"}
    goal     = json.loads(raw)
    plan_raw = r.get(f"{PLAN_PREFIX}{goal_id}")
    if plan_raw:
        goal["plan"] = json.loads(plan_raw)
    return goal


@app.patch("/goals/{goal_id}/progress")
async def update_progress(goal_id: str, req: UpdateProgressRequest):
    """Update goal progress. Triggers completion if progress >= 1.0."""
    raw = r.get(f"{GOAL_PREFIX}{goal_id}")
    if not raw:
        return {"error": "Goal not found"}

    goal             = json.loads(raw)
    old_progress     = goal.get("progress", 0.0)
    goal["progress"] = min(1.0, max(0.0, req.progress))

    if req.note:
        goal.setdefault("notes", []).append({
            "note": req.note, "timestamp": time.time(),
            "progress": req.progress
        })

    if goal["progress"] >= 1.0:
        goal["status"]    = GoalStatus.COMPLETED
        goal["completed"] = time.time()
        r.zrem(GOALS_KEY, goal_id)

        # Archive to history
        r.lpush(HISTORY_KEY, json.dumps({
            "goal_id":    goal_id,
            "title":      goal["title"],
            "completed":  time.time(),
            "created":    goal["created"],
            "duration_h": (time.time() - goal["created"]) / 3600
        }))
        r.ltrim(HISTORY_KEY, 0, 499)

        events.publish("experience.new", {
            "experience_type": "FEEDBACK_SIGNAL",
            "content": f"Goal completed: '{goal['title']}'",
            "intensity": 0.9,
            "session_id": goal.get("session_id", "")
        })

    r.set(f"{GOAL_PREFIX}{goal_id}", json.dumps(goal))

    return {
        "goal_id":    goal_id,
        "progress":   goal["progress"],
        "status":     goal["status"],
        "delta":      round(goal["progress"] - old_progress, 3)
    }


@app.get("/agenda")
async def generate_agenda(session_id: str = ""):
    """
    Generate an autonomous work agenda based on active goals.
    Returns the highest-priority actionable tasks right now.
    This is the system's autonomous work queue.
    """
    goal_ids = r.zrevrange(GOALS_KEY, 0, 9, withscores=True)
    agenda   = []

    for gid, priority in goal_ids:
        raw = r.get(f"{GOAL_PREFIX}{gid}")
        if not raw:
            continue

        g = json.loads(raw)
        if g.get("status") != GoalStatus.ACTIVE:
            continue

        plan_raw = r.get(f"{PLAN_PREFIX}{gid}")
        if not plan_raw:
            # Goal active but no plan yet
            agenda.append({
                "goal_id":     gid,
                "goal_title":  g["title"],
                "task":        "Decompose goal into sub-tasks",
                "can_automate": False,
                "priority":    priority
            })
            continue

        plan        = json.loads(plan_raw)
        progress    = g.get("progress", 0.0)
        sub_goals   = plan.get("sub_goals", [])

        if not sub_goals:
            continue

        # Find current sub-goal based on progress
        sg_idx     = int(progress * len(sub_goals))
        sg_idx     = min(sg_idx, len(sub_goals) - 1)
        current_sg = sub_goals[sg_idx]

        for task in current_sg.get("tasks", [])[:2]:
            agenda.append({
                "goal_id":      gid,
                "goal_title":   g["title"],
                "sub_goal":     current_sg.get("title", ""),
                "task":         task.get("description", ""),
                "can_automate": task.get("can_automate", False),
                "priority":     priority,
                "est_minutes":  task.get("estimated_minutes", 30)
            })

    agenda.sort(key=lambda x: (x["can_automate"], x["priority"]), reverse=True)

    return {
        "agenda":       agenda[:8],
        "active_goals": len(goal_ids),
        "automatable":  sum(1 for a in agenda if a["can_automate"]),
        "generated_at": time.time()
    }


@app.get("/history")
async def get_history(limit: int = 20):
    raw = r.lrange(HISTORY_KEY, 0, limit - 1)
    return {"history": [json.loads(h) for h in raw]}


@app.delete("/goals/{goal_id}")
async def abandon_goal(goal_id: str, reason: str = ""):
    raw = r.get(f"{GOAL_PREFIX}{goal_id}")
    if not raw:
        return {"error": "Not found"}
    g = json.loads(raw)
    g["status"]    = GoalStatus.ABANDONED
    g["abandoned"] = time.time()
    g["abandon_reason"] = reason
    r.set(f"{GOAL_PREFIX}{goal_id}", json.dumps(g))
    r.zrem(GOALS_KEY, goal_id)
    return {"abandoned": goal_id}
