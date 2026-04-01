# ── Goals: Persistent Agency and Long-Horizon Planning ───────────────────────
# The system can have goals that persist across sessions, across days.
# Goals decompose into sub-goals, sub-goals into tasks.
# The system pursues them autonomously when not responding to direct input.
# This is what enables "working on something for a week" behavior.

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

app    = FastAPI(title="Goals", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
r      = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
http   = httpx.AsyncClient(timeout=60.0)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MAIN_MODEL    = os.getenv("DEFAULT_CLAUDE_MODEL", "claude-sonnet-4-20250514")
THALAMUS_URL  = os.getenv("THALAMUS_URL",  "http://thalamus:8000")

GOALS_KEY     = "goals:active"      # sorted set: score=priority
GOAL_DATA_KEY = "goals:data:{id}"
PLAN_KEY      = "goals:plan:{id}"
HISTORY_KEY   = "goals:history"


class GoalStatus(str, Enum):
    PENDING    = "pending"
    ACTIVE     = "active"
    PAUSED     = "paused"
    COMPLETED  = "completed"
    FAILED     = "failed"
    ABANDONED  = "abandoned"


class Goal(BaseModel):
    id:          str   = ""
    title:       str
    description: str
    priority:    float = 0.5    # 0=low, 1=critical
    horizon:     str   = "week" # hour | day | week | month
    domain:      str   = "general"
    status:      str   = GoalStatus.PENDING
    progress:    float = 0.0
    sub_goals:   list  = []
    tasks:       list  = []
    created:     float = 0.0
    deadline:    Optional[float] = None
    session_id:  str   = ""


class CreateGoalRequest(BaseModel):
    title:       str
    description: str
    priority:    float = 0.5
    horizon:     str   = "week"
    domain:      str   = "general"
    session_id:  str   = ""
    deadline:    Optional[float] = None
    auto_plan:   bool  = True    # auto-decompose into sub-goals


@app.get("/health")
async def health():
    active_count = r.zcard(GOALS_KEY)
    return {
        "status":        "alive",
        "service":       "goals",
        "active_goals":  active_count
    }


@app.post("/goals", response_model=BrainResponse)
async def create_goal(req: CreateGoalRequest,
                      background: BackgroundTasks):
    """
    Create a new persistent goal.
    Optionally auto-decomposes into sub-goals and tasks.
    """
    start   = time.time()
    goal_id = uuid.uuid4().hex
    
    goal = Goal(
        id          = goal_id,
        title       = req.title,
        description = req.description,
        priority    = req.priority,
        horizon     = req.horizon,
        domain      = req.domain,
        status      = GoalStatus.PENDING,
        progress    = 0.0,
        created     = time.time(),
        deadline    = req.deadline,
        session_id  = req.session_id
    )
    
    # Store in Redis
    r.set(GOAL_DATA_KEY.format(id=goal_id), json.dumps(goal.dict()))
    r.zadd(GOALS_KEY, {goal_id: req.priority})
    
    # Auto-decompose if requested
    if req.auto_plan and ANTHROPIC_KEY:
        background.add_task(decompose_goal, goal_id)
    
    events.publish("experience.new", {
        "experience_type": "TEMPORAL_PATTERN",
        "content": f"New goal created: '{req.title}' (horizon={req.horizon})",
        "intensity": req.priority,
        "session_id": req.session_id
    })
    
    return BrainResponse(
        request_id = "goals",
        service    = "goals",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"goal_id": goal_id, "goal": goal.dict()}
    )


async def decompose_goal(goal_id: str):
    """
    Use Claude to decompose a goal into sub-goals and tasks.
    This is the hierarchical planning component.
    """
    goal_raw = r.get(GOAL_DATA_KEY.format(id=goal_id))
    if not goal_raw:
        return
    
    goal = json.loads(goal_raw)
    
    prompt = f"""Decompose this goal into concrete, executable sub-goals and tasks.

GOAL: {goal['title']}
DESCRIPTION: {goal['description']}
HORIZON: {goal['horizon']}
DOMAIN: {goal['domain']}

Create a hierarchical plan:
- Sub-goals: major milestones (3-5)
- Tasks: concrete actions for each sub-goal (2-4 per sub-goal)
- Each task should be completable in one focused session

Respond with JSON:
{{
    "sub_goals": [
        {{
            "id": "sg_1",
            "title": "...",
            "description": "...",
            "sequence": 1,
            "tasks": [
                {{
                    "id": "t_1",
                    "description": "specific action to take",
                    "estimated_minutes": 30,
                    "can_automate": true/false
                }}
            ]
        }}
    ],
    "critical_path": ["sg_1", "sg_2"],
    "estimated_total_hours": 5
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
                "model":      MAIN_MODEL,
                "max_tokens": 1000,
                "messages":   [{"role": "user", "content": prompt}]
            }
        )
        
        text = resp.json()["content"][0]["text"].strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        plan = json.loads(text)
        
        # Store plan
        r.set(PLAN_KEY.format(id=goal_id), json.dumps(plan))
        
        # Update goal with sub-goals
        goal["sub_goals"] = [sg["id"] for sg in plan.get("sub_goals", [])]
        goal["status"]    = GoalStatus.ACTIVE
        r.set(GOAL_DATA_KEY.format(id=goal_id), json.dumps(goal))
        
        print(f"Goal {goal_id} decomposed into "
              f"{len(plan.get('sub_goals', []))} sub-goals")
    
    except Exception as e:
        print(f"Goal decomposition error: {e}")


@app.get("/goals")
async def list_goals(status: str = None):
    """List all goals, optionally filtered by status."""
    goal_ids = r.zrevrange(GOALS_KEY, 0, -1, withscores=True)
    goals    = []
    
    for goal_id, priority in goal_ids:
        raw = r.get(GOAL_DATA_KEY.format(id=goal_id))
        if raw:
            g = json.loads(raw)
            if status is None or g.get("status") == status:
                goals.append(g)
    
    return {"goals": goals, "count": len(goals)}


@app.get("/goals/{goal_id}/plan")
async def get_plan(goal_id: str):
    """Get the decomposed plan for a goal."""
    plan_raw = r.get(PLAN_KEY.format(id=goal_id))
    if not plan_raw:
        return {"plan": None, "message": "No plan yet — decomposition pending"}
    return {"plan": json.loads(plan_raw)}


@app.patch("/goals/{goal_id}/progress")
async def update_progress(goal_id: str, progress: float,
                           completed_task: str = None):
    """Update goal progress. Trigger completion check."""
    raw = r.get(GOAL_DATA_KEY.format(id=goal_id))
    if not raw:
        return {"error": "Goal not found"}
    
    goal             = json.loads(raw)
    goal["progress"] = min(1.0, max(0.0, progress))
    
    if goal["progress"] >= 1.0:
        goal["status"]   = GoalStatus.COMPLETED
        goal["completed"] = time.time()
        r.zrem(GOALS_KEY, goal_id)
        
        # Log to history
        r.lpush(HISTORY_KEY, json.dumps({
            "goal_id": goal_id,
            "title":   goal["title"],
            "completed": time.time(),
            "horizon": goal["horizon"]
        }))
        r.ltrim(HISTORY_KEY, 0, 99)
    
    r.set(GOAL_DATA_KEY.format(id=goal_id), json.dumps(goal))
    
    return {"goal_id": goal_id, "progress": goal["progress"],
            "status": goal["status"]}


@app.post("/agenda")
async def generate_agenda(session_id: str = ""):
    """
    Generate an autonomous agenda based on active goals.
    What should the brain work on right now?
    This enables truly autonomous operation.
    """
    # Get active goals sorted by priority
    goal_ids = r.zrevrange(GOALS_KEY, 0, 9, withscores=True)
    active   = []
    
    for goal_id, priority in goal_ids:
        raw = r.get(GOAL_DATA_KEY.format(id=goal_id))
        if raw:
            g = json.loads(raw)
            if g.get("status") == GoalStatus.ACTIVE:
                plan_raw = r.get(PLAN_KEY.format(id=goal_id))
                g["plan"] = json.loads(plan_raw) if plan_raw else {}
                active.append((g, priority))
    
    if not active:
        return {"agenda": [], "message": "No active goals"}
    
    # Find next actionable tasks
    agenda = []
    for goal, priority in active[:3]:
        plan    = goal.get("plan", {})
        sub_goals = plan.get("sub_goals", [])
        progress  = goal.get("progress", 0)
        
        # Find the current sub-goal based on progress
        if sub_goals:
            current_sg_idx = int(progress * len(sub_goals))
            current_sg_idx = min(current_sg_idx, len(sub_goals) - 1)
            current_sg     = sub_goals[current_sg_idx]
            
            for task in current_sg.get("tasks", [])[:2]:
                agenda.append({
                    "goal_id":    goal["id"],
                    "goal_title": goal["title"],
                    "sub_goal":   current_sg["title"],
                    "task":       task["description"],
                    "can_automate": task.get("can_automate", False),
                    "priority":   priority
                })
    
    # Sort by priority
    agenda.sort(key=lambda x: x["priority"], reverse=True)
    
    return {
        "agenda":      agenda[:5],
        "active_goals": len(active),
        "generated":   time.time()
    }
