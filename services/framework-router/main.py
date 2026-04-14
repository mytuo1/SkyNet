# services/framework-router/main.py
# ── Framework Router: Multi-Agent Organizational Structures ──────────────────
# Selects the right thinking structure for complex problems.
# Four frameworks, each optimal for different task types:
#
#   PARALLEL  — diverse simultaneous perspectives, synthesized
#   COLONY    — specialized roles, sequential pipeline (scout→analyze→verify→synthesize)
#   HIVE      — shared memory, collective knowledge building, iterative convergence
#   SWARM     — decentralized exploration, emergent discovery, pheromone signals

import asyncio
import json
import os
import re
import sys
import time
import uuid
from typing import Optional

sys.path.insert(0, "/shared")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from knobs import KnobsClient
from llm_provider import complete, has_api_key

app = FastAPI(title="FrameworkRouter", version="1.0.0",
              description="Multi-agent frameworks: Parallel, Colony, Hive, Swarm")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs = KnobsClient()


class FrameworkRequest(BaseModel):
    request_id:      str = "framework"
    session_id:      str = ""
    user_input:      str
    amygdala_result: dict = {}
    context:         str  = ""
    framework:       Optional[str] = None   # force a specific framework


def select_framework(task: str, amygdala: dict, k) -> str:
    """
    Choose the optimal framework based on task characteristics.
    """
    task_lower  = task.lower()
    complexity  = amygdala.get("complexity", "medium")
    word_count  = len(task.split())

    # Synthesis / knowledge compilation → Hive
    if any(w in task_lower for w in [
        "synthesize", "compile", "comprehensive", "everything about",
        "full overview", "complete picture", "gather all"
    ]):
        return "hive"

    # Multi-step production / research + write → Colony
    if any(w in task_lower for w in [
        "research and write", "plan and execute", "analyze then",
        "full pipeline", "end to end", "step by step and"
    ]) or (complexity == "high" and word_count > 80):
        return "colony"

    # Creative exploration / brainstorming → Swarm
    if any(w in task_lower for w in [
        "brainstorm", "possibilities", "creative ideas", "innovate",
        "unexplored", "novel ways", "think outside", "what if"
    ]) or k.creativity > 0.85:
        return "swarm"

    # Default → Parallel (diverse perspectives)
    return "parallel"


# ── PARALLEL FRAMEWORK ────────────────────────────────────────────────────────

async def run_parallel(task: str, context: str, k) -> dict:
    """
    4 agents, each with a distinct lens, process simultaneously.
    Synthesizer combines independent views.
    Best for: risk assessment, complex decisions, diverse perspectives.
    """
    lenses = [
        ("ANALYTICAL",  "Analyze with pure logic and structured reasoning."),
        ("CREATIVE",    "Find unexpected angles, unconventional possibilities."),
        ("CRITICAL",    "Find weaknesses, gaps, risks, and what could go wrong."),
        ("EMPIRICAL",   "Focus on evidence, examples, and what is actually known."),
    ]

    n_agents = 3 if k.reasoning_depth < 0.5 else 4

    async def agent_think(role: str, instruction: str) -> str:
        prompt = f"{instruction}\n\nCONTEXT:\n{context}\n\nTASK: {task}"
        return await complete(prompt, system=f"You are {role}. {instruction}")

    tasks         = [agent_think(r, i) for r, i in lenses[:n_agents]]
    perspectives  = await asyncio.gather(*tasks)

    # Synthesize
    combined = "\n\n".join([
        f"[{lenses[i][0]}]: {perspectives[i]}"
        for i in range(len(perspectives))
    ])

    synthesis_prompt = f"""Multiple independent agents analyzed this task:

TASK: {task}

PERSPECTIVES:
{combined}

Synthesize into a unified response that:
1. Captures points of agreement (most reliable)
2. Notes important disagreements (need attention)
3. Integrates the strongest insights
4. Reaches a better conclusion than any single perspective"""

    synthesis = await complete(synthesis_prompt, model_tier="main", max_tokens=1500)

    return {
        "framework":    "parallel",
        "perspectives": [
            {"role": lenses[i][0], "view": perspectives[i]}
            for i in range(len(perspectives))
        ],
        "synthesis":     synthesis,
        "final_output":  synthesis
    }


# ── COLONY FRAMEWORK ─────────────────────────────────────────────────────────

async def run_colony(task: str, context: str) -> dict:
    """
    Pipeline: Scout → Analyze → Verify → Synthesize.
    Each stage uses the previous stage's output.
    Best for: multi-step production, research workflows.
    """
    # Phase 1: Scout (explore the territory)
    scout1, scout2 = await asyncio.gather(
        complete(
            f"CONTEXT: {context}\n\nTASK: {task}\n\n"
            f"You are a scout. Map what we need to know: "
            f"key concepts, relevant facts, important questions to answer.",
            system="You are an explorer. Find what matters for this task."
        ),
        complete(
            f"CONTEXT: {context}\n\nTASK: {task}\n\n"
            f"You are a second scout with a different angle. "
            f"What gaps, assumptions, or hidden aspects need exploring?",
            system="You are a scout looking for what others miss."
        )
    )

    scout_output = f"Scout 1:\n{scout1}\n\nScout 2:\n{scout2}"

    # Phase 2: Analyze (structure the findings)
    analyst1, analyst2 = await asyncio.gather(
        complete(
            f"TASK: {task}\n\nSCOUTED INFORMATION:\n{scout_output}\n\n"
            f"Structure and analyze these findings. Find patterns and key insights.",
            system="You are an analyst. Structure findings, find patterns."
        ),
        complete(
            f"TASK: {task}\n\nSCOUTED INFORMATION:\n{scout_output}\n\n"
            f"From a different angle, what is the most important insight?",
            system="You are an analyst focused on core insights."
        )
    )

    analysis_output = f"Analysis 1:\n{analyst1}\n\nAnalysis 2:\n{analyst2}"

    # Phase 3: Verify (check quality)
    verification = await complete(
        f"TASK: {task}\n\nANALYSIS:\n{analysis_output}\n\n"
        f"Verify this analysis. Find: errors, unsupported claims, "
        f"gaps, and what still needs addressing.",
        system="You are a rigorous verifier. Find what is wrong or missing."
    )

    # Phase 4: Synthesize (final output)
    synthesis = await complete(
        f"TASK: {task}\n\nANALYSIS:\n{analysis_output}\n\n"
        f"VERIFICATION NOTES:\n{verification}\n\n"
        f"Produce the final comprehensive response.",
        system="You are a synthesizer. Produce the definitive final output.",
        model_tier="main",
        max_tokens=2000
    )

    return {
        "framework":  "colony",
        "phases": {
            "scouting":     [scout1, scout2],
            "analysis":     [analyst1, analyst2],
            "verification": verification,
            "synthesis":    synthesis
        },
        "final_output": synthesis
    }


# ── HIVE FRAMEWORK ───────────────────────────────────────────────────────────

async def run_hive(task: str, context: str, iterations: int = 3) -> dict:
    """
    Shared collective knowledge state, built iteratively.
    Each agent reads the hive state and contributes.
    Best for: knowledge synthesis, building comprehensive understanding.
    """
    hive_mind = {
        "established_facts":  [],
        "open_questions":     [],
        "working_hypotheses": [],
        "confidence_map":     {}
    }

    def format_hive() -> str:
        hm = hive_mind
        lines = [
            f"FACTS ({len(hm['established_facts'])}):",
            *[f"  • {f}" for f in hm["established_facts"][-8:]],
            f"\nOPEN QUESTIONS ({len(hm['open_questions'])}):",
            *[f"  ? {q}" for q in hm["open_questions"][-4:]],
            f"\nHYPOTHESES ({len(hm['working_hypotheses'])}):",
            *[f"  ~ {h}" for h in hm["working_hypotheses"][-4:]],
        ]
        return "\n".join(lines)

    for iteration in range(iterations):
        hive_state = format_hive()

        # 5 agents contribute simultaneously
        contribute_tasks = [
            complete(
                f"TASK: {task}\n\nCOLLECTIVE KNOWLEDGE:\n{hive_state}\n\n"
                f"Contribute what is missing. Correct what is wrong. "
                f"Do not repeat established facts.\n\n"
                f"Respond with JSON:\n"
                f'{{"add_facts": [], "add_questions": [], '
                f'"add_hypotheses": [], "corrections": []}}',
                system="You are part of a collective intelligence. "
                       "Contribute only what adds new value. "
                       "Respond only with valid JSON."
            )
            for _ in range(5)
        ]

        contributions = await asyncio.gather(*contribute_tasks)

        # Integrate all contributions
        for contrib_text in contributions:
            contrib = None
            try:
                if "```" in contrib_text:
                    contrib_text = contrib_text.split("```")[1].lstrip("json").strip()
                contrib = json.loads(contrib_text)
            except (json.JSONDecodeError, IndexError):
                # Try to extract JSON object from response
                match = re.search(r'\{.*\}', contrib_text, re.DOTALL)
                if match:
                    try:
                        contrib = json.loads(match.group())
                    except json.JSONDecodeError:
                        pass
                if not contrib:
                    print(f"[Hive] Warning: could not parse contribution: "
                          f"{contrib_text[:120]}")

            if contrib:
                for fact in contrib.get("add_facts", []):
                    if fact and fact not in hive_mind["established_facts"]:
                        hive_mind["established_facts"].append(fact)
                for q in contrib.get("add_questions", []):
                    if q and q not in hive_mind["open_questions"]:
                        hive_mind["open_questions"].append(q)
                for h in contrib.get("add_hypotheses", []):
                    if h and h not in hive_mind["working_hypotheses"]:
                        hive_mind["working_hypotheses"].append(h)

    # Crystallize final output
    final = await complete(
        f"TASK: {task}\n\nCOLLECTIVE KNOWLEDGE:\n{format_hive()}\n\n"
        f"Synthesize this collective knowledge into a final comprehensive answer.",
        model_tier="main",
        max_tokens=2000
    )

    convergence = len(hive_mind["established_facts"]) / max(
        1, len(hive_mind["established_facts"]) + len(hive_mind["open_questions"])
    )

    return {
        "framework":   "hive",
        "hive_mind":   hive_mind,
        "convergence": round(convergence, 3),
        "iterations":  iterations,
        "final_output": final
    }


# ── SWARM FRAMEWORK ──────────────────────────────────────────────────────────

async def run_swarm(task: str, context: str, iterations: int = 3) -> dict:
    """
    Decentralized exploration. No leader. Emergent discovery.
    Agents sense pheromone trails (high-signal results) and follow them.
    Best for: creative exploration, novel solutions, large solution spaces.
    """
    angles = [
        "the simplest possible solution — remove all assumptions",
        "the most creative unconventional approach",
        "work backwards from failure — what must not happen?",
        "first principles — ignore all conventions",
        "what has worked for analogous problems in history?",
        "the minimum viable path — what is absolutely necessary?",
        "the user's deepest underlying need, not the surface request",
        "second-order effects — what happens after the solution?"
    ]

    pheromones: list = []   # high-signal discoveries that attract other agents
    all_results: list = []

    for iteration in range(iterations):
        # Format current pheromone trails
        if pheromones:
            trail_text = "\n".join([
                f"Signal {i+1} (strength={p['strength']:.1f}): {p['content'][:120]}"
                for i, p in enumerate(pheromones[-4:])
            ])
        else:
            trail_text = "No signals yet — you are pioneering."

        # All agents explore simultaneously
        explore_tasks = [
            complete(
                f"You are one agent in a decentralized swarm.\n\n"
                f"TASK: {task}\n"
                f"CONTEXT: {context}\n"
                f"YOUR STARTING ANGLE: {angles[i % len(angles)]}\n\n"
                f"SIGNALS FROM OTHER AGENTS:\n{trail_text}\n\n"
                f"Rules: EXPLORE your angle. ATTRACT toward high-signal results. "
                f"REPEL from dead ends. SIGNAL others if you find something strong.\n\n"
                f"End your response with:\n"
                f"SIGNAL STRENGTH: [0-10]\n"
                f"POSITION: [brief description of where you are in solution space]",
                system="You are a swarm agent. Explore. Discover. Signal."
            )
            for i in range(7)
        ]

        results = await asyncio.gather(*explore_tasks)

        # Extract signals and deposit pheromones
        for i, result in enumerate(results):
            # Extract signal strength
            match = re.search(r"SIGNAL STRENGTH:\s*(\d+)", result)
            strength = int(match.group(1)) if match else 5
            strength = min(10, max(0, strength))

            all_results.append({
                "result":   result,
                "strength": strength,
                "angle":    angles[i % len(angles)],
                "iteration": iteration
            })

            if strength > 6:
                pheromones.append({
                    "content":  result[:200],
                    "strength": float(strength),
                    "iteration": iteration
                })

        # Evaporate weak pheromones
        pheromones = [p for p in pheromones if p["strength"] > 2.0]
        for p in pheromones:
            p["strength"] *= 0.75

    # Best result = highest signal strength
    if all_results:
        best = max(all_results, key=lambda x: x["strength"])
    else:
        best = {"result": "No results", "strength": 0}

    # Synthesize best discoveries
    top_results = sorted(all_results, key=lambda x: x["strength"], reverse=True)[:3]
    synthesis   = await complete(
        f"TASK: {task}\n\nBEST DISCOVERIES FROM SWARM:\n" +
        "\n\n".join([r["result"][:400] for r in top_results]) +
        "\n\nSynthesize the strongest insights into a final answer.",
        model_tier="main",
        max_tokens=1500
    )

    emergence_score = len([r for r in all_results if r["strength"] > 7]) / max(1, len(all_results))

    return {
        "framework":       "swarm",
        "best_signal":     best["strength"],
        "emergence_score": round(emergence_score, 3),
        "iterations":      iterations,
        "agents_run":      len(all_results),
        "final_output":    synthesis
    }


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "alive", "service": "framework-router", "version": "1.0.0"}


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


@app.post("/process", response_model=BrainResponse)
async def process(req: FrameworkRequest):
    """
    Select and run the appropriate multi-agent framework.
    Returns structured results including final_output for thalamus.
    """
    start = time.time()
    k     = knobs.get()

    # Select framework
    chosen = (
        req.framework
        if req.framework in ("parallel", "colony", "hive", "swarm")
        else select_framework(req.user_input, req.amygdala_result, k)
    )

    print(f"[FrameworkRouter] Selected: {chosen.upper()} for: "
          f"'{req.user_input[:50]}...'")

    if chosen == "parallel":
        result = await run_parallel(req.user_input, req.context, k)
    elif chosen == "colony":
        result = await run_colony(req.user_input, req.context)
    elif chosen == "hive":
        iterations = 2 if k.reasoning_depth < 0.5 else 3
        result     = await run_hive(req.user_input, req.context, iterations)
    elif chosen == "swarm":
        iterations = 2 if k.reasoning_depth < 0.5 else 3
        result     = await run_swarm(req.user_input, req.context, iterations)
    else:
        result = await run_parallel(req.user_input, req.context, k)

    return BrainResponse(
        request_id = req.request_id,
        service    = "framework-router",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = result
    )


@app.get("/frameworks")
async def list_frameworks():
    """Describe available frameworks and their optimal use cases."""
    return {
        "parallel": {
            "description": "Multiple simultaneous perspectives, synthesized",
            "best_for":    "Risk assessment, complex decisions, diverse viewpoints"
        },
        "colony": {
            "description": "Specialized roles in pipeline: scout→analyze→verify→synthesize",
            "best_for":    "Research workflows, multi-step production tasks"
        },
        "hive": {
            "description": "Shared collective knowledge, iterative convergence",
            "best_for":    "Knowledge synthesis, building comprehensive understanding"
        },
        "swarm": {
            "description": "Decentralized exploration with pheromone signals",
            "best_for":    "Creative brainstorming, novel solutions, large solution spaces"
        }
    }
