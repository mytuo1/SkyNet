# ── Simulator: Ground Claims in Executable Reality ────────────────────────────
# The closest we can get to world-model grounding without sensors.
# Executes code to verify claims about computable reality.
# "What happens if I sort this list?" → actually runs it
# "What's 2^100?" → actually computes it
# "Does this regex match?" → actually tests it
# This grounds language in ACTUAL outcomes, not predicted outcomes.

import asyncio
import json
import os
import sys
import time
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, "/shared")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import BrainResponse
from events import EventBus

app    = FastAPI(title="Simulator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()

# Safe builtins — no file I/O, no network, no subprocess
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin,
    "bool": bool, "callable": callable, "chr": chr, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset,
    "getattr": getattr, "hasattr": hasattr, "hash": hash,
    "hex": hex, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len,
    "list": list, "map": map, "max": max, "min": min,
    "next": next, "oct": oct, "ord": ord, "pow": pow,
    "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None
}

SAFE_MODULES = {
    "math":       __import__("math"),
    "statistics": __import__("statistics"),
    "re":         __import__("re"),
    "json":       __import__("json"),
    "datetime":   __import__("datetime"),
    "random":     __import__("random"),
    "itertools":  __import__("itertools"),
    "collections":__import__("collections"),
    "string":     __import__("string"),
    "decimal":    __import__("decimal"),
    "fractions":  __import__("fractions"),
}


class SimulateRequest(BaseModel):
    code:         str
    description:  str = ""
    timeout_ms:   int = 3000   # max execution time
    session_id:   str = ""


class HypothesisRequest(BaseModel):
    hypothesis:   str    # natural language hypothesis
    test_code:    str    # code that tests the hypothesis
    expected:     str    # expected result (natural language)
    session_id:   str = ""


@app.get("/health")
async def health():
    return {"status": "alive", "service": "simulator"}


@app.post("/run", response_model=BrainResponse)
async def run_code(req: SimulateRequest):
    """
    Execute code in a sandboxed environment.
    Returns stdout, result, and any errors.
    This grounds abstract claims in executable reality.
    """
    start = time.time()
    
    stdout_buffer = StringIO()
    result_buffer = {}
    error         = None
    
    # Build safe namespace
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        **SAFE_MODULES
    }
    
    try:
        # Capture stdout
        with redirect_stdout(stdout_buffer):
            # Execute with timeout via asyncio
            loop   = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None,
                lambda: exec(req.code, safe_globals, result_buffer)
            )
            await asyncio.wait_for(
                future,
                timeout=req.timeout_ms / 1000
            )
        
        stdout_output = stdout_buffer.getvalue()
        
        # Extract result variable if set
        result_value = result_buffer.get("result", None)
        
        return BrainResponse(
            request_id = "sim",
            service    = "simulator",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {
                "success":     True,
                "stdout":      stdout_output,
                "result":      str(result_value) if result_value is not None else None,
                "description": req.description
            }
        )
    
    except asyncio.TimeoutError:
        error = f"Timeout after {req.timeout_ms}ms"
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
    
    return BrainResponse(
        request_id = "sim",
        service    = "simulator",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {
            "success": False,
            "error":   error,
            "stdout":  stdout_buffer.getvalue()
        }
    )


@app.post("/test-hypothesis")
async def test_hypothesis(req: HypothesisRequest):
    """
    Test a hypothesis by running code and comparing to expected result.
    This is how the system grounds its beliefs in reality:
    1. System claims X
    2. Write code that tests X
    3. Run it
    4. Compare to expected
    5. Update belief based on actual outcome
    """
    # Run the test code
    run_result = await run_code(SimulateRequest(
        code       = req.test_code,
        description = f"Testing: {req.hypothesis}",
        session_id  = req.session_id
    ))
    
    success     = run_result.data.get("success", False)
    actual      = run_result.data.get("stdout", "") + \
                  str(run_result.data.get("result", ""))
    error       = run_result.data.get("error", "")
    
    # Simple pass/fail: did it run without error?
    hypothesis_supported = success and not error
    
    # Log to causal graph if meaningful
    if success:
        events.publish("experience.new", {
            "experience_type": "EPISTEMIC_SURPRISE" if not hypothesis_supported else "FEEDBACK_SIGNAL",
            "content": f"Hypothesis '{req.hypothesis}': {'supported' if hypothesis_supported else 'refuted'} by execution",
            "intensity": 0.6,
            "session_id": req.session_id
        })
    
    return {
        "hypothesis":           req.hypothesis,
        "hypothesis_supported": hypothesis_supported,
        "actual_output":        actual[:500],
        "expected":             req.expected,
        "execution_success":    success,
        "error":                error
    }
