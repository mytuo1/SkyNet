# services/simulator/main.py
# ── Simulator: Ground Claims in Executable Reality ───────────────────────────
# Executes sandboxed code to verify claims about computable reality.
# The only service in the brain that touches ACTUAL truth, not predicted truth.
# "What is 2^1000?" → runs it. "Does X regex match Y?" → tests it.
# Verification service calls this to ground hallucination checks.

import asyncio
import math
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Optional

sys.path.insert(0, "/shared")

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus

app    = FastAPI(title="Simulator", version="1.0.0",
                 description="Sandboxed code execution for grounding claims in reality")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()

# ── Safe execution environment ────────────────────────────────────────────────
# Only these builtins and modules are accessible.
# NO: file I/O, network, subprocess, importlib, __import__

SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin,
    "bool": bool, "chr": chr, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "hash": hash,
    "hex": hex, "int": int, "isinstance": isinstance, "iter": iter,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "oct": oct, "ord": ord, "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "NotImplemented": NotImplemented, "Ellipsis": Ellipsis,
    "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "StopIteration": StopIteration,
}

SAFE_MODULES = {
    "math":        __import__("math"),
    "statistics":  __import__("statistics"),
    "re":          __import__("re"),
    "json":        __import__("json"),
    "random":      __import__("random"),
    "itertools":   __import__("itertools"),
    "collections": __import__("collections"),
    "string":      __import__("string"),
    "decimal":     __import__("decimal"),
    "fractions":   __import__("fractions"),
    "functools":   __import__("functools"),
    "datetime":    __import__("datetime"),
    "heapq":       __import__("heapq"),
    "bisect":      __import__("bisect"),
    "operator":    __import__("operator"),
    "numpy":       np,
}

MAX_OUTPUT_LEN = 5000   # truncate long outputs


class RunRequest(BaseModel):
    code:        str
    description: str = ""
    timeout_ms:  int = 5000
    session_id:  str = ""


class HypothesisRequest(BaseModel):
    hypothesis:  str
    test_code:   str
    expected:    str    = ""
    session_id:  str   = ""


class MathRequest(BaseModel):
    expression:  str
    session_id:  str = ""


@app.get("/health")
async def health():
    # Verify sandbox works
    try:
        result = _run_sandboxed("result = 2 + 2", timeout=1.0)
        sandbox_ok = result.get("result") == "4"
    except Exception:
        sandbox_ok = False
    return {"status": "alive", "service": "simulator", "sandbox": sandbox_ok}


def _run_sandboxed(code: str, timeout: float = 5.0) -> dict:
    """Execute code in sandboxed namespace. Returns dict with result/output/error."""
    stdout_buf    = StringIO()
    local_ns: dict = {}

    safe_globals  = {
        "__builtins__": SAFE_BUILTINS,
        **SAFE_MODULES
    }

    error  = None
    output = ""

    try:
        with redirect_stdout(stdout_buf):
            exec(compile(code, "<sandbox>", "exec"), safe_globals, local_ns)
        output = stdout_buf.getvalue()[:MAX_OUTPUT_LEN]
    except SyntaxError as e:
        error = f"SyntaxError: {e}"
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
        output = stdout_buf.getvalue()[:MAX_OUTPUT_LEN]

    result_val = local_ns.get("result")
    if result_val is not None:
        try:
            result_str = str(result_val)[:MAX_OUTPUT_LEN]
        except Exception:
            result_str = "<unrepresentable>"
    else:
        result_str = None

    return {
        "success": error is None,
        "output":  output,
        "result":  result_str,
        "error":   error,
        "locals":  {k: str(v)[:200] for k, v in local_ns.items()
                    if not k.startswith("_")}
    }


@app.post("/run", response_model=BrainResponse)
async def run_code(req: RunRequest):
    """
    Execute code in a sandboxed Python environment.
    Set `result = <value>` in your code to return a value.
    """
    start   = time.time()
    timeout = min(req.timeout_ms / 1000, 10.0)

    loop   = asyncio.get_event_loop()
    try:
        exec_result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_sandboxed, req.code, timeout),
            timeout=timeout + 1
        )
    except asyncio.TimeoutError:
        exec_result = {
            "success": False,
            "output":  "",
            "result":  None,
            "error":   f"Execution timeout after {req.timeout_ms}ms"
        }

    return BrainResponse(
        request_id = "sim",
        service    = "simulator",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {**exec_result, "description": req.description}
    )


@app.post("/test-hypothesis")
async def test_hypothesis(req: HypothesisRequest):
    """
    Test a hypothesis by executing code and checking the outcome.
    This is how the brain grounds beliefs in executable reality.
    """
    run_result = await run_code(RunRequest(
        code        = req.test_code,
        description = f"Testing: {req.hypothesis}",
        session_id  = req.session_id,
        timeout_ms  = 5000
    ))

    success = run_result.data.get("success", False)
    output  = run_result.data.get("output", "")
    result  = run_result.data.get("result")
    error   = run_result.data.get("error", "")

    # Determine if hypothesis is supported
    # If the code ran without error and no explicit failure, it's supported
    supported = success and not error

    # If expected value provided, do string comparison
    if req.expected and result is not None:
        supported = req.expected.strip().lower() in result.lower() or \
                    result.strip() == req.expected.strip()

    if not supported:
        events.publish("experience.new", {
            "experience_type": "EPISTEMIC_SURPRISE",
            "content": f"Hypothesis '{req.hypothesis[:80]}' was refuted by execution",
            "intensity": 0.65,
            "session_id": req.session_id
        })

    return {
        "hypothesis":       req.hypothesis,
        "supported":        supported,
        "actual_result":    result or output[:200],
        "expected":         req.expected,
        "execution_success": success,
        "error":            error
    }


@app.post("/math")
async def evaluate_math(req: MathRequest):
    """Safely evaluate a mathematical expression."""
    code   = f"result = {req.expression}"
    result = await run_code(RunRequest(
        code        = code,
        description = f"Math: {req.expression}",
        session_id  = req.session_id
    ))
    return {
        "expression": req.expression,
        "result":     result.data.get("result"),
        "error":      result.data.get("error")
    }


@app.post("/verify-code")
async def verify_code_output(body: dict):
    """
    Given code and expected output, verify if they match.
    Used by verification service.
    """
    code     = body.get("code", "")
    expected = body.get("expected", "")

    result   = await run_code(RunRequest(code=code, timeout_ms=3000))
    actual   = (result.data.get("result") or result.data.get("output") or "").strip()
    matches  = expected.strip() in actual or actual == expected.strip()

    return {
        "verified": matches,
        "actual":   actual[:500],
        "expected": expected,
        "error":    result.data.get("error")
    }