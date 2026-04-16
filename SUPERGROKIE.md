# Brain System — Architecture Guide for supergrokie

This repository contains a 23-service cognitive AI architecture.
Each service is a FastAPI microservice in a Docker container.
Services communicate only via HTTP — they never import each other.

## Service Map

```
PORT    SERVICE           PURPOSE
8000    thalamus          Master orchestrator — all input enters here
8001    amygdala          Urgency and emotion detection (regex, fast)
8002    hippocampus       Episodic memory — EdgeDB
8003    cerebellum        Skill registry — EdgeDB
8004    neocortex         Semantic knowledge — Elasticsearch
8005    working-memory    Active context — Redis
8006    perception        Experience detection (21 types)
8007    verification      3-stage fact checking
8008    cross-domain      Structural analogy reasoning
8009    compass           Ethical orientation (7 dimensions)
8010    framework-router  Multi-agent: Parallel/Colony/Hive/Swarm
8011    self-update       Code modification — you are here
8012    causal-graph      Cause-effect graph — Redis
8013    simulator         Sandboxed Python executor
8014    metacognition     Calibrated uncertainty
8015    goals             Long-horizon planning — Redis
8016    learner           Continual local model learning
8017    vision            Camera + CLIP visual memory
8018    autonomy          Autonomous self-improvement loop
8019    hypothalamus      Hormonal system — 16 hormones via 7 glands
8020    insight-engine    Epiphany detection via graph analysis
8080    dashboard         Web UI — terminal, monitoring, logs, controls
8090    knobs             Dynamic behavioral parameters — Redis
```

## Mandatory Patterns

### Every service file starts with:
```python
import sys
import time
import os
sys.path.insert(0, "/shared")

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus
from llm_provider import complete, complete_json   # never call APIs directly

app    = FastAPI(title="service-name", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=30.0)
```

### Inter-service calls use retries:
The thalamus `call()` and `call_get()` helpers automatically retry failed
requests (2 attempts with exponential back-off). Errors are logged and
surfaced in the response trace. If you add a new endpoint that calls
other services, use the same httpx client with proper error handling:
```python
# Pattern for calling other services with retry
async def call_with_retry(url, payload, retries=2):
    last_error = None
    for attempt in range(retries):
        try:
            resp = await http.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
    return {"error": str(last_error)}
```

### Embedding calls must guard against failures:
All embed() functions return `[]` on failure (never zero vectors, which
crash Elasticsearch cosine similarity). Use 5s timeout and retry once:
```python
async def embed(text):
    for attempt in range(2):
        try:
            r = await http.post(OLLAMA_URL + "/api/embeddings",
                                json={"model": EMBED_MODEL, "prompt": text},
                                timeout=5.0)
            vec = r.json().get("embedding", [])
            if not vec or all(v == 0.0 for v in vec[:10]):
                return []
            return vec
        except Exception:
            if attempt == 0:
                await asyncio.sleep(0.5)
    return []
```

### Every service has these two endpoints (mandatory):
```python
@app.get("/health")
async def health():
    return {"status": "alive", "service": "service-name"}

@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()
```

### Standard endpoint pattern:
```python
@app.post("/endpoint", response_model=BrainResponse)
async def my_endpoint(req: BrainRequest):
    start = time.time()
    k = knobs.get()   # read on every request — never cache
    try:
        # logic here
        result = do_something(req.user_input)
        return BrainResponse(
            request_id = req.request_id,
            service    = "service-name",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"result": result}
        )
    except Exception as e:
        return BrainResponse(
            request_id = req.request_id,
            service    = "service-name",
            success    = False,
            latency_ms = round((time.time() - start) * 1000, 2),
            error      = str(e),
            data       = {}
        )
```

### Environment variables:
```python
# Service URLs come from env vars — never hardcode
OTHER_SERVICE_URL = os.getenv("OTHER_SERVICE_URL", "http://other-service:PORT")

# Always provide a default that works in Docker network
REDIS_URL    = os.getenv("REDIS_URL",   "redis://redis:6379")
ES_URL       = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
OLLAMA_URL   = os.getenv("OLLAMA_URL",  "http://host.docker.internal:11434")
```

### Background tasks (fire and forget):
```python
asyncio.create_task(some_background_function())
# or with FastAPI:
background.add_task(some_function, arg1, arg2)
```

### Publishing events:
```python
events.publish("experience.new", {
    "experience_type": "SEMANTIC_NOVELTY",
    "content":         "description of what happened",
    "intensity":       0.75,
    "session_id":      req.session_id
})
```

## What NOT to do

- Never `import` another service's Python module
- Never call `anthropic.Anthropic()` or `openai.OpenAI()` directly — use `llm_provider.complete()`
- Never store mutable state in module-level variables (use Redis)
- Never use `time.sleep()` in async functions (use `asyncio.sleep()`)
- Never let unhandled exceptions crash the service
- Never add a `if __name__ == "__main__":` block
- Never hardcode API keys or service URLs
- Never hardcode timeouts — use retries with exponential back-off
- Never truncate conversation data stored in working memory
- Never return zero-magnitude vectors from embed() — return `[]` instead
- Never use `r.get()` on Redis list keys — check `r.type(key)` first

## Adding a new endpoint

When adding an endpoint, always:
1. Follow the BrainRequest/BrainResponse pattern exactly
2. Read knobs at the start: `k = knobs.get()`
3. Wrap in try/except
4. Return latency_ms in every response
5. Place it after the existing endpoints, before any __main__ block

## Testing a change

After any modification, verify:
- `/health` endpoint still returns `{"status": "alive"}`
- `/knobs` endpoint still returns valid JSON
- No syntax errors (the service must import cleanly)
- New endpoints follow BrainResponse pattern