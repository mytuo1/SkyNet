"""
SkyNet Brain Dashboard — single-file FastAPI application.

Serves a fully-featured monitoring / control dashboard for the SkyNet
cognitive architecture.  The HTML, CSS and JS are embedded as a single
template string so the service can be deployed without any build step.

Pages
-----
1. Terminal    — chat with the brain via thalamus /process (STT + webcam)
2. Monitoring  — per-service metrics, latency, error rates, sparklines
3. Logging     — structured log viewer + live event stream
4. Controls    — knob sliders, presets, perception feed, autonomy
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THALAMUS_URL = os.getenv("THALAMUS_URL", "http://thalamus:8000")
KNOBS_URL = os.getenv("KNOBS_URL", "http://knobs:8090")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

ALL_SERVICES = {
    "thalamus": 8000, "amygdala": 8001, "hippocampus": 8002,
    "cerebellum": 8003, "neocortex": 8004, "working-memory": 8005,
    "perception": 8006, "verification": 8007, "cross-domain": 8008,
    "compass": 8009, "framework-router": 8010, "self-update": 8011,
    "causal-graph": 8012, "simulator": 8013, "metacognition": 8014,
    "goals": 8015, "learner": 8016, "vision": 8017, "knobs": 8090,
    "autonomy": 8018, "dashboard": 8080,
}

PUBSUB_CHANNELS = [
    "experience.new",
    "memory.consolidate",
    "knowledge.updated",
    "skill.learned",
    "knobs.updated",
    "verification.failed",
    "verification.contested",
    "self.update.triggered",
    "session.started",
    "session.ended",
    "analogical.insight",
    "pattern.registered",
]

logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO)

STARTUP_TIME: float = time.time()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

http_client: httpx.AsyncClient | None = None
redis_pool: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, redis_pool
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))
    try:
        redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_pool.ping()
        logger.info("Redis connected")
    except Exception:
        redis_pool = None
        logger.warning("Redis unavailable — event streaming disabled")
    yield
    await http_client.aclose()
    if redis_pool:
        await redis_pool.aclose()


app = FastAPI(title="SkyNet Dashboard", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------


async def _proxy_get(url: str) -> JSONResponse:
    try:
        r = await http_client.get(url)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable", "url": url}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _proxy_post(url: str, body: dict) -> JSONResponse:
    try:
        r = await http_client.post(url, json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable", "url": url}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _proxy_patch(url: str, params: dict | None = None, body: dict | None = None) -> JSONResponse:
    try:
        r = await http_client.patch(url, params=params, json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable", "url": url}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# API routes — proxied to internal services
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def api_health():
    return await _proxy_get(f"{THALAMUS_URL}/health")


@app.post("/api/process")
async def api_process(body: dict):
    return await _proxy_post(f"{THALAMUS_URL}/process", body)


@app.get("/api/knobs")
async def api_knobs():
    return await _proxy_get(f"{KNOBS_URL}/current")


@app.get("/api/knobs/presets")
async def api_knobs_presets():
    return await _proxy_get(f"{KNOBS_URL}/presets")


@app.get("/api/knobs/history")
async def api_knobs_history():
    return await _proxy_get(f"{KNOBS_URL}/history")


@app.patch("/api/knobs/adjust")
async def api_knobs_adjust(knob_name: str, value: float):
    return await _proxy_patch(
        f"{KNOBS_URL}/adjust",
        params={"knob_name": knob_name, "value": str(value)},
    )


@app.patch("/api/knobs/adjust-many")
async def api_knobs_adjust_many(body: dict):
    return await _proxy_patch(f"{KNOBS_URL}/adjust-many", body=body)


@app.post("/api/knobs/preset/{name}")
async def api_knobs_preset(name: str):
    return await _proxy_post(f"{KNOBS_URL}/preset/{name}", {})


@app.post("/api/knobs/reset")
async def api_knobs_reset():
    return await _proxy_post(f"{KNOBS_URL}/reset", {})


@app.get("/api/service/{service_name}/knobs")
async def api_service_knobs(service_name: str):
    """Hit a specific service's /knobs endpoint (read-only view)."""
    port = ALL_SERVICES.get(service_name)
    if port is None:
        return JSONResponse({"error": f"Unknown service: {service_name}"}, status_code=404)
    return await _proxy_get(f"http://{service_name}:{port}/knobs")


# ---------------------------------------------------------------------------
# Metrics endpoints (Redis-backed)
# ---------------------------------------------------------------------------


def _hour_bucket(ts: float) -> str:
    """Return an ISO hour string for the given epoch timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:00Z")


@app.post("/api/metrics/record")
async def api_metrics_record(body: dict):
    """Record a single request metric from thalamus."""
    if redis_pool is None:
        return JSONResponse({"error": "Redis unavailable"}, status_code=503)

    service = body.get("service", "unknown")
    latency = float(body.get("latency_ms", 0))
    status = body.get("status", "ok")
    ts = float(body.get("timestamp", time.time()))
    hour = _hour_bucket(ts)

    pipe = redis_pool.pipeline()
    pipe.zadd(f"metrics:{service}:latency", {f"{ts}:{latency}": ts})
    pipe.zadd(f"metrics:{service}:requests", {f"{ts}:{status}": ts})
    if status == "error":
        pipe.zadd(f"metrics:{service}:errors", {f"{ts}": ts})
    pipe.hincrby(f"metrics:{service}:hourly:{hour}", "requests", 1)
    pipe.hincrbyfloat(f"metrics:{service}:hourly:{hour}", "total_latency", latency)
    if status == "error":
        pipe.hincrby(f"metrics:{service}:hourly:{hour}", "errors", 1)
    await pipe.execute()

    return JSONResponse({"ok": True})


@app.get("/api/metrics/overview")
async def api_metrics_overview():
    """Per-service stats for the monitoring page."""
    if redis_pool is None:
        return JSONResponse({"services": {}})

    now = time.time()
    day_ago = now - 86400
    services_data: dict = {}

    for svc in ALL_SERVICES:
        total_req = 0
        total_lat = 0.0
        total_err = 0
        latencies: list[float] = []
        hourly_buckets: list[dict] = []

        req_members = await redis_pool.zrangebyscore(
            f"metrics:{svc}:requests", day_ago, now
        )
        total_req = len(req_members)

        lat_members = await redis_pool.zrangebyscore(
            f"metrics:{svc}:latency", day_ago, now
        )
        for m in lat_members:
            parts = str(m).rsplit(":", 1)
            if len(parts) == 2:
                try:
                    latencies.append(float(parts[1]))
                except ValueError:
                    pass
        total_lat = sum(latencies) if latencies else 0.0

        err_members = await redis_pool.zrangebyscore(
            f"metrics:{svc}:errors", day_ago, now
        )
        total_err = len(err_members)

        for h in range(24):
            hour_ts = now - (23 - h) * 3600
            hour_key = _hour_bucket(hour_ts)
            raw = await redis_pool.hgetall(f"metrics:{svc}:hourly:{hour_key}")
            hourly_buckets.append({
                "hour": hour_key,
                "requests": int(raw.get("requests", 0)),
                "avg_latency": round(
                    float(raw.get("total_latency", 0)) / max(int(raw.get("requests", 0)), 1), 2
                ),
                "errors": int(raw.get("errors", 0)),
            })

        avg_lat = round(total_lat / max(len(latencies), 1), 2)
        sorted_lat = sorted(latencies)
        p95 = round(sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0, 2)
        error_rate = round(total_err / max(total_req, 1) * 100, 2)
        ok_hours = sum(1 for b in hourly_buckets if b["requests"] > 0 or b["errors"] == 0)
        uptime_pct = round(ok_hours / 24 * 100, 1)

        services_data[svc] = {
            "total_requests": total_req,
            "avg_latency_ms": avg_lat,
            "error_rate": error_rate,
            "p95_latency": p95,
            "last_24h_requests": hourly_buckets,
            "uptime_pct": uptime_pct,
        }

    return JSONResponse({"services": services_data})


@app.get("/api/metrics/history")
async def api_metrics_history(
    service: str = Query(...),
    days: int = Query(1, ge=1, le=7),
):
    """Time series for a specific service over N days."""
    if redis_pool is None:
        return JSONResponse({"service": service, "hourly": []})

    now = time.time()
    start = now - days * 86400
    total_hours = days * 24
    hourly: list[dict] = []

    for h in range(total_hours):
        hour_ts = start + h * 3600
        hour_key = _hour_bucket(hour_ts)
        raw = await redis_pool.hgetall(f"metrics:{service}:hourly:{hour_key}")
        reqs = int(raw.get("requests", 0))
        hourly.append({
            "hour": hour_key,
            "requests": reqs,
            "avg_latency": round(
                float(raw.get("total_latency", 0)) / max(reqs, 1), 2
            ),
            "errors": int(raw.get("errors", 0)),
        })

    return JSONResponse({"service": service, "hourly": hourly})


@app.get("/api/metrics/summary")
async def api_metrics_summary():
    """High-level system stats."""
    if redis_pool is None:
        return JSONResponse({
            "total_requests": 0,
            "avg_response_time_ms": 0,
            "active_services": 0,
            "uptime_hours": round((time.time() - STARTUP_TIME) / 3600, 2),
        })

    now = time.time()
    day_ago = now - 86400
    total_reqs = 0
    total_lat = 0.0
    lat_count = 0
    active = 0

    for svc in ALL_SERVICES:
        req_members = await redis_pool.zrangebyscore(
            f"metrics:{svc}:requests", day_ago, now
        )
        count = len(req_members)
        total_reqs += count
        if count > 0:
            active += 1

        lat_members = await redis_pool.zrangebyscore(
            f"metrics:{svc}:latency", day_ago, now
        )
        for m in lat_members:
            parts = str(m).rsplit(":", 1)
            if len(parts) == 2:
                try:
                    total_lat += float(parts[1])
                    lat_count += 1
                except ValueError:
                    pass

    return JSONResponse({
        "total_requests": total_reqs,
        "avg_response_time_ms": round(total_lat / max(lat_count, 1), 2),
        "active_services": active,
        "uptime_hours": round((time.time() - STARTUP_TIME) / 3600, 2),
    })


# ---------------------------------------------------------------------------
# Log endpoints (Redis-backed)
# ---------------------------------------------------------------------------

MAX_LOGS_PER_SERVICE = 1000


@app.post("/api/logs/record")
async def api_logs_record(body: dict):
    """Record a structured log entry."""
    if redis_pool is None:
        return JSONResponse({"error": "Redis unavailable"}, status_code=503)

    service = body.get("service", "unknown")
    entry = json.dumps({
        "service": service,
        "level": body.get("level", "info"),
        "message": body.get("message", ""),
        "timestamp": body.get("timestamp", time.time()),
    })
    pipe = redis_pool.pipeline()
    pipe.lpush(f"logs:{service}", entry)
    pipe.ltrim(f"logs:{service}", 0, MAX_LOGS_PER_SERVICE - 1)
    pipe.lpush("logs:_all", entry)
    pipe.ltrim("logs:_all", 0, MAX_LOGS_PER_SERVICE - 1)
    await pipe.execute()
    return JSONResponse({"ok": True})


@app.get("/api/logs/{service}")
async def api_logs_service(
    service: str,
    limit: int = Query(100, ge=1, le=1000),
    level: str = Query("all"),
):
    """Get recent logs for a specific service."""
    if redis_pool is None:
        return JSONResponse({"logs": []})

    raw = await redis_pool.lrange(f"logs:{service}", 0, limit - 1)
    logs = []
    for r in raw:
        try:
            entry = json.loads(r)
        except (json.JSONDecodeError, TypeError):
            continue
        if level != "all" and entry.get("level") != level:
            continue
        logs.append(entry)
    return JSONResponse({"logs": logs})


@app.get("/api/logs/all")
async def api_logs_all(
    limit: int = Query(50, ge=1, le=1000),
):
    """Get recent logs across all services."""
    if redis_pool is None:
        return JSONResponse({"logs": []})

    raw = await redis_pool.lrange("logs:_all", 0, limit - 1)
    logs = []
    for r in raw:
        try:
            logs.append(json.loads(r))
        except (json.JSONDecodeError, TypeError):
            continue
    return JSONResponse({"logs": logs})


# ---------------------------------------------------------------------------
# WebSocket — Redis pub/sub event stream
# ---------------------------------------------------------------------------


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    if redis_pool is None:
        await ws.send_json({"type": "error", "message": "Redis unavailable"})
        await ws.close()
        return

    pubsub = redis_pool.pubsub()
    try:
        await pubsub.subscribe(*PUBSUB_CHANNELS)
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    data = {"raw": msg["data"]}
                await ws.send_json({
                    "channel": msg["channel"],
                    "data": data,
                    "ts": time.time(),
                })
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WS error: %s", exc)
    finally:
        await pubsub.unsubscribe(*PUBSUB_CHANNELS)
        await pubsub.aclose()


# ---------------------------------------------------------------------------
# Dashboard HTML (served at root)
# ---------------------------------------------------------------------------


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SkyNet Brain Dashboard</title>
<style>
/* == Reset & Variables ================================================ */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e17;--surface:#141926;--surface2:#1b2236;--border:#232b42;
  --text:#e0e6ed;--text-dim:#7b8a9e;--accent:#00d4ff;--accent-dim:#0097b8;
  --success:#00c853;--warning:#ffd600;--error:#ff1744;
  --emo-gratitude:#ffd700;--emo-belonging:#4fc3f7;--emo-regret:#ce93d8;
  --emo-alarm:#ff1744;--emo-joy:#00e676;--emo-curiosity:#26c6da;
  --emo-neutral:#90a4ae;--emo-urgency:#ff9100;--emo-distress:#ef5350;
  --radius:8px;
  --font:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'SF Mono',SFMono-Regular,ui-monospace,'Cascadia Code',Menlo,monospace;
  --brain-thalamus:#d4c5a9;--brain-amygdala:#c47d7d;--brain-hippocampus:#6d9e9e;
  --brain-cerebellum:#7d9e7d;--brain-neocortex:#7d8ea0;--brain-prefrontal:#9e8eb3;
  --brain-temporal:#b3a07d;--brain-parietal:#7da0b3;--brain-occipital:#8e7da0;
  --brain-brainstem:#a09e7d;
}
html{font-size:14px}
body{font-family:var(--font);background:var(--bg);color:var(--text);
  min-height:100vh;line-height:1.5;overflow:hidden}
a{color:var(--accent);text-decoration:none}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
button{font-family:var(--font)}

/* == Layout =========================================================== */
#app{display:grid;grid-template-rows:auto auto 1fr;height:100vh;overflow:hidden}
.header{display:flex;align-items:center;gap:12px;padding:10px 20px;
  background:var(--surface);border-bottom:1px solid var(--border);position:relative;z-index:20}
.header h1{font-size:1.15rem;font-weight:700;background:linear-gradient(135deg,var(--accent),#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:.5px;white-space:nowrap}
.pulse-dot{width:8px;height:8px;border-radius:50%;background:var(--success);
  box-shadow:0 0 6px var(--success);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.header .meta-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.header .session{font-size:.75rem;color:var(--text-dim);font-family:var(--mono)}
.header .global-status{font-size:.75rem;padding:3px 10px;border-radius:12px;font-weight:600}

.nav{display:flex;gap:2px;padding:0 20px;background:var(--surface);border-bottom:1px solid var(--border)}
.nav-btn{padding:9px 20px;font-size:.82rem;font-weight:600;color:var(--text-dim);cursor:pointer;
  border:none;background:none;border-bottom:2px solid transparent;transition:.2s}
.nav-btn:hover{color:var(--text)}
.nav-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

.main{overflow:hidden;position:relative}
.page{display:none;position:absolute;inset:0;overflow-y:auto;padding:16px 20px}
.page.active{display:flex;flex-direction:column}

/* == Brain Schematic ================================================== */
.brain-thumb{position:fixed;top:54px;right:12px;width:120px;height:100px;z-index:30;
  cursor:pointer;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:4px;transition:.2s;opacity:.9}
.brain-thumb:hover{opacity:1;border-color:var(--accent)}
.brain-thumb svg{width:100%;height:100%}
.brain-modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:200;
  display:none;align-items:center;justify-content:center}
.brain-modal-overlay.open{display:flex}
.brain-modal{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:24px;width:540px;max-width:95vw;position:relative}
.brain-modal svg{width:100%;height:auto}
.brain-modal .close-btn{position:absolute;top:10px;right:14px;background:none;border:none;
  color:var(--text-dim);font-size:1.4rem;cursor:pointer}
.brain-modal h3{text-align:center;margin-bottom:12px;font-size:.95rem;color:var(--text-dim)}
.brain-region{fill:rgba(255,255,255,.04);stroke:rgba(255,255,255,.12);stroke-width:1.2;
  transition:fill .3s ease-in,opacity .3s;cursor:pointer}
.brain-region:hover{stroke:rgba(255,255,255,.4)}
.brain-region.active-thalamus{fill:var(--brain-thalamus);opacity:.85}
.brain-region.active-amygdala{fill:var(--brain-amygdala);opacity:.85}
.brain-region.active-hippocampus{fill:var(--brain-hippocampus);opacity:.85}
.brain-region.active-cerebellum{fill:var(--brain-cerebellum);opacity:.85}
.brain-region.active-neocortex{fill:var(--brain-neocortex);opacity:.85}
.brain-region.active-prefrontal{fill:var(--brain-prefrontal);opacity:.85}
.brain-region.active-temporal{fill:var(--brain-temporal);opacity:.85}
.brain-region.active-parietal{fill:var(--brain-parietal);opacity:.85}
.brain-region.active-occipital{fill:var(--brain-occipital);opacity:.85}
.brain-region.active-brainstem{fill:var(--brain-brainstem);opacity:.85}
.brain-label{font-size:7px;fill:rgba(255,255,255,.5);pointer-events:none;font-family:var(--font);text-anchor:middle}
.brain-tooltip{position:absolute;background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;padding:6px 10px;font-size:.75rem;pointer-events:none;z-index:210;
  white-space:nowrap;display:none;color:var(--text)}

/* == Terminal ========================================================= */
#page-terminal{gap:12px}
.chat-log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px;
  padding:10px;background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);min-height:0}
.msg{padding:10px 14px;border-radius:var(--radius);max-width:88%;word-wrap:break-word;font-size:.88rem}
.msg.user{align-self:flex-end;background:var(--accent-dim);color:#fff}
.msg.bot{align-self:flex-start;background:var(--surface2);border:1px solid var(--border)}
.msg .meta{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;font-size:.73rem}
.msg .meta span{padding:2px 8px;border-radius:12px;font-weight:600}
.badge-emotion{color:#111}
.badge-latency{background:var(--surface);color:var(--accent);border:1px solid var(--border)}
.badge-confidence{background:var(--surface);color:var(--warning);border:1px solid var(--border)}
.badge-path{background:var(--surface);color:var(--emo-curiosity);border:1px solid var(--border)}
.badge-urgency{color:#111}
.msg .trace-toggle{display:inline-block;margin-top:6px;cursor:pointer;color:var(--accent);font-size:.73rem}
.msg pre.trace{margin-top:6px;padding:8px;background:var(--bg);border-radius:4px;
  font-family:var(--mono);font-size:.7rem;overflow-x:auto;white-space:pre-wrap;display:none;
  max-height:240px;overflow-y:auto;border:1px solid var(--border)}
.chat-controls{display:flex;gap:8px;align-items:center}
.chat-controls input[type=text]{flex:1;padding:10px 14px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--surface);color:var(--text);font-size:.9rem;outline:none;font-family:var(--font)}
.chat-controls input:focus{border-color:var(--accent)}
.chat-btn{padding:10px 20px;border:none;border-radius:var(--radius);
  background:var(--accent);color:#0a0e17;font-weight:700;cursor:pointer;font-size:.88rem;transition:.15s}
.chat-btn:hover{background:#33dfff}
.chat-btn:disabled{opacity:.5;cursor:not-allowed}
.media-btn{width:38px;height:38px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--surface);color:var(--text-dim);cursor:pointer;font-size:1.1rem;
  display:flex;align-items:center;justify-content:center;transition:.15s;position:relative}
.media-btn:hover{border-color:var(--accent);color:var(--accent)}
.media-btn.active-media{border-color:var(--error);color:var(--error)}
.rec-dot{position:absolute;top:3px;right:3px;width:6px;height:6px;border-radius:50%;
  background:var(--error);display:none;animation:pulse 1s infinite}
.media-btn.active-media .rec-dot{display:block}
.webcam-preview{width:48px;height:36px;border-radius:4px;border:1px solid var(--border);
  object-fit:cover;display:none}

/* == Monitoring ======================================================= */
#page-monitoring{gap:14px}
.overview-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.ov-card{padding:16px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);text-align:center}
.ov-card .ov-val{font-size:1.6rem;font-weight:800;font-family:var(--mono);color:var(--accent)}
.ov-card .ov-label{font-size:.75rem;color:var(--text-dim);margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
.time-range{display:flex;gap:4px;align-items:center}
.time-range button{padding:4px 12px;border-radius:14px;border:1px solid var(--border);
  background:var(--surface);color:var(--text-dim);font-size:.75rem;cursor:pointer;transition:.15s}
.time-range button:hover{border-color:var(--accent);color:var(--accent)}
.time-range button.active{background:var(--accent);color:#0a0e17;border-color:var(--accent);font-weight:700}
.mon-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.mon-header h2{font-size:1rem;font-weight:700}
.svc-metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;
  flex:1;overflow-y:auto;min-height:0}
.svc-metric-card{padding:14px;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);transition:.15s}
.svc-metric-card:hover{border-color:var(--accent)}
.svc-metric-card .svc-m-name{font-weight:700;font-size:.85rem;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.svc-metric-card .svc-m-name .dot-sm{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.svc-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-size:.75rem;margin-bottom:8px}
.svc-stats .stat-label{color:var(--text-dim)}
.svc-stats .stat-val{font-family:var(--mono);text-align:right;color:var(--text)}
.sparkline{height:28px;display:flex;align-items:flex-end;gap:1px}
.sparkline .bar{flex:1;background:var(--accent);opacity:.5;border-radius:1px 1px 0 0;min-width:2px;transition:height .3s}
.sparkline .bar:hover{opacity:1}

/* == Logging ========================================================== */
#page-logging{gap:10px}
.log-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.log-toolbar select,.log-toolbar input{padding:6px 10px;border-radius:4px;
  border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:.82rem;outline:none}
.log-toolbar select:focus,.log-toolbar input:focus{border-color:var(--accent)}
.log-toolbar .ws-status{font-size:.75rem;display:flex;align-items:center;gap:4px;margin-left:auto}
.log-toolbar .small-btn{padding:4px 10px;border-radius:4px;border:1px solid var(--border);
  background:var(--surface);color:var(--text-dim);cursor:pointer;font-size:.75rem}
.log-toolbar .small-btn:hover{border-color:var(--accent);color:var(--accent)}
.dot-sm{width:8px;height:8px;border-radius:50%;flex-shrink:0;display:inline-block}
.dot-alive{background:var(--success);box-shadow:0 0 4px var(--success)}
.dot-degraded{background:var(--warning);box-shadow:0 0 4px var(--warning)}
.dot-unreachable{background:var(--error);box-shadow:0 0 4px var(--error)}
.log-scroll{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:3px;
  padding:8px;background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);
  font-family:var(--mono);font-size:.76rem;min-height:0}
.log-line{padding:5px 10px;border-radius:4px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.log-line.level-error{background:rgba(255,23,68,.08);border-left:3px solid var(--error)}
.log-line.level-warn{border-left:3px solid var(--warning)}
.log-line.level-info{border-left:3px solid var(--accent)}
.log-line .log-ts{color:var(--text-dim);font-size:.7rem;flex-shrink:0}
.log-line .log-svc{font-weight:700;font-size:.73rem;flex-shrink:0}
.log-line .log-lvl{font-size:.68rem;padding:1px 6px;border-radius:8px;font-weight:700;flex-shrink:0}
.log-line .log-msg{flex:1;word-break:break-word;color:var(--text)}
.lvl-info{background:rgba(0,212,255,.12);color:var(--accent)}
.lvl-warn{background:rgba(255,214,0,.12);color:var(--warning)}
.lvl-error{background:rgba(255,23,68,.15);color:var(--error)}
.live-events{margin-top:8px}
.live-events h3{font-size:.85rem;font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:6px}
.live-events-scroll{max-height:180px;overflow-y:auto;display:flex;flex-direction:column;gap:3px;
  padding:8px;background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);
  font-family:var(--mono);font-size:.74rem}
.live-event{padding:4px 8px;border-radius:4px;background:var(--surface2)}
.live-event .ev-ch{color:var(--accent);font-weight:600}
.live-event .ev-ts{color:var(--text-dim);font-size:.68rem}

/* == Controls ========================================================= */
#page-controls{gap:14px}
.ctrl-section{padding:14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius)}
.ctrl-section h3{font-size:.9rem;font-weight:700;margin-bottom:10px;color:var(--accent)}
.knobs-presets{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.knobs-presets button{padding:6px 14px;border-radius:16px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);font-size:.78rem;cursor:pointer;transition:.15s}
.knobs-presets button:hover{border-color:var(--accent);color:var(--accent)}
.knobs-presets button.active-preset{background:var(--accent);color:#0a0e17;border-color:var(--accent);font-weight:700}
.knob-groups{display:flex;flex-direction:column;gap:12px}
.knob-group h4{font-size:.82rem;font-weight:700;margin-bottom:6px;color:var(--text-dim)}
.knob-row{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.knob-row label{width:170px;font-size:.8rem;flex-shrink:0}
.knob-row input[type=range]{flex:1;accent-color:var(--accent);height:4px}
.knob-row .knob-val{width:42px;text-align:right;font-family:var(--mono);font-size:.78rem;color:var(--accent)}
.perc-list{display:flex;flex-direction:column;gap:8px}
.perc-card{padding:10px 14px;border-radius:var(--radius);background:var(--surface2);
  border:1px solid var(--border);display:flex;align-items:center;gap:12px}
.perc-card .perc-icon{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:1rem;flex-shrink:0}
.perc-card .perc-body{flex:1}
.perc-card .perc-type{font-weight:700;font-size:.82rem}
.perc-card .perc-detail{font-size:.75rem;color:var(--text-dim);margin-top:2px}
.perc-card .perc-bar{width:50px}
.perc-card .perc-bar .bar{height:5px;border-radius:3px;background:var(--border);overflow:hidden}
.perc-card .perc-bar .bar-fill{height:100%;border-radius:3px;transition:width .3s}
.perc-card .perc-bar .val{font-size:.68rem;color:var(--text-dim);text-align:right;margin-top:2px}
.perc-empty{text-align:center;color:var(--text-dim);padding:30px;font-size:.85rem}
.autonomy-ctrl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
.autonomy-ctrl button{padding:6px 16px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--surface2);color:var(--text);cursor:pointer;font-size:.8rem;transition:.15s}
.autonomy-ctrl button:hover{border-color:var(--accent);color:var(--accent)}
.autonomy-status{font-size:.78rem;color:var(--text-dim)}
.thought-list{margin-top:8px;display:flex;flex-direction:column;gap:4px;max-height:160px;overflow-y:auto}
.thought-item{font-size:.75rem;padding:4px 8px;background:var(--surface2);border-radius:4px;
  font-family:var(--mono);color:var(--text-dim)}

/* == Shared =========================================================== */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;
  align-items:center;justify-content:center;z-index:100}
.modal-overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px;max-width:560px;width:90%;max-height:70vh;overflow-y:auto}
.modal h3{margin-bottom:10px;font-size:1rem}
.modal pre{font-family:var(--mono);font-size:.76rem;background:var(--bg);padding:12px;
  border-radius:4px;white-space:pre-wrap;overflow-x:auto;border:1px solid var(--border)}
.modal .close-btn{float:right;cursor:pointer;color:var(--text-dim);font-size:1.2rem;
  background:none;border:none;line-height:1}
.loading{display:inline-block;width:14px;height:14px;border:2px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{padding:10px 14px;border-radius:var(--radius);background:rgba(255,23,68,.1);
  border:1px solid var(--error);color:var(--error);font-size:.82rem}
.scroll-area{flex:1;overflow-y:auto;min-height:0}
</style>
</head>
<body>
<div id="app">

<!-- == Header ========================================================= -->
<div class="header">
  <div class="pulse-dot" id="global-dot"></div>
  <h1>SkyNet Brain Dashboard</h1>
  <div class="meta-right">
    <span class="global-status" id="global-status" style="background:var(--success);color:#111">Online</span>
    <span class="session" id="session-display">session: ...</span>
  </div>
</div>

<!-- == Nav ============================================================ -->
<div class="nav" id="nav-bar">
  <button class="nav-btn active" data-page="page-terminal">&#x1F9E0; Terminal</button>
  <button class="nav-btn" data-page="page-monitoring">&#x1F4CA; Monitoring</button>
  <button class="nav-btn" data-page="page-logging">&#x1F4DC; Logging</button>
  <button class="nav-btn" data-page="page-controls">&#x1F39B; Controls</button>
</div>

<!-- == Main =========================================================== -->
<div class="main">

<!-- PAGE: Terminal -->
<div class="page active" id="page-terminal">
  <div class="chat-log" id="chat-log"></div>
  <div class="chat-controls">
    <button class="media-btn" id="btn-stt" title="Speech-to-Text" style="display:none">&#x1F3A4;<span class="rec-dot"></span></button>
    <button class="media-btn" id="btn-cam" title="Webcam capture" style="display:none">&#x1F4F7;<span class="rec-dot"></span></button>
    <video id="cam-preview" class="webcam-preview" autoplay muted playsinline></video>
    <input type="text" id="chat-input" placeholder="Talk to the brain..." autocomplete="off"/>
    <button class="chat-btn" id="chat-send">Send</button>
  </div>
</div>

<!-- PAGE: Monitoring -->
<div class="page" id="page-monitoring">
  <div class="mon-header">
    <h2>System Monitoring</h2>
    <div class="time-range" id="time-range">
      <button class="active" data-hours="1">1h</button>
      <button data-hours="6">6h</button>
      <button data-hours="24">24h</button>
      <button data-hours="168">7d</button>
    </div>
  </div>
  <div class="overview-cards" id="overview-cards">
    <div class="ov-card"><div class="ov-val" id="ov-reqs">&mdash;</div><div class="ov-label">Total Requests</div></div>
    <div class="ov-card"><div class="ov-val" id="ov-latency">&mdash;</div><div class="ov-label">Avg Response Time</div></div>
    <div class="ov-card"><div class="ov-val" id="ov-active">&mdash;</div><div class="ov-label">Active Services</div></div>
    <div class="ov-card"><div class="ov-val" id="ov-uptime">&mdash;</div><div class="ov-label">System Uptime</div></div>
  </div>
  <div class="svc-metrics-grid" id="svc-metrics-grid"></div>
</div>

<!-- PAGE: Logging -->
<div class="page" id="page-logging">
  <div class="log-toolbar">
    <select id="log-svc-select">
      <option value="all">All Services</option>
    </select>
    <select id="log-level-select">
      <option value="all">All Levels</option>
      <option value="info">Info</option>
      <option value="warn">Warn</option>
      <option value="error">Error</option>
    </select>
    <input type="text" id="log-search" placeholder="Search logs..."/>
    <button class="small-btn" id="log-clear-btn">Clear</button>
    <label style="font-size:.75rem;display:flex;align-items:center;gap:4px;color:var(--text-dim)">
      <input type="checkbox" id="log-autoscroll" checked/> Auto-scroll
    </label>
    <span class="ws-status"><span class="dot-sm dot-unreachable" id="ws-dot"></span><span id="ws-label">Disconnected</span></span>
  </div>
  <div class="log-scroll" id="log-scroll"></div>
  <div class="live-events">
    <h3><span class="dot-sm dot-alive"></span> Live Event Stream</h3>
    <div class="live-events-scroll" id="live-events"></div>
  </div>
</div>

<!-- PAGE: Controls -->
<div class="page" id="page-controls">
  <div class="scroll-area" style="display:flex;flex-direction:column;gap:14px">
    <div class="ctrl-section">
      <h3>&#x1F39B; Behavioral Knobs</h3>
      <div class="knobs-presets" id="knobs-presets"></div>
      <div class="knob-groups" id="knob-groups"></div>
    </div>
    <div class="ctrl-section">
      <h3>&#x1F9E0; Perception Feed</h3>
      <div class="perc-list" id="perc-list">
        <div class="perc-empty">No experiences detected yet.</div>
      </div>
    </div>
    <div class="ctrl-section">
      <h3>&#x1F916; Autonomy Controls</h3>
      <div class="autonomy-status" id="autonomy-status">Status: unknown</div>
      <div class="autonomy-ctrl">
        <button id="autonomy-on">Activate</button>
        <button id="autonomy-off">Deactivate</button>
      </div>
      <div class="thought-list" id="thought-list"></div>
    </div>
  </div>
</div>

</div><!-- /main -->
</div><!-- /app -->

<!-- == Brain Schematic (thumbnail) ==================================== -->
<div class="brain-thumb" id="brain-thumb" title="Click to expand brain view">
  <svg viewBox="0 0 200 160" xmlns="http://www.w3.org/2000/svg" id="brain-svg-sm">
    <g id="brain-g-sm">
      <path d="M40,130 C20,120 10,95 12,75 C14,55 25,35 50,22 C70,12 100,8 130,12 C155,16 175,30 183,50 C190,68 188,90 178,108 C170,122 155,132 135,136 C115,140 90,138 70,136 C55,134 45,133 40,130 Z" fill="none" stroke="rgba(255,255,255,.15)" stroke-width="1.5"/>
      <ellipse class="brain-region" id="sm-neocortex" cx="100" cy="38" rx="55" ry="22"/>
      <ellipse class="brain-region" id="sm-prefrontal" cx="45" cy="50" rx="22" ry="20"/>
      <ellipse class="brain-region" id="sm-parietal" cx="110" cy="48" rx="24" ry="16"/>
      <ellipse class="brain-region" id="sm-occipital" cx="165" cy="55" rx="18" ry="20"/>
      <ellipse class="brain-region" id="sm-temporal" cx="60" cy="95" rx="28" ry="18"/>
      <ellipse class="brain-region" id="sm-thalamus" cx="105" cy="78" rx="18" ry="14"/>
      <ellipse class="brain-region" id="sm-amygdala" cx="82" cy="92" rx="12" ry="10"/>
      <ellipse class="brain-region" id="sm-hippocampus" cx="120" cy="95" rx="16" ry="10"/>
      <ellipse class="brain-region" id="sm-cerebellum" cx="160" cy="110" rx="22" ry="18"/>
      <path class="brain-region" id="sm-brainstem" d="M100,105 C98,115 95,130 93,145 L107,145 C105,130 102,115 100,105 Z"/>
    </g>
  </svg>
</div>

<!-- == Brain Modal ===================================================== -->
<div class="brain-modal-overlay" id="brain-modal-overlay">
  <div class="brain-modal">
    <button class="close-btn" id="brain-modal-close">&times;</button>
    <h3>Cognitive Architecture</h3>
    <div style="position:relative" id="brain-modal-wrap">
      <svg viewBox="0 0 400 320" xmlns="http://www.w3.org/2000/svg" id="brain-svg-lg">
        <path d="M80,260 C40,240 20,190 24,150 C28,110 50,70 100,44 C140,24 200,16 260,24 C310,32 350,60 366,100 C380,136 376,180 356,216 C340,244 310,264 270,272 C230,280 180,276 140,272 C110,268 90,265 80,260 Z" fill="none" stroke="rgba(255,255,255,.15)" stroke-width="2"/>
        <ellipse class="brain-region" id="lg-neocortex" cx="200" cy="76" rx="110" ry="44" data-name="Neocortex" data-svc="neocortex"/>
        <ellipse class="brain-region" id="lg-prefrontal" cx="90" cy="100" rx="44" ry="40" data-name="Prefrontal Cortex" data-svc="compass"/>
        <ellipse class="brain-region" id="lg-parietal" cx="220" cy="96" rx="48" ry="32" data-name="Parietal Lobe" data-svc="working-memory"/>
        <ellipse class="brain-region" id="lg-occipital" cx="330" cy="110" rx="36" ry="40" data-name="Occipital Lobe" data-svc="vision"/>
        <ellipse class="brain-region" id="lg-temporal" cx="120" cy="190" rx="56" ry="36" data-name="Temporal Lobe" data-svc="cross-domain"/>
        <ellipse class="brain-region" id="lg-thalamus" cx="210" cy="156" rx="36" ry="28" data-name="Thalamus (Hub)" data-svc="thalamus"/>
        <ellipse class="brain-region" id="lg-amygdala" cx="164" cy="184" rx="24" ry="20" data-name="Amygdala" data-svc="amygdala"/>
        <ellipse class="brain-region" id="lg-hippocampus" cx="240" cy="190" rx="32" ry="20" data-name="Hippocampus" data-svc="hippocampus"/>
        <ellipse class="brain-region" id="lg-cerebellum" cx="320" cy="220" rx="44" ry="36" data-name="Cerebellum" data-svc="cerebellum"/>
        <path class="brain-region" id="lg-brainstem" d="M200,210 C196,230 190,260 186,290 L214,290 C210,260 204,230 200,210 Z" data-name="Brainstem" data-svc="framework-router"/>
        <text class="brain-label" x="200" y="80">Neocortex</text>
        <text class="brain-label" x="90" y="103">Prefrontal</text>
        <text class="brain-label" x="220" y="100">Parietal</text>
        <text class="brain-label" x="330" y="113">Occipital</text>
        <text class="brain-label" x="120" y="193">Temporal</text>
        <text class="brain-label" x="210" y="159">Thalamus</text>
        <text class="brain-label" x="164" y="187">Amygdala</text>
        <text class="brain-label" x="240" y="193">Hippocampus</text>
        <text class="brain-label" x="320" y="223">Cerebellum</text>
        <text class="brain-label" x="200" y="260">Brainstem</text>
      </svg>
      <div class="brain-tooltip" id="brain-tooltip"></div>
    </div>
  </div>
</div>

<!-- == Generic Modal =================================================== -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <button class="close-btn" id="modal-close">&times;</button>
    <h3 id="modal-title">Detail</h3>
    <pre id="modal-body"></pre>
  </div>
</div>

<script>
/* =====================================================================
   SkyNet Dashboard Client-Side Logic
   ===================================================================== */
var $=function(s){return document.querySelector(s)};
var $$=function(s){return document.querySelectorAll(s)};
var esc=function(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML};

/* -- Session ---------------------------------------------------------- */
var SESSION=localStorage.getItem('skynet_session');
if(!SESSION){SESSION=crypto.randomUUID();localStorage.setItem('skynet_session',SESSION)}
$('#session-display').textContent='session: '+SESSION.slice(0,8);

/* -- Navigation ------------------------------------------------------- */
$$('.nav-btn').forEach(function(b){b.addEventListener('click',function(){
  $$('.nav-btn').forEach(function(x){x.classList.remove('active')});
  $$('.page').forEach(function(x){x.classList.remove('active')});
  b.classList.add('active');
  $('#'+b.dataset.page).classList.add('active');
  if(b.dataset.page==='page-monitoring') refreshMonitoring();
  if(b.dataset.page==='page-logging') refreshLogs();
  if(b.dataset.page==='page-controls') loadKnobs();
})});

/* -- Emotion helpers -------------------------------------------------- */
function emotionColor(e){
  var m={gratitude:'var(--emo-gratitude)',belonging:'var(--emo-belonging)',
    regret:'var(--emo-regret)',alarm:'var(--emo-alarm)',joy:'var(--emo-joy)',
    curiosity:'var(--emo-curiosity)',urgency:'var(--emo-urgency)',
    distress:'var(--emo-distress)',neutral:'var(--emo-neutral)'};
  return m[e]||'var(--emo-neutral)';
}
function urgencyColor(u){return u>7?'var(--error)':u>4?'var(--warning)':'var(--success)'}

/* Region mapping */
var svcToRegion={thalamus:'thalamus',amygdala:'amygdala',hippocampus:'hippocampus',
  cerebellum:'cerebellum',neocortex:'neocortex',compass:'prefrontal',
  'cross-domain':'temporal',perception:'temporal','working-memory':'parietal',
  vision:'occipital','framework-router':'brainstem',goals:'brainstem'};

/* -- Brain Schematic -------------------------------------------------- */
$('#brain-thumb').addEventListener('click',function(){$('#brain-modal-overlay').classList.add('open')});
$('#brain-modal-close').addEventListener('click',function(){$('#brain-modal-overlay').classList.remove('open')});
$('#brain-modal-overlay').addEventListener('click',function(e){if(e.target===e.currentTarget)e.currentTarget.classList.remove('open')});

$$('#brain-svg-lg .brain-region').forEach(function(el){
  el.addEventListener('mouseenter',function(e){
    var tip=$('#brain-tooltip');
    var name=el.dataset.name||'';
    var svc=el.dataset.svc||'';
    tip.textContent=name+(svc?' ('+svc+')':'');
    tip.style.display='block';
    var r=$('#brain-modal-wrap').getBoundingClientRect();
    tip.style.left=(e.clientX-r.left+10)+'px';
    tip.style.top=(e.clientY-r.top-30)+'px';
  });
  el.addEventListener('mousemove',function(e){
    var tip=$('#brain-tooltip');
    var r=$('#brain-modal-wrap').getBoundingClientRect();
    tip.style.left=(e.clientX-r.left+10)+'px';
    tip.style.top=(e.clientY-r.top-30)+'px';
  });
  el.addEventListener('mouseleave',function(){$('#brain-tooltip').style.display='none'});
});

function lightRegion(regionKey,duration){
  var cls='active-'+regionKey;
  ['sm-','lg-'].forEach(function(prefix){
    var el=$('#'+prefix+regionKey);
    if(el){el.classList.add(cls);setTimeout(function(){el.classList.remove(cls)},duration||1600)}
  });
}

function animateBrainPipeline(cogPath){
  lightRegion('thalamus',1400);
  setTimeout(function(){lightRegion('amygdala',1200)},300);
  setTimeout(function(){lightRegion('parietal',1200)},600);
  if(Array.isArray(cogPath)){
    cogPath.forEach(function(svc,i){
      var region=svcToRegion[svc];
      if(region) setTimeout(function(){lightRegion(region,1000)},900+i*300);
    });
  }
}

/* -- Chat / Terminal -------------------------------------------------- */
var busy=false;
var camStream=null,camEnabled=false,sttActive=false;

$('#chat-send').addEventListener('click',sendMessage);
$('#chat-input').addEventListener('keydown',function(e){if(e.key==='Enter'&&!busy)sendMessage()});

function sendMessage(){
  var inp=$('#chat-input');
  var text=inp.value.trim();
  if(!text||busy)return;
  busy=true;inp.value='';$('#chat-send').disabled=true;
  appendMsg('user',esc(text));
  animateBrainPipeline([]);
  var t0=performance.now();
  var payload={input:text,session_id:SESSION};
  if(camEnabled){var frame=captureFrame();if(frame)payload.image=frame}
  fetch('/api/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
    .then(function(r){return r.json()})
    .then(function(d){
      var latency=((performance.now()-t0)/1000).toFixed(2);
      appendBotMsg(d,latency);
      if(d.cognitive_path||d.path) animateBrainPipeline(d.cognitive_path||d.path);
      if(d.experiences) addExperiences(d.experiences);
    })
    .catch(function(e){appendMsg('bot','<span class="error-box">Error: '+esc(e.message)+'</span>')})
    .finally(function(){busy=false;$('#chat-send').disabled=false});
}

function appendMsg(role,html){
  var log=$('#chat-log');
  var div=document.createElement('div');
  div.className='msg '+role;div.innerHTML=html;
  log.appendChild(div);log.scrollTop=log.scrollHeight;
}

function appendBotMsg(d,latency){
  var emotion=d.emotion||d.dominant_emotion||'neutral';
  var urgency=d.urgency;if(urgency===undefined||urgency===null)urgency='';
  var confidence=d.confidence;if(confidence===undefined||confidence===null)confidence='';
  var path=d.cognitive_path||d.path||'';
  var text=d.response||d.reply||d.text||JSON.stringify(d);
  var meta='<div class="meta">';
  meta+='<span class="badge-emotion" style="background:'+emotionColor(emotion)+'">'+esc(emotion)+'</span>';
  if(urgency!=='')meta+='<span class="badge-urgency" style="background:'+urgencyColor(urgency)+'">urgency '+urgency+'</span>';
  if(confidence!=='')meta+='<span class="badge-confidence">\u2691 '+Math.round(confidence*100)+'%</span>';
  meta+='<span class="badge-latency">\u23F1 '+latency+'s</span>';
  if(path)meta+='<span class="badge-path">\u21A3 '+esc(Array.isArray(path)?path.join(' \u2192 '):path)+'</span>';
  meta+='</div>';
  var trace='';
  if(d.trace){
    trace='<span class="trace-toggle" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'block\'?\'none\':\'block\'">\u25B6 trace</span>';
    trace+='<pre class="trace">'+esc(JSON.stringify(d.trace,null,2))+'</pre>';
  }
  appendMsg('bot',esc(text)+meta+trace);
}

/* -- Speech-to-Text --------------------------------------------------- */
var SpeechRec=window.SpeechRecognition||window.webkitSpeechRecognition;
if(SpeechRec){
  $('#btn-stt').style.display='flex';
  var recognition=new SpeechRec();
  recognition.continuous=false;recognition.interimResults=true;recognition.lang='en-US';
  recognition.onresult=function(e){
    var final_text='',interim='';
    for(var i=e.resultIndex;i<e.results.length;i++){
      if(e.results[i].isFinal)final_text+=e.results[i][0].transcript;
      else interim+=e.results[i][0].transcript;
    }
    $('#chat-input').value=final_text||interim;
  };
  recognition.onend=function(){
    sttActive=false;$('#btn-stt').classList.remove('active-media');
    var v=$('#chat-input').value.trim();
    if(v&&!busy)sendMessage();
  };
  recognition.onerror=function(){sttActive=false;$('#btn-stt').classList.remove('active-media')};
  $('#btn-stt').addEventListener('click',function(){
    if(sttActive){recognition.stop();sttActive=false;$('#btn-stt').classList.remove('active-media')}
    else{recognition.start();sttActive=true;$('#btn-stt').classList.add('active-media')}
  });
}

/* -- Webcam ----------------------------------------------------------- */
if(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia){
  $('#btn-cam').style.display='flex';
  var vid=$('#cam-preview');
  $('#btn-cam').addEventListener('click',function(){
    if(camEnabled){
      camEnabled=false;
      if(camStream){camStream.getTracks().forEach(function(t){t.stop()});camStream=null}
      vid.style.display='none';vid.srcObject=null;
      $('#btn-cam').classList.remove('active-media');
    }else{
      navigator.mediaDevices.getUserMedia({video:true}).then(function(stream){
        camStream=stream;vid.srcObject=stream;vid.style.display='block';
        camEnabled=true;$('#btn-cam').classList.add('active-media');
      }).catch(function(e){console.warn('Webcam denied',e)});
    }
  });
}

function captureFrame(){
  if(!camStream)return null;
  var v=$('#cam-preview');
  var c=document.createElement('canvas');
  c.width=v.videoWidth||320;c.height=v.videoHeight||240;
  c.getContext('2d').drawImage(v,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',0.7);
}

/* -- Monitoring ------------------------------------------------------- */
$$('#time-range button').forEach(function(b){b.addEventListener('click',function(){
  $$('#time-range button').forEach(function(x){x.classList.remove('active')});
  b.classList.add('active');refreshMonitoring();
})});

function refreshMonitoring(){
  Promise.all([fetch('/api/metrics/summary'),fetch('/api/metrics/overview')])
    .then(function(rs){return Promise.all(rs.map(function(r){return r.json()}))})
    .then(function(arr){
      var summary=arr[0],overview=arr[1];
      $('#ov-reqs').textContent=summary.total_requests.toLocaleString();
      $('#ov-latency').textContent=summary.avg_response_time_ms.toFixed(1)+'ms';
      $('#ov-active').textContent=summary.active_services;
      $('#ov-uptime').textContent=summary.uptime_hours.toFixed(1)+'h';
      renderServiceMetrics(overview.services||{});
    })
    .catch(function(e){$('#svc-metrics-grid').innerHTML='<div class="error-box">'+esc(e.message)+'</div>'});
}

var ALL_SVC_LIST=["thalamus","amygdala","hippocampus","cerebellum","neocortex","working-memory",
  "perception","verification","cross-domain","compass","framework-router","self-update",
  "causal-graph","simulator","metacognition","goals","learner","vision","knobs","autonomy","dashboard"];

function renderServiceMetrics(services){
  var grid=$('#svc-metrics-grid');
  grid.innerHTML=ALL_SVC_LIST.map(function(name){
    var s=services[name]||{total_requests:0,avg_latency_ms:0,error_rate:0,p95_latency:0,last_24h_requests:[],uptime_pct:100};
    var dotCls=s.total_requests>0?'dot-alive':'dot-unreachable';
    var buckets=s.last_24h_requests||[];
    var maxReq=Math.max.apply(null,buckets.map(function(b){return b.requests||0}).concat([1]));
    var bars=buckets.map(function(b){
      var h=Math.max(Math.round((b.requests||0)/maxReq*100),2);
      return '<div class="bar" style="height:'+h+'%" title="'+(b.hour||'')+': '+(b.requests||0)+' reqs"></div>';
    }).join('');
    return '<div class="svc-metric-card"><div class="svc-m-name"><span class="dot-sm '+dotCls+'"></span>'+esc(name)+'</div>'+
      '<div class="svc-stats"><span class="stat-label">Requests</span><span class="stat-val">'+s.total_requests+'</span>'+
      '<span class="stat-label">Avg Latency</span><span class="stat-val">'+s.avg_latency_ms.toFixed(1)+'ms</span>'+
      '<span class="stat-label">Error Rate</span><span class="stat-val">'+s.error_rate.toFixed(1)+'%</span>'+
      '<span class="stat-label">P95 Latency</span><span class="stat-val">'+s.p95_latency.toFixed(1)+'ms</span></div>'+
      '<div class="sparkline">'+bars+'</div></div>';
  }).join('');
}

setInterval(function(){
  if($('#page-monitoring').classList.contains('active'))refreshMonitoring();
},30000);

/* -- Logging ---------------------------------------------------------- */
var svcColors={};
ALL_SVC_LIST.forEach(function(s,i){svcColors[s]='hsl('+(i*17)%360+',55%,65%)'});
ALL_SVC_LIST.forEach(function(s){var o=document.createElement('option');o.value=s;o.textContent=s;$('#log-svc-select').appendChild(o)});

function refreshLogs(){
  var svc=$('#log-svc-select').value;
  var level=$('#log-level-select').value;
  var url=svc==='all'?'/api/logs/all?limit=200':('/api/logs/'+encodeURIComponent(svc)+'?limit=200&level='+level);
  fetch(url).then(function(r){return r.json()}).then(function(d){renderLogs(d.logs||[])}).catch(function(){});
}

function renderLogs(logs){
  var c=$('#log-scroll');
  var search=$('#log-search').value.toLowerCase();
  var level=$('#log-level-select').value;
  c.innerHTML='';
  logs.forEach(function(l){
    if(level!=='all'&&l.level!==level)return;
    if(search&&(l.message||'').toLowerCase().indexOf(search)<0&&(l.service||'').toLowerCase().indexOf(search)<0)return;
    var div=document.createElement('div');
    div.className='log-line level-'+(l.level||'info');
    var ts=new Date((l.timestamp||0)*1000);
    var timeStr=ts.toLocaleTimeString();
    var color=svcColors[l.service]||'var(--text-dim)';
    div.innerHTML='<span class="log-ts">'+timeStr+'</span>'+
      '<span class="log-svc" style="color:'+color+'">'+esc(l.service||'?')+'</span>'+
      '<span class="log-lvl lvl-'+(l.level||'info')+'">'+esc(l.level||'info')+'</span>'+
      '<span class="log-msg">'+esc(l.message||'')+'</span>';
    c.appendChild(div);
  });
  if($('#log-autoscroll').checked)c.scrollTop=c.scrollHeight;
}

$('#log-svc-select').addEventListener('change',refreshLogs);
$('#log-level-select').addEventListener('change',refreshLogs);
$('#log-search').addEventListener('input',refreshLogs);
$('#log-clear-btn').addEventListener('click',function(){$('#log-scroll').innerHTML='';$('#live-events').innerHTML=''});

/* -- WebSocket event stream ------------------------------------------- */
var ws,wsRetry=1000;
function connectWs(){
  var proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(proto+'://'+location.host+'/ws/events');
  ws.onopen=function(){$('#ws-dot').className='dot-sm dot-alive';$('#ws-label').textContent='Connected';wsRetry=1000};
  ws.onclose=function(){$('#ws-dot').className='dot-sm dot-unreachable';$('#ws-label').textContent='Disconnected';setTimeout(connectWs,wsRetry);wsRetry=Math.min(wsRetry*2,30000)};
  ws.onmessage=function(e){try{var m=JSON.parse(e.data);addLiveEvent(m)}catch(err){}};
}
connectWs();

var MAX_LIVE=150;
function addLiveEvent(m){
  var c=$('#live-events');
  var t=new Date((m.ts||0)*1000).toLocaleTimeString();
  var div=document.createElement('div');
  div.className='live-event';
  div.innerHTML='<span class="ev-ts">'+t+'</span> <span class="ev-ch">'+esc(m.channel||'')+'</span> '+esc(JSON.stringify(m.data||{})).slice(0,200);
  c.appendChild(div);
  if(c.children.length>MAX_LIVE)c.removeChild(c.firstChild);
  if($('#log-autoscroll').checked)c.scrollTop=c.scrollHeight;
}

/* -- Knob Controls ---------------------------------------------------- */
var knobGroups={
  Personality:['openness','conscientiousness','extraversion','agreeableness','neuroticism',
    'assertiveness','empathy','optimism','adventurousness','humility'],
  Cognition:['creativity','focus','verbosity','humor','formality','confidence_threshold',
    'curiosity','skepticism','patience','adaptability'],
  Framework:['analogical_weight','first_principles_weight','bayesian_weight',
    'dialectical_weight','systems_weight'],
  Ethics:['moral_weight','caution','transparency','autonomy_level','privacy_sensitivity',
    'fairness_weight'],
  'Higher Emotions':['gratitude','belonging','regret','alarm']
};

function loadKnobs(){
  Promise.all([fetch('/api/knobs'),fetch('/api/knobs/presets')])
    .then(function(rs){return Promise.all(rs.map(function(r){return r.json()}))})
    .then(function(arr){renderPresets(arr[1]);renderKnobs(arr[0])})
    .catch(function(e){$('#knob-groups').innerHTML='<div class="error-box">'+esc(e.message)+'</div>'});
}
function renderPresets(presets){
  var c=$('#knobs-presets');
  var names=Array.isArray(presets)?presets:Object.keys(presets||{});
  c.innerHTML=names.map(function(n){return '<button onclick="applyPreset(\''+esc(n)+'\')">'+esc(n)+'</button>'}).join('')+
    '<button onclick="resetKnobs()">\u21BA Reset</button>';
}
function applyPreset(name){fetch('/api/knobs/preset/'+encodeURIComponent(name),{method:'POST'}).then(function(){loadKnobs()})}
function resetKnobs(){fetch('/api/knobs/reset',{method:'POST'}).then(function(){loadKnobs()})}
function renderKnobs(knobs){
  var c=$('#knob-groups');
  c.innerHTML=Object.keys(knobGroups).map(function(group){
    var keys=knobGroups[group];
    var rows=keys.filter(function(k){return knobs[k]!==undefined}).map(function(k){
      var v=knobs[k];
      return '<div class="knob-row"><label>'+esc(k)+'</label><input type="range" min="0" max="1" step="0.01" value="'+v+'" data-knob="'+k+'" oninput="adjustKnob(this)"><span class="knob-val">'+Number(v).toFixed(2)+'</span></div>';
    }).join('');
    if(!rows)return '';
    return '<h4>'+esc(group)+'</h4>'+rows;
  }).join('');
}
var adjustTimer;
function adjustKnob(el){
  el.nextElementSibling.textContent=Number(el.value).toFixed(2);
  clearTimeout(adjustTimer);
  adjustTimer=setTimeout(function(){
    fetch('/api/knobs/adjust?knob_name='+encodeURIComponent(el.dataset.knob)+'&value='+el.value,{method:'PATCH'});
  },300);
}

/* -- Perception Feed -------------------------------------------------- */
var experiences=[];
function addExperiences(arr){
  if(!Array.isArray(arr))return;
  arr.forEach(function(ex){experiences.unshift(ex)});
  if(experiences.length>30)experiences.length=30;
  renderExperiences();
}
var expIcons={success:'\u2728',failure:'\u26A0',social:'\u{1F91D}',learning:'\u{1F4DA}',
  creative:'\u{1F3A8}',ethical_dilemma:'\u2696',routine:'\u{1F504}',surprise:'\u2757',
  emotional:'\u{1F496}','default':'\u{1F4A1}'};
var expColors={success:'var(--success)',failure:'var(--error)',social:'var(--emo-belonging)',
  learning:'var(--emo-curiosity)',creative:'var(--emo-regret)',ethical_dilemma:'var(--warning)',
  routine:'var(--text-dim)',surprise:'var(--emo-urgency)',emotional:'var(--emo-alarm)',
  'default':'var(--text-dim)'};
function renderExperiences(){
  var c=$('#perc-list');
  if(!experiences.length){c.innerHTML='<div class="perc-empty">No experiences yet.</div>';return}
  c.innerHTML=experiences.map(function(ex){
    var icon=expIcons[ex.type]||expIcons['default'];
    var color=expColors[ex.type]||expColors['default'];
    var pct=Math.round((ex.intensity||0)*100);
    return '<div class="perc-card"><div class="perc-icon" style="background:'+color+'22;color:'+color+'">'+icon+'</div>'+
      '<div class="perc-body"><div class="perc-type" style="color:'+color+'">'+esc(ex.type)+'</div>'+
      '<div class="perc-detail">'+esc(ex.detail)+'</div></div>'+
      '<div class="perc-bar"><div class="bar"><div class="bar-fill" style="width:'+pct+'%;background:'+color+'"></div></div><div class="val">'+pct+'%</div></div></div>';
  }).join('');
}

/* -- Autonomy Controls ------------------------------------------------ */
$('#autonomy-on').addEventListener('click',function(){
  fetch('/api/service/autonomy/knobs').then(function(r){return r.json()}).then(function(d){
    $('#autonomy-status').textContent='Status: active';
    $('#thought-list').innerHTML='<div class="thought-item">Autonomy activated</div>';
  }).catch(function(e){$('#autonomy-status').textContent='Status: error - '+e.message});
});
$('#autonomy-off').addEventListener('click',function(){$('#autonomy-status').textContent='Status: deactivated';$('#thought-list').innerHTML=''});

/* -- Modal ------------------------------------------------------------ */
$('#modal-close').addEventListener('click',function(){$('#modal-overlay').classList.remove('open')});
$('#modal-overlay').addEventListener('click',function(e){if(e.target===e.currentTarget)e.currentTarget.classList.remove('open')});

/* -- Init ------------------------------------------------------------- */
fetch('/api/health').then(function(r){
  if(!r.ok)throw new Error('offline');
  $('#global-status').style.background='var(--success)';$('#global-status').style.color='#111';$('#global-status').textContent='Online';
}).catch(function(){$('#global-status').style.background='var(--error)';$('#global-status').textContent='Offline'});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.get("/health")
async def health():
    return {"status": "alive", "service": "dashboard"}
