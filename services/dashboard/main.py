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


import logging as _log

def _parse_response(r: httpx.Response) -> JSONResponse:
    """Return upstream JSON when possible; fall back to raw text so errors are never swallowed."""
    try:
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception:
        return JSONResponse({"error": r.text or f"Upstream returned {r.status_code}"}, status_code=r.status_code)


async def _proxy_get(url: str) -> JSONResponse:
    try:
        r = await http_client.get(url)
        return _parse_response(r)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable"}, status_code=502)
    except Exception as exc:
        _log.exception("proxy GET %s failed", url)
        return JSONResponse({"error": "Internal proxy error"}, status_code=500)


async def _proxy_post(url: str, body: dict) -> JSONResponse:
    try:
        r = await http_client.post(url, json=body)
        return _parse_response(r)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable"}, status_code=502)
    except Exception as exc:
        _log.exception("proxy POST %s failed", url)
        return JSONResponse({"error": "Internal proxy error"}, status_code=500)


async def _proxy_patch(url: str, params: dict | None = None, body: dict | None = None) -> JSONResponse:
    try:
        r = await http_client.patch(url, params=params, json=body)
        return _parse_response(r)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable"}, status_code=502)
    except Exception as exc:
        _log.exception("proxy PATCH %s failed", url)
        return JSONResponse({"error": "Internal proxy error"}, status_code=500)


# ---------------------------------------------------------------------------
# API routes — proxied to internal services
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def api_health():
    return await _proxy_get(f"{THALAMUS_URL}/health")


@app.post("/api/process")
async def api_process(body: dict):
    return await _proxy_post(f"{THALAMUS_URL}/process", body)


@app.post("/api/teach-emotion")
async def api_teach_emotion(body: dict):
    return await _proxy_post(f"{THALAMUS_URL}/teach-emotion", body)


@app.get("/api/recent-learnings")
async def api_recent_learnings(session_id: str = ""):
    return await _proxy_get(
        f"{THALAMUS_URL}/recent-learnings?session_id={session_id}&limit=20")


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
                channel = msg["channel"]
                ts = time.time()
                await ws.send_json({
                    "channel": channel,
                    "data": data,
                    "ts": ts,
                })
                # Also persist to service logs so the Logging page shows them
                svc_name = _channel_to_service(channel, data)
                log_entry = json.dumps({
                    "service":   svc_name,
                    "level":     "info",
                    "message":   f"[{channel}] {json.dumps(data)[:200]}",
                    "timestamp": ts,
                })
                try:
                    pipe = redis_pool.pipeline()
                    pipe.lpush(f"logs:{svc_name}", log_entry)
                    pipe.ltrim(f"logs:{svc_name}", 0, MAX_LOGS_PER_SERVICE - 1)
                    pipe.lpush("logs:_all", log_entry)
                    pipe.ltrim("logs:_all", 0, MAX_LOGS_PER_SERVICE - 1)
                    await pipe.execute()
                except Exception:
                    pass
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WS error: %s", exc)
    finally:
        await pubsub.unsubscribe(*PUBSUB_CHANNELS)
        await pubsub.aclose()


# Map pubsub channels to service names for log storage
_CHANNEL_SERVICE_MAP = {
    "experience.new": "perception",
    "memory.consolidate": "hippocampus",
    "knowledge.updated": "neocortex",
    "skill.learned": "cerebellum",
    "knobs.updated": "knobs",
    "verification.failed": "verification",
    "verification.contested": "verification",
    "self.update.triggered": "self-update",
    "session.started": "thalamus",
    "session.ended": "thalamus",
    "analogical.insight": "cross-domain",
    "pattern.registered": "learner",
}


def _channel_to_service(channel: str, data: dict) -> str:
    """Derive a service name from a pubsub channel and payload."""
    if isinstance(channel, bytes):
        channel = channel.decode()
    return _CHANNEL_SERVICE_MAP.get(channel, data.get("service", "system"))


# ---------------------------------------------------------------------------
# Dashboard HTML (served at root)
# ---------------------------------------------------------------------------


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SkyNet Brain Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e17;--surface-solid:#141926;--surface:rgba(20,25,38,.85);--surface2:#1b2236;--border:#232b42;
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
  --glass-bg:rgba(20,25,38,.85);--glass-blur:12px;--glow-accent:rgba(0,212,255,.15);
}
html{font-size:14px}
body{font-family:var(--font);background:linear-gradient(135deg,#0a0e17 0%,#141926 50%,#1b2236 100%);
  color:var(--text);min-height:100vh;line-height:1.5;overflow:hidden}
a{color:var(--accent);text-decoration:none}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--surface-solid)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
button{font-family:var(--font)}

#app{display:grid;grid-template-rows:auto 1fr;height:100vh;overflow:hidden}
.header{display:flex;align-items:center;gap:12px;padding:10px 20px 10px 56px;
  background:linear-gradient(135deg,rgba(10,14,23,.95),rgba(20,25,38,.9),rgba(27,34,54,.95));
  backdrop-filter:blur(var(--glass-blur));border-bottom:1px solid var(--border);position:relative;z-index:20}
.header h1{font-size:1.15rem;font-weight:700;background:linear-gradient(135deg,var(--accent),#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:1px;white-space:nowrap}
.pulse-dot{width:8px;height:8px;border-radius:50%;background:var(--success);
  box-shadow:0 0 8px var(--success),0 0 16px rgba(0,200,83,.3);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 8px var(--success),0 0 16px rgba(0,200,83,.3)}50%{opacity:.4;box-shadow:0 0 4px var(--success)}}
.header .meta-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.header .session{font-size:.75rem;color:var(--text-dim);font-family:var(--mono);letter-spacing:.5px}
.header .global-status{font-size:.75rem;padding:3px 10px;border-radius:12px;font-weight:600;letter-spacing:.3px}

.nav{display:none}
.nav-btn{display:none}

/* == Hamburger Menu ==================================================== */
.hamburger{position:fixed;top:12px;left:12px;z-index:50;width:36px;height:36px;
  background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));border:1px solid var(--border);
  border-radius:var(--radius);cursor:pointer;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:4px;transition:all .25s ease}
.hamburger:hover{border-color:var(--accent);box-shadow:0 0 10px var(--glow-accent)}
.hamburger span{display:block;width:18px;height:2px;background:var(--text-dim);border-radius:1px;transition:all .3s ease}
.hamburger.open span:nth-child(1){transform:rotate(45deg) translate(4px,4px)}
.hamburger.open span:nth-child(2){opacity:0}
.hamburger.open span:nth-child(3){transform:rotate(-45deg) translate(4px,-4px)}
.hamburger.open{border-color:var(--accent)}
.hamburger.open span{background:var(--accent)}
.slide-menu{position:fixed;top:0;left:-300px;width:280px;height:100vh;z-index:45;
  background:linear-gradient(180deg,rgba(10,14,23,.98),rgba(20,25,38,.98));
  backdrop-filter:blur(16px);border-right:1px solid var(--border);
  transition:left .3s cubic-bezier(.4,0,.2,1);padding:60px 16px 16px;overflow-y:auto;
  box-shadow:4px 0 30px rgba(0,0,0,.4)}
.slide-menu.open{left:0}
.slide-menu-overlay{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:40;
  display:none;backdrop-filter:blur(2px)}
.slide-menu-overlay.open{display:block}
.menu-item{display:flex;align-items:center;gap:10px;padding:12px 14px;margin-bottom:4px;
  border-radius:var(--radius);color:var(--text-dim);cursor:pointer;font-size:.85rem;
  font-weight:600;letter-spacing:.3px;transition:all .2s ease;border:1px solid transparent}
.menu-item:hover{background:rgba(0,212,255,.06);color:var(--text);border-color:rgba(0,212,255,.15)}
.menu-item.active{background:rgba(0,212,255,.1);color:var(--accent);border-color:rgba(0,212,255,.25)}
.menu-item .mi-icon{font-size:1.1rem;width:24px;text-align:center}
.menu-divider{height:1px;background:var(--border);margin:12px 0}
.menu-label{font-size:.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;
  padding:8px 14px;font-weight:700}

.main{overflow:hidden;position:relative}
.page{display:none;position:absolute;inset:0;overflow-y:auto;padding:16px 20px;opacity:0;transition:opacity .3s ease}
.page.active{display:flex;flex-direction:column;opacity:1}
#page-terminal.active{padding:8px 0 0 0}

/* == Emotion Panel ==================================================== */
.emotion-panel{position:fixed;top:54px;left:12px;width:200px;z-index:10;
  background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:var(--radius);padding:12px;
  transition:all .3s ease;box-shadow:0 4px 20px rgba(0,0,0,.3);
  max-height:calc(100vh - 70px);overflow-y:auto}
.emotion-panel:hover{border-color:rgba(0,212,255,.25);box-shadow:0 4px 20px rgba(0,0,0,.3),0 0 15px var(--glow-accent)}
.emotion-panel.expanded{width:320px;max-height:500px;overflow-y:auto}
.emo-face-wrap{text-align:center;margin-bottom:8px}
.emo-face-wrap canvas{border-radius:6px;image-rendering:pixelated}
.emo-face-label{font-size:.7rem;color:var(--text-dim);margin-top:4px;letter-spacing:.5px;text-transform:uppercase}
.emo-current{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.emo-current .emo-emoji{font-size:1.5rem}
.emo-current .emo-name{font-weight:700;font-size:.85rem;letter-spacing:.3px}
.emo-badge{font-size:.68rem;padding:2px 8px;border-radius:10px;font-weight:700;color:#111;display:inline-block;margin-right:3px;margin-bottom:3px}
.emo-mixed-wrap{margin-bottom:6px}
.emo-mixed-label{font-size:.68rem;color:var(--text-dim);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.emo-mixed-list{display:flex;flex-wrap:wrap;gap:3px}
.emo-mixed-tag{font-size:.65rem;padding:2px 6px;border-radius:8px;font-weight:600;
  background:rgba(255,255,255,.08);color:var(--text-dim);border:1px solid rgba(255,255,255,.06)}
.emo-mixed-tag .tag-score{font-size:.58rem;opacity:.7;margin-left:2px}
.emo-mood{margin-bottom:8px}
.emo-mood-label{font-size:.7rem;color:var(--text-dim);margin-bottom:4px;letter-spacing:.3px}
.emo-mood-bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.emo-mood-bar-fill{height:100%;border-radius:3px;
  background:linear-gradient(90deg,var(--error),var(--warning),var(--success));transition:width .5s ease}
.emo-expand-btn{width:100%;padding:4px;border:none;background:none;color:var(--text-dim);
  cursor:pointer;font-size:.75rem;transition:.2s}
.emo-expand-btn:hover{color:var(--accent)}
.emo-history{max-height:220px;overflow-y:auto}
.emo-hist-item{display:flex;align-items:center;gap:6px;padding:4px 0;font-size:.72rem;
  border-bottom:1px solid rgba(255,255,255,.04)}
.emo-hist-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.emo-hist-ts{color:var(--text-dim);font-family:var(--mono);font-size:.68rem}
.emo-hist-name{font-weight:600;flex:1}
.emo-hist-conf{color:var(--text-dim);font-family:var(--mono)}
.emo-hist-mixed{font-size:.62rem;color:var(--text-dim);margin-left:4px}

/* == 3D Brain ========================================================= */
.brain-thumb{position:fixed;top:54px;right:12px;width:180px;height:140px;z-index:10;
  cursor:pointer;background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:var(--radius);
  padding:4px;transition:all .35s ease;overflow:hidden;transform-origin:top right}
.brain-thumb.expanded{width:270px;height:210px;z-index:25;
  box-shadow:0 8px 30px rgba(0,0,0,.5);border-color:rgba(0,212,255,.3)}
.brain-thumb:hover{border-color:rgba(0,212,255,.4);box-shadow:0 0 15px var(--glow-accent)}
.brain-thumb canvas{width:100%;height:100%;display:block;border-radius:calc(var(--radius) - 4px)}
.brain-modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:200;
  display:none;align-items:center;justify-content:center;backdrop-filter:blur(4px)}
.brain-modal-overlay.open{display:flex}
.brain-modal{background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:12px;padding:24px;width:650px;max-width:95vw;
  position:relative;box-shadow:0 8px 40px rgba(0,0,0,.5)}
.brain-modal canvas{width:100%;height:auto;border-radius:8px;cursor:grab;display:block}
.brain-modal canvas:active{cursor:grabbing}
.brain-modal .close-btn{position:absolute;top:10px;right:14px;background:none;border:none;
  color:var(--text-dim);font-size:1.4rem;cursor:pointer;transition:.2s}
.brain-modal .close-btn:hover{color:var(--accent)}
.brain-modal h3{text-align:center;margin-bottom:12px;font-size:.95rem;color:var(--text-dim);letter-spacing:.5px}
.brain-tooltip{position:absolute;background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;padding:6px 10px;font-size:.75rem;pointer-events:none;z-index:210;
  white-space:nowrap;display:none;color:var(--text);box-shadow:0 2px 10px rgba(0,0,0,.3)}

/* == Terminal ========================================================= */
#page-terminal{gap:0;position:relative;display:flex;flex-direction:column;overflow:hidden;
  padding-bottom:0 !important}
.chat-log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px;
  padding:10px 12%;background:var(--glass-bg);backdrop-filter:blur(8px);
  border-radius:var(--radius);border:1px solid var(--border);min-height:0;transition:all .25s ease;
  margin:0 0 0 0}
.chat-log:hover{border-color:rgba(0,212,255,.12)}

/* == Media Bar (above chat input) ===================================== */
.media-bar{display:flex;align-items:center;gap:8px;padding:8px 10px;
  background:var(--glass-bg);backdrop-filter:blur(8px);border:1px solid var(--border);
  border-bottom:none;border-radius:var(--radius) var(--radius) 0 0;margin-top:8px}
.media-btn{width:38px;height:38px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--glass-bg);color:var(--text-dim);cursor:pointer;font-size:1.1rem;
  display:flex;align-items:center;justify-content:center;transition:all .2s ease;position:relative;
  flex-shrink:0}
.media-btn:hover{border-color:var(--accent);color:var(--accent);box-shadow:0 0 10px var(--glow-accent)}
.media-btn.active-media{border-color:var(--success);color:var(--success)}
.media-btn.active-media.cam-active{border-color:var(--error);color:var(--error)}
.rec-dot{position:absolute;top:3px;right:3px;width:6px;height:6px;border-radius:50%;
  background:var(--error);display:none;animation:pulse 1s infinite}
.media-btn.active-media .rec-dot{display:block}
.webcam-preview{width:48px;height:36px;border-radius:4px;border:1px solid var(--border);
  object-fit:cover;display:none}

/* == Audio Waveform Visualizer ======================================== */
.waveform-wrap{flex:1;height:36px;display:none;align-items:center;border-radius:var(--radius);
  overflow:hidden;background:rgba(0,200,83,.06);border:1px solid rgba(0,200,83,.15);padding:0 4px}
.waveform-wrap.active{display:flex}
.waveform-canvas{width:100%;height:100%;display:block}
.media-label{font-size:.68rem;color:var(--text-dim);font-family:var(--mono);letter-spacing:.5px;
  white-space:nowrap}
.msg{padding:10px 14px;border-radius:var(--radius);max-width:88%;word-wrap:break-word;font-size:.88rem;
  animation:msgFade .3s ease}
@keyframes msgFade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-self:flex-end;background:linear-gradient(135deg,var(--accent-dim),#0077a8);color:#fff}
.msg.bot{align-self:flex-start;background:var(--glass-bg);backdrop-filter:blur(8px);border:1px solid var(--border)}
.msg .meta{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;font-size:.73rem}
.msg .meta span{padding:2px 8px;border-radius:12px;font-weight:600}
.badge-emotion{color:#111}
.badge-latency{background:var(--surface-solid);color:var(--accent);border:1px solid var(--border)}
.badge-confidence{background:var(--surface-solid);color:var(--warning);border:1px solid var(--border)}
.badge-path{background:var(--surface-solid);color:var(--emo-curiosity);border:1px solid var(--border)}
.badge-urgency{color:#111}
.msg .trace-toggle{display:inline-block;margin-top:6px;cursor:pointer;color:var(--accent);font-size:.73rem}
.msg pre.trace{margin-top:6px;padding:8px;background:var(--bg);border-radius:4px;
  font-family:var(--mono);font-size:.7rem;overflow-x:auto;white-space:pre-wrap;display:none;
  max-height:240px;overflow-y:auto;border:1px solid var(--border)}
.chat-controls{display:flex;gap:8px;align-items:center;
  border-radius:0 0 var(--radius) var(--radius);background:var(--glass-bg);backdrop-filter:blur(8px);
  border:1px solid var(--border);border-top:none;padding:8px 10px}
.chat-controls input[type=text]{flex:1;padding:10px 14px;border-radius:var(--radius);border:1px solid var(--border);
  background:rgba(10,14,23,.5);color:var(--text);font-size:.9rem;outline:none;
  font-family:var(--font);transition:all .25s ease}
.chat-controls input:focus{border-color:var(--accent);box-shadow:0 0 10px var(--glow-accent)}
.chat-btn{padding:10px 20px;border:none;border-radius:var(--radius);
  background:linear-gradient(135deg,var(--accent),#0097b8);color:#0a0e17;font-weight:700;
  cursor:pointer;font-size:.88rem;transition:all .2s ease;letter-spacing:.3px}
.chat-btn:hover{background:linear-gradient(135deg,#33dfff,var(--accent));box-shadow:0 0 15px var(--glow-accent)}
.chat-btn:disabled{opacity:.5;cursor:not-allowed}

/* == Monitoring ======================================================= */
#page-monitoring{gap:14px}
.overview-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.ov-card{padding:16px;background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:var(--radius);text-align:center;transition:all .25s ease}
.ov-card:hover{border-color:rgba(0,212,255,.25);box-shadow:0 0 15px var(--glow-accent)}
.ov-card .ov-val{font-size:1.6rem;font-weight:800;font-family:var(--mono);color:var(--accent);letter-spacing:.5px}
.ov-card .ov-label{font-size:.75rem;color:var(--text-dim);margin-top:4px;text-transform:uppercase;letter-spacing:.8px}
.time-range{display:flex;gap:4px;align-items:center}
.time-range button{padding:4px 12px;border-radius:14px;border:1px solid var(--border);
  background:var(--glass-bg);color:var(--text-dim);font-size:.75rem;cursor:pointer;transition:all .2s ease}
.time-range button:hover{border-color:var(--accent);color:var(--accent)}
.time-range button.active{background:linear-gradient(135deg,var(--accent),#0097b8);color:#0a0e17;
  border-color:var(--accent);font-weight:700}
.mon-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.mon-header h2{font-size:1rem;font-weight:700;letter-spacing:.3px}
.svc-metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;
  flex:1;overflow-y:auto;min-height:0}
.svc-metric-card{padding:14px;background:var(--glass-bg);backdrop-filter:blur(8px);
  border:1px solid var(--border);border-radius:var(--radius);transition:all .25s ease}
.svc-metric-card:hover{border-color:rgba(0,212,255,.25);box-shadow:0 0 15px var(--glow-accent)}
.svc-metric-card .svc-m-name{font-weight:700;font-size:.85rem;margin-bottom:8px;display:flex;align-items:center;gap:6px;letter-spacing:.3px}
.svc-metric-card .svc-m-name .dot-sm{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.svc-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-size:.75rem;margin-bottom:8px}
.svc-stats .stat-label{color:var(--text-dim)}
.svc-stats .stat-val{font-family:var(--mono);text-align:right;color:var(--text)}
.sparkline{height:28px;display:flex;align-items:flex-end;gap:1px}
.sparkline .bar{flex:1;background:linear-gradient(180deg,var(--accent),var(--accent-dim));opacity:.5;
  border-radius:1px 1px 0 0;min-width:2px;transition:height .3s}
.sparkline .bar:hover{opacity:1}

/* == Logging ========================================================== */
#page-logging{gap:10px}
.log-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.log-toolbar select,.log-toolbar input{padding:6px 10px;border-radius:4px;
  border:1px solid var(--border);background:var(--glass-bg);color:var(--text);font-size:.82rem;outline:none;transition:all .2s ease}
.log-toolbar select:focus,.log-toolbar input:focus{border-color:var(--accent);box-shadow:0 0 8px var(--glow-accent)}
.log-toolbar .ws-status{font-size:.75rem;display:flex;align-items:center;gap:4px;margin-left:auto}
.log-toolbar .small-btn{padding:4px 10px;border-radius:4px;border:1px solid var(--border);
  background:var(--glass-bg);color:var(--text-dim);cursor:pointer;font-size:.75rem;transition:all .2s ease}
.log-toolbar .small-btn:hover{border-color:var(--accent);color:var(--accent)}
.dot-sm{width:8px;height:8px;border-radius:50%;flex-shrink:0;display:inline-block}
.dot-alive{background:var(--success);box-shadow:0 0 4px var(--success)}
.dot-degraded{background:var(--warning);box-shadow:0 0 4px var(--warning)}
.dot-unreachable{background:var(--error);box-shadow:0 0 4px var(--error)}
.log-scroll{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:3px;
  padding:8px;background:var(--glass-bg);backdrop-filter:blur(8px);border-radius:var(--radius);
  border:1px solid var(--border);font-family:var(--mono);font-size:.76rem;min-height:0}
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
  padding:8px;background:var(--glass-bg);backdrop-filter:blur(8px);border-radius:var(--radius);
  border:1px solid var(--border);font-family:var(--mono);font-size:.74rem}
.live-event{padding:4px 8px;border-radius:4px;background:var(--surface2)}
.live-event .ev-ch{color:var(--accent);font-weight:600}
.live-event .ev-ts{color:var(--text-dim);font-size:.68rem}

/* == Controls ========================================================= */
#page-controls{gap:14px}
.ctrl-section{padding:14px;background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:var(--radius);transition:all .25s ease}
.ctrl-section:hover{border-color:rgba(0,212,255,.15)}
.ctrl-section h3{font-size:.9rem;font-weight:700;margin-bottom:10px;color:var(--accent);letter-spacing:.3px}
.knobs-presets{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.knobs-presets button{padding:6px 14px;border-radius:16px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);font-size:.78rem;cursor:pointer;transition:all .2s ease}
.knobs-presets button:hover{border-color:var(--accent);color:var(--accent);box-shadow:0 0 8px var(--glow-accent)}
.knobs-presets button.active-preset{background:linear-gradient(135deg,var(--accent),#0097b8);
  color:#0a0e17;border-color:var(--accent);font-weight:700}
.knob-groups{display:flex;flex-direction:column;gap:16px}
.knob-group h4{font-size:.82rem;font-weight:700;margin-bottom:8px;color:var(--text-dim);letter-spacing:.3px}
.knob-grid{display:flex;flex-wrap:wrap;gap:14px}
.rotary-knob{display:flex;flex-direction:column;align-items:center;width:88px;cursor:grab;user-select:none}
.rotary-knob:active{cursor:grabbing}
.rotary-knob .dial-wrap{position:relative;width:64px;height:64px}
.rotary-knob svg.dial{width:64px;height:64px;display:block}
.rotary-knob .dial-track{fill:none;stroke:var(--border);stroke-width:5;stroke-linecap:round}
.rotary-knob .dial-fill{fill:none;stroke:var(--accent);stroke-width:5;stroke-linecap:round;transition:stroke-dashoffset .08s}
.rotary-knob .dial-dot{fill:var(--accent);transition:transform .08s}
.rotary-knob .dial-val{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:.72rem;color:var(--accent);font-weight:700;pointer-events:none}
.rotary-knob .dial-label{font-size:.68rem;text-align:center;margin-top:4px;color:var(--text-dim);
  line-height:1.15;max-width:88px;word-break:break-word}
.perc-list{display:flex;flex-direction:column;gap:8px}
.perc-card{padding:10px 14px;border-radius:var(--radius);background:var(--surface2);
  border:1px solid var(--border);display:flex;align-items:center;gap:12px;transition:all .2s ease}
.perc-card:hover{border-color:rgba(0,212,255,.2)}
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
  background:var(--surface2);color:var(--text);cursor:pointer;font-size:.8rem;transition:all .2s ease}
.autonomy-ctrl button:hover{border-color:var(--accent);color:var(--accent);box-shadow:0 0 8px var(--glow-accent)}
.autonomy-status{font-size:.78rem;color:var(--text-dim)}
.thought-list{margin-top:8px;display:flex;flex-direction:column;gap:4px;max-height:160px;overflow-y:auto}
.thought-item{font-size:.75rem;padding:4px 8px;background:var(--surface2);border-radius:4px;
  font-family:var(--mono);color:var(--text-dim)}

/* == Shared =========================================================== */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;
  align-items:center;justify-content:center;z-index:100;backdrop-filter:blur(4px)}
.modal-overlay.open{display:flex}
.modal{background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:var(--radius);
  padding:20px;max-width:560px;width:90%;max-height:70vh;overflow-y:auto}
.modal h3{margin-bottom:10px;font-size:1rem}
.modal pre{font-family:var(--mono);font-size:.76rem;background:var(--bg);padding:12px;
  border-radius:4px;white-space:pre-wrap;overflow-x:auto;border:1px solid var(--border)}
.modal .close-btn{float:right;cursor:pointer;color:var(--text-dim);font-size:1.2rem;
  background:none;border:none;line-height:1;transition:.2s}
.modal .close-btn:hover{color:var(--accent)}
.loading{display:inline-block;width:14px;height:14px;border:2px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{padding:10px 14px;border-radius:var(--radius);background:rgba(255,23,68,.1);
  border:1px solid var(--error);color:var(--error);font-size:.82rem}
.scroll-area{flex:1;overflow-y:auto;min-height:0}
.brain-thumb.hidden-panel,.emotion-panel.hidden-panel{display:none !important}

/* == Session History ==================================================== */
.session-history{margin-top:12px}
.session-history .sh-label{font-size:.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;padding:0 14px 6px;font-weight:700}
.session-item{display:flex;align-items:center;gap:8px;padding:8px 14px;margin-bottom:2px;
  border-radius:var(--radius);color:var(--text-dim);cursor:pointer;font-size:.78rem;transition:all .2s ease;border:1px solid transparent}
.session-item:hover{background:rgba(0,212,255,.06);color:var(--text);border-color:rgba(0,212,255,.15)}
.session-item.active-session{background:rgba(0,212,255,.1);color:var(--accent);border-color:rgba(0,212,255,.25)}
.session-item .si-id{font-family:var(--mono);font-size:.72rem;letter-spacing:.3px}
.session-item .si-ts{font-size:.66rem;color:var(--text-dim);margin-left:auto}
.session-new-btn{width:100%;padding:8px;margin-top:4px;border:1px dashed var(--border);border-radius:var(--radius);
  background:none;color:var(--text-dim);cursor:pointer;font-size:.78rem;transition:all .2s ease}
.session-new-btn:hover{border-color:var(--accent);color:var(--accent)}
</style>
</head>
<body>
<div id="app">

<!-- == Hamburger Menu ================================================== -->
<button class="hamburger" id="hamburger-btn" title="Menu">
  <span></span><span></span><span></span>
</button>
<div class="slide-menu-overlay" id="slide-menu-overlay"></div>
<div class="slide-menu" id="slide-menu">
  <div class="menu-label">Navigate</div>
  <div class="menu-item active" data-page="page-terminal"><span class="mi-icon">&#x1F9E0;</span> Terminal</div>
  <div class="menu-divider"></div>
  <div class="menu-label">System</div>
  <div class="menu-item" data-page="page-monitoring"><span class="mi-icon">&#x1F4CA;</span> Monitoring</div>
  <div class="menu-item" data-page="page-logging"><span class="mi-icon">&#x1F4DC;</span> Logging</div>
  <div class="menu-item" data-page="page-controls"><span class="mi-icon">&#x1F39B;</span> Controls</div>
  <div class="menu-divider"></div>
  <div class="session-history" id="session-history">
    <div class="sh-label">Sessions</div>
    <div id="session-list"></div>
    <button class="session-new-btn" id="new-session-btn">+ New Session</button>
  </div>
</div>

<!-- == Header ========================================================= -->
<div class="header">
  <div class="pulse-dot" id="global-dot"></div>
  <h1>SkyNet Brain Dashboard</h1>
  <div class="meta-right">
    <span class="global-status" id="global-status" style="background:var(--success);color:#111">Online</span>
    <span class="session" id="session-display">session: ...</span>
  </div>
</div>

<!-- == Nav (hidden, kept for compatibility) ============================ -->
<div class="nav" id="nav-bar" style="display:none">
  <button class="nav-btn active" data-page="page-terminal">&#x1F9E0; Terminal</button>
  <button class="nav-btn" data-page="page-monitoring">&#x1F4CA; Monitoring</button>
  <button class="nav-btn" data-page="page-logging">&#x1F4DC; Logging</button>
  <button class="nav-btn" data-page="page-controls">&#x1F39B; Controls</button>
</div>

<!-- == Main =========================================================== -->
<div class="main">

<!-- PAGE: Terminal -->
<div class="page active" id="page-terminal">
  <div class="chat-log" id="chat-log" style="margin:0 210px"></div>
  <div class="media-bar" id="media-bar" style="margin:0 210px">
    <button class="media-btn" id="btn-stt" title="Microphone">&#x1F3A4;<span class="rec-dot"></span></button>
    <button class="media-btn" id="btn-cam" title="Webcam">&#x1F4F7;<span class="rec-dot"></span></button>
    <video id="cam-preview" class="webcam-preview" autoplay muted playsinline></video>
    <div class="waveform-wrap" id="waveform-wrap">
      <canvas id="waveform-canvas" class="waveform-canvas" height="34"></canvas>
    </div>
    <span class="media-label" id="media-label"></span>
  </div>
  <div class="chat-controls" style="margin:0 210px">
    <input type="text" id="chat-input" placeholder="Talk to the brain... (type /teach to enter teacher mode)" autocomplete="off"/>
    <button class="chat-btn" id="chat-send">Send</button>
  </div>
  <!-- Teacher Mode Bar -->
  <div id="teacher-bar" style="display:none;margin:0 210px;padding:6px 12px;background:rgba(255,214,0,.08);border:1px solid var(--warning);border-top:none;border-radius:0 0 var(--radius) var(--radius);font-size:.78rem;align-items:center;color:var(--warning)">
    <span style="font-weight:700">&#x1F4DA; Teacher Mode Active</span>
    <span style="color:var(--text-dim);margin-left:8px">— your messages teach the brain. Type <code>/teach</code> again to exit.</span>
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

<!-- == Emotion Panel (fixed, outside page flow) ======================== -->
<div class="emotion-panel" id="emotion-tracker">
  <div class="emo-face-wrap">
    <canvas id="pixel-face" width="80" height="80"></canvas>
    <div class="emo-face-label" id="emo-face-label">neutral</div>
  </div>
  <div class="emo-current">
    <span class="emo-emoji" id="emo-emoji">&#x1F610;</span>
    <span class="emo-name" id="emo-name">neutral</span>
  </div>
  <div style="margin-bottom:6px"><span class="emo-badge" id="emo-badge" style="background:var(--emo-neutral)">neutral</span></div>
  <div class="emo-mixed-wrap" id="emo-mixed-wrap" style="display:none">
    <div class="emo-mixed-label">Mixed Feelings</div>
    <div class="emo-mixed-list" id="emo-mixed-list"></div>
  </div>
  <div class="emo-mood">
    <div class="emo-mood-label">Overall Mood</div>
    <div class="emo-mood-bar"><div class="emo-mood-bar-fill" id="mood-bar-fill" style="width:50%"></div></div>
  </div>
  <button class="emo-expand-btn" id="emo-expand-btn">&#x25BC; History</button>
  <div class="emo-history" id="emo-history"></div>
</div>

<!-- == 3D Brain Thumbnail ============================================= -->
<div class="brain-thumb" id="brain-thumb" title="Click to expand brain view">
  <canvas id="brain-canvas-sm"></canvas>
</div>

<!-- == 3D Brain Modal ================================================= -->
<div class="brain-modal-overlay" id="brain-modal-overlay">
  <div class="brain-modal">
    <button class="close-btn" id="brain-modal-close">&times;</button>
    <h3>Cognitive Architecture</h3>
    <div style="position:relative" id="brain-modal-wrap">
      <canvas id="brain-canvas-lg" width="600" height="450"></canvas>
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

/* -- Device ID & Session Management ----------------------------------- */
var DEVICE_ID=localStorage.getItem('skynet_device_id');
if(!DEVICE_ID){DEVICE_ID=crypto.randomUUID();localStorage.setItem('skynet_device_id',DEVICE_ID)}
var SESSION=localStorage.getItem('skynet_session');
if(!SESSION){SESSION=crypto.randomUUID();localStorage.setItem('skynet_session',SESSION)}
/* Track session list: [{id, created, label}] */
var sessionList=JSON.parse(localStorage.getItem('skynet_sessions')||'[]');
if(!sessionList.find(function(s){return s.id===SESSION})){
  sessionList.unshift({id:SESSION,created:Date.now(),label:''});
  localStorage.setItem('skynet_sessions',JSON.stringify(sessionList));
}
$('#session-display').textContent='session: '+SESSION.slice(0,8);

function switchSession(sid){
  SESSION=sid;localStorage.setItem('skynet_session',sid);
  $('#session-display').textContent='session: '+sid.slice(0,8);
  $('#chat-log').innerHTML='';
  renderSessionList();
}
function newSession(){
  var sid=crypto.randomUUID();
  sessionList.unshift({id:sid,created:Date.now(),label:''});
  localStorage.setItem('skynet_sessions',JSON.stringify(sessionList));
  switchSession(sid);
}
function renderSessionList(){
  var c=$('#session-list');if(!c)return;
  c.innerHTML=sessionList.slice(0,20).map(function(s){
    var d=new Date(s.created);
    var ts=d.toLocaleDateString()+' '+d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
    var cls=s.id===SESSION?'session-item active-session':'session-item';
    return '<div class="'+cls+'" data-sid="'+s.id+'">'
      +'<span class="si-id">'+s.id.slice(0,8)+'</span>'
      +'<span class="si-ts">'+ts+'</span></div>';
  }).join('');
  c.querySelectorAll('.session-item').forEach(function(el){
    el.addEventListener('click',function(){switchSession(el.dataset.sid)});
  });
}
$('#new-session-btn').addEventListener('click',newSession);
renderSessionList();

/* -- Hamburger Menu --------------------------------------------------- */
var slideOpen=false;
function toggleMenu(){
  slideOpen=!slideOpen;
  $('#hamburger-btn').classList.toggle('open',slideOpen);
  $('#slide-menu').classList.toggle('open',slideOpen);
  $('#slide-menu-overlay').classList.toggle('open',slideOpen);
}
$('#hamburger-btn').addEventListener('click',toggleMenu);
$('#slide-menu-overlay').addEventListener('click',toggleMenu);

$$('.menu-item').forEach(function(b){b.addEventListener('click',function(){
  $$('.menu-item').forEach(function(x){x.classList.remove('active')});
  $$('.page').forEach(function(x){x.classList.remove('active')});
  b.classList.add('active');
  $('#'+b.dataset.page).classList.add('active');
  /* Show brain/emotions only on terminal page */
  var isTerminal=b.dataset.page==='page-terminal';
  $('#emotion-tracker').classList.toggle('hidden-panel',!isTerminal);
  $('#brain-thumb').classList.toggle('hidden-panel',!isTerminal);
  if(b.dataset.page==='page-monitoring') refreshMonitoring();
  if(b.dataset.page==='page-logging') refreshLogs();
  if(b.dataset.page==='page-controls') loadKnobs();
  if(slideOpen) toggleMenu();
})});

/* -- Navigation (legacy compat) --------------------------------------- */
$$('.nav-btn').forEach(function(b){b.addEventListener('click',function(){
  $$('.nav-btn').forEach(function(x){x.classList.remove('active')});
  $$('.page').forEach(function(x){x.classList.remove('active')});
  b.classList.add('active');
  $('#'+b.dataset.page).classList.add('active');
  if(b.dataset.page==='page-monitoring') refreshMonitoring();
  if(b.dataset.page==='page-logging') refreshLogs();
  if(b.dataset.page==='page-controls') loadKnobs();
})});

/* -- Emotion helpers (GoEmotions taxonomy) ----------------------------- */
var GOEMO_COLORS={
  admiration:'#ffd700',amusement:'#ffb300',approval:'#66bb6a',caring:'#f48fb1',
  desire:'#ff7043',excitement:'#ff5722',gratitude:'#ffd700',joy:'#00e676',
  love:'#e91e63',optimism:'#ffeb3b',pride:'#7e57c2',relief:'#81c784',
  anger:'#f44336',annoyance:'#ef5350',disappointment:'#9e9e9e',disapproval:'#78909c',
  disgust:'#795548',embarrassment:'#ce93d8',fear:'#ff1744',grief:'#5c6bc0',
  nervousness:'#ffb74d',remorse:'#ce93d8',sadness:'#42a5f5',
  confusion:'#b0bec5',curiosity:'#26c6da',realization:'#ffee58',surprise:'#ff9100',
  neutral:'#90a4ae',belonging:'#4fc3f7',urgency:'#ff9100',distress:'#ef5350',
  alarm:'#ff1744',regret:'#ce93d8',creative:'#7e57c2'
};
var GOEMO_VALENCE={
  admiration:0.8,amusement:0.7,approval:0.7,caring:0.75,desire:0.6,excitement:0.85,
  gratitude:0.9,joy:0.9,love:0.9,optimism:0.75,pride:0.8,relief:0.7,
  anger:0.15,annoyance:0.3,disappointment:0.25,disapproval:0.3,disgust:0.15,
  embarrassment:0.3,fear:0.15,grief:0.1,nervousness:0.3,remorse:0.2,sadness:0.15,
  confusion:0.45,curiosity:0.6,realization:0.6,surprise:0.55,
  neutral:0.5,belonging:0.7,urgency:0.25,distress:0.15,alarm:0.1,regret:0.2,creative:0.65
};
function emotionColor(e){
  return GOEMO_COLORS[e]||GOEMO_COLORS.neutral;
}
function urgencyColor(u){return u>7?'var(--error)':u>4?'var(--warning)':'var(--success)'}

/* Region mapping */
var svcToRegion={thalamus:'thalamus',amygdala:'amygdala',hippocampus:'hippocampus',
  cerebellum:'cerebellum',neocortex:'neocortex',compass:'prefrontal',
  'cross-domain':'temporal',perception:'temporal','working-memory':'parietal',
  vision:'occipital','framework-router':'brainstem',goals:'brainstem'};

/* =====================================================================
   Three.js 3D Brain
   ===================================================================== */
var brainDefs={
  neocortex:{px:0,py:1.2,pz:0,sx:2.5,sy:1.0,sz:2.0,color:0x7d8ea0,name:'Neocortex',svc:'neocortex'},
  prefrontal:{px:-1.0,py:0.8,pz:1.0,sx:0.9,sy:0.8,sz:0.8,color:0x9e8eb3,name:'Prefrontal Cortex',svc:'compass'},
  parietal:{px:0.2,py:1.5,pz:-0.2,sx:1.0,sy:0.6,sz:1.0,color:0x7da0b3,name:'Parietal Lobe',svc:'working-memory'},
  occipital:{px:0,py:0.8,pz:-1.5,sx:0.9,sy:0.8,sz:0.7,color:0x8e7da0,name:'Occipital Lobe',svc:'vision'},
  temporal:{px:-1.4,py:-0.2,pz:0.4,sx:0.7,sy:0.6,sz:1.1,color:0xb3a07d,name:'Temporal Lobe',svc:'cross-domain'},
  thalamus:{px:0,py:0.2,pz:0,sx:0.7,sy:0.5,sz:0.6,color:0xd4c5a9,name:'Thalamus (Hub)',svc:'thalamus'},
  amygdala:{px:-0.5,py:-0.1,pz:0.3,sx:0.4,sy:0.35,sz:0.35,color:0xc47d7d,name:'Amygdala',svc:'amygdala'},
  hippocampus:{px:0.6,py:-0.1,pz:0.3,sx:0.6,sy:0.3,sz:0.5,color:0x6d9e9e,name:'Hippocampus',svc:'hippocampus'},
  cerebellum:{px:0,py:-0.8,pz:-1.0,sx:1.2,sy:0.7,sz:0.9,color:0x7d9e7d,name:'Cerebellum',svc:'cerebellum'},
  brainstem:{px:0,py:-1.5,pz:-0.3,sx:0.35,sy:0.8,sz:0.35,color:0xa09e7d,name:'Brainstem',svc:'framework-router'}
};

var brainScene,brainGroup,brainMeshes={};
var thumbRenderer,thumbCamera,thumbAnimId;
var modalRenderer,modalCamera,modalAnimId;
var brainModalOpen=false;

function initBrainScene(){
  brainScene=new THREE.Scene();
  brainScene.background=new THREE.Color(0x0a0e17);
  brainScene.add(new THREE.AmbientLight(0x404060,0.6));
  var pt1=new THREE.PointLight(0xffffff,0.8,20);pt1.position.set(3,3,5);brainScene.add(pt1);
  var pt2=new THREE.PointLight(0x4488ff,0.3,15);pt2.position.set(-3,-2,3);brainScene.add(pt2);
  brainGroup=new THREE.Group();
  brainScene.add(brainGroup);
  var keys=Object.keys(brainDefs);
  for(var i=0;i<keys.length;i++){
    var key=keys[i];var def=brainDefs[key];
    var geom=new THREE.SphereGeometry(0.5,20,14);
    geom.scale(def.sx,def.sy,def.sz);
    var mat=new THREE.MeshPhongMaterial({
      color:def.color,transparent:true,opacity:0.15,
      emissive:def.color,emissiveIntensity:0.05,
      shininess:30,specular:0x222222
    });
    var mesh=new THREE.Mesh(geom,mat);
    mesh.position.set(def.px,def.py,def.pz);
    mesh.userData={key:key,name:def.name,svc:def.svc,baseColor:def.color};
    brainGroup.add(mesh);
    brainMeshes[key]=mesh;
  }
}

function initThumbRenderer(){
  var canvas=$('#brain-canvas-sm');
  var w=180,h=140;
  thumbRenderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true,alpha:false});
  thumbRenderer.setSize(w,h);
  thumbRenderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  thumbCamera=new THREE.PerspectiveCamera(38,w/h,0.1,100);
  thumbCamera.position.set(0,0.5,6.5);
  thumbCamera.lookAt(0,0,0);
  function animate(){
    thumbAnimId=requestAnimationFrame(animate);
    if(!brainModalOpen){
      brainGroup.rotation.y+=0.002;
      thumbRenderer.render(brainScene,thumbCamera);
    }
  }
  animate();
}

function initModalRenderer(){
  var canvas=$('#brain-canvas-lg');
  var w=600,h=450;
  modalRenderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true,alpha:false});
  modalRenderer.setSize(w,h);
  modalRenderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  modalCamera=new THREE.PerspectiveCamera(38,w/h,0.1,100);
  modalCamera.position.set(0,0.5,6.5);
  modalCamera.lookAt(0,0,0);

  var isDrag=false,prevMX=0,prevMY=0;
  canvas.addEventListener('mousedown',function(e){isDrag=true;prevMX=e.clientX;prevMY=e.clientY});
  document.addEventListener('mousemove',function(e){
    if(!isDrag||!brainModalOpen)return;
    brainGroup.rotation.y+=(e.clientX-prevMX)*0.005;
    brainGroup.rotation.x+=(e.clientY-prevMY)*0.005;
    prevMX=e.clientX;prevMY=e.clientY;
  });
  document.addEventListener('mouseup',function(){isDrag=false});

  var raycaster=new THREE.Raycaster();
  var mouse=new THREE.Vector2();
  canvas.addEventListener('mousemove',function(e){
    if(isDrag)return;
    var rect=canvas.getBoundingClientRect();
    mouse.x=((e.clientX-rect.left)/rect.width)*2-1;
    mouse.y=-((e.clientY-rect.top)/rect.height)*2+1;
    raycaster.setFromCamera(mouse,modalCamera);
    var hits=raycaster.intersectObjects(brainGroup.children);
    var tip=$('#brain-tooltip');
    if(hits.length>0){
      var obj=hits[0].object;
      tip.textContent=obj.userData.name+' ('+obj.userData.svc+')';
      tip.style.display='block';
      var wr=$('#brain-modal-wrap').getBoundingClientRect();
      tip.style.left=(e.clientX-wr.left+10)+'px';
      tip.style.top=(e.clientY-wr.top-30)+'px';
    }else{tip.style.display='none'}
  });
  canvas.addEventListener('mouseleave',function(){$('#brain-tooltip').style.display='none'});

  function animate(){
    modalAnimId=requestAnimationFrame(animate);
    if(brainModalOpen){
      if(!isDrag)brainGroup.rotation.y+=0.002;
      modalRenderer.render(brainScene,modalCamera);
    }
  }
  animate();
}

/* -- Brain open/close ------------------------------------------------- */
$('#brain-thumb').addEventListener('click',function(){
  var thumb=$('#brain-thumb');
  if(thumb.classList.contains('expanded')){
    thumb.classList.remove('expanded');
    brainModalOpen=false;
  }else{
    thumb.classList.add('expanded');
    brainModalOpen=true;
  }
});
$('#brain-thumb').addEventListener('dblclick',function(){
  /* Double-click opens the full modal for detailed inspection */
  brainModalOpen=true;
  $('#brain-modal-overlay').classList.add('open');
});
$('#brain-modal-close').addEventListener('click',function(){
  brainModalOpen=false;$('#brain-modal-overlay').classList.remove('open');
});
$('#brain-modal-overlay').addEventListener('click',function(e){
  if(e.target===e.currentTarget){brainModalOpen=false;e.currentTarget.classList.remove('open')}
});

/* -- Light region (3D) ------------------------------------------------ */
function lightRegion(regionKey,duration){
  var mesh=brainMeshes[regionKey];
  if(!mesh)return;
  var mat=mesh.material;
  mat.opacity=0.85;
  mat.emissiveIntensity=0.4;
  setTimeout(function(){mat.opacity=0.15;mat.emissiveIntensity=0.05},duration||1600);
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

/* =====================================================================
   Pixelated Face — Full GoEmotions Expressions
   Each of the 27 GoEmotions gets a unique color scheme.
   Eye and mouth shapes vary per emotion. Some share shapes but differ
   in color — the expandable emotion tracker disambiguates.
   ===================================================================== */
var currentEmotion='neutral';
var currentMixed=[];
var faceGrid=[
  [0,0,0,1,1,1,1,0,0,0],
  [0,0,1,1,1,1,1,1,0,0],
  [0,1,1,1,1,1,1,1,1,0],
  [0,1,1,1,1,1,1,1,1,0],
  [0,1,1,1,1,1,1,1,1,0],
  [0,1,1,1,1,1,1,1,1,0],
  [0,1,1,1,1,1,1,1,1,0],
  [0,0,1,1,1,1,1,1,0,0],
  [0,0,0,1,1,1,1,0,0,0],
  [0,0,0,0,0,0,0,0,0,0]
];

function drawPixelFace(emotion){
  var cv=$('#pixel-face');if(!cv)return;
  var ctx=cv.getContext('2d');
  var s=8;
  /* Color schemes: bg, face, eye, mouth, extra (blush/tear/etc) */
  var schemes={
    /* ── Positive ────────────────────────────────────── */
    admiration:    {bg:'#18150a',face:'#3a3218',eye:'#ffd700',mouth:'#e6be00',x:'#ff980040'},
    amusement:     {bg:'#1a1505',face:'#3a3010',eye:'#ffb300',mouth:'#ff9800',x:'#ffcc0040'},
    approval:      {bg:'#0a1a0e',face:'#1a3520',eye:'#66bb6a',mouth:'#4caf50',x:'#81c78440'},
    caring:        {bg:'#1a0a12',face:'#351a28',eye:'#f48fb1',mouth:'#e91e63',x:'#f8bbd040'},
    desire:        {bg:'#1a0e08',face:'#351c10',eye:'#ff7043',mouth:'#e64a19',x:'#ffab9140'},
    excitement:    {bg:'#1a0a05',face:'#351808',eye:'#ff5722',mouth:'#e64a19',x:'#ff8a6540'},
    gratitude:     {bg:'#1a1800',face:'#3a3520',eye:'#ffd700',mouth:'#ffcc00',x:'#ffe08240'},
    joy:           {bg:'#081a08',face:'#1a351a',eye:'#00e676',mouth:'#00c853',x:'#69f0ae40'},
    love:          {bg:'#1a0510',face:'#35081f',eye:'#e91e63',mouth:'#c2185b',x:'#f48fb140'},
    optimism:      {bg:'#1a1a05',face:'#353508',eye:'#ffeb3b',mouth:'#fdd835',x:'#fff17640'},
    pride:         {bg:'#12081a',face:'#281035',eye:'#7e57c2',mouth:'#673ab7',x:'#b39ddb40'},
    relief:        {bg:'#0a1a10',face:'#1a3520',eye:'#81c784',mouth:'#66bb6a',x:'#a5d6a740'},
    /* ── Negative ────────────────────────────────────── */
    anger:         {bg:'#1a0505',face:'#350808',eye:'#f44336',mouth:'#d32f2f',x:'#ef535040'},
    annoyance:     {bg:'#1a0a0a',face:'#351515',eye:'#ef5350',mouth:'#e53935',x:'#ef9a9a40'},
    disappointment:{bg:'#10100e',face:'#252520',eye:'#9e9e9e',mouth:'#757575',x:'#bdbdbd40'},
    disapproval:   {bg:'#0e1012',face:'#1e2228',eye:'#78909c',mouth:'#546e7a',x:'#b0bec540'},
    disgust:       {bg:'#120e08',face:'#281c10',eye:'#795548',mouth:'#5d4037',x:'#a1887f40'},
    embarrassment: {bg:'#150a18',face:'#2a1535',eye:'#ce93d8',mouth:'#ab47bc',x:'#f48fb180'},
    fear:          {bg:'#1a0508',face:'#350810',eye:'#ff1744',mouth:'#d50000',x:'#ff8a8040'},
    grief:         {bg:'#08081a',face:'#101035',eye:'#5c6bc0',mouth:'#3949ab',x:'#7986cb40'},
    nervousness:   {bg:'#1a1208',face:'#352510',eye:'#ffb74d',mouth:'#ff9800',x:'#ffe0b240'},
    remorse:       {bg:'#150a18',face:'#2a1535',eye:'#ce93d8',mouth:'#b070c0',x:'#e1bee740'},
    sadness:       {bg:'#081018',face:'#152030',eye:'#42a5f5',mouth:'#1e88e5',x:'#90caf940'},
    /* ── Ambiguous ───────────────────────────────────── */
    confusion:     {bg:'#0e1012',face:'#1e2228',eye:'#b0bec5',mouth:'#90a4ae',x:'#cfd8dc40'},
    curiosity:     {bg:'#0a1a1a',face:'#1a3535',eye:'#26c6da',mouth:'#00acc1',x:'#80deea40'},
    realization:   {bg:'#1a1a08',face:'#353510',eye:'#ffee58',mouth:'#fdd835',x:'#fff59d40'},
    surprise:      {bg:'#1a1208',face:'#352510',eye:'#ff9100',mouth:'#e65100',x:'#ffcc8040'},
    neutral:       {bg:'#0d1117',face:'#2a2e3a',eye:'#8899aa',mouth:'#6a7080',x:'#00000000'},
    /* ── Legacy compat ───────────────────────────────── */
    alarm:         {bg:'#1a0505',face:'#350808',eye:'#ff1744',mouth:'#d50000',x:'#ff8a8040'},
    urgency:       {bg:'#1a0a05',face:'#351808',eye:'#ff5722',mouth:'#e64a19',x:'#ff8a6540'},
    distress:      {bg:'#150a18',face:'#2a1535',eye:'#ce93d8',mouth:'#b070c0',x:'#e1bee740'},
    regret:        {bg:'#150a18',face:'#2a1535',eye:'#ce93d8',mouth:'#b070c0',x:'#e1bee740'},
    belonging:     {bg:'#081018',face:'#152030',eye:'#4fc3f7',mouth:'#3aa0d0',x:'#b3e5fc40'},
    creative:      {bg:'#12081a',face:'#281035',eye:'#7e57c2',mouth:'#673ab7',x:'#b39ddb40'}
  };
  var sc=schemes[emotion]||schemes.neutral;

  /* Clear + draw face base */
  ctx.globalAlpha=1;
  ctx.fillStyle=sc.bg;ctx.fillRect(0,0,80,80);
  ctx.fillStyle=sc.face;
  for(var r=0;r<10;r++)for(var c=0;c<10;c++)if(faceGrid[r][c])ctx.fillRect(c*s,r*s,s,s);

  /* ── Eyes ─────────────────────────────────────────── */
  ctx.fillStyle=sc.eye;
  var ey=emotion;
  /* Wide eyes (fear, alarm, surprise, excitement, realization) */
  if(ey==='fear'||ey==='alarm'||ey==='surprise'||ey==='excitement'||ey==='realization'){
    ctx.fillRect(2*s,3*s,2*s,2*s);ctx.fillRect(6*s,3*s,2*s,2*s);
  }
  /* Raised brow / asymmetric (curiosity, confusion, amusement) */
  else if(ey==='curiosity'||ey==='confusion'){
    ctx.fillRect(2*s,2*s,2*s,s);ctx.fillRect(2*s,3*s,2*s,2*s);ctx.fillRect(6*s,3*s,s,s);
  }
  else if(ey==='amusement'){
    ctx.fillRect(2*s,3*s,s,s);ctx.fillRect(3*s,2*s,s,s);ctx.fillRect(6*s,3*s,s,s);ctx.fillRect(7*s,2*s,s,s);
  }
  /* Droopy / sad eyes (sadness, grief, disappointment, remorse, regret, distress) */
  else if(ey==='sadness'||ey==='grief'||ey==='disappointment'||ey==='remorse'||ey==='regret'||ey==='distress'){
    ctx.fillRect(2*s,4*s,2*s,s);ctx.fillRect(6*s,4*s,2*s,s);
  }
  /* Squinting / narrow (anger, annoyance, disapproval, disgust) */
  else if(ey==='anger'||ey==='annoyance'||ey==='disapproval'||ey==='disgust'){
    ctx.fillRect(2*s,3*s,s,s);ctx.fillRect(3*s,4*s,s,s);ctx.fillRect(6*s,4*s,s,s);ctx.fillRect(7*s,3*s,s,s);
  }
  /* Soft / glowing eyes (love, caring, belonging, relief) */
  else if(ey==='love'||ey==='caring'||ey==='belonging'||ey==='relief'){
    ctx.fillRect(3*s,3*s,s,s);ctx.fillRect(6*s,3*s,s,s);
    ctx.globalAlpha=0.3;ctx.fillRect(2*s,3*s,s,s);ctx.fillRect(7*s,3*s,s,s);ctx.globalAlpha=1;
  }
  /* Nervous flickering eyes (nervousness, embarrassment) */
  else if(ey==='nervousness'||ey==='embarrassment'){
    ctx.fillRect(3*s,3*s,s,s);ctx.fillRect(7*s,3*s,s,s);
  }
  /* Starry eyes (admiration, pride, optimism, gratitude, desire) */
  else if(ey==='admiration'||ey==='pride'||ey==='optimism'||ey==='gratitude'||ey==='desire'){
    ctx.fillRect(2*s,3*s,2*s,s);ctx.fillRect(6*s,3*s,2*s,s);
    ctx.globalAlpha=0.4;ctx.fillRect(2*s,2*s,s,s);ctx.fillRect(7*s,2*s,s,s);ctx.globalAlpha=1;
  }
  /* Default normal eyes (neutral, approval, joy) */
  else{
    ctx.fillRect(3*s,3*s,s,s);ctx.fillRect(6*s,3*s,s,s);
  }

  /* ── Mouth ────────────────────────────────────────── */
  ctx.fillStyle=sc.mouth;
  /* Big smile (joy, excitement, amusement, love) */
  if(ey==='joy'||ey==='excitement'||ey==='amusement'||ey==='love'){
    ctx.fillRect(3*s,6*s,s,s);ctx.fillRect(4*s,7*s,2*s,s);ctx.fillRect(6*s,6*s,s,s);
  }
  /* Gentle smile (gratitude, admiration, approval, caring, pride, optimism, relief, belonging) */
  else if(ey==='gratitude'||ey==='admiration'||ey==='approval'||ey==='caring'||
          ey==='pride'||ey==='optimism'||ey==='relief'||ey==='belonging'){
    ctx.fillRect(3*s,6*s,4*s,s);ctx.fillRect(4*s,7*s,2*s,s);
  }
  /* O mouth — shock (fear, alarm, surprise, realization) */
  else if(ey==='fear'||ey==='alarm'||ey==='surprise'||ey==='realization'){
    ctx.fillRect(4*s,6*s,2*s,2*s);
  }
  /* Frown (sadness, grief, disappointment, remorse, regret, distress) */
  else if(ey==='sadness'||ey==='grief'||ey==='disappointment'||ey==='remorse'||ey==='regret'||ey==='distress'){
    ctx.fillRect(3*s,7*s,s,s);ctx.fillRect(4*s,6*s,2*s,s);ctx.fillRect(6*s,7*s,s,s);
  }
  /* Gritted / tight (anger, disgust) */
  else if(ey==='anger'||ey==='disgust'){
    ctx.fillRect(3*s,6*s,s,s);ctx.fillRect(4*s,7*s,s,s);ctx.fillRect(5*s,7*s,s,s);ctx.fillRect(6*s,6*s,s,s);
  }
  /* Flat / tense line (annoyance, disapproval, nervousness) */
  else if(ey==='annoyance'||ey==='disapproval'||ey==='nervousness'){
    ctx.fillRect(3*s,7*s,4*s,s);
  }
  /* Small O (confusion, embarrassment) */
  else if(ey==='confusion'||ey==='embarrassment'){
    ctx.fillRect(4*s,6*s,2*s,s);
  }
  /* Small opening (curiosity, desire) */
  else if(ey==='curiosity'||ey==='desire'){
    ctx.fillRect(4*s,6*s,s,s);ctx.fillRect(5*s,7*s,s,s);
  }
  /* Neutral straight line */
  else{
    ctx.fillRect(3*s,6*s,4*s,s);
  }

  /* ── Extra detail — blush, tears, sparkle ────────── */
  if(sc.x&&sc.x!=='#00000000'){
    ctx.fillStyle=sc.x;
    /* Blush cheeks (embarrassment, love, caring, nervousness) */
    if(ey==='embarrassment'||ey==='love'||ey==='caring'||ey==='nervousness'){
      ctx.fillRect(1*s,5*s,2*s,s);ctx.fillRect(7*s,5*s,2*s,s);
    }
    /* Tear drops (grief, sadness) */
    else if(ey==='grief'||ey==='sadness'){
      ctx.globalAlpha=0.5;ctx.fillRect(4*s,5*s,s,s);ctx.fillRect(6*s,5*s,s,s);ctx.globalAlpha=1;
    }
    /* Sparkle above (admiration, realization, gratitude, pride) */
    else if(ey==='admiration'||ey==='realization'||ey==='gratitude'||ey==='pride'){
      ctx.globalAlpha=0.4;ctx.fillRect(1*s,1*s,s,s);ctx.fillRect(8*s,1*s,s,s);ctx.globalAlpha=1;
    }
    /* Sweat drop (fear, alarm, nervousness) */
    else if(ey==='fear'||ey==='alarm'){
      ctx.globalAlpha=0.5;ctx.fillRect(8*s,2*s,s,s);ctx.globalAlpha=1;
    }
  }

  $('#emo-face-label').textContent=emotion;
}

/* =====================================================================
   Emotion Tracker — GoEmotions with Mixed Emotions
   ===================================================================== */
var emotionHistory=[];
var emotionExpanded=false;

function updateEmotionTracker(emotion,confidence,mixedEmotions){
  var entry={emotion:emotion,confidence:confidence,timestamp:new Date().toLocaleTimeString(),mixed:mixedEmotions||[]};
  emotionHistory.unshift(entry);
  if(emotionHistory.length>30)emotionHistory.length=30;
  currentEmotion=emotion;
  currentMixed=mixedEmotions||[];
  drawPixelFace(emotion);
  renderEmotionTracker();
}

function renderEmotionTracker(){
  var emojiMap={
    admiration:'\uD83E\uDD29',amusement:'\uD83D\uDE04',approval:'\uD83D\uDC4D',caring:'\uD83E\uDD7A',
    desire:'\uD83D\uDE0D',excitement:'\uD83E\uDD29',gratitude:'\uD83D\uDE4F',joy:'\uD83D\uDE0A',
    love:'\u2764\uFE0F',optimism:'\uD83C\uDF1F',pride:'\uD83D\uDE0E',relief:'\uD83D\uDE0C',
    anger:'\uD83D\uDE21',annoyance:'\uD83D\uDE12',disappointment:'\uD83D\uDE1E',disapproval:'\uD83D\uDC4E',
    disgust:'\uD83E\uDD22',embarrassment:'\uD83D\uDE33',fear:'\uD83D\uDE28',grief:'\uD83D\uDE2D',
    nervousness:'\uD83D\uDE30',remorse:'\uD83D\uDE14',sadness:'\uD83D\uDE22',
    confusion:'\uD83D\uDE15',curiosity:'\uD83E\uDD14',realization:'\uD83D\uDCA1',surprise:'\uD83D\uDE32',
    neutral:'\uD83D\uDE10',alarm:'\uD83D\uDEA8',urgency:'\u26A0\uFE0F',distress:'\uD83D\uDE1F',
    regret:'\uD83D\uDE14',belonging:'\uD83D\uDC9C',creative:'\uD83C\uDFA8'
  };
  var emoji=emojiMap[currentEmotion]||emojiMap.neutral;
  $('#emo-emoji').textContent=emoji;
  $('#emo-name').textContent=currentEmotion;
  var badgeEl=$('#emo-badge');
  badgeEl.style.background=emotionColor(currentEmotion);
  badgeEl.textContent=currentEmotion;

  /* Show mixed emotions if more than 1 */
  var mw=$('#emo-mixed-wrap');
  var ml=$('#emo-mixed-list');
  if(currentMixed.length>1){
    mw.style.display='block';
    ml.innerHTML=currentMixed.map(function(m){
      return '<span class="emo-mixed-tag" style="border-color:'+emotionColor(m.emotion)+'">'+
        esc(m.emotion)+'<span class="tag-score">'+Math.round((m.score||0)*100)+'%</span></span>';
    }).join('');
  }else{
    mw.style.display='none';ml.innerHTML='';
  }

  /* Overall mood bar — use GoEmotions valence map */
  var recent=emotionHistory.slice(0,10);
  var avgVal=0.5;
  if(recent.length>0){
    var sum=0;
    for(var i=0;i<recent.length;i++) sum+=(GOEMO_VALENCE[recent[i].emotion]||0.5);
    avgVal=sum/recent.length;
  }
  $('#mood-bar-fill').style.width=(avgVal*100)+'%';

  if(emotionExpanded){
    var h='';
    for(var j=0;j<emotionHistory.length;j++){
      var e=emotionHistory[j];
      var mixStr='';
      if(e.mixed&&e.mixed.length>1){
        mixStr=' <span class="emo-hist-mixed">(+'+e.mixed.slice(1).map(function(m){return m.emotion}).join(', ')+')</span>';
      }
      h+='<div class="emo-hist-item">';
      h+='<span class="emo-hist-dot" style="background:'+emotionColor(e.emotion)+'"></span>';
      h+='<span class="emo-hist-ts">'+esc(e.timestamp)+'</span>';
      h+='<span class="emo-hist-name">'+esc(e.emotion)+mixStr+'</span>';
      h+='<span class="emo-hist-conf">'+(e.confidence?Math.round(e.confidence*100)+'%':'')+'</span>';
      h+='</div>';
    }
    $('#emo-history').innerHTML=h;
  }
}

$('#emo-expand-btn').addEventListener('click',function(){
  emotionExpanded=!emotionExpanded;
  var panel=$('#emotion-tracker');
  if(emotionExpanded){
    panel.classList.add('expanded');
    this.textContent='\u25B2 Hide';
    renderEmotionTracker();
  }else{
    panel.classList.remove('expanded');
    this.textContent='\u25BC History';
    $('#emo-history').innerHTML='';
  }
});

/* -- Chat / Terminal -------------------------------------------------- */
var busy=false;
var camStream=null,camEnabled=false,sttActive=false;
var teacherMode=false;

$('#chat-send').addEventListener('click',sendMessage);
$('#chat-input').addEventListener('keydown',function(e){if(e.key==='Enter'&&!busy)sendMessage()});

function toggleTeacherMode(){
  teacherMode=!teacherMode;
  var bar=$('#teacher-bar');
  var btn=$('#chat-send');
  var inp=$('#chat-input');
  if(teacherMode){
    bar.style.display='flex';
    btn.textContent='\uD83D\uDCDA Teach';
    btn.style.background='linear-gradient(135deg,var(--warning),#e6a800)';
    inp.placeholder='Teach the brain... (e.g. "that was curiosity, not alarm")';
    appendMsg('bot','<span style="color:var(--warning);font-weight:700">\uD83D\uDCDA Teacher Mode ON</span> — I\'m listening as your student. Tell me what I should feel, know, or correct. Type <code>/teach</code> to exit.');
  }else{
    bar.style.display='none';
    btn.textContent='Send';
    btn.style.background='';
    inp.placeholder='Talk to the brain... (type /teach to enter teacher mode)';
    appendMsg('bot','<span style="color:var(--text-dim)">\uD83D\uDCDA Teacher Mode OFF</span> — back to normal conversation.');
  }
}

function sendMessage(){
  var inp=$('#chat-input');
  var text=inp.value.trim();
  if(!text||busy)return;

  /* Slash commands */
  if(text.toLowerCase()==='/teach'){
    inp.value='';
    toggleTeacherMode();
    return;
  }
  if(text.toLowerCase()==='/learnings'||text.toLowerCase()==='/learned'){
    inp.value='';
    appendMsg('user',esc(text));
    fetch('/api/recent-learnings?session_id='+encodeURIComponent(SESSION))
      .then(function(r){return r.json()})
      .then(function(d){
        var html='<b>\uD83E\uDDE0 What I\'ve Learned</b><br>';
        html+=esc(d.summary||'No learnings yet.')+'<br><br>';
        if(d.emotions&&d.emotions.learned&&d.emotions.learned.length>0){
          html+='<b>New emotions:</b> '+d.emotions.learned.map(esc).join(', ')+'<br>';
        }
        if(d.working_memory&&d.working_memory.corrections&&d.working_memory.corrections.length>0){
          html+='<b>Emotion corrections:</b><br>';
          d.working_memory.corrections.forEach(function(c){
            html+='&bull; '+esc(c.detected_emotion||'')+' \u2192 '+esc(c.correct_emotion||'')+' for "'+esc((c.input||'').slice(0,40))+'"<br>';
          });
        }
        html+='<br><span style="color:var(--text-dim)">Memories: '+
          (d.hippocampus?d.hippocampus.total_episodes:0)+
          ' | Buffer: '+(d.learner?d.learner.replay_buffer_size:0)+
          ' | Training runs: '+(d.learner?d.learner.training_runs:0)+'</span>';
        appendMsg('bot',html);
      })
      .catch(function(e){appendMsg('bot','<span class="error-box">Error: '+esc(e.message)+'</span>')});
    return;
  }
  if(text.toLowerCase()==='/help'){
    inp.value='';
    appendMsg('bot',
      '<b>Commands:</b><br>'+
      '<code>/teach</code> — toggle teacher mode (correct emotions, teach knowledge)<br>'+
      '<code>/learnings</code> — show what the brain has learned<br>'+
      '<code>/help</code> — show this help'
    );
    return;
  }

  busy=true;inp.value='';$('#chat-send').disabled=true;
  var role=teacherMode?'user teacher':'user';
  appendMsg(role,esc(text));
  animateBrainPipeline([]);
  var t0=performance.now();
  var payload={user_input:text,session_id:SESSION,teacher_mode:teacherMode};
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
  var cls=role.split(' ')[0];  /* 'user' or 'bot' */
  div.className='msg '+cls;
  if(role.indexOf('teacher')>=0){
    div.style.borderLeft='3px solid var(--warning)';
    html='<span style="font-size:.68rem;color:var(--warning);font-weight:700;letter-spacing:.5px">\uD83D\uDCDA TEACHER</span><br>'+html;
  }
  div.innerHTML=html;
  log.appendChild(div);log.scrollTop=log.scrollHeight;
}

function appendBotMsg(d,latency){
  var emotion=d.emotion||d.dominant_emotion||'neutral';
  var urgency=d.urgency;if(urgency===undefined||urgency===null)urgency='';
  var confidence=d.confidence;if(confidence===undefined||confidence===null)confidence='';
  var path=d.cognitive_path||d.path||'';
  var text=d.response||d.reply||d.text||JSON.stringify(d);
  var mixedEmotions=d.emotions||[];
  var meta='<div class="meta">';
  meta+='<span class="badge-emotion" style="background:'+emotionColor(emotion)+'">'+esc(emotion)+'</span>';
  if(mixedEmotions.length>1){
    for(var mi=1;mi<Math.min(mixedEmotions.length,4);mi++){
      meta+='<span class="badge-emotion" style="background:'+emotionColor(mixedEmotions[mi].emotion)+';opacity:.7">'+esc(mixedEmotions[mi].emotion)+'</span>';
    }
  }
  if(urgency!=='')meta+='<span class="badge-urgency" style="background:'+urgencyColor(urgency)+'">urgency '+urgency+'</span>';
  if(confidence!=='')meta+='<span class="badge-confidence">\u2691 '+Math.round(confidence*100)+'%</span>';
  meta+='<span class="badge-latency">\u23F1 '+latency+'s</span>';
  if(d.execution_ms)meta+='<span class="badge-latency" style="color:var(--text-dim)">\u2699 '+Math.round(d.execution_ms)+'ms server</span>';
  if(path)meta+='<span class="badge-path">\u21A3 '+esc(Array.isArray(path)?path.join(' \u2192 '):path)+'</span>';
  if(d.teacher_mode)meta+='<span style="padding:2px 8px;border-radius:12px;font-weight:600;background:rgba(255,214,0,.15);color:var(--warning);border:1px solid var(--warning)">\uD83D\uDCDA teacher</span>';
  meta+='</div>';

  /* Trace with timing waterfall */
  var trace='';
  if(d.trace){
    trace='<span class="trace-toggle" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'block\'?\'none\':\'block\'">';
    trace+='\u25B6 trace</span>';
    /* Build timing waterfall if step_times_ms exists */
    var traceContent='';
    var st=d.trace.step_times_ms;
    if(st){
      var totalMs=d.execution_ms||0;
      traceContent+='<div style="margin-bottom:8px;font-size:.72rem;font-family:var(--mono)">';
      traceContent+='<div style="font-weight:700;margin-bottom:4px;color:var(--accent)">Step Timing (ms)</div>';
      var steps=Object.keys(st);
      for(var si=0;si<steps.length;si++){
        var sn=steps[si],sv=st[sn];
        var pct=totalMs>0?Math.min(100,Math.round(sv/totalMs*100)):0;
        var barColor=sv>5000?'var(--error)':sv>2000?'var(--warning)':'var(--accent)';
        traceContent+='<div style="display:flex;align-items:center;gap:6px;margin:2px 0">';
        traceContent+='<span style="width:140px;color:var(--text-dim);text-align:right;flex-shrink:0">'+esc(sn)+'</span>';
        traceContent+='<div style="flex:1;height:10px;background:var(--border);border-radius:3px;overflow:hidden">';
        traceContent+='<div style="width:'+pct+'%;height:100%;background:'+barColor+';border-radius:3px;transition:width .3s"></div></div>';
        traceContent+='<span style="width:60px;text-align:right;color:'+(sv>5000?'var(--error)':'var(--text)')+'">'+Math.round(sv)+'</span>';
        traceContent+='</div>';
      }
      traceContent+='<div style="margin-top:4px;color:var(--text-dim)">Total: '+Math.round(totalMs)+'ms</div>';
      traceContent+='</div>';
    }
    /* Show errors if any */
    if(d.trace.errors&&d.trace.errors.length>0){
      traceContent+='<div style="margin-bottom:8px;font-size:.72rem">';
      traceContent+='<div style="font-weight:700;color:var(--error);margin-bottom:4px">\u26A0 Errors ('+d.trace.errors.length+')</div>';
      for(var ei=0;ei<d.trace.errors.length;ei++){
        var err=d.trace.errors[ei];
        traceContent+='<div style="color:var(--error);padding:2px 0">'+esc(err.step)+': '+esc(String(err.error).slice(0,120))+'</div>';
      }
      traceContent+='</div>';
    }
    /* Show emotion teaching result */
    if(d.trace.emotion_taught){
      traceContent+='<div style="margin-bottom:8px;font-size:.72rem;color:var(--warning)">';
      traceContent+='\uD83D\uDCDA Emotion corrected: '+esc(d.trace.emotion_taught.from)+' \u2192 '+esc(d.trace.emotion_taught.to);
      traceContent+='</div>';
    }
    /* Show DB operations */
    var dbOps=d.trace.db_operations;
    if(dbOps&&dbOps.length>0){
      traceContent+='<div style="margin-bottom:8px;font-size:.72rem">';
      traceContent+='<div style="font-weight:700;color:var(--accent);margin-bottom:4px">\uD83D\uDCBE DB Operations</div>';
      for(var di=0;di<dbOps.length;di++){
        var op=dbOps[di];
        var okColor=op.ok?'var(--success)':'var(--error)';
        var okIcon=op.ok?'\u2713':'\u2717';
        traceContent+='<div style="color:'+okColor+';padding:1px 0">'+okIcon+' '+esc(op.op);
        if(op.error)traceContent+=' — <span style="color:var(--error)">'+esc(String(op.error).slice(0,80))+'</span>';
        traceContent+='</div>';
      }
      traceContent+='</div>';
    }
    traceContent+='<pre style="margin:0;font-size:.68rem">'+esc(JSON.stringify(d.trace,null,2))+'</pre>';
    trace+='<div class="trace" style="display:none">'+traceContent+'</div>';
  }
  appendMsg('bot',esc(text)+meta+trace);
  updateEmotionTracker(emotion,confidence||null,mixedEmotions);
}

/* -- Speech-to-Text + Audio Waveform ---------------------------------- */
var SpeechRec=window.SpeechRecognition||window.webkitSpeechRecognition;
var audioCtx=null,micAnalyser=null,micSource=null,micStream=null,waveAnimId=null;

function startWaveform(stream){
  if(!audioCtx) audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  micSource=audioCtx.createMediaStreamSource(stream);
  micAnalyser=audioCtx.createAnalyser();
  micAnalyser.fftSize=256;
  micSource.connect(micAnalyser);
  var canvas=$('#waveform-canvas');
  var ctx=canvas.getContext('2d');
  var bufLen=micAnalyser.frequencyBinCount;
  var data=new Uint8Array(bufLen);
  $('#waveform-wrap').classList.add('active');
  $('#media-label').textContent='Listening...';
  function draw(){
    waveAnimId=requestAnimationFrame(draw);
    micAnalyser.getByteTimeDomainData(data);
    var w=canvas.width,h=canvas.height;
    ctx.fillStyle='rgba(0,20,8,0.3)';ctx.fillRect(0,0,w,h);
    ctx.lineWidth=2;ctx.strokeStyle='#00c853';
    ctx.shadowColor='#00e676';ctx.shadowBlur=6;
    ctx.beginPath();
    var sliceW=w/bufLen;var x=0;
    for(var i=0;i<bufLen;i++){
      var v=data[i]/128.0;var y=v*h/2;
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
      x+=sliceW;
    }
    ctx.lineTo(w,h/2);ctx.stroke();
    ctx.shadowBlur=0;
    /* Draw equalizer bars at bottom */
    micAnalyser.getByteFrequencyData(data);
    var barW=Math.max(2,w/64-1);
    ctx.globalAlpha=0.4;
    for(var b=0;b<64;b++){
      var bh=(data[b]/255)*h*0.4;
      var hue=120-(data[b]/255)*30;
      ctx.fillStyle='hsl('+hue+',80%,50%)';
      ctx.fillRect(b*(barW+1),h-bh,barW,bh);
    }
    ctx.globalAlpha=1;
  }
  draw();
}

function stopWaveform(){
  if(waveAnimId){cancelAnimationFrame(waveAnimId);waveAnimId=null}
  if(micSource){micSource.disconnect();micSource=null}
  if(micStream){micStream.getTracks().forEach(function(t){t.stop()});micStream=null}
  micAnalyser=null;
  $('#waveform-wrap').classList.remove('active');
  $('#media-label').textContent='';
  var canvas=$('#waveform-canvas');
  var ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
}

if(SpeechRec){
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
    stopWaveform();
    var v=$('#chat-input').value.trim();
    if(v&&!busy)sendMessage();
  };
  recognition.onerror=function(){sttActive=false;$('#btn-stt').classList.remove('active-media');stopWaveform()};
  $('#btn-stt').addEventListener('click',function(){
    if(sttActive){
      recognition.stop();sttActive=false;$('#btn-stt').classList.remove('active-media');
      stopWaveform();
    }else{
      navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
        micStream=stream;
        startWaveform(stream);
        recognition.start();sttActive=true;$('#btn-stt').classList.add('active-media');
      }).catch(function(e){console.warn('Mic denied',e)});
    }
  });
}

/* -- Webcam ----------------------------------------------------------- */
if(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia){
  var vid=$('#cam-preview');
  $('#btn-cam').addEventListener('click',function(){
    if(camEnabled){
      camEnabled=false;
      if(camStream){camStream.getTracks().forEach(function(t){t.stop()});camStream=null}
      vid.style.display='none';vid.srcObject=null;
      $('#btn-cam').classList.remove('active-media');$('#btn-cam').classList.remove('cam-active');
    }else{
      navigator.mediaDevices.getUserMedia({video:true}).then(function(stream){
        camStream=stream;vid.srcObject=stream;vid.style.display='block';
        camEnabled=true;$('#btn-cam').classList.add('active-media');$('#btn-cam').classList.add('cam-active');
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
      return '<div class="bar" style="height:'+h+'%"></div>';
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
  c.innerHTML=names.map(function(n){return '<button onclick="applyPreset(\''+esc(n)+'\')">' +esc(n)+'</button>'}).join('')+
    '<button onclick="resetKnobs()">\u21BA Reset</button>';
}
function applyPreset(name){fetch('/api/knobs/preset/'+encodeURIComponent(name),{method:'POST'}).then(function(){loadKnobs()})}
function resetKnobs(){fetch('/api/knobs/reset',{method:'POST'}).then(function(){loadKnobs()})}
function renderKnobs(knobs){
  var c=$('#knob-groups');
  c.innerHTML=Object.keys(knobGroups).map(function(group){
    var keys=knobGroups[group];
    var dials=keys.filter(function(k){return knobs[k]!==undefined}).map(function(k){
      var v=Number(knobs[k]);
      return buildRotaryKnob(k,v);
    }).join('');
    if(!dials)return '';
    return '<h4>'+esc(group)+'</h4><div class="knob-grid">'+dials+'</div>';
  }).join('');
  initRotaryKnobs();
}

/* ---- Rotary Knob SVG builder ---------------------------------------- */
function buildRotaryKnob(name,val){
  var pct=Math.max(0,Math.min(1,val));
  var id='rk-'+name.replace(/[^a-zA-Z0-9]/g,'_');
  return '<div class="rotary-knob" data-knob="'+esc(name)+'" data-value="'+pct+'" id="'+id+'">'
    +'<div class="dial-wrap">'
    +'<svg class="dial" viewBox="0 0 64 64">'
    +'<circle class="dial-track" cx="32" cy="32" r="26" stroke-dasharray="122.52" stroke-dashoffset="0" '
    +'transform="rotate(135 32 32)"></circle>'
    +'<circle class="dial-fill" cx="32" cy="32" r="26" stroke-dasharray="122.52" '
    +'stroke-dashoffset="'+(122.52 - pct*122.52).toFixed(2)+'" transform="rotate(135 32 32)"></circle>'
    +'<circle class="dial-dot" cx="32" cy="8" r="3" transform="rotate('+(135+pct*270)+' 32 32)"></circle>'
    +'</svg>'
    +'<div class="dial-val">'+pct.toFixed(2)+'</div>'
    +'</div>'
    +'<div class="dial-label">'+esc(name.replace(/_/g,' '))+'</div>'
    +'</div>';
}

/* ---- Make rotary knobs interactive via mouse drag ------------------- */
function initRotaryKnobs(){
  document.querySelectorAll('.rotary-knob').forEach(function(el){
    var svg=el.querySelector('svg.dial');
    if(!svg)return;
    var dragging=false, startY=0, startVal=0;

    function onDown(e){
      e.preventDefault();
      dragging=true;
      startY=(e.touches?e.touches[0].clientY:e.clientY);
      startVal=parseFloat(el.dataset.value)||0;
      document.addEventListener('mousemove',onMove);
      document.addEventListener('mouseup',onUp);
      document.addEventListener('touchmove',onMove,{passive:false});
      document.addEventListener('touchend',onUp);
    }
    function onMove(e){
      if(!dragging)return;
      e.preventDefault();
      var curY=(e.touches?e.touches[0].clientY:e.clientY);
      var delta=(startY-curY)/150;
      var nv=Math.max(0,Math.min(1,startVal+delta));
      updateDial(el,nv);
    }
    function onUp(){
      if(!dragging)return;
      dragging=false;
      document.removeEventListener('mousemove',onMove);
      document.removeEventListener('mouseup',onUp);
      document.removeEventListener('touchmove',onMove);
      document.removeEventListener('touchend',onUp);
      sendKnobValue(el.dataset.knob, parseFloat(el.dataset.value));
    }
    svg.addEventListener('mousedown',onDown);
    svg.addEventListener('touchstart',onDown,{passive:false});

    svg.addEventListener('wheel',function(e){
      e.preventDefault();
      var cur=parseFloat(el.dataset.value)||0;
      var step=e.deltaY>0?-0.02:0.02;
      var nv=Math.max(0,Math.min(1,cur+step));
      updateDial(el,nv);
      clearTimeout(el._wheelTimer);
      el._wheelTimer=setTimeout(function(){
        sendKnobValue(el.dataset.knob,parseFloat(el.dataset.value));
      },400);
    },{passive:false});
  });
}

function updateDial(el,val){
  val=Math.max(0,Math.min(1,val));
  el.dataset.value=val;
  var fill=el.querySelector('.dial-fill');
  var dot=el.querySelector('.dial-dot');
  var valEl=el.querySelector('.dial-val');
  if(fill)fill.setAttribute('stroke-dashoffset',(122.52-val*122.52).toFixed(2));
  if(dot)dot.setAttribute('transform','rotate('+(135+val*270)+' 32 32)');
  if(valEl)valEl.textContent=val.toFixed(2);
}

var _knobSendTimer={};
function sendKnobValue(name,val){
  clearTimeout(_knobSendTimer[name]);
  _knobSendTimer[name]=setTimeout(function(){
    fetch('/api/knobs/adjust?knob_name='+encodeURIComponent(name)+'&value='+val.toFixed(3),{method:'PATCH'});
  },250);
}
/* ---- Legacy compat -------------------------------------------------- */
function adjustKnob(el){
  el.nextElementSibling.textContent=Number(el.value).toFixed(2);
  sendKnobValue(el.dataset.knob,Number(el.value));
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
initBrainScene();
initThumbRenderer();
initModalRenderer();
drawPixelFace('neutral');

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
