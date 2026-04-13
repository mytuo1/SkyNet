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
        return JSONResponse({"error": "Service unreachable"}, status_code=502)
    except Exception:
        return JSONResponse({"error": "Internal proxy error"}, status_code=500)


async def _proxy_post(url: str, body: dict) -> JSONResponse:
    try:
        r = await http_client.post(url, json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable"}, status_code=502)
    except Exception:
        return JSONResponse({"error": "Internal proxy error"}, status_code=500)


async def _proxy_patch(url: str, params: dict | None = None, body: dict | None = None) -> JSONResponse:
    try:
        r = await http_client.patch(url, params=params, json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        return JSONResponse({"error": "Service unreachable"}, status_code=502)
    except Exception:
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

#app{display:grid;grid-template-rows:auto auto 1fr;height:100vh;overflow:hidden}
.header{display:flex;align-items:center;gap:12px;padding:10px 20px;
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

.nav{display:flex;gap:2px;padding:0 20px;background:var(--glass-bg);
  backdrop-filter:blur(var(--glass-blur));border-bottom:1px solid var(--border)}
.nav-btn{padding:9px 20px;font-size:.82rem;font-weight:600;color:var(--text-dim);cursor:pointer;
  border:none;background:none;border-bottom:2px solid transparent;transition:all .25s ease;letter-spacing:.3px}
.nav-btn:hover{color:var(--text);text-shadow:0 0 8px var(--glow-accent)}
.nav-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

.main{overflow:hidden;position:relative}
.page{display:none;position:absolute;inset:0;overflow-y:auto;padding:16px 20px;opacity:0;transition:opacity .3s ease}
.page.active{display:flex;flex-direction:column;opacity:1}

/* == Emotion Panel ==================================================== */
.emotion-panel{position:absolute;top:12px;left:12px;width:200px;z-index:10;
  background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:var(--radius);padding:12px;
  transition:all .3s ease;box-shadow:0 4px 20px rgba(0,0,0,.3)}
.emotion-panel:hover{border-color:rgba(0,212,255,.25);box-shadow:0 4px 20px rgba(0,0,0,.3),0 0 15px var(--glow-accent)}
.emotion-panel.expanded{width:320px;max-height:400px;overflow-y:auto}
.emo-face-wrap{text-align:center;margin-bottom:8px}
.emo-face-wrap canvas{border-radius:6px;image-rendering:pixelated}
.emo-face-label{font-size:.7rem;color:var(--text-dim);margin-top:4px;letter-spacing:.5px;text-transform:uppercase}
.emo-current{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.emo-current .emo-emoji{font-size:1.5rem}
.emo-current .emo-name{font-weight:700;font-size:.85rem;letter-spacing:.3px}
.emo-badge{font-size:.68rem;padding:2px 8px;border-radius:10px;font-weight:700;color:#111}
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

/* == 3D Brain ========================================================= */
.brain-thumb{position:fixed;top:54px;right:12px;width:180px;height:140px;z-index:30;
  cursor:pointer;background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--border);border-radius:var(--radius);
  padding:4px;transition:all .25s ease;overflow:hidden}
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
#page-terminal{gap:12px;position:relative}
.chat-log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px;
  padding:10px;padding-left:224px;background:var(--glass-bg);backdrop-filter:blur(8px);
  border-radius:var(--radius);border:1px solid var(--border);min-height:0;transition:all .25s ease}
.chat-log:hover{border-color:rgba(0,212,255,.12)}
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
.chat-controls{display:flex;gap:8px;align-items:center}
.chat-controls input[type=text]{flex:1;padding:10px 14px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--glass-bg);backdrop-filter:blur(8px);color:var(--text);font-size:.9rem;outline:none;
  font-family:var(--font);transition:all .25s ease}
.chat-controls input:focus{border-color:var(--accent);box-shadow:0 0 10px var(--glow-accent)}
.chat-btn{padding:10px 20px;border:none;border-radius:var(--radius);
  background:linear-gradient(135deg,var(--accent),#0097b8);color:#0a0e17;font-weight:700;
  cursor:pointer;font-size:.88rem;transition:all .2s ease;letter-spacing:.3px}
.chat-btn:hover{background:linear-gradient(135deg,#33dfff,var(--accent));box-shadow:0 0 15px var(--glow-accent)}
.chat-btn:disabled{opacity:.5;cursor:not-allowed}
.media-btn{width:38px;height:38px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--glass-bg);color:var(--text-dim);cursor:pointer;font-size:1.1rem;
  display:flex;align-items:center;justify-content:center;transition:all .2s ease;position:relative}
.media-btn:hover{border-color:var(--accent);color:var(--accent);box-shadow:0 0 10px var(--glow-accent)}
.media-btn.active-media{border-color:var(--error);color:var(--error)}
.rec-dot{position:absolute;top:3px;right:3px;width:6px;height:6px;border-radius:50%;
  background:var(--error);display:none;animation:pulse 1s infinite}
.media-btn.active-media .rec-dot{display:block}
.webcam-preview{width:48px;height:36px;border-radius:4px;border:1px solid var(--border);
  object-fit:cover;display:none}

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
    <div class="emo-mood">
      <div class="emo-mood-label">Overall Mood</div>
      <div class="emo-mood-bar"><div class="emo-mood-bar-fill" id="mood-bar-fill" style="width:50%"></div></div>
    </div>
    <button class="emo-expand-btn" id="emo-expand-btn">&#x25BC; History</button>
    <div class="emo-history" id="emo-history"></div>
  </div>
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
   Pixelated Face
   ===================================================================== */
var currentEmotion='neutral';
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
  var schemes={
    neutral:{bg:'#0d1117',face:'#2a2e3a',eye:'#8899aa',mouth:'#6a7080'},
    joy:{bg:'#1a1800',face:'#3a3520',eye:'#ffd700',mouth:'#ffcc00'},
    gratitude:{bg:'#1a1800',face:'#3a3520',eye:'#ffd700',mouth:'#ffcc00'},
    curiosity:{bg:'#0a1a1a',face:'#1a3535',eye:'#26c6da',mouth:'#20a0b0'},
    alarm:{bg:'#1a0808',face:'#3a1515',eye:'#ff4444',mouth:'#ff3333'},
    urgency:{bg:'#1a0808',face:'#3a1515',eye:'#ff4444',mouth:'#ff3333'},
    regret:{bg:'#150a18',face:'#2a1535',eye:'#ce93d8',mouth:'#b070c0'},
    distress:{bg:'#150a18',face:'#2a1535',eye:'#ce93d8',mouth:'#b070c0'},
    belonging:{bg:'#081018',face:'#152030',eye:'#4fc3f7',mouth:'#3aa0d0'}
  };
  var sc=schemes[emotion]||schemes.neutral;
  ctx.fillStyle=sc.bg;ctx.fillRect(0,0,80,80);
  ctx.fillStyle=sc.face;
  for(var r=0;r<10;r++)for(var c=0;c<10;c++)if(faceGrid[r][c])ctx.fillRect(c*s,r*s,s,s);
  ctx.fillStyle=sc.eye;
  if(emotion==='alarm'||emotion==='urgency'){
    ctx.fillRect(2*s,3*s,2*s,2*s);ctx.fillRect(6*s,3*s,2*s,2*s);
  }else if(emotion==='curiosity'){
    ctx.fillRect(2*s,2*s,2*s,s);ctx.fillRect(2*s,3*s,2*s,2*s);ctx.fillRect(6*s,3*s,s,s);
  }else if(emotion==='regret'||emotion==='distress'){
    ctx.fillRect(2*s,4*s,2*s,s);ctx.fillRect(6*s,4*s,2*s,s);
  }else if(emotion==='belonging'){
    ctx.fillRect(3*s,3*s,s,s);ctx.fillRect(6*s,3*s,s,s);
    ctx.globalAlpha=0.3;ctx.fillRect(2*s,3*s,s,s);ctx.fillRect(7*s,3*s,s,s);ctx.globalAlpha=1;
  }else{
    ctx.fillRect(3*s,3*s,s,s);ctx.fillRect(6*s,3*s,s,s);
  }
  ctx.fillStyle=sc.mouth;
  if(emotion==='joy'||emotion==='gratitude'){
    ctx.fillRect(3*s,6*s,s,s);ctx.fillRect(4*s,7*s,2*s,s);ctx.fillRect(6*s,6*s,s,s);
  }else if(emotion==='alarm'||emotion==='urgency'){
    ctx.fillRect(4*s,6*s,2*s,2*s);
  }else if(emotion==='regret'||emotion==='distress'){
    ctx.fillRect(3*s,7*s,s,s);ctx.fillRect(4*s,6*s,2*s,s);ctx.fillRect(6*s,7*s,s,s);
  }else if(emotion==='belonging'){
    ctx.fillRect(3*s,6*s,4*s,s);ctx.fillRect(4*s,7*s,2*s,s);
  }else if(emotion==='curiosity'){
    ctx.fillRect(4*s,6*s,2*s,s);
  }else{
    ctx.fillRect(3*s,6*s,4*s,s);
  }
  $('#emo-face-label').textContent=emotion;
}

/* =====================================================================
   Emotion Tracker
   ===================================================================== */
var emotionHistory=[];
var emotionExpanded=false;

function updateEmotionTracker(emotion,confidence){
  emotionHistory.unshift({emotion:emotion,confidence:confidence,timestamp:new Date().toLocaleTimeString()});
  if(emotionHistory.length>20)emotionHistory.length=20;
  currentEmotion=emotion;
  drawPixelFace(emotion);
  renderEmotionTracker();
}

function renderEmotionTracker(){
  var emojiMap={neutral:'\uD83D\uDE10',joy:'\uD83D\uDE0A',gratitude:'\uD83D\uDE4F',
    curiosity:'\uD83E\uDD14',alarm:'\uD83D\uDEA8',urgency:'\u26A0',
    regret:'\uD83D\uDE14',distress:'\uD83D\uDE1F',belonging:'\uD83D\uDC9C'};
  var emoji=emojiMap[currentEmotion]||emojiMap.neutral;
  $('#emo-emoji').textContent=emoji;
  $('#emo-name').textContent=currentEmotion;
  var badgeEl=$('#emo-badge');
  badgeEl.style.background=emotionColor(currentEmotion);
  badgeEl.textContent=currentEmotion;

  var valence={joy:0.9,gratitude:0.85,belonging:0.7,curiosity:0.6,
    neutral:0.5,alarm:0.3,urgency:0.25,regret:0.2,distress:0.15};
  var recent=emotionHistory.slice(0,10);
  var avgVal=0.5;
  if(recent.length>0){
    var sum=0;for(var i=0;i<recent.length;i++)sum+=(valence[recent[i].emotion]||0.5);
    avgVal=sum/recent.length;
  }
  $('#mood-bar-fill').style.width=(avgVal*100)+'%';

  if(emotionExpanded){
    var h='';
    for(var j=0;j<emotionHistory.length;j++){
      var e=emotionHistory[j];
      h+='<div class="emo-hist-item">';
      h+='<span class="emo-hist-dot" style="background:'+emotionColor(e.emotion)+'"></span>';
      h+='<span class="emo-hist-ts">'+esc(e.timestamp)+'</span>';
      h+='<span class="emo-hist-name">'+esc(e.emotion)+'</span>';
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
    trace='<span class="trace-toggle" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'block\'?\'none\':\'block\'">';
    trace+='\u25B6 trace</span>';
    trace+='<pre class="trace">'+esc(JSON.stringify(d.trace,null,2))+'</pre>';
  }
  appendMsg('bot',esc(text)+meta+trace);
  updateEmotionTracker(emotion,confidence||null);
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
