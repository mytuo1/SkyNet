"""
SkyNet Brain Dashboard — single-file FastAPI application.

Serves a fully-featured monitoring / control dashboard for the SkyNet
cognitive architecture.  The HTML, CSS and JS are embedded as a single
template string so the service can be deployed without any build step.

Panels
------
1. Terminal       — chat with the brain via thalamus /process
2. Service Monitor — live health grid for all 19 services
3. Logging Panel  — Redis pub/sub event stream over WebSocket
4. Knob Controls  — sliders + presets for behavioural knobs
5. Perception Feed — recent detected experiences
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THALAMUS_URL = os.getenv("THALAMUS_URL", "http://thalamus:8000")
KNOBS_URL = os.getenv("KNOBS_URL", "http://knobs:8090")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

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
    service_ports = {
        "thalamus": 8000, "amygdala": 8001, "hippocampus": 8002,
        "cerebellum": 8003, "neocortex": 8004, "working-memory": 8005,
        "perception": 8006, "verification": 8007, "cross-domain": 8008,
        "compass": 8009, "framework-router": 8010, "self-update": 8011,
        "causal-graph": 8012, "simulator": 8013, "metacognition": 8014,
        "goals": 8015, "learner": 8016, "vision": 8017, "knobs": 8090,
    }
    port = service_ports.get(service_name)
    if port is None:
        return JSONResponse({"error": f"Unknown service: {service_name}"}, status_code=404)
    return await _proxy_get(f"http://{service_name}:{port}/knobs")


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
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SkyNet Brain Dashboard</title>
<style>
/* ── CSS Reset & Variables ───────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e17;--surface:#141926;--surface2:#1b2236;--border:#232b42;
  --text:#e0e6ed;--text-dim:#7b8a9e;--accent:#00d4ff;--accent-dim:#0097b8;
  --success:#00c853;--warning:#ffd600;--error:#ff1744;
  --emo-gratitude:#ffd700;--emo-belonging:#4fc3f7;--emo-regret:#ce93d8;
  --emo-alarm:#ff1744;--emo-joy:#00e676;--emo-curiosity:#26c6da;
  --emo-neutral:#90a4ae;--emo-urgency:#ff9100;--emo-distress:#ef5350;
  --radius:8px;--font:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'SF Mono',SFMono-Regular,ui-monospace,'Cascadia Code',Menlo,monospace;
}
html{font-size:14px}
body{font-family:var(--font);background:var(--bg);color:var(--text);
  min-height:100vh;line-height:1.5}
a{color:var(--accent);text-decoration:none}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* ── Layout ──────────────────────────────────────────────────────────── */
.header{display:flex;align-items:center;gap:12px;padding:12px 20px;
  background:var(--surface);border-bottom:1px solid var(--border)}
.header h1{font-size:1.2rem;font-weight:700;background:linear-gradient(135deg,var(--accent),#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:.5px}
.header .dot{width:8px;height:8px;border-radius:50%;background:var(--success);
  box-shadow:0 0 6px var(--success);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.header .session{margin-left:auto;font-size:.78rem;color:var(--text-dim);font-family:var(--mono)}

.tabs{display:flex;gap:2px;padding:0 20px;background:var(--surface);border-bottom:1px solid var(--border)}
.tab{padding:8px 18px;font-size:.82rem;font-weight:600;color:var(--text-dim);
  cursor:pointer;border-bottom:2px solid transparent;transition:.2s}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}

.panels{display:flex;flex-direction:column;height:calc(100vh - 90px);overflow:hidden}
.panel{display:none;flex:1;overflow:auto;padding:16px 20px}
.panel.active{display:flex;flex-direction:column}

/* ── Terminal ────────────────────────────────────────────────────────── */
#terminal-panel{gap:12px}
.chat-log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px;
  padding:8px;background:var(--surface);border-radius:var(--radius);border:1px solid var(--border)}
.msg{padding:10px 14px;border-radius:var(--radius);max-width:92%;word-wrap:break-word;font-size:.88rem}
.msg.user{align-self:flex-end;background:var(--accent-dim);color:#fff}
.msg.bot{align-self:flex-start;background:var(--surface2);border:1px solid var(--border)}
.msg .meta{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;font-size:.75rem}
.msg .meta span{padding:2px 8px;border-radius:12px;font-weight:600}
.badge-emotion{color:#111}
.badge-latency{background:var(--surface);color:var(--accent);border:1px solid var(--border)}
.badge-confidence{background:var(--surface);color:var(--warning);border:1px solid var(--border)}
.badge-path{background:var(--surface);color:var(--emo-curiosity);border:1px solid var(--border)}
.badge-urgency{color:#111}
.msg .trace-toggle{display:inline-block;margin-top:6px;cursor:pointer;color:var(--accent);font-size:.75rem}
.msg pre.trace{margin-top:6px;padding:8px;background:var(--bg);border-radius:4px;
  font-family:var(--mono);font-size:.72rem;overflow-x:auto;white-space:pre-wrap;display:none;
  max-height:260px;overflow-y:auto;border:1px solid var(--border)}

.chat-input{display:flex;gap:8px}
.chat-input input{flex:1;padding:10px 14px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--surface);color:var(--text);font-size:.9rem;outline:none;font-family:var(--font)}
.chat-input input:focus{border-color:var(--accent)}
.chat-input button{padding:10px 24px;border:none;border-radius:var(--radius);
  background:var(--accent);color:#0a0e17;font-weight:700;cursor:pointer;font-size:.88rem;
  transition:.15s}
.chat-input button:hover{background:#33dfff}
.chat-input button:disabled{opacity:.5;cursor:not-allowed}

/* ── Service Monitor ─────────────────────────────────────────────────── */
.svc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.svc-card{padding:14px;border-radius:var(--radius);background:var(--surface);
  border:1px solid var(--border);cursor:pointer;transition:.15s}
.svc-card:hover{border-color:var(--accent);transform:translateY(-1px)}
.svc-card .svc-header{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.svc-card .dot-sm{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.dot-alive{background:var(--success);box-shadow:0 0 4px var(--success)}
.dot-degraded{background:var(--warning);box-shadow:0 0 4px var(--warning)}
.dot-unreachable{background:var(--error);box-shadow:0 0 4px var(--error)}
.svc-card .svc-name{font-weight:700;font-size:.88rem}
.svc-card .svc-port{font-size:.75rem;color:var(--text-dim);font-family:var(--mono)}
.svc-card .svc-status{font-size:.75rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.5px}
.monitor-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.monitor-header h2{font-size:1rem;font-weight:700}
.monitor-header .counts{display:flex;gap:12px;font-size:.8rem}
.monitor-header .counts span{display:flex;align-items:center;gap:4px}
.refresh-btn{padding:4px 12px;border-radius:4px;border:1px solid var(--border);
  background:var(--surface);color:var(--text-dim);cursor:pointer;font-size:.75rem}
.refresh-btn:hover{border-color:var(--accent);color:var(--accent)}

/* modal for service /knobs detail */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;
  align-items:center;justify-content:center;z-index:100}
.modal-overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px;max-width:560px;width:90%;max-height:70vh;overflow-y:auto}
.modal h3{margin-bottom:10px;font-size:1rem}
.modal pre{font-family:var(--mono);font-size:.78rem;background:var(--bg);padding:12px;
  border-radius:4px;white-space:pre-wrap;overflow-x:auto;border:1px solid var(--border)}
.modal .close-btn{float:right;cursor:pointer;color:var(--text-dim);font-size:1.2rem;
  background:none;border:none;line-height:1}

/* ── Logging Panel ───────────────────────────────────────────────────── */
#logging-panel{gap:10px}
.log-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.log-controls input,.log-controls select{padding:6px 10px;border-radius:4px;
  border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:.82rem}
.log-controls .ws-status{font-size:.75rem;display:flex;align-items:center;gap:4px;margin-left:auto}
.log-entries{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;
  padding:8px;background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);
  font-family:var(--mono);font-size:.78rem}
.log-entry{padding:6px 10px;border-radius:4px;background:var(--surface2);border:1px solid var(--border)}
.log-entry .log-channel{color:var(--accent);font-weight:600}
.log-entry .log-time{color:var(--text-dim);font-size:.72rem}

/* ── Trace Panel (inside logging) ────────────────────────────────────── */
.trace-section{margin-top:10px}
.trace-section h3{font-size:.88rem;font-weight:700;margin-bottom:6px}
.trace-card{padding:10px;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:6px;font-size:.78rem}
.trace-card .trace-header{display:flex;justify-content:space-between;margin-bottom:4px}
.trace-card .trace-header strong{color:var(--accent)}
.trace-card .trace-steps{font-family:var(--mono);font-size:.72rem;color:var(--text-dim)}

/* ── Knob Controls ───────────────────────────────────────────────────── */
#knobs-panel{gap:14px}
.knobs-presets{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:4px}
.knobs-presets button{padding:6px 14px;border-radius:16px;border:1px solid var(--border);
  background:var(--surface);color:var(--text);font-size:.78rem;cursor:pointer;transition:.15s}
.knobs-presets button:hover{border-color:var(--accent);color:var(--accent)}
.knobs-presets button.active-preset{background:var(--accent);color:#0a0e17;border-color:var(--accent);font-weight:700}
.knob-groups{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:14px}
.knob-group{padding:12px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius)}
.knob-group h3{font-size:.85rem;font-weight:700;margin-bottom:8px;color:var(--accent)}
.knob-row{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.knob-row label{width:180px;font-size:.82rem;flex-shrink:0}
.knob-row input[type=range]{flex:1;accent-color:var(--accent);height:4px}
.knob-row .knob-val{width:42px;text-align:right;font-family:var(--mono);font-size:.8rem;color:var(--accent)}

/* ── Perception Feed ─────────────────────────────────────────────────── */
.perc-list{display:flex;flex-direction:column;gap:8px}
.perc-card{padding:12px 16px;border-radius:var(--radius);background:var(--surface);
  border:1px solid var(--border);display:flex;align-items:center;gap:14px;transition:.2s}
.perc-card.highlight{border-color:var(--emo-gratitude);box-shadow:0 0 8px rgba(255,215,0,.15)}
.perc-card .perc-icon{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:1.1rem;flex-shrink:0}
.perc-card .perc-body{flex:1}
.perc-card .perc-type{font-weight:700;font-size:.85rem}
.perc-card .perc-detail{font-size:.78rem;color:var(--text-dim);margin-top:2px}
.perc-card .perc-intensity{width:60px;text-align:right}
.perc-card .perc-intensity .bar{height:6px;border-radius:3px;background:var(--border);overflow:hidden}
.perc-card .perc-intensity .bar-fill{height:100%;border-radius:3px;transition:width .3s}
.perc-card .perc-intensity .val{font-size:.72rem;color:var(--text-dim);text-align:right;margin-top:2px}
.perc-empty{text-align:center;color:var(--text-dim);padding:40px;font-size:.9rem}

/* ── Utilities ───────────────────────────────────────────────────────── */
.loading{display:inline-block;width:14px;height:14px;border:2px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{padding:10px 14px;border-radius:var(--radius);background:rgba(255,23,68,.1);
  border:1px solid var(--error);color:var(--error);font-size:.82rem}
</style>
</head>
<body>

<!-- ── Header ────────────────────────────────────────────────────────── -->
<div class="header">
  <div class="dot" id="global-dot"></div>
  <h1>SkyNet Brain Dashboard</h1>
  <span class="session" id="session-display">session: —</span>
</div>

<!-- ── Tabs ───────────────────────────────────────────────────────────── -->
<div class="tabs" id="tab-bar">
  <div class="tab active" data-panel="terminal-panel">&#x1F4AC; Terminal</div>
  <div class="tab" data-panel="monitor-panel">&#x1F4E1; Services</div>
  <div class="tab" data-panel="logging-panel">&#x1F4DC; Logs</div>
  <div class="tab" data-panel="knobs-panel">&#x1F39B; Knobs</div>
  <div class="tab" data-panel="perception-panel">&#x1F9E0; Perception</div>
</div>

<!-- ── Panels ─────────────────────────────────────────────────────────── -->
<div class="panels">

  <!-- TERMINAL -->
  <div class="panel active" id="terminal-panel">
    <div class="chat-log" id="chat-log"></div>
    <div class="chat-input">
      <input type="text" id="chat-input" placeholder="Talk to the brain…" autocomplete="off" />
      <button id="chat-send">Send</button>
    </div>
  </div>

  <!-- SERVICE MONITOR -->
  <div class="panel" id="monitor-panel">
    <div class="monitor-header">
      <h2>Service Health</h2>
      <div class="counts" id="svc-counts"></div>
      <button class="refresh-btn" id="refresh-health">&#x21BB; Refresh</button>
    </div>
    <div class="svc-grid" id="svc-grid"></div>
  </div>

  <!-- LOGGING -->
  <div class="panel" id="logging-panel">
    <div class="log-controls">
      <input type="text" id="log-filter-svc" placeholder="Filter service…" />
      <select id="log-filter-type">
        <option value="">All events</option>
        <option value="experience.new">experience.new</option>
        <option value="memory.consolidate">memory.consolidate</option>
        <option value="knowledge.updated">knowledge.updated</option>
        <option value="skill.learned">skill.learned</option>
        <option value="knobs.updated">knobs.updated</option>
        <option value="verification.failed">verification.failed</option>
        <option value="verification.contested">verification.contested</option>
        <option value="self.update.triggered">self.update.triggered</option>
        <option value="session.started">session.started</option>
        <option value="session.ended">session.ended</option>
        <option value="analogical.insight">analogical.insight</option>
        <option value="pattern.registered">pattern.registered</option>
      </select>
      <span class="ws-status"><span class="dot-sm dot-unreachable" id="ws-dot"></span><span id="ws-label">Disconnected</span></span>
    </div>
    <div class="log-entries" id="log-entries"></div>
    <div class="trace-section">
      <h3>Recent Cognitive Traces</h3>
      <div id="trace-list"></div>
    </div>
  </div>

  <!-- KNOBS -->
  <div class="panel" id="knobs-panel">
    <div class="knobs-presets" id="knobs-presets"></div>
    <div class="knob-groups" id="knob-groups"></div>
  </div>

  <!-- PERCEPTION -->
  <div class="panel" id="perception-panel">
    <h2 style="margin-bottom:12px;font-size:1rem">Recent Experiences</h2>
    <div class="perc-list" id="perc-list">
      <div class="perc-empty">No experiences detected yet. Chat with the brain to generate them.</div>
    </div>
  </div>

</div>

<!-- ── Modal overlay ─────────────────────────────────────────────────── -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <button class="close-btn" id="modal-close">&times;</button>
    <h3 id="modal-title">Service Detail</h3>
    <pre id="modal-body">Loading…</pre>
  </div>
</div>

<script>
/* =====================================================================
   SkyNet Dashboard — Client-side JavaScript
   ===================================================================== */

// ── State ───────────────────────────────────────────────────────────
const SESSION_KEY = 'skynet_session_id';
let sessionId = localStorage.getItem(SESSION_KEY);
if (!sessionId) {
  sessionId = 'dash-' + crypto.randomUUID().slice(0, 12);
  localStorage.setItem(SESSION_KEY, sessionId);
}
document.getElementById('session-display').textContent = 'session: ' + sessionId;

const cognitiveTraces = [];   // last 5 traces
const experiences = [];       // last 10 perception events
let healthTimer = null;
let ws = null;

// ── Tab switching ───────────────────────────────────────────────────
document.getElementById('tab-bar').addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  tab.classList.add('active');
  document.getElementById(tab.dataset.panel).classList.add('active');

  if (tab.dataset.panel === 'monitor-panel') refreshHealth();
  if (tab.dataset.panel === 'knobs-panel') loadKnobs();
  if (tab.dataset.panel === 'perception-panel') renderPerception();
});

// ── Helpers ─────────────────────────────────────────────────────────
function esc(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}
function emotionColor(emotion) {
  const map = {
    gratitude:'var(--emo-gratitude)', belonging:'var(--emo-belonging)',
    regret:'var(--emo-regret)', alarm:'var(--emo-alarm)', joy:'var(--emo-joy)',
    curiosity:'var(--emo-curiosity)', neutral:'var(--emo-neutral)',
    urgency:'var(--emo-urgency)', distress:'var(--emo-distress)',
    awe:'var(--emo-gratitude)', reverence:'var(--emo-gratitude)',
    nostalgia:'var(--emo-belonging)', longing:'var(--emo-regret)',
    pride:'var(--emo-joy)', serenity:'var(--emo-curiosity)',
  };
  return map[emotion] || 'var(--emo-neutral)';
}
function urgencyColor(u) {
  if (u === 'critical' || u === 'distressed') return 'var(--error)';
  if (u === 'high') return 'var(--emo-urgency)';
  return 'var(--emo-neutral)';
}
function experienceIcon(type) {
  const m = {
    gratitude:'🙏', belonging:'🤝', regret:'😔', threat:'⚠️',
    insight:'💡', curiosity:'🔍', joy:'😊', humor:'😄',
    nostalgia:'🕰️', awe:'✨', flow:'🌊', resonance:'🎵',
    dissonance:'💢', growth:'🌱', pattern:'🔗', anomaly:'❓',
    convergence:'🔀', emergence:'🌟',
  };
  return m[type] || '🧠';
}
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

// ── 1. Terminal ─────────────────────────────────────────────────────
const chatLog = document.getElementById('chat-log');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');

function appendMsg(role, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  chatLog.appendChild(d);
  chatLog.scrollTop = chatLog.scrollHeight;
  return d;
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = '';
  chatSend.disabled = true;

  appendMsg('user', esc(text));
  const loading = appendMsg('bot', '<span class="loading"></span> Thinking…');

  const t0 = performance.now();
  try {
    const data = await api('POST', '/api/process', {
      session_id: sessionId,
      user_input: text,
      metadata: {},
    });
    const latency = ((performance.now() - t0) / 1000).toFixed(2);

    if (data.error) {
      loading.innerHTML = '<span class="error-box">' + esc(data.error) + '</span>';
      chatSend.disabled = false;
      return;
    }

    const emotion = data.emotion || 'neutral';
    const urgency = data.urgency || 'low';
    const confidence = data.confidence != null ? (data.confidence * 100).toFixed(0) + '%' : '—';
    const path = data.cognitive_path || '—';
    const traceId = 'trace-' + Date.now();

    let html = esc(data.response || JSON.stringify(data));
    html += '<div class="meta">';
    html += '<span class="badge-emotion" style="background:' + emotionColor(emotion) + '">' + esc(emotion) + '</span>';
    html += '<span class="badge-urgency" style="background:' + urgencyColor(urgency) + '">' + esc(urgency) + '</span>';
    html += '<span class="badge-latency">⏱ ' + latency + 's</span>';
    html += '<span class="badge-confidence">🎯 ' + confidence + '</span>';
    html += '<span class="badge-path">⚡ ' + esc(path) + '</span>';
    html += '</div>';

    if (data.trace) {
      html += '<span class="trace-toggle" onclick="toggleTrace(\'' + traceId + '\')">▶ Show trace</span>';
      html += '<pre class="trace" id="' + traceId + '">' + esc(JSON.stringify(data.trace, null, 2)) + '</pre>';
    }

    loading.innerHTML = html;

    // Save cognitive trace
    cognitiveTraces.unshift({
      ts: Date.now(),
      input: text.slice(0, 80),
      latency: latency,
      path: path,
      emotion: emotion,
      trace: data.trace || {},
    });
    if (cognitiveTraces.length > 5) cognitiveTraces.pop();
    renderTraces();

    // Collect experiences from response
    if (data.experiences && typeof data.experiences === 'number' && data.trace) {
      // experiences count is in data.experiences; details may be in trace
    }
  } catch (err) {
    loading.innerHTML = '<span class="error-box">Network error: ' + esc(err.message) + '</span>';
  }
  chatSend.disabled = false;
  chatInput.focus();
}

chatSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

function toggleTrace(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const showing = el.style.display === 'block';
  el.style.display = showing ? 'none' : 'block';
  el.previousElementSibling.textContent = showing ? '▶ Show trace' : '▼ Hide trace';
}

// ── 2. Service Monitor ──────────────────────────────────────────────
const ALL_SERVICES = [
  {name:'thalamus',port:8000},{name:'amygdala',port:8001},{name:'hippocampus',port:8002},
  {name:'cerebellum',port:8003},{name:'neocortex',port:8004},{name:'working-memory',port:8005},
  {name:'perception',port:8006},{name:'verification',port:8007},{name:'cross-domain',port:8008},
  {name:'compass',port:8009},{name:'framework-router',port:8010},{name:'self-update',port:8011},
  {name:'causal-graph',port:8012},{name:'simulator',port:8013},{name:'metacognition',port:8014},
  {name:'goals',port:8015},{name:'learner',port:8016},{name:'vision',port:8017},{name:'knobs',port:8090},
];

async function refreshHealth() {
  const grid = document.getElementById('svc-grid');
  try {
    const data = await api('GET', '/api/health');
    const regions = data.regions || {};
    let alive = 0, degraded = 0, unreachable = 0;

    grid.innerHTML = ALL_SERVICES.map(s => {
      const info = regions[s.name];
      let status = 'unreachable', dotCls = 'dot-unreachable';
      if (info) {
        status = info.status || 'alive';
        dotCls = status === 'alive' ? 'dot-alive' : status === 'degraded' ? 'dot-degraded' : 'dot-unreachable';
      }
      if (status === 'alive') alive++;
      else if (status === 'degraded') degraded++;
      else unreachable++;

      return '<div class="svc-card" data-svc="' + s.name + '">' +
        '<div class="svc-header"><span class="dot-sm ' + dotCls + '"></span>' +
        '<span class="svc-name">' + esc(s.name) + '</span></div>' +
        '<div class="svc-port">:' + s.port + '</div>' +
        '<div class="svc-status">' + esc(status) + '</div></div>';
    }).join('');

    document.getElementById('svc-counts').innerHTML =
      '<span><span class="dot-sm dot-alive"></span> ' + alive + ' alive</span>' +
      '<span><span class="dot-sm dot-degraded"></span> ' + degraded + ' degraded</span>' +
      '<span><span class="dot-sm dot-unreachable"></span> ' + unreachable + ' unreachable</span>';

    const gd = document.getElementById('global-dot');
    if (unreachable > 5) { gd.style.background = 'var(--error)'; gd.style.boxShadow = '0 0 6px var(--error)'; }
    else if (degraded > 0 || unreachable > 0) { gd.style.background = 'var(--warning)'; gd.style.boxShadow = '0 0 6px var(--warning)'; }
    else { gd.style.background = 'var(--success)'; gd.style.boxShadow = '0 0 6px var(--success)'; }
  } catch {
    grid.innerHTML = '<div class="error-box">Could not reach thalamus /health</div>';
  }
}

document.getElementById('svc-grid').addEventListener('click', async e => {
  const card = e.target.closest('.svc-card');
  if (!card) return;
  const svc = card.dataset.svc;
  document.getElementById('modal-title').textContent = svc + ' /knobs';
  document.getElementById('modal-body').textContent = 'Loading…';
  document.getElementById('modal-overlay').classList.add('open');
  try {
    const data = await api('GET', '/api/service/' + svc + '/knobs');
    document.getElementById('modal-body').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    document.getElementById('modal-body').textContent = 'Error: ' + err.message;
  }
});

document.getElementById('modal-close').addEventListener('click', () => {
  document.getElementById('modal-overlay').classList.remove('open');
});
document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === e.currentTarget) e.currentTarget.classList.remove('open');
});
document.getElementById('refresh-health').addEventListener('click', refreshHealth);

// Auto-refresh every 10s
healthTimer = setInterval(refreshHealth, 10000);
refreshHealth();

// ── 3. Logging Panel ────────────────────────────────────────────────
const logEntries = document.getElementById('log-entries');
const logFilterSvc = document.getElementById('log-filter-svc');
const logFilterType = document.getElementById('log-filter-type');
const allLogs = [];
const MAX_LOGS = 200;

function renderLogs() {
  const svcF = logFilterSvc.value.toLowerCase();
  const typeF = logFilterType.value;
  const filtered = allLogs.filter(l => {
    if (typeF && l.channel !== typeF) return false;
    if (svcF && !l.channel.toLowerCase().includes(svcF) && !JSON.stringify(l.data).toLowerCase().includes(svcF)) return false;
    return true;
  });
  logEntries.innerHTML = filtered.slice(0, 100).map(l => {
    const time = new Date(l.ts * 1000).toLocaleTimeString();
    return '<div class="log-entry"><span class="log-channel">' + esc(l.channel) +
      '</span> <span class="log-time">' + time + '</span><br>' +
      esc(JSON.stringify(l.data).slice(0, 300)) + '</div>';
  }).join('');
}

logFilterSvc.addEventListener('input', renderLogs);
logFilterType.addEventListener('change', renderLogs);

function renderTraces() {
  const el = document.getElementById('trace-list');
  if (!cognitiveTraces.length) {
    el.innerHTML = '<div style="color:var(--text-dim);font-size:.82rem">No traces yet. Send messages in the terminal to generate traces.</div>';
    return;
  }
  el.innerHTML = cognitiveTraces.map(t => {
    const time = new Date(t.ts).toLocaleTimeString();
    const steps = Object.keys(t.trace).slice(0, 8).join(' → ') || '—';
    return '<div class="trace-card"><div class="trace-header"><strong>' + esc(t.input) +
      '</strong><span>' + time + ' · ' + t.latency + 's · <span style="color:' + emotionColor(t.emotion) + '">' + esc(t.emotion) +
      '</span></span></div><div class="trace-steps">' + esc(t.path) + ' · ' + esc(steps) + '</div></div>';
  }).join('');
}
renderTraces();

// WebSocket connection
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws/events');
  const dot = document.getElementById('ws-dot');
  const label = document.getElementById('ws-label');

  ws.onopen = () => {
    dot.className = 'dot-sm dot-alive';
    label.textContent = 'Connected';
  };
  ws.onclose = () => {
    dot.className = 'dot-sm dot-unreachable';
    label.textContent = 'Disconnected';
    setTimeout(connectWS, 5000);
  };
  ws.onerror = () => {
    dot.className = 'dot-sm dot-unreachable';
    label.textContent = 'Error';
  };
  ws.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'error') {
        label.textContent = msg.message;
        return;
      }
      allLogs.unshift(msg);
      if (allLogs.length > MAX_LOGS) allLogs.pop();
      renderLogs();

      // If it's an experience event, add to perception feed
      if (msg.channel === 'experience.new' && msg.data) {
        addExperience(msg.data);
      }
    } catch {}
  };
}
connectWS();

// ── 4. Knob Controls ────────────────────────────────────────────────
const KNOB_GROUPS = {
  'Personality': ['humor','honesty','empathy','formality','verbosity','creativity','skepticism','assertiveness','curiosity','warmth'],
  'Cognition': ['reasoning_depth','analogy_seeking','verification_threshold','memory_weight','novelty_sensitivity'],
  'Framework': ['swarm_preference','framework_threshold'],
  'Ethics': ['ethical_weight','care_explicitness'],
  'Higher Emotions': ['gratitude_sensitivity','belonging_weight'],
};

const PRESET_NAMES = ['default','therapist','researcher','creative_partner','technical_expert','friend','socratic','concise_professional','philosopher','coach'];

let currentKnobs = {};

async function loadKnobs() {
  try {
    const data = await api('GET', '/api/knobs');
    if (data && !data.error) {
      currentKnobs = data;
      renderKnobSliders();
    }
  } catch {}

  try {
    const presets = await api('GET', '/api/knobs/presets');
    renderPresets(presets);
  } catch {}
}

function renderPresets(presetsData) {
  const el = document.getElementById('knobs-presets');
  // Combine built-in names with any custom presets
  const customNames = presetsData && !presetsData.error ? Object.keys(presetsData).filter(n => !PRESET_NAMES.includes(n)) : [];
  const allNames = [...PRESET_NAMES, ...customNames];
  el.innerHTML = allNames.map(n =>
    '<button data-preset="' + n + '">' + n.replace(/_/g, ' ') + '</button>'
  ).join('') + '<button data-preset="__reset" style="border-color:var(--error);color:var(--error)">⟲ Reset</button>';

  el.addEventListener('click', async e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    const preset = btn.dataset.preset;
    if (preset === '__reset') {
      await api('POST', '/api/knobs/reset');
    } else {
      await api('POST', '/api/knobs/preset/' + preset);
    }
    el.querySelectorAll('button').forEach(b => b.classList.remove('active-preset'));
    btn.classList.add('active-preset');
    await loadKnobs();
  });
}

function renderKnobSliders() {
  const container = document.getElementById('knob-groups');
  container.innerHTML = Object.entries(KNOB_GROUPS).map(([group, knobs]) => {
    const rows = knobs.map(k => {
      const val = currentKnobs[k] != null ? currentKnobs[k] : 0.5;
      const isInt = k === 'framework_threshold';
      const max = isInt ? 200 : 1;
      const step = isInt ? 1 : 0.01;
      const display = isInt ? val : parseFloat(val).toFixed(2);
      return '<div class="knob-row"><label>' + k.replace(/_/g, ' ') +
        '</label><input type="range" min="0" max="' + max + '" step="' + step +
        '" value="' + val + '" data-knob="' + k + '" />' +
        '<span class="knob-val">' + display + '</span></div>';
    }).join('');
    return '<div class="knob-group"><h3>' + group + '</h3>' + rows + '</div>';
  }).join('');

  // Debounced slider events
  let debounce = {};
  container.addEventListener('input', e => {
    if (e.target.tagName !== 'INPUT') return;
    const knob = e.target.dataset.knob;
    const val = e.target.value;
    const isInt = knob === 'framework_threshold';
    e.target.nextElementSibling.textContent = isInt ? val : parseFloat(val).toFixed(2);

    clearTimeout(debounce[knob]);
    debounce[knob] = setTimeout(async () => {
      const numVal = isInt ? parseInt(val) : parseFloat(val);
      await api('PATCH', '/api/knobs/adjust?knob_name=' + knob + '&value=' + numVal);
    }, 300);
  });
}

loadKnobs();

// ── 5. Perception Feed ──────────────────────────────────────────────
const HIGHLIGHT_TYPES = ['gratitude', 'belonging', 'regret'];

function addExperience(exp) {
  const entry = {
    type: exp.experience_type || exp.type || 'unknown',
    intensity: exp.intensity || exp.score || 0.5,
    detail: exp.description || exp.content || exp.trigger || JSON.stringify(exp).slice(0, 120),
    ts: Date.now(),
  };
  experiences.unshift(entry);
  if (experiences.length > 10) experiences.pop();
  renderPerception();
}

function renderPerception() {
  const el = document.getElementById('perc-list');
  if (!experiences.length) {
    el.innerHTML = '<div class="perc-empty">No experiences detected yet. Chat with the brain to generate them.</div>';
    return;
  }
  el.innerHTML = experiences.map(ex => {
    const hl = HIGHLIGHT_TYPES.includes(ex.type) ? ' highlight' : '';
    const color = emotionColor(ex.type);
    const pct = Math.round(ex.intensity * 100);
    return '<div class="perc-card' + hl + '">' +
      '<div class="perc-icon" style="background:' + color + '22;border:1px solid ' + color + '">' +
      experienceIcon(ex.type) + '</div>' +
      '<div class="perc-body"><div class="perc-type" style="color:' + color + '">' + esc(ex.type) + '</div>' +
      '<div class="perc-detail">' + esc(ex.detail) + '</div></div>' +
      '<div class="perc-intensity"><div class="bar"><div class="bar-fill" style="width:' + pct +
      '%;background:' + color + '"></div></div><div class="val">' + pct + '%</div></div></div>';
  }).join('');
}

</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.get("/health")
async def health():
    return {"status": "alive", "service": "dashboard"}
