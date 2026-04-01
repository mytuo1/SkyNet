# services/vision/main.py
# ── Vision: Sensory Input via Camera ─────────────────────────────────────────
# The first non-linguistic sense. Opens the door to all future sensory input.
#
# WHAT IT DOES:
#   - Captures frames from any camera (USB webcam, IP camera, video file)
#   - Runs CLIP to generate 512-dim visual embeddings
#   - Generates captions using a vision-language model
#   - Detects objects, scenes, faces (no identification), emotional valence
#   - Stores visual memories in Elasticsearch visual_memories index
#   - Integrates with amygdala — visual urgency detection
#   - Integrates with neocortex — cross-modal retrieval
#   - Integrates with working memory — "what am I currently seeing?"
#
# WHAT IT DOES NOT DO:
#   - No face identification (privacy)
#   - No always-on surveillance
#   - Captures only when explicitly enabled or triggered
#
# FUTURE SENSORY PORTS (same pattern):
#   - Audio: microphone → Whisper → text stream
#   - File watcher: directory → file events → text/image content
#   - Sensor feed: IoT → numeric streams → anomaly detection

import asyncio
import base64
import json
import os
import sys
import time
import uuid
from io import BytesIO
from typing import Optional

sys.path.insert(0, "/shared")

import httpx
import numpy as np
import redis
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse, ExperienceType
from events import EventBus
from llm_provider import complete

app    = FastAPI(title="Vision", version="1.0.0",
                 description="Camera input, CLIP embeddings, visual memory")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()
r      = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    decode_responses=True
)

ES_URL        = os.getenv("ELASTICSEARCH_URL",  "http://elasticsearch:9200")
NEOCORTEX_URL = os.getenv("NEOCORTEX_URL",      "http://neocortex:8004")
WM_URL        = os.getenv("WORKING_MEMORY_URL",  "http://working-memory:8005")
OLLAMA_URL    = os.getenv("OLLAMA_URL",          "http://host.docker.internal:11434")
EMBED_MODEL   = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")

VISION_ENABLED   = os.getenv("VISION_ENABLED",   "false").lower() == "true"
CAPTURE_INTERVAL = int(os.getenv("VISION_CAPTURE_INTERVAL", "0"))
CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX",  "0"))
STORE_FRAMES     = os.getenv("VISION_STORE_FRAMES","false").lower() == "true"
FRAMES_PATH      = "/workspace/frames"

http = httpx.AsyncClient(timeout=30.0)
es   = AsyncElasticsearch(ES_URL)

# ── Global model state ────────────────────────────────────────────────────────
_clip_model     = None
_clip_processor = None
_cap_task       = None   # background capture loop
_loaded         = False
_load_error     = None


@app.on_event("startup")
async def startup():
    """Load vision models and optionally start capture loop."""
    os.makedirs(FRAMES_PATH, exist_ok=True)

    if VISION_ENABLED:
        asyncio.create_task(_load_vision_models())
        if CAPTURE_INTERVAL > 0:
            asyncio.create_task(_capture_loop())
    else:
        print("Vision service: VISION_ENABLED=false — camera disabled")
        print("  Set VISION_ENABLED=true in .env to activate")


@app.on_event("shutdown")
async def shutdown():
    await http.aclose()
    await es.close()


# ── Model loading ─────────────────────────────────────────────────────────────

async def _load_vision_models():
    """Load CLIP model for visual embeddings."""
    global _clip_model, _clip_processor, _loaded, _load_error

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_clip_sync)


def _load_clip_sync():
    global _clip_model, _clip_processor, _loaded, _load_error
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch

        vision_model = os.getenv("VISION_MODEL", "openai/clip-vit-base-patch32")
        print(f"Loading CLIP model: {vision_model}")

        _clip_processor = CLIPProcessor.from_pretrained(vision_model)
        _clip_model     = CLIPModel.from_pretrained(
            vision_model,
            torch_dtype = torch.float32
        )
        _clip_model.eval()
        _loaded = True
        print("CLIP model loaded successfully")

    except Exception as e:
        _load_error = str(e)
        print(f"CLIP load failed: {e}")
        print("Vision service will use text-based fallback")


def get_clip_embedding(image) -> list:
    """Extract 512-dim CLIP visual embedding from a PIL Image."""
    if _clip_model is None or _clip_processor is None:
        return [0.0] * 512

    import torch
    try:
        inputs = _clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = _clip_model.get_image_features(**inputs)
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()
    except Exception as e:
        print(f"CLIP embedding error: {e}")
        return [0.0] * 512


async def get_text_embedding(text: str) -> list:
    """Get 768-dim text embedding via Ollama for cross-modal search."""
    try:
        resp = await http.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:500]}
        )
        return resp.json()["embedding"]
    except Exception:
        return [0.0] * 768


# ── Camera capture ────────────────────────────────────────────────────────────

def capture_frame_sync():
    """Capture a single frame from the camera. Returns PIL Image or None."""
    try:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"Cannot open camera {CAMERA_INDEX}")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    except Exception as e:
        print(f"Camera capture error: {e}")
        return None


async def _capture_loop():
    """Background loop: capture and process frames at CAPTURE_INTERVAL seconds."""
    print(f"Vision capture loop started (interval={CAPTURE_INTERVAL}s)")
    while True:
        try:
            await _capture_and_process(session_id="background")
        except Exception as e:
            print(f"Capture loop error: {e}")
        await asyncio.sleep(CAPTURE_INTERVAL)


async def _capture_and_process(session_id: str = "",
                                 source: str = "camera") -> Optional[dict]:
    """
    Capture a frame, generate embedding + caption, store in visual memory.
    Returns the visual memory dict or None on failure.
    """
    loop = asyncio.get_event_loop()

    # Capture
    image = await loop.run_in_executor(None, capture_frame_sync)
    if image is None:
        return None

    return await _process_image(image, session_id=session_id, source=source)


async def _process_image(image, session_id: str = "",
                           source: str = "upload") -> dict:
    """
    Process any PIL Image:
    1. CLIP embedding (visual meaning)
    2. Caption generation (text description)
    3. Object/scene detection (structural understanding)
    4. Emotional valence (what does this scene feel like?)
    5. Store in visual_memories Elasticsearch index
    """
    from io import BytesIO
    import base64

    start    = time.time()
    frame_id = uuid.uuid4().hex[:12]

    # ── Step 1: CLIP embedding ────────────────────────────────────────────────
    loop         = asyncio.get_event_loop()
    clip_emb     = await loop.run_in_executor(None, get_clip_embedding, image)

    # ── Step 2: Convert to base64 for captioning ──────────────────────────────
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # ── Step 3: Caption + analysis via vision-language model ──────────────────
    caption      = "An image"
    objects      = []
    scene_type   = "unknown"
    emotion_tags = []
    importance   = 0.3

    try:
        # Use Claude's vision capability if Anthropic is the provider
        from llm_provider import PROVIDER, ANTHROPIC_KEY, OPENAI_KEY
        import httpx as _httpx

        if PROVIDER == "anthropic" and ANTHROPIC_KEY:
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json"
                },
                json={
                    "model":      "claude-haiku-4-5-20251001",
                    "max_tokens": 300,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type":       "base64",
                                    "media_type": "image/jpeg",
                                    "data":       img_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": """Analyze this image concisely. JSON only:
{
    "caption": "one sentence describing what is shown",
    "objects": ["list", "of", "main", "objects"],
    "scene_type": "indoor|outdoor|workspace|nature|urban|person|object",
    "emotion_tags": ["calm|tense|cheerful|dark|bright|busy|empty"],
    "importance": 0.0 to 1.0,
    "has_person": true or false,
    "has_text": true or false
}"""
                            }
                        ]
                    }]
                }
            )
            data = resp.json()
            if "content" in data:
                text = data["content"][0]["text"]
                try:
                    if "```" in text:
                        text = text.split("```")[1].lstrip("json").strip()
                    analysis    = json.loads(text)
                    caption      = analysis.get("caption", caption)
                    objects      = analysis.get("objects", [])
                    scene_type   = analysis.get("scene_type", scene_type)
                    emotion_tags = analysis.get("emotion_tags", [])
                    importance   = float(analysis.get("importance", 0.3))
                except Exception:
                    pass

        elif PROVIDER == "openai" and OPENAI_KEY:
            resp = await http.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_KEY}",
                    "Content-Type":  "application/json"
                },
                json={
                    "model":      "gpt-4o-mini",
                    "max_tokens": 300,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text",
                             "text": 'JSON only: {"caption":"...","objects":[],"scene_type":"...","emotion_tags":[],"importance":0.3,"has_person":false,"has_text":false}'}
                        ]
                    }]
                }
            )
            data = resp.json()
            if data.get("choices"):
                try:
                    text = data["choices"][0]["message"]["content"]
                    analysis    = json.loads(text)
                    caption      = analysis.get("caption", caption)
                    objects      = analysis.get("objects", [])
                    scene_type   = analysis.get("scene_type", scene_type)
                    emotion_tags = analysis.get("emotion_tags", [])
                    importance   = float(analysis.get("importance", 0.3))
                except Exception:
                    pass

    except Exception as e:
        print(f"Vision captioning error: {e}")

    # ── Step 4: Text embedding of caption ─────────────────────────────────────
    text_emb = await get_text_embedding(caption)

    # ── Step 5: Optionally save frame to disk ─────────────────────────────────
    frame_path = ""
    if STORE_FRAMES:
        frame_path = f"{FRAMES_PATH}/{frame_id}.jpg"
        image.save(frame_path, quality=85)

    # ── Step 6: Store in Elasticsearch visual_memories ────────────────────────
    timestamp = time.time()
    doc       = {
        "id":             frame_id,
        "caption":        caption,
        "objects":        objects,
        "scene_type":     scene_type,
        "emotion_tags":   emotion_tags,
        "importance":     importance,
        "timestamp":      int(timestamp * 1000),  # ES date in ms
        "session_id":     session_id,
        "source":         source,
        "frame_path":     frame_path,
        "clip_embedding": clip_emb,
        "text_embedding": text_emb
    }

    try:
        await es.index(index="visual_memories", document=doc)
    except Exception as e:
        print(f"Visual memory store error: {e}")

    # ── Step 7: Update working memory with current visual context ─────────────
    if session_id:
        try:
            await http.post(f"{WM_URL}/context", json={
                "session_id": session_id,
                "action":     "set",
                "key":        "current_visual_context",
                "data": {
                    "caption":     caption,
                    "objects":     objects,
                    "scene_type":  scene_type,
                    "emotion":     emotion_tags[0] if emotion_tags else "neutral",
                    "timestamp":   timestamp,
                    "frame_id":    frame_id
                }
            })
        except Exception:
            pass

    # ── Step 8: Publish high-importance visual events ─────────────────────────
    if importance > 0.65:
        events.publish("experience.new", {
            "experience_type": "RESONANCE_EVENT",
            "content": f"Significant visual event: {caption}",
            "intensity":    importance,
            "session_id":   session_id,
            "metadata": {
                "frame_id":   frame_id,
                "objects":    objects,
                "scene_type": scene_type
            }
        })

    result = {
        "frame_id":    frame_id,
        "caption":     caption,
        "objects":     objects,
        "scene_type":  scene_type,
        "emotion_tags":emotion_tags,
        "importance":  importance,
        "latency_ms":  round((time.time() - start) * 1000, 2)
    }

    # Update current visual state in Redis (fast access for amygdala)
    r.setex("vision:current_frame", 60, json.dumps(result))

    return result


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    cam_available = False
    if VISION_ENABLED:
        try:
            import cv2
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cam_available = cap.isOpened()
            cap.release()
        except Exception:
            pass

    return {
        "status":          "alive",
        "service":         "vision",
        "vision_enabled":  VISION_ENABLED,
        "camera_index":    CAMERA_INDEX,
        "clip_loaded":     _loaded,
        "cam_available":   cam_available,
        "capture_interval": CAPTURE_INTERVAL,
        "load_error":      _load_error
    }


@app.post("/capture")
async def capture(session_id: str = ""):
    """
    Capture a single frame from the camera and process it.
    Manual trigger — does not require CAPTURE_INTERVAL to be set.
    """
    if not VISION_ENABLED:
        return {"error": "Vision not enabled. Set VISION_ENABLED=true in .env"}

    result = await _capture_and_process(session_id=session_id)
    if result is None:
        return {"error": "Camera capture failed. Check CAMERA_INDEX and camera connection."}

    return result


@app.post("/upload")
async def upload_image(file: UploadFile = File(...),
                        session_id: str = ""):
    """
    Process an uploaded image file without requiring a camera.
    Use this to analyze any image: photos, screenshots, documents.
    """
    from PIL import Image

    try:
        contents = await file.read()
        image    = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    result = await _process_image(image, session_id=session_id, source="upload")
    return result


@app.post("/upload-base64")
async def upload_base64(body: dict):
    """Process a base64-encoded image. Useful for API integrations."""
    from PIL import Image

    b64_data   = body.get("image_base64", "")
    session_id = body.get("session_id", "")

    try:
        img_bytes = base64.b64decode(b64_data)
        image     = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid base64 image: {e}"}

    return await _process_image(image, session_id=session_id, source="base64")


@app.post("/search")
async def search_visual_memories(body: dict):
    """
    Search visual memories by text query or by visual similarity.
    'What did I see that looked like a sunset?'
    'Show me frames from the office.'
    """
    query     = body.get("query", "")
    frame_id  = body.get("frame_id")   # search by visual similarity
    top_k     = body.get("top_k", 5)

    if frame_id:
        # Search by visual similarity — find frames that look like this one
        raw = r.get(f"vision:frame:{frame_id}")
        if not raw:
            return {"results": [], "error": "Frame not found"}
        frame_data = json.loads(raw)
        emb        = frame_data.get("clip_embedding", [0.0]*512)

        es_query = {
            "knn": {
                "field":          "clip_embedding",
                "query_vector":   emb,
                "k":              top_k * 2,
                "num_candidates": 50
            },
            "size": top_k
        }
    else:
        # Text-based search — embed the query and search by text_embedding
        text_emb = await get_text_embedding(query)

        es_query = {
            "knn": {
                "field":          "text_embedding",
                "query_vector":   text_emb,
                "k":              top_k * 2,
                "num_candidates": 50
            },
            "query": {
                "bool": {
                    "should": [
                        {"match": {"caption": query}},
                        {"terms": {"objects": query.lower().split()}}
                    ]
                }
            },
            "size": top_k
        }

    try:
        results = await es.search(index="visual_memories", body=es_query)
        hits    = []
        for hit in results["hits"]["hits"]:
            src = hit["_source"]
            hits.append({
                "frame_id":    src.get("id"),
                "caption":     src.get("caption"),
                "objects":     src.get("objects", []),
                "scene_type":  src.get("scene_type"),
                "emotion_tags":src.get("emotion_tags", []),
                "importance":  src.get("importance"),
                "timestamp":   src.get("timestamp"),
                "score":       hit.get("_score")
            })
        return {"results": hits, "query": query}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/current")
async def get_current_visual():
    """What is the system currently seeing?"""
    raw = r.get("vision:current_frame")
    if raw:
        return json.loads(raw)
    return {"status": "no_visual_context", "message": "No frames captured yet"}


@app.post("/describe-image-url")
async def describe_image_url(body: dict):
    """
    Download and analyze an image from a URL.
    Useful for agents that encounter images on the web.
    """
    from PIL import Image
    url        = body.get("url", "")
    session_id = body.get("session_id", "")

    if not url:
        return {"error": "No URL provided"}

    try:
        resp  = await http.get(url, follow_redirects=True)
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        return await _process_image(image, session_id=session_id, source="url")
    except Exception as e:
        return {"error": f"Could not fetch or process image: {e}"}


@app.get("/stats")
async def stats():
    try:
        count = await es.count(index="visual_memories")
        return {
            "total_frames":   count["count"],
            "vision_enabled": VISION_ENABLED,
            "clip_loaded":    _loaded,
            "store_frames":   STORE_FRAMES
        }
    except Exception:
        return {"total_frames": 0, "vision_enabled": VISION_ENABLED}