# services/neocortex/main.py
# ── Neocortex: Long-Term Semantic Memory ──────────────────────────────────────
# Owns the Elasticsearch connection.
# Stores distilled knowledge. Serves vector + keyword hybrid search.
# Also stores logical patterns for cross-domain reasoning.

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, "/shared")

import httpx
import numpy as np
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainRequest, BrainResponse
from knobs import KnobsClient
from events import EventBus

app = FastAPI(title="Neocortex", version="1.0.0",
              description="Long-term semantic memory and knowledge storage")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

knobs  = KnobsClient()
events = EventBus()
http   = httpx.AsyncClient(timeout=30.0)

ES_URL     = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
OLLAMA_URL = os.getenv("OLLAMA_URL",        "http://host.docker.internal:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

es = AsyncElasticsearch(ES_URL)

# ── Index Mappings ────────────────────────────────────────────────────────────

KNOWLEDGE_MAPPING = {
    "settings": {
        "number_of_shards":   3,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "knowledge_analyzer": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter":    ["lowercase", "stop", "snowball"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content":          {"type": "text", "analyzer": "knowledge_analyzer"},
            "summary":          {"type": "text"},
            "domain":           {"type": "keyword"},
            "subdomain":        {"type": "keyword"},
            "source_type":      {"type": "keyword"},
            "confidence":       {"type": "float"},
            "created_at":       {"type": "date"},
            "updated_at":       {"type": "date"},
            "access_count":     {"type": "integer"},
            "related_concepts": {"type": "keyword"},
            "contradicts":      {"type": "keyword"},
            "embedding": {
                "type":       "dense_vector",
                "dims":       768,
                "index":      True,
                "similarity": "cosine"
            }
        }
    }
}

PATTERN_MAPPING = {
    "settings": {"number_of_shards": 2, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "pattern_type":   {"type": "keyword"},
            "domain":         {"type": "keyword"},
            "entities":       {"type": "keyword"},
            "relationships":  {"type": "text"},
            "constraints":    {"type": "text"},
            "dynamics":       {"type": "text"},
            "source_content": {"type": "text"},
            "confidence":     {"type": "float"},
            "created_at":     {"type": "date"},
            "embedding": {
                "type":       "dense_vector",
                "dims":       768,
                "index":      True,
                "similarity": "cosine"
            }
        }
    }
}


@app.on_event("startup")
async def startup():
    for index, mapping in [
        ("neocortex_knowledge", KNOWLEDGE_MAPPING),
        ("neocortex_patterns",  PATTERN_MAPPING)
    ]:
        try:
            exists = await es.indices.exists(index=index)
            if not exists:
                await es.indices.create(index=index, body=mapping)
                print(f"Created index: {index}")
        except Exception as e:
            print(f"Index setup error for {index}: {e}")


@app.on_event("shutdown")
async def shutdown():
    await es.close()
    await http.aclose()


# ── Utilities ─────────────────────────────────────────────────────────────────

async def embed(text: str) -> list:
    try:
        resp = await http.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:2000]}
        )
        return resp.json()["embedding"]
    except Exception:
        return [0.0] * 768


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        info      = await es.cluster.health()
        es_status = info.get("status", "unknown")
    except Exception:
        es_status = "unreachable"
    return {"status": "alive", "service": "neocortex", "elasticsearch": es_status}


@app.get("/knobs")
async def get_knobs():
    return knobs.get().dict()


class StoreRequest(BaseModel):
    content:          str
    domain:           str   = "general"
    subdomain:        str   = ""
    source_type:      str   = "direct"
    confidence:       float = 0.8
    related_concepts: list  = []
    embedding:        Optional[list] = None
    summary:          str   = ""


@app.post("/store", response_model=BrainResponse)
async def store_knowledge(req: StoreRequest):
    """Store a piece of semantic knowledge."""
    start = time.time()
    now   = datetime.now(timezone.utc).isoformat()

    embedding = req.embedding or await embed(req.content)

    doc = {
        "content":          req.content[:10000],
        "summary":          req.summary or req.content[:300],
        "domain":           req.domain,
        "subdomain":        req.subdomain,
        "source_type":      req.source_type,
        "confidence":       req.confidence,
        "related_concepts": req.related_concepts,
        "embedding":        embedding,
        "created_at":       now,
        "updated_at":       now,
        "access_count":     0
    }

    result = await es.index(index="neocortex_knowledge", document=doc)

    events.publish("knowledge.updated", {
        "doc_id": result["_id"],
        "domain": req.domain
    })

    return BrainResponse(
        request_id = "store",
        service    = "neocortex",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"doc_id": result["_id"]}
    )


@app.post("/search", response_model=BrainResponse)
async def search(req: BrainRequest, domain: str = None, top_k: int = 8):
    """
    Hybrid retrieval: vector similarity (KNN) + keyword match (BM25).
    Increments access counts on returned documents.
    """
    start     = time.time()
    embedding = await embed(req.user_input)

    must_clauses = []
    if domain and domain not in ("general", ""):
        must_clauses.append({"term": {"domain": domain}})

    body: dict = {
        "knn": {
            "field":          "embedding",
            "query_vector":   embedding,
            "k":              top_k * 2,
            "num_candidates": 200,
        },
        "size": top_k
    }

    if req.user_input.strip():
        body["query"] = {
            "bool": {
                "must": must_clauses,
                "should": [{
                    "multi_match": {
                        "query":  req.user_input,
                        "fields": ["content^2", "summary"],
                        "type":   "best_fields"
                    }
                }],
                "minimum_should_match": 0
            }
        }

    try:
        results = await es.search(index="neocortex_knowledge", body=body)
    except Exception as e:
        return BrainResponse(
            request_id = req.request_id,
            service    = "neocortex",
            success    = False,
            error      = str(e),
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"results": []}
        )

    hits = []
    ids_to_bump = []

    for hit in results["hits"]["hits"]:
        src = hit["_source"]
        hits.append({
            "id":           hit["_id"],
            "content":      src.get("content", ""),
            "summary":      src.get("summary", ""),
            "domain":       src.get("domain", ""),
            "confidence":   src.get("confidence", 1.0),
            "score":        hit.get("_score", 0),
            "access_count": src.get("access_count", 0)
        })
        ids_to_bump.append(hit["_id"])

    # Bump access counts asynchronously
    async def bump_counts():
        for doc_id in ids_to_bump:
            try:
                await es.update(
                    index = "neocortex_knowledge",
                    id    = doc_id,
                    body  = {"script": "ctx._source.access_count++"}
                )
            except Exception:
                pass

    asyncio.create_task(bump_counts())

    return BrainResponse(
        request_id = req.request_id,
        service    = "neocortex",
        latency_ms = round((time.time() - start) * 1000, 2),
        data       = {"results": hits}
    )


@app.put("/update/{doc_id}")
async def update_knowledge(doc_id: str, updates: dict):
    """Partial update to an existing knowledge document."""
    now = datetime.now(timezone.utc).isoformat()
    await es.update(
        index = "neocortex_knowledge",
        id    = doc_id,
        body  = {"doc": {**updates, "updated_at": now}}
    )
    return {"updated": doc_id}


@app.post("/store-pattern")
async def store_pattern(pattern: dict):
    """Store a logical pattern for the cross-domain reasoning engine."""
    pattern["created_at"] = datetime.now(timezone.utc).isoformat()
    result = await es.index(index="neocortex_patterns", document=pattern)
    events.publish("pattern.registered", {"pattern_id": result["_id"],
                                           "domain": pattern.get("domain")})
    return {"pattern_id": result["_id"]}


@app.post("/search-patterns")
async def search_patterns(body: dict):
    """Search for structurally similar logic patterns."""
    embedding    = body.get("embedding", [])
    pattern_type = body.get("pattern_type")
    top_k        = body.get("top_k", 20)

    search_body: dict = {
        "knn": {
            "field":          "embedding",
            "query_vector":   embedding,
            "k":              top_k,
            "num_candidates": 200,
        },
        "size": top_k
    }

    if pattern_type:
        search_body["query"] = {
            "bool": {"must": [{"term": {"pattern_type": pattern_type}}]}
        }

    try:
        results = await es.search(index="neocortex_patterns", body=search_body)
    except Exception as e:
        return {"patterns": [], "error": str(e)}

    return {
        "patterns": [
            {
                "id":            h["_id"],
                "pattern_type":  h["_source"].get("pattern_type"),
                "domain":        h["_source"].get("domain"),
                "entities":      h["_source"].get("entities", []),
                "relationships": h["_source"].get("relationships", []),
                "dynamics":      h["_source"].get("dynamics", []),
                "source_content":h["_source"].get("source_content", ""),
                "embedding":     h["_source"].get("embedding"),
                "confidence":    h["_source"].get("confidence", 0.8),
                "score":         h.get("_score", 0)
            }
            for h in results["hits"]["hits"]
        ]
    }


@app.get("/stats")
async def stats():
    try:
        k_count = await es.count(index="neocortex_knowledge")
        p_count = await es.count(index="neocortex_patterns")
        return {
            "knowledge_items": k_count["count"],
            "pattern_items":   p_count["count"]
        }
    except Exception as e:
        return {"error": str(e)}
