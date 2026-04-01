# databases/elasticsearch/init.py
# Creates all Elasticsearch indices with correct mappings.
# Run once on first deploy, safe to re-run (uses UNLESS EXISTS).

import os
import time
import json
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
ES_USER = os.getenv("ELASTICSEARCH_USER", "")
ES_PASS = os.getenv("ELASTICSEARCH_PASSWORD", "")


def wait_for_es(max_retries: int = 40):
    kwargs = {"hosts": [ES_URL]}
    if ES_USER and ES_PASS:
        kwargs["basic_auth"] = (ES_USER, ES_PASS)

    for i in range(max_retries):
        try:
            es = Elasticsearch(**kwargs)
            health = es.cluster.health(wait_for_status="yellow", timeout="10s")
            print(f"Elasticsearch ready after {i+1} attempts: {health['status']}")
            return es
        except Exception as e:
            print(f"ES not ready ({i+1}/{max_retries}): {str(e)[:60]}")
            time.sleep(3)
    raise RuntimeError("Elasticsearch never became available")


# ── Index definitions ──────────────────────────────────────────────────────────

INDICES = {
    "neocortex_knowledge": {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "knowledge_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"]
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
                    "type": "dense_vector", "dims": 768,
                    "index": True, "similarity": "cosine"
                }
            }
        }
    },
    "neocortex_patterns": {
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
                    "type": "dense_vector", "dims": 768,
                    "index": True, "similarity": "cosine"
                }
            }
        }
    },
    "visual_memories": {
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "caption":       {"type": "text"},
                "objects":       {"type": "keyword"},
                "scene_type":    {"type": "keyword"},
                "emotion_tags":  {"type": "keyword"},
                "timestamp":     {"type": "date"},
                "session_id":    {"type": "keyword"},
                "frame_path":    {"type": "keyword"},
                "importance":    {"type": "float"},
                "clip_embedding": {
                    "type": "dense_vector", "dims": 512,
                    "index": True, "similarity": "cosine"
                },
                "text_embedding": {
                    "type": "dense_vector", "dims": 768,
                    "index": True, "similarity": "cosine"
                }
            }
        }
    },
    "causal_events": {
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "cause":      {"type": "text"},
                "effect":     {"type": "text"},
                "mechanism":  {"type": "text"},
                "domain":     {"type": "keyword"},
                "strength":   {"type": "float"},
                "evidence":   {"type": "integer"},
                "created_at": {"type": "date"},
                "embedding": {
                    "type": "dense_vector", "dims": 768,
                    "index": True, "similarity": "cosine"
                }
            }
        }
    }
}


def init_elasticsearch(es: Elasticsearch):
    """Create all indices if they don't exist."""
    for index_name, config in INDICES.items():
        try:
            exists = es.indices.exists(index=index_name)
            if not exists:
                es.indices.create(
                    index=index_name,
                    settings=config.get("settings", {}),
                    mappings=config.get("mappings", {})
                )
                print(f"  Created index: {index_name}")
            else:
                print(f"  Index exists:  {index_name}")
        except Exception as e:
            print(f"  Error creating {index_name}: {e}")

    # Verify cluster health
    health = es.cluster.health()
    print(f"\nCluster health: {health['status']}")
    print(f"Total indices: {len(INDICES)}")


if __name__ == "__main__":
    print("Initializing Elasticsearch indices...")
    es = wait_for_es()
    init_elasticsearch(es)
    print("Elasticsearch initialization complete")
