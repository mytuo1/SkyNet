# databases/redis/init.py
# Run this to verify Redis setup and pre-populate default structures.
# Called by healthcheck container on first deploy.

import redis
import json
import os
import time

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

def wait_for_redis(max_retries: int = 30):
    for i in range(max_retries):
        try:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            r.ping()
            print(f"Redis connected after {i+1} attempts")
            return r
        except Exception as e:
            print(f"Redis not ready ({i+1}/{max_retries}): {e}")
            time.sleep(2)
    raise RuntimeError("Redis never became available")


def init_redis(r: redis.Redis):
    """Initialize Redis with brain defaults."""

    # Brain metadata
    r.set("brain:initialized", "true")
    r.set("brain:start_time", time.time())
    r.set("brain:version", "2.0.0")

    # Default knobs (if not already set)
    if not r.exists("brain:knobs:current"):
        from pathlib import Path
        import sys
        sys.path.insert(0, "/shared")
        # Import default KnobSet
        default_knobs = {
            "humor": 0.4, "honesty": 0.95, "empathy": 0.8,
            "formality": 0.4, "verbosity": 0.5, "creativity": 0.6,
            "skepticism": 0.6, "assertiveness": 0.5, "curiosity": 0.8,
            "warmth": 0.75, "reasoning_depth": 0.7, "analogy_seeking": 0.7,
            "verification_threshold": 0.6, "memory_weight": 0.6,
            "novelty_sensitivity": 0.7, "swarm_preference": 0.4,
            "framework_threshold": 50, "ethical_weight": 0.9,
            "care_explicitness": 0.5
        }
        r.set("brain:knobs:current", json.dumps(default_knobs))
        print("Set default knobs")

    # Calibration namespace priming
    domains = ["technology", "science", "general", "creative", "emotional"]
    for domain in domains:
        key = f"meta:calibration:{domain}"
        if not r.exists(key):
            r.set(key, json.dumps({
                "total": 0, "correct": 0,
                "avg_predicted_confidence": 0.7
            }))

    print("Redis initialization complete")
    print(f"  Keys set: {r.dbsize()}")


if __name__ == "__main__":
    r = wait_for_redis()
    init_redis(r)
