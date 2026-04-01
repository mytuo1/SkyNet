# shared/events.py
import redis
import json
import os
from typing import Callable


class EventBus:
    """
    Redis pub/sub event bus for async communication between brain regions.
    Services publish events — other services subscribe to relevant channels.
    This is how brain regions stay in sync without direct coupling.
    """

    CHANNELS = {
        "experience.new":         "New experience detected by perception",
        "memory.consolidate":     "Trigger hippocampus→neocortex consolidation",
        "knowledge.updated":      "Neocortex has new semantic knowledge",
        "skill.learned":          "Cerebellum learned a new skill",
        "knobs.updated":          "Behavioral parameters have changed",
        "verification.failed":    "Fact check found a contradiction",
        "verification.contested": "Conflicting knowledge found",
        "self.update.triggered":  "Self-modification has been applied",
        "session.started":        "New session began",
        "session.ended":          "Session completed",
        "analogical.insight":     "Cross-domain analogy discovered",
        "pattern.registered":     "New logic pattern stored",
    }

    def __init__(self, redis_url: str = None):
        url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379")
        self._pub = redis.from_url(url, decode_responses=True)
        self._sub = redis.from_url(url, decode_responses=True)

    def publish(self, channel: str, data: dict):
        """Publish an event to a channel."""
        try:
            self._pub.publish(channel, json.dumps(data))
        except Exception as e:
            print(f"EventBus publish error on {channel}: {e}")

    def subscribe(self, channels: list, callback: Callable):
        """
        Subscribe to channels and call callback on each message.
        Blocking — run in a background thread.
        """
        pubsub = self._sub.pubsub()
        pubsub.subscribe(*channels)

        for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    callback(message["channel"], data)
                except Exception as e:
                    print(f"EventBus handler error: {e}")

    def publish_experience(self, experience_type: str, content: str,
                           intensity: float, session_id: str,
                           metadata: dict = None):
        """Convenience method for publishing experience events."""
        import time
        self.publish("experience.new", {
            "experience_type": experience_type,
            "content":         content,
            "intensity":       intensity,
            "session_id":      session_id,
            "timestamp":       time.time(),
            "metadata":        metadata or {}
        })
