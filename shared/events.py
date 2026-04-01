import redis
import json
from typing import Callable

class EventBus:
    """
    Pub/Sub for async communication between brain regions.
    Services publish events. Other services subscribe to relevant channels.
    """
    
    CHANNELS = {
        "experience.new":        "New experience detected",
        "memory.consolidate":    "Trigger consolidation",
        "knowledge.updated":     "Neocortex has new knowledge",
        "skill.learned":         "Cerebellum learned a skill",
        "knobs.updated":         "Behavioral parameters changed",
        "verification.failed":   "Fact check found contradiction",
        "self.update.triggered": "Self-modification requested",
        "session.started":       "New session began",
        "session.ended":         "Session ended",
    }
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.pub = redis.from_url(redis_url, decode_responses=True)
        self.sub = redis.from_url(redis_url, decode_responses=True)
    
    def publish(self, channel: str, data: dict):
        self.pub.publish(channel, json.dumps(data))
    
    def subscribe(self, channels: list, callback: Callable):
        pubsub = self.sub.pubsub()
        pubsub.subscribe(*channels)
        
        for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    callback(message["channel"], data)
                except Exception as e:
                    print(f"Event handler error: {e}")