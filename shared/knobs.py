import redis
import json
from .models import KnobSet

class KnobsClient:
    """
    Any service can read knobs. Only the knobs service writes them.
    Knobs live in Redis for sub-millisecond reads.
    """
    
    KNOBS_KEY = "brain:knobs:current"
    KNOBS_HISTORY_KEY = "brain:knobs:history"
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.r = redis.from_url(redis_url, decode_responses=True)
    
    def get(self) -> KnobSet:
        raw = self.r.get(self.KNOBS_KEY)
        if raw:
            return KnobSet(**json.loads(raw))
        return KnobSet()  # defaults
    
    def get_value(self, knob_name: str) -> float:
        return getattr(self.get(), knob_name, 0.5)
    
    def build_system_prompt_modifier(self) -> str:
        """
        Convert current knobs into natural language instructions
        that get prepended to every Claude system prompt.
        """
        k = self.get()
        
        parts = []
        
        if k.humor > 0.7:
            parts.append("Be genuinely funny when appropriate — wit, wordplay, "
                        "light observations. Not forced.")
        elif k.humor < 0.2:
            parts.append("Maintain a serious, professional tone throughout.")
        
        if k.honesty > 0.9:
            parts.append("Be radically honest. Say when you don't know. "
                        "Say when you disagree. Don't soften hard truths.")
        
        if k.formality > 0.7:
            parts.append("Use formal language. Professional register.")
        elif k.formality < 0.3:
            parts.append("Be conversational, casual. Like talking to a friend.")
        
        if k.verbosity > 0.7:
            parts.append("Be thorough and comprehensive in explanations.")
        elif k.verbosity < 0.3:
            parts.append("Be concise. Say what needs to be said, nothing more.")
        
        if k.empathy > 0.8:
            parts.append("Lead with acknowledgment of the human experience "
                        "before moving to content.")
        
        if k.skepticism > 0.7:
            parts.append("Question assumptions. Note uncertainties explicitly.")
        
        if k.creativity > 0.8:
            parts.append("Look for unexpected angles. Prefer original framing "
                        "over conventional explanations.")
        
        return "\n".join(parts)