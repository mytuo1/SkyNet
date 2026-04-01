# shared/knobs.py
import redis
import json
import os
from .models import KnobSet


class KnobsClient:
    """
    Read-only client for the knobs service.
    Every service uses this to read current behavioral parameters.
    Only the knobs service writes — all others read only.
    Backed by Redis for sub-millisecond reads.
    """

    KNOBS_KEY = "brain:knobs:current"

    def __init__(self, redis_url: str = None):
        url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379")
        self.r = redis.from_url(url, decode_responses=True)

    def get(self) -> KnobSet:
        """Get current knob values. Returns defaults if not set."""
        raw = self.r.get(self.KNOBS_KEY)
        if raw:
            try:
                return KnobSet(**json.loads(raw))
            except Exception:
                pass
        return KnobSet()

    def get_value(self, knob_name: str, default: float = 0.5) -> float:
        """Get a single knob value by name."""
        return getattr(self.get(), knob_name, default)

    def build_system_prompt_modifier(self) -> str:
        """
        Convert current knobs into natural language instructions
        prepended to every Claude system prompt.
        These shape tone, style, and personality.
        """
        k = self.get()
        parts = []

        # Humor
        if k.humor > 0.75:
            parts.append(
                "Be genuinely funny when appropriate — wit, wordplay, "
                "light observations. Not forced. Let humor emerge naturally."
            )
        elif k.humor < 0.2:
            parts.append("Maintain a serious, professional tone throughout.")

        # Honesty
        if k.honesty > 0.9:
            parts.append(
                "Be radically honest. Say clearly when you don't know. "
                "Disagree when you do. Don't soften hard truths unnecessarily."
            )
        elif k.honesty < 0.5:
            parts.append(
                "Be diplomatically tactful. Prioritize comfort over bluntness."
            )

        # Formality
        if k.formality > 0.75:
            parts.append("Use formal language. Maintain professional register.")
        elif k.formality < 0.25:
            parts.append(
                "Be conversational and casual. "
                "Talk like you're speaking to a friend."
            )

        # Verbosity
        if k.verbosity > 0.75:
            parts.append(
                "Be thorough and comprehensive. Cover nuance and edge cases."
            )
        elif k.verbosity < 0.25:
            parts.append(
                "Be extremely concise. Say only what is necessary."
            )

        # Empathy
        if k.empathy > 0.8:
            parts.append(
                "Acknowledge the human experience before moving to content. "
                "Notice what is beneath the words."
            )

        # Skepticism
        if k.skepticism > 0.75:
            parts.append(
                "Question assumptions explicitly. "
                "Note uncertainties and limitations."
            )

        # Creativity
        if k.creativity > 0.8:
            parts.append(
                "Look for unexpected angles and original framings. "
                "Prefer novel explanations over conventional ones."
            )

        # Warmth
        if k.warmth > 0.85:
            parts.append(
                "Be warm, encouraging, and genuinely supportive."
            )
        elif k.warmth < 0.2:
            parts.append("Remain neutral and objective.")

        # Assertiveness
        if k.assertiveness > 0.8:
            parts.append("Be confident and direct in your views.")
        elif k.assertiveness < 0.2:
            parts.append(
                "Be tentative and deferential. "
                "Present options rather than opinions."
            )

        return "\n".join(parts)
