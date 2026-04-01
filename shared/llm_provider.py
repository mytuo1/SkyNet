# shared/llm_provider.py
# ── LLM Provider Abstraction ──────────────────────────────────────────────────
# Drop-in replacement for direct Anthropic calls.
# Switch between Anthropic and OpenAI via LLM_PROVIDER env var.
# All services import from here — never call APIs directly.
# To switch to OpenAI: set LLM_PROVIDER=openai in .env. Done.

import json
import os
from typing import Optional

import httpx

_http = httpx.AsyncClient(timeout=120.0)

# ── Provider configuration ────────────────────────────────────────────────────

PROVIDER       = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")

# Model mapping — same logical role, different model names
MODELS = {
    "anthropic": {
        "main": os.getenv("ANTHROPIC_MAIN_MODEL", "claude-sonnet-4-20250514"),
        "fast": os.getenv("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5-20251001"),
    },
    "openai": {
        "main": os.getenv("OPENAI_MAIN_MODEL", "gpt-4o"),
        "fast": os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini"),
    }
}

def get_model(tier: str = "main") -> str:
    """Get the right model name for current provider and tier."""
    return MODELS.get(PROVIDER, MODELS["anthropic"]).get(tier, "")

def has_api_key() -> bool:
    """Check if current provider has an API key configured."""
    if PROVIDER == "anthropic":
        return bool(ANTHROPIC_KEY)
    if PROVIDER == "openai":
        return bool(OPENAI_KEY)
    return bool(ANTHROPIC_KEY)   # default to anthropic


# ── Core completion function ──────────────────────────────────────────────────

async def complete(
    prompt:     str,
    system:     str       = "",
    model_tier: str       = "fast",      # "main" or "fast"
    model:      str       = None,        # override model tier
    max_tokens: int       = 1000,
    temperature: float    = 0.7,
    json_mode:  bool      = False,       # request JSON output
) -> str:
    """
    Universal completion function.
    Same call signature regardless of which provider is active.
    Returns the text response as a string.
    """
    if not has_api_key():
        return f"[No API key for provider={PROVIDER}]"

    chosen_model = model or get_model(model_tier)

    if json_mode and system:
        system += "\n\nRespond only with valid JSON. No markdown code blocks."
    elif json_mode:
        system = "Respond only with valid JSON. No markdown code blocks."

    if PROVIDER == "openai":
        return await _openai_complete(prompt, system, chosen_model,
                                      max_tokens, temperature, json_mode)
    else:
        return await _anthropic_complete(prompt, system, chosen_model,
                                         max_tokens, temperature)


async def complete_json(
    prompt:     str,
    system:     str    = "",
    model_tier: str    = "fast",
    model:      str    = None,
    max_tokens: int    = 800,
) -> dict:
    """
    Convenience: complete and parse JSON response.
    Returns empty dict on parse failure.
    """
    text = await complete(
        prompt     = prompt,
        system     = system,
        model_tier = model_tier,
        model      = model,
        max_tokens = max_tokens,
        json_mode  = True
    )
    try:
        # Strip markdown fences if present
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        # Try harder: find JSON object in response
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}


# ── Provider implementations ──────────────────────────────────────────────────

async def _anthropic_complete(prompt: str, system: str, model: str,
                               max_tokens: int, temperature: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    body = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   messages,
    }
    if system:
        body["system"] = system
    # Anthropic doesn't support temperature=0 in the same way, skip if default
    if temperature != 0.7:
        body["temperature"] = temperature

    try:
        resp = await _http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json"
            },
            json=body
        )
        data = resp.json()
        if "content" in data and data["content"]:
            return data["content"][0]["text"]
        return f"[Anthropic error: {data.get('error', data)}]"
    except Exception as e:
        return f"[Anthropic call failed: {str(e)[:80]}]"


async def _openai_complete(prompt: str, system: str, model: str,
                            max_tokens: int, temperature: float,
                            json_mode: bool) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages":    messages,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        resp = await _http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type":  "application/json"
            },
            json=body
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        return f"[OpenAI error: {data.get('error', data)}]"
    except Exception as e:
        return f"[OpenAI call failed: {str(e)[:80]}]"


# ── Convenience aliases (backward compat with existing service code) ──────────

async def claude(prompt: str, system: str = "",
                 model: str = None, max_tokens: int = 1000) -> str:
    """Backward-compatible alias. Works with both providers."""
    return await complete(prompt, system, model=model, max_tokens=max_tokens)


async def claude_json(prompt: str, system: str = "",
                      model: str = None, max_tokens: int = 800) -> dict:
    """Backward-compatible alias."""
    return await complete_json(prompt, system, model=model, max_tokens=max_tokens)


def provider_info() -> dict:
    return {
        "provider":    PROVIDER,
        "has_key":     has_api_key(),
        "main_model":  get_model("main"),
        "fast_model":  get_model("fast"),
    }
