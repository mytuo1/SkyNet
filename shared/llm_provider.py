# shared/llm_provider.py
# ── LLM Provider Abstraction ──────────────────────────────────────────────────
# Universal provider layer: Anthropic, OpenAI, Gemini, Ollama, DeepSeek,
# Groq, OpenRouter, or any OpenAI-compatible endpoint.
# All services import from here — never call APIs directly.
# Set LLM_PROVIDER env var to switch providers.

import asyncio
import json
import os
import re
from typing import Optional

import httpx

_http = httpx.AsyncClient(timeout=120.0)

# ── Provider configuration ────────────────────────────────────────────────────

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

# API keys
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY      = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GENERIC_API_KEY    = os.getenv("GENERIC_API_KEY", "")
GENERIC_BASE_URL   = os.getenv("GENERIC_BASE_URL", "")

# Ollama config (no key needed — local)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")

# Token limit overrides
MAX_TOKENS_MAIN = int(os.getenv("MAX_TOKENS_MAIN", "0")) or None
MAX_TOKENS_FAST = int(os.getenv("MAX_TOKENS_FAST", "0")) or None

# Model mapping — same logical role, different model names per provider
MODELS = {
    "anthropic": {
        "main": os.getenv("ANTHROPIC_MAIN_MODEL", "claude-sonnet-4-20250514"),
        "fast": os.getenv("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5-20251001"),
    },
    "openai": {
        "main": os.getenv("OPENAI_MAIN_MODEL", "gpt-4o"),
        "fast": os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini"),
    },
    "gemini": {
        "main": os.getenv("GEMINI_MAIN_MODEL", "gemini-2.5-pro"),
        "fast": os.getenv("GEMINI_FAST_MODEL", "gemini-2.0-flash"),
    },
    "ollama": {
        "main": os.getenv("OLLAMA_LLM_MODEL", "phi4"),
        "fast": os.getenv("OLLAMA_LLM_MODEL", "phi4"),
    },
    "deepseek": {
        "main": os.getenv("DEEPSEEK_MAIN_MODEL", "deepseek-reasoner"),
        "fast": os.getenv("DEEPSEEK_FAST_MODEL", "deepseek-chat"),
    },
    "groq": {
        "main": os.getenv("GROQ_MAIN_MODEL", "llama-3.3-70b-versatile"),
        "fast": os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
    },
    "openrouter": {
        "main": os.getenv("OPENROUTER_MAIN_MODEL", ""),
        "fast": os.getenv("OPENROUTER_FAST_MODEL", ""),
    },
    "generic": {
        "main": os.getenv("GENERIC_MAIN_MODEL", ""),
        "fast": os.getenv("GENERIC_FAST_MODEL", ""),
    },
}

# Maps provider name → (api_key, base_url) for OpenAI-compatible providers
_OPENAI_COMPAT_PROVIDERS = {
    "openai":     lambda: (OPENAI_KEY, "https://api.openai.com/v1"),
    "deepseek":   lambda: (DEEPSEEK_API_KEY, "https://api.deepseek.com/v1"),
    "groq":       lambda: (GROQ_API_KEY, "https://api.groq.com/openai/v1"),
    "openrouter": lambda: (OPENROUTER_API_KEY, "https://openrouter.ai/api/v1"),
    "generic":    lambda: (GENERIC_API_KEY, GENERIC_BASE_URL),
}

_API_KEY_MAP = {
    "anthropic":  lambda: ANTHROPIC_KEY,
    "openai":     lambda: OPENAI_KEY,
    "gemini":     lambda: GEMINI_API_KEY,
    "ollama":     lambda: True,  # no key needed
    "deepseek":   lambda: DEEPSEEK_API_KEY,
    "groq":       lambda: GROQ_API_KEY,
    "openrouter": lambda: OPENROUTER_API_KEY,
    "generic":    lambda: GENERIC_API_KEY,
}


def get_model(tier: str = "main") -> str:
    """Get the right model name for current provider and tier."""
    return MODELS.get(PROVIDER, MODELS["anthropic"]).get(tier, "")


def has_api_key() -> bool:
    """Check if current provider has an API key configured."""
    getter = _API_KEY_MAP.get(PROVIDER)
    if getter:
        return bool(getter())
    return False


def _effective_max_tokens(max_tokens: int, model_tier: str) -> int:
    """Apply per-tier env-var overrides for max_tokens."""
    if model_tier == "main" and MAX_TOKENS_MAIN:
        return MAX_TOKENS_MAIN
    if model_tier == "fast" and MAX_TOKENS_FAST:
        return MAX_TOKENS_FAST
    return max_tokens


# ── Core completion function ──────────────────────────────────────────────────

async def complete(
    prompt:      str,
    system:      str   = "",
    model_tier:  str   = "fast",
    model:       str   = None,
    max_tokens:  int   = 4096,
    temperature: float = 0.7,
    json_mode:   bool  = False,
) -> str:
    """
    Universal completion function.
    Same call signature regardless of which provider is active.
    Returns the text response as a string.
    """
    if not has_api_key():
        return f"[No API key for provider={PROVIDER}]"

    chosen_model = model or get_model(model_tier)
    max_tokens = _effective_max_tokens(max_tokens, model_tier)

    # Rough token-limit warning (1 token ≈ 4 chars)
    approx_input_tokens = (len(prompt) + len(system)) // 4
    if approx_input_tokens + max_tokens > 128_000:
        print(
            f"[llm_provider] WARNING: estimated input ({approx_input_tokens}) "
            f"+ max_tokens ({max_tokens}) may exceed provider context window"
        )

    if json_mode and system:
        system += "\n\nRespond only with valid JSON. No markdown code blocks."
    elif json_mode:
        system = "Respond only with valid JSON. No markdown code blocks."

    # Route to the right backend
    if PROVIDER == "anthropic":
        return await _with_retry(
            _anthropic_complete, prompt, system, chosen_model,
            max_tokens, temperature)
    if PROVIDER == "gemini":
        return await _with_retry(
            _gemini_complete, prompt, system, chosen_model,
            max_tokens, temperature)
    if PROVIDER == "ollama":
        return await _with_retry(
            _ollama_complete, prompt, system, chosen_model,
            max_tokens, temperature)
    if PROVIDER in _OPENAI_COMPAT_PROVIDERS:
        api_key, base_url = _OPENAI_COMPAT_PROVIDERS[PROVIDER]()
        return await _with_retry(
            _openai_compatible_complete, base_url, api_key, prompt, system,
            chosen_model, max_tokens, temperature, json_mode)

    return f"[Unknown provider: {PROVIDER}]"


async def complete_json(
    prompt:     str,
    system:     str    = "",
    model_tier: str    = "fast",
    model:      str    = None,
    max_tokens: int    = 2048,
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
        json_mode  = True,
    )
    try:
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}


# ── Retry wrapper ─────────────────────────────────────────────────────────────

async def _with_retry(fn, *args, _retries: int = 1, **kwargs) -> str:
    """Call fn(*args, **kwargs); on failure, retry once after 1 s."""
    result = await fn(*args, **kwargs)
    if result.startswith("[") and ("error" in result.lower() or "failed" in result.lower()):
        await asyncio.sleep(1)
        result = await fn(*args, **kwargs)
    return result


# ── Provider implementations ──────────────────────────────────────────────────

async def _anthropic_complete(prompt: str, system: str, model: str,
                               max_tokens: int, temperature: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    body: dict = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   messages,
    }
    if system:
        body["system"] = system
    if temperature != 0.7:
        body["temperature"] = temperature

    try:
        resp = await _http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json=body,
        )
        data = resp.json()
        if "content" in data and data["content"]:
            return data["content"][0]["text"]
        return f"[Anthropic error: {data.get('error', data)}]"
    except Exception as e:
        return f"[Anthropic call failed: {str(e)[:80]}]"


async def _openai_compatible_complete(
    base_url: str, api_key: str, prompt: str, system: str,
    model: str, max_tokens: int, temperature: float,
    json_mode: bool,
) -> str:
    """Shared implementation for OpenAI, DeepSeek, Groq, OpenRouter, generic."""
    messages: list = []
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

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers: dict = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = await _http.post(url, headers=headers, json=body)
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        return f"[{PROVIDER} error: {data.get('error', data)}]"
    except Exception as e:
        return f"[{PROVIDER} call failed: {str(e)[:80]}]"


async def _gemini_complete(prompt: str, system: str, model: str,
                            max_tokens: int, temperature: float) -> str:
    """Google Gemini via REST (generativelanguage.googleapis.com)."""
    contents: list = []
    if system:
        contents.append({
            "role": "user",
            "parts": [{"text": system}],
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Understood."}],
        })
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}],
    })

    body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     temperature,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )

    try:
        resp = await _http.post(
            url,
            headers={"Content-Type": "application/json"},
            json=body,
        )
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        error = data.get("error", {})
        return f"[Gemini error: {error.get('message', data)}]"
    except Exception as e:
        return f"[Gemini call failed: {str(e)[:80]}]"


async def _ollama_complete(prompt: str, system: str, model: str,
                            max_tokens: int, temperature: float) -> str:
    """Local Ollama instance via /api/chat."""
    messages: list = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "options": {
            "num_predict":  max_tokens,
            "temperature":  temperature,
        },
    }

    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"

    try:
        resp = await _http.post(url, json=body)
        data = resp.json()
        msg = data.get("message", {})
        if msg.get("content"):
            return msg["content"]
        return f"[Ollama error: {data.get('error', data)}]"
    except Exception as e:
        return f"[Ollama call failed: {str(e)[:80]}]"


# ── Convenience aliases (backward compat with existing service code) ──────────

async def claude(prompt: str, system: str = "",
                 model: str = None, max_tokens: int = 4096) -> str:
    """Backward-compatible alias. Works with any provider."""
    return await complete(prompt, system, model=model, max_tokens=max_tokens)


async def claude_json(prompt: str, system: str = "",
                      model: str = None, max_tokens: int = 2048) -> dict:
    """Backward-compatible alias."""
    return await complete_json(prompt, system, model=model, max_tokens=max_tokens)


# ── Info / diagnostics ────────────────────────────────────────────────────────

def provider_info() -> dict:
    """Return info about active provider and all configured providers."""
    configured = []
    for name, key_fn in _API_KEY_MAP.items():
        if bool(key_fn()):
            configured.append(name)

    return {
        "provider":             PROVIDER,
        "has_key":              has_api_key(),
        "main_model":           get_model("main"),
        "fast_model":           get_model("fast"),
        "configured_providers": configured,
    }
