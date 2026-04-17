# shared/llm_provider.py
# ── LLM Provider Abstraction ──────────────────────────────────────────────────
# Universal provider layer: Anthropic, OpenAI, Gemini, Ollama, DeepSeek,
# Groq, xAI (Grok), OpenRouter, or any OpenAI-compatible endpoint.
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
XAI_API_KEY        = os.getenv("XAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GENERIC_API_KEY    = os.getenv("GENERIC_API_KEY", "")
GENERIC_BASE_URL   = os.getenv("GENERIC_BASE_URL", "")

# Ollama config (no key needed — local)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")

# Web search — master toggle (set in .env or per-service env)
# Default: true — the model decides when to actually search based on the prompt
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"

# Providers that support server-side web search via their API
_WEB_SEARCH_PROVIDERS = {"openai", "anthropic", "xai", "gemini"}

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
        "main": os.getenv("OPENAI_MAIN_MODEL", "gpt-5.4-mini"),
        "fast": os.getenv("OPENAI_FAST_MODEL", "gpt-5.4-nano"),
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
        "main": os.getenv("DEEPSEEK_MAIN_MODEL", "deepseek-r1"),
        "fast": os.getenv("DEEPSEEK_FAST_MODEL", "deepseek-chat"),
    },
    "groq": {
        "main": os.getenv("GROQ_MAIN_MODEL", "llama-3.3-70b-versatile"),
        "fast": os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
    },
    "xai": {
        "main": os.getenv("XAI_MAIN_MODEL", "grok-4-20"),
        "fast": os.getenv("XAI_FAST_MODEL", "grok-3-mini-fast"),
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
    "xai":        lambda: (XAI_API_KEY, "https://api.x.ai/v1"),
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
    "xai":        lambda: XAI_API_KEY,
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
    max_tokens:  int   = 8192,
    temperature: float = 0.7,
    json_mode:   bool  = False,
    web_search:  bool  = False,
) -> str:
    """
    Universal completion function.
    Same call signature regardless of which provider is active.
    Returns the text response as a string.

    web_search: When True, enables server-side web search for providers that
    support it (OpenAI, Anthropic, xAI, Gemini). The model decides when to
    search based on the prompt. Falls back to normal completion for other providers.
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

    # ── Web search path (Responses API for OpenAI/xAI, native tools for Anthropic/Gemini)
    if web_search and PROVIDER in _WEB_SEARCH_PROVIDERS:
        if PROVIDER == "openai":
            api_key, base_url = _OPENAI_COMPAT_PROVIDERS["openai"]()
            return await _with_retry(
                _responses_api_search, base_url, api_key, prompt, system,
                chosen_model, max_tokens, temperature)
        if PROVIDER == "xai":
            api_key, base_url = _OPENAI_COMPAT_PROVIDERS["xai"]()
            return await _with_retry(
                _responses_api_search, base_url, api_key, prompt, system,
                chosen_model, max_tokens, temperature)
        if PROVIDER == "anthropic":
            return await _with_retry(
                _anthropic_search_complete, prompt, system, chosen_model,
                max_tokens, temperature)
        if PROVIDER == "gemini":
            return await _with_retry(
                _gemini_search_complete, prompt, system, chosen_model,
                max_tokens, temperature)

    # ── Standard completion path (no web search)
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


async def complete_with_search_details(
    prompt:      str,
    system:      str   = "",
    model_tier:  str   = "main",
    model:       str   = None,
    max_tokens:  int   = 8192,
    temperature: float = 0.7,
) -> tuple:
    """Complete with web search enabled, returning (text, search_details).

    search_details is a dict with:
      searched: bool — whether the model actually performed a web search
      queries:  list[str] — search queries the model used
      sources:  list[{url, title}] — pages the model consulted
      citations: list[{url, title}] — inline citations in the response

    The model decides autonomously whether to search based on the prompt.
    If the current provider doesn't support web search, falls back to
    normal completion with search_details = {"searched": False}.
    """
    if not has_api_key():
        return f"[No API key for provider={PROVIDER}]", {"searched": False}

    chosen_model = model or get_model(model_tier)
    max_tokens = _effective_max_tokens(max_tokens, model_tier)
    no_search = {"searched": False, "queries": [], "sources": [], "citations": []}

    if PROVIDER not in _WEB_SEARCH_PROVIDERS:
        text = await complete(prompt, system, model_tier=model_tier, model=model,
                              max_tokens=max_tokens, temperature=temperature)
        return text, no_search

    try:
        if PROVIDER == "openai":
            api_key, base_url = _OPENAI_COMPAT_PROVIDERS["openai"]()
            return await _responses_api_search_detailed(
                base_url, api_key, prompt, system, chosen_model,
                max_tokens, temperature)
        if PROVIDER == "xai":
            api_key, base_url = _OPENAI_COMPAT_PROVIDERS["xai"]()
            return await _responses_api_search_detailed(
                base_url, api_key, prompt, system, chosen_model,
                max_tokens, temperature)
        if PROVIDER == "anthropic":
            return await _anthropic_search_detailed(
                prompt, system, chosen_model, max_tokens, temperature)
        if PROVIDER == "gemini":
            return await _gemini_search_detailed(
                prompt, system, chosen_model, max_tokens, temperature)
    except Exception as e:
        print(f"[llm_provider] web search failed, falling back: {e}", flush=True)

    # Fallback to normal completion
    text = await complete(prompt, system, model_tier=model_tier, model=model,
                          max_tokens=max_tokens, temperature=temperature)
    return text, no_search


# ── Retry wrapper ─────────────────────────────────────────────────────────────

async def _with_retry(fn, *args, _retries: int = 1, **kwargs) -> str:
    """Call fn(*args, **kwargs); on failure, retry once after 1 s."""
    result = await fn(*args, **kwargs)
    if result.startswith("[") and ("error" in result.lower() or "failed" in result.lower()):
        await asyncio.sleep(1)
        result = await fn(*args, **kwargs)
    return result


# ── Web Search implementations ────────────────────────────────────────────────
# These use each provider's native server-side web search tools.
# The model decides when to search based on the prompt content.
# Each returns (text, search_details_dict) when called via complete_with_search_details().

def _extract_search_details_openai(data: dict) -> dict:
    """Extract search queries, sources, and citations from OpenAI/xAI Responses API output."""
    details: dict = {"searched": False, "queries": [], "sources": [], "citations": []}
    for item in data.get("output", []):
        if item.get("type") == "web_search_call":
            details["searched"] = True
            action = item.get("action", {})
            if isinstance(action, dict):
                # search action may include queries
                q = action.get("query") or action.get("queries")
                if q:
                    if isinstance(q, list):
                        details["queries"].extend(q)
                    else:
                        details["queries"].append(str(q))
                # sources list
                for src in action.get("sources", []):
                    details["sources"].append({
                        "url": src.get("url", ""),
                        "title": src.get("title", ""),
                    })
        if item.get("type") == "message":
            for part in item.get("content", []):
                for ann in part.get("annotations", []):
                    if ann.get("type") == "url_citation":
                        details["citations"].append({
                            "url": ann.get("url", ""),
                            "title": ann.get("title", ""),
                        })
    return details


def _extract_search_details_anthropic(data: dict) -> dict:
    """Extract search details from Anthropic Messages API response with web_search tool."""
    details: dict = {"searched": False, "queries": [], "sources": [], "citations": []}
    for block in data.get("content", []):
        btype = block.get("type", "")
        if btype == "server_tool_use" and block.get("name") == "web_search":
            details["searched"] = True
            inp = block.get("input", {})
            if inp.get("query"):
                details["queries"].append(inp["query"])
        if btype == "web_search_tool_result":
            for result in block.get("content", []):
                if result.get("type") == "web_search_result":
                    details["sources"].append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                    })
        if btype == "text":
            for ann in block.get("annotations", []):
                if ann.get("type") == "web_search_result_location":
                    details["citations"].append({
                        "url": ann.get("url", ""),
                        "title": ann.get("title", ""),
                    })
    return details


async def _responses_api_search(
    base_url: str, api_key: str, prompt: str, system: str,
    model: str, max_tokens: int, temperature: float,
) -> str:
    """OpenAI / xAI Responses API with web_search tool.

    Uses /v1/responses — the model performs live web search server-side
    and returns the answer with inline URL citations.
    """
    input_msgs: list = []
    if system:
        input_msgs.append({"role": "system", "content": system})
    input_msgs.append({"role": "user", "content": prompt})

    body: dict = {
        "model":            model,
        "input":            input_msgs,
        "tools":            [{"type": "web_search"}],
        "max_output_tokens": max_tokens,
    }
    if temperature != 0.7:
        body["temperature"] = temperature

    url = f"{base_url.rstrip('/')}/responses"
    headers: dict = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = await _http.post(url, headers=headers, json=body)
        data = resp.json()

        # Top-level output_text is the clean final answer
        if data.get("output_text"):
            return data["output_text"]

        # Fallback: walk the output array for the message block
        for item in reversed(data.get("output", [])):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and part.get("text"):
                        return part["text"]

        return f"[{PROVIDER} search error: {data.get('error', data)}]"
    except Exception as e:
        return f"[{PROVIDER} search failed: {str(e)[:80]}]"


async def _responses_api_search_detailed(
    base_url: str, api_key: str, prompt: str, system: str,
    model: str, max_tokens: int, temperature: float,
) -> tuple:
    """Like _responses_api_search but returns (text, search_details)."""
    input_msgs: list = []
    if system:
        input_msgs.append({"role": "system", "content": system})
    input_msgs.append({"role": "user", "content": prompt})

    body: dict = {
        "model":            model,
        "input":            input_msgs,
        "tools":            [{"type": "web_search"}],
        "max_output_tokens": max_tokens,
    }
    if temperature != 0.7:
        body["temperature"] = temperature

    url = f"{base_url.rstrip('/')}/responses"
    headers: dict = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = await _http.post(url, headers=headers, json=body)
        data = resp.json()
        details = _extract_search_details_openai(data)

        text = data.get("output_text", "")
        if not text:
            for item in reversed(data.get("output", [])):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text" and part.get("text"):
                            text = part["text"]
                            break
                    if text:
                        break

        if not text:
            text = f"[{PROVIDER} search error: {data.get('error', data)}]"
        return text, details
    except Exception as e:
        return f"[{PROVIDER} search failed: {str(e)[:80]}]", {"searched": False}


async def _anthropic_search_complete(
    prompt: str, system: str, model: str,
    max_tokens: int, temperature: float,
) -> str:
    """Anthropic Messages API with web_search server-side tool.

    Claude decides when to search based on the prompt. The search runs
    server-side — no client round-trips needed. Returns the final text
    block which includes inline citations.
    """
    messages = [{"role": "user", "content": prompt}]
    body: dict = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   messages,
        "tools":      [{"type": "web_search_20250305", "name": "web_search"}],
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

        # Extract the last text block (after any tool_use / search results)
        if "content" in data and data["content"]:
            for block in reversed(data["content"]):
                if block.get("type") == "text" and block.get("text"):
                    return block["text"]

        return f"[Anthropic search error: {data.get('error', data)}]"
    except Exception as e:
        return f"[Anthropic search failed: {str(e)[:80]}]"


async def _anthropic_search_detailed(
    prompt: str, system: str, model: str,
    max_tokens: int, temperature: float,
) -> tuple:
    """Like _anthropic_search_complete but returns (text, search_details)."""
    messages = [{"role": "user", "content": prompt}]
    body: dict = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   messages,
        "tools":      [{"type": "web_search_20250305", "name": "web_search"}],
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
        details = _extract_search_details_anthropic(data)

        text = ""
        if "content" in data and data["content"]:
            for block in reversed(data["content"]):
                if block.get("type") == "text" and block.get("text"):
                    text = block["text"]
                    break

        if not text:
            text = f"[Anthropic search error: {data.get('error', data)}]"
        return text, details
    except Exception as e:
        return f"[Anthropic search failed: {str(e)[:80]}]", {"searched": False}


async def _gemini_search_complete(
    prompt: str, system: str, model: str,
    max_tokens: int, temperature: float,
) -> str:
    """Google Gemini with Google Search grounding tool.

    Adds the google_search tool — Gemini fetches live search results
    and incorporates them into the response.
    """
    contents: list = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    body = {
        "contents": contents,
        "tools": [{"google_search": {}}],
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
        return f"[Gemini search error: {error.get('message', data)}]"
    except Exception as e:
        return f"[Gemini search failed: {str(e)[:80]}]"


async def _gemini_search_detailed(
    prompt: str, system: str, model: str,
    max_tokens: int, temperature: float,
) -> tuple:
    """Like _gemini_search_complete but returns (text, search_details)."""
    contents: list = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    body = {
        "contents": contents,
        "tools": [{"google_search": {}}],
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
        details: dict = {"searched": False, "queries": [], "sources": [], "citations": []}

        # Gemini grounding metadata
        grounding = data.get("candidates", [{}])[0].get("groundingMetadata", {})
        if grounding:
            details["searched"] = True
            for q in grounding.get("webSearchQueries", []):
                details["queries"].append(q)
            for chunk in grounding.get("groundingChunks", []):
                web = chunk.get("web", {})
                if web:
                    details["sources"].append({
                        "url": web.get("uri", ""),
                        "title": web.get("title", ""),
                    })

        text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")

        if not text:
            error = data.get("error", {})
            text = f"[Gemini search error: {error.get('message', data)}]"
        return text, details
    except Exception as e:
        return f"[Gemini search failed: {str(e)[:80]}]", {"searched": False}


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
        "temperature": temperature,
        "messages":    messages,
    }
    # GPT-5.x models use max_completion_tokens instead of max_tokens
    if model and ("gpt-5" in model or "o1" in model or "o3" in model):
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
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
        "keep_alive": -1,
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
    except (httpx.ConnectTimeout, httpx.ConnectError) as e:
        print(f"[llm_provider] ERROR: Ollama unreachable at {OLLAMA_URL} ({type(e).__name__})", flush=True)
        return f"[Ollama unreachable at {OLLAMA_URL}: {type(e).__name__}]"
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
        "web_search_enabled":   WEB_SEARCH_ENABLED,
        "web_search_supported": PROVIDER in _WEB_SEARCH_PROVIDERS,
    }
