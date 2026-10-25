"""
llm/client.py
The single place where this project talks to an LLM.

Every module (entity extraction, query planning, Cypher generation, answer
generation) calls chat() or chat_json() from here, so provider selection,
rate limiting and retries are implemented exactly once.

Providers (all reached through the OpenAI-compatible chat API):
  gemini — Google Gemini       GEMINI_API_KEY   (generous free tier)
  groq   — Llama on Groq       GROQ_API_KEY     (fast; tight free-tier limits)
  openai — GPT models          OPENAI_API_KEY
  ollama — any local model     no key           (no rate limits at all)

Environment variables (all optional):
  LLM_PROVIDER            Force a provider instead of auto-detecting from keys
  <PROVIDER>_MODEL        Model name,        e.g. GROQ_MODEL=llama-3.1-8b-instant
  <PROVIDER>_RPM          Requests / minute, e.g. GROQ_RPM=30
  <PROVIDER>_TPM          Tokens / minute,   e.g. GROQ_TPM=8000
  LLM_REASONING_EFFORT    Thinking budget for reasoning models (see REASONING_DEFAULTS)
  LLM_MAX_WAIT_SECONDS    Longest we will sleep on a 429 before giving up (120)
  OLLAMA_BASE_URL         Default http://localhost:11434/v1

How rate limits are handled:
  1. Before each call, the shared RateLimiter waits until the request fits
     under the provider's RPM/TPM budget (proactive — avoids most 429s).
  2. If the server still returns 429, we read its "retry after" hint, pause
     all callers for that long, and retry.
  3. If the hint is very long (e.g. a daily quota is used up) we raise
     RateLimitExhausted instead of hanging, so callers can stop cleanly.
"""

from __future__ import annotations
import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

from dotenv import load_dotenv

from llm.rate_limiter import RateLimiter

load_dotenv()


# ── Provider table ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Provider:
    name:          str
    base_url:      Optional[str]   # None = OpenAI's default endpoint
    api_key_env:   Optional[str]   # None = no key needed (Ollama)
    default_model: str
    default_rpm:   int             # 0 = unlimited
    default_tpm:   int             # 0 = unlimited


# Default limits are conservative free-tier values. Raise them with
# <PROVIDER>_RPM / <PROVIDER>_TPM if your account allows more.
PROVIDERS: Dict[str, Provider] = {
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.0-flash",
        default_rpm=15, default_tpm=1_000_000,
    ),
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        default_rpm=30, default_tpm=6_000,
    ),
    "openai": Provider(
        name="openai",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
        default_rpm=500, default_tpm=30_000,
    ),
    "ollama": Provider(
        name="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key_env=None,
        default_model="llama3.1:8b",
        default_rpm=0, default_tpm=0,
    ),
}

# Order used by provider="auto" when LLM_PROVIDER is not set.
AUTO_DETECT_ORDER = ["gemini", "groq", "openai"]

# Reasoning ("thinking") models spend hidden tokens before answering. Those
# tokens count against max_tokens AND the tokens/min budget, so we keep the
# thinking to a minimum. Per model-family defaults; LLM_REASONING_EFFORT overrides.
REASONING_DEFAULTS = {
    "openai/gpt-oss": "low",    # lowest setting gpt-oss supports
    "qwen/qwen3":     "none",   # qwen3 can switch thinking off entirely
}
# Extra output budget for models that still think (gpt-oss at "low" ≈ 500 tokens).
REASONING_HEADROOM_TOKENS = 1024

RATE_LIMIT_RETRIES = 6
TRANSIENT_RETRIES  = 3
CHARS_PER_TOKEN    = 4   # rough estimate; real usage is recorded after each call


# Tokens and calls per model since the process started (see usage_summary()).
_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "tokens": 0})
_usage_lock = threading.Lock()


def usage_summary() -> Dict[str, Dict[str, int]]:
    """{model: {"calls": N, "tokens": M}} for every LLM call made so far."""
    with _usage_lock:
        return {model: dict(counts) for model, counts in _usage.items()}


class RateLimitExhausted(RuntimeError):
    """The provider's quota is used up for longer than we are willing to wait."""

    def __init__(self, provider: str, wait_seconds: Optional[float], message: str):
        self.provider     = provider
        self.wait_seconds = wait_seconds
        hint = f" (server says retry in ~{wait_seconds / 60:.0f} min)" if wait_seconds else ""
        super().__init__(f"{provider} rate limit exhausted{hint}: {message}")


# ── Provider selection ────────────────────────────────────────────────────────

def resolve_provider(provider: str = "auto") -> str:
    """
    Turn "auto" into a concrete provider name.

    Priority: explicit argument → LLM_PROVIDER env var → first API key found.
    """
    provider = (provider or "auto").lower()
    if provider == "auto":
        provider = os.getenv("LLM_PROVIDER", "auto").lower()

    if provider != "auto":
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{provider}'. "
                             f"Choose from: auto, {', '.join(PROVIDERS)}")
        return provider

    for name in AUTO_DETECT_ORDER:
        if os.getenv(PROVIDERS[name].api_key_env):
            return name

    raise EnvironmentError(
        "No LLM configured. Set GEMINI_API_KEY, GROQ_API_KEY or OPENAI_API_KEY in .env, "
        "or LLM_PROVIDER=ollama to use a local model."
    )


def model_for(provider: str) -> str:
    """Model name for a provider, overridable via <PROVIDER>_MODEL."""
    return os.getenv(f"{provider.upper()}_MODEL", PROVIDERS[provider].default_model)


@lru_cache(maxsize=None)
def _get_client(provider: str):
    """One reusable HTTP client per provider (the SDK's own retries are off)."""
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    if cfg.api_key_env:
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise EnvironmentError(f"{cfg.api_key_env} is not set in .env")
    else:
        api_key = "not-needed"

    return OpenAI(api_key=api_key, base_url=cfg.base_url, max_retries=0, timeout=120)


@lru_cache(maxsize=None)
def _get_limiter(provider: str) -> RateLimiter:
    """One shared limiter per provider, so all callers draw from one budget."""
    cfg = PROVIDERS[provider]
    rpm = int(os.getenv(f"{provider.upper()}_RPM", cfg.default_rpm))
    tpm = int(os.getenv(f"{provider.upper()}_TPM", cfg.default_tpm))
    return RateLimiter(provider, rpm=rpm, tpm=tpm)


# ── Public API ────────────────────────────────────────────────────────────────

def chat(
    system:      str,
    user:        str,
    provider:    str = "auto",
    model:       Optional[str] = None,
    max_tokens:  int = 1024,
    temperature: float = 0.0,
    json_mode:   bool = False,
) -> str:
    """
    Send one system + user message pair and return the reply text.

    Waits as needed to respect rate limits and retries transient failures.
    Raises RateLimitExhausted if the provider's quota will not recover soon.
    """
    import openai

    provider = resolve_provider(provider)
    model    = model or model_for(provider)
    client   = _get_client(provider)
    limiter  = _get_limiter(provider)

    request = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    effort = _reasoning_effort(provider, model)
    if effort:
        request["extra_body"] = {"reasoning_effort": effort}
        if effort != "none":
            request["max_tokens"] = max_tokens = max_tokens + REASONING_HEADROOM_TOKENS

    # Providers count max_tokens against the per-minute budget up front,
    # so reserve prompt + max_tokens and correct it after the call.
    estimated_tokens = (len(system) + len(user)) // CHARS_PER_TOKEN + max_tokens

    rate_limit_hits = 0
    transient_hits  = 0
    while True:
        reservation = limiter.acquire(estimated_tokens)
        try:
            response = client.chat.completions.create(**request)

        except openai.RateLimitError as exc:
            limiter.cancel(reservation)
            if "request too large" in str(exc).lower():
                # Bigger than the whole tokens/min budget — waiting cannot help.
                raise RateLimitExhausted(
                    provider, None,
                    "single request exceeds the tokens/min limit; use smaller "
                    "chunks or a lower max_tokens. " + _short(exc)) from exc
            rate_limit_hits += 1
            wait = _retry_after_seconds(exc)
            max_wait = float(os.getenv("LLM_MAX_WAIT_SECONDS", 120))

            if _is_daily_quota(exc) or (wait and wait > max_wait) \
                    or rate_limit_hits > RATE_LIMIT_RETRIES:
                raise RateLimitExhausted(provider, wait, _short(exc)) from exc

            wait = wait or min(60.0, 5.0 * 2 ** (rate_limit_hits - 1))
            print(f"  [rate-limit] {provider} returned 429 — waiting {wait:.1f}s "
                  f"(retry {rate_limit_hits}/{RATE_LIMIT_RETRIES})")
            limiter.block_for(wait)
            continue

        except (openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as exc:
            limiter.cancel(reservation)
            transient_hits += 1
            if transient_hits > TRANSIENT_RETRIES:
                raise
            backoff = 2.0 ** transient_hits
            print(f"  [llm] {provider} transient error ({type(exc).__name__}), "
                  f"retrying in {backoff:.0f}s")
            time.sleep(backoff)
            continue

        except Exception:
            limiter.cancel(reservation)
            raise

        if response.usage:
            limiter.record_usage(reservation, response.usage.total_tokens)
            with _usage_lock:
                _usage[model]["calls"]  += 1
                _usage[model]["tokens"] += response.usage.total_tokens
        return response.choices[0].message.content or ""


def chat_json(system: str, user: str, **kwargs) -> dict:
    """Like chat(), but asks for JSON output and returns the parsed dict."""
    text = chat(system, user, json_mode=True, **kwargs)
    return parse_json_reply(text)


def parse_json_reply(text: str) -> dict:
    """Parse a JSON reply, tolerating ```json fences some local models add."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def _reasoning_effort(provider: str, model: str) -> Optional[str]:
    """reasoning_effort to send for this model, or None for non-reasoning models."""
    if provider != "groq":
        return None
    for prefix, default in REASONING_DEFAULTS.items():
        if model.startswith(prefix):
            return os.getenv("LLM_REASONING_EFFORT", default)
    return None


# ── 429 helpers ───────────────────────────────────────────────────────────────

# Matches "try again in 7.66s", "try again in 1m26.4s", "retry in 41.5s", "in 340ms"
_RETRY_IN_PATTERN = re.compile(r"(?:try again|retry) in\s+((?:[\d.]+(?:ms|h|m|s))+)", re.I)
_DURATION_PART    = re.compile(r"([\d.]+)(ms|h|m|s)", re.I)
_UNIT_SECONDS     = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """How long the server asked us to wait, from the header or the message."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header) + 1.0
        except ValueError:
            pass

    match = _RETRY_IN_PATTERN.search(str(exc))
    if match:
        seconds = sum(float(value) * _UNIT_SECONDS[unit.lower()]
                      for value, unit in _DURATION_PART.findall(match.group(1)))
        return seconds + 1.0
    return None


def _is_daily_quota(exc: Exception) -> bool:
    """True when a per-day limit (not a per-minute one) was hit."""
    message = str(exc).lower()
    return any(marker in message for marker in
               ("per day", "perday", "(tpd)", "(rpd)", "daily"))


def _short(exc: Exception, limit: int = 300) -> str:
    text = str(exc)
    return text if len(text) <= limit else text[:limit] + "…"
