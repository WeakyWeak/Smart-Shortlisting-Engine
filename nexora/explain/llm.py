"""Stage 8b - Groq client. Only ever receives the fact-sheet from factsheet.py, never
a resume or the JD. Every failure path falls back to the template."""
from __future__ import annotations

import json
import os

from .. import config
from .prompts import CHAT_SYSTEM, CHAT_USER, EXPLAIN_SYSTEM, EXPLAIN_USER

_CLIENT = None
_UNAVAILABLE = False


def _client():
    """Build the client lazily. Returns None if unusable, never raises."""
    global _CLIENT, _UNAVAILABLE
    if _UNAVAILABLE:
        return None
    if _CLIENT is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(config.ROOT / ".env")
        except ImportError:
            pass
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            _UNAVAILABLE = True
            return None
        try:
            from groq import Groq
            _CLIENT = Groq(api_key=key, timeout=config.GROQ_TIMEOUT,
                           max_retries=config.GROQ_MAX_RETRIES)
        except Exception:
            _UNAVAILABLE = True
            return None
    return _CLIENT


def available() -> bool:
    return _client() is not None


def _complete(system: str, user: str, max_tokens: int = 900) -> str | None:
    client = _client()
    if client is None:
        return None
    kwargs = {}
    if "gpt-oss" in config.GROQ_MODEL or "qwen" in config.GROQ_MODEL:
        kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    try:
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:  # noqa: BLE001
        print(f"  ! LLM call failed ({type(exc).__name__}: {exc}); using template fallback")
        return None


def explain(facts: dict) -> str | None:
    return _complete(
        EXPLAIN_SYSTEM,
        EXPLAIN_USER.format(facts=json.dumps(facts, indent=2), rank=facts["rank"]),
    )


def chat(facts: dict, question: str) -> str | None:
    return _complete(
        CHAT_SYSTEM,
        CHAT_USER.format(facts=json.dumps(facts, indent=2), question=question),
        max_tokens=1200,
    )
