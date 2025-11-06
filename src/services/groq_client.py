"""
Groq adapter using the OpenAI-compatible client shipped by the `openai` Python package.

Groq exposes an OpenAI-compatible surface and can be used by configuring the OpenAI client
with a custom base_url and API key, for example (from Groq docs):

    import os
    import openai
    client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))

This module provides a Thin `GroqClient` wrapper that constructs such a client and exposes
`chat.completions.create` and `chat.completions.stream` (via the underlying OpenAI client)
so `LlmBroker` can use it interchangeably with the OpenAI client.

Security: Do NOT hardcode API keys. Read them from env `GROQ_API_KEY`.
"""
from __future__ import annotations
import os
from typing import Optional

try:
    import openai
except Exception as e:  # pragma: no cover - import problems handled at runtime
    openai = None

DEFAULT_GROQ_BASE = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")

class GroqClient:
    """Adapter that instantiates an OpenAI-compatible client pointing at Groq.

    Usage:
        GroqClient(api_key=os.environ['GROQ_API_KEY'])

    The returned object exposes `.chat.completions.create(...)` and `.chat.completions.stream(...)`
    because it delegates to `openai.OpenAI(...).chat`.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if openai is None:
            raise RuntimeError("The 'openai' package is required for Groq adapter but is not installed.")
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not provided to GroqClient")
        base = base_url or DEFAULT_GROQ_BASE
        # Use the OpenAI-compatible client but point it to the Groq base_url
        # `openai.OpenAI` accepts base_url and api_key per Groq docs
        try:
            self._client = openai.OpenAI(base_url=base, api_key=key)
        except TypeError:
            # Some versions of the openai package accept different param names; try alternative init
            try:
                self._client = openai.OpenAI(api_key=key, base_url=base)
            except Exception:
                raise
        # Expose `.chat` so callers (LlmBroker) can use `.chat.completions.create` and `.chat.completions.stream`
        self.chat = self._client.chat

    def __repr__(self) -> str:  # helpful for debugging
        return f"<GroqClient base={self._client._base_url if hasattr(self._client,'_base_url') else 'unknown'}>"

__all__ = ["GroqClient"]
