"""
pipeline/models.py — Centralized LLM Model Initializations

This module serves as the single source of truth for all LLM model initializations
across the SDLC pipeline. Nodes import their models directly from here rather than
instantiating models locally.

Centralizing model initialization:
1. Keeps provider configuration and API key handling in one place.
2. Allows model swapping, temperature tuning, and fallback routing without
   modifying individual node implementations.
3. FallbackChain wraps two providers so that if the primary fails the secondary
   is tried automatically — no node code changes needed for resilience.

FallbackChain design:
- Drop-in replacement for any single LangChain chat model instance.
- invoke() tries providers in order, logs failures, raises only if all fail.
- with_structured_output() returns a _FallbackStructuredChain that proxies
  the same try-in-order logic through each provider's structured variant.
- None providers are filtered out at init (missing env vars -> skip silently).
- Embedding model is intentionally excluded from FallbackChain: switching
  embedding providers mid-run would corrupt the vector index.
- Thread timeout (35s per provider): ChatGoogleGenerativeAI ignores SDK-level
  timeout params, so we enforce a wall-clock deadline via ThreadPoolExecutor.
"""

import concurrent.futures
import logging
import os
from typing import Any, Type

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

_PROVIDER_TIMEOUT = 35  # seconds per provider; enforced via thread, not SDK


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------

class _FallbackStructuredChain:
    """
    Wraps a list of (model, structured_variant) pairs and tries each in order.
    Returned by FallbackChain.with_structured_output().
    """

    def __init__(self, structured_pairs: list[tuple[BaseChatModel, Any]]) -> None:
        self._pairs = structured_pairs

    def invoke(self, messages: list[BaseMessage], **kwargs) -> Any:
        last_exc: Exception | None = None
        for raw_model, structured in self._pairs:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    _fut = _ex.submit(structured.invoke, messages, **kwargs)
                    try:
                        return _fut.result(timeout=_PROVIDER_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        provider = getattr(raw_model, "model_name", repr(raw_model))
                        raise TimeoutError(
                            f"Provider {provider!r} timed out after {_PROVIDER_TIMEOUT}s"
                        )
            except Exception as exc:
                provider = getattr(raw_model, "model_name", repr(raw_model))
                logger.warning(
                    "Structured LLM call failed, trying next provider",
                    extra={"provider": provider, "error": str(exc)},
                )
                last_exc = exc
        raise RuntimeError(
            "All providers failed in _FallbackStructuredChain"
        ) from last_exc


class FallbackChain:
    """
    Drop-in replacement for a single LangChain chat model.

    Usage (identical to a raw model):
        chain = FallbackChain([primary_model, fallback_model])
        chain.invoke(messages)
        chain.with_structured_output(Schema).invoke(messages)

    Providers are tried left-to-right. None entries are removed at init so
    that optional providers (e.g. Groq when GROQ_API_KEY is unset) are
    silently skipped.

    Raises ValueError on construction if the resulting provider list is empty.
    """

    def __init__(self, models: list[BaseChatModel | None]) -> None:
        filtered = [m for m in models if m is not None]
        if not filtered:
            raise ValueError(
                "FallbackChain requires at least one non-None model. "
                "Check that the required API key env vars are set."
            )
        self._models = filtered

    def invoke(self, messages: list[BaseMessage], **kwargs) -> Any:
        last_exc: Exception | None = None
        for model in self._models:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    _fut = _ex.submit(model.invoke, messages, **kwargs)
                    try:
                        return _fut.result(timeout=_PROVIDER_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        provider = getattr(model, "model_name", repr(model))
                        raise TimeoutError(
                            f"Provider {provider!r} timed out after {_PROVIDER_TIMEOUT}s"
                        )
            except Exception as exc:
                provider = getattr(model, "model_name", repr(model))
                logger.warning(
                    "LLM call failed, trying next provider",
                    extra={"provider": provider, "error": str(exc)},
                )
                last_exc = exc
        raise RuntimeError(
            "All providers failed in FallbackChain"
        ) from last_exc

    def with_structured_output(
        self,
        schema: Type,
        **kwargs,
    ) -> _FallbackStructuredChain:
        """
        Returns a _FallbackStructuredChain that tries each provider's
        structured variant in order.
        """
        pairs = []
        for model in self._models:
            structured = model.with_structured_output(schema, **kwargs)
            pairs.append((model, structured))
        return _FallbackStructuredChain(pairs)


# ---------------------------------------------------------------------------
# Provider instantiation helpers
# ---------------------------------------------------------------------------

def _make_gemini(key_env: str = "GOOGLE_API_KEY") -> ChatGoogleGenerativeAI | None:
    """Returns a Gemini Flash instance, or None if the key is missing."""
    key = os.getenv(key_env)
    if not key:
        logger.warning("GOOGLE_API_KEY not set — Gemini provider unavailable")
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=key,
        max_retries=0,  # Don't retry — FallbackChain handles fallback
    )


def _gemini_all() -> list[ChatGoogleGenerativeAI]:
    """
    One ChatGoogleGenerativeAI per configured key, tried left-to-right.
    Keys MUST be from different GCP projects for independent quota pools.
    .env: GOOGLE_API_KEY (required), GOOGLE_API_KEY_2, GOOGLE_API_KEY_3
    """
    return [m for m in (
        _make_gemini("GOOGLE_API_KEY"),
        _make_gemini("GOOGLE_API_KEY_2"),
        _make_gemini("GOOGLE_API_KEY_3"),
    ) if m is not None]


def _make_groq() -> BaseChatModel | None:
    """Returns a Groq instance, or None if the key is missing."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        from langchain_groq import ChatGroq  # noqa: PLC0415
        return ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=key,
            max_retries=0,  # Don't retry — FallbackChain handles fallback
        )
    except Exception as exc:
        logger.warning("Groq provider init failed", extra={"error": str(exc)})
        return None


# ---------------------------------------------------------------------------
# Exported model objects
# ---------------------------------------------------------------------------

# All three model objects use _gemini_all() so that quota is spread across
# up to three independent GCP projects before falling back to Groq.
review_model: FallbackChain = FallbackChain([*_gemini_all(), _make_groq()])
coding_model: FallbackChain = FallbackChain([*_gemini_all(), _make_groq()])
flash_model:  FallbackChain = FallbackChain([*_gemini_all(), _make_groq()])

# Intentionally NOT wrapped in FallbackChain — switching embedding providers
# mid-run would produce incompatible vector representations and corrupt the index.
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
)
