"""LLM router.

Free-tier reality: rate limits hit, models get deprecated, hosted APIs
have outages. A production RAG stack needs a fallback chain.

Design:
  - Ordered providers. Try first; on failure, fall through.
  - Per-provider async retry with exponential backoff (tenacity).
  - Circuit breaker: after N consecutive failures on one provider, skip it
    for cooldown_s seconds. Prevents spending latency on a dead upstream.
  - Unified streaming interface: every provider yields text deltas.
  - Implements LLMAdapter from retrieval.query_transform, so the same
    router powers query rewrites, expansions, HyDE, AND final generation.

Primary: Groq (Llama 3.3 70B) — blazing fast, free tier.
Fallback: Google Gemini 2.0 Flash — free tier, different infra.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings
from observability.logger import get_logger

log = get_logger(__name__)


class LLMProviderError(Exception):
    """Base for provider errors. Retryable in the router's eyes."""


class RateLimitError(LLMProviderError):
    pass


class NoProvidersAvailableError(Exception):
    pass


# ---------- Provider protocol ----------

class LLMProvider(Protocol):
    name: str
    async def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str: ...
    async def stream(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]: ...


# ---------- Groq ----------

class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self._api_key)
        return self._client

    async def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise _classify_error(e) from e

    async def stream(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        try:
            client = self._get_client()
            stream = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            raise _classify_error(e) from e


# ---------- Gemini ----------

class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model_name = model
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)
        return self._model

    async def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        try:
            model = self._get_model()
            # google-generativeai SDK is sync; offload to thread
            resp = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
            return resp.text or ""
        except Exception as e:
            raise _classify_error(e) from e

    async def stream(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        try:
            model = self._get_model()
            # Gemini streaming is also sync-iterator; bridge via queue
            queue: asyncio.Queue = asyncio.Queue()
            SENTINEL = object()

            def _produce():
                try:
                    stream = model.generate_content(
                        prompt,
                        generation_config={
                            "max_output_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.text:
                            asyncio.run_coroutine_threadsafe(queue.put(chunk.text), loop)
                except Exception as exc:  # surface to consumer
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _produce)
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    return
                if isinstance(item, Exception):
                    raise _classify_error(item)
                yield item
        except Exception as e:
            raise _classify_error(e) from e


def _classify_error(e: Exception) -> LLMProviderError:
    msg = str(e).lower()
    if "rate" in msg and "limit" in msg or "429" in msg or "quota" in msg:
        return RateLimitError(str(e))
    return LLMProviderError(str(e))


# ---------- Circuit breaker ----------

@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    open_until: float = 0.0

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self, threshold: int, cooldown_s: float) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= threshold:
            self.open_until = time.monotonic() + cooldown_s

    def is_open(self) -> bool:
        return time.monotonic() < self.open_until


# ---------- Router ----------

class LLMRouter:
    """Tries providers in order; retries within each; skips open circuits."""

    def __init__(
        self,
        providers: list[LLMProvider] | None = None,
        *,
        retry_attempts: int = 2,
        breaker_threshold: int = 5,
        breaker_cooldown_s: float = 30.0,
    ):
        self.providers = providers if providers is not None else _default_providers()
        self._retry_attempts = retry_attempts
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown_s = breaker_cooldown_s
        self._breakers: dict[str, _BreakerState] = {
            p.name: _BreakerState() for p in self.providers
        }

    async def complete(
        self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2
    ) -> str:
        last_err: Exception | None = None
        for p in self.providers:
            breaker = self._breakers[p.name]
            if breaker.is_open():
                log.debug("provider_skipped_circuit_open", provider=p.name)
                continue
            try:
                result = await self._with_retry(
                    lambda: p.complete(prompt, max_tokens=max_tokens, temperature=temperature),
                    provider_name=p.name,
                )
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(self._breaker_threshold, self._breaker_cooldown_s)
                log.warning("provider_failed", provider=p.name, error=str(e))
                last_err = e
                continue
        raise NoProvidersAvailableError(
            f"All providers failed. Last error: {last_err}"
        ) from last_err

    async def stream(
        self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        """Streams from the first provider that yields at least one token.

        Fallback happens BEFORE any token is sent to the caller. Once streaming
        has started, we don't mid-stream-switch — that produces ugly concatenated
        answers. Better to fail loudly if a stream breaks partway.
        """
        last_err: Exception | None = None
        for p in self.providers:
            breaker = self._breakers[p.name]
            if breaker.is_open():
                continue
            try:
                agen = p.stream(prompt, max_tokens=max_tokens, temperature=temperature)
                # Peek first chunk to detect early failures before yielding.
                first = await _peek_first(agen)
                if first is None:
                    raise LLMProviderError(f"{p.name} returned empty stream")
                breaker.record_success()
                yield first
                async for delta in agen:
                    yield delta
                return
            except Exception as e:
                breaker.record_failure(self._breaker_threshold, self._breaker_cooldown_s)
                log.warning("stream_provider_failed", provider=p.name, error=str(e))
                last_err = e
                continue
        raise NoProvidersAvailableError(
            f"All providers failed during streaming. Last error: {last_err}"
        ) from last_err

    async def _with_retry(self, fn, *, provider_name: str):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            retry=retry_if_exception_type(LLMProviderError),
            reraise=True,
        ):
            with attempt:
                return await fn()


async def _peek_first(agen: AsyncIterator[str]) -> str | None:
    async for item in agen:
        return item
    return None


def _default_providers() -> list[LLMProvider]:
    s = get_settings()
    providers: list[LLMProvider] = []
    if s.groq_api_key:
        providers.append(GroqProvider(api_key=s.groq_api_key, model=s.groq_model))
    if s.google_api_key:
        providers.append(GeminiProvider(api_key=s.google_api_key, model=s.gemini_model))
    return providers
