"""Drop-in replacement for ``aibridge_exec.providers``, backed by the Rust engine.

Same names, same arguments, same return types and the same errors, so switching is one import:

    # from aibridge_exec.providers import ChatClient, ProviderError
    from aibridge_core.providers import ChatClient, ProviderError

    with ChatClient() as client:
        reply = client.chat("anthropic", api_key=KEY, model="<model id>",
                            messages=[{"role": "user", "content": "Hello"}])
        print(reply.text, reply.usage)

        for event in client.stream("openai", api_key=KEY, model="<model id>",
                                   messages=[{"role": "user", "content": "Hi"}]):
            if event.type == "delta":
                print(event.text, end="")

Differences from aibridge-exec:
- ``http=`` (an httpx client) isn't accepted: the HTTP stack is Rust. Use ``base_urls`` and ``timeout``.
- ``register_provider`` with a Python adapter isn't supported; add adapters in Rust (see README).
- New: ``cache=CacheOptions(...)`` turns on the in-process cache for model lists and chat replies.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from . import _native
from .errors import ProviderError
from .types import ChatMessage, ChatRequest, ChatResponse, ModelInfo, ProviderInfo, StreamEvent, Usage

Messages = Iterable["ChatMessage | Mapping[str, Any]"]

#: (connect, read) seconds, the same defaults as aibridge-exec.
DEFAULT_TIMEOUT = (10.0, 300.0)
MAX_MODEL_PAGES = 20


@dataclass(frozen=True)
class CacheOptions:
    """In-process cache (off unless you pass one). Keys are never stored: entries are indexed by a
    salted hash of the key, so one key never sees another key's results.

    ``models_ttl``: seconds to keep model lists. ``chat_ttl``: seconds to keep chat replies.
    ``chat_policy``: ``"deterministic"`` caches only ``temperature=0`` requests, ``"all"`` caches every one.
    """

    models_ttl: float | None = None
    chat_ttl: float | None = None
    chat_policy: str = "deterministic"
    max_entries: int = 10_000


@dataclass(frozen=True)
class ProviderHandle:
    """What ``get_provider`` returns: the id and the provider's ``ProviderInfo``."""

    info: ProviderInfo

    @property
    def id(self) -> str:
        return self.info.id


def _load_providers() -> dict[str, ProviderHandle]:
    out = {}
    for id_, label, key_url, base_url, hint in _native.providers():
        out[id_] = ProviderHandle(ProviderInfo(id=id_, label=label, key_url=key_url, default_base_url=base_url,
                                               key_hint=hint))
    return out


_PROVIDERS = _load_providers()


def get_provider(provider_id: str) -> ProviderHandle:
    try:
        return _PROVIDERS[provider_id]
    except KeyError:
        raise ProviderError("unknown_provider", f"unknown provider {provider_id!r}", provider=provider_id) from None


def list_providers() -> list[ProviderHandle]:
    return list(_PROVIDERS.values())


def register_provider(adapter: object, *, replace: bool = False) -> None:
    raise NotImplementedError(
        "aibridge_core adapters are written in Rust (implement aibridge_core::ProviderAdapter). "
        "For any OpenAI-compatible endpoint, use provider 'openai_compatible' with base_url."
    )


# ---------------------------------------------------------------- shared helpers
def _request(model: str, messages: Messages, system: str | None, temperature: float | None,
             max_tokens: int | None) -> ChatRequest:
    try:
        return ChatRequest.build(model, messages, system=system, temperature=temperature, max_tokens=max_tokens)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderError("bad_request", f"invalid request: {exc}") from None


def _require_key(provider_id: str, api_key: str | None) -> str:
    if not api_key or not api_key.strip():
        raise ProviderError("auth", "an API key is required", provider=provider_id)
    return api_key.strip()


def _native_args(request: ChatRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "messages": [(m.role, m.content) for m in request.messages],
        "system": request.system,
        "temperature": None if request.temperature is None else float(request.temperature),
        "max_tokens": request.max_tokens,
    }


def _reply(raw: tuple) -> ChatResponse:
    text, model, finish, input_tokens, output_tokens = raw
    return ChatResponse(text=text, model=model, finish_reason=finish, usage=Usage(input_tokens, output_tokens))


def _event(raw: tuple) -> StreamEvent:
    kind, first, model, input_tokens, output_tokens = raw
    if kind == "delta":
        return StreamEvent("delta", text=first)
    return StreamEvent("done", finish_reason=first, usage=Usage(input_tokens, output_tokens), model=model)


def _models(raw: list[tuple]) -> list[ModelInfo]:
    return [ModelInfo(id_, name, context) for id_, name, context in raw]


def _timeouts(timeout: Any) -> tuple[float, float]:
    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, (int, float)):
        return float(timeout), float(timeout)
    if isinstance(timeout, tuple):
        return float(timeout[0]), float(timeout[1])
    # httpx.Timeout or anything with connect/read attributes
    connect = getattr(timeout, "connect", None) or DEFAULT_TIMEOUT[0]
    read = getattr(timeout, "read", None) or DEFAULT_TIMEOUT[1]
    return float(connect), float(read)


class _Base:
    def __init__(self, *, http: Any = None, base_urls: Mapping[str, str] | None = None, timeout: Any = DEFAULT_TIMEOUT,
                 cache: CacheOptions | None = None) -> None:
        if http is not None:
            raise TypeError("aibridge_core doesn't use httpx: pass base_urls= and timeout= instead of http=")
        self.base_urls = dict(base_urls or {})
        connect, read = _timeouts(timeout)
        cache = cache or CacheOptions()
        self._native = _native.NativeClient(
            base_urls=self.base_urls, connect_timeout=connect, read_timeout=read,
            cache_models_ttl=cache.models_ttl, cache_chat_ttl=cache.chat_ttl, cache_chat_policy=cache.chat_policy,
            cache_max_entries=cache.max_entries,
        )

    def cache_stats(self) -> dict | None:
        """Hit and miss counters per cache, or None when caching is off."""
        return self._native.cache_stats()


class ChatClient(_Base):
    """Synchronous client. Network waits release the GIL.

    >>> client = ChatClient()
    >>> reply = client.chat("openai", api_key="sk-...", model="gpt-4o-mini",
    ...                     messages=[{"role": "user", "content": "Hello"}])
    >>> reply.text
    """

    def close(self) -> None:
        """Nothing to release (kept for aibridge-exec compatibility)."""

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(self, provider: str, *, api_key: str, model: str, messages: Messages, system: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             base_url: str | None = None) -> ChatResponse:
        get_provider(provider)
        key = _require_key(provider, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        return _reply(self._native.chat(provider, key, base_url=base_url, **_native_args(request)))

    def stream(self, provider: str, *, api_key: str, model: str, messages: Messages, system: str | None = None,
               temperature: float | None = None, max_tokens: int | None = None,
               base_url: str | None = None) -> Iterator[StreamEvent]:
        # A generator, like aibridge-exec: nothing runs (or raises) until the first next().
        get_provider(provider)
        key = _require_key(provider, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        native = self._native.stream(provider, key, base_url=base_url, **_native_args(request))
        try:
            for batch in native:  # the engine hands over every event already decoded at once
                for raw in batch:
                    yield _event(raw)
        finally:
            native.close()

    def list_models(self, provider: str, *, api_key: str, base_url: str | None = None) -> list[ModelInfo]:
        get_provider(provider)
        key = _require_key(provider, api_key)
        return _models(self._native.list_models(provider, key, base_url=base_url))


class AsyncChatClient(_Base):
    """Asynchronous client with the same methods as ``ChatClient``; ``stream`` is an async iterator."""

    async def aclose(self) -> None:
        """Nothing to release (kept for aibridge-exec compatibility)."""

    async def __aenter__(self) -> "AsyncChatClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def chat(self, provider: str, *, api_key: str, model: str, messages: Messages, system: str | None = None,
                   temperature: float | None = None, max_tokens: int | None = None,
                   base_url: str | None = None) -> ChatResponse:
        get_provider(provider)
        key = _require_key(provider, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        return _reply(await self._native.achat(provider, key, base_url=base_url, **_native_args(request)))

    async def stream(self, provider: str, *, api_key: str, model: str, messages: Messages,
                     system: str | None = None, temperature: float | None = None, max_tokens: int | None = None,
                     base_url: str | None = None) -> AsyncIterator[StreamEvent]:
        get_provider(provider)
        key = _require_key(provider, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        native = self._native.astream(provider, key, base_url=base_url, **_native_args(request))
        try:
            async for batch in native:
                for raw in batch:
                    yield _event(raw)
        finally:
            await native.aclose()

    async def list_models(self, provider: str, *, api_key: str, base_url: str | None = None) -> list[ModelInfo]:
        get_provider(provider)
        key = _require_key(provider, api_key)
        return _models(await self._native.alist_models(provider, key, base_url=base_url))


def stream_event_json(event: StreamEvent) -> str:
    import json

    return json.dumps(event.to_dict(), separators=(",", ":"))


__all__ = [
    "AsyncChatClient",
    "CacheOptions",
    "ChatClient",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ModelInfo",
    "ProviderError",
    "ProviderHandle",
    "ProviderInfo",
    "StreamEvent",
    "Usage",
    "get_provider",
    "list_providers",
    "register_provider",
    "stream_event_json",
]
