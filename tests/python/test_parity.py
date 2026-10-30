"""aibridge_core.providers must behave exactly like aibridge_exec.providers.

Every scenario runs through both libraries against tests/parity/mock_provider.py and the results
(dataclasses, stream events, errors) must be equal. Tests that need aibridge-exec are skipped when
it isn't installed; the aibridge-core-only tests always run.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aibridge_core import providers as core

try:
    from aibridge_exec import providers as exec_providers  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    exec_providers = None

needs_exec = pytest.mark.skipif(exec_providers is None, reason="aibridge-exec is not installed")

MOCK = Path(__file__).resolve().parents[1] / "parity" / "mock_provider.py"
GOOD = "sk-test-good-key-1234"


@pytest.fixture(scope="session")
def mock_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(MOCK), str(port)])
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait()


@pytest.fixture
def base_urls(mock_url):
    return {"openai": f"{mock_url}/v1", "anthropic": f"{mock_url}/v1", "gemini": f"{mock_url}/v1beta",
            "groq": "http://127.0.0.1:9/unreachable"}


def outcome(fn):
    """Normalized result or error of a call, comparable across the two libraries."""
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "ProviderError":
            return ("error", exc.code, exc.message, exc.provider, exc.status)
        return ("exception", type(exc).__name__)
    if isinstance(value, list):
        return [v.to_dict() for v in value]
    if hasattr(value, "__next__"):
        out = []
        try:
            for event in value:
                out.append(event.to_dict())
        except Exception as exc:  # noqa: BLE001
            out.append(("error", exc.code, exc.message, exc.provider, exc.status))
        return out
    return value.to_dict()


def own(client):
    """The providers module of the library this client belongs to."""
    client = getattr(client, "_client", client)
    return exec_providers if type(client).__module__.startswith("aibridge_exec") else core


def msgs(*texts):
    return [{"role": "user", "content": t} for t in texts]


SCENARIOS = [
    ("models openai", lambda c: c.list_models("openai", api_key=GOOD)),
    ("models anthropic paginated", lambda c: c.list_models("anthropic", api_key=GOOD)),
    ("models gemini paginated", lambda c: c.list_models("gemini", api_key=GOOD)),
    ("models bad key", lambda c: c.list_models("openai", api_key="sk-wrong-key-999")),
    ("models no key", lambda c: c.list_models("openai", api_key="  ")),
    ("models unknown provider", lambda c: c.list_models("nope", api_key=GOOD)),
    ("models unreachable", lambda c: c.list_models("groq", api_key=GOOD)),
    ("models openai_compatible without base_url", lambda c: c.list_models("openai_compatible", api_key=GOOD)),
    ("chat openai", lambda c: c.chat("openai", api_key=GOOD, model="m", system="sys", temperature=0.2, max_tokens=7,
                                     messages=[{"role": "system", "content": "x"}, *msgs("hi")])),
    ("chat anthropic merged turns", lambda c: c.chat("anthropic", api_key=GOOD, model="m", temperature=1.7,
                                                     messages=msgs("a", "b"))),
    ("chat gemini", lambda c: c.chat("gemini", api_key=GOOD, model="models/g", messages=msgs("hi"))),
    ("chat unicode", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=msgs("unicode"))),
    ("chat via ChatMessage objects", lambda c: c.chat("openai", api_key=GOOD, model="m",
                                                      messages=[own(c).ChatMessage("user", "hi")])),
    ("chat rate limited", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=msgs("please fail"))),
    ("chat 500 html", lambda c: c.chat("anthropic", api_key=GOOD, model="m", messages=msgs("server error"))),
    ("chat invalid json", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=msgs("bad json"))),
    ("chat key redacted", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=msgs("echo key"))),
    ("chat unknown model", lambda c: c.chat("openai", api_key=GOOD, model="missing-model", messages=msgs("x"))),
    ("chat bad role", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=[{"role": "bot", "content": "x"}])),
    ("chat no messages", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=[])),
    ("chat temperature too high", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=msgs("x"), temperature=3)),
    ("chat missing content key", lambda c: c.chat("openai", api_key=GOOD, model="m", messages=[{"role": "user"}])),
    ("stream openai", lambda c: c.stream("openai", api_key=GOOD, model="m", messages=msgs("hello there"))),
    ("stream anthropic", lambda c: c.stream("anthropic", api_key=GOOD, model="m", system="s", messages=msgs("hey"))),
    ("stream gemini", lambda c: c.stream("gemini", api_key=GOOD, model="g", max_tokens=3, messages=msgs("yo"))),
    ("stream unicode", lambda c: c.stream("anthropic", api_key=GOOD, model="m", messages=msgs("unicode"))),
    ("stream fails mid-way", lambda c: c.stream("openai", api_key=GOOD, model="m", messages=msgs("fail mid"))),
    ("stream anthropic error event", lambda c: c.stream("anthropic", api_key=GOOD, model="m", messages=msgs("fail mid"))),
    ("stream rate limited", lambda c: c.stream("openai", api_key=GOOD, model="m", messages=msgs("please fail"))),
    ("stream unknown provider", lambda c: c.stream("nope", api_key=GOOD, model="m", messages=msgs("x"))),
]


@needs_exec
@pytest.mark.parametrize("name,scenario", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_sync_parity(name, scenario, base_urls):
    with exec_providers.ChatClient(base_urls=base_urls) as a, core.ChatClient(base_urls=base_urls) as b:
        expected, actual = outcome(lambda: scenario(a)), outcome(lambda: scenario(b))
    assert actual == expected
    assert GOOD not in repr(actual)


@needs_exec
@pytest.mark.parametrize("name,scenario", SCENARIOS, ids=[s[0] for s in SCENARIOS])
async def test_async_parity(name, scenario, base_urls):
    async def run(client):
        call = scenario(_AsyncShim(client))
        try:
            if hasattr(call, "__anext__"):
                out = []
                try:
                    async for event in call:
                        out.append(event.to_dict())
                except Exception as exc:  # noqa: BLE001
                    out.append(("error", exc.code, exc.message, exc.provider, exc.status))
                return out
            value = await call
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "ProviderError":
                return ("error", exc.code, exc.message, exc.provider, exc.status)
            return ("exception", type(exc).__name__)
        return [v.to_dict() for v in value] if isinstance(value, list) else value.to_dict()

    async with exec_providers.AsyncChatClient(base_urls=base_urls) as a:
        expected = await run(a)
    async with core.AsyncChatClient(base_urls=base_urls) as b:
        actual = await run(b)
    assert actual == expected


class _AsyncShim:
    """Lets the sync scenario lambdas drive an async client (returns the coroutine / async iterator)."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)


@needs_exec
def test_provider_listing_matches():
    assert [p.info.to_dict() for p in core.list_providers()] == [p.info.to_dict() for p in exec_providers.list_providers()]
    a, b = core.get_provider("gemini"), exec_providers.get_provider("gemini")
    assert (a.id, a.info.to_dict(), a.info.default_base_url) == (b.id, b.info.to_dict(), b.info.default_base_url)


@needs_exec
def test_types_and_error_are_interchangeable_shapes():
    assert core.ChatRequest.build("m", msgs("x")) == core.ChatRequest.build("m", msgs("x"))
    error = core.ProviderError("auth", "nope", provider="openai", status=401)
    other = exec_providers.ProviderError("auth", "nope", provider="openai", status=401)
    assert error.to_dict() == other.to_dict() and repr(error) == repr(other)


# ---------------------------------------------------------------- aibridge-core only
def test_stream_is_lazy_like_a_generator(base_urls):
    client = core.ChatClient(base_urls=base_urls)
    events = client.stream("nope", api_key=GOOD, model="m", messages=msgs("x"))  # no error yet
    with pytest.raises(core.ProviderError) as info:
        next(events)
    assert info.value.code == "unknown_provider"


def test_breaking_out_of_a_stream_is_clean(base_urls):
    client = core.ChatClient(base_urls=base_urls)
    for event in client.stream("openai", api_key=GOOD, model="m", messages=msgs("one two three four")):
        assert event.type == "delta"
        break
    assert client.chat("openai", api_key=GOOD, model="m", messages=msgs("still works")).text


def test_cache_for_models_and_deterministic_chat(base_urls):
    client = core.ChatClient(base_urls=base_urls, cache=core.CacheOptions(models_ttl=60, chat_ttl=60))
    first = client.list_models("openai", api_key=GOOD)
    assert client.list_models("openai", api_key=GOOD) == first
    args = dict(api_key=GOOD, model="m", messages=msgs("cache me"), temperature=0)
    reply = client.chat("openai", **args)
    assert client.chat("openai", **args) == reply
    streamed = list(client.stream("openai", **args))
    assert streamed[0].text == reply.text and streamed[-1].type == "done"
    client.chat("openai", api_key=GOOD, model="m", messages=msgs("not cached"), temperature=0.5)
    stats = client.cache_stats()
    assert stats["models"]["hits"] == 1 and stats["chat"]["hits"] == 2
    # Another key never sees cached entries (it reaches the provider and is rejected).
    with pytest.raises(core.ProviderError):
        client.list_models("openai", api_key="sk-another-key-000")


@needs_exec
def test_accepts_aibridge_exec_message_objects(base_urls):
    reply = core.ChatClient(base_urls=base_urls).chat(
        "openai", api_key=GOOD, model="m", messages=[exec_providers.ChatMessage("user", "mixed")])
    assert "mixed" in reply.text


def test_no_cache_by_default(base_urls):
    assert core.ChatClient(base_urls=base_urls).cache_stats() is None


def test_httpx_client_is_rejected_with_a_clear_message():
    with pytest.raises(TypeError, match="base_urls"):
        core.ChatClient(http=object())


def test_register_provider_explains_the_alternative():
    with pytest.raises(NotImplementedError, match="openai_compatible"):
        core.register_provider(object())


def test_threads_run_in_parallel(base_urls):
    """The GIL is released during I/O, so concurrent sync calls overlap."""
    from concurrent.futures import ThreadPoolExecutor

    client = core.ChatClient(base_urls=base_urls)
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda i: client.chat("openai", api_key=GOOD, model="m", messages=msgs(f"n{i}")).text,
                                range(16)))
    assert all(f"n{i}" in text for i, text in enumerate(replies))
