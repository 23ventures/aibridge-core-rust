"""Request-level parity: every provider x model x option set sends byte-identical requests and
parses the replies identically in aibridge_core and aibridge_exec.

The mock's recording mode (``/rec/<provider>/...``) answers with the exact request it received, in
the provider's own protocol (text, tool call or function call; with cache usage), so equal
outcomes mean equal requests (path, query, auth and protocol headers, JSON body) and equal parsing.
"""
from __future__ import annotations

import json

import pytest

from aibridge_core import providers as core

from test_parity import GOOD, mock_url, outcome  # noqa: F401 - fixture re-export

exec_providers = pytest.importorskip("aibridge_exec.providers")

PROVIDERS = [p.id for p in exec_providers.list_providers()]
SCHEMA = {"type": "object", "properties": {"name": {"type": ["string", "null"], "description": "Who"},
                                          "tags": {"type": "array", "items": {"type": "string"}},
                                          "level": {"type": "integer", "enum": [1, 2]}},
          "required": ["name"], "additionalProperties": False}
MODELS = ["m", "gpt-5-mini", "gpt-5.1", "o3", "o1-mini", "kimi-k2.5", "claude-opus-4-6", "claude-sonnet-4-20250514",
          "nvidia/nemotron-3-nano", "nvidia/llama-3.3-nemotron-super-49b-v1.5", "thinker", "qwen-tool", "glm-4.6"]
OPTIONS = {
    "plain": dict(system="Be brief.", temperature=0.7, max_tokens=50,
                  messages=[{"role": "user", "content": "Héllo"}, {"role": "assistant", "content": "Hi"},
                            {"role": "user", "content": "Again 😀"}]),
    "schema": dict(temperature=1.5, response_format={"schema": SCHEMA}, reasoning_effort="lowest", thinking=False,
                   messages=[{"role": "system", "content": "Stable rules", "cache": True},
                             {"role": "user", "content": "Fill it"}]),
    "json-any": dict(response_format={}, reasoning_effort="high", thinking=True,
                     messages=[{"role": "assistant", "content": "Earlier"},
                               {"role": "system", "content": "s1", "cache": True},
                               {"role": "system", "content": "s2", "cache": True},
                               {"role": "user", "content": "u1", "cache": True},
                               {"role": "user", "content": "u2"}]),
    "effort-none": dict(reasoning_effort="none", max_tokens=1, messages=[{"role": "user", "content": "x"}]),
    "custom-tool": dict(response_format={"schema": {"type": "object", "properties": {}}, "name": "Pick-1",
                                         "description": "Pick one"},
                        messages=[{"role": "user", "content": "y"}]),
}


def _base_urls(mock: str) -> dict[str, str]:
    urls = {p: f"{mock}/rec/{p}" for p in PROVIDERS}
    urls["alibaba"] = f"{mock}/rec/alibaba/api/v1"
    return urls


@pytest.fixture(scope="module")
def clients(mock_url):  # noqa: F811
    urls = _base_urls(mock_url)
    with exec_providers.ChatClient(base_urls=urls) as a, core.ChatClient(base_urls=urls) as b:
        yield a, b


def _compare(clients, call):
    a, b = clients
    expected, actual = outcome(lambda: call(a)), outcome(lambda: call(b))
    assert actual == expected
    return actual


CASES = [(p, m, o) for p in PROVIDERS for m in MODELS for o in OPTIONS]


@pytest.mark.parametrize("provider,model,option", CASES, ids=[f"{p}-{m}-{o}" for p, m, o in CASES])
def test_chat_request_and_reply(clients, provider, model, option):
    result = _compare(clients, lambda c: c.chat(provider, api_key=GOOD, model=model, **OPTIONS[option]))
    assert not (isinstance(result, tuple) and result[0] == "exception")


@pytest.mark.parametrize("provider,option", [(p, o) for p in PROVIDERS for o in OPTIONS])
@pytest.mark.parametrize("model", ["m", "thinker", "kimi-k2.5", "claude-opus-4-6", "qwen-tool"])
def test_stream_request_and_events(clients, provider, model, option):
    _compare(clients, lambda c: c.stream(provider, api_key=GOOD, model=model, **OPTIONS[option]))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_models_request(clients, provider):
    _compare(clients, lambda c: c.list_models(provider, api_key=GOOD))


def test_echo_really_is_the_request(clients):
    """Guard against a vacuous comparison: the reply text carries the request body."""
    reply = clients[1].chat("anthropic", api_key=GOOD, model="claude-x", **OPTIONS["plain"])
    sent = json.loads(reply.text)
    assert sent["path"] == "/messages" and sent["body"]["system"] == "Be brief."
    assert sent["headers"]["x-api-key"] == "<good-key>"
    assert reply.usage.cache_read_tokens == 300 and reply.usage.input_tokens == 405


def test_keyless_ollama(clients):
    _compare(clients, lambda c: c.chat("ollama", api_key="", model="m", messages=[{"role": "user", "content": "x"}]))
    assert "authorization" not in json.loads(clients[1].chat("ollama", api_key=" ", model="m",
                                                             messages=[{"role": "user", "content": "x"}]).text)["headers"]


@pytest.mark.parametrize("bad", [dict(reasoning_effort="max"), dict(response_format={"name": "bad name"}),
                                 dict(response_format={"schema": "x"}), dict(response_format={"bogus": 1})])
def test_invalid_options(clients, bad):
    _compare(clients, lambda c: c.chat("openai", api_key=GOOD, model="m", messages=[{"role": "user", "content": "x"}],
                                       **bad))


def test_openai_compatible_with_explicit_base_url(clients, mock_url):  # noqa: F811
    _compare(clients, lambda c: c.chat("openai_compatible", api_key=GOOD, model="m", base_url=f"{mock_url}/rec/x/v1",
                                       **OPTIONS["schema"]))


def test_provider_capabilities_match():
    for theirs in exec_providers.list_providers():
        ours = core.get_provider(theirs.id)
        assert (ours.requires_key, ours.needs_custom_base_url, ours.stream_format) == (
            theirs.requires_key, theirs.needs_custom_base_url, theirs.stream_format), theirs.id
