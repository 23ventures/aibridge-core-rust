# aibridge-core

The provider layer of [aibridge-exec](https://github.com/23ventures/aibridge-exec), rewritten in
Rust. It gives one interface to many LLM providers, with the caller's own API key on every call.
It is stateless, stores nothing and logs no keys.

It plugs in where aibridge-exec is used, without code changes on the other side:

- **HTTP server:** `aibridge-core serve`. Same endpoints, JSON, SSE stream and `AIBRIDGE_EXEC_*`
  settings as `aibridge-exec serve`. aibridge-playground runs on it unchanged.
- **Python library:** `aibridge_core.providers`. Same API as `aibridge_exec.providers` (`ChatClient`,
  `AsyncChatClient`, `ProviderError`, the same dataclasses).
- **Rust crate:** `aibridge-core`, for Rust services, with a `Layer` hook for caching and other
  optimizations.

Compared with aibridge-exec on one CPU core ([full results and method](docs/benchmarks.md)):

| | aibridge-exec | aibridge-core | |
|---|---:|---:|---:|
| Server: chat, req/s | 324 | 13,376 | 41× |
| Server: streamed chat (100 chunks), req/s | 80 | 733 | 9× |
| Server: 500 concurrent slow streams, p99 latency | 34.7 s | 0.52 s | 66× lower |
| Server: CPU per streamed chat | 12.4 ms | 0.73 ms | 17× less |
| Server: peak memory | 66 MB | 11 MB | 6× less |
| Python library: async chat, 50 concurrent, ops/s | 342 | 5,608 | 16× |

## Scope of v0.1

| aibridge-exec module | In aibridge-core? |
| --- | --- |
| `providers` (chat, streaming, model lists; 10 providers) | Yes: Rust, plus Python bindings |
| HTTP server | Yes: wire-compatible |
| `media` (image and video generation) | Not yet: keep using aibridge-exec |
| `contracts` (tool and agent types) | No: plain Python types, nothing to speed up |
| `code_runtime` | No: subprocess orchestration, nothing to speed up |

Providers: `openai`, `anthropic`, `gemini`, `xai`, `openrouter`, `groq`, `deepseek`, `mistral`,
`together`, and `openai_compatible` (any OpenAI-compatible URL).

## Run the server

```bash
cargo build --release -p aibridge-core-server
AIBRIDGE_EXEC_API_TOKENS=change-me ./target/release/aibridge-core serve --host 0.0.0.0 --port 8000

# or Docker (distroless image, runs as non-root)
docker build -t aibridge-core .
docker run -p 8000:8000 -e AIBRIDGE_EXEC_API_TOKENS=change-me aibridge-core
```

The API is the same as aibridge-exec. See that README for request examples. Interactive docs
are at `/docs`, and the OpenAPI document is at `/openapi.json`.

```bash
curl -N localhost:8000/v1/chat -H "Authorization: Bearer change-me" -H "X-Provider-Key: $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"provider": "openai", "model": "<model id>", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Configuration

Every aibridge-exec variable works unchanged. An `AIBRIDGE_CORE_` spelling of the same name takes
precedence if both are set.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AIBRIDGE_EXEC_API_TOKENS` | empty | Comma-separated service tokens (`Authorization: Bearer`). Empty means open, for local use only. |
| `AIBRIDGE_EXEC_PROVIDERS` | all | Provider ids to expose |
| `AIBRIDGE_EXEC_CORS_ORIGINS` | empty | Browser origins allowed to call the server |
| `AIBRIDGE_EXEC_ALLOW_CUSTOM_BASE_URL` | `false` | Let callers send `base_url`. Keep it off on public servers. |
| `AIBRIDGE_EXEC_BASE_URL_<PROVIDER>` | provider default | Server-side URL override |
| `AIBRIDGE_EXEC_MAX_MESSAGES` / `_MAX_MESSAGE_CHARS` / `_MAX_TOTAL_CHARS` | 500 / 200000 / 1000000 | Request size limits |
| `AIBRIDGE_CORE_CACHE_MODELS_TTL` | off | Seconds to cache each key's model list |
| `AIBRIDGE_CORE_CACHE_CHAT_TTL` | off | Seconds to cache chat replies |
| `AIBRIDGE_CORE_CACHE_CHAT_POLICY` | `deterministic` | `deterministic` (`temperature=0` only) or `all` |
| `AIBRIDGE_CORE_CACHE_MAX_ENTRIES` | 10000 | Entries per cache |
| `RUST_LOG` | `info` | Log filter. One access line per request; headers and bodies are never logged. |

Added endpoint: `GET /v1/stats` returns cache hit and miss counters. `/health` also reports
`"engine": "aibridge-core"`.

## Python

```bash
pip install maturin && maturin develop --release     # or install a wheel built by CI
```

```python
from aibridge_core.providers import ChatClient, CacheOptions

with ChatClient(cache=CacheOptions(models_ttl=300)) as client:      # cache is optional
    models = client.list_models("anthropic", api_key=KEY)
    reply = client.chat("anthropic", api_key=KEY, model=models[0].id,
                        messages=[{"role": "user", "content": "What is a suspension bridge?"}])
    for event in client.stream("openai", api_key=KEY, model="<model id>",
                               messages=[{"role": "user", "content": "Count to five"}]):
        print(event.text if event.type == "delta" else event.usage)
```

Network waits release the GIL. The differences from aibridge-exec (no `http=` argument, no Python
adapters) are listed in [docs/integration.md](docs/integration.md).

## Rust

```rust
use std::time::Duration;
use aibridge_core::{CacheConfig, ChatClient, ChatMessage, ChatRequest, Role};
use futures_util::StreamExt;

let client = ChatClient::builder()
    .base_url("openai", "https://gateway.internal/openai/v1")      // optional
    .cache(CacheConfig { models_ttl: Some(Duration::from_secs(300)), ..Default::default() })
    .build()?;
let request = ChatRequest::new("<model id>", vec![ChatMessage::new(Role::User, "Hello")]);
let mut stream = client.stream("anthropic", &key, request, None).await?;
while let Some(event) = stream.next().await { println!("{:?}", event?); }
```

Extension points:
- **New provider:** implement `ProviderAdapter`, which describes HTTP calls and parses responses
  with no I/O, then `Registry::register` it. See `crates/core/src/providers/anthropic.rs`.
- **Caching or other optimization:** implement `Layer`. It can answer a call itself or observe
  completed calls, and it receives a salted key fingerprint, never the key. `CacheLayer` is the
  built-in example; a Redis cache, request coalescing or metrics would be further layers.

## Layout

```
crates/core/      aibridge-core: types, errors, SSE decoder, adapters, client, registry, layers
crates/server/    aibridge-core-server: axum server and the `aibridge-core` CLI
                  examples/mock_upstream.rs, examples/loadgen.rs (benchmark tools)
crates/python/    PyO3 extension (aibridge_core._native)
python/           aibridge_core package: providers.py, types.py, errors.py
tests/parity/     mock provider and HTTP parity runner against aibridge-exec
tests/python/     Python library parity tests against aibridge-exec
tests/bench/      run_bench.py (servers), bench_python.py (libraries)
docs/             integration.md, benchmarks.md
```

## Development

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude aibridge-core-py

# parity with aibridge-exec (needs it installed: pip install "../aibridge-exec[server]")
cargo build --release -p aibridge-core-server --examples
maturin develop --release && pip install pytest pytest-asyncio httpx
pytest                                   # Python library, sync and async
python tests/parity/run_parity.py        # HTTP server, byte-for-byte
python tests/bench/run_bench.py          # benchmarks
```

CI (`.github/workflows/ci.yml`) runs all of these, builds the Docker image, and builds wheels on
`v*` tags. The parity job checks out aibridge-exec. It is public today; if it becomes private,
add an `EXEC_REPO_TOKEN` secret, as aibridge-playground does.

## Next steps

- Port `media` (image and video generation): the next hot path the platform calls.
- Native asyncio streaming transport, if async streaming CPU matters (see benchmarks).
- Shared cache `Layer` (Redis) for multi-instance deployments, and request coalescing for
  identical concurrent calls.

## License

Private and proprietary for now (`license = "LicenseRef-Proprietary"`). Choose a license before
the repository is made public.
