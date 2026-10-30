# Plugging aibridge-core in where aibridge-exec is used

aibridge-core replaces two parts of aibridge-exec, and both are drop-in:

| aibridge-exec part | Replacement | Who uses it today |
| --- | --- | --- |
| HTTP server (`aibridge-exec serve`) | `aibridge-core serve` (binary or Docker image) | aibridge-playground |
| `aibridge_exec.providers` (chat, stream, models) | `aibridge_core.providers` (Python wheel) | available to aibridge-platform |

Not replaced (keep aibridge-exec installed for these): `aibridge_exec.contracts`,
`aibridge_exec.code_runtime` and `aibridge_exec.media`.

How compatibility is checked:
- **HTTP server:** `tests/parity/run_parity.py` sends 65 requests (success, error, validation and
  streaming cases, and every newer provider in both modes) to both servers. Status, content type
  and body must match byte for byte.
- **Python library:** `tests/python/test_parity.py` runs 30 scenarios through both libraries,
  sync and async, and requires equal results and errors.
- **Requests, per provider:** `tests/python/test_request_parity.py` sends every provider ×
  13 models × 5 option sets (plain, schema + `reasoning_effort="lowest"` + `thinking=False`,
  any-JSON + cache breakpoints + assistant-first turns, `reasoning_effort="none"`, custom tool name)
  through both libraries, plus streams and model lists (1,898 cases with the other suites). The mock
  answers with the exact request it received (path, query, auth and protocol headers, JSON body,
  key order included) in each provider's protocol, as text, tool call or function call with cache
  usage, so equal results mean byte-identical requests and identical parsing.
- **aibridge-platform:** its test suite run on aibridge-core (see below) passes except three tests
  that inject an `httpx.MockTransport`, which aibridge-core can't accept by design.
- **Playground:** its own Playwright suite (`e2e/chat.spec.ts`) passes against aibridge-core with
  no playground changes.

## aibridge-playground

Nothing in the playground changes. It only needs `AIBRIDGE_EXEC_URL` to point at aibridge-core.
All `AIBRIDGE_EXEC_*` server variables are read unchanged, so the same secrets work.

**Docker Compose** (`docker-compose.yml`): replace the `exec` service's build:

```yaml
  exec:
    build: ../aibridge-core          # was ../aibridge-exec
    environment:
      AIBRIDGE_EXEC_API_TOKENS: ${AIBRIDGE_EXEC_TOKEN:-local-dev-token}
      # optional: cache each key's model list for 5 minutes
      AIBRIDGE_CORE_CACHE_MODELS_TTL: "300"
    ports: ["8000:8000"]
```

**CI** (`.github/workflows/ci.yml`): check out `aibridge-core` instead of `aibridge-exec`, then
replace the server start line:

```yaml
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --release -p aibridge-core-server --manifest-path aibridge-core/Cargo.toml
      # ...
          AIBRIDGE_EXEC_API_TOKENS=ci-token \
          AIBRIDGE_EXEC_BASE_URL_OPENAI=http://127.0.0.1:9100/v1 \
          AIBRIDGE_EXEC_BASE_URL_ANTHROPIC=http://127.0.0.1:9100/v1 \
            aibridge-core/target/release/aibridge-core serve --port 8000 &
```

**AWS (ECS):** in aibridge-exec's task definition (`deploy/ecs/exec.json`), swap the image for
the aibridge-core image.
- Port, health path, environment and secrets stay the same.
- The image is distroless and its health check is `aibridge-core healthcheck --port 8000`.
- The task can drop to the smallest CPU and memory size (peak RSS was 11 MB under load).

To roll back, point the service at the exec image again: the API is identical.

## aibridge-platform

The platform imports aibridge-exec for `contracts`, `code_runtime` and `media`. Those stay on
aibridge-exec. Since platform #69 its text models (`aibridge/execution/text/`) call
`aibridge_exec.providers` with the 0.4.0 options (structured output, cache breakpoints, reasoning
effort, thinking), which aibridge-core implements identically.

To swap, every `aibridge_exec.providers` import in `aibridge/execution/text/` (client, components,
legacy, model, errors, cache_usage) and `domain/services/validate_api_keys.py` moves to
`aibridge_core.providers` (`HTTP_STATUS_FOR_CODE` is in `aibridge_core.errors`). Keep the classes from
one library throughout: `ProviderError` from one isn't the other's. Safest is one module that
re-exports them, chosen by a setting, so QA can flip between engines without a code change:

```python
# aibridge/execution/text/engine.py
import os
if os.getenv("AIBRIDGE_TEXT_ENGINE") == "core":
    from aibridge_core.providers import *           # noqa: F403
    from aibridge_core.errors import HTTP_STATUS_FOR_CODE
else:
    from aibridge_exec.providers import *           # noqa: F403
    from aibridge_exec.providers.errors import HTTP_STATUS_FOR_CODE
```

To try it without code changes, put a `sitecustomize.py` on `PYTHONPATH` that maps
`sys.modules["aibridge_exec.providers"]` (and `.errors`, `.types`) to the `aibridge_core` modules;
that is how the platform suite above was run.

The swap is one import per module:

```python
# from aibridge_exec.providers import ChatClient, AsyncChatClient, ProviderError
from aibridge_core.providers import ChatClient, AsyncChatClient, ProviderError
```

Same method names, arguments, return types (`ChatResponse`, `StreamEvent`, `ModelInfo`,
`Usage`), error class with the same `code` values, and the same lazy-generator behavior for
`stream`. It also accepts aibridge-exec's `ChatMessage` objects, so modules can move one at a
time.

Differences:
- `ChatClient(http=httpx.Client(...))` isn't accepted, because the HTTP stack is Rust. Use
  `base_urls=` and `timeout=`. Tests that used `httpx.MockTransport` should point `base_urls`
  at a local mock server instead (see `tests/parity/mock_provider.py`).
- `register_provider` with a Python adapter isn't supported. Add the adapter in Rust, or use
  `openai_compatible` with `base_url` for OpenAI-compatible endpoints.

**Name clash to watch:** the platform already depends on `aibridgecore==1.5.24` (import name
`aibridgecore`, no underscore). The new package is `aibridge-core` with import name
`aibridge_core`. Python treats them as separate packages, but they are easy to confuse in review.
Renaming this one (for example `aibridge-engine`) before it spreads is cheap now.

### Installing the wheel

The wheel contains a compiled extension, so install a built wheel rather than building from
source in the platform image:

```bash
# CI builds wheels for Linux x86_64/arm64 and macOS arm64 on every v* tag (job "wheels")
pip install aibridge_core-0.2.0-cp311-abi3-manylinux_2_28_x86_64.whl
```

One `abi3` wheel covers Python 3.11 and newer. For local development, run
`pip install maturin && maturin develop --release` in this repo.

In the platform's Dockerfile, mirror the existing aibridge-exec stage:

```dockerfile
# build stage: wheel for aibridge-core (needs Rust only in this stage)
FROM rust:1-slim-bookworm AS core-wheel
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && python3 -m venv /v && /v/bin/pip install "maturin>=1.7,<2"
COPY --from=aibridge_core . /src/aibridge-core
RUN cd /src/aibridge-core && /v/bin/maturin build --release --out /core-wheels

# runtime stage
RUN --mount=type=bind,from=core-wheel,source=/core-wheels,target=/core-wheels \
    pip install --no-index --find-links=/core-wheels aibridge-core
```

and build with `--build-context aibridge_core=../aibridge-core`, like aibridge-exec.

## Caching (new; off by default)

| Setting | Server variable | Python |
| --- | --- | --- |
| Model list TTL | `AIBRIDGE_CORE_CACHE_MODELS_TTL` (seconds) | `CacheOptions(models_ttl=...)` |
| Chat reply TTL | `AIBRIDGE_CORE_CACHE_CHAT_TTL` (seconds) | `CacheOptions(chat_ttl=...)` |
| Which chats | `AIBRIDGE_CORE_CACHE_CHAT_POLICY`: `deterministic` (default, `temperature=0` only) or `all` | `chat_policy=` |
| Size bound | `AIBRIDGE_CORE_CACHE_MAX_ENTRIES` (default 10000 per cache) | `max_entries=` |

Entries are keyed by provider, base URL, the full request and a salted hash of the API key. The
key itself is never stored, and one key never sees another key's entries. A cached chat that is
requested as a stream is replayed as one `delta` followed by `done`. Counters are at `GET /v1/stats`
(server) and `client.cache_stats()` (Python).

The cache is the first implementation of the `Layer` trait (`crates/core/src/layer.rs`). A shared
cache (Redis) or metrics layer is another `Layer`; the client and server don't change.
