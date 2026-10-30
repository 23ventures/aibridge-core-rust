# Changelog

## 0.2.0 (2026-10-30)

Matches aibridge-exec 0.4.0 (`8a5d31a`), the version aibridge-platform runs in production.

- Ten new providers, in aibridge-exec's order: `kimi`, `nvidia`, `zai`, `minimax`, `arcee`,
  `ai21` (OpenAI protocol with per-provider rules), `cohere` (v2 chat), `alibaba` (native
  DashScope), `meta` (Responses protocol) and `ollama` (native API, keyless, NDJSON streams).
- Provider-neutral options: `response_format` (`ResponseFormat`), `ChatMessage.cache`,
  `reasoning_effort`, `thinking`. `Usage` gains `cache_read_tokens` / `cache_write_tokens`.
- Anthropic: up to four cache breakpoints (system blocks and turns), a leading user turn when a
  conversation starts with the assistant, no temperature on Opus 4.6+ / Sonnet 4.5+, forced tool
  for structured output, cache-inclusive input token counts.
- Gemini: a leading user turn, JSON mode and `responseSchema` converted from JSON Schema.
- OpenAI-compatible: per-provider structured output (JSON schema, JSON mode or a tool), reasoning
  model rules (OpenAI, Kimi, thinking switches, NVIDIA Nemotron), `<think>` stripping.
- Adapter capabilities `requires_key`, `stream_format` and `needs_custom_base_url`. The server
  exposes `ollama` and `openai_compatible` only with a configured base URL or custom URLs allowed.
- Python bindings mirror `aibridge_exec.providers` 0.4.0, including `ProviderHandle.requires_key`,
  `needs_custom_base_url` and `stream_format`.
- Request-level parity tests: the mock echoes each request in the provider's own protocol.

## 0.1.0 (2026-10-30)

First version: the aibridge-exec provider layer and HTTP server, rewritten in Rust.

- `aibridge-core` crate: 10 providers (OpenAI and 6 OpenAI-compatible, Anthropic, Gemini, custom
  OpenAI-compatible URL), chat, SSE streaming, paginated model lists, typed errors with key redaction.
- `Layer` hook with an in-memory TTL cache for model lists and chat replies (off by default).
- `aibridge-core serve`: wire-compatible with `aibridge-exec serve` (42/42 parity scenarios
  identical); reads the same `AIBRIDGE_EXEC_*` settings. New `GET /v1/stats`.
- `aibridge_core.providers` Python wheel (PyO3, abi3 for Python 3.11+): same API as
  `aibridge_exec.providers`, sync and async (70 parity tests).
- aibridge-playground's end-to-end suite passes against it unchanged.
