# Changelog

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
