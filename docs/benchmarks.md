# Benchmarks: aibridge-exec (Python) vs aibridge-core (Rust)

Measured 2026-10-30 on a 2-core x86_64 Linux VM, aibridge-exec `25b80c8` (FastAPI + uvicorn[standard]
+ httpx, Python 3.13) against aibridge-core 0.1.0 (axum + reqwest, release build).

Both run the same workloads against the same fake provider. The numbers are the cost of the
gateway itself, not of a real provider. Re-run them with:

```bash
cargo build --release -p aibridge-core-server --examples
python tests/bench/run_bench.py        # HTTP servers
python tests/bench/bench_python.py     # Python libraries
```

## Method

- **Upstream:** `examples/mock_upstream` (Rust), which replies instantly with fixed payloads:
  50 models; a 100-word chat reply; 100-chunk SSE streams.
- **Client:** `examples/loadgen` (Rust), so the load generator never limits the result. Every
  response body is read to the end.
- **Isolation:** each gateway runs alone, pinned to **one core** with one worker. The mock and
  load generator run on the other core. This compares per-core efficiency: with more cores, both
  scale out (uvicorn workers vs Tokio threads).
- **Warm-up:** each workload runs once at 10% of its size first, then is measured.
- **CPU ms/req:** user + system CPU time of the gateway process (from `/proc`) divided by
  requests. This is the number that sets how many cores you pay for at a given traffic level.

## HTTP server (what the playground calls)

| Workload | exec req/s | core req/s | Speed-up | exec p50 / p99 ms | core p50 / p99 ms | exec CPU ms/req | core CPU ms/req |
|---|---:|---:|---:|---:|---:|---:|---:|
| models list | 346 | 4,557 | **13.2×** | 75.4 / 299.2 | 7.5 / 11.6 | 2.83 | 0.22 |
| chat (non-stream) | 324 | 13,376 | **41.3×** | 80.6 / 321.6 | 2.5 / 3.9 | 3.04 | 0.07 |
| stream openai, 100 chunks | 80 | 733 | **9.2×** | 317.0 / 1365.1 | 44.4 / 63.5 | 12.36 | 0.73 |
| stream anthropic, 100 chunks | 84 | 718 | **8.6×** | 311.2 / 1194.7 | 44.5 / 64.1 | 11.76 | 0.75 |
| latency, 1 at a time (non-stream) | 441 | 3,622 | **8.2×** | 2.0 / 4.6 | 0.3 / 0.4 | 2.12 | 0.15 |
| 500 concurrent streams, 300 ms upstream | 58 | 1,192 | **20.7×** | 6,938 / 34,686 | 352 / 522 | 16.72 | 0.39 |

32 concurrent clients unless noted. Peak memory: aibridge-exec 66 MB, aibridge-core 11 MB. With
no gateway in between, the mock serves 27,027 req/s, the ceiling for any gateway on this machine.

**Reading it:**
- **CPU per request is 10–40× lower.** That is the cost saving: a core that handles about 80
  streamed chats per second under exec handles about 730 under core.
- **Streaming throughput** in this setup is partly limited by the mock and load generator sharing
  one core. The CPU column shows the gateway's own headroom, which is larger than the req/s ratio.
- **500 concurrent slow streams** is the closest to production (many users waiting on slow
  models). The floor latency is 300 ms. Core stays near it (352 ms p50). Exec queues: 7 s p50,
  35 s p99.
  - Part of exec's gap here is configuration: httpx's default pool allows 100 connections, so
    streams 101–500 wait for a slot.
  - Raising that limit in aibridge-exec would help it. The CPU cost per stream (16.7 vs 0.39 ms)
    would remain.
- **Single-request latency** added by the gateway drops from about 2 ms to about 0.3 ms. Next to
  a real model's 300 ms to 30 s this rarely matters; CPU and concurrency are the wins.

## Python library (what the platform would import)

`aibridge_exec.providers` vs `aibridge_core.providers`, called in-process from one Python thread
(or one asyncio loop) pinned to one core.

| Workload | exec ops/s | core ops/s | Speed-up | exec CPU ms/op | core CPU ms/op |
|---|---:|---:|---:|---:|---:|
| sync chat, sequential | 1,399 | 5,020 | **3.6×** | 0.64 | 0.14 |
| sync stream (100 chunks), sequential | 249 | 964 | **3.9×** | 3.71 | 0.79 |
| sync list_models (50 models) | 1,179 | 3,493 | **3.0×** | 0.72 | 0.17 |
| async chat, 50 concurrent | 342 | 5,608 | **16.4×** | 2.88 | 0.18 |
| async stream (100 chunks), 50 concurrent | 180 | 203 | **1.1×** | 5.52 | 4.88 |

- **Sync calls** are 3–4× faster. The GIL is released during network waits, so threads (for
  example uwsgi or Celery threads in the platform) overlap instead of serializing on httpx.
- **Async chat** is 16× faster: httpx's async stack is the expensive part under concurrency.
- **Async streaming is on par.** Each event that reaches an `async for` loop is one hop from the
  Rust I/O thread into asyncio (about 45 µs).
  - This mock sends 100 events back to back, which makes that hop dominate. A real provider
    spaces tokens 10–30 ms apart, so the hop is well under 1% of the time.
  - Events already decoded together are handed over as one batch. A native asyncio transport
    is the follow-up if this matters.
