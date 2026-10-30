"""Run the same requests against the aibridge-exec server and the aibridge-core server and diff them.

    pip install "aibridge-exec[server] @ git+https://github.com/23ventures/aibridge-exec" httpx
    cargo build --release -p aibridge-core-server
    python tests/parity/run_parity.py                 # parity only
    python tests/parity/run_parity.py --bench         # plus a throughput comparison

Both servers get identical settings and point every provider at tests/parity/mock_provider.py.
Responses must match exactly: status, content type, and body (SSE streams byte for byte). The
only allowed difference is the extra "engine" field aibridge-core adds to /health.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOKEN = "parity-token"
GOOD = "sk-test-good-key-1234"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
KEYED = {**AUTH, "X-Provider-Key": GOOD}


def msgs(*texts, role="user"):
    return [{"role": role, "content": t} for t in texts]


def chat(provider, last="Hello", stream=True, **extra):
    body = {"provider": provider, "model": extra.pop("model", f"{provider}-model"), "messages": msgs(last),
            "stream": stream}
    body.update(extra)
    return body


# (name, method, path, headers, json body or raw bytes)
SCENARIOS: list[tuple[str, str, str, dict, object]] = [
    ("health", "GET", "/health", {}, None),
    ("providers: no token", "GET", "/v1/providers", {}, None),
    ("providers: wrong token", "GET", "/v1/providers", {"Authorization": "Bearer nope"}, None),
    ("providers", "GET", "/v1/providers", AUTH, None),
    ("models: openai", "POST", "/v1/models", KEYED, {"provider": "openai"}),
    ("models: anthropic (2 pages)", "POST", "/v1/models", KEYED, {"provider": "anthropic"}),
    ("models: gemini (2 pages)", "POST", "/v1/models", KEYED, {"provider": "gemini"}),
    ("models: groq (no override, unreachable host not called)", "POST", "/v1/models", AUTH, {"provider": "groq"}),
    ("models: no key", "POST", "/v1/models", AUTH, {"provider": "openai"}),
    ("models: bad key", "POST", "/v1/models", {**AUTH, "X-Provider-Key": "sk-wrong-key-999"}, {"provider": "openai"}),
    ("models: unknown provider", "POST", "/v1/models", KEYED, {"provider": "nope"}),
    ("models: openai_compatible disabled", "POST", "/v1/models", KEYED, {"provider": "openai_compatible"}),
    ("models: custom base_url disabled", "POST", "/v1/models", KEYED, {"provider": "openai", "base_url": "http://x"}),
    ("models: missing provider field", "POST", "/v1/models", KEYED, {}),
    ("models: no token beats bad body", "POST", "/v1/models", {}, {}),
    ("chat: openai stream", "POST", "/v1/chat", KEYED,
     {**chat("openai"), "system": "Be brief", "temperature": 0.7, "max_tokens": 50,
      "messages": [{"role": "system", "content": "extra"}, *msgs("first"), {"role": "assistant", "content": "ok"},
                   *msgs("Hello")]}),
    ("chat: openai non-stream", "POST", "/v1/chat", KEYED, {**chat("openai", stream=False), "max_tokens": 5}),
    ("chat: anthropic stream (merged turns, clamped temperature)", "POST", "/v1/chat", KEYED,
     {**chat("anthropic"), "temperature": 1.8, "system": "sys", "messages": msgs("one", "two")}),
    ("chat: anthropic non-stream", "POST", "/v1/chat", KEYED, chat("anthropic", stream=False)),
    ("chat: gemini stream", "POST", "/v1/chat", KEYED,
     {**chat("gemini", model="models/gemini-a"), "max_tokens": 9,
      "messages": [*msgs("a"), {"role": "assistant", "content": "b"}, *msgs("c")]}),
    ("chat: gemini non-stream", "POST", "/v1/chat", KEYED, {**chat("gemini", stream=False), "system": "s"}),
    ("chat: unicode stream", "POST", "/v1/chat", KEYED, chat("openai", "unicode")),
    ("chat: unicode non-stream", "POST", "/v1/chat", KEYED, chat("anthropic", "unicode", stream=False)),
    ("chat: rate limited before stream", "POST", "/v1/chat", KEYED, chat("openai", "please fail")),
    ("chat: openai fails mid-stream", "POST", "/v1/chat", KEYED, chat("openai", "fail mid")),
    ("chat: anthropic fails mid-stream", "POST", "/v1/chat", KEYED, chat("anthropic", "fail mid")),
    ("chat: upstream 500 html", "POST", "/v1/chat", KEYED, chat("openai", "server error", stream=False)),
    ("chat: provider sends invalid json", "POST", "/v1/chat", KEYED, chat("openai", "bad json", stream=False)),
    ("chat: key echoed by provider is redacted", "POST", "/v1/chat", KEYED, chat("openai", "echo key")),
    ("chat: unknown model", "POST", "/v1/chat", KEYED, chat("openai", model="missing-model")),
    ("chat: bad key", "POST", "/v1/chat", {**AUTH, "X-Provider-Key": "sk-wrong-key-999"}, chat("anthropic")),
    ("chat: stream as string 'false'", "POST", "/v1/chat", KEYED, {**chat("openai"), "stream": "false"}),
    ("chat: too many messages", "POST", "/v1/chat", KEYED, {**chat("openai"), "messages": msgs(*["x"] * 51)}),
    ("chat: message too long", "POST", "/v1/chat", KEYED, chat("openai", "y" * 2001)),
    ("chat: conversation too long", "POST", "/v1/chat", KEYED,
     {**chat("openai"), "messages": msgs(*["z" * 1500] * 3)}),
    ("chat: validation errors", "POST", "/v1/chat", KEYED,
     {"provider": "", "messages": [{"role": "robot", "content": 1}], "temperature": 3}),
    ("chat: empty messages", "POST", "/v1/chat", KEYED, {"provider": "openai", "model": "m", "messages": []}),
    ("chat: max_tokens fraction", "POST", "/v1/chat", KEYED, {**chat("openai"), "max_tokens": 1.5}),
    ("chat: invalid json body", "POST", "/v1/chat", KEYED, b"{not json"),
    ("chat: body is a list", "POST", "/v1/chat", KEYED, []),
    ("unknown route", "GET", "/nope", {}, None),
    ("wrong method", "DELETE", "/v1/chat", {}, None),
]


def start(cmd: str, env: dict, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(shlex.split(cmd.format(port=port)), env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, start_new_session=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"failed to start: {cmd}\n{proc.stderr.read().decode()}")
        try:
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    raise SystemExit(f"timed out starting: {cmd}")


def start_mock(port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, str(HERE / "mock_provider.py"), str(port)], start_new_session=True)
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=0.5)
            return proc
        except httpx.HTTPError:
            time.sleep(0.05)
    raise SystemExit("mock provider did not start")


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()


def call(client: httpx.Client, base: str, method: str, path: str, headers: dict, body: object):
    kwargs: dict = {"headers": dict(headers)}
    if isinstance(body, bytes):
        kwargs["content"] = body
        kwargs["headers"]["Content-Type"] = "application/json"
    elif body is not None:
        kwargs["json"] = body
    response = client.request(method, base + path, **kwargs)
    return response.status_code, response.headers.get("content-type", ""), response.text


def normalize(name: str, text: str) -> str:
    if name == "health":
        data = json.loads(text)
        data.pop("engine", None)
        data.pop("version", None)
        return json.dumps(data)
    return text


def parity(exec_base: str, core_base: str) -> int:
    failures = 0
    with httpx.Client(timeout=30) as client:
        for name, method, path, headers, body in SCENARIOS:
            a = call(client, exec_base, method, path, headers, body)
            b = call(client, core_base, method, path, headers, body)
            a_norm = (a[0], a[1], normalize(name, a[2]))
            b_norm = (b[0], b[1], normalize(name, b[2]))
            same = a_norm == b_norm
            failures += not same
            print(f"{'PASS' if same else 'FAIL'}  {a[0]}  {name}")
            if not same:
                for label, (status, ctype, text) in (("exec", a_norm), ("core", b_norm)):
                    print(f"      {label}: {status} {ctype}\n        {text[:600]!r}")
            if GOOD in a[2] or GOOD in b[2]:
                failures += 1
                print("FAIL  provider key leaked into a response")
    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios identical")
    return failures


async def bench_one(base: str, label: str, requests: int, concurrency: int) -> dict:
    body = {**chat("openai", stream=False), "max_tokens": 5}
    stream_body = chat("anthropic")
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        results = {}
        for kind, path, payload in (("models", "/v1/models", {"provider": "openai"}),
                                    ("chat", "/v1/chat", body), ("stream", "/v1/chat", stream_body)):
            semaphore = asyncio.Semaphore(concurrency)
            latencies: list[float] = []

            async def one():
                async with semaphore:
                    started = time.perf_counter()
                    response = await client.post(base + path, json=payload, headers=KEYED)
                    assert response.status_code == 200, response.text
                    latencies.append(time.perf_counter() - started)

            started = time.perf_counter()
            await asyncio.gather(*(one() for _ in range(requests)))
            elapsed = time.perf_counter() - started
            latencies.sort()
            results[kind] = {"rps": requests / elapsed, "p50_ms": latencies[len(latencies) // 2] * 1000,
                             "p99_ms": latencies[int(len(latencies) * 0.99) - 1] * 1000}
        return results


def rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-cmd", default="aibridge-exec serve --port {port}")
    parser.add_argument("--core-cmd", default=str(ROOT / "target/release/aibridge-core") + " serve --port {port}")
    parser.add_argument("--mock-port", type=int, default=9100)
    parser.add_argument("--exec-port", type=int, default=8001)
    parser.add_argument("--core-port", type=int, default=8002)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-requests", type=int, default=2000)
    parser.add_argument("--bench-concurrency", type=int, default=50)
    args = parser.parse_args()

    mock_url = f"http://127.0.0.1:{args.mock_port}"
    env = {**os.environ,
           "AIBRIDGE_EXEC_API_TOKENS": TOKEN,
           "AIBRIDGE_EXEC_BASE_URL_OPENAI": f"{mock_url}/v1",
           "AIBRIDGE_EXEC_BASE_URL_ANTHROPIC": f"{mock_url}/v1",
           "AIBRIDGE_EXEC_BASE_URL_GEMINI": f"{mock_url}/v1beta",
           "AIBRIDGE_EXEC_BASE_URL_GROQ": "http://127.0.0.1:9/never-called",
           "AIBRIDGE_EXEC_MAX_MESSAGES": "50",
           "AIBRIDGE_EXEC_MAX_MESSAGE_CHARS": "2000",
           "AIBRIDGE_EXEC_MAX_TOTAL_CHARS": "4000",
           "RUST_LOG": "warn"}
    for name in list(env):
        if name.startswith("AIBRIDGE_CORE_"):
            env.pop(name)
    procs = []
    try:
        procs.append(start_mock(args.mock_port))
        exec_proc = start(args.exec_cmd + " --log-level warning", env, args.exec_port)
        procs.append(exec_proc)
        core_proc = start(args.core_cmd, env, args.core_port)
        procs.append(core_proc)
        exec_base, core_base = f"http://127.0.0.1:{args.exec_port}", f"http://127.0.0.1:{args.core_port}"
        failures = parity(exec_base, core_base)
        if args.bench:
            print(f"\nBenchmark: {args.bench_requests} requests per kind, concurrency {args.bench_concurrency}, "
                  "mock provider on localhost (measures gateway overhead, not provider latency)")
            for label, base, proc in (("aibridge-exec", exec_base, exec_proc), ("aibridge-core", core_base, core_proc)):
                results = asyncio.run(bench_one(base, label, args.bench_requests, args.bench_concurrency))
                print(f"  {label:14} RSS {rss_mb(proc.pid):6.1f} MB")
                for kind, r in results.items():
                    print(f"    {kind:7} {r['rps']:8.0f} req/s   p50 {r['p50_ms']:7.1f} ms   p99 {r['p99_ms']:7.1f} ms")
        return 1 if failures else 0
    finally:
        for proc in reversed(procs):
            stop(proc)


if __name__ == "__main__":
    sys.exit(main())
