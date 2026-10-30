"""Benchmark the Python libraries in-process: aibridge_exec.providers vs aibridge_core.providers.

    cargo build --release -p aibridge-core-server --examples
    python tests/bench/bench_python.py [--quick]

The upstream is the Rust mock (examples/mock_upstream) pinned to other cores; this process is
pinned to one core, so the numbers compare per-core cost of the client library itself.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import resource
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOCK = ROOT / "target" / "release" / "examples" / "mock_upstream"
CORES = os.cpu_count() or 1
KEY = "sk-bench-key-123456"
MESSAGES = [{"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Summarize the plot of a long novel. " * 20}]


def cpu_now() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def measure(label: str, fn, ops: int) -> dict:
    fn(max(5, ops // 10))  # warm-up
    wall, cpu = time.perf_counter(), cpu_now()
    fn(ops)
    wall, cpu = time.perf_counter() - wall, cpu_now() - cpu
    return {"name": label, "ops_s": ops / wall, "cpu_ms_per_op": cpu * 1000 / ops}


def suite(lib, base_urls: dict, n: int) -> list[dict]:
    client = lib.ChatClient(base_urls=base_urls)
    aclient_holder: dict = {}

    def chat(k):
        for _ in range(k):
            client.chat("openai", api_key=KEY, model="m", messages=MESSAGES, max_tokens=64)

    def stream(k):
        for _ in range(k):
            for _event in client.stream("anthropic", api_key=KEY, model="m", messages=MESSAGES):
                pass

    def models(k):
        for _ in range(k):
            client.list_models("openai", api_key=KEY)

    def run_async(coro_factory, k, concurrency=50):
        async def main():
            if "c" not in aclient_holder:
                aclient_holder["c"] = lib.AsyncChatClient(base_urls=base_urls)
            sem = asyncio.Semaphore(concurrency)

            async def one():
                async with sem:
                    await coro_factory(aclient_holder["c"])

            await asyncio.gather(*(one() for _ in range(k)))
        loop.run_until_complete(main())

    async def achat(c):
        await c.chat("openai", api_key=KEY, model="m", messages=MESSAGES)

    async def astream(c):
        async for _event in c.stream("openai", api_key=KEY, model="m", messages=MESSAGES):
            pass

    loop = asyncio.new_event_loop()
    try:
        return [
            measure("sync chat, sequential", chat, n),
            measure("sync stream (100 chunks), sequential", stream, n // 4),
            measure("sync list_models (50 models)", models, n),
            measure("async chat, 50 concurrent", lambda k: run_async(achat, k), n * 2),
            measure("async stream (100 chunks), 50 concurrent", lambda k: run_async(astream, k), n // 2),
        ]
    finally:
        loop.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not MOCK.exists():
        raise SystemExit("build first: cargo build --release -p aibridge-core-server --examples")

    if CORES > 1:
        os.sched_setaffinity(0, {CORES - 1})
    mock = subprocess.Popen(["taskset", "-c", ",".join(map(str, range(CORES - 1))) or "0", str(MOCK), "9300"],
                            stdout=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:9300/", timeout=1).read()
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        base_urls = {"openai": "http://127.0.0.1:9300/v1", "anthropic": "http://127.0.0.1:9300/v1"}
        n = 200 if args.quick else 1000
        from aibridge_core import providers as core
        from aibridge_exec import providers as exec_

        results = {}
        for label, lib in (("aibridge-exec", exec_), ("aibridge-core", core)):
            results[label] = {r["name"]: r for r in suite(lib, base_urls, n)}
            for r in results[label].values():
                print(f"  {label:14} {r['name']:44} {r['ops_s']:8.0f} ops/s  cpu {r['cpu_ms_per_op']:6.2f} ms/op", flush=True)
    finally:
        os.killpg(mock.pid, signal.SIGTERM)

    lines = ["| Workload (Python library, one core) | exec ops/s | core ops/s | Speed-up | exec CPU ms/op | core CPU ms/op |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, e in results["aibridge-exec"].items():
        c = results["aibridge-core"][name]
        lines.append(f"| {name} | {e['ops_s']:,.0f} | {c['ops_s']:,.0f} | **{c['ops_s'] / e['ops_s']:.1f}×** | "
                     f"{e['cpu_ms_per_op']:.2f} | {c['cpu_ms_per_op']:.2f} |")
    table = "\n".join(lines)
    print("\n" + table)
    if args.out:
        args.out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
