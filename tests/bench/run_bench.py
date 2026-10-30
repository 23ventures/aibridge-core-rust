"""Benchmark aibridge-exec (Python) against aibridge-core (Rust) on identical workloads.

    cargo build --release -p aibridge-core-server --examples
    pip install "aibridge-exec[server] @ git+https://github.com/23ventures/aibridge-exec"
    python tests/bench/run_bench.py [--quick] [--out bench-results.md]

Setup, chosen so the gateway is the only thing measured:
- The upstream is `examples/mock_upstream` (Rust) and the client is `examples/loadgen` (Rust), so
  neither saturates before the gateway does.
- Each gateway runs alone, pinned to one CPU core (`taskset`), with one worker: this compares
  per-core efficiency. The mock and load generator share the other core(s).
- Every workload is run once to warm up, then measured.

Reported per workload: throughput, latency percentiles, CPU time the gateway process spent per
request (user + system, from /proc) and peak resident memory.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "target" / "release"
TOKEN = "bench-token"
KEY = "sk-bench-key-123456"
CLK_TCK = os.sysconf("SC_CLK_TCK")
CORES = os.cpu_count() or 1
GATEWAY_CPU = str(CORES - 1)
OTHER_CPUS = ",".join(str(c) for c in range(CORES - 1)) or "0"


def pinned(cpus: str, cmd: list[str]) -> list[str]:
    return ["taskset", "-c", cpus, *cmd] if CORES > 1 else cmd


def wait_http(url: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise SystemExit(f"{url} did not come up")


def spawn(cmd: list[str], env: dict | None = None) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()


def tree_pids(pid: int) -> list[int]:
    pids, frontier = [pid], [pid]
    while frontier:
        parent = frontier.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            children = []
        for child in map(int, children):
            pids.append(child)
            frontier.append(child)
    return pids


def cpu_seconds(pid: int) -> float:
    total = 0.0
    for p in tree_pids(pid):
        try:
            fields = Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()
            total += (int(fields[11]) + int(fields[12])) / CLK_TCK  # utime + stime
        except (OSError, IndexError):
            pass
    return total


def peak_rss_mb(pid: int) -> float:
    total = 0.0
    for p in tree_pids(pid):
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    total += int(line.split()[1]) / 1024
        except OSError:
            pass
    return total


def loadgen(url: str, body: dict | None, requests: int, concurrency: int, headers: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(body or {}, handle)
    cmd = [str(RELEASE / "examples" / "loadgen"), url, handle.name if body is not None else "-", str(requests),
           str(concurrency), *[f"{k}:{v}" for k, v in headers.items()]]
    out = subprocess.run(pinned(OTHER_CPUS, cmd), capture_output=True, text=True, check=True).stdout
    os.unlink(handle.name)
    return json.loads(out.strip().splitlines()[-1])


def chat_body(provider: str, stream: bool) -> dict:
    return {"provider": provider, "model": "mock-model", "stream": stream, "max_tokens": 256,
            "messages": [{"role": "system", "content": "You are terse."},
                         {"role": "user", "content": "Summarize the plot of a long novel. " * 20}]}


def workloads(quick: bool) -> list[dict]:
    n = 400 if quick else 3000
    s = 200 if quick else 1500
    return [
        {"name": "models list", "path": "/v1/models", "body": {"provider": "openai"}, "requests": n, "concurrency": 32},
        {"name": "chat (non-stream)", "path": "/v1/chat", "body": chat_body("openai", False), "requests": n,
         "concurrency": 32},
        {"name": "stream openai, 100 chunks", "path": "/v1/chat", "body": chat_body("openai", True), "requests": s,
         "concurrency": 32},
        {"name": "stream anthropic, 100 chunks", "path": "/v1/chat", "body": chat_body("anthropic", True),
         "requests": s, "concurrency": 32},
        {"name": "latency, 1 at a time (non-stream)", "path": "/v1/chat", "body": chat_body("openai", False),
         "requests": n // 4, "concurrency": 1},
        {"name": "500 concurrent streams, 300ms upstream", "path": "/v1/chat", "body": chat_body("openai", True),
         "requests": 500 if quick else 2000, "concurrency": 500, "slow": True},
    ]


def run_gateway(label: str, cmd: list[str], env: dict, port: int, loads: list[dict]) -> list[dict]:
    proc = spawn(pinned(GATEWAY_CPU, cmd), env)
    try:
        wait_http(f"http://127.0.0.1:{port}/health")
        headers = {"Authorization": f"Bearer {TOKEN}", "X-Provider-Key": KEY}
        results = []
        for load in loads:
            url = f"http://127.0.0.1:{port + (1 if load.get('slow') else 0)}{load['path']}"
            if load.get("slow"):
                continue  # measured separately against the slow upstream
            loadgen(url, load["body"], max(50, load["requests"] // 10), load["concurrency"], headers)  # warm-up
            before = cpu_seconds(proc.pid)
            result = loadgen(url, load["body"], load["requests"], load["concurrency"], headers)
            result["cpu_ms_per_req"] = (cpu_seconds(proc.pid) - before) * 1000 / max(result["requests"], 1)
            result["name"] = load["name"]
            results.append(result)
            print(f"  {label:14} {load['name']:40} {result['rps']:9.0f} req/s  p50 {result['p50_ms']:7.1f} ms  "
                  f"p99 {result['p99_ms']:7.1f} ms  cpu {result['cpu_ms_per_req']:6.2f} ms/req  errors {result['errors']}",
                  flush=True)
        return results + [{"name": "_peak_rss_mb", "value": peak_rss_mb(proc.pid)}]
    finally:
        stop(proc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-cmd", default="aibridge-exec serve --workers 1 --log-level warning --port {port}")
    parser.add_argument("--core-cmd", default=f"{RELEASE / 'aibridge-core'} serve --workers 1 --log-level warn --port {{port}}")
    parser.add_argument("--quick", action="store_true", help="fewer requests (smoke run)")
    parser.add_argument("--out", type=Path, help="write a markdown report here")
    args = parser.parse_args()

    for binary in ("aibridge-core", "examples/mock_upstream", "examples/loadgen"):
        if not (RELEASE / binary).exists():
            raise SystemExit("build first: cargo build --release -p aibridge-core-server --examples")

    fast_mock = spawn(pinned(OTHER_CPUS, [str(RELEASE / "examples" / "mock_upstream"), "9200"]))
    slow_mock = spawn(pinned(OTHER_CPUS, [str(RELEASE / "examples" / "mock_upstream"), "9201"]),
                      {**os.environ, "MOCK_DELAY_MS": "300", "MOCK_CHUNKS": "20"})
    loads = workloads(args.quick)
    report: dict[str, dict[str, dict]] = {}
    try:
        wait_http("http://127.0.0.1:9200/")
        wait_http("http://127.0.0.1:9201/")
        print(f"{CORES} CPU cores; gateway pinned to core {GATEWAY_CPU}, mock + load generator on {OTHER_CPUS}\n")

        # Baseline: the load generator straight against the mock (the ceiling for any gateway).
        direct = loadgen("http://127.0.0.1:9200/v1/chat/completions", {"model": "m", "messages": [], "stream": False},
                         loads[1]["requests"], 32, {})
        print(f"  {'direct to mock':14} {'chat (non-stream)':40} {direct['rps']:9.0f} req/s  p50 {direct['p50_ms']:7.1f} ms\n")

        for label, template, port in (("aibridge-exec", args.exec_cmd, 8101), ("aibridge-core", args.core_cmd, 8201)):
            base_env = {k: v for k, v in os.environ.items() if not k.startswith("AIBRIDGE_")}
            env = {**base_env, "AIBRIDGE_EXEC_API_TOKENS": TOKEN, "RUST_LOG": "warn",
                   "AIBRIDGE_EXEC_BASE_URL_OPENAI": "http://127.0.0.1:9200/v1",
                   "AIBRIDGE_EXEC_BASE_URL_ANTHROPIC": "http://127.0.0.1:9200/v1"}
            results = run_gateway(label, shlex.split(template.format(port=port)), env, port, loads)
            # Second instance against the slow upstream for the concurrency test.
            slow_env = {**env, "AIBRIDGE_EXEC_BASE_URL_OPENAI": "http://127.0.0.1:9201/v1"}
            slow = [dict(l, slow=False) for l in loads if l.get("slow")]
            results += [r for r in run_gateway(label, shlex.split(template.format(port=port + 1)), slow_env, port + 1, slow)
                        if not r["name"].startswith("_")]
            report[label] = {r["name"]: r for r in results}
            print()
    finally:
        stop(fast_mock)
        stop(slow_mock)

    lines = ["| Workload | exec req/s | core req/s | Speed-up | exec p50 / p99 ms | core p50 / p99 ms | exec CPU ms/req | core CPU ms/req |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for load in loads:
        e, c = report["aibridge-exec"].get(load["name"]), report["aibridge-core"].get(load["name"])
        if not e or not c:
            continue
        lines.append(
            f"| {load['name']} | {e['rps']:,.0f} | {c['rps']:,.0f} | **{c['rps'] / e['rps']:.1f}×** | "
            f"{e['p50_ms']:.1f} / {e['p99_ms']:.1f} | {c['p50_ms']:.1f} / {c['p99_ms']:.1f} | "
            f"{e['cpu_ms_per_req']:.2f} | {c['cpu_ms_per_req']:.2f} |")
    e_rss = report["aibridge-exec"]["_peak_rss_mb"]["value"]
    c_rss = report["aibridge-core"]["_peak_rss_mb"]["value"]
    lines.append(f"\nPeak memory (fast-upstream run): aibridge-exec {e_rss:.0f} MB, aibridge-core {c_rss:.0f} MB.")
    lines.append(f"Direct to mock, no gateway: {direct['rps']:,.0f} req/s (ceiling).")
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
    errors = sum(r.get("errors", 0) for side in report.values() for r in side.values() if isinstance(r, dict))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
