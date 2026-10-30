//! Minimal HTTP load generator for the benchmark (keeps the client from being the bottleneck).
//!
//!     cargo run --release --example loadgen -- <url> <body.json|-> <requests> <concurrency> [header:value ...]
//!
//! Reads every response to the end (so streamed replies count in full) and prints one JSON line:
//! {"requests", "errors", "seconds", "rps", "p50_ms", "p90_ms", "p99_ms", "bytes"}

use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: loadgen <url> <body.json|-> <requests> <concurrency> [header:value ...]");
        std::process::exit(2);
    }
    let url = args[1].clone();
    let body: Option<Vec<u8>> = (args[2] != "-").then(|| std::fs::read(&args[2]).expect("body file"));
    let requests: usize = args[3].parse().expect("requests");
    let concurrency: usize = args[4].parse().expect("concurrency");
    let headers: Vec<(String, String)> = args[5..]
        .iter()
        .filter_map(|h| h.split_once(':').map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned())))
        .collect();

    let client = reqwest::Client::builder().pool_max_idle_per_host(concurrency).build().unwrap();
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let started = Instant::now();
    let mut workers = Vec::new();
    for _ in 0..concurrency {
        let (client, url, body, headers, next) =
            (client.clone(), url.clone(), body.clone(), headers.clone(), next.clone());
        workers.push(tokio::spawn(async move {
            let (mut latencies, mut errors, mut bytes) = (Vec::new(), 0usize, 0usize);
            while next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < requests {
                let t = Instant::now();
                let mut request = match &body {
                    Some(b) => client.post(&url).body(b.clone()).header("content-type", "application/json"),
                    None => client.get(&url),
                };
                for (k, v) in &headers {
                    request = request.header(k, v);
                }
                match request.send().await {
                    Ok(response) => {
                        let ok = response.status().is_success();
                        let mut stream = response.bytes_stream();
                        while let Some(chunk) = stream.next().await {
                            match chunk {
                                Ok(c) => bytes += c.len(),
                                Err(_) => break,
                            }
                        }
                        if !ok {
                            errors += 1;
                        }
                    }
                    Err(_) => errors += 1,
                }
                latencies.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            (latencies, errors, bytes)
        }));
    }
    let (mut all, mut errors, mut bytes) = (Vec::new(), 0, 0);
    for worker in workers {
        let (l, e, b) = worker.await.unwrap();
        all.extend(l);
        errors += e;
        bytes += b;
    }
    let seconds = started.elapsed().as_secs_f64();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct =
        |p: f64| all.get(((all.len() as f64 * p) as usize).min(all.len().saturating_sub(1))).copied().unwrap_or(0.0);
    println!(
        "{{\"requests\":{},\"errors\":{errors},\"seconds\":{seconds:.3},\"rps\":{:.1},\"p50_ms\":{:.2},\"p90_ms\":{:.2},\"p99_ms\":{:.2},\"bytes\":{bytes}}}",
        all.len(),
        all.len() as f64 / seconds,
        pct(0.50),
        pct(0.90),
        pct(0.99)
    );
}
