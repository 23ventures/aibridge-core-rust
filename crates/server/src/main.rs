//! `aibridge-core serve | providers | healthcheck`

use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::Arc;

use aibridge_core::Registry;
use aibridge_core_server::{router, AppState, ServerSettings};
use axum::serve::ListenerExt;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "aibridge-core", version, about = "Stateless LLM provider gateway (Rust aibridge-exec)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP server
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// Worker threads (default: one per CPU)
        #[arg(long)]
        workers: Option<usize>,
        #[arg(long, default_value = "info")]
        log_level: String,
    },
    /// List built-in providers
    Providers,
    /// Exit 0 when the local server answers /health (for container health checks)
    Healthcheck {
        #[arg(long, default_value_t = 8000)]
        port: u16,
    },
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Providers => {
            for adapter in Registry::builtin().list() {
                println!("{:20} {}", adapter.id(), adapter.info().label);
            }
            ExitCode::SUCCESS
        }
        Command::Healthcheck { port } => healthcheck(port),
        Command::Serve { host, port, workers, log_level } => serve(host, port, workers, log_level),
    }
}

fn healthcheck(port: u16) -> ExitCode {
    use std::io::{Read, Write};
    let attempt = || -> std::io::Result<bool> {
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port))?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(3)))?;
        stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        Ok(response.starts_with("HTTP/1.1 200"))
    };
    if attempt().unwrap_or(false) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn serve(host: String, port: u16, workers: Option<usize>, log_level: String) -> ExitCode {
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| format!("{log_level},hyper=warn,reqwest=warn"));
    tracing_subscriber::fmt().with_env_filter(filter).with_target(false).init();

    let settings = match ServerSettings::from_env() {
        Ok(settings) => settings,
        Err(message) => {
            eprintln!("configuration error: {message}");
            return ExitCode::FAILURE;
        }
    };
    if settings.api_tokens.is_empty() {
        tracing::warn!("AIBRIDGE_EXEC_API_TOKENS is not set: the server accepts unauthenticated requests");
    }
    if settings.cache.is_enabled() {
        tracing::info!(
            "cache on: models ttl {:?}, chat ttl {:?} ({:?}), max {} entries",
            settings.cache.models_ttl,
            settings.cache.chat_ttl,
            settings.cache.chat_policy,
            settings.cache.max_entries
        );
    }
    let state = match AppState::new(settings) {
        Ok(state) => Arc::new(state),
        Err(error) => {
            eprintln!("startup error: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    if let Some(workers) = workers.filter(|w| *w > 0) {
        runtime.worker_threads(workers);
    }
    let runtime = match runtime.enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("could not start the runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async move {
        let addr: SocketAddr = match format!("{host}:{port}").parse().or_else(|_| {
            use std::net::ToSocketAddrs;
            format!("{host}:{port}")
                .to_socket_addrs()
                .map_err(|e| e.to_string())?
                .next()
                .ok_or_else(|| "no address".to_owned())
        }) {
            Ok(addr) => addr,
            Err(error) => {
                eprintln!("invalid --host/--port: {error}");
                return ExitCode::FAILURE;
            }
        };
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => listener,
            Err(error) => {
                eprintln!("could not listen on {addr}: {error}");
                return ExitCode::FAILURE;
            }
        };
        tracing::info!("aibridge-core {} listening on http://{addr}", aibridge_core::VERSION);
        // Send each SSE frame as soon as it's written (no Nagle delay between streamed tokens).
        let listener = listener.tap_io(|tcp| {
            let _ = tcp.set_nodelay(true);
        });
        let served = axum::serve(listener, router(state)).with_graceful_shutdown(shutdown()).await;
        match served {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("server error: {error}");
                ExitCode::FAILURE
            }
        }
    })
}

async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
    tracing::info!("shutting down");
}
