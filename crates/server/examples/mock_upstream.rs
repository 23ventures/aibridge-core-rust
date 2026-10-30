//! Fast fake provider for benchmarks: OpenAI-compatible and Anthropic endpoints, any key.
//!
//!     cargo run --release --example mock_upstream -- 9200
//!
//! MOCK_CHUNKS   delta events per streamed reply (default 100)
//! MOCK_DELAY_MS provider "thinking" time before answering (default 0)

use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::{Json, Router};
use serde_json::{json, Value};

fn env_num(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

async fn delay() {
    let ms = env_num("MOCK_DELAY_MS", 0);
    if ms > 0 {
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }
}

fn sse(frames: Vec<String>) -> Response {
    let stream =
        futures_util::stream::iter(frames.into_iter().map(|f| Ok::<_, std::convert::Infallible>(Bytes::from(f))));
    ([(header::CONTENT_TYPE, "text/event-stream")], Body::from_stream(stream)).into_response()
}

async fn models() -> Json<Value> {
    delay().await;
    let data: Vec<Value> =
        (0..50).map(|i| json!({"id": format!("mock-model-{i:02}"), "context_length": 128000})).collect();
    Json(json!({"object": "list", "data": data}))
}

async fn openai(Json(body): Json<Value>) -> Response {
    delay().await;
    let model = body["model"].as_str().unwrap_or("mock").to_owned();
    let chunks = env_num("MOCK_CHUNKS", 100);
    if body["stream"].as_bool() != Some(true) {
        let text = "word ".repeat(chunks as usize);
        return Json(json!({"model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 12, "completion_tokens": chunks}})).into_response();
    }
    let mut frames: Vec<String> = (0..chunks)
        .map(|i| {
            format!(
                "data: {}\n\n",
                json!({"id": "c", "model": model, "choices": [{"index": 0, "delta": {"content": format!("tok{i} ")}}]})
            )
        })
        .collect();
    frames.push(format!("data: {}\n\n", json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})));
    frames.push(format!(
        "data: {}\n\n",
        json!({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": chunks}})
    ));
    frames.push("data: [DONE]\n\n".into());
    sse(frames)
}

async fn anthropic(Json(body): Json<Value>) -> Response {
    delay().await;
    let model = body["model"].as_str().unwrap_or("mock").to_owned();
    let chunks = env_num("MOCK_CHUNKS", 100);
    let mut frames = vec![format!(
        "event: message_start\ndata: {}\n\n",
        json!({"type": "message_start", "message": {"model": model, "usage": {"input_tokens": 12, "output_tokens": 1}}})
    )];
    frames.extend((0..chunks).map(|i| {
        format!("event: content_block_delta\ndata: {}\n\n", json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": format!("tok{i} ")}}))
    }));
    frames.push(format!(
        "event: message_delta\ndata: {}\n\n",
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": chunks}})
    ));
    frames.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".into());
    sse(frames)
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args().nth(1).and_then(|p| p.parse().ok()).unwrap_or(9200);
    let app = Router::new()
        .route("/", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(openai))
        .route("/v1/messages", post(anthropic));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("bind").tap_io(|tcp| {
        let _ = tcp.set_nodelay(true);
    });
    axum::serve(listener, app).await.expect("serve");
}
