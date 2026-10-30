//! HTTP API over the provider adapters. Stateless: keys arrive per request and are never stored.
//!
//! Endpoints (identical to aibridge-exec):
//!     GET  /health
//!     GET  /v1/providers
//!     POST /v1/models     {"provider"}                              header X-Provider-Key
//!     POST /v1/chat       {"provider", "model", "messages", ...}    header X-Provider-Key
//! Added by aibridge-core:
//!     GET  /v1/stats      cache counters (when a cache is configured)

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use aibridge_core::{CacheLayer, ChatClient, ChatRequest, ErrorCode, ProviderError, StreamEvent, VERSION};
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures_util::StreamExt;
use serde_json::{json, Value};
use subtle::ConstantTimeEq;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::body::{self, Details};
use crate::settings::ServerSettings;

pub const KEY_HEADER: &str = "x-provider-key";

pub struct AppState {
    pub settings: ServerSettings,
    pub client: ChatClient,
    pub cache: Option<Arc<CacheLayer>>,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    pub fn new(settings: ServerSettings) -> Result<Self, ProviderError> {
        Self::with_builder(settings, ChatClient::builder())
    }

    /// Build with a pre-configured client builder (tests, custom registries or layers).
    pub fn with_builder(
        settings: ServerSettings,
        builder: aibridge_core::ChatClientBuilder,
    ) -> Result<Self, ProviderError> {
        let cache = settings.cache.is_enabled().then(|| Arc::new(CacheLayer::new(settings.cache.clone())));
        let mut builder = builder.base_urls(settings.base_urls.clone());
        if let Some(cache) = &cache {
            builder = builder.layer(cache.clone());
        }
        Ok(Self { client: builder.build()?, settings, cache })
    }
}

// ------------------------------------------------------------------ responses

/// Like Starlette's JSONResponse: compact, UTF-8.
fn json_response(status: StatusCode, value: &Value) -> Response {
    let body = serde_json::to_vec(value).unwrap_or_default();
    (status, [(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))], body).into_response()
}

fn error_response(error: &ProviderError) -> Response {
    let status = StatusCode::from_u16(error.code.http_status()).unwrap_or(StatusCode::BAD_REQUEST);
    json_response(status, &json!({"error": error}))
}

fn validation_response(details: Details) -> Response {
    json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        &json!({"error": {"code": "bad_request", "message": "invalid request body", "details": details}}),
    )
}

enum ApiError {
    Provider(ProviderError),
    Validation(Details),
}

impl From<ProviderError> for ApiError {
    fn from(error: ProviderError) -> Self {
        ApiError::Provider(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Provider(error) => error_response(&error),
            ApiError::Validation(details) => validation_response(details),
        }
    }
}

type ApiResult = Result<Response, ApiError>;

/// `json.dumps(payload, separators=(",", ":"))`: compact with non-ASCII escaped, as the Python
/// server writes SSE data lines.
pub fn ascii_json(value: &impl serde::Serialize) -> String {
    let text = serde_json::to_string(value).unwrap_or_default();
    if text.is_ascii() {
        return text;
    }
    let mut out = String::with_capacity(text.len() + 16);
    for ch in text.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            let mut units = [0u16; 2];
            for unit in ch.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

fn sse_frame(kind: &str, payload: &impl serde::Serialize) -> Bytes {
    Bytes::from(format!("event: {kind}\ndata: {}\n\n", ascii_json(payload)))
}

// ------------------------------------------------------------------ checks

fn authorize(settings: &ServerSettings, headers: &HeaderMap) -> Result<(), ProviderError> {
    if settings.api_tokens.is_empty() {
        return Ok(());
    }
    let raw = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).unwrap_or("");
    let token = raw.strip_prefix("Bearer ").unwrap_or(raw).trim();
    let ok = settings.api_tokens.iter().any(|t| bool::from(token.as_bytes().ct_eq(t.as_bytes())));
    if ok {
        Ok(())
    } else {
        Err(ProviderError::new(ErrorCode::Auth, "missing or invalid service token"))
    }
}

fn enabled(state: &AppState, provider_id: &str) -> bool {
    let settings = &state.settings;
    if !settings.provider_enabled(provider_id) {
        return false;
    }
    // Unknown ids are reported by resolve().
    let Ok(adapter) = state.client.registry().get(provider_id) else { return true };
    // openai_compatible and ollama have no public endpoint: only with a URL to call.
    !adapter.needs_custom_base_url() || settings.allow_custom_base_url || settings.base_urls.contains_key(provider_id)
}

fn resolve(state: &AppState, provider_id: &str, base_url: Option<&str>) -> Result<(), ProviderError> {
    if !enabled(state, provider_id) {
        return Err(ProviderError::new(
            ErrorCode::UnknownProvider,
            format!("provider {} is not enabled on this server", aibridge_core::registry::py_repr(provider_id)),
        )
        .provider(provider_id));
    }
    state.client.provider(provider_id)?;
    let base_url = base_url.filter(|u| !u.is_empty());
    if base_url.is_some() && !state.settings.allow_custom_base_url {
        return Err(
            ProviderError::new(ErrorCode::Config, "custom base_url is disabled on this server").provider(provider_id)
        );
    }
    if let Some(url) = base_url {
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err(ProviderError::new(ErrorCode::Config, "base_url must be an http(s) URL").provider(provider_id));
        }
    }
    Ok(())
}

fn check_size(settings: &ServerSettings, body: &body::ChatBody) -> Result<(), ProviderError> {
    let bad = |message: String| ProviderError::new(ErrorCode::BadRequest, message);
    if body.messages.len() > settings.max_messages {
        return Err(bad(format!("too many messages (max {})", settings.max_messages)));
    }
    let mut total = body.system.as_deref().map(|s| s.chars().count()).unwrap_or(0);
    for message in &body.messages {
        let length = message.content.chars().count();
        if length > settings.max_message_chars {
            return Err(bad(format!("message too long (max {} chars)", settings.max_message_chars)));
        }
        total += length;
    }
    if total > settings.max_total_chars {
        return Err(bad(format!("conversation too long (max {} chars)", settings.max_total_chars)));
    }
    Ok(())
}

fn provider_key(headers: &HeaderMap) -> String {
    headers.get(KEY_HEADER).and_then(|v| v.to_str().ok()).unwrap_or("").to_owned()
}

/// FastAPI order: JSON decoding first, then the service token, then field validation.
fn parse_then_authorize(
    state: &AppState,
    headers: &HeaderMap,
    raw: &Bytes,
) -> Result<serde_json::Map<String, Value>, ApiError> {
    let object = body::parse_object(raw);
    if let Err(details) = &object {
        let is_json_error = details.first().is_some_and(|d| d["msg"] == "JSON decode error");
        if is_json_error {
            return Err(ApiError::Validation(details.clone()));
        }
    }
    authorize(&state.settings, headers)?;
    object.map_err(ApiError::Validation)
}

// ------------------------------------------------------------------ handlers

async fn health() -> Response {
    json_response(StatusCode::OK, &json!({"status": "ok", "version": VERSION, "engine": "aibridge-core"}))
}

async fn providers(State(state): State<SharedState>, headers: HeaderMap) -> ApiResult {
    authorize(&state.settings, &headers)?;
    let list: Vec<&aibridge_core::ProviderInfo> = state
        .client
        .registry()
        .list()
        .iter()
        .filter(|adapter| enabled(&state, adapter.id()))
        .map(|adapter| adapter.info())
        .collect();
    Ok(json_response(StatusCode::OK, &json!({"providers": list})))
}

async fn models(State(state): State<SharedState>, headers: HeaderMap, raw: Bytes) -> ApiResult {
    let object = parse_then_authorize(&state, &headers, &raw)?;
    let body = body::models_body(&object).map_err(ApiError::Validation)?;
    resolve(&state, &body.provider, body.base_url.as_deref())?;
    let found = state.client.list_models(&body.provider, &provider_key(&headers), body.base_url.as_deref()).await?;
    Ok(json_response(StatusCode::OK, &json!({"models": found})))
}

async fn chat(State(state): State<SharedState>, headers: HeaderMap, raw: Bytes) -> ApiResult {
    let object = parse_then_authorize(&state, &headers, &raw)?;
    let body = body::chat_body(&object).map_err(ApiError::Validation)?;
    resolve(&state, &body.provider, body.base_url.as_deref())?;
    check_size(&state.settings, &body)?;
    let key = provider_key(&headers);
    let mut request = ChatRequest::new(body.model, body.messages);
    request.system = body.system;
    request.temperature = body.temperature;
    request.max_tokens = body.max_tokens;
    let base_url = body.base_url.as_deref();

    if !body.stream {
        let reply = state.client.chat(&body.provider, &key, request, base_url).await?;
        return Ok(json_response(StatusCode::OK, &json!(reply)));
    }

    // Fail fast (auth, unknown model) with a normal HTTP error before the stream starts.
    let mut stream = state.client.stream(&body.provider, &key, request, base_url).await?;
    let first = match stream.next().await {
        Some(Ok(event)) => event,
        Some(Err(error)) => return Err(error.into()),
        None => return Err(ProviderError::new(ErrorCode::Upstream, "the provider closed the stream").into()),
    };

    let events = async_frames(first, stream);
    let mut response = Response::new(Body::from_stream(events));
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    Ok(response)
}

fn async_frames(
    first: StreamEvent,
    rest: aibridge_core::ChatStream,
) -> impl futures_util::Stream<Item = Result<Bytes, Infallible>> + Send {
    let head = futures_util::stream::once(async move { sse_frame(first.type_name(), &first) });
    let tail = rest.scan(false, |stopped, item| {
        if *stopped {
            return futures_util::future::ready(None);
        }
        let frame = match item {
            Ok(event) => sse_frame(event.type_name(), &event),
            Err(error) => {
                *stopped = true;
                sse_frame("error", &json!({"type": "error", "error": error}))
            }
        };
        futures_util::future::ready(Some(frame))
    });
    head.chain(tail).map(Ok)
}

async fn stats(State(state): State<SharedState>, headers: HeaderMap) -> ApiResult {
    authorize(&state.settings, &headers)?;
    let cache = state.cache.as_ref().map(|c| json!(c.stats()));
    Ok(json_response(StatusCode::OK, &json!({"engine": "aibridge-core", "version": VERSION, "cache": cache})))
}

async fn openapi() -> Response {
    let spec: Value = serde_json::from_str(include_str!("openapi.json")).unwrap_or(Value::Null);
    let mut spec = spec;
    spec["info"]["version"] = json!(VERSION);
    json_response(StatusCode::OK, &spec)
}

async fn docs() -> Response {
    let html = r##"<!doctype html>
<html><head><meta charset="utf-8"><title>aibridge-core - API docs</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url: "/openapi.json", dom_id: "#swagger-ui"})</script></body></html>"##;
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], html).into_response()
}

async fn not_found() -> Response {
    json_response(StatusCode::NOT_FOUND, &json!({"detail": "Not Found"}))
}

async fn method_not_allowed() -> Response {
    json_response(StatusCode::METHOD_NOT_ALLOWED, &json!({"detail": "Method Not Allowed"}))
}

/// One line per request: method, path, status, duration. Never headers or bodies.
async fn access_log(request: Request, next: Next) -> Response {
    let (method, path, started) = (request.method().clone(), request.uri().path().to_owned(), Instant::now());
    let response = next.run(request).await;
    tracing::info!(
        target: "aibridge_core_server::access",
        "{method} {path} {} {:.1}ms",
        response.status().as_u16(),
        started.elapsed().as_secs_f64() * 1000.0
    );
    response
}

pub fn router(state: SharedState) -> Router {
    let cors_origins = state.settings.cors_origins.clone();
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/v1/providers", get(providers))
        .route("/v1/models", post(models))
        .route("/v1/chat", post(chat))
        .route("/v1/stats", get(stats))
        .route("/openapi.json", get(openapi))
        .route("/docs", get(docs))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .with_state(state);
    if !cors_origins.is_empty() {
        let origins: Vec<HeaderValue> = cors_origins.iter().filter_map(|o| o.parse().ok()).collect();
        app = app.layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(origins))
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE, HeaderName::from_static(KEY_HEADER)]),
        );
    }
    app.layer(middleware::from_fn(access_log))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_json_matches_python_json_dumps() {
        assert_eq!(ascii_json(&json!({"text": "h\u{e9}llo \u{1f600}"})), r#"{"text":"h\u00e9llo \ud83d\ude00"}"#);
    }
}
