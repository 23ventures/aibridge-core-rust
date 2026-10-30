//! Runs adapter calls over HTTP. The API key is passed per call and only placed in headers.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{Stream, StreamExt};
use serde_json::Value;

use crate::adapter::{HttpCall, Method, ProviderAdapter};
use crate::error::{redact, ErrorCode, ProviderError, Result};
use crate::layer::{CacheConfig, CacheLayer, CallInfo, KeyFingerprint, Layer};
use crate::registry::Registry;
use crate::sse::SseDecoder;
use crate::types::{ChatRequest, ChatResponse, ModelInfo, StreamEvent};

pub const MAX_MODEL_PAGES: usize = 20;

/// Stream of chat events; the last item is a `done` event unless an error ends it early.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

#[derive(Debug, Clone)]
pub struct Timeouts {
    pub connect: Duration,
    /// Longest wait for the next bytes of a response (streams can run for minutes).
    pub read: Duration,
}

impl Default for Timeouts {
    /// Same as aibridge-exec: connect 10s, read 300s.
    fn default() -> Self {
        Self { connect: Duration::from_secs(10), read: Duration::from_secs(300) }
    }
}

pub struct ChatClientBuilder {
    registry: Registry,
    base_urls: HashMap<String, String>,
    timeouts: Timeouts,
    layers: Vec<Arc<dyn Layer>>,
    http: Option<reqwest::Client>,
    user_agent: String,
}

impl ChatClientBuilder {
    pub fn registry(mut self, registry: Registry) -> Self {
        self.registry = registry;
        self
    }

    /// Route a built-in provider through another URL for every call (gateways, proxies, tests).
    pub fn base_url(mut self, provider: impl Into<String>, url: impl Into<String>) -> Self {
        self.base_urls.insert(provider.into(), url.into());
        self
    }

    pub fn base_urls(mut self, urls: HashMap<String, String>) -> Self {
        self.base_urls.extend(urls);
        self
    }

    pub fn timeouts(mut self, timeouts: Timeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// Add a layer. Layers are consulted in the order they are added.
    pub fn layer(mut self, layer: Arc<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Shortcut for `layer(CacheLayer::new(config))` (no-op when the config enables nothing).
    pub fn cache(self, config: CacheConfig) -> Self {
        if config.is_enabled() {
            self.layer(Arc::new(CacheLayer::new(config)))
        } else {
            self
        }
    }

    /// Bring your own `reqwest::Client` (timeouts above are then ignored).
    pub fn http(mut self, http: reqwest::Client) -> Self {
        self.http = Some(http);
        self
    }

    pub fn build(self) -> Result<ChatClient> {
        let http = match self.http {
            Some(http) => http,
            None => reqwest::Client::builder()
                .connect_timeout(self.timeouts.connect)
                .read_timeout(self.timeouts.read)
                .pool_idle_timeout(Duration::from_secs(90))
                .tcp_nodelay(true)
                .user_agent(self.user_agent)
                .build()
                .map_err(|e| ProviderError::new(ErrorCode::Config, format!("could not build the HTTP client: {e}")))?,
        };
        Ok(ChatClient {
            inner: Arc::new(Inner {
                http,
                registry: self.registry,
                base_urls: self.base_urls,
                layers: self.layers,
                salt: rand::random(),
            }),
        })
    }
}

struct Inner {
    http: reqwest::Client,
    registry: Registry,
    base_urls: HashMap<String, String>,
    layers: Vec<Arc<dyn Layer>>,
    salt: [u8; 32],
}

/// Async client. Cheap to clone (shares one connection pool).
#[derive(Clone)]
pub struct ChatClient {
    inner: Arc<Inner>,
}

impl ChatClient {
    pub fn builder() -> ChatClientBuilder {
        ChatClientBuilder {
            registry: Registry::builtin(),
            base_urls: HashMap::new(),
            timeouts: Timeouts::default(),
            layers: Vec::new(),
            http: None,
            user_agent: format!("aibridge-core/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Client with the built-in providers and default timeouts.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    pub fn registry(&self) -> &Registry {
        &self.inner.registry
    }

    pub fn provider(&self, provider_id: &str) -> Result<Arc<dyn ProviderAdapter>> {
        self.inner.registry.get(provider_id)
    }

    pub fn configured_base_url(&self, provider_id: &str) -> Option<&str> {
        self.inner.base_urls.get(provider_id).map(String::as_str)
    }

    fn effective_base_url<'a>(&'a self, adapter: &dyn ProviderAdapter, base_url: Option<&'a str>) -> Option<&'a str> {
        base_url.filter(|u| !u.is_empty()).or_else(|| self.configured_base_url(adapter.id()))
    }

    pub async fn chat(
        &self,
        provider: &str,
        api_key: &str,
        request: ChatRequest,
        base_url: Option<&str>,
    ) -> Result<ChatResponse> {
        let adapter = self.provider(provider)?;
        let key = require_key(adapter.as_ref(), api_key)?;
        validate(&request)?;
        let base_url = self.effective_base_url(adapter.as_ref(), base_url);
        let fingerprint = KeyFingerprint::new(&self.inner.salt, key);
        let info = CallInfo { provider: adapter.id(), base_url, key: &fingerprint, stream: false };
        for layer in &self.inner.layers {
            if let Some(hit) = layer.lookup_chat(&info, &request).await {
                return Ok(hit);
            }
        }
        let call = adapter.chat_call(&request, key, base_url, false)?;
        let response = self.send(adapter.as_ref(), &call, key).await?;
        let payload = read_json(adapter.as_ref(), response).await?;
        let reply = adapter.parse_chat(&payload, &request).map_err(|e| redact_error(e, key))?;
        for layer in &self.inner.layers {
            layer.on_chat(&info, &request, &reply).await;
        }
        Ok(reply)
    }

    /// Start a streamed chat. Errors before the provider starts streaming (unknown provider,
    /// bad key, unknown model) are returned here; later errors arrive as stream items.
    pub async fn stream(
        &self,
        provider: &str,
        api_key: &str,
        request: ChatRequest,
        base_url: Option<&str>,
    ) -> Result<ChatStream> {
        let adapter = self.provider(provider)?;
        let key = require_key(adapter.as_ref(), api_key)?.to_owned();
        validate(&request)?;
        let base_url = self.effective_base_url(adapter.as_ref(), base_url).map(str::to_owned);
        let fingerprint = KeyFingerprint::new(&self.inner.salt, &key);
        {
            let info =
                CallInfo { provider: adapter.id(), base_url: base_url.as_deref(), key: &fingerprint, stream: true };
            for layer in &self.inner.layers {
                if let Some(hit) = layer.lookup_chat(&info, &request).await {
                    return Ok(replay(hit));
                }
            }
        }
        let call = adapter.chat_call(&request, &key, base_url.as_deref(), true)?;
        let response = self.send(adapter.as_ref(), &call, &key).await?;
        let parser = adapter.stream_parser(&request);
        let layers = self.inner.layers.clone();
        let provider_id = adapter.id().to_owned();

        Ok(Box::pin(async_stream::stream! {
            let mut parser = parser;
            let mut decoder = SseDecoder::new();
            let mut body = response.bytes_stream();
            let mut text = String::new();
            let mut failed = false;
            'read: loop {
                let (pairs, last) = match body.next().await {
                    Some(Ok(chunk)) => (decoder.feed(&chunk), false),
                    Some(Err(error)) => {
                        yield Err(transport_error(&provider_id, &error, &key));
                        failed = true;
                        break 'read;
                    }
                    None => (decoder.flush(), true),
                };
                for (event, data) in pairs {
                    match parser.feed(event.as_deref(), &data) {
                        Ok(events) => {
                            for event in events {
                                if let StreamEvent::Delta { text: delta } = &event {
                                    if !layers.is_empty() {
                                        text.push_str(delta);
                                    }
                                }
                                yield Ok(event);
                            }
                        }
                        Err(error) => {
                            yield Err(redact_error(error, &key));
                            failed = true;
                            break 'read;
                        }
                    }
                }
                if last {
                    break 'read;
                }
            }
            if !failed {
                let done = parser.finish();
                if !layers.is_empty() {
                    if let StreamEvent::Done { finish_reason, model, usage } = &done {
                        let reply = ChatResponse {
                            text: std::mem::take(&mut text),
                            model: model.clone().unwrap_or_else(|| request.model.clone()),
                            finish_reason: *finish_reason,
                            usage: *usage,
                        };
                        let info = CallInfo { provider: &provider_id, base_url: base_url.as_deref(), key: &fingerprint, stream: true };
                        for layer in &layers {
                            layer.on_chat(&info, &request, &reply).await;
                        }
                    }
                }
                yield Ok(done);
            }
        }))
    }

    pub async fn list_models(&self, provider: &str, api_key: &str, base_url: Option<&str>) -> Result<Vec<ModelInfo>> {
        let adapter = self.provider(provider)?;
        let key = require_key(adapter.as_ref(), api_key)?;
        let base_url = self.effective_base_url(adapter.as_ref(), base_url);
        let fingerprint = KeyFingerprint::new(&self.inner.salt, key);
        let info = CallInfo { provider: adapter.id(), base_url, key: &fingerprint, stream: false };
        for layer in &self.inner.layers {
            if let Some(hit) = layer.lookup_models(&info).await {
                return Ok(hit);
            }
        }
        let mut call = Some(adapter.models_call(key, base_url)?);
        let mut models = Vec::new();
        for _ in 0..MAX_MODEL_PAGES {
            let Some(current) = call.take() else { break };
            let response = self.send(adapter.as_ref(), &current, key).await?;
            let payload = read_json(adapter.as_ref(), response).await?;
            models.extend(adapter.parse_models(&payload));
            call = adapter.next_models_page(&payload, &current);
        }
        for layer in &self.inner.layers {
            layer.on_models(&info, &models).await;
        }
        Ok(models)
    }

    fn request(&self, call: &HttpCall) -> reqwest::RequestBuilder {
        let method = match call.method {
            Method::Get => reqwest::Method::GET,
            Method::Post => reqwest::Method::POST,
        };
        let mut builder = self.inner.http.request(method, &call.url);
        for (name, value) in &call.headers {
            builder = builder.header(name, value);
        }
        if !call.params.is_empty() {
            builder = builder.query(&call.params);
        }
        if let Some(json) = &call.json {
            builder = builder.body(serde_json::to_vec(json).unwrap_or_default());
        }
        builder
    }

    /// Send and check the status; the body is left unread.
    async fn send(&self, adapter: &dyn ProviderAdapter, call: &HttpCall, key: &str) -> Result<reqwest::Response> {
        let response = self.request(call).send().await.map_err(|error| transport_error(adapter.id(), &error, key))?;
        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.bytes().await.unwrap_or_default();
            return Err(adapter.error_from_response(status, &String::from_utf8_lossy(&body), Some(key)));
        }
        Ok(response)
    }
}

fn replay(hit: ChatResponse) -> ChatStream {
    let mut events = Vec::with_capacity(2);
    if !hit.text.is_empty() {
        events.push(Ok(StreamEvent::delta(hit.text)));
    }
    events.push(Ok(StreamEvent::Done { finish_reason: hit.finish_reason, model: Some(hit.model), usage: hit.usage }));
    Box::pin(futures_util::stream::iter(events))
}

fn require_key<'a>(adapter: &dyn ProviderAdapter, api_key: &'a str) -> Result<&'a str> {
    let key = api_key.trim();
    if key.is_empty() {
        return Err(ProviderError::new(ErrorCode::Auth, "an API key is required").provider(adapter.id()));
    }
    Ok(key)
}

fn validate(request: &ChatRequest) -> Result<()> {
    request
        .validate()
        .map_err(|message| ProviderError::new(ErrorCode::BadRequest, format!("invalid request: {message}")))
}

fn redact_error(mut error: ProviderError, key: &str) -> ProviderError {
    error.message = redact(&error.message, Some(key));
    error
}

async fn read_json(adapter: &dyn ProviderAdapter, response: reqwest::Response) -> Result<Value> {
    let status = response.status().as_u16();
    let invalid = || {
        ProviderError::new(ErrorCode::Upstream, "the provider returned an invalid response")
            .provider(adapter.id())
            .status(status)
    };
    let body = match response.bytes().await {
        Ok(body) => body,
        Err(error) if error.is_timeout() => {
            return Err(
                ProviderError::new(ErrorCode::Timeout, "the provider did not respond in time").provider(adapter.id())
            )
        }
        Err(_) => return Err(invalid()),
    };
    serde_json::from_slice(&body).map_err(|_| invalid())
}

fn transport_error(provider_id: &str, error: &reqwest::Error, key: &str) -> ProviderError {
    if error.is_timeout() {
        return ProviderError::new(ErrorCode::Timeout, "the provider did not respond in time").provider(provider_id);
    }
    let kind = if error.is_connect() {
        "ConnectError"
    } else if error.is_body() || error.is_decode() {
        "ReadError"
    } else {
        "TransportError"
    };
    ProviderError::new(ErrorCode::Network, redact(&format!("could not reach the provider: {kind}"), Some(key)))
        .provider(provider_id)
}
