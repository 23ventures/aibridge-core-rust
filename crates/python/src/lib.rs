//! `aibridge_core._native`: the Rust engine behind `aibridge_core.providers`.
//!
//! The public Python API lives in `python/aibridge_core/`; this module only moves data. Network
//! I/O runs on a shared Tokio runtime with the GIL released, so other Python threads keep running.
//! Results are plain tuples, which the Python layer turns into the same dataclasses aibridge-exec
//! returns.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aibridge_core::{
    CacheConfig, CacheLayer, ChatCachePolicy, ChatClient, ChatMessage, ChatRequest, ChatResponse, ChatStream,
    ModelInfo, ProviderError, Role, StreamEvent, Timeouts,
};
use futures_util::StreamExt;
use pyo3::exceptions::{PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// (text, model, finish_reason, input_tokens, output_tokens)
type ReplyTuple = (String, String, Option<&'static str>, Option<u64>, Option<u64>);
/// ("delta", text, None, None, None) or ("done", finish_reason, model, input_tokens, output_tokens)
type EventTuple = (&'static str, Option<String>, Option<String>, Option<u64>, Option<u64>);
/// (id, name, context_length)
type ModelTuple = (String, Option<String>, Option<u64>);

fn runtime() -> &'static tokio::runtime::Runtime {
    pyo3_async_runtimes::tokio::get_runtime()
}

/// Raise `aibridge_core.errors.ProviderError(code, message, provider=..., status=...)`.
fn to_pyerr(error: ProviderError) -> PyErr {
    Python::attach(|py| {
        let build = || -> PyResult<PyErr> {
            let class = py.import("aibridge_core.errors")?.getattr("ProviderError")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("provider", error.provider.as_deref())?;
            kwargs.set_item("status", error.status)?;
            let instance = class.call((error.code.as_str(), error.message.as_str()), Some(&kwargs))?;
            Ok(PyErr::from_value(instance))
        };
        build().unwrap_or_else(|e| e)
    })
}

fn reply_tuple(reply: ChatResponse) -> ReplyTuple {
    (
        reply.text,
        reply.model,
        reply.finish_reason.map(|f| f.as_str()),
        reply.usage.input_tokens,
        reply.usage.output_tokens,
    )
}

fn event_tuple(event: StreamEvent) -> EventTuple {
    match event {
        StreamEvent::Delta { text } => ("delta", Some(text), None, None, None),
        StreamEvent::Done { finish_reason, model, usage } => {
            ("done", finish_reason.map(|f| f.as_str().to_owned()), model, usage.input_tokens, usage.output_tokens)
        }
    }
}

fn model_tuples(models: Vec<ModelInfo>) -> Vec<ModelTuple> {
    models.into_iter().map(|m| (m.id, m.name, m.context_length)).collect()
}

fn build_request(
    model: String,
    messages: Vec<(String, String)>,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
) -> PyResult<ChatRequest> {
    let messages = messages
        .into_iter()
        .map(|(role, content)| {
            Role::parse(&role)
                .map(|role| ChatMessage { role, content })
                .ok_or_else(|| PyValueError::new_err(format!("invalid role {role:?}")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(ChatRequest { model, messages, system, temperature, max_tokens })
}

#[derive(Clone)]
struct StreamArgs {
    client: ChatClient,
    provider: String,
    api_key: String,
    request: ChatRequest,
    base_url: Option<String>,
}

enum StreamState {
    /// Not started: the request is sent on the first `next()`, like a Python generator.
    Pending(StreamArgs),
    Running(ChatStream),
    /// An error that arrived after some events of a batch; raised on the next call.
    Failed(ProviderError),
    Finished,
}

/// Most events handed to Python per wake-up.
const BATCH: usize = 256;

impl StreamState {
    async fn next(&mut self) -> Option<Result<StreamEvent, ProviderError>> {
        if let StreamState::Pending(args) = self {
            let args = args.clone();
            match args.client.stream(&args.provider, &args.api_key, args.request, args.base_url.as_deref()).await {
                Ok(stream) => *self = StreamState::Running(stream),
                Err(error) => {
                    *self = StreamState::Finished;
                    return Some(Err(error));
                }
            }
        }
        if let StreamState::Failed(_) = self {
            let StreamState::Failed(error) = std::mem::replace(self, StreamState::Finished) else { unreachable!() };
            return Some(Err(error));
        }
        let StreamState::Running(stream) = self else {
            return None;
        };
        match stream.next().await {
            Some(Ok(event)) => Some(Ok(event)),
            Some(Err(error)) => {
                *self = StreamState::Finished;
                Some(Err(error))
            }
            None => {
                *self = StreamState::Finished;
                None
            }
        }
    }
}

impl StreamState {
    /// Wait for one event, then take every event that is already decoded without waiting.
    /// Crossing into Python once per batch instead of once per event is what keeps streaming
    /// cheap, especially from asyncio (each wake-up is a cross-thread call).
    /// `Ok(empty)` means the stream ended.
    async fn next_batch(&mut self) -> Result<Vec<EventTuple>, ProviderError> {
        let mut batch = Vec::new();
        match self.next().await {
            Some(Ok(event)) => batch.push(event_tuple(event)),
            Some(Err(error)) => return Err(error),
            None => return Ok(batch),
        }
        while batch.len() < BATCH {
            let StreamState::Running(stream) = self else {
                break;
            };
            match futures_util::FutureExt::now_or_never(stream.next()) {
                Some(Some(Ok(event))) => batch.push(event_tuple(event)),
                Some(Some(Err(error))) => {
                    *self = StreamState::Failed(error);
                    break;
                }
                Some(None) => {
                    *self = StreamState::Finished;
                    break;
                }
                None => break, // nothing ready yet
            }
        }
        Ok(batch)
    }
}

/// Blocking iterator over batches of stream events (GIL released while waiting).
#[pyclass(module = "aibridge_core._native")]
struct NativeStream {
    state: Mutex<StreamState>,
}

#[pymethods]
impl NativeStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Next batch of events (a non-empty list); StopIteration at the end.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Vec<EventTuple>>> {
        let result = py.detach(|| {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            runtime().block_on(state.next_batch())
        });
        match result {
            Ok(batch) if batch.is_empty() => Ok(None),
            Ok(batch) => Ok(Some(batch)),
            Err(error) => Err(to_pyerr(error)),
        }
    }

    /// Stop reading and drop the upstream connection.
    fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *state = StreamState::Finished;
    }
}

/// Async iterator over batches of stream events.
#[pyclass(module = "aibridge_core._native")]
struct NativeAsyncStream {
    state: Arc<tokio::sync::Mutex<StreamState>>,
}

#[pymethods]
impl NativeAsyncStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut state = state.lock().await;
            match state.next_batch().await {
                Ok(batch) if batch.is_empty() => Err(PyStopAsyncIteration::new_err(())),
                Ok(batch) => Ok(batch),
                Err(error) => Err(to_pyerr(error)),
            }
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            *state.lock().await = StreamState::Finished;
            Ok(())
        })
    }
}

#[pyclass(module = "aibridge_core._native", frozen)]
struct NativeClient {
    client: ChatClient,
    cache: Option<Arc<CacheLayer>>,
}

#[pymethods]
impl NativeClient {
    #[new]
    #[pyo3(signature = (base_urls=None, connect_timeout=10.0, read_timeout=300.0, cache_models_ttl=None, cache_chat_ttl=None, cache_chat_policy="deterministic", cache_max_entries=10_000))]
    fn new(
        base_urls: Option<HashMap<String, String>>,
        connect_timeout: f64,
        read_timeout: f64,
        cache_models_ttl: Option<f64>,
        cache_chat_ttl: Option<f64>,
        cache_chat_policy: &str,
        cache_max_entries: usize,
    ) -> PyResult<Self> {
        let seconds = |value: f64, name: &str| -> PyResult<Duration> {
            Duration::try_from_secs_f64(value)
                .map_err(|_| PyValueError::new_err(format!("{name} must be a positive number")))
        };
        let positive = |value: Option<f64>| value.filter(|v| *v > 0.0).map(Duration::from_secs_f64);
        let policy = ChatCachePolicy::parse(cache_chat_policy)
            .ok_or_else(|| PyValueError::new_err("cache chat_policy must be 'deterministic' or 'all'"))?;
        let config = CacheConfig {
            models_ttl: positive(cache_models_ttl),
            chat_ttl: positive(cache_chat_ttl),
            chat_policy: policy,
            max_entries: cache_max_entries,
        };
        let cache = config.is_enabled().then(|| Arc::new(CacheLayer::new(config)));
        let mut builder = ChatClient::builder().base_urls(base_urls.unwrap_or_default()).timeouts(Timeouts {
            connect: seconds(connect_timeout, "connect timeout")?,
            read: seconds(read_timeout, "read timeout")?,
        });
        if let Some(cache) = &cache {
            builder = builder.layer(cache.clone());
        }
        // reqwest's client must be built inside the runtime it will be used on.
        let client = runtime().block_on(async { builder.build() }).map_err(to_pyerr)?;
        Ok(Self { client, cache })
    }

    #[pyo3(signature = (provider, api_key, model, messages, system=None, temperature=None, max_tokens=None, base_url=None))]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        py: Python<'_>,
        provider: String,
        api_key: String,
        model: String,
        messages: Vec<(String, String)>,
        system: Option<String>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        base_url: Option<String>,
    ) -> PyResult<ReplyTuple> {
        let request = build_request(model, messages, system, temperature, max_tokens)?;
        let client = self.client.clone();
        py.detach(|| runtime().block_on(client.chat(&provider, &api_key, request, base_url.as_deref())))
            .map(reply_tuple)
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (provider, api_key, model, messages, system=None, temperature=None, max_tokens=None, base_url=None))]
    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        provider: String,
        api_key: String,
        model: String,
        messages: Vec<(String, String)>,
        system: Option<String>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        base_url: Option<String>,
    ) -> PyResult<NativeStream> {
        let request = build_request(model, messages, system, temperature, max_tokens)?;
        let args = StreamArgs { client: self.client.clone(), provider, api_key, request, base_url };
        Ok(NativeStream { state: Mutex::new(StreamState::Pending(args)) })
    }

    #[pyo3(signature = (provider, api_key, base_url=None))]
    fn list_models(
        &self,
        py: Python<'_>,
        provider: String,
        api_key: String,
        base_url: Option<String>,
    ) -> PyResult<Vec<ModelTuple>> {
        let client = self.client.clone();
        py.detach(|| runtime().block_on(client.list_models(&provider, &api_key, base_url.as_deref())))
            .map(model_tuples)
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (provider, api_key, model, messages, system=None, temperature=None, max_tokens=None, base_url=None))]
    #[allow(clippy::too_many_arguments)]
    fn achat<'py>(
        &self,
        py: Python<'py>,
        provider: String,
        api_key: String,
        model: String,
        messages: Vec<(String, String)>,
        system: Option<String>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        base_url: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = build_request(model, messages, system, temperature, max_tokens)?;
        let client = self.client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.chat(&provider, &api_key, request, base_url.as_deref()).await.map(reply_tuple).map_err(to_pyerr)
        })
    }

    #[pyo3(signature = (provider, api_key, model, messages, system=None, temperature=None, max_tokens=None, base_url=None))]
    #[allow(clippy::too_many_arguments)]
    fn astream(
        &self,
        provider: String,
        api_key: String,
        model: String,
        messages: Vec<(String, String)>,
        system: Option<String>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        base_url: Option<String>,
    ) -> PyResult<NativeAsyncStream> {
        let request = build_request(model, messages, system, temperature, max_tokens)?;
        let args = StreamArgs { client: self.client.clone(), provider, api_key, request, base_url };
        Ok(NativeAsyncStream { state: Arc::new(tokio::sync::Mutex::new(StreamState::Pending(args))) })
    }

    #[pyo3(signature = (provider, api_key, base_url=None))]
    fn alist_models<'py>(
        &self,
        py: Python<'py>,
        provider: String,
        api_key: String,
        base_url: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.list_models(&provider, &api_key, base_url.as_deref()).await.map(model_tuples).map_err(to_pyerr)
        })
    }

    /// Cache counters, or None when no cache is configured.
    fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(cache) = &self.cache else {
            return Ok(None);
        };
        let stats = cache.stats();
        let out = PyDict::new(py);
        for (name, counter) in [("models", stats.models), ("chat", stats.chat)] {
            let entry = PyDict::new(py);
            entry.set_item("hits", counter.hits)?;
            entry.set_item("misses", counter.misses)?;
            entry.set_item("entries", counter.entries)?;
            out.set_item(name, entry)?;
        }
        Ok(Some(out))
    }
}

/// Built-in providers as (id, label, key_url, default_base_url, key_hint), in registry order.
#[pyfunction]
fn providers() -> Vec<(String, String, String, String, String)> {
    aibridge_core::Registry::builtin()
        .list()
        .iter()
        .map(|a| {
            let info = a.info();
            (
                info.id.clone(),
                info.label.clone(),
                info.key_url.clone(),
                info.default_base_url.clone(),
                info.key_hint.clone(),
            )
        })
        .collect()
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", aibridge_core::VERSION)?;
    module.add_class::<NativeClient>()?;
    module.add_class::<NativeStream>()?;
    module.add_class::<NativeAsyncStream>()?;
    module.add_function(wrap_pyfunction!(providers, module)?)?;
    Ok(())
}
