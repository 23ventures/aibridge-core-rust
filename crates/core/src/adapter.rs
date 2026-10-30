//! Adapter interface: each provider describes HTTP calls and parses responses.
//!
//! Adapters do no I/O. [`crate::ChatClient`] runs the calls, so adapters are tested without a
//! network and new providers are one type implementing [`ProviderAdapter`].

use serde_json::{Map, Value};

use crate::error::{redact, truncate_chars, ErrorCode, ProviderError, Result};
use crate::types::{ChatRequest, ChatResponse, ModelInfo, ProviderInfo, StreamEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HttpCall {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub params: Vec<(String, String)>,
    pub json: Option<Value>,
}

impl HttpCall {
    pub fn get(url: String, headers: Vec<(String, String)>) -> Self {
        Self { method: Method::Get, url, headers, params: Vec::new(), json: None }
    }

    pub fn post(url: String, headers: Vec<(String, String)>, json: Value) -> Self {
        Self { method: Method::Post, url, headers, params: Vec::new(), json: Some(json) }
    }

    pub fn with_params(mut self, params: &[(&str, &str)]) -> Self {
        for (key, value) in params {
            self.set_param(key, value);
        }
        self
    }

    /// Set or replace one query parameter (used for pagination).
    pub fn set_param(&mut self, key: &str, value: &str) {
        match self.params.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = value.to_owned(),
            None => self.params.push((key.to_owned(), value.to_owned())),
        }
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// How a provider frames streamed replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFormat {
    /// Server-Sent Events.
    Sse,
    /// Newline-delimited JSON (Ollama's native API).
    Ndjson,
    /// NDJSON unless the response says `text/event-stream`.
    Auto,
}

/// Stateful parser for one streamed response.
pub trait StreamParser: Send {
    /// Handle one event (an SSE event, or one NDJSON line with no event name).
    fn feed(&mut self, event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>>;
    /// Events still buffered when the stream ends (before `finish`).
    fn flush_events(&mut self) -> Vec<StreamEvent> {
        Vec::new()
    }
    /// The final `done` event (usage, finish reason) once the stream ends.
    fn finish(&mut self) -> StreamEvent;
}

pub trait ProviderAdapter: Send + Sync {
    fn info(&self) -> &ProviderInfo;

    /// `false` for servers that take no key (a local Ollama); a key is then sent only if given.
    fn requires_key(&self) -> bool {
        true
    }

    fn stream_format(&self) -> StreamFormat {
        StreamFormat::Sse
    }

    /// Only usable with a caller- or server-supplied URL (no public default endpoint).
    fn needs_custom_base_url(&self) -> bool {
        false
    }

    fn id(&self) -> &str {
        &self.info().id
    }

    fn base_url(&self, override_url: Option<&str>) -> Result<String> {
        let url = override_url.filter(|u| !u.is_empty()).unwrap_or(&self.info().default_base_url);
        Ok(url.trim_end_matches('/').to_owned())
    }

    fn chat_call(&self, request: &ChatRequest, api_key: &str, base_url: Option<&str>, stream: bool)
        -> Result<HttpCall>;

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse>;

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser>;

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall>;

    fn parse_models(&self, payload: &Value) -> Vec<ModelInfo>;

    /// Follow-up call for paginated model lists, if any.
    fn next_models_page(&self, _payload: &Value, _call: &HttpCall) -> Option<HttpCall> {
        None
    }

    fn error_from_response(&self, status: u16, body: &str, api_key: Option<&str>) -> ProviderError {
        let message = error_message(body).unwrap_or_else(|| format!("HTTP {status}"));
        ProviderError::new(ErrorCode::for_status(status), truncate_chars(&redact(&message, api_key), 500))
            .provider(self.id())
            .status(status)
    }
}

/// Best-effort message from a provider error body (`_error_message` in Python).
pub fn error_message(body: &str) -> Option<String> {
    let payload: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => {
            let text = truncate_chars(body.trim(), 300);
            return (!text.is_empty()).then_some(text);
        }
    };
    let payload = match &payload {
        Value::Array(items) if !items.is_empty() => &items[0],
        other => other,
    };
    let object = payload.as_object()?;
    let error = object.get("error").unwrap_or(payload);
    if let Some(error) = error.as_object() {
        let message = truthy(error.get("message")).or_else(|| truthy(error.get("detail")));
        if let Some(Value::String(text)) = message {
            return Some(text.clone());
        }
    }
    if let Some(text) = error.as_str() {
        return Some(text.to_owned());
    }
    truthy(object.get("message")).or_else(|| truthy(object.get("detail"))).and_then(|v| v.as_str().map(str::to_owned))
}

/// Error object found inside a 200 response or a stream (`stream_error` in Python).
pub fn stream_error(adapter_id: &str, payload: &Value, api_key: Option<&str>) -> ProviderError {
    let error = match payload.get("error") {
        Some(inner @ Value::Object(_)) => inner,
        _ => payload,
    };
    let message = match truthy(error.get("message")) {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => "stream error".to_owned(),
    };
    let kind = truthy(error.get("type")).or_else(|| truthy(error.get("code"))).map(py_str).unwrap_or_default();
    let code = if kind.contains("rate") { ErrorCode::RateLimited } else { ErrorCode::Upstream };
    ProviderError::new(code, truncate_chars(&redact(&message, api_key), 500)).provider(adapter_id)
}

// ------------------------------------------------------------------ JSON helpers
// Small helpers that follow Python truthiness, so parsing matches the reference adapters.

/// `value` if it is "truthy" in Python terms.
pub fn truthy(value: Option<&Value>) -> Option<&Value> {
    value.filter(|v| match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    })
}

/// Non-empty string field.
pub fn str_field<'a>(object: &'a Value, key: &str) -> Option<&'a str> {
    object.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// Truthy object field, or an empty map.
pub fn obj_field<'a>(object: &'a Value, key: &str) -> &'a Map<String, Value> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    object.get(key).and_then(Value::as_object).unwrap_or_else(|| EMPTY.get_or_init(Map::new))
}

/// Truthy array field, or an empty slice.
pub fn arr_field<'a>(object: &'a Value, key: &str) -> &'a [Value] {
    object.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

/// Integer token counts (floats are truncated, like `int()`).
pub fn as_count(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(n) => n.as_u64().or_else(|| n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)),
        _ => None,
    }
}

/// `str(value)` for scalars.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `json.dumps(value)` with Python's defaults: `", "` / `": "` separators and non-ASCII escaped.
/// Used where the reference adapters re-serialize provider JSON (tool inputs), so reply text
/// matches byte for byte.
pub fn py_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_py_json(value, &mut out);
    out
}

fn write_py_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(text) => write_py_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_py_string(key, out);
                out.push_str(": ");
                write_py_json(item, out);
            }
            out.push('}');
        }
    }
}

fn write_py_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || !c.is_ascii() => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_dumps() {
        let value = json!({"name": "Café ☕", "list": [1, 2.5, null, true], "nested": {"q": "a\"b\n"}, "e": {}});
        assert_eq!(
            py_json_dumps(&value),
            r#"{"name": "Caf\u00e9 \u2615", "list": [1, 2.5, null, true], "nested": {"q": "a\"b\n"}, "e": {}}"#
        );
    }

    #[test]
    fn error_messages_from_common_shapes() {
        assert_eq!(error_message(r#"{"error":{"message":"m"}}"#).as_deref(), Some("m"));
        assert_eq!(error_message(r#"[{"error":{"message":"g"}}]"#).as_deref(), Some("g"));
        assert_eq!(error_message(r#"{"error":"s"}"#).as_deref(), Some("s"));
        assert_eq!(error_message(r#"{"detail":"d"}"#).as_deref(), Some("d"));
        assert_eq!(error_message("  plain text ").as_deref(), Some("plain text"));
        assert_eq!(error_message(""), None);
        assert_eq!(error_message("[]"), None);
    }

    #[test]
    fn stream_errors_classify_rate_limits() {
        let error = stream_error("x", &json!({"error": {"type": "rate_limit_error", "message": "slow"}}), None);
        assert_eq!((error.code, error.message.as_str()), (ErrorCode::RateLimited, "slow"));
        let error = stream_error("x", &json!({"type": "error"}), None);
        assert_eq!((error.code, error.message.as_str()), (ErrorCode::Upstream, "stream error"));
    }
}
