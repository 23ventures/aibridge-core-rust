//! Ollama's native API (`/api/chat`, `/api/tags`).
//!
//! Local servers take no key; Ollama's cloud (`https://ollama.com`) takes a Bearer key, sent only
//! when one is given. Streams are newline-delimited JSON. There is no public default server, so
//! callers pass `base_url` (the server's root, e.g. `http://gpu-box:11434`).

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, py_str, str_field, stream_error, truthy, HttpCall, ProviderAdapter, StreamFormat, StreamParser,
};
use crate::error::Result;
use crate::providers::flat_messages;
use crate::types::{ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage};

#[derive(Debug, Clone)]
pub struct OllamaAdapter {
    info: ProviderInfo,
}

impl Default for OllamaAdapter {
    fn default() -> Self {
        Self {
            info: ProviderInfo::new(
                "ollama",
                "Ollama",
                "http://localhost:11434",
                "https://ollama.com/settings/keys",
                "",
            ),
        }
    }
}

impl OllamaAdapter {
    fn auth_headers(&self, api_key: &str) -> Vec<(String, String)> {
        let mut headers = vec![("Content-Type".to_owned(), "application/json".to_owned())];
        if !api_key.is_empty() {
            headers.push(("Authorization".to_owned(), format!("Bearer {api_key}")));
        }
        headers
    }
}

impl ProviderAdapter for OllamaAdapter {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn requires_key(&self) -> bool {
        false
    }

    fn stream_format(&self) -> StreamFormat {
        StreamFormat::Ndjson
    }

    fn needs_custom_base_url(&self) -> bool {
        true
    }

    /// Accepts the server root, `/api` or an OpenAI-style `/v1` URL.
    fn base_url(&self, override_url: Option<&str>) -> Result<String> {
        let url = override_url.filter(|u| !u.is_empty()).unwrap_or(&self.info.default_base_url);
        let mut url = url.trim_end_matches('/');
        for suffix in ["/api", "/v1"] {
            url = url.strip_suffix(suffix).unwrap_or(url);
        }
        Ok(url.to_owned())
    }

    fn chat_call(
        &self,
        request: &ChatRequest,
        api_key: &str,
        base_url: Option<&str>,
        stream: bool,
    ) -> Result<HttpCall> {
        let mut body = Map::new();
        body.insert("model".into(), json!(request.model));
        body.insert("messages".into(), Value::Array(flat_messages(request)));
        body.insert("stream".into(), json!(stream));
        let mut options = Map::new();
        if let Some(temperature) = request.temperature {
            options.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            options.insert("num_predict".into(), json!(max_tokens));
        }
        if !options.is_empty() {
            body.insert("options".into(), Value::Object(options));
        }
        if let Some(format) = &request.response_format {
            body.insert("format".into(), if format.has_schema() { format.json_schema() } else { json!("json") });
        }
        if let Some(thinking) = request.thinking {
            body.insert("think".into(), json!(thinking));
        }
        let url = format!("{}/api/chat", self.base_url(base_url)?);
        Ok(HttpCall::post(url, self.auth_headers(api_key), Value::Object(body)))
    }

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse> {
        if let Some(error) = truthy(payload.get("error")) {
            return Err(stream_error(self.id(), &json!({"message": py_str(error)}), None));
        }
        Ok(ChatResponse {
            text: content(payload),
            model: str_field(payload, "model").unwrap_or(&request.model).to_owned(),
            finish_reason: finish(payload),
            usage: usage(payload),
        })
    }

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser> {
        Box::new(OllamaStreamParser {
            provider_id: self.info.id.clone(),
            model: request.model.clone(),
            finish_reason: None,
            usage: Usage::default(),
        })
    }

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall> {
        Ok(HttpCall::get(format!("{}/api/tags", self.base_url(base_url)?), self.auth_headers(api_key)))
    }

    fn parse_models(&self, payload: &Value) -> Vec<ModelInfo> {
        let mut models: Vec<ModelInfo> = arr_field(payload, "models")
            .iter()
            .filter(|item| item.is_object())
            .filter_map(|item| {
                let id = truthy(item.get("model")).or_else(|| truthy(item.get("name")))?;
                Some(ModelInfo::new(py_str(id), item.get("name").and_then(Value::as_str).map(str::to_owned), None))
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }
}

fn content(payload: &Value) -> String {
    payload.get("message").and_then(|m| m.get("content")).and_then(Value::as_str).unwrap_or("").to_owned()
}

fn finish(payload: &Value) -> Option<FinishReason> {
    truthy(payload.get("done"))?;
    let reason = truthy(payload.get("done_reason")).map(py_str).unwrap_or_else(|| "stop".into());
    Some(match reason.as_str() {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        _ => FinishReason::Other,
    })
}

fn usage(payload: &Value) -> Usage {
    Usage::new(as_count(payload.get("prompt_eval_count")), as_count(payload.get("eval_count")))
}

struct OllamaStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl StreamParser for OllamaStreamParser {
    fn feed(&mut self, _event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        let Ok(payload) = serde_json::from_str::<Value>(data) else { return Ok(Vec::new()) };
        if !payload.is_object() {
            return Ok(Vec::new());
        }
        if let Some(error) = truthy(payload.get("error")) {
            return Err(stream_error(&self.provider_id, &json!({"message": error}), None));
        }
        if let Some(model) = str_field(&payload, "model") {
            self.model = model.to_owned();
        }
        if truthy(payload.get("done")).is_some() {
            self.finish_reason = finish(&payload);
            self.usage = usage(&payload);
        }
        let text = content(&payload);
        Ok(if text.is_empty() { Vec::new() } else { vec![StreamEvent::delta(text)] })
    }

    fn finish(&mut self) -> StreamEvent {
        StreamEvent::Done { finish_reason: self.finish_reason, model: Some(self.model.clone()), usage: self.usage }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ResponseFormat, Role};

    fn request() -> ChatRequest {
        ChatRequest::new("llama3", vec![ChatMessage::new(Role::User, "q")])
    }

    #[test]
    fn keyless_calls_and_base_urls() {
        let adapter = OllamaAdapter::default();
        let call = adapter
            .chat_call(
                &request().with_response_format(ResponseFormat::default()).with_thinking(false).with_temperature(0.1),
                "",
                Some("http://gpu:11434/v1/"),
                true,
            )
            .unwrap();
        assert_eq!(call.url, "http://gpu:11434/api/chat");
        assert_eq!(call.header("Authorization"), None);
        assert_eq!(
            call.json.unwrap(),
            json!({"model": "llama3", "messages": [{"role": "user", "content": "q"}], "stream": true,
                   "options": {"temperature": 0.1}, "format": "json", "think": false})
        );
        assert_eq!(
            adapter.models_call("key", Some("https://ollama.com/api")).unwrap().header("Authorization"),
            Some("Bearer key")
        );
        assert!(!adapter.requires_key() && adapter.needs_custom_base_url());
    }

    #[test]
    fn parse_stream_and_models() {
        let adapter = OllamaAdapter::default();
        let mut parser = adapter.stream_parser(&request());
        assert_eq!(
            parser.feed(None, r#"{"model": "llama3:8b", "message": {"content": "hi"}, "done": false}"#).unwrap(),
            vec![StreamEvent::delta("hi")]
        );
        parser
            .feed(None, r#"{"done": true, "done_reason": "length", "prompt_eval_count": 3, "eval_count": 4}"#)
            .unwrap();
        assert_eq!(
            parser.finish(),
            StreamEvent::Done {
                finish_reason: Some(FinishReason::Length),
                model: Some("llama3:8b".into()),
                usage: Usage::new(Some(3), Some(4))
            }
        );
        let err = adapter.parse_chat(&json!({"error": "model not found"}), &request()).unwrap_err();
        assert_eq!(err.message, "model not found");
        let models =
            adapter.parse_models(&json!({"models": [{"name": "b:latest", "model": "b:latest"}, {"name": "a"}]}));
        assert_eq!(
            models,
            vec![
                ModelInfo::new("a", Some("a".into()), None),
                ModelInfo::new("b:latest", Some("b:latest".into()), None)
            ]
        );
    }
}
