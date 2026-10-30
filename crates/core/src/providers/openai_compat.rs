//! OpenAI Chat Completions and the providers that speak the same protocol.

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, headers, obj_field, str_field, stream_error, truthy, HttpCall, ProviderAdapter, StreamParser,
};
use crate::error::{ErrorCode, ProviderError, Result};
use crate::types::{ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage};

/// Non-chat model families in OpenAI's /models list.
const OPENAI_NON_CHAT: &[&str] = &[
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "davinci",
    "babbage",
    "transcribe",
    "realtime",
    "audio",
    "image",
    "search",
    "computer-use",
    "sora",
];

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleAdapter {
    info: ProviderInfo,
    max_tokens_field: &'static str,
    stream_usage_option: bool,
    extra_headers: Vec<(String, String)>,
    non_chat_markers: Vec<String>,
    /// `openai_compatible`: no default URL, the caller must pass one.
    requires_base_url: bool,
}

impl OpenAiCompatibleAdapter {
    pub fn new(info: ProviderInfo) -> Self {
        Self {
            info,
            max_tokens_field: "max_tokens",
            stream_usage_option: false,
            extra_headers: Vec::new(),
            non_chat_markers: Vec::new(),
            requires_base_url: false,
        }
    }

    pub fn max_tokens_field(mut self, field: &'static str) -> Self {
        self.max_tokens_field = field;
        self
    }

    /// Ask for usage in the final stream chunk (`stream_options.include_usage`).
    pub fn stream_usage(mut self) -> Self {
        self.stream_usage_option = true;
        self
    }

    pub fn extra_header(mut self, name: &str, value: &str) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    pub fn non_chat(mut self, markers: &[&str]) -> Self {
        self.non_chat_markers = markers.iter().map(|m| (*m).to_owned()).collect();
        self
    }

    fn auth_headers(&self, api_key: &str) -> Vec<(String, String)> {
        let mut out = headers(&[("Authorization", &format!("Bearer {api_key}")), ("Content-Type", "application/json")]);
        out.extend(self.extra_headers.iter().cloned());
        out
    }
}

impl ProviderAdapter for OpenAiCompatibleAdapter {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn base_url(&self, override_url: Option<&str>) -> Result<String> {
        match override_url.filter(|u| !u.is_empty()) {
            Some(url) => Ok(url.trim_end_matches('/').to_owned()),
            None if self.requires_base_url => {
                Err(ProviderError::new(ErrorCode::Config, "base_url is required for openai_compatible")
                    .provider(self.id()))
            }
            None => Ok(self.info.default_base_url.trim_end_matches('/').to_owned()),
        }
    }

    fn chat_call(
        &self,
        request: &ChatRequest,
        api_key: &str,
        base_url: Option<&str>,
        stream: bool,
    ) -> Result<HttpCall> {
        let (system, turns) = request.split_system();
        let mut messages = Vec::with_capacity(turns.len() + 1);
        if let Some(system) = system {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.extend(turns.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})));
        let mut body = Map::new();
        body.insert("model".into(), json!(request.model));
        body.insert("messages".into(), Value::Array(messages));
        body.insert("stream".into(), json!(stream));
        if let Some(temperature) = request.temperature {
            body.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            body.insert(self.max_tokens_field.into(), json!(max_tokens));
        }
        if stream && self.stream_usage_option {
            body.insert("stream_options".into(), json!({"include_usage": true}));
        }
        let url = format!("{}/chat/completions", self.base_url(base_url)?);
        Ok(HttpCall::post(url, self.auth_headers(api_key), Value::Object(body)))
    }

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse> {
        if payload.get("error").is_some_and(Value::is_object) {
            return Err(stream_error(self.id(), payload, None));
        }
        let empty = Value::Object(Map::new());
        let choice = truthy(payload.get("choices")).and_then(|c| c.get(0)).unwrap_or(&empty);
        let message = truthy(choice.get("message")).unwrap_or(&empty);
        Ok(ChatResponse {
            text: content_text(message.get("content")),
            model: str_field(payload, "model").unwrap_or(&request.model).to_owned(),
            finish_reason: finish(choice.get("finish_reason")),
            usage: usage(payload),
        })
    }

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser> {
        Box::new(OpenAiStreamParser {
            provider_id: self.info.id.clone(),
            model: request.model.clone(),
            finish_reason: None,
            usage: Usage::default(),
        })
    }

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall> {
        Ok(HttpCall::get(format!("{}/models", self.base_url(base_url)?), self.auth_headers(api_key)))
    }

    fn parse_models(&self, payload: &Value) -> Vec<ModelInfo> {
        let items: &[Value] = match payload {
            Value::Object(_) => arr_field(payload, "data"),
            Value::Array(items) => items,
            _ => &[],
        };
        let mut models = Vec::new();
        for item in items {
            if !item.is_object() {
                continue;
            }
            let Some(id_value) = truthy(item.get("id")) else {
                continue;
            };
            let model_id = crate::adapter::py_str(id_value);
            let lower = model_id.to_lowercase();
            if self.non_chat_markers.iter().any(|marker| lower.contains(marker.as_str())) {
                continue;
            }
            if let Some(kind) = truthy(item.get("type")) {
                if !matches!(kind.as_str(), Some("chat" | "model" | "language" | "code")) {
                    continue; // Together: image, embedding, rerank, ...
                }
            }
            if item.get("capabilities").and_then(|c| c.get("completion_chat")) == Some(&Value::Bool(false)) {
                continue; // Mistral
            }
            let context = truthy(item.get("context_length"))
                .or_else(|| truthy(item.get("context_window")))
                .or_else(|| truthy(item.get("max_context_length")));
            let name = str_field(item, "name").or_else(|| str_field(item, "display_name")).map(str::to_owned);
            models.push(ModelInfo::new(model_id, name, as_count(context)));
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }
}

struct OpenAiStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl StreamParser for OpenAiStreamParser {
    fn feed(&mut self, _event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        if data.trim() == "[DONE]" {
            return Ok(Vec::new());
        }
        let Ok(payload) = serde_json::from_str::<Value>(data) else {
            return Ok(Vec::new());
        };
        if !payload.is_object() {
            return Ok(Vec::new());
        }
        if payload.get("error").is_some_and(Value::is_object) {
            return Err(stream_error(&self.provider_id, &payload, None));
        }
        if let Some(model) = str_field(&payload, "model") {
            self.model = model.to_owned();
        }
        let chunk_usage = usage(&payload);
        if chunk_usage.input_tokens.is_some() || chunk_usage.output_tokens.is_some() {
            self.usage = chunk_usage;
        }
        let mut events = Vec::new();
        for choice in arr_field(&payload, "choices") {
            let index = choice.get("index").and_then(Value::as_i64).unwrap_or(0);
            if index != 0 {
                continue;
            }
            let delta = choice.get("delta").filter(|d| d.is_object());
            let text = content_text(delta.and_then(|d| d.get("content")));
            if !text.is_empty() {
                events.push(StreamEvent::delta(text));
            }
            if truthy(choice.get("finish_reason")).is_some() {
                self.finish_reason = finish(choice.get("finish_reason"));
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> StreamEvent {
        StreamEvent::Done { finish_reason: self.finish_reason, model: Some(self.model.clone()), usage: self.usage }
    }
}

fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        // content parts
        Some(Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect(),
        _ => String::new(),
    }
}

fn finish(value: Option<&Value>) -> Option<FinishReason> {
    let value = truthy(value)?;
    Some(match crate::adapter::py_str(value).as_str() {
        "stop" | "eos" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        _ => FinishReason::Other,
    })
}

fn usage(payload: &Value) -> Usage {
    let usage = truthy(payload.get("usage")).or_else(|| truthy(obj_field(payload, "x_groq").get("usage")));
    match usage {
        Some(usage @ Value::Object(_)) => {
            Usage::new(as_count(usage.get("prompt_tokens")), as_count(usage.get("completion_tokens")))
        }
        _ => Usage::default(),
    }
}

fn info(id: &str, label: &str, base_url: &str, key_url: &str, hint: &str) -> ProviderInfo {
    ProviderInfo::new(id, label, base_url, key_url, hint)
}

/// Built-in OpenAI-compatible providers, in registry order.
pub fn builtin() -> Vec<OpenAiCompatibleAdapter> {
    vec![
        OpenAiCompatibleAdapter::new(info(
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            "https://platform.openai.com/api-keys",
            "sk-...",
        ))
        .max_tokens_field("max_completion_tokens")
        .stream_usage()
        .non_chat(OPENAI_NON_CHAT),
        OpenAiCompatibleAdapter::new(info(
            "xai",
            "xAI (Grok)",
            "https://api.x.ai/v1",
            "https://console.x.ai",
            "xai-...",
        ))
        .stream_usage()
        .non_chat(&["image", "imagine"]),
        OpenAiCompatibleAdapter::new(info(
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/keys",
            "sk-or-...",
        ))
        .stream_usage(),
        OpenAiCompatibleAdapter::new(info(
            "groq",
            "Groq",
            "https://api.groq.com/openai/v1",
            "https://console.groq.com/keys",
            "gsk_...",
        ))
        .non_chat(&["whisper", "tts", "guard"]),
        OpenAiCompatibleAdapter::new(info(
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com",
            "https://platform.deepseek.com/api_keys",
            "sk-...",
        ))
        .stream_usage(),
        OpenAiCompatibleAdapter::new(info(
            "mistral",
            "Mistral",
            "https://api.mistral.ai/v1",
            "https://console.mistral.ai/api-keys",
            "",
        ))
        .non_chat(&["embed", "moderation", "ocr"]),
        OpenAiCompatibleAdapter::new(info(
            "together",
            "Together AI",
            "https://api.together.xyz/v1",
            "https://api.together.ai/settings/api-keys",
            "",
        )),
    ]
}

/// Any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, gateways). Needs `base_url`.
pub fn custom() -> OpenAiCompatibleAdapter {
    let mut adapter =
        OpenAiCompatibleAdapter::new(info("openai_compatible", "OpenAI-compatible (custom URL)", "", "", ""));
    adapter.requires_base_url = true;
    adapter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Role};

    fn openai() -> OpenAiCompatibleAdapter {
        builtin().remove(0)
    }

    fn request() -> ChatRequest {
        ChatRequest::new("gpt-x", vec![ChatMessage::new(Role::System, "be brief"), ChatMessage::new(Role::User, "Hi")])
            .with_max_tokens(10)
            .with_temperature(0.5)
    }

    #[test]
    fn chat_call_shape() {
        let call = openai().chat_call(&request(), "sk-key", None, true).unwrap();
        assert_eq!(call.url, "https://api.openai.com/v1/chat/completions");
        assert_eq!(call.header("authorization"), Some("Bearer sk-key"));
        assert_eq!(
            call.json.unwrap(),
            json!({"model": "gpt-x", "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "Hi"}],
                   "stream": true, "temperature": 0.5, "max_completion_tokens": 10, "stream_options": {"include_usage": true}})
        );
    }

    #[test]
    fn parse_chat_and_stream() {
        let reply = openai()
            .parse_chat(
                &json!({"model": "gpt-x-1", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1}}),
                &request(),
            )
            .unwrap();
        assert_eq!((reply.text.as_str(), reply.model.as_str()), ("ok", "gpt-x-1"));
        assert_eq!(reply.usage, Usage::new(Some(3), Some(1)));

        let mut parser = openai().stream_parser(&request());
        let mut out = parser.feed(None, r#"{"choices":[{"index":0,"delta":{"content":"He"}}]}"#).unwrap();
        out.extend(parser.feed(None, r#"{"choices":[{"index":1,"delta":{"content":"zz"}}]}"#).unwrap());
        out.extend(parser.feed(None, r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#).unwrap());
        out.extend(parser.feed(None, "[DONE]").unwrap());
        out.push(parser.finish());
        assert_eq!(out[0], StreamEvent::delta("He"));
        assert_eq!(
            out[1],
            StreamEvent::Done {
                finish_reason: Some(FinishReason::Length),
                model: Some("gpt-x".into()),
                usage: Usage::new(Some(2), Some(1))
            }
        );
        let error = parser.feed(None, r#"{"error":{"message":"overloaded","type":"server_error"}}"#).unwrap_err();
        assert_eq!(error.code, ErrorCode::Upstream);
    }

    #[test]
    fn models_filter_non_chat_and_sort() {
        let models = openai()
            .parse_models(&json!({"data": [{"id": "whisper-1"}, {"id": "gpt-b"}, {"id": "gpt-a"}, {"id": ""}]}));
        assert_eq!(models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["gpt-a", "gpt-b"]);
        let together = builtin().pop().unwrap();
        let models = together.parse_models(
            &json!([{"id": "a", "type": "image"}, {"id": "b", "type": "chat", "context_length": 8192.0}]),
        );
        assert_eq!(models, vec![ModelInfo::new("b", None, Some(8192))]);
    }

    #[test]
    fn custom_requires_base_url() {
        let error = custom().chat_call(&request(), "k", None, false).unwrap_err();
        assert_eq!(error.code, ErrorCode::Config);
        let call = custom().chat_call(&request(), "k", Some("http://localhost:11434/v1/"), false).unwrap();
        assert_eq!(call.url, "http://localhost:11434/v1/chat/completions");
    }
}
