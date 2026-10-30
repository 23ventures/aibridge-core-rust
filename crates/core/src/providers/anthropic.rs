//! Anthropic Messages API.

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, headers, obj_field, py_str, str_field, stream_error, truthy, HttpCall, ProviderAdapter,
    StreamParser,
};
use crate::error::Result;
use crate::types::{
    merge_consecutive, ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage,
};

pub const API_VERSION: &str = "2023-06-01";
pub const DEFAULT_MAX_TOKENS: u64 = 4096;

#[derive(Debug, Clone)]
pub struct AnthropicAdapter {
    info: ProviderInfo,
}

impl Default for AnthropicAdapter {
    fn default() -> Self {
        Self {
            info: ProviderInfo::new(
                "anthropic",
                "Anthropic",
                "https://api.anthropic.com/v1",
                "https://console.anthropic.com/settings/keys",
                "sk-ant-...",
            ),
        }
    }
}

impl AnthropicAdapter {
    fn auth_headers(&self, api_key: &str) -> Vec<(String, String)> {
        headers(&[("x-api-key", api_key), ("anthropic-version", API_VERSION), ("content-type", "application/json")])
    }
}

impl ProviderAdapter for AnthropicAdapter {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn chat_call(
        &self,
        request: &ChatRequest,
        api_key: &str,
        base_url: Option<&str>,
        stream: bool,
    ) -> Result<HttpCall> {
        let (system, turns) = request.split_system();
        let messages: Vec<Value> =
            merge_consecutive(turns).iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect();
        let mut body = Map::new();
        body.insert("model".into(), json!(request.model));
        body.insert("max_tokens".into(), json!(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)));
        body.insert("messages".into(), Value::Array(messages));
        if let Some(system) = system {
            body.insert("system".into(), json!(system));
        }
        if let Some(temperature) = request.temperature {
            body.insert("temperature".into(), json!(temperature.min(1.0))); // Anthropic accepts 0..1
        }
        if stream {
            body.insert("stream".into(), json!(true));
        }
        let url = format!("{}/messages", self.base_url(base_url)?);
        Ok(HttpCall::post(url, self.auth_headers(api_key), Value::Object(body)))
    }

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse> {
        if payload.get("type").and_then(Value::as_str) == Some("error") {
            return Err(stream_error(self.id(), payload, None));
        }
        let text: String = arr_field(payload, "content")
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect();
        let usage = obj_field(payload, "usage");
        Ok(ChatResponse {
            text,
            model: str_field(payload, "model").unwrap_or(&request.model).to_owned(),
            finish_reason: finish(payload.get("stop_reason")),
            usage: Usage::new(as_count(usage.get("input_tokens")), as_count(usage.get("output_tokens"))),
        })
    }

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser> {
        Box::new(AnthropicStreamParser {
            provider_id: self.info.id.clone(),
            model: request.model.clone(),
            finish_reason: None,
            usage: Usage::default(),
        })
    }

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall> {
        Ok(HttpCall::get(format!("{}/models", self.base_url(base_url)?), self.auth_headers(api_key))
            .with_params(&[("limit", "1000")]))
    }

    fn parse_models(&self, payload: &Value) -> Vec<ModelInfo> {
        arr_field(payload, "data")
            .iter()
            .filter(|item| item.is_object())
            .filter_map(|item| {
                let id = truthy(item.get("id"))?;
                Some(ModelInfo::new(py_str(id), str_field(item, "display_name").map(str::to_owned), None))
            })
            .collect()
    }

    fn next_models_page(&self, payload: &Value, call: &HttpCall) -> Option<HttpCall> {
        truthy(payload.get("has_more"))?;
        let last_id = truthy(payload.get("last_id"))?;
        let mut next = call.clone();
        next.set_param("after_id", &py_str(last_id));
        Some(next)
    }
}

struct AnthropicStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl StreamParser for AnthropicStreamParser {
    fn feed(&mut self, event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        let Ok(payload) = serde_json::from_str::<Value>(data) else {
            return Ok(Vec::new());
        };
        if !payload.is_object() {
            return Ok(Vec::new());
        }
        let kind = str_field(&payload, "type").or(event.filter(|e| !e.is_empty())).unwrap_or("");
        match kind {
            "error" => return Err(stream_error(&self.provider_id, &payload, None)),
            "message_start" => {
                let message = obj_field(&payload, "message");
                if let Some(model) = message.get("model").and_then(Value::as_str).filter(|m| !m.is_empty()) {
                    self.model = model.to_owned();
                }
                let usage = message.get("usage").and_then(Value::as_object);
                self.usage = Usage::new(
                    as_count(usage.and_then(|u| u.get("input_tokens"))),
                    as_count(usage.and_then(|u| u.get("output_tokens"))),
                );
            }
            "content_block_delta" => {
                let delta = obj_field(&payload, "delta");
                if delta.get("type").and_then(Value::as_str) == Some("text_delta") {
                    if let Some(text) = delta.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                        return Ok(vec![StreamEvent::delta(text)]);
                    }
                }
            }
            "message_delta" => {
                if let Some(reason) = truthy(obj_field(&payload, "delta").get("stop_reason")) {
                    self.finish_reason = finish(Some(reason));
                }
                let usage = obj_field(&payload, "usage");
                if let Some(output) = as_count(usage.get("output_tokens")) {
                    self.usage.output_tokens = Some(output);
                }
                if let Some(input) = as_count(usage.get("input_tokens")) {
                    self.usage.input_tokens = Some(input);
                }
            }
            _ => {}
        }
        Ok(Vec::new())
    }

    fn finish(&mut self) -> StreamEvent {
        StreamEvent::Done { finish_reason: self.finish_reason, model: Some(self.model.clone()), usage: self.usage }
    }
}

fn finish(value: Option<&Value>) -> Option<FinishReason> {
    let value = truthy(value)?;
    Some(match py_str(value).as_str() {
        "end_turn" | "stop_sequence" | "pause_turn" => FinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Role};

    fn request() -> ChatRequest {
        ChatRequest::new(
            "claude-x",
            vec![
                ChatMessage::new(Role::System, "sys"),
                ChatMessage::new(Role::User, "a"),
                ChatMessage::new(Role::User, "b"),
            ],
        )
        .with_temperature(1.5)
    }

    #[test]
    fn chat_call_merges_turns_and_clamps_temperature() {
        let call = AnthropicAdapter::default().chat_call(&request(), "sk-ant", None, false).unwrap();
        assert_eq!(call.header("x-api-key"), Some("sk-ant"));
        assert_eq!(
            call.json.unwrap(),
            json!({"model": "claude-x", "max_tokens": 4096, "messages": [{"role": "user", "content": "a\n\nb"}],
                   "system": "sys", "temperature": 1.0})
        );
    }

    #[test]
    fn stream_events() {
        let mut parser = AnthropicAdapter::default().stream_parser(&request());
        let mut out = Vec::new();
        for (event, data) in [
            (
                "message_start",
                r#"{"type":"message_start","message":{"model":"claude-x-1","usage":{"input_tokens":5,"output_tokens":1}}}"#,
            ),
            ("content_block_delta", r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}"#),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":7}}"#,
            ),
        ] {
            out.extend(parser.feed(Some(event), data).unwrap());
        }
        out.push(parser.finish());
        assert_eq!(out[0], StreamEvent::delta("Hi"));
        assert_eq!(
            out[1],
            StreamEvent::Done {
                finish_reason: Some(FinishReason::Length),
                model: Some("claude-x-1".into()),
                usage: Usage::new(Some(5), Some(7))
            }
        );
    }

    #[test]
    fn models_pagination() {
        let adapter = AnthropicAdapter::default();
        let call = adapter.models_call("k", None).unwrap();
        let next = adapter.next_models_page(&json!({"has_more": true, "last_id": "m2"}), &call).unwrap();
        assert_eq!(
            next.params,
            vec![("limit".to_owned(), "1000".to_owned()), ("after_id".to_owned(), "m2".to_owned())]
        );
        assert!(adapter.next_models_page(&json!({"has_more": false}), &call).is_none());
    }
}
