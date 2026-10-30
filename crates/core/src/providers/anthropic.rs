//! Anthropic Messages API.

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, obj_field, py_json_dumps, py_str, str_field, stream_error, truthy, HttpCall, ProviderAdapter,
    StreamParser,
};
use crate::error::Result;
use crate::types::{
    merge_consecutive, ChatMessage, ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, Role,
    StreamEvent, Usage,
};

pub const API_VERSION: &str = "2023-06-01";
pub const DEFAULT_MAX_TOKENS: u64 = 4096;
/// Anthropic accepts at most four cache breakpoints per request.
pub const MAX_CACHE_BREAKPOINTS: usize = 4;
/// Sent as the first turn when the conversation would otherwise start with the assistant.
pub const LEADING_USER_TURN: &str = "Continue.";

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

/// Opus 4.6+ and Sonnet 4.5+ reject a custom temperature.
///
/// Same matching as the reference regex `(opus|sonnet)-(\d{1,2})(?:-(\d{1,2}))?(?!\d)`: date
/// suffixes such as `-20250514` are not minor versions.
pub fn skips_temperature(model: &str) -> bool {
    for (index, _) in model.char_indices() {
        let rest = &model[index..];
        let (family, after) = if let Some(after) = rest.strip_prefix("opus-") {
            ("opus", after)
        } else if let Some(after) = rest.strip_prefix("sonnet-") {
            ("sonnet", after)
        } else {
            continue;
        };
        if let Some((major, minor)) = parse_version(after) {
            let threshold = if family == "opus" { (4, 6) } else { (4, 5) };
            return (major, minor) >= threshold;
        }
    }
    false
}

/// `(\d{1,2})(?:-(\d{1,2}))?(?!\d)` at the start of `text`, trying matches in regex order.
fn parse_version(text: &str) -> Option<(u32, u32)> {
    let digits = |s: &str| s.bytes().take_while(u8::is_ascii_digit).count();
    let not_digit_at = |s: &str, at: usize| !s.as_bytes().get(at).is_some_and(u8::is_ascii_digit);
    let major_run = digits(text);
    for major_len in (1..=major_run.min(2)).rev() {
        let major: u32 = text[..major_len].parse().ok()?;
        let after_major = &text[major_len..];
        if let Some(minor_text) = after_major.strip_prefix('-') {
            let minor_run = digits(minor_text);
            for minor_len in (1..=minor_run.min(2)).rev() {
                if not_digit_at(minor_text, minor_len) {
                    return Some((major, minor_text[..minor_len].parse().ok()?));
                }
            }
        }
        if not_digit_at(text, major_len) {
            return Some((major, 0));
        }
    }
    None
}

impl AnthropicAdapter {
    fn auth_headers(&self, api_key: &str) -> Vec<(String, String)> {
        crate::adapter::headers(&[
            ("x-api-key", api_key),
            ("anthropic-version", API_VERSION),
            ("content-type", "application/json"),
        ])
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
        let (systems, turns) = request.system_parts();
        let mut turns = merge_consecutive(turns);
        if turns.first().is_some_and(|t| t.role == Role::Assistant) {
            turns.insert(0, ChatMessage::new(Role::User, LEADING_USER_TURN));
        }
        let mut budget = MAX_CACHE_BREAKPOINTS;
        let mut block = |message: &ChatMessage| -> Value {
            let mut out = json!({"type": "text", "text": message.content});
            if message.cache && budget > 0 {
                out["cache_control"] = json!({"type": "ephemeral"});
                budget -= 1;
            }
            out
        };
        let system_blocks: Vec<Value> = systems.iter().map(&mut block).collect();
        let messages: Vec<Value> = turns
            .iter()
            .map(|turn| {
                if turn.cache {
                    json!({"role": turn.role.as_str(), "content": [block(turn)]})
                } else {
                    json!({"role": turn.role.as_str(), "content": turn.content})
                }
            })
            .collect();
        let mut body = Map::new();
        body.insert("model".into(), json!(request.model));
        body.insert("max_tokens".into(), json!(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)));
        body.insert("messages".into(), Value::Array(messages));
        if !system_blocks.is_empty() {
            if system_blocks.iter().any(|b| b.get("cache_control").is_some()) {
                body.insert("system".into(), Value::Array(system_blocks));
            } else {
                let texts: Vec<&str> = system_blocks.iter().filter_map(|b| b["text"].as_str()).collect();
                body.insert("system".into(), json!(texts.join("\n\n")));
            }
        }
        if let Some(temperature) = request.temperature {
            if !skips_temperature(&request.model) {
                body.insert("temperature".into(), json!(temperature.min(1.0))); // Anthropic accepts 0..1
            }
        }
        if let Some(format) = &request.response_format {
            body.insert(
                "tools".into(),
                json!([{"name": format.name, "description": format.description, "input_schema": format.json_schema()}]),
            );
            body.insert("tool_choice".into(), json!({"type": "tool", "name": format.name}));
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
        let blocks = arr_field(payload, "content");
        let mut text: String = blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect();
        if text.is_empty() {
            // structured output: the forced tool's input is the reply
            if let Some(tool) = blocks.iter().find(|b| b.get("type").and_then(Value::as_str) == Some("tool_use")) {
                text = py_json_dumps(truthy(tool.get("input")).unwrap_or(&json!({})));
            }
        }
        Ok(ChatResponse {
            text,
            model: str_field(payload, "model").unwrap_or(&request.model).to_owned(),
            finish_reason: finish(payload.get("stop_reason")),
            usage: usage(obj_field(payload, "usage"), None),
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

/// Anthropic reports uncached input separately from cache reads and writes; `input_tokens` here
/// is the total, like every other provider's prompt count.
fn usage(usage: &Map<String, Value>, previous: Option<Usage>) -> Usage {
    let previous = previous.unwrap_or_default();
    let uncached = as_count(usage.get("input_tokens"));
    let mut write = as_count(usage.get("cache_creation_input_tokens"));
    let mut read = as_count(usage.get("cache_read_input_tokens"));
    let input = if uncached.is_none() && write.is_none() && read.is_none() {
        write = previous.cache_write_tokens;
        read = previous.cache_read_tokens;
        previous.input_tokens
    } else {
        Some(uncached.unwrap_or(0) + write.unwrap_or(0) + read.unwrap_or(0))
    };
    let output = as_count(usage.get("output_tokens")).or(previous.output_tokens);
    Usage::new(input, output).with_cache(read, write)
}

struct AnthropicStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl StreamParser for AnthropicStreamParser {
    fn feed(&mut self, event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        let Ok(payload) = serde_json::from_str::<Value>(data) else { return Ok(Vec::new()) };
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
                let empty = Map::new();
                self.usage = usage(message.get("usage").and_then(Value::as_object).unwrap_or(&empty), None);
            }
            "content_block_delta" => {
                let delta = obj_field(&payload, "delta");
                let field = match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => "text",
                    Some("input_json_delta") => "partial_json",
                    _ => return Ok(Vec::new()),
                };
                if let Some(text) = delta.get(field).and_then(Value::as_str).filter(|t| !t.is_empty()) {
                    return Ok(vec![StreamEvent::delta(text)]);
                }
            }
            "message_delta" => {
                if let Some(reason) = truthy(obj_field(&payload, "delta").get("stop_reason")) {
                    self.finish_reason = finish(Some(reason));
                }
                self.usage = usage(obj_field(&payload, "usage"), Some(self.usage));
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
    use crate::types::ResponseFormat;

    #[test]
    fn temperature_rule() {
        for model in [
            "claude-opus-4-6",
            "claude-sonnet-4-5-20250929",
            "claude-opus-5",
            "claude-sonnet-5-1",
            "opus-10-2",
            "opus-5-",
            "sonnet-4-5x",
        ] {
            assert!(skips_temperature(model), "{model}");
        }
        for model in [
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet",
            "claude-haiku-4-5",
            "sonnet-123",
            "x-opus-4-sonnet-5",
            "opus-4-123",
        ] {
            assert!(!skips_temperature(model), "{model}");
        }
    }

    #[test]
    fn cache_breakpoints_system_blocks_and_leading_user() {
        let mut messages: Vec<ChatMessage> =
            (0..5).map(|i| ChatMessage::new(Role::System, format!("s{i}")).cached()).collect();
        messages.push(ChatMessage::new(Role::System, "clock"));
        messages.push(ChatMessage::new(Role::Assistant, "earlier"));
        messages.push(ChatMessage::new(Role::User, "Hi"));
        let request = ChatRequest::new("claude-sonnet-4-5", messages).with_temperature(0.3);
        let body = AnthropicAdapter::default().chat_call(&request, "k", None, false).unwrap().json.unwrap();
        let flags: Vec<bool> =
            body["system"].as_array().unwrap().iter().map(|b| b.get("cache_control").is_some()).collect();
        assert_eq!(flags, [true, true, true, true, false, false]);
        assert_eq!(body["messages"][0], json!({"role": "user", "content": "Continue."}));
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn structured_output_and_cache_usage() {
        let adapter = AnthropicAdapter::default();
        let request = ChatRequest::new("claude-x", vec![ChatMessage::new(Role::User, "q")])
            .with_response_format(ResponseFormat::default());
        let body = adapter.chat_call(&request, "k", None, false).unwrap().json.unwrap();
        assert_eq!(body["tool_choice"], json!({"type": "tool", "name": "Give_Information"}));
        let reply = adapter
            .parse_chat(
                &json!({"content": [{"type": "tool_use", "input": {"name": "Café"}}], "stop_reason": "tool_use",
                        "usage": {"input_tokens": 20, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 3000,
                                  "output_tokens": 7}}),
                &request,
            )
            .unwrap();
        assert_eq!(reply.text, "{\"name\": \"Caf\\u00e9\"}");
        assert_eq!(reply.usage, Usage::new(Some(4020), Some(7)).with_cache(Some(3000), Some(1000)));
    }
}
