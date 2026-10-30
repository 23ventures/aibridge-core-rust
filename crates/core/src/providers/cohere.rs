//! Cohere v2 Chat API (native).

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, headers, obj_field, py_str, stream_error, truthy, HttpCall, ProviderAdapter, StreamParser,
};
use crate::error::Result;
use crate::providers::flat_messages;
use crate::types::{ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage};

#[derive(Debug, Clone)]
pub struct CohereAdapter {
    info: ProviderInfo,
}

impl Default for CohereAdapter {
    fn default() -> Self {
        Self {
            info: ProviderInfo::new(
                "cohere",
                "Cohere",
                "https://api.cohere.com",
                "https://dashboard.cohere.com/api-keys",
                "",
            ),
        }
    }
}

impl CohereAdapter {
    fn auth_headers(&self, api_key: &str) -> Vec<(String, String)> {
        headers(&[
            ("Authorization", &format!("Bearer {api_key}")),
            ("Content-Type", "application/json"),
            ("Accept", "application/json"),
        ])
    }
}

impl ProviderAdapter for CohereAdapter {
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
        let mut body = Map::new();
        body.insert("model".into(), json!(request.model));
        body.insert("messages".into(), Value::Array(flat_messages(request)));
        body.insert("stream".into(), json!(stream));
        if let Some(temperature) = request.temperature {
            body.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            body.insert("max_tokens".into(), json!(max_tokens));
        }
        if let Some(format) = &request.response_format {
            let mut response_format = json!({"type": "json_object"});
            if format.has_schema() {
                response_format["json_schema"] = format.json_schema();
            }
            body.insert("response_format".into(), response_format);
        }
        if let Some(thinking) = request.thinking {
            body.insert("thinking".into(), json!({"type": if thinking { "enabled" } else { "disabled" }}));
        }
        let url = format!("{}/v2/chat", self.base_url(base_url)?);
        Ok(HttpCall::post(url, self.auth_headers(api_key), Value::Object(body)))
    }

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse> {
        let message_is_text = truthy(payload.get("message")).is_some_and(Value::is_string);
        if payload.get("type").and_then(Value::as_str) == Some("error") || message_is_text {
            return Err(stream_error(self.id(), payload, None));
        }
        let parts = obj_field(payload, "message").get("content").and_then(Value::as_array);
        let text: String = parts
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        Ok(ChatResponse {
            text,
            model: request.model.clone(),
            finish_reason: finish(payload.get("finish_reason")),
            usage: usage(payload.get("usage")),
        })
    }

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser> {
        Box::new(CohereStreamParser {
            provider_id: self.info.id.clone(),
            model: request.model.clone(),
            finish_reason: None,
            usage: Usage::default(),
        })
    }

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall> {
        Ok(HttpCall::get(format!("{}/v1/models", self.base_url(base_url)?), self.auth_headers(api_key))
            .with_params(&[("endpoint", "chat"), ("page_size", "1000")]))
    }

    fn parse_models(&self, payload: &Value) -> Vec<ModelInfo> {
        let mut models: Vec<ModelInfo> = arr_field(payload, "models")
            .iter()
            .filter(|item| item.is_object())
            .filter_map(|item| {
                let name = truthy(item.get("name"))?;
                Some(ModelInfo::new(py_str(name), None, as_count(item.get("context_length"))))
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    fn next_models_page(&self, payload: &Value, call: &HttpCall) -> Option<HttpCall> {
        let token = truthy(payload.get("next_page_token"))?;
        let mut next = call.clone();
        next.set_param("page_token", &py_str(token));
        Some(next)
    }
}

struct CohereStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl StreamParser for CohereStreamParser {
    fn feed(&mut self, event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        let Ok(payload) = serde_json::from_str::<Value>(data) else { return Ok(Vec::new()) };
        if !payload.is_object() {
            return Ok(Vec::new());
        }
        let kind = truthy(payload.get("type")).and_then(Value::as_str).or(event).unwrap_or("");
        if kind == "error" || payload.get("error").is_some_and(Value::is_object) {
            return Err(stream_error(&self.provider_id, &payload, None));
        }
        let delta = obj_field(&payload, "delta");
        match kind {
            "content-delta" => {
                let text = delta
                    .get("message")
                    .and_then(Value::as_object)
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_object)
                    .and_then(|c| c.get("text"))
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty());
                return Ok(text.map(|t| vec![StreamEvent::delta(t)]).unwrap_or_default());
            }
            "message-end" => {
                self.finish_reason = finish(delta.get("finish_reason"));
                self.usage = usage(delta.get("usage"));
            }
            _ => {}
        }
        Ok(Vec::new())
    }

    fn finish(&mut self) -> StreamEvent {
        StreamEvent::Done { finish_reason: self.finish_reason, model: Some(self.model.clone()), usage: self.usage }
    }
}

fn usage(value: Option<&Value>) -> Usage {
    let Some(usage) = value.filter(|v| v.is_object()) else { return Usage::default() };
    let tokens = truthy(usage.get("tokens")).or_else(|| truthy(usage.get("billed_units")));
    let Some(tokens) = tokens.filter(|t| t.is_object()) else { return Usage::default() };
    Usage::new(as_count(tokens.get("input_tokens")), as_count(tokens.get("output_tokens")))
}

fn finish(value: Option<&Value>) -> Option<FinishReason> {
    let value = truthy(value)?;
    Some(match py_str(value).as_str() {
        "COMPLETE" | "STOP_SEQUENCE" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "TOOL_CALL" => FinishReason::ToolCalls,
        "ERROR" | "TIMEOUT" => FinishReason::Error,
        _ => FinishReason::Other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ResponseFormat, Role};

    fn request() -> ChatRequest {
        ChatRequest::new("command-a", vec![ChatMessage::new(Role::System, "s"), ChatMessage::new(Role::User, "q")])
    }

    #[test]
    fn chat_call_options() {
        let request = request()
            .with_temperature(0.2)
            .with_response_format(ResponseFormat::with_schema(json!({"type": "object"})))
            .with_thinking(false);
        let call = CohereAdapter::default().chat_call(&request, "k", None, true).unwrap();
        assert_eq!(call.url, "https://api.cohere.com/v2/chat");
        assert_eq!(call.header("Accept"), Some("application/json"));
        assert_eq!(
            call.json.unwrap(),
            json!({"model": "command-a", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                   "stream": true, "temperature": 0.2,
                   "response_format": {"type": "json_object", "json_schema": {"type": "object"}},
                   "thinking": {"type": "disabled"}})
        );
    }

    #[test]
    fn parse_and_stream() {
        let adapter = CohereAdapter::default();
        let reply = adapter
            .parse_chat(
                &json!({"message": {"content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "hi"}]},
                        "finish_reason": "MAX_TOKENS", "usage": {"tokens": {"input_tokens": 3, "output_tokens": 1}}}),
                &request(),
            )
            .unwrap();
        assert_eq!((reply.text.as_str(), reply.finish_reason), ("hi", Some(FinishReason::Length)));
        assert_eq!(reply.usage, Usage::new(Some(3), Some(1)));
        assert!(adapter.parse_chat(&json!({"message": "invalid api token"}), &request()).is_err());

        let mut parser = adapter.stream_parser(&request());
        let events = parser
            .feed(None, r#"{"type": "content-delta", "delta": {"message": {"content": {"text": "he"}}}}"#)
            .unwrap();
        assert_eq!(events, vec![StreamEvent::delta("he")]);
        parser
            .feed(None, r#"{"type": "message-end", "delta": {"finish_reason": "COMPLETE", "usage": {"billed_units": {"input_tokens": 2, "output_tokens": 5}}}}"#)
            .unwrap();
        assert_eq!(
            parser.finish(),
            StreamEvent::Done {
                finish_reason: Some(FinishReason::Stop),
                model: Some("command-a".into()),
                usage: Usage::new(Some(2), Some(5))
            }
        );
    }
}
