//! The Responses protocol (`POST /responses`), used natively by Meta's Model API.
//!
//! Request: `model`, `input` (role/content items; system included), `temperature`,
//! `max_output_tokens`, flat function `tools`. Streaming events: `response.output_text.delta`,
//! `response.function_call_arguments.delta`, then `response.completed` with usage.

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, py_json_dumps, py_str, str_field, stream_error, truthy, HttpCall, ProviderAdapter,
    StreamParser,
};
use crate::error::Result;
use crate::providers::{bearer_headers, flat_messages};
use crate::types::{ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage};

#[derive(Debug, Clone)]
pub struct ResponsesAdapter {
    info: ProviderInfo,
}

impl ResponsesAdapter {
    pub fn new(info: ProviderInfo) -> Self {
        Self { info }
    }
}

/// Meta's Model API (Muse models).
pub fn meta() -> ResponsesAdapter {
    ResponsesAdapter::new(ProviderInfo::new("meta", "Meta (Muse)", "https://api.meta.ai/v1", "https://dev.meta.ai", ""))
}

impl ProviderAdapter for ResponsesAdapter {
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
        body.insert("input".into(), Value::Array(flat_messages(request)));
        body.insert("stream".into(), json!(stream));
        if let Some(temperature) = request.temperature {
            body.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            body.insert("max_output_tokens".into(), json!(max_tokens));
        }
        if let Some(format) = &request.response_format {
            body.insert(
                "tools".into(),
                json!([{"type": "function", "name": format.name, "description": format.description,
                        "parameters": format.json_schema()}]),
            );
            body.insert("tool_choice".into(), json!({"type": "function", "name": format.name}));
        }
        if let Some(effort) = request.reasoning_effort.as_deref().filter(|e| !e.is_empty()) {
            if effort != "lowest" && effort != "none" {
                body.insert("reasoning".into(), json!({"effort": effort}));
            }
        }
        let url = format!("{}/responses", self.base_url(base_url)?);
        Ok(HttpCall::post(url, bearer_headers(api_key), Value::Object(body)))
    }

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse> {
        if payload.get("error").is_some_and(Value::is_object) {
            return Err(stream_error(self.id(), payload, None));
        }
        Ok(ChatResponse {
            text: output_text(payload),
            model: str_field(payload, "model").unwrap_or(&request.model).to_owned(),
            finish_reason: finish(payload),
            usage: usage(payload.get("usage")),
        })
    }

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser> {
        Box::new(ResponsesStreamParser {
            provider_id: self.info.id.clone(),
            model: request.model.clone(),
            finish_reason: None,
            usage: Usage::default(),
            saw_text: false,
        })
    }

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall> {
        Ok(HttpCall::get(format!("{}/models", self.base_url(base_url)?), bearer_headers(api_key)))
    }

    fn parse_models(&self, payload: &Value) -> Vec<ModelInfo> {
        let mut models: Vec<ModelInfo> = arr_field(payload, "data")
            .iter()
            .filter(|item| item.is_object())
            .filter_map(|item| truthy(item.get("id")).map(|id| ModelInfo::new(py_str(id), None, None)))
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }
}

fn is_type(item: &Value, kind: &str) -> bool {
    item.get("type").and_then(Value::as_str) == Some(kind)
}

fn output_text(response: &Value) -> String {
    let mut text = String::new();
    let mut arguments: Option<&Value> = None;
    let mut seen_call = false;
    for item in arr_field(response, "output").iter().filter(|i| i.is_object()) {
        if is_type(item, "message") {
            for part in arr_field(item, "content") {
                if is_type(part, "output_text") || is_type(part, "text") {
                    if let Some(chunk) = part.get("text").and_then(Value::as_str) {
                        text.push_str(chunk);
                    }
                }
            }
        } else if is_type(item, "function_call") && !seen_call {
            // Python keeps the first call whose `arguments` is not None
            if let Some(value) = item.get("arguments").filter(|v| !v.is_null()) {
                arguments = Some(value);
                seen_call = true;
            }
        }
    }
    if !text.is_empty() {
        return text;
    }
    if let Some(arguments) = truthy(arguments) {
        return match arguments {
            Value::String(s) => s.clone(),
            other => py_json_dumps(other),
        };
    }
    response.get("output_text").and_then(Value::as_str).unwrap_or("").to_owned()
}

fn finish(response: &Value) -> Option<FinishReason> {
    if arr_field(response, "output").iter().any(|i| i.is_object() && is_type(i, "function_call")) {
        return Some(FinishReason::ToolCalls);
    }
    let status = truthy(response.get("status"))?;
    match py_str(status).as_str() {
        "incomplete" => {
            let reason = response.get("incomplete_details").and_then(|d| d.get("reason")).and_then(Value::as_str);
            Some(if reason == Some("content_filter") { FinishReason::ContentFilter } else { FinishReason::Length })
        }
        "completed" => Some(FinishReason::Stop),
        "failed" => Some(FinishReason::Error),
        _ => None,
    }
}

fn usage(value: Option<&Value>) -> Usage {
    let Some(usage) = value.filter(|v| v.is_object()) else { return Usage::default() };
    let cached = usage
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .filter(|c| *c > 0);
    Usage::new(as_count(usage.get("input_tokens")), as_count(usage.get("output_tokens"))).with_cache(cached, None)
}

struct ResponsesStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
    saw_text: bool,
}

impl StreamParser for ResponsesStreamParser {
    fn feed(&mut self, event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        let Ok(payload) = serde_json::from_str::<Value>(data) else { return Ok(Vec::new()) };
        if !payload.is_object() {
            return Ok(Vec::new());
        }
        let kind = truthy(payload.get("type")).and_then(Value::as_str).or(event).unwrap_or("");
        match kind {
            "error" => return Err(stream_error(&self.provider_id, &payload, None)),
            "response.failed" => {
                let body = payload.get("response").filter(|r| r.is_object()).unwrap_or(&payload);
                return Err(stream_error(&self.provider_id, body, None));
            }
            "response.output_text.delta" => {
                if let Some(delta) = truthy(payload.get("delta")) {
                    self.saw_text = true;
                    return Ok(vec![StreamEvent::delta(py_str(delta))]);
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(delta) = truthy(payload.get("delta")).filter(|_| !self.saw_text) {
                    return Ok(vec![StreamEvent::delta(py_str(delta))]);
                }
            }
            "response.completed" | "response.incomplete" => {
                let empty = json!({});
                let response = truthy(payload.get("response")).unwrap_or(&empty);
                if let Some(model) = str_field(response, "model") {
                    self.model = model.to_owned();
                }
                self.usage = usage(truthy(response.get("usage")).or(payload.get("usage")));
                self.finish_reason = finish(response).or(Some(if kind == "response.incomplete" {
                    FinishReason::Length
                } else {
                    FinishReason::Stop
                }));
            }
            _ => {}
        }
        Ok(Vec::new())
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
        ChatRequest::new("muse-1", vec![ChatMessage::new(Role::User, "q")]).with_system("s")
    }

    #[test]
    fn chat_call_shape() {
        let request =
            request().with_response_format(ResponseFormat::default()).with_reasoning_effort("high").with_max_tokens(7);
        let call = meta().chat_call(&request, "k", None, false).unwrap();
        assert_eq!(call.url, "https://api.meta.ai/v1/responses");
        assert_eq!(
            call.json.unwrap(),
            json!({"model": "muse-1", "input": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                   "stream": false, "max_output_tokens": 7,
                   "tools": [{"type": "function", "name": "Give_Information",
                              "description": "Generate output in given key and valid json format", "parameters": {"type": "object"}}],
                   "tool_choice": {"type": "function", "name": "Give_Information"}, "reasoning": {"effort": "high"}})
        );
        let lowest = request.clone().with_reasoning_effort("none");
        assert!(meta().chat_call(&lowest, "k", None, false).unwrap().json.unwrap().get("reasoning").is_none());
    }

    #[test]
    fn parse_output() {
        let reply = meta()
            .parse_chat(
                &json!({"model": "muse-1-0", "status": "completed",
                        "output": [{"type": "reasoning"}, {"type": "function_call", "arguments": {"a": "é"}}],
                        "usage": {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 6}}}),
                &request(),
            )
            .unwrap();
        assert_eq!(reply.text, "{\"a\": \"\\u00e9\"}");
        assert_eq!(reply.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(reply.usage, Usage::new(Some(10), Some(2)).with_cache(Some(6), None));
        let incomplete = meta()
            .parse_chat(&json!({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output_text": "x"}), &request())
            .unwrap();
        assert_eq!((incomplete.text.as_str(), incomplete.finish_reason), ("x", Some(FinishReason::Length)));
    }

    #[test]
    fn stream_prefers_text_over_arguments() {
        let mut parser = meta().stream_parser(&request());
        assert_eq!(
            parser.feed(None, r#"{"type": "response.output_text.delta", "delta": "hi"}"#).unwrap(),
            vec![StreamEvent::delta("hi")]
        );
        assert!(parser
            .feed(None, r#"{"type": "response.function_call_arguments.delta", "delta": "{"}"#)
            .unwrap()
            .is_empty());
        parser
            .feed(None, r#"{"type": "response.completed", "response": {"model": "m2", "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}}}"#)
            .unwrap();
        assert_eq!(
            parser.finish(),
            StreamEvent::Done {
                finish_reason: Some(FinishReason::Stop),
                model: Some("m2".into()),
                usage: Usage::new(Some(1), Some(1))
            }
        );
        let err = parser
            .feed(None, r#"{"type": "response.failed", "response": {"error": {"message": "boom"}}}"#)
            .unwrap_err();
        assert_eq!(err.message, "boom");
    }
}
