//! Alibaba Cloud Model Studio (Qwen) through the native DashScope API.
//!
//! `POST {base}/services/aigc/text-generation/generation` with `input.messages` and
//! `parameters`; streaming is SSE enabled by the `X-DashScope-SSE` header, with
//! `incremental_output` so each event carries only new text. The default base is the US
//! (Virginia) endpoint; set a base URL for another region, e.g.
//! `https://dashscope-intl.aliyuncs.com/api/v1`.

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, obj_field, py_json_dumps, py_str, stream_error, truthy, HttpCall, ProviderAdapter,
    StreamParser,
};
use crate::error::Result;
use crate::providers::{bearer_headers, flat_messages};
use crate::types::{ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage};

#[derive(Debug, Clone)]
pub struct DashScopeAdapter {
    info: ProviderInfo,
}

impl Default for DashScopeAdapter {
    fn default() -> Self {
        Self {
            info: ProviderInfo::new(
                "alibaba",
                "Alibaba Cloud (Qwen)",
                "https://dashscope-us.aliyuncs.com/api/v1",
                "https://modelstudio.console.alibabacloud.com/?tab=playground#/api-key",
                "sk-...",
            ),
        }
    }
}

impl ProviderAdapter for DashScopeAdapter {
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
        let mut parameters = Map::new();
        parameters.insert("result_format".into(), json!("message"));
        if let Some(temperature) = request.temperature {
            parameters.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            parameters.insert("max_tokens".into(), json!(max_tokens));
        }
        if stream {
            parameters.insert("incremental_output".into(), json!(true));
        }
        if request.response_format.is_some() {
            parameters.insert("response_format".into(), json!({"type": "json_object"}));
        }
        if let Some(thinking) = request.thinking {
            parameters.insert("enable_thinking".into(), json!(thinking));
        }
        let body =
            json!({"model": request.model, "input": {"messages": flat_messages(request)}, "parameters": parameters});
        let mut headers = bearer_headers(api_key);
        if stream {
            headers.push(("X-DashScope-SSE".into(), "enable".into()));
        }
        let url = format!("{}/services/aigc/text-generation/generation", self.base_url(base_url)?);
        Ok(HttpCall::post(url, headers, body))
    }

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse> {
        if is_error(payload) {
            return Err(stream_error(self.id(), payload, None));
        }
        let (text, finish_reason) = choice(payload);
        Ok(ChatResponse { text, model: request.model.clone(), finish_reason, usage: usage(payload) })
    }

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser> {
        Box::new(DashScopeStreamParser {
            provider_id: self.info.id.clone(),
            model: request.model.clone(),
            finish_reason: None,
            usage: Usage::default(),
        })
    }

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall> {
        // The native API has no model list; the same host serves one under compatible-mode.
        let base = self.base_url(base_url)?;
        let root = base.strip_suffix("/api/v1").unwrap_or(&base);
        Ok(HttpCall::get(format!("{root}/compatible-mode/v1/models"), bearer_headers(api_key)))
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

fn is_error(payload: &Value) -> bool {
    truthy(payload.get("code")).is_some() && truthy(payload.get("output")).is_none()
}

struct DashScopeStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl StreamParser for DashScopeStreamParser {
    fn feed(&mut self, event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        let Ok(payload) = serde_json::from_str::<Value>(data) else { return Ok(Vec::new()) };
        if !payload.is_object() {
            return Ok(Vec::new());
        }
        if event == Some("error") || is_error(&payload) {
            return Err(stream_error(&self.provider_id, &payload, None));
        }
        let latest = usage(&payload);
        if latest.input_tokens.is_some() || latest.output_tokens.is_some() {
            self.usage = latest;
        }
        let (text, finish_reason) = choice(&payload);
        if finish_reason.is_some() {
            self.finish_reason = finish_reason;
        }
        Ok(if text.is_empty() { Vec::new() } else { vec![StreamEvent::delta(text)] })
    }

    fn finish(&mut self) -> StreamEvent {
        StreamEvent::Done { finish_reason: self.finish_reason, model: Some(self.model.clone()), usage: self.usage }
    }
}

fn choice(payload: &Value) -> (String, Option<FinishReason>) {
    let output = obj_field(payload, "output");
    let choices = output.get("choices").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    let (text, reason) = if let Some(first) = choices.first() {
        let message = first.get("message").filter(|m| m.is_object());
        let mut text = match message.and_then(|m| m.get("content")) {
            Some(Value::Array(parts)) => {
                parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<String>()
            }
            Some(Value::String(content)) => content.clone(),
            _ => String::new(),
        };
        if text.is_empty() {
            let calls = message.map(|m| arr_field(m, "tool_calls")).unwrap_or(&[]);
            for call in calls.iter().filter(|c| c.is_object()) {
                let arguments = call.get("function").filter(|f| f.is_object()).and_then(|f| truthy(f.get("arguments")));
                if let Some(arguments) = arguments {
                    text = match arguments {
                        Value::String(s) => s.clone(),
                        other => py_json_dumps(other),
                    };
                    break;
                }
            }
        }
        (text, first.get("finish_reason"))
    } else {
        // result_format "text"
        (output.get("text").and_then(Value::as_str).unwrap_or("").to_owned(), output.get("finish_reason"))
    };
    let finish = truthy(reason).map(py_str).filter(|r| r != "null").map(|r| match r.as_str() {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    });
    (text, finish)
}

fn usage(payload: &Value) -> Usage {
    match payload.get("usage") {
        Some(usage @ Value::Object(_)) => {
            Usage::new(as_count(usage.get("input_tokens")), as_count(usage.get("output_tokens")))
        }
        _ => Usage::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ResponseFormat, Role};

    fn request() -> ChatRequest {
        ChatRequest::new("qwen-plus", vec![ChatMessage::new(Role::User, "q")]).with_system("s")
    }

    #[test]
    fn chat_call_shape() {
        let request = request().with_response_format(ResponseFormat::default()).with_thinking(true).with_max_tokens(9);
        let call = DashScopeAdapter::default().chat_call(&request, "k", None, true).unwrap();
        assert_eq!(call.url, "https://dashscope-us.aliyuncs.com/api/v1/services/aigc/text-generation/generation");
        assert_eq!(call.header("X-DashScope-SSE"), Some("enable"));
        assert_eq!(
            call.json.unwrap(),
            json!({"model": "qwen-plus", "input": {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]},
                   "parameters": {"result_format": "message", "max_tokens": 9, "incremental_output": true,
                                  "response_format": {"type": "json_object"}, "enable_thinking": true}})
        );
        let models =
            DashScopeAdapter::default().models_call("k", Some("https://dashscope-intl.aliyuncs.com/api/v1/")).unwrap();
        assert_eq!(models.url, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models");
    }

    #[test]
    fn parse_tool_arguments_errors_and_stream() {
        let adapter = DashScopeAdapter::default();
        let reply = adapter
            .parse_chat(
                &json!({"output": {"choices": [{"message": {"content": "", "tool_calls": [{"function": {"arguments": "{\"a\": 1}"}}]},
                                                "finish_reason": "tool_calls"}]},
                        "usage": {"input_tokens": 4, "output_tokens": 2}}),
                &request(),
            )
            .unwrap();
        assert_eq!((reply.text.as_str(), reply.finish_reason), ("{\"a\": 1}", Some(FinishReason::ToolCalls)));
        assert!(adapter.parse_chat(&json!({"code": "InvalidApiKey", "message": "bad"}), &request()).is_err());

        let mut parser = adapter.stream_parser(&request());
        let first = parser
            .feed(None, r#"{"output": {"choices": [{"message": {"content": "he"}, "finish_reason": "null"}]}}"#)
            .unwrap();
        assert_eq!(first, vec![StreamEvent::delta("he")]);
        parser
            .feed(None, r#"{"output": {"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}]}, "usage": {"input_tokens": 1, "output_tokens": 2}}"#)
            .unwrap();
        assert_eq!(
            parser.finish(),
            StreamEvent::Done {
                finish_reason: Some(FinishReason::Stop),
                model: Some("qwen-plus".into()),
                usage: Usage::new(Some(1), Some(2))
            }
        );
        assert!(parser.feed(Some("error"), r#"{"code": "x", "message": "m"}"#).is_err());
    }
}
