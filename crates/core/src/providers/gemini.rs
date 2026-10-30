//! Google Gemini API (Generative Language API, API-key auth).

use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, headers, obj_field, str_field, stream_error, truthy, HttpCall, ProviderAdapter, StreamParser,
};
use crate::error::Result;
use crate::types::{
    merge_consecutive, ChatMessage, ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, Role,
    StreamEvent, Usage,
};

/// Python's `quote(s, safe="-._")`: unreserved characters stay, everything else is encoded.
const MODEL_PATH: &AsciiSet = &NON_ALPHANUMERIC.remove(b'-').remove(b'.').remove(b'_').remove(b'~');

#[derive(Debug, Clone)]
pub struct GeminiAdapter {
    info: ProviderInfo,
}

impl Default for GeminiAdapter {
    fn default() -> Self {
        Self {
            info: ProviderInfo::new(
                "gemini",
                "Google Gemini",
                "https://generativelanguage.googleapis.com/v1beta",
                "https://aistudio.google.com/apikey",
                "AIza...",
            ),
        }
    }
}

impl GeminiAdapter {
    fn auth_headers(&self, api_key: &str) -> Vec<(String, String)> {
        headers(&[("x-goog-api-key", api_key), ("Content-Type", "application/json")])
    }
}

impl ProviderAdapter for GeminiAdapter {
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
        let mut turns = merge_consecutive(turns);
        if turns.first().is_some_and(|t| t.role == Role::Assistant) {
            // Gemini conversations must open with a user turn
            turns.insert(0, ChatMessage::new(Role::User, "Continue."));
        }
        let contents: Vec<Value> = turns
            .iter()
            .map(|m| {
                let role = if m.role == Role::Assistant { "model" } else { "user" };
                json!({"role": role, "parts": [{"text": m.content}]})
            })
            .collect();
        let mut body = Map::new();
        body.insert("contents".into(), Value::Array(contents));
        if let Some(system) = system {
            body.insert("systemInstruction".into(), json!({"parts": [{"text": system}]}));
        }
        let mut config = Map::new();
        if let Some(temperature) = request.temperature {
            config.insert("temperature".into(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            config.insert("maxOutputTokens".into(), json!(max_tokens));
        }
        if let Some(format) = &request.response_format {
            config.insert("responseMimeType".into(), json!("application/json"));
            if format.has_schema() {
                config.insert("responseSchema".into(), to_gemini_schema(&format.json_schema()));
            }
        }
        if !config.is_empty() {
            body.insert("generationConfig".into(), Value::Object(config));
        }
        let model_id = request.model.strip_prefix("models/").unwrap_or(&request.model);
        let model = utf8_percent_encode(model_id, MODEL_PATH);
        let method = if stream { "streamGenerateContent" } else { "generateContent" };
        let url = format!("{}/models/{model}:{method}", self.base_url(base_url)?);
        let call = HttpCall::post(url, self.auth_headers(api_key), Value::Object(body));
        Ok(if stream { call.with_params(&[("alt", "sse")]) } else { call })
    }

    fn parse_chat(&self, payload: &Value, request: &ChatRequest) -> Result<ChatResponse> {
        if payload.get("error").is_some_and(Value::is_object) {
            return Err(stream_error(self.id(), payload, None));
        }
        let (text, finish_reason) = candidate(payload);
        Ok(ChatResponse {
            text,
            model: str_field(payload, "modelVersion").unwrap_or(&request.model).to_owned(),
            finish_reason,
            usage: usage(payload).unwrap_or_default(),
        })
    }

    fn stream_parser(&self, request: &ChatRequest) -> Box<dyn StreamParser> {
        Box::new(GeminiStreamParser {
            provider_id: self.info.id.clone(),
            model: request.model.clone(),
            finish_reason: None,
            usage: Usage::default(),
        })
    }

    fn models_call(&self, api_key: &str, base_url: Option<&str>) -> Result<HttpCall> {
        Ok(HttpCall::get(format!("{}/models", self.base_url(base_url)?), self.auth_headers(api_key))
            .with_params(&[("pageSize", "1000")]))
    }

    fn parse_models(&self, payload: &Value) -> Vec<ModelInfo> {
        let mut models: Vec<ModelInfo> = arr_field(payload, "models")
            .iter()
            .filter(|item| {
                arr_field(item, "supportedGenerationMethods").iter().any(|m| m.as_str() == Some("generateContent"))
            })
            .filter_map(|item| {
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let id = name.strip_prefix("models/").unwrap_or(name);
                (!id.is_empty()).then(|| {
                    ModelInfo::new(
                        id,
                        str_field(item, "displayName").map(str::to_owned),
                        as_count(item.get("inputTokenLimit")),
                    )
                })
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    fn next_models_page(&self, payload: &Value, call: &HttpCall) -> Option<HttpCall> {
        let token = str_field(payload, "nextPageToken")?;
        let mut next = call.clone();
        next.set_param("pageToken", token);
        Some(next)
    }
}

struct GeminiStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl StreamParser for GeminiStreamParser {
    fn feed(&mut self, _event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        let Ok(payload) = serde_json::from_str::<Value>(data) else {
            return Ok(Vec::new());
        };
        if !payload.is_object() {
            return Ok(Vec::new());
        }
        if payload.get("error").is_some_and(Value::is_object) {
            return Err(stream_error(&self.provider_id, &payload, None));
        }
        if let Some(model) = str_field(&payload, "modelVersion") {
            self.model = model.to_owned();
        }
        if let Some(usage) = usage(&payload) {
            self.usage = usage;
        }
        let (text, finish_reason) = candidate(&payload);
        if finish_reason.is_some() {
            self.finish_reason = finish_reason;
        }
        Ok(if text.is_empty() { Vec::new() } else { vec![StreamEvent::delta(text)] })
    }

    fn finish(&mut self) -> StreamEvent {
        StreamEvent::Done { finish_reason: self.finish_reason, model: Some(self.model.clone()), usage: self.usage }
    }
}

fn candidate(payload: &Value) -> (String, Option<FinishReason>) {
    let candidates = arr_field(payload, "candidates");
    let Some(first) = candidates.first() else {
        let blocked = truthy(obj_field(payload, "promptFeedback").get("blockReason")).is_some();
        return (String::new(), blocked.then_some(FinishReason::ContentFilter));
    };
    let text: String = first
        .get("content")
        .map(|content| arr_field(content, "parts"))
        .unwrap_or(&[])
        .iter()
        .filter(|part| part.is_object() && truthy(part.get("thought")).is_none())
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect();
    let finish = match first.get("finishReason").and_then(Value::as_str) {
        None | Some("") | Some("FINISH_REASON_UNSPECIFIED") => None,
        Some("STOP") => Some(FinishReason::Stop),
        Some("MAX_TOKENS") => Some(FinishReason::Length),
        Some("SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII") => {
            Some(FinishReason::ContentFilter)
        }
        Some("MALFORMED_FUNCTION_CALL") => Some(FinishReason::Error),
        Some(_) => Some(FinishReason::Other),
    };
    (text, finish)
}

fn usage(payload: &Value) -> Option<Usage> {
    let meta = payload.get("usageMetadata")?.as_object()?;
    let mut output = as_count(meta.get("candidatesTokenCount"));
    if let (Some(out), Some(thoughts)) = (output, as_count(truthy(meta.get("thoughtsTokenCount")))) {
        output = Some(out + thoughts);
    }
    Some(Usage::new(as_count(meta.get("promptTokenCount")), output))
}

/// JSON Schema to Gemini's `responseSchema` (an OpenAPI subset with upper-case types).
///
/// Keeps type, description, enum, required, nullable, properties and items; drops the rest.
pub fn to_gemini_schema(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else { return json!({"type": "STRING"}) };
    let mut nullable = false;
    let kind = match object.get("type") {
        None => "object".to_owned(),
        Some(Value::Array(kinds)) => {
            nullable = kinds.iter().any(|k| k.as_str() == Some("null"));
            kinds.iter().find(|k| k.as_str() != Some("null")).map(py_scalar_str).unwrap_or_else(|| "string".into())
        }
        Some(other) => py_scalar_str(other),
    };
    let kind = match kind.to_lowercase().as_str() {
        "object" => "OBJECT",
        "array" => "ARRAY",
        "boolean" => "BOOLEAN",
        "number" => "NUMBER",
        "integer" => "INTEGER",
        _ => "STRING",
    };
    let mut out = Map::new();
    out.insert("type".into(), json!(kind));
    if nullable || truthy(object.get("nullable")).is_some() {
        out.insert("nullable".into(), json!(true));
    }
    if let Some(description) = object.get("description").filter(|d| d.is_string()) {
        out.insert("description".into(), description.clone());
    }
    if let Some(Value::Array(values)) = object.get("enum") {
        out.insert("enum".into(), Value::Array(values.iter().map(|v| json!(py_scalar_str(v))).collect()));
    }
    if kind == "OBJECT" {
        let empty = Map::new();
        let properties = truthy(object.get("properties")).and_then(Value::as_object).unwrap_or(&empty);
        let converted: Map<String, Value> =
            properties.iter().map(|(key, value)| (key.clone(), to_gemini_schema(value))).collect();
        out.insert("properties".into(), Value::Object(converted));
        let required: Vec<Value> = truthy(object.get("required"))
            .and_then(Value::as_array)
            .map(|keys| {
                keys.iter().filter(|k| k.as_str().is_some_and(|k| properties.contains_key(k))).cloned().collect()
            })
            .unwrap_or_default();
        if !required.is_empty() {
            out.insert("required".into(), Value::Array(required));
        }
    } else if kind == "ARRAY" {
        let default = json!({"type": "string"});
        out.insert("items".into(), to_gemini_schema(truthy(object.get("items")).unwrap_or(&default)));
    }
    Value::Object(out)
}

/// Python's `str()` of a JSON scalar (`True`, `None`, `1.5`).
fn py_scalar_str(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ResponseFormat;

    fn request() -> ChatRequest {
        ChatRequest::new(
            "models/gemini x",
            vec![ChatMessage::new(Role::User, "a"), ChatMessage::new(Role::Assistant, "b")],
        )
        .with_max_tokens(5)
    }

    #[test]
    fn chat_call_shape() {
        let call = GeminiAdapter::default().chat_call(&request(), "AIza", None, true).unwrap();
        assert_eq!(
            call.url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini%20x:streamGenerateContent"
        );
        assert_eq!(call.params, vec![("alt".to_owned(), "sse".to_owned())]);
        assert_eq!(
            call.json.unwrap(),
            json!({"contents": [{"role": "user", "parts": [{"text": "a"}]}, {"role": "model", "parts": [{"text": "b"}]}],
                   "generationConfig": {"maxOutputTokens": 5}})
        );
    }

    #[test]
    fn parse_skips_thoughts_and_counts_them_as_output() {
        let reply = GeminiAdapter::default()
            .parse_chat(
                &json!({"candidates": [{"content": {"parts": [{"text": "hmm", "thought": true}, {"text": "ok"}]}, "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "thoughtsTokenCount": 3},
                        "modelVersion": "gemini-x-001"}),
                &request(),
            )
            .unwrap();
        assert_eq!(reply.text, "ok");
        assert_eq!(reply.model, "gemini-x-001");
        assert_eq!(reply.finish_reason, Some(FinishReason::Stop));
        assert_eq!(reply.usage, Usage::new(Some(4), Some(5)));
        let blocked = GeminiAdapter::default()
            .parse_chat(&json!({"promptFeedback": {"blockReason": "SAFETY"}}), &request())
            .unwrap();
        assert_eq!(blocked.finish_reason, Some(FinishReason::ContentFilter));
    }

    #[test]
    fn leading_user_turn_and_structured_output() {
        let schema = json!({"type": "object", "properties": {
            "name": {"type": ["string", "null"], "description": "who"},
            "tags": {"type": "array"}, "level": {"enum": [1, true, null, "x"], "type": "integer"}},
            "required": ["name", "missing"], "additionalProperties": false});
        let request = request().with_response_format(ResponseFormat::with_schema(schema));
        let mut request = request;
        request.messages.reverse();
        let body = GeminiAdapter::default().chat_call(&request, "k", None, false).unwrap().json.unwrap();
        assert_eq!(body["contents"][0], json!({"role": "user", "parts": [{"text": "Continue."}]}));
        assert_eq!(
            body["generationConfig"],
            json!({"maxOutputTokens": 5, "responseMimeType": "application/json", "responseSchema": {
                "type": "OBJECT", "properties": {
                    "name": {"type": "STRING", "nullable": true, "description": "who"},
                    "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "level": {"type": "INTEGER", "enum": ["1", "True", "None", "x"]}},
                "required": ["name"]}})
        );
        let plain = request.clone().with_response_format(ResponseFormat::default());
        let body = GeminiAdapter::default().chat_call(&plain, "k", None, false).unwrap().json.unwrap();
        assert!(body["generationConfig"].get("responseSchema").is_none());
    }

    #[test]
    fn models_need_generate_content() {
        let models = GeminiAdapter::default().parse_models(&json!({"models": [
            {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-b", "displayName": "B", "inputTokenLimit": 100, "supportedGenerationMethods": ["generateContent"]}
        ]}));
        assert_eq!(models, vec![ModelInfo::new("gemini-b", Some("B".into()), Some(100))]);
    }
}
