//! OpenAI Chat Completions, and the providers whose own API uses the same protocol.
//!
//! Each provider is a first-class adapter with its own id, URL and model rules (Kimi, NVIDIA,
//! Z.ai, MiniMax, Arcee and AI21 speak this protocol natively); `openai_compatible` is only for
//! arbitrary endpoints the caller points at.

use serde_json::{json, Map, Value};

use crate::adapter::{
    arr_field, as_count, obj_field, py_json_dumps, str_field, stream_error, truthy, HttpCall, ProviderAdapter,
    StreamParser,
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

/// How an adapter asks for JSON when `ChatRequest::response_format` is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structured {
    /// `response_format = {"type": "json_schema", ...}`
    JsonSchema,
    /// `response_format = {"type": "json_object"}` (the schema stays in the prompt)
    JsonObject,
    /// One function tool whose arguments are the JSON reply.
    Tool,
}

/// Model-specific request rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rules {
    None,
    /// gpt-5 / o-series: no temperature; `reasoning_effort` from the family table.
    OpenAi,
    /// kimi-k*: `max_completion_tokens`, no temperature; others: temperature at most 1.
    Kimi,
    /// `thinking: {"type": on | "disabled"}` (Z.ai GLM, MiniMax).
    ThinkingSwitch {
        on: &'static str,
    },
    /// Nemotron thinking switches.
    Nvidia,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleAdapter {
    info: ProviderInfo,
    max_tokens_field: &'static str,
    stream_usage_option: bool,
    extra_headers: Vec<(String, String)>,
    non_chat_markers: Vec<String>,
    structured: Option<Structured>,
    force_tool: bool,
    strip_think: bool,
    rules: Rules,
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
            structured: Some(Structured::JsonObject),
            force_tool: true,
            strip_think: false,
            rules: Rules::None,
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

    pub fn structured(mut self, mode: Option<Structured>) -> Self {
        self.structured = mode;
        self
    }

    /// For `Structured::Tool`: let the model choose (`"auto"`) instead of forcing the tool.
    pub fn optional_tool(mut self) -> Self {
        self.force_tool = false;
        self
    }

    pub fn strip_think(mut self) -> Self {
        self.strip_think = true;
        self
    }

    pub fn rules(mut self, rules: Rules) -> Self {
        self.rules = rules;
        self
    }

    fn auth_headers(&self, api_key: &str) -> Vec<(String, String)> {
        let mut out = vec![("Content-Type".to_owned(), "application/json".to_owned())];
        out.extend(self.extra_headers.iter().cloned());
        if !api_key.is_empty() {
            out.push(("Authorization".to_owned(), format!("Bearer {api_key}")));
        }
        out
    }

    fn apply_structured(&self, body: &mut Map<String, Value>, request: &ChatRequest) {
        let (Some(format), Some(mode)) = (&request.response_format, self.structured) else { return };
        match mode {
            Structured::JsonObject => {
                body.insert("response_format".into(), json!({"type": "json_object"}));
            }
            Structured::JsonSchema => {
                body.insert(
                    "response_format".into(),
                    json!({"type": "json_schema", "json_schema": {
                        "name": format.name, "description": format.description,
                        "schema": format.json_schema(), "strict": false}}),
                );
            }
            Structured::Tool => {
                body.insert(
                    "tools".into(),
                    json!([{"type": "function", "function": {
                        "name": format.name, "description": format.description, "parameters": format.json_schema()}}]),
                );
                let choice = if self.force_tool {
                    json!({"type": "function", "function": {"name": format.name}})
                } else {
                    json!("auto")
                };
                body.insert("tool_choice".into(), choice);
            }
        }
    }

    fn tune(&self, body: &mut Map<String, Value>, request: &ChatRequest) {
        match &self.rules {
            Rules::None => {}
            Rules::OpenAi => tune_openai(body, request),
            Rules::Kimi => tune_kimi(body, request),
            Rules::ThinkingSwitch { on } => {
                if let Some(thinking) = request.thinking {
                    body.insert("thinking".into(), json!({"type": if thinking { *on } else { "disabled" }}));
                } else if request.response_format.is_some() {
                    // Tool calls are not allowed while thinking.
                    body.insert("thinking".into(), json!({"type": "disabled"}));
                }
            }
            Rules::Nvidia => tune_nvidia(body, request),
        }
    }
}

// ------------------------------------------------------------------ model rules

/// Reasoning models: the efforts each family accepts, for `reasoning_effort = "lowest"`.
const OPENAI_EFFORTS: &[(&str, &[&str])] = &[
    ("gpt-5.6", &["none", "low", "medium", "high"]),
    ("gpt-5.4", &["none", "low", "medium", "high"]),
    ("gpt-5.2", &["none", "low", "medium", "high"]),
    ("gpt-5.1", &["none", "low", "medium", "high"]),
    ("gpt-5", &["minimal", "low", "medium", "high"]),
    ("o4-mini", &["low", "medium", "high"]),
    ("o3-mini", &["low", "medium", "high"]),
    ("o3", &["low", "medium", "high"]),
    ("o1-mini", &[]),
    ("o1", &["low", "medium", "high"]),
];
const LOWEST: &[&str] = &["none", "minimal", "low"];

fn effort_family(model: &str) -> Option<&'static [&'static str]> {
    let mut families: Vec<&(&str, &[&str])> = OPENAI_EFFORTS.iter().collect();
    families.sort_by_key(|(prefix, _)| std::cmp::Reverse(prefix.len()));
    families.into_iter().find(|(prefix, _)| model.starts_with(prefix)).map(|(_, allowed)| *allowed)
}

/// gpt-5 and the o-series take no temperature and think before answering.
pub fn is_openai_reasoning_model(model: &str) -> bool {
    let mut chars = model.chars();
    model.starts_with("gpt-5") || (chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit()))
}

fn tune_openai(body: &mut Map<String, Value>, request: &ChatRequest) {
    let model = request.model.as_str();
    if is_openai_reasoning_model(model) {
        body.shift_remove("temperature"); // only the default (1) is accepted
    }
    let (Some(effort), Some(allowed)) = (request.reasoning_effort.as_deref(), effort_family(model)) else { return };
    if allowed.is_empty() {
        return;
    }
    let effort = if effort == "lowest" { LOWEST.iter().copied().find(|e| allowed.contains(e)) } else { Some(effort) };
    if let Some(effort) = effort.filter(|e| allowed.contains(e)) {
        body.insert("reasoning_effort".into(), json!(effort));
    }
}

/// Moonshot: kimi-k* reasoning models take `max_completion_tokens` and no temperature.
fn tune_kimi(body: &mut Map<String, Value>, request: &ChatRequest) {
    if request.model.starts_with("kimi-k") {
        body.shift_remove("temperature");
        if let Some(max_tokens) = body.shift_remove("max_tokens") {
            body.insert("max_completion_tokens".into(), max_tokens);
        }
    } else if let Some(temperature) = body.get("temperature").and_then(Value::as_f64) {
        body.insert("temperature".into(), json!(temperature.min(1.0))); // Moonshot accepts 0..1
    }
    if let Some(thinking) = request.thinking {
        body.insert("thinking".into(), json!({"type": if thinking { "enabled" } else { "disabled" }}));
    }
    if let Some(effort) = request.reasoning_effort.as_deref().filter(|e| *e != "lowest") {
        body.insert("reasoning_effort".into(), json!(effort));
    }
}

const NVIDIA_PROMPT_SWITCHED: &[&str] = &["nvidia-nemotron-nano-9b-v2", "llama-3.3-nemotron-super-49b-v1.5"];

/// NVIDIA NIM: Nemotron models switch thinking per request.
fn tune_nvidia(body: &mut Map<String, Value>, request: &ChatRequest) {
    let thinking = match request.thinking {
        Some(thinking) => thinking,
        None if request.response_format.is_some() => false,
        None => return,
    };
    let model = request.model.as_str();
    if model.contains("nemotron-3-") {
        body.insert("chat_template_kwargs".into(), json!({"enable_thinking": thinking}));
    } else if NVIDIA_PROMPT_SWITCHED.iter().any(|name| model.contains(name)) {
        let switch = if thinking { "/think" } else { "/no_think" };
        let Some(Value::Array(messages)) = body.get_mut("messages") else { return };
        let first_is_system = messages.first().and_then(|m| m.get("role")).and_then(Value::as_str) == Some("system");
        if first_is_system {
            let content = messages[0].get("content").and_then(Value::as_str).unwrap_or("").to_owned();
            messages[0] = json!({"role": "system", "content": format!("{switch}\n{content}")});
        } else {
            messages.insert(0, json!({"role": "system", "content": switch}));
        }
    }
}

impl ProviderAdapter for OpenAiCompatibleAdapter {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn needs_custom_base_url(&self) -> bool {
        self.requires_base_url
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
        self.apply_structured(&mut body, request);
        self.tune(&mut body, request);
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
        let mut text = content_text(message.get("content"));
        if self.strip_think {
            text = strip_think(&text);
        }
        if text.is_empty() {
            text = tool_arguments(message);
        }
        Ok(ChatResponse {
            text,
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
            think: self.strip_think.then(ThinkFilter::default),
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
            let Some(id_value) = truthy(item.get("id")) else { continue };
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

// ------------------------------------------------------------------ parsing helpers

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Hides `<think>...</think>` spans in streamed text, even when tags are split across chunks.
#[derive(Debug, Default)]
pub struct ThinkFilter {
    buffer: String,
    inside: bool,
}

impl ThinkFilter {
    pub fn feed(&mut self, text: &str) -> String {
        self.buffer.push_str(text);
        let mut out = String::new();
        while !self.buffer.is_empty() {
            if self.inside {
                match self.buffer.find(THINK_CLOSE) {
                    None => {
                        // keep a possible partial closing tag (last 8 characters)
                        self.buffer = last_chars(&self.buffer, THINK_CLOSE.len()).to_owned();
                        break;
                    }
                    Some(end) => {
                        self.buffer = self.buffer[end + THINK_CLOSE.len()..].to_owned();
                        self.inside = false;
                    }
                }
            } else {
                match self.buffer.find(THINK_OPEN) {
                    None => {
                        let keep = partial_tag_suffix(&self.buffer, THINK_OPEN);
                        let cut = self.buffer.len() - keep;
                        out.push_str(&self.buffer[..cut]);
                        self.buffer = self.buffer[cut..].to_owned();
                        break;
                    }
                    Some(start) => {
                        out.push_str(&self.buffer[..start]);
                        self.buffer = self.buffer[start + THINK_OPEN.len()..].to_owned();
                        self.inside = true;
                    }
                }
            }
        }
        out
    }

    pub fn flush(&mut self) -> String {
        let tail = if self.inside { String::new() } else { std::mem::take(&mut self.buffer) };
        self.buffer.clear();
        tail
    }
}

fn last_chars(text: &str, count: usize) -> &str {
    match text.char_indices().rev().nth(count.saturating_sub(1)) {
        Some((index, _)) if count > 0 => &text[index..],
        _ => text,
    }
}

/// Length (bytes; the tag is ASCII) of the longest proper prefix of `tag` that `text` ends with.
fn partial_tag_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len().min(text.len() + 1)).rev().find(|&size| text.ends_with(&tag[..size])).unwrap_or(0)
}

/// Text after the last `</think>`; before an unclosed `<think>`; else unchanged.
pub fn strip_think(text: &str) -> String {
    if let Some(index) = text.rfind(THINK_CLOSE) {
        return text[index + THINK_CLOSE.len()..].trim_start().to_owned();
    }
    if let Some(index) = text.find(THINK_OPEN) {
        return text[..index].trim_end().to_owned();
    }
    text.to_owned()
}

fn arguments_text(arguments: &Value) -> String {
    match arguments {
        Value::String(text) => text.clone(),
        other => py_json_dumps(other),
    }
}

/// The JSON reply of a structured call: the first tool call's (or legacy function call's) arguments.
fn tool_arguments(message: &Value) -> String {
    for call in arr_field(message, "tool_calls") {
        if let Some(arguments) = call.get("function").filter(|f| f.is_object()).and_then(|f| truthy(f.get("arguments")))
        {
            return arguments_text(arguments);
        }
    }
    if let Some(arguments) =
        message.get("function_call").filter(|f| f.is_object()).and_then(|f| truthy(f.get("arguments")))
    {
        return arguments_text(arguments);
    }
    String::new()
}

struct OpenAiStreamParser {
    provider_id: String,
    model: String,
    finish_reason: Option<FinishReason>,
    usage: Usage,
    think: Option<ThinkFilter>,
}

impl StreamParser for OpenAiStreamParser {
    fn feed(&mut self, _event: Option<&str>, data: &str) -> Result<Vec<StreamEvent>> {
        if data.trim() == "[DONE]" {
            return Ok(Vec::new());
        }
        let Ok(payload) = serde_json::from_str::<Value>(data) else { return Ok(Vec::new()) };
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
            let empty = Value::Object(Map::new());
            let delta = truthy(choice.get("delta")).filter(|d| d.is_object()).unwrap_or(&empty);
            let mut text = content_text(delta.get("content"));
            if let Some(think) = self.think.as_mut() {
                if !text.is_empty() {
                    text = think.feed(&text);
                }
            }
            if text.is_empty() {
                // structured output arrives as tool-call argument fragments
                for call in arr_field(delta, "tool_calls") {
                    if let Some(fragment) = call
                        .get("function")
                        .filter(|f| f.is_object())
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        text.push_str(fragment);
                    }
                }
            }
            if !text.is_empty() {
                events.push(StreamEvent::delta(text));
            }
            if truthy(choice.get("finish_reason")).is_some() {
                self.finish_reason = finish(choice.get("finish_reason"));
            }
        }
        Ok(events)
    }

    fn flush_events(&mut self) -> Vec<StreamEvent> {
        match self.think.as_mut().map(ThinkFilter::flush) {
            Some(tail) if !tail.is_empty() => vec![StreamEvent::delta(tail)],
            _ => Vec::new(),
        }
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
            let cached = obj_field(usage, "prompt_tokens_details").get("cached_tokens").and_then(Value::as_u64);
            Usage::new(as_count(usage.get("prompt_tokens")), as_count(usage.get("completion_tokens")))
                .with_cache(cached.filter(|c| *c > 0), None)
        }
        _ => Usage::default(),
    }
}

fn info(id: &str, label: &str, base_url: &str, key_url: &str, hint: &str) -> ProviderInfo {
    ProviderInfo::new(id, label, base_url, key_url, hint)
}

/// Built-in providers that use this protocol, in registry order.
pub fn builtin() -> Vec<OpenAiCompatibleAdapter> {
    use Structured::{JsonSchema, Tool};
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
        .non_chat(OPENAI_NON_CHAT)
        .structured(Some(Tool))
        .optional_tool()
        .rules(Rules::OpenAi),
        OpenAiCompatibleAdapter::new(info(
            "xai",
            "xAI (Grok)",
            "https://api.x.ai/v1",
            "https://console.x.ai",
            "xai-...",
        ))
        .stream_usage()
        .non_chat(&["image", "imagine"])
        .structured(Some(Tool))
        .optional_tool(),
        OpenAiCompatibleAdapter::new(info(
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/keys",
            "sk-or-...",
        ))
        .stream_usage()
        .structured(None),
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
        .stream_usage()
        .structured(Some(Tool))
        .optional_tool(),
        OpenAiCompatibleAdapter::new(info(
            "mistral",
            "Mistral",
            "https://api.mistral.ai/v1",
            "https://console.mistral.ai/api-keys",
            "",
        ))
        .non_chat(&["embed", "moderation", "ocr"])
        .structured(Some(Tool))
        .optional_tool(),
        OpenAiCompatibleAdapter::new(info(
            "together",
            "Together AI",
            "https://api.together.xyz/v1",
            "https://api.together.ai/settings/api-keys",
            "",
        )),
        OpenAiCompatibleAdapter::new(info(
            "kimi",
            "Kimi (Moonshot)",
            "https://api.moonshot.ai/v1",
            "https://platform.moonshot.ai/console/api-keys",
            "sk-...",
        ))
        .stream_usage()
        .structured(Some(JsonSchema))
        .rules(Rules::Kimi),
        OpenAiCompatibleAdapter::new(info(
            "nvidia",
            "NVIDIA NIM",
            "https://integrate.api.nvidia.com/v1",
            "https://build.nvidia.com/settings/api-keys",
            "nvapi-...",
        ))
        .stream_usage()
        .structured(Some(Tool))
        .strip_think()
        .rules(Rules::Nvidia),
        OpenAiCompatibleAdapter::new(info(
            "zai",
            "Z.ai (GLM)",
            "https://api.z.ai/api/paas/v4",
            "https://z.ai/manage-apikey/apikey-list",
            "",
        ))
        .stream_usage()
        .structured(Some(Tool))
        .rules(Rules::ThinkingSwitch { on: "enabled" }),
        OpenAiCompatibleAdapter::new(info(
            "minimax",
            "MiniMax",
            "https://api.minimax.io/v1",
            "https://platform.minimax.io/user-center/basic-information/interface-key",
            "",
        ))
        .stream_usage()
        .structured(Some(Tool))
        .rules(Rules::ThinkingSwitch { on: "adaptive" }),
        OpenAiCompatibleAdapter::new(info(
            "arcee",
            "Arcee AI",
            "https://api.arcee.ai/api/v1",
            "https://chat.arcee.ai",
            "",
        ))
        .stream_usage()
        .structured(Some(Tool)),
        OpenAiCompatibleAdapter::new(info(
            "ai21",
            "AI21 (Jamba)",
            "https://api.ai21.com/studio/v1",
            "https://studio.ai21.com/account/api-key",
            "",
        )),
    ]
}

/// Any OpenAI-compatible endpoint (vLLM, LM Studio, gateways). Needs `base_url`.
pub fn custom() -> OpenAiCompatibleAdapter {
    let mut adapter =
        OpenAiCompatibleAdapter::new(info("openai_compatible", "OpenAI-compatible (custom URL)", "", "", ""))
            .structured(None);
    adapter.requires_base_url = true;
    adapter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ResponseFormat, Role};

    fn by_id(id: &str) -> OpenAiCompatibleAdapter {
        builtin().into_iter().find(|a| a.info.id == id).unwrap()
    }

    fn request() -> ChatRequest {
        ChatRequest::new("gpt-x", vec![ChatMessage::new(Role::System, "be brief"), ChatMessage::new(Role::User, "Hi")])
            .with_max_tokens(10)
            .with_temperature(0.5)
    }

    fn body(adapter: &OpenAiCompatibleAdapter, request: &ChatRequest) -> Value {
        adapter.chat_call(request, "sk-key", None, false).unwrap().json.unwrap()
    }

    #[test]
    fn chat_call_shape() {
        let call = by_id("openai").chat_call(&request(), "sk-key", None, true).unwrap();
        assert_eq!(call.url, "https://api.openai.com/v1/chat/completions");
        assert_eq!(call.header("authorization"), Some("Bearer sk-key"));
        assert_eq!(
            call.json.unwrap(),
            json!({"model": "gpt-x", "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "Hi"}],
                   "stream": true, "temperature": 0.5, "max_completion_tokens": 10, "stream_options": {"include_usage": true}})
        );
    }

    #[test]
    fn openai_reasoning_models() {
        let cases = [
            ("gpt-5.1-mini", Some("none")),
            ("gpt-5", Some("minimal")),
            ("o3-mini", Some("low")),
            ("o1-mini", None),
            ("gpt-4o", None),
        ];
        for (model, effort) in cases {
            let mut req = request().with_reasoning_effort("lowest");
            req.model = model.into();
            let b = body(&by_id("openai"), &req);
            assert_eq!(b.get("reasoning_effort").and_then(Value::as_str), effort, "{model}");
            assert_eq!(b.get("temperature").is_some(), model == "gpt-4o", "{model}");
        }
    }

    #[test]
    fn structured_modes() {
        let schema = json!({"type": "object", "properties": {"n": {"type": "string"}}});
        let req = request().with_response_format(ResponseFormat::with_schema(schema.clone()));
        let b = body(&by_id("openai"), &req);
        assert_eq!(b["tools"][0]["function"]["parameters"], schema);
        assert_eq!(b["tool_choice"], "auto");
        let b = body(&by_id("nvidia"), &req);
        assert_eq!(b["tool_choice"], json!({"type": "function", "function": {"name": "Give_Information"}}));
        let mut kimi_req = req.clone();
        kimi_req.model = "kimi-k3".into();
        let b = body(&by_id("kimi"), &kimi_req);
        assert_eq!(b["response_format"]["json_schema"]["schema"], schema);
        assert!(b.get("temperature").is_none() && b["max_completion_tokens"] == 10);
        assert_eq!(body(&by_id("ai21"), &req)["response_format"], json!({"type": "json_object"}));
        assert!(body(&by_id("openrouter"), &req).get("response_format").is_none());
        let b = body(&by_id("zai"), &req);
        assert_eq!(b["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn tool_arguments_are_the_reply() {
        let reply = by_id("openai")
            .parse_chat(
                &json!({"choices": [{"message": {"content": null, "tool_calls": [{"function": {"arguments": "{\"a\":1}"}}]},
                                     "finish_reason": "tool_calls"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 80}}}),
                &request(),
            )
            .unwrap();
        assert_eq!(reply.text, "{\"a\":1}");
        assert_eq!(reply.usage, Usage::new(Some(100), Some(5)).with_cache(Some(80), None));
    }

    #[test]
    fn think_filter_and_strip() {
        let mut filter = ThinkFilter::default();
        let out: String =
            ["Hi <thi", "nk>secret ", "stuff</th", "ink> there", " <", "b>"].iter().map(|c| filter.feed(c)).collect();
        assert_eq!(out + &filter.flush(), "Hi  there <b>");
        assert_eq!(strip_think("<think>x</think>\nanswer"), "answer");
        assert_eq!(strip_think("answer <think>unfinished"), "answer");
    }

    #[test]
    fn nvidia_prompt_switch() {
        let mut req = request().with_thinking(false);
        req.model = "nvidia/nvidia-nemotron-nano-9b-v2".into();
        assert_eq!(
            body(&by_id("nvidia"), &req)["messages"][0],
            json!({"role": "system", "content": "/no_think\nbe brief"})
        );
    }

    #[test]
    fn custom_requires_base_url() {
        let error = custom().chat_call(&request(), "k", None, false).unwrap_err();
        assert_eq!(error.code, ErrorCode::Config);
        assert!(custom().needs_custom_base_url());
    }
}
