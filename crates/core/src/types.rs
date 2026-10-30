//! Provider-neutral chat types. JSON shapes match `aibridge_exec.providers.types`.

use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};

/// Message role. `system` messages are merged into the system prompt by the adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    pub fn parse(value: &str) -> Option<Role> {
        match value {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// One chat completion call. `system` and any `system` messages are merged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self { model: model.into(), messages, system: None, temperature: None, max_tokens: None }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Same rules and messages as the Python `ChatRequest.__post_init__`.
    pub fn validate(&self) -> Result<(), String> {
        if self.model.is_empty() {
            return Err("model is required".into());
        }
        if self.messages.is_empty() {
            return Err("at least one message is required".into());
        }
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err("temperature must be between 0 and 2".into());
            }
        }
        if self.max_tokens == Some(0) {
            return Err("max_tokens must be positive".into());
        }
        Ok(())
    }

    /// System text (from `system` plus system messages) and the remaining turns.
    pub fn split_system(&self) -> (Option<String>, Vec<ChatMessage>) {
        let mut parts: Vec<&str> = Vec::new();
        if let Some(system) = self.system.as_deref() {
            if !system.is_empty() {
                parts.push(system);
            }
        }
        let mut turns = Vec::with_capacity(self.messages.len());
        for message in &self.messages {
            if message.role == Role::System {
                if !message.content.is_empty() {
                    parts.push(&message.content);
                }
            } else {
                turns.push(message.clone());
            }
        }
        let joined = parts.join("\n\n");
        ((!joined.is_empty()).then_some(joined), turns)
    }
}

/// Join back-to-back turns with the same role (providers that require alternation).
pub fn merge_consecutive(turns: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut merged: Vec<ChatMessage> = Vec::with_capacity(turns.len());
    for turn in turns {
        match merged.last_mut() {
            Some(last) if last.role == turn.role => {
                last.content.push_str("\n\n");
                last.content.push_str(&turn.content);
            }
            _ => merged.push(turn),
        }
    }
    merged
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Error,
    Other,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::Error => "error",
            FinishReason::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

impl Usage {
    pub fn new(input_tokens: Option<u64>, output_tokens: Option<u64>) -> Self {
        Self { input_tokens, output_tokens }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub text: String,
    pub model: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Usage,
}

/// `delta` events carry text; one final `done` event carries usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum StreamEvent {
    Delta { text: String },
    Done { finish_reason: Option<FinishReason>, model: Option<String>, usage: Usage },
}

impl StreamEvent {
    pub fn delta(text: impl Into<String>) -> Self {
        StreamEvent::Delta { text: text.into() }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            StreamEvent::Delta { .. } => "delta",
            StreamEvent::Done { .. } => "done",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: Option<String>,
    pub context_length: Option<u64>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>, name: Option<String>, context_length: Option<u64>) -> Self {
        Self { id: id.into(), name, context_length }
    }
}

/// Serialized like the Python `to_dict`: `name` falls back to `id`.
impl Serialize for ModelInfo {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("name", self.name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&self.id))?;
        map.serialize_entry("context_length", &self.context_length)?;
        map.end()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderInfo {
    pub id: String,
    pub label: String,
    pub key_url: String,
    pub default_base_url: String,
    pub key_hint: String,
}

impl ProviderInfo {
    pub fn new(id: &str, label: &str, default_base_url: &str, key_url: &str, key_hint: &str) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            key_url: key_url.into(),
            default_base_url: default_base_url.into(),
            key_hint: key_hint.into(),
        }
    }
}

/// Serialized like the Python `to_dict` (no base URL).
impl Serialize for ProviderInfo {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("label", &self.label)?;
        map.serialize_entry("key_url", &self.key_url)?;
        map.serialize_entry("key_hint", &self.key_hint)?;
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_system_merges_system_prompt_and_messages() {
        let request =
            ChatRequest::new("m", vec![ChatMessage::new(Role::System, "a"), ChatMessage::new(Role::User, "hi")])
                .with_system("top");
        let (system, turns) = request.split_system();
        assert_eq!(system.as_deref(), Some("top\n\na"));
        assert_eq!(turns, vec![ChatMessage::new(Role::User, "hi")]);
    }

    #[test]
    fn validation_messages_match_python() {
        assert_eq!(ChatRequest::new("", vec![]).validate().unwrap_err(), "model is required");
        assert_eq!(ChatRequest::new("m", vec![]).validate().unwrap_err(), "at least one message is required");
        let one = vec![ChatMessage::new(Role::User, "x")];
        assert!(ChatRequest::new("m", one.clone()).with_temperature(2.5).validate().is_err());
        assert!(ChatRequest::new("m", one).with_max_tokens(0).validate().is_err());
    }

    #[test]
    fn json_shapes_match_python_to_dict() {
        let done = StreamEvent::Done {
            finish_reason: Some(FinishReason::Stop),
            model: Some("m".into()),
            usage: Usage::new(Some(1), None),
        };
        assert_eq!(
            serde_json::to_string(&done).unwrap(),
            r#"{"type":"done","finish_reason":"stop","model":"m","usage":{"input_tokens":1,"output_tokens":null}}"#
        );
        assert_eq!(serde_json::to_string(&StreamEvent::delta("x")).unwrap(), r#"{"type":"delta","text":"x"}"#);
        let model = ModelInfo::new("id", None, Some(5));
        assert_eq!(serde_json::to_string(&model).unwrap(), r#"{"id":"id","name":"id","context_length":5}"#);
    }
}
