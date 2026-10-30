//! Built-in provider adapters.

pub mod anthropic;
pub mod cohere;
pub mod dashscope;
pub mod gemini;
pub mod ollama;
pub mod openai_compat;
pub mod responses;

use serde_json::{json, Value};

use crate::types::ChatRequest;

/// `[{"role": "system", "content": system}, *turns]` with system text joined, as the native
/// adapters that take a flat role/content list send it.
pub(crate) fn flat_messages(request: &ChatRequest) -> Vec<Value> {
    let (system, turns) = request.split_system();
    let mut messages = Vec::with_capacity(turns.len() + 1);
    if let Some(system) = system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.extend(turns.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})));
    messages
}

/// `Authorization: Bearer` and JSON content type.
pub(crate) fn bearer_headers(api_key: &str) -> Vec<(String, String)> {
    crate::adapter::headers(&[("Authorization", &format!("Bearer {api_key}")), ("Content-Type", "application/json")])
}
