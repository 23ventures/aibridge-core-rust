//! Provider registry. Start from [`Registry::builtin`] and [`Registry::register`] your own adapters.

use std::sync::Arc;

use crate::adapter::ProviderAdapter;
use crate::error::{ErrorCode, ProviderError, Result};
use crate::providers::{anthropic::AnthropicAdapter, gemini::GeminiAdapter, openai_compat};

#[derive(Clone, Default)]
pub struct Registry {
    adapters: Vec<Arc<dyn ProviderAdapter>>,
}

impl Registry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The built-in providers, in the same order as aibridge-exec.
    pub fn builtin() -> Self {
        let mut registry = Self::new();
        for adapter in openai_compat::builtin() {
            registry.adapters.push(Arc::new(adapter));
        }
        registry.adapters.push(Arc::new(AnthropicAdapter::default()));
        registry.adapters.push(Arc::new(GeminiAdapter::default()));
        registry.adapters.push(Arc::new(openai_compat::custom()));
        registry
    }

    /// Add an adapter. Fails if the id is taken, unless `replace` is set.
    pub fn register(&mut self, adapter: Arc<dyn ProviderAdapter>, replace: bool) -> Result<(), String> {
        match self.adapters.iter().position(|a| a.id() == adapter.id()) {
            Some(_) if !replace => Err(format!("provider '{}' is already registered", adapter.id())),
            Some(index) => {
                self.adapters[index] = adapter;
                Ok(())
            }
            None => {
                self.adapters.push(adapter);
                Ok(())
            }
        }
    }

    pub fn get(&self, provider_id: &str) -> Result<Arc<dyn ProviderAdapter>> {
        self.adapters.iter().find(|a| a.id() == provider_id).cloned().ok_or_else(|| {
            ProviderError::new(ErrorCode::UnknownProvider, format!("unknown provider {}", py_repr(provider_id)))
                .provider(provider_id)
        })
    }

    pub fn list(&self) -> &[Arc<dyn ProviderAdapter>] {
        &self.adapters
    }
}

/// `repr()` of a Python string, used so error messages read exactly like aibridge-exec's.
pub fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_order_and_lookup() {
        let registry = Registry::builtin();
        let ids: Vec<&str> = registry.list().iter().map(|a| a.id()).collect();
        assert_eq!(
            ids,
            [
                "openai",
                "xai",
                "openrouter",
                "groq",
                "deepseek",
                "mistral",
                "together",
                "anthropic",
                "gemini",
                "openai_compatible"
            ]
        );
        let error = registry.get("nope").err().unwrap();
        assert_eq!((error.code, error.message.as_str()), (ErrorCode::UnknownProvider, "unknown provider 'nope'"));
    }

    #[test]
    fn repr_quoting() {
        assert_eq!(py_repr("a'b"), "\"a'b\"");
        assert_eq!(py_repr("a\"b'"), "'a\"b\\''");
    }
}
