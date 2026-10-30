//! Server configuration from environment variables.
//!
//! Every `AIBRIDGE_EXEC_*` variable of the Python server is honored unchanged, so an existing
//! deployment switches by swapping the image. `AIBRIDGE_CORE_*` spellings take precedence when
//! both are set. Cache settings are new and exist only as `AIBRIDGE_CORE_*`.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use aibridge_core::{CacheConfig, ChatCachePolicy};

#[derive(Debug, Clone)]
pub struct ServerSettings {
    /// Service tokens callers must send as `Authorization: Bearer <token>`. Empty = open (local dev only).
    pub api_tokens: Vec<String>,
    /// Browser origins allowed to call the server directly (CORS). Empty = none.
    pub cors_origins: Vec<String>,
    /// Provider ids to expose. Empty = every registered provider except `openai_compatible` and `ollama`
    /// (unless custom base URLs are allowed or a base URL is configured for them).
    pub providers: Vec<String>,
    /// Let callers send their own `base_url` (enables `openai_compatible`). Off by default: it lets
    /// callers make the server send requests to any URL, including your internal network.
    pub allow_custom_base_url: bool,
    /// Server-side base URL overrides, e.g. `{"openai": "https://gateway.internal/openai/v1"}`.
    pub base_urls: HashMap<String, String>,
    pub max_messages: usize,
    pub max_message_chars: usize,
    pub max_total_chars: usize,
    pub cache: CacheConfig,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            api_tokens: Vec::new(),
            cors_origins: Vec::new(),
            providers: Vec::new(),
            allow_custom_base_url: false,
            base_urls: HashMap::new(),
            max_messages: 500,
            max_message_chars: 200_000,
            max_total_chars: 1_000_000,
            cache: CacheConfig::default(),
        }
    }
}

fn list(value: Option<String>) -> Vec<String> {
    value.unwrap_or_default().split(',').map(str::trim).filter(|item| !item.is_empty()).map(str::to_owned).collect()
}

fn boolean(value: Option<String>) -> bool {
    matches!(value.unwrap_or_default().trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

impl ServerSettings {
    pub fn from_env() -> Result<Self, String> {
        Self::from_vars(std::env::vars())
    }

    pub fn from_vars(vars: impl IntoIterator<Item = (String, String)>) -> Result<Self, String> {
        let env: BTreeMap<String, String> = vars.into_iter().collect();
        // AIBRIDGE_CORE_X wins over AIBRIDGE_EXEC_X.
        let get = |name: &str| -> Option<String> {
            env.get(&format!("AIBRIDGE_CORE_{name}")).or_else(|| env.get(&format!("AIBRIDGE_EXEC_{name}"))).cloned()
        };
        let number = |name: &str, default: usize| -> Result<usize, String> {
            match get(name) {
                Some(value) => {
                    value.trim().parse().map_err(|_| format!("{name} must be a whole number, got {value:?}"))
                }
                None => Ok(default),
            }
        };
        let seconds = |name: &str| -> Result<Option<Duration>, String> {
            match env.get(&format!("AIBRIDGE_CORE_{name}")) {
                None => Ok(None),
                Some(value) => {
                    let secs: f64 = value
                        .trim()
                        .parse()
                        .map_err(|_| format!("AIBRIDGE_CORE_{name} must be a number of seconds"))?;
                    Ok((secs > 0.0).then(|| Duration::from_secs_f64(secs)))
                }
            }
        };

        let mut base_urls = HashMap::new();
        for prefix in ["AIBRIDGE_EXEC_BASE_URL_", "AIBRIDGE_CORE_BASE_URL_"] {
            for (key, value) in &env {
                if let Some(provider) = key.strip_prefix(prefix) {
                    if !value.is_empty() {
                        base_urls.insert(provider.to_ascii_lowercase(), value.clone());
                    }
                }
            }
        }

        let defaults = Self::default();
        let chat_policy = match env.get("AIBRIDGE_CORE_CACHE_CHAT_POLICY") {
            None => ChatCachePolicy::Deterministic,
            Some(value) => ChatCachePolicy::parse(value)
                .ok_or_else(|| "AIBRIDGE_CORE_CACHE_CHAT_POLICY must be 'deterministic' or 'all'".to_owned())?,
        };
        Ok(Self {
            api_tokens: list(get("API_TOKENS")),
            cors_origins: list(get("CORS_ORIGINS")),
            providers: list(get("PROVIDERS")),
            allow_custom_base_url: boolean(get("ALLOW_CUSTOM_BASE_URL")),
            base_urls,
            max_messages: number("MAX_MESSAGES", defaults.max_messages)?,
            max_message_chars: number("MAX_MESSAGE_CHARS", defaults.max_message_chars)?,
            max_total_chars: number("MAX_TOTAL_CHARS", defaults.max_total_chars)?,
            cache: CacheConfig {
                models_ttl: seconds("CACHE_MODELS_TTL")?,
                chat_ttl: seconds("CACHE_CHAT_TTL")?,
                chat_policy,
                max_entries: env
                    .get("AIBRIDGE_CORE_CACHE_MAX_ENTRIES")
                    .map(|v| v.trim().parse().map_err(|_| "AIBRIDGE_CORE_CACHE_MAX_ENTRIES must be a whole number"))
                    .transpose()?
                    .unwrap_or(10_000),
            },
        })
    }

    pub fn provider_enabled(&self, provider_id: &str) -> bool {
        if !self.providers.is_empty() {
            return self.providers.iter().any(|p| p == provider_id);
        }
        !matches!(provider_id, "openai_compatible" | "ollama")
            || self.allow_custom_base_url
            || self.base_urls.contains_key(provider_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect()
    }

    #[test]
    fn reads_exec_variables_and_core_overrides() {
        let settings = ServerSettings::from_vars(vars(&[
            ("AIBRIDGE_EXEC_API_TOKENS", "a, b,"),
            ("AIBRIDGE_EXEC_MAX_MESSAGES", "10"),
            ("AIBRIDGE_CORE_MAX_MESSAGES", "20"),
            ("AIBRIDGE_EXEC_BASE_URL_OPENAI", "http://exec"),
            ("AIBRIDGE_CORE_BASE_URL_OPENAI", "http://core"),
            ("AIBRIDGE_EXEC_ALLOW_CUSTOM_BASE_URL", "Yes"),
            ("AIBRIDGE_CORE_CACHE_MODELS_TTL", "300"),
        ]))
        .unwrap();
        assert_eq!(settings.api_tokens, ["a", "b"]);
        assert_eq!(settings.max_messages, 20);
        assert_eq!(settings.base_urls["openai"], "http://core");
        assert!(settings.allow_custom_base_url);
        assert_eq!(settings.cache.models_ttl, Some(Duration::from_secs(300)));
        assert_eq!(settings.cache.chat_ttl, None);
    }

    #[test]
    fn provider_filter() {
        let settings = ServerSettings::default();
        assert!(settings.provider_enabled("openai") && !settings.provider_enabled("openai_compatible"));
        let only = ServerSettings { providers: vec!["anthropic".into()], ..Default::default() };
        assert!(!only.provider_enabled("openai"));
    }

    #[test]
    fn bad_numbers_are_reported() {
        assert!(ServerSettings::from_vars(vars(&[("AIBRIDGE_EXEC_MAX_MESSAGES", "x")])).is_err());
    }
}
