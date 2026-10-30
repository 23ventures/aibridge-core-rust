//! Pluggable layers around provider calls: caching today, other optimizations later.
//!
//! A [`Layer`] can answer a call itself (a cache hit) or observe a finished call (to store it,
//! record metrics, ...). Layers never see the caller's API key: they get a salted SHA-256
//! [`KeyFingerprint`] instead, so per-key results stay separate without the key being held.
//!
//! [`CacheLayer`] is the built-in implementation, an in-memory TTL cache for model lists and
//! (opt-in) chat replies. A shared cache (Redis, ...) is another `Layer` implementation.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::types::{ChatRequest, ChatResponse, ModelInfo};

/// Salted hash of an API key. The salt is random per client, so fingerprints mean nothing
/// outside the process that made them.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyFingerprint(pub [u8; 32]);

impl KeyFingerprint {
    pub fn new(salt: &[u8; 32], api_key: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(salt);
        hasher.update(api_key.as_bytes());
        Self(hasher.finalize().into())
    }
}

impl std::fmt::Debug for KeyFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KeyFingerprint({}…)", hex::encode(&self.0[..4]))
    }
}

/// What a layer knows about a call.
#[derive(Debug, Clone, Copy)]
pub struct CallInfo<'a> {
    pub provider: &'a str,
    /// Effective base URL override, if any (part of the cache identity).
    pub base_url: Option<&'a str>,
    pub key: &'a KeyFingerprint,
    pub stream: bool,
}

#[async_trait]
pub trait Layer: Send + Sync {
    /// Answer a chat call without contacting the provider (streamed calls are replayed).
    async fn lookup_chat(&self, _call: &CallInfo<'_>, _request: &ChatRequest) -> Option<ChatResponse> {
        None
    }

    /// A chat call finished successfully (for streams: the `done` event arrived).
    async fn on_chat(&self, _call: &CallInfo<'_>, _request: &ChatRequest, _response: &ChatResponse) {}

    async fn lookup_models(&self, _call: &CallInfo<'_>) -> Option<Vec<ModelInfo>> {
        None
    }

    async fn on_models(&self, _call: &CallInfo<'_>, _models: &[ModelInfo]) {}
}

/// Which chat replies [`CacheLayer`] keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatCachePolicy {
    /// Only requests with `temperature == 0` (deterministic).
    Deterministic,
    /// Every request: identical requests get identical replies until the entry expires.
    All,
}

impl ChatCachePolicy {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "deterministic" => Some(Self::Deterministic),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    fn allows(self, request: &ChatRequest) -> bool {
        match self {
            Self::All => true,
            Self::Deterministic => request.temperature == Some(0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// How long model lists are kept. `None` disables the model cache.
    pub models_ttl: Option<Duration>,
    /// How long chat replies are kept. `None` disables the chat cache.
    pub chat_ttl: Option<Duration>,
    pub chat_policy: ChatCachePolicy,
    /// Upper bound per cache; the oldest entries are evicted first.
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { models_ttl: None, chat_ttl: None, chat_policy: ChatCachePolicy::Deterministic, max_entries: 10_000 }
    }
}

impl CacheConfig {
    pub fn is_enabled(&self) -> bool {
        self.models_ttl.is_some() || self.chat_ttl.is_some()
    }
}

/// In-memory TTL caches for model lists and chat replies.
pub struct CacheLayer {
    config: CacheConfig,
    models: TtlCache<Vec<ModelInfo>>,
    chat: TtlCache<ChatResponse>,
}

impl CacheLayer {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            models: TtlCache::new(config.models_ttl.unwrap_or_default(), config.max_entries),
            chat: TtlCache::new(config.chat_ttl.unwrap_or_default(), config.max_entries),
            config,
        }
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats { models: self.models.stats(), chat: self.chat.stats() }
    }

    fn models_key(call: &CallInfo<'_>) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"models\0");
        hasher.update(call.provider.as_bytes());
        hasher.update([0]);
        hasher.update(call.base_url.unwrap_or("").as_bytes());
        hasher.update([0]);
        hasher.update(call.key.0);
        hasher.finalize().into()
    }

    fn chat_key(call: &CallInfo<'_>, request: &ChatRequest) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"chat\0");
        hasher.update(call.provider.as_bytes());
        hasher.update([0]);
        hasher.update(call.base_url.unwrap_or("").as_bytes());
        hasher.update([0]);
        hasher.update(call.key.0);
        // Field order of ChatRequest is fixed, so the JSON is a stable identity.
        hasher.update(serde_json::to_vec(request).unwrap_or_default());
        hasher.finalize().into()
    }
}

#[async_trait]
impl Layer for CacheLayer {
    async fn lookup_chat(&self, call: &CallInfo<'_>, request: &ChatRequest) -> Option<ChatResponse> {
        if self.config.chat_ttl.is_none() || !self.config.chat_policy.allows(request) {
            return None;
        }
        self.chat.get(&Self::chat_key(call, request))
    }

    async fn on_chat(&self, call: &CallInfo<'_>, request: &ChatRequest, response: &ChatResponse) {
        if self.config.chat_ttl.is_some() && self.config.chat_policy.allows(request) {
            self.chat.insert(Self::chat_key(call, request), response.clone());
        }
    }

    async fn lookup_models(&self, call: &CallInfo<'_>) -> Option<Vec<ModelInfo>> {
        self.config.models_ttl?;
        self.models.get(&Self::models_key(call))
    }

    async fn on_models(&self, call: &CallInfo<'_>, models: &[ModelInfo]) {
        if self.config.models_ttl.is_some() {
            self.models.insert(Self::models_key(call), models.to_vec());
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct CounterStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct CacheStats {
    pub models: CounterStats,
    pub chat: CounterStats,
}

struct Entry<V> {
    value: V,
    inserted: Instant,
}

struct TtlInner<V> {
    entries: HashMap<[u8; 32], Entry<V>>,
    hits: u64,
    misses: u64,
}

/// Small TTL map with a size bound. Expired entries are dropped lazily.
pub struct TtlCache<V> {
    ttl: Duration,
    max_entries: usize,
    inner: Mutex<TtlInner<V>>,
}

impl<V: Clone> TtlCache<V> {
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            ttl,
            max_entries: max_entries.max(1),
            inner: Mutex::new(TtlInner { entries: HashMap::new(), hits: 0, misses: 0 }),
        }
    }

    pub fn get(&self, key: &[u8; 32]) -> Option<V> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let fresh = inner.entries.get(key).filter(|e| e.inserted.elapsed() < self.ttl).map(|e| e.value.clone());
        match fresh {
            Some(value) => {
                inner.hits += 1;
                Some(value)
            }
            None => {
                inner.entries.remove(key);
                inner.misses += 1;
                None
            }
        }
    }

    pub fn insert(&self, key: [u8; 32], value: V) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.entries.len() >= self.max_entries && !inner.entries.contains_key(&key) {
            let ttl = self.ttl;
            inner.entries.retain(|_, e| e.inserted.elapsed() < ttl);
            if inner.entries.len() >= self.max_entries {
                if let Some(oldest) = inner.entries.iter().min_by_key(|(_, e)| e.inserted).map(|(k, _)| *k) {
                    inner.entries.remove(&oldest);
                }
            }
        }
        inner.entries.insert(key, Entry { value, inserted: Instant::now() });
    }

    pub fn stats(&self) -> CounterStats {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        CounterStats { hits: inner.hits, misses: inner.misses, entries: inner.entries.len() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Role, Usage};

    fn reply() -> ChatResponse {
        ChatResponse { text: "hi".into(), model: "m".into(), finish_reason: None, usage: Usage::default() }
    }

    #[tokio::test]
    async fn chat_cache_respects_policy_and_key() {
        let layer = CacheLayer::new(CacheConfig { chat_ttl: Some(Duration::from_secs(60)), ..Default::default() });
        let salt = [7u8; 32];
        let (a, b) = (KeyFingerprint::new(&salt, "key-a"), KeyFingerprint::new(&salt, "key-b"));
        let call_a = CallInfo { provider: "openai", base_url: None, key: &a, stream: false };
        let call_b = CallInfo { key: &b, ..call_a };
        let request = ChatRequest::new("m", vec![ChatMessage::new(Role::User, "x")]);
        let deterministic = request.clone().with_temperature(0.0);

        layer.on_chat(&call_a, &request, &reply()).await;
        assert!(layer.lookup_chat(&call_a, &request).await.is_none(), "non-deterministic requests are not cached");
        layer.on_chat(&call_a, &deterministic, &reply()).await;
        assert_eq!(layer.lookup_chat(&call_a, &deterministic).await, Some(reply()));
        assert!(layer.lookup_chat(&call_b, &deterministic).await.is_none(), "another key never sees the entry");
    }

    #[test]
    fn ttl_and_eviction() {
        let cache = TtlCache::new(Duration::from_millis(30), 2);
        cache.insert([1; 32], 1);
        cache.insert([2; 32], 2);
        cache.insert([3; 32], 3);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.get(&[1; 32]), None, "oldest entry evicted");
        assert_eq!(cache.get(&[3; 32]), Some(3));
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(cache.get(&[3; 32]), None, "expired");
    }
}
