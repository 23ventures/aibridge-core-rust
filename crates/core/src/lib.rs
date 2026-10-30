//! # aibridge-core
//!
//! Rust rewrite of the `aibridge-exec` provider layer: one interface to many LLM providers,
//! with the caller's own API key on every call. Stateless: no key is stored or logged.
//!
//! ```no_run
//! use aibridge_core::{ChatClient, ChatMessage, ChatRequest, Role};
//! use futures_util::StreamExt;
//!
//! # async fn demo() -> aibridge_core::Result<()> {
//! let client = ChatClient::new()?;
//! let request = ChatRequest::new("<model id>", vec![ChatMessage::new(Role::User, "Hello")]);
//! let reply = client.chat("anthropic", "sk-ant-...", request.clone(), None).await?;
//! println!("{}", reply.text);
//!
//! let mut stream = client.stream("openai", "sk-...", request, None).await?;
//! while let Some(event) = stream.next().await {
//!     println!("{:?}", event?);
//! }
//! # Ok(()) }
//! ```
//!
//! Wire formats (JSON shapes, error codes, SSE events) match aibridge-exec exactly, so the
//! server binary and the Python bindings are drop-in replacements.

pub mod adapter;
pub mod client;
pub mod error;
pub mod layer;
pub mod providers;
pub mod registry;
pub mod sse;
pub mod types;

pub use adapter::{HttpCall, Method, ProviderAdapter, StreamFormat, StreamParser};
pub use client::{ChatClient, ChatClientBuilder, ChatStream, Timeouts};
pub use error::{ErrorCode, ProviderError, Result};
pub use layer::{CacheConfig, CacheLayer, CacheStats, CallInfo, ChatCachePolicy, KeyFingerprint, Layer};
pub use registry::Registry;
pub use types::{
    ChatMessage, ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, ResponseFormat, Role, StreamEvent,
    Usage, REASONING_EFFORTS,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
