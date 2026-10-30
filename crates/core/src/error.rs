//! Errors raised by provider adapters. Messages never contain the caller's API key.

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use std::fmt;

/// Stable error codes, identical to `aibridge_exec.providers.errors.ErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// Key missing, invalid or not allowed for this model.
    Auth,
    /// Provider throttled the request or quota exhausted.
    RateLimited,
    /// Provider rejected the request (unknown model, bad parameters).
    BadRequest,
    /// Model or endpoint does not exist.
    NotFound,
    /// Provider-side failure (5xx, overloaded).
    Upstream,
    Timeout,
    Network,
    UnknownProvider,
    Config,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Auth => "auth",
            ErrorCode::RateLimited => "rate_limited",
            ErrorCode::BadRequest => "bad_request",
            ErrorCode::NotFound => "not_found",
            ErrorCode::Upstream => "upstream",
            ErrorCode::Timeout => "timeout",
            ErrorCode::Network => "network",
            ErrorCode::UnknownProvider => "unknown_provider",
            ErrorCode::Config => "config",
        }
    }

    /// HTTP status the server answers with (`HTTP_STATUS_FOR_CODE`).
    pub fn http_status(self) -> u16 {
        match self {
            ErrorCode::Auth => 401,
            ErrorCode::RateLimited => 429,
            ErrorCode::BadRequest => 400,
            ErrorCode::NotFound => 404,
            ErrorCode::Upstream => 502,
            ErrorCode::Timeout => 504,
            ErrorCode::Network => 502,
            ErrorCode::UnknownProvider => 404,
            ErrorCode::Config => 400,
        }
    }

    pub fn parse(value: &str) -> Option<ErrorCode> {
        Some(match value {
            "auth" => ErrorCode::Auth,
            "rate_limited" => ErrorCode::RateLimited,
            "bad_request" => ErrorCode::BadRequest,
            "not_found" => ErrorCode::NotFound,
            "upstream" => ErrorCode::Upstream,
            "timeout" => ErrorCode::Timeout,
            "network" => ErrorCode::Network,
            "unknown_provider" => ErrorCode::UnknownProvider,
            "config" => ErrorCode::Config,
            _ => return None,
        })
    }

    pub fn for_status(status: u16) -> ErrorCode {
        match status {
            401 | 403 => ErrorCode::Auth,
            429 => ErrorCode::RateLimited,
            404 => ErrorCode::NotFound,
            400..=499 => ErrorCode::BadRequest,
            _ => ErrorCode::Upstream,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ProviderError {
    pub code: ErrorCode,
    pub message: String,
    pub provider: Option<String>,
    /// HTTP status returned by the provider, when there was one.
    pub status: Option<u16>,
}

impl ProviderError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), provider: None, status: None }
    }

    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }
}

/// `{"code", "message", "provider", "upstream_status"}`, like the Python `to_dict`.
impl Serialize for ProviderError {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("code", self.code.as_str())?;
        map.serialize_entry("message", &self.message)?;
        map.serialize_entry("provider", &self.provider)?;
        map.serialize_entry("upstream_status", &self.status)?;
        map.end()
    }
}

pub type Result<T, E = ProviderError> = std::result::Result<T, E>;

/// Replace the key with `***` (keys shorter than 6 characters are left alone, as in Python).
pub fn redact(text: &str, secret: Option<&str>) -> String {
    match secret {
        Some(secret) if secret.chars().count() >= 6 => text.replace(secret, "***"),
        _ => text.to_owned(),
    }
}

/// First `max` characters (Python slicing semantics, not bytes).
pub fn truncate_chars(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((index, _)) => text[..index].to_owned(),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping() {
        assert_eq!(ErrorCode::for_status(403), ErrorCode::Auth);
        assert_eq!(ErrorCode::for_status(422), ErrorCode::BadRequest);
        assert_eq!(ErrorCode::for_status(503), ErrorCode::Upstream);
    }

    #[test]
    fn redaction_and_truncation() {
        assert_eq!(redact("bad key sk-123456", Some("sk-123456")), "bad key ***");
        assert_eq!(redact("abc", Some("abc")), "abc");
        assert_eq!(truncate_chars("héllo", 2), "hé");
    }

    #[test]
    fn serializes_like_to_dict() {
        let error = ProviderError::new(ErrorCode::Auth, "nope").provider("openai").status(401);
        assert_eq!(
            serde_json::to_string(&error).unwrap(),
            r#"{"code":"auth","message":"nope","provider":"openai","upstream_status":401}"#
        );
    }
}
