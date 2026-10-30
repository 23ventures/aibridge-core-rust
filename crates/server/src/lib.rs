//! aibridge-core HTTP server, wire-compatible with the aibridge-exec server.

pub mod app;
pub mod body;
pub mod settings;

pub use app::{router, AppState, SharedState};
pub use settings::ServerSettings;
