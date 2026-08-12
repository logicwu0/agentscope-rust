//! `AgentScope` for Rust.

#![forbid(unsafe_code)]

pub mod message;

pub use message::{ContentBlock, Metadata, Msg, Role, TextBlock};

/// The current crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_package_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
