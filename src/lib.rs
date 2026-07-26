//! AgentScope for Rust.
//!
//! The public API is intentionally empty while the project scope and
//! compatibility goals are being designed.

#![forbid(unsafe_code)]

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
