//! Configuration and pre-dispatch selection errors.

use crate::ModelError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Invalid model-router allowlist. Configuration never invokes targets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelRouterConfigError {
    Empty,
    UnknownRoute(String),
    DuplicateRoute(String),
    EmptyDescription { route: String },
}

impl fmt::Display for ModelRouterConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("model router requires an explicit nonempty route catalog"),
            Self::UnknownRoute(route) => {
                write!(f, "model router route {route:?} is not registered")
            }
            Self::DuplicateRoute(route) => write!(f, "model router route {route:?} is duplicated"),
            Self::EmptyDescription { route } => {
                write!(f, "model router route {route:?} needs a description")
            }
        }
    }
}
impl std::error::Error for ModelRouterConfigError {}

/// Selection failed before target dispatch. Local validation errors omit raw
/// classifier output. Provider errors are preserved and may contain private data.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum RouteSelectionError {
    /// The selector does not advertise schema-constrained output.
    UnsupportedModel,
    /// Input has no nonblank visible text; media and tool payloads are not classified.
    EmptyInput,
    /// The selector explicitly declined to choose a suitable route.
    NoMatch,
    /// Invalid, partial, ambiguous or non-allowlisted structured selection.
    InvalidResponse,
    /// Original model failure, including its code and retryable flag.
    Model(ModelError),
}

impl fmt::Display for RouteSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModel => {
                f.write_str("route selector requires structured-output support")
            }
            Self::EmptyInput => f.write_str("route selection requires nonblank visible input text"),
            Self::NoMatch => f.write_str("route selector found no suitable allowed route"),
            Self::InvalidResponse => {
                f.write_str("route selector returned an invalid or disallowed selection")
            }
            Self::Model(error) => write!(f, "route selector failed: {error}"),
        }
    }
}
impl std::error::Error for RouteSelectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}
