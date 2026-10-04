//! Configuration and attributed runtime failures for explicit routing.

use super::RouteSelectionError;
use crate::AgentError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Invalid immutable route configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RoutedConfigError {
    /// At least one route must be registered.
    Empty,
    /// One-based registration entry with a blank route key.
    EmptyRoute { entry: usize },
    /// Duplicate exact key; an existing target is never replaced.
    DuplicateRoute(String),
    /// Selected registration has a blank agent name.
    EmptyAgentName { route: String },
}

impl fmt::Display for RoutedConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("routed pipeline requires at least one route"),
            Self::EmptyRoute { entry } => write!(f, "route entry {entry} has a blank key"),
            Self::DuplicateRoute(route) => write!(f, "route key {route:?} is duplicated"),
            Self::EmptyAgentName { route } => write!(f, "route {route:?} has a blank agent name"),
        }
    }
}
impl std::error::Error for RoutedConfigError {}

/// Why one routed run stopped. No failure authorizes retrying external effects.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum RoutedFailure {
    /// The exact route key is unregistered; no target was invoked.
    UnknownRoute,
    /// Another run or stream on this pipeline or a clone holds the shared lock.
    Busy,
    /// The pipeline interrupted the active reply; effects may be uncertain.
    Interrupted,
    /// Model selection failed before any target was invoked.
    Selection(RouteSelectionError),
    /// The original error, including confirmation or uncertain tool execution.
    Agent(Box<AgentError>),
    /// Storage failed or conflicted; inspect the authoritative record before retrying.
    Store(String),
    /// Missing, invalid, incompatible or already dispatched checkpoint progress.
    UnsafeResume(String),
}

/// Diagnostic failure attributed to the caller's requested route.
///
/// This is not a checkpoint or permission to replay. Inspect the selected
/// agent's own state and external effects before any retry or reconciliation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RoutedError {
    /// Exact requested or stored route key, including an unknown key. Empty when
    /// a checkpoint or model-selection operation has not obtained its selection yet.
    pub route: String,
    /// Captured target name when the requested or saved selection is available.
    /// `None` for unknown routes and failures before obtaining a selection.
    pub agent_name: Option<String>,
    pub cause: RoutedFailure,
}

impl fmt::Display for RoutedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "route {:?}", self.route)?;
        if let Some(name) = &self.agent_name {
            write!(f, " (agent {name:?})")?;
        }
        match &self.cause {
            RoutedFailure::UnknownRoute => f.write_str(" is unknown; no agent was dispatched"),
            RoutedFailure::Busy => {
                f.write_str(" cannot run: routed pipeline already has an active run")
            }
            RoutedFailure::Interrupted => {
                f.write_str(" interrupted; prior effects are not rolled back")
            }
            RoutedFailure::Selection(error) => write!(f, " selection failed: {error}"),
            RoutedFailure::Agent(error) => write!(f, " stopped: {error}"),
            RoutedFailure::Store(reason) => write!(f, " checkpoint storage failed: {reason}"),
            RoutedFailure::UnsafeResume(reason) => {
                write!(f, " cannot resume or reconcile: {reason}")
            }
        }
    }
}

impl std::error::Error for RoutedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            RoutedFailure::Agent(error) => Some(error.as_ref()),
            RoutedFailure::Selection(error) => Some(error),
            _ => None,
        }
    }
}
