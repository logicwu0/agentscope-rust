//! Message sender roles.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The role of a message sender.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions supplied by the application.
    System,
    /// Input supplied by an end user.
    User,
    /// Output produced by an agent or model.
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        })
    }
}
