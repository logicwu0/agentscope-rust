//! Serializable, versioned agent state snapshots.

use serde::{Deserialize, Serialize};

use crate::Msg;

/// The state format version emitted by this crate.
pub const AGENT_STATE_VERSION: u32 = 1;

/// A complete, restartable snapshot of one agent's conversation state.
///
/// Runtime-only controls such as interruption handles are deliberately not
/// included. Restore a snapshot only while no reply is running on the target
/// agent.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AgentState {
    format_version: u32,
    agent_name: String,
    messages: Vec<Msg>,
}

impl AgentState {
    /// Creates a snapshot in the current format.
    #[must_use]
    pub fn new(agent_name: impl Into<String>, messages: Vec<Msg>) -> Self {
        Self {
            format_version: AGENT_STATE_VERSION,
            agent_name: agent_name.into(),
            messages,
        }
    }

    /// Returns the serialized state format version.
    #[must_use]
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the name of the agent that created this snapshot.
    #[must_use]
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// Returns the conversation history in chronological order.
    #[must_use]
    pub fn messages(&self) -> &[Msg] {
        &self.messages
    }

    pub(crate) fn into_messages(self) -> Vec<Msg> {
        self.messages
    }
}

#[cfg(test)]
mod tests {
    use super::{AGENT_STATE_VERSION, AgentState};
    use crate::Msg;

    #[test]
    fn state_round_trips_through_json_with_an_explicit_version() {
        let state = AgentState::new("Friday", vec![Msg::user("Hello")]);
        let json = serde_json::to_value(&state).unwrap();

        assert_eq!(json["format_version"], AGENT_STATE_VERSION);
        assert_eq!(json["agent_name"], "Friday");
        assert_eq!(serde_json::from_value::<AgentState>(json).unwrap(), state);
    }
}
