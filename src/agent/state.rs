//! Serializable, versioned agent state snapshots.

use serde::{Deserialize, Serialize};

use crate::Msg;

use super::PendingToolCalls;

/// The state format version emitted by this crate.
pub const AGENT_STATE_VERSION: u32 = 2;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending_tool_calls: Option<PendingToolCalls>,
}

impl AgentState {
    /// Creates a snapshot in the current format.
    #[must_use]
    pub fn new(agent_name: impl Into<String>, messages: Vec<Msg>) -> Self {
        Self {
            format_version: AGENT_STATE_VERSION,
            agent_name: agent_name.into(),
            messages,
            pending_tool_calls: None,
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

    /// Returns the tool calls awaiting human confirmation, when paused.
    #[must_use]
    pub const fn pending_tool_calls(&self) -> Option<&PendingToolCalls> {
        self.pending_tool_calls.as_ref()
    }

    pub(crate) fn with_pending_tool_calls(mut self, pending: Option<PendingToolCalls>) -> Self {
        self.pending_tool_calls = pending;
        self
    }

    pub(crate) fn into_parts(self) -> (Vec<Msg>, Option<PendingToolCalls>) {
        (self.messages, self.pending_tool_calls)
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
