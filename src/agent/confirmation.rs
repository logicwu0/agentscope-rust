//! Serializable human confirmation and resumable tool-call checkpoints.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{ToolCallBlock, ToolCallState};

/// A batch of tool calls paused for explicit user confirmation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PendingToolCalls {
    reply_id: String,
    step: usize,
    calls: Vec<ToolCallBlock>,
}

impl PendingToolCalls {
    pub(crate) fn new(reply_id: String, step: usize, calls: Vec<ToolCallBlock>) -> Self {
        Self {
            reply_id,
            step,
            calls,
        }
    }

    /// Returns the assistant reply identifier used to resume this checkpoint.
    #[must_use]
    pub fn reply_id(&self) -> &str {
        &self.reply_id
    }

    /// Returns the model step that requested the tools.
    #[must_use]
    pub const fn step(&self) -> usize {
        self.step
    }

    /// Returns the tool calls awaiting decisions.
    #[must_use]
    pub fn calls(&self) -> &[ToolCallBlock] {
        &self.calls
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.reply_id.trim().is_empty() {
            return Err("pending tool reply id cannot be empty".to_owned());
        }
        if self.step == 0 {
            return Err("pending tool step must be greater than zero".to_owned());
        }
        if self.calls.is_empty() {
            return Err("pending tool calls cannot be empty".to_owned());
        }
        let mut ids = BTreeSet::new();
        for call in &self.calls {
            if call.state != ToolCallState::Asking {
                return Err(format!(
                    "pending tool call `{}` must be in asking state",
                    call.id()
                ));
            }
            if !ids.insert(call.id()) {
                return Err(format!("duplicate pending tool call id `{}`", call.id()));
            }
        }
        Ok(())
    }
}

/// A durable record that approved tools may already have produced side effects.
///
/// The agent writes this checkpoint before invoking any approved tool. If the
/// process stops before results are persisted, callers must reconcile the
/// external operation and submit its terminal results explicitly.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PendingToolExecution {
    execution_id: String,
    confirmation: PendingToolCalls,
    decisions: Vec<ToolConfirmation>,
}

impl PendingToolExecution {
    pub(crate) fn new(
        execution_id: String,
        confirmation: PendingToolCalls,
        decisions: Vec<ToolConfirmation>,
    ) -> Self {
        Self {
            execution_id,
            confirmation,
            decisions,
        }
    }

    /// Returns the stable identifier for this execution attempt.
    #[must_use]
    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    /// Returns the original confirmation checkpoint.
    #[must_use]
    pub const fn confirmation(&self) -> &PendingToolCalls {
        &self.confirmation
    }

    /// Returns the decisions that authorized this execution.
    #[must_use]
    pub fn decisions(&self) -> &[ToolConfirmation] {
        &self.decisions
    }

    /// Returns the stable idempotency key supplied to one approved tool.
    #[must_use]
    pub fn idempotency_key(&self, tool_call_id: &str) -> Option<String> {
        self.decisions.iter().find_map(|confirmation| {
            (confirmation.tool_call_id() == tool_call_id
                && matches!(confirmation.decision(), ToolConfirmationDecision::Approve))
            .then(|| format!("{}:{tool_call_id}", self.execution_id))
        })
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.execution_id.trim().is_empty() {
            return Err("pending tool execution id cannot be empty".to_owned());
        }
        self.confirmation.validate()?;
        let expected = self
            .confirmation
            .calls()
            .iter()
            .map(ToolCallBlock::id)
            .collect::<BTreeSet<_>>();
        let mut found = BTreeSet::new();
        let mut approved = 0;
        for decision in &self.decisions {
            if !expected.contains(decision.tool_call_id()) {
                return Err(format!(
                    "tool call `{}` is not part of the execution checkpoint",
                    decision.tool_call_id()
                ));
            }
            if !found.insert(decision.tool_call_id()) {
                return Err(format!(
                    "tool call `{}` has more than one execution decision",
                    decision.tool_call_id()
                ));
            }
            if matches!(decision.decision(), ToolConfirmationDecision::Approve) {
                approved += 1;
            }
        }
        if found.len() != expected.len() {
            return Err("execution checkpoint is missing tool decisions".to_owned());
        }
        if approved == 0 {
            return Err("execution checkpoint has no approved tool calls".to_owned());
        }
        Ok(())
    }
}

/// The user's decision for one pending tool call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
pub enum ToolConfirmationDecision {
    /// Permit execution of the tool.
    Approve,
    /// Do not execute the tool and return a denied result to the model.
    Deny {
        /// Human-readable reason shown to the model.
        reason: String,
    },
}

/// A decision associated with one provider-assigned tool-call identifier.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolConfirmation {
    tool_call_id: String,
    decision: ToolConfirmationDecision,
}

impl ToolConfirmation {
    /// Approves one pending tool call.
    #[must_use]
    pub fn approve(tool_call_id: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            decision: ToolConfirmationDecision::Approve,
        }
    }

    /// Denies one pending tool call with a reason visible to the model.
    #[must_use]
    pub fn deny(tool_call_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            decision: ToolConfirmationDecision::Deny {
                reason: reason.into(),
            },
        }
    }

    /// Returns the provider-assigned tool-call identifier.
    #[must_use]
    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    /// Returns the confirmation decision.
    #[must_use]
    pub const fn decision(&self) -> &ToolConfirmationDecision {
        &self.decision
    }

    pub(crate) fn into_parts(self) -> (String, ToolConfirmationDecision) {
        (self.tool_call_id, self.decision)
    }
}

#[cfg(test)]
mod tests {
    use super::{ToolConfirmation, ToolConfirmationDecision};

    #[test]
    fn confirmations_round_trip_through_json() {
        let cases = [
            ToolConfirmation::approve("call-1"),
            ToolConfirmation::deny("call-2", "not authorized"),
        ];

        for confirmation in &cases {
            let encoded = serde_json::to_string(&confirmation).unwrap();
            let decoded: ToolConfirmation = serde_json::from_str(&encoded).unwrap();
            assert_eq!(&decoded, confirmation);
        }
        assert!(matches!(
            cases[0].decision(),
            ToolConfirmationDecision::Approve
        ));
    }
}
