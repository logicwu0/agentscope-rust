//! Serializable lifecycle of one explicitly selected agent stream.

use super::{RoutedError, RoutedOutput};
use crate::AgentEvent;
use serde::{Deserialize, Serialize};

/// Route-attributed events without modifying the selected agent's payloads.
///
/// A fully consumed stream emits exactly one terminal [`Self::Finished`] or
/// [`Self::Error`]. Dropping it emits no synthetic terminal event. Payloads may
/// include private thinking, metadata or data blocks; select visible content
/// before displaying or sharing events.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoutedEvent {
    /// The route is selected; the agent operation has not yet been invoked.
    /// Checkpointed streams have committed the `InFlight` dispatch fence.
    RouteStarted {
        /// Exact caller-selected route key.
        route: String,
        /// Agent name captured when constructing the pipeline.
        agent_name: String,
    },
    /// An unmodified event from the selected agent, including its local step.
    /// In checkpointed streams, even an agent terminal event is an observation,
    /// not a durable acknowledgement; consume the routed terminal event.
    Agent {
        /// Exact caller-selected route key.
        route: String,
        /// Agent name captured when constructing the pipeline.
        agent_name: String,
        /// Original event, not a synthesized or filtered message.
        event: AgentEvent,
    },
    /// The agent finished; contains its complete original reply. Checkpointed
    /// streams emit this only after committing the completed checkpoint.
    Finished { output: RoutedOutput },
    /// Terminal failure; no other route is dispatched and no retry is authorized.
    /// In checkpointed streams, an agent failure acknowledges a committed
    /// `Failed` result; interruption or storage errors do not acknowledge it.
    Error { error: RoutedError },
}
