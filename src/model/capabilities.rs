//! Chat model capability declarations.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// One provider-neutral feature supported by a chat model.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    /// The model can emit incremental responses.
    Streaming,
    /// The model can request tools.
    ToolCalls,
    /// The model supports schema-constrained output.
    StructuredOutput,
    /// The model accepts multimodal input blocks.
    MultimodalInput,
}

/// Provider-neutral capabilities advertised by a chat model.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ModelCapabilities(BTreeSet<ModelCapability>);

impl ModelCapabilities {
    /// Creates an empty capability declaration.
    #[must_use]
    pub const fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Creates a declaration containing every currently known capability.
    #[must_use]
    pub fn all() -> Self {
        Self(BTreeSet::from([
            ModelCapability::Streaming,
            ModelCapability::ToolCalls,
            ModelCapability::StructuredOutput,
            ModelCapability::MultimodalInput,
        ]))
    }

    /// Returns whether `capability` is supported.
    #[must_use]
    pub fn supports(&self, capability: ModelCapability) -> bool {
        self.0.contains(&capability)
    }

    /// Enables or disables one capability.
    #[must_use]
    pub fn with(mut self, capability: ModelCapability, enabled: bool) -> Self {
        if enabled {
            self.0.insert(capability);
        } else {
            self.0.remove(&capability);
        }
        self
    }

    /// Iterates over supported capabilities in stable wire order.
    pub fn iter(&self) -> impl Iterator<Item = ModelCapability> + '_ {
        self.0.iter().copied()
    }
}
