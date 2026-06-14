// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! The extraction adapter trait and the data-transfer types it exchanges with
//! the pipeline. An adapter turns a chunk of client-supplied text into proposed
//! entities, edges, and ontology proposals.

use serde::{Deserialize, Serialize};

use crate::error::ExtractionError;
use crate::ontology::{Ontology, RawProposal};

/// Identifier of a source document a chunk belongs to.
pub type DocId = String;

/// One client-supplied chunk of text, the unit of extraction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkContent {
    pub doc_id: DocId,
    pub chunk_id: u64,
    pub text: String,
    /// Optional embedding, used for entity dedup when present.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// A baked-in, semver-like prompt version. Bumping it invalidates the cache.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptVersion(pub &'static str);

impl PromptVersion {
    /// The version string as a `&str`.
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for PromptVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// An entity the LLM proposed from a chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposedEntity {
    /// Entity label; must be in the ontology or it triggers a proposal.
    pub label: String,
    /// Canonical name; the dedup key within a label.
    pub name: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub confidence: f32,
    /// Exact source span cited by the LLM.
    pub source_text: String,
    /// Local id within this extraction, referenced by edges.
    pub ref_id: String,
}

/// An edge the LLM proposed from a chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposedEdge {
    /// `ref_id` of a proposed entity, or `@chunk` for the content node.
    pub source_ref: String,
    pub target_ref: String,
    pub edge_type: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub confidence: f32,
    pub source_text: String,
}

/// The full result of extracting one chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub entities: Vec<ProposedEntity>,
    pub edges: Vec<ProposedEdge>,
    /// LLM-suggested new labels or edge types.
    #[serde(default)]
    pub ontology_proposals: Vec<RawProposal>,
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    /// True when `input_tokens`/`output_tokens` came from the provider's own
    /// usage block; false when they were estimated locally because the provider
    /// returned no usage. Lets post-job actuals be honest about their source.
    #[serde(default)]
    pub usage_reported: bool,
}

/// A cost preview for extracting a batch of chunks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostEstimate {
    pub chunks: usize,
    pub estimated_input_tokens: u64,
    pub estimated_output_tokens: u64,
    /// 0.0 for unknown or self-hosted models.
    pub estimated_cost_usd: f64,
    pub model: String,
    pub pricing_known: bool,
}

/// A provider adapter that extracts entities and edges from a chunk.
#[async_trait::async_trait]
pub trait ExtractionAdapter: Send + Sync {
    /// Extract entities, edges, and proposals from a single chunk.
    async fn extract(
        &self,
        chunk: &ChunkContent,
        ontology: &Ontology,
        prompt_version: PromptVersion,
    ) -> Result<ExtractionResult, ExtractionError>;

    /// Estimate the cost of extracting the given chunks without calling the LLM.
    fn estimate_cost(&self, chunks: &[ChunkContent]) -> CostEstimate;

    /// Stable provider id (for example derived from the base url host).
    fn provider_id(&self) -> &str;

    /// The model id this adapter targets.
    fn model_id(&self) -> &str;
}
