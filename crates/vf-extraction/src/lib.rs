// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! LLM-driven extraction layer for SwarnDB hybrid collections.
//!
//! This crate sits above the graph layer and below the server. It turns
//! client-supplied chunks of text into typed entities and edges via an LLM
//! adapter, with at-rest key sealing, token-cost estimation, an ontology of
//! allowed labels and edge types, and a result cache.

pub mod adapter;
pub mod cache;
pub mod config;
pub mod corpus;
pub mod cost;
pub mod crypto;
pub mod diff;
pub mod error;
pub mod manager;
pub mod ontology;
pub mod openai;
pub mod pipeline;
pub mod prompt;
pub mod resolution;
pub mod writer;

// ── Re-exports ───────────────────────────────────────────────────────

pub use adapter::{
    ChunkContent, CostEstimate, DocId, ExtractionAdapter, ExtractionResult, ProposedEdge,
    ProposedEntity, PromptVersion,
};
pub use cache::{
    cache_key, chunk_content_hash, custom_prompt_hash, normalize_text, ExtractionCache,
    GLOBAL_CACHE_DIR,
};
pub use corpus::{CorpusDocProgress, CorpusDocState, CorpusJobStatus};
pub use diff::{ChunkDiff, ChunkDiffAction, ReextractSummary};
pub use config::{LlmConfig, RedactedLlmConfig, SealedLlmConfig};
pub use cost::{HeuristicEstimator, ModelPrice, PricingTable, TokenEstimator};
pub use crypto::MasterKey;
pub use error::ExtractionError;
pub use ontology::{
    derive_proposal_id, ecommerce_catalog, internal_docs, legal_contracts, research_papers,
    support_tickets, template_by_name, EdgeTypeDef, EntityLabelDef, EntityResolution, Ontology,
    OntologyProposal, ProposalKind, ProposalStatus, RawProposal,
};
pub use resolution::fuzzy_name_match;
pub use prompt::{
    build_extraction_prompt, build_system_prompt, contract_footer, default_framing,
    passage_link_guidance, system_prompt, PROMPT_VERSION,
};
pub use manager::ExtractionManager;
pub use openai::OpenAICompatibleAdapter;
pub use pipeline::{
    cosine, normalize_entity_name, ChunkError, ChunkOutcome, JobState, JobStatus, RejectRule,
    ALIASES_KEY, DEFAULT_DEDUP_THRESHOLD, NAME_NORM_KEY,
};
pub use writer::GraphWriter;
