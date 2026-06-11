// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Per-chunk extraction processing: cache lookup, ontology validation, the
//! re-extraction replace policy, reject-list filtering, entity dedup, and the
//! actual node and edge writes through the graph-write seam.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use vf_graph::{now_millis, Node, NodeId, NodeSource, Provenance, TypedEdge as Edge};

use crate::adapter::{ChunkContent, ExtractionResult, PromptVersion};
use crate::error::ExtractionError;
use crate::ontology::{Ontology, OntologyProposal};
use crate::writer::GraphWriter;

/// The reference name an edge uses to point at the chunk's content node.
const CHUNK_REF: &str = "@chunk";
/// Default cosine-similarity threshold for embedding-based entity dedup.
pub const DEFAULT_DEDUP_THRESHOLD: f32 = 0.92;
/// Upper bound on how many per-chunk failure samples a job status retains. The
/// `failed_chunks` counter still counts every failure; only the stored sample is
/// bounded so a million failing chunks cannot balloon a job's status in memory.
pub const MAX_CHUNK_ERRORS: usize = 100;

/// Lifecycle state of an extraction job.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Completed,
    CompletedWithErrors,
    Failed,
    Cancelled,
}

/// A single chunk's failure, recorded as a bounded sample on the job status so
/// callers can surface which chunks failed and why without losing the whole job.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkError {
    pub doc_id: String,
    pub chunk_id: u64,
    pub error: String,
}

/// A snapshot of an extraction job's progress.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStatus {
    pub job_id: String,
    pub collection: String,
    pub state: JobState,
    pub total_chunks: usize,
    pub processed_chunks: usize,
    pub entities_written: usize,
    pub edges_written: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    /// Count of all chunks that failed (unbounded count, not the sample size).
    pub failed_chunks: usize,
    /// A bounded sample of per-chunk failures, capped at `MAX_CHUNK_ERRORS`.
    pub chunk_errors: Vec<ChunkError>,
    pub error: Option<String>,
    /// Actual input tokens consumed by the LLM calls this job ran (Area 4).
    /// Cache hits contribute zero tokens. Accumulated as the job runs.
    pub actual_input_tokens: u64,
    /// Actual output tokens produced by the LLM calls this job ran (Area 4).
    pub actual_output_tokens: u64,
    /// Actual cost in USD priced via the pricing table from the actual tokens.
    /// Zero when the model's pricing is unknown (never fabricated).
    pub actual_cost_usd: f64,
    /// True only while every priced LLM call reported usage itself; flips false
    /// the first time a call's tokens had to be estimated, so callers know the
    /// actuals are not fully provider-reported.
    pub usage_provider_reported: bool,
}

impl JobStatus {
    /// A fresh queued status for a job over `total_chunks` chunks.
    pub fn queued(job_id: String, collection: String, total_chunks: usize) -> Self {
        Self {
            job_id,
            collection,
            state: JobState::Queued,
            total_chunks,
            processed_chunks: 0,
            entities_written: 0,
            edges_written: 0,
            cache_hits: 0,
            cache_misses: 0,
            failed_chunks: 0,
            chunk_errors: Vec::new(),
            error: None,
            actual_input_tokens: 0,
            actual_output_tokens: 0,
            actual_cost_usd: 0.0,
            // Starts true; the absence of any priced call leaves it true, and the
            // first estimated call flips it false (see record_actual_usage).
            usage_provider_reported: true,
        }
    }
}

/// A reject rule: a pattern of auto-edges that must never be re-created from a
/// given source. Empty fields are wildcards; a rule matches an edge when every
/// set field matches.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RejectRule {
    pub source_doc: Option<String>,
    pub source_chunk_id: Option<u64>,
    pub edge_type: String,
    pub source_name: Option<String>,
    pub target_name: Option<String>,
}

/// The per-chunk outcome reported back to the manager.
#[derive(Clone, Debug, Default)]
pub struct ChunkOutcome {
    pub entities_written: usize,
    pub edges_written: usize,
    pub cache_hit: bool,
    /// Newly emitted pending proposals for unknown-but-suggested types.
    pub new_proposals: Vec<OntologyProposal>,
}

/// Property key holding the deterministic normalized entity name (ADR-015).
pub const NAME_NORM_KEY: &str = "name_norm";

/// Property key holding the list of merged surface forms for an entity node when
/// fuzzy resolution (ADR-020) collapses a variant onto a canonical node. Preserves
/// provenance: the canonical `name` is the display name, `aliases` records the
/// other surface forms that resolved to it.
pub const ALIASES_KEY: &str = "aliases";

/// Canonical ADR-015 entity-name normalizer. Relocated to `vf-core` so the graph
/// store index can compute it without depending on this crate. Re-exported here
/// so `vf_extraction::normalize_entity_name` stays a stable path for all callers.
pub use vf_core::text_norm::normalize_entity_name;

/// Cosine similarity of two equal-length vectors. Returns 0.0 for a length
/// mismatch or a zero-magnitude vector.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// True when a reject rule matches an edge being considered for this chunk.
fn rule_matches(
    rule: &RejectRule,
    chunk: &ChunkContent,
    edge_type: &str,
    source_name: Option<&str>,
    target_name: Option<&str>,
) -> bool {
    if rule.edge_type != edge_type {
        return false;
    }
    if let Some(doc) = &rule.source_doc {
        if doc != &chunk.doc_id {
            return false;
        }
    }
    if let Some(cid) = rule.source_chunk_id {
        if cid != chunk.chunk_id {
            return false;
        }
    }
    if let Some(sn) = &rule.source_name {
        match source_name {
            Some(name) if name.eq_ignore_ascii_case(sn) => {}
            _ => return false,
        }
    }
    if let Some(tn) = &rule.target_name {
        match target_name {
            Some(name) if name.eq_ignore_ascii_case(tn) => {}
            _ => return false,
        }
    }
    true
}

/// Process a single chunk end to end and report what was written.
///
/// All graph reads and writes go through `writer`; this function never touches
/// the server's locks. The caller (the worker loop) resolves the extraction
/// result (from cache or a fresh LLM call) and the cache-hit flag outside this
/// function so the cache lock never crosses the LLM await. This function is
/// synchronous: it performs ontology validation, re-extraction, reject-list
/// filtering, entity dedup, and the graph writes only. The caller also supplies
/// the active merged ontology and the per-collection reject list.
#[allow(clippy::too_many_arguments)]
pub fn process_chunk(
    result: &ExtractionResult,
    cache_hit: bool,
    ontology: &Ontology,
    writer: &dyn GraphWriter,
    reject_list: &[RejectRule],
    model: &str,
    prompt_version: PromptVersion,
    chunk: &ChunkContent,
) -> Result<ChunkOutcome, ExtractionError> {
    let mut outcome = ChunkOutcome {
        cache_hit,
        ..Default::default()
    };

    // 2. Re-extraction: drop prior auto-edges from this same source that are
    //    neither manual nor verified, before writing new ones.
    let prior = writer.edges_from_chunk(&chunk.doc_id, chunk.chunk_id);
    for edge in prior {
        if edge.is_manual || edge.verified {
            continue;
        }
        let lsn = writer.next_lsn();
        writer.delete_edge(edge.id, lsn)?;
    }

    // 3. Validate entities against the ontology and dedup them to node ids.
    //    An unknown label that the LLM also surfaced as a proposal becomes a
    //    pending proposal and the entity is skipped; an unknown label with no
    //    matching proposal is skipped with a warning.
    let mut ref_to_node: HashMap<String, ResolvedEntity> = HashMap::new();

    emit_unknown_proposals(result, ontology, chunk, &mut outcome);

    for entity in &result.entities {
        if !ontology.has_label(&entity.label) {
            // Skipped: either turned into a proposal above or simply unknown.
            if !result
                .ontology_proposals
                .iter()
                .any(|p| p.name == entity.label)
            {
                tracing::warn!(
                    label = %entity.label,
                    doc = %chunk.doc_id,
                    chunk = chunk.chunk_id,
                    "skipping entity with unknown label and no proposal"
                );
            }
            continue;
        }

        let node_id = resolve_entity_node(writer, entity, ontology.entity_resolution)?;
        outcome.entities_written += 1;
        ref_to_node.insert(
            entity.ref_id.clone(),
            ResolvedEntity {
                node_id,
                name: entity.name.clone(),
            },
        );
    }

    // 4. Ensure the chunk's content node exists for any "@chunk" edge ref.
    let chunk_node_id = NodeId(chunk.chunk_id);
    let needs_chunk_node = result
        .edges
        .iter()
        .any(|e| e.source_ref == CHUNK_REF || e.target_ref == CHUNK_REF);
    if needs_chunk_node {
        ensure_chunk_node(writer, chunk, chunk_node_id)?;
    }

    // 5. Write edges: resolve refs, filter rejects, validate edge type.
    for edge in &result.edges {
        if !ontology.has_edge_type(&edge.edge_type) {
            if !result
                .ontology_proposals
                .iter()
                .any(|p| p.name == edge.edge_type)
            {
                tracing::warn!(
                    edge_type = %edge.edge_type,
                    doc = %chunk.doc_id,
                    chunk = chunk.chunk_id,
                    "skipping edge with unknown type and no proposal"
                );
            }
            continue;
        }

        let source = resolve_ref(&edge.source_ref, chunk_node_id, &ref_to_node);
        let target = resolve_ref(&edge.target_ref, chunk_node_id, &ref_to_node);
        let (source_id, source_name) = match source {
            Some(r) => r,
            None => continue,
        };
        let (target_id, target_name) = match target {
            Some(r) => r,
            None => continue,
        };

        // Reject-list filter.
        let rejected = reject_list.iter().any(|rule| {
            rule_matches(
                rule,
                chunk,
                &edge.edge_type,
                source_name.as_deref(),
                target_name.as_deref(),
            )
        });
        if rejected {
            continue;
        }

        // Dedup-skip: never write a duplicate of a surviving manual/verified edge.
        if writer.manual_or_verified_edge_exists(source_id, target_id, &edge.edge_type) {
            continue;
        }

        let extracted_at = now_millis();
        let edge_id = writer.alloc_edge_id();
        let mut graph_edge = Edge {
            id: edge_id,
            source: source_id,
            target: target_id,
            edge_type: edge.edge_type.as_str().into(),
            properties: json_map_to_hashmap(&edge.properties),
            provenance: Provenance {
                source_doc: Some(chunk.doc_id.clone()),
                source_chunk_id: Some(chunk.chunk_id),
                source_text: Some(edge.source_text.clone()),
                model: Some(model.to_string()),
                prompt_version: Some(prompt_version.as_str().to_string()),
                extracted_at: Some(extracted_at),
                cache_hit_at: if cache_hit { Some(extracted_at) } else { None },
            },
            confidence: edge.confidence,
            verified: false,
            is_manual: false,
            created_at: extracted_at,
            valid_from: None,
            valid_until: None,
            temporal_context: None,
            history: Vec::new(),
        };
        // Belt-and-suspenders: never let an extracted edge claim manual/verified.
        graph_edge.is_manual = false;
        graph_edge.verified = false;
        // One audit entry recording the extraction.
        graph_edge.record_audit("extracted", None, extracted_at);

        let lsn = writer.next_lsn();
        writer.put_edge(graph_edge, lsn)?;
        outcome.edges_written += 1;
    }

    Ok(outcome)
}

/// A deduped entity: its node id and the canonical name (for reject matching).
struct ResolvedEntity {
    node_id: NodeId,
    name: String,
}

/// Resolve an edge ref to a node id and optional name. `@chunk` resolves to the
/// content node (which has no entity name). An unknown ref resolves to `None`,
/// which causes the edge to be skipped.
fn resolve_ref(
    reference: &str,
    chunk_node_id: NodeId,
    ref_to_node: &HashMap<String, ResolvedEntity>,
) -> Option<(NodeId, Option<String>)> {
    if reference == CHUNK_REF {
        return Some((chunk_node_id, None));
    }
    ref_to_node
        .get(reference)
        .map(|r| (r.node_id, Some(r.name.clone())))
}

/// Turn unknown labels and edge types that the LLM surfaced as raw proposals
/// into pending ontology proposals recorded on the outcome.
fn emit_unknown_proposals(
    result: &ExtractionResult,
    ontology: &Ontology,
    chunk: &ChunkContent,
    outcome: &mut ChunkOutcome,
) {
    use crate::ontology::ProposalKind;
    for raw in &result.ontology_proposals {
        let already_known = match raw.kind {
            ProposalKind::EntityLabel => ontology.has_label(&raw.name),
            ProposalKind::EdgeType => ontology.has_edge_type(&raw.name),
        };
        if already_known {
            continue;
        }
        let proposal =
            OntologyProposal::pending(raw, Some(chunk.doc_id.clone()), Some(chunk.chunk_id));
        outcome.new_proposals.push(proposal);
    }
}

/// Find or allocate the node for a proposed entity, applying dedup.
///
/// `resolution` selects the matching strategy (ADR-020). `Normalized` (the
/// default) uses the ADR-015 exact normalized match and is byte-identical to the
/// prior behavior. `Fuzzy` additionally applies the conservative deterministic
/// resolver via `find_entity_fuzzy` (which falls back to the exact match when no
/// fuzzy rule is confident), so a fresh node is still allocated when in doubt.
fn resolve_entity_node(
    writer: &dyn GraphWriter,
    entity: &crate::adapter::ProposedEntity,
    resolution: crate::ontology::EntityResolution,
) -> Result<NodeId, ExtractionError> {
    use crate::ontology::EntityResolution;
    // Label + name match. Normalized mode is the ADR-015 exact normalized match;
    // Fuzzy mode adds the conservative deterministic rules (ADR-020). Both are
    // label-scoped inside the writer.
    let existing = match resolution {
        EntityResolution::Normalized => writer.find_entity(&entity.label, &entity.name),
        EntityResolution::Fuzzy => writer.find_entity_fuzzy(&entity.label, &entity.name),
    };
    if let Some(existing) = existing {
        return Ok(existing);
    }

    // Allocate a fresh entity node carrying its name in the property bag.
    let id = writer.alloc_node_id();
    let mut node = Node::entity(id, entity.label.clone(), NodeSource::Extracted);
    node.properties = json_map_to_hashmap(&entity.properties);
    node.properties.insert(
        "name".to_string(),
        serde_json::Value::String(entity.name.clone()),
    );
    // ADR-015. Store the normalized name once so find_entity matches on it
    // without re-normalizing every stored node on every lookup.
    node.properties.insert(
        NAME_NORM_KEY.to_string(),
        serde_json::Value::String(normalize_entity_name(&entity.name)),
    );
    let lsn = writer.next_lsn();
    writer.put_node(node, lsn)?;
    Ok(id)
}

/// Ensure the chunk's content node exists, creating it with text and doc_id if
/// it is absent. Existing content nodes are left untouched.
fn ensure_chunk_node(
    writer: &dyn GraphWriter,
    chunk: &ChunkContent,
    chunk_node_id: NodeId,
) -> Result<(), ExtractionError> {
    if writer.node_exists(chunk_node_id) {
        return Ok(());
    }
    // Re-check via get_node in case node_exists is a cheaper probe; either way
    // we only create when truly absent.
    if writer.get_node(chunk_node_id).is_some() {
        return Ok(());
    }
    let mut node = Node::content(chunk_node_id, chunk.embedding.clone(), NodeSource::Extracted);
    node.properties.insert(
        "text".to_string(),
        serde_json::Value::String(chunk.text.clone()),
    );
    node.properties.insert(
        "doc_id".to_string(),
        serde_json::Value::String(chunk.doc_id.clone()),
    );
    let lsn = writer.next_lsn();
    writer.put_node(node, lsn)
}

/// Convert a serde_json map into the graph's `HashMap` property bag.
fn json_map_to_hashmap(
    map: &serde_json::Map<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}
