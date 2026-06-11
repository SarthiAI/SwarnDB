// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! The extraction prompt: a baked-in version, the system prompt, and the builder
//! that embeds the allowed labels and edge types plus the required JSON schema.

use std::fmt::Write as _;

use crate::adapter::PromptVersion;
use crate::ontology::Ontology;

/// The baked-in prompt version. Bumping this invalidates the extraction cache
/// for every collection. A per-collection custom prompt is folded into the cache
/// key instead (see `cache::cache_key`), so it does not need a version bump.
pub const PROMPT_VERSION: PromptVersion = PromptVersion("1.0.0");

/// The default role and task framing. This is the part a per-collection
/// `system_prompt` override replaces; the machine contract below is always kept.
pub fn default_framing() -> &'static str {
    "You are an information extraction engine. Given a text passage and an \
ontology of allowed entity labels and edge types, extract the entities and the \
typed relationships between them."
}

/// ADR-017. The passage-linking guidance injected when `link_passages` is set.
/// Ported verbatim in intent from the proven `examples/realtest/run.py` guidance
/// (the text that produced ~1095 passage->entity edges and the real GraphRAG
/// lift). ADR-012 only injects the `mentions` edge TYPE into the ontology; this
/// is the missing half that actually instructs the LLM to PRODUCE the edges.
/// It references the engine's `@chunk` token, which the pipeline resolves to the
/// chunk's content node, so a `@chunk -> entity` edge becomes a real
/// passage->entity link. Kept generic (no answer, no domain assumptions) so it
/// composes with any ontology and any user `extra_guidance`.
pub fn passage_link_guidance() -> &'static str {
    "For EVERY chunk, also emit a 'mentions' edge from the passage to EACH \
entity the passage names. The passage is referenced by the special ref \
'@chunk', so each 'mentions' edge has source_ref '@chunk' and target_ref the \
entity's ref_id. Create a 'mentions' edge for every entity you extract. Prefer \
canonical entity names (the same person, place, or thing must use the same name \
across passages) so the graph connects."
}

/// The non-negotiable machine contract appended to every system prompt, default
/// or overridden. It pins the strict-JSON output, the use-only-allowed-types
/// rule (else propose), the cite-span-plus-confidence rule, and the no-invention
/// rule, so a custom prompt can never make a response fail to parse.
pub fn contract_footer() -> &'static str {
    "Respond with a single strict JSON object of the given shape and nothing \
else. Use only the allowed entity labels and edge types. Only when a recurring \
pattern clearly does not fit any allowed label or edge type, add it to \
ontology_proposals instead of forcing it into an existing type. Cite the exact \
source span for every entity and edge in its source_text field, and give each a \
confidence between 0 and 1. Do not invent facts that are not supported by the \
text."
}

/// The system prompt describing the task and the strict-JSON contract. Kept for
/// callers that want the default prompt with no per-collection customization; it
/// is exactly `build_system_prompt(None, None, false)`.
pub fn system_prompt() -> String {
    build_system_prompt(None, None, false)
}

/// Compose the system message for one extraction call.
///
/// The result is always [framing] then the always-on [contract footer], with an
/// optional domain-guidance block in between:
///   - framing is `custom_system` when set and non-empty, else `default_framing()`;
///   - when `extra_guidance` is set and non-empty, a clearly labeled
///     "Additional domain guidance:" block is appended on top of the framing;
///   - when `link_passages` is set (ADR-017), the passage-linking guidance is
///     appended on top of any domain guidance so the LLM is actually instructed
///     to emit `@chunk -> entity` "mentions" edges (ADR-012 only added the edge
///     type to the ontology, not the instruction to produce the edges);
///   - `contract_footer()` is always appended last so the machine contract holds
///     even under a full framing override.
///
/// Empty or whitespace-only custom inputs are treated as unset so a blank
/// override never silently drops the default framing or emits an empty guidance
/// block. When `link_passages` is false and no custom inputs are set, the output
/// is byte-identical to the pre-existing default prompt.
pub fn build_system_prompt(
    custom_system: Option<&str>,
    extra_guidance: Option<&str>,
    link_passages: bool,
) -> String {
    let framing = match custom_system.map(str::trim) {
        Some(s) if !s.is_empty() => s,
        _ => default_framing(),
    };
    let guidance = extra_guidance.map(str::trim).filter(|s| !s.is_empty());
    let link_block = if link_passages {
        Some(passage_link_guidance())
    } else {
        None
    };

    // framing + optional guidance block + optional passage-link block + contract
    // footer, separated by blank lines for readability in the request body.
    let footer = contract_footer();
    let mut out = String::with_capacity(
        framing.len()
            + guidance.map(|g| g.len() + 32).unwrap_or(0)
            + link_block.map(|g| g.len() + 32).unwrap_or(0)
            + footer.len()
            + 4,
    );
    out.push_str(framing);
    if let Some(g) = guidance {
        out.push_str("\n\nAdditional domain guidance:\n");
        out.push_str(g);
    }
    if let Some(g) = link_block {
        out.push_str("\n\nPassage linking (required):\n");
        out.push_str(g);
    }
    out.push_str("\n\n");
    out.push_str(footer);
    out
}

/// Build the user prompt for one chunk: it embeds the allowed labels and edge
/// types and the required JSON output schema. The schema mirrors the serde shape
/// of `ExtractionResult` so the response parses directly.
pub fn build_extraction_prompt(text: &str, ontology: &Ontology) -> String {
    let mut out = String::with_capacity(text.len() + 2048);

    out.push_str("Allowed entity labels:\n");
    if ontology.entity_labels.is_empty() {
        out.push_str("  (none)\n");
    } else {
        for l in &ontology.entity_labels {
            // Best-effort formatting; a write to a String cannot fail.
            let _ = writeln!(out, "  - {}: {}", l.label, l.description);
        }
    }

    out.push_str("\nAllowed edge types:\n");
    if ontology.edge_types.is_empty() {
        out.push_str("  (none)\n");
    } else {
        for t in &ontology.edge_types {
            let src = if t.source_labels.is_empty() {
                "any".to_string()
            } else {
                t.source_labels.join("|")
            };
            let tgt = if t.target_labels.is_empty() {
                "any".to_string()
            } else {
                t.target_labels.join("|")
            };
            let _ = writeln!(
                out,
                "  - {} ({} -> {}): {}",
                t.edge_type, src, tgt, t.description
            );
        }
    }

    out.push_str(OUTPUT_SCHEMA);

    out.push_str("\nText to extract from:\n\"\"\"\n");
    out.push_str(text);
    out.push_str("\n\"\"\"\n");

    out
}

/// The required JSON output schema. Field names mirror the serde shape of
/// `ExtractionResult`, `ProposedEntity`, `ProposedEdge`, and `RawProposal`.
const OUTPUT_SCHEMA: &str = "\nRespond with a JSON object of exactly this shape:\n\
{\n\
\x20 \"entities\": [\n\
\x20   {\n\
\x20     \"label\": \"<allowed entity label>\",\n\
\x20     \"name\": \"<canonical name>\",\n\
\x20     \"properties\": { },\n\
\x20     \"confidence\": 0.0,\n\
\x20     \"source_text\": \"<exact cited span>\",\n\
\x20     \"ref_id\": \"<local id, referenced by edges>\"\n\
\x20   }\n\
\x20 ],\n\
\x20 \"edges\": [\n\
\x20   {\n\
\x20     \"source_ref\": \"<ref_id of an entity, or @chunk>\",\n\
\x20     \"target_ref\": \"<ref_id of an entity, or @chunk>\",\n\
\x20     \"edge_type\": \"<allowed edge type>\",\n\
\x20     \"properties\": { },\n\
\x20     \"confidence\": 0.0,\n\
\x20     \"source_text\": \"<exact cited span>\"\n\
\x20   }\n\
\x20 ],\n\
\x20 \"ontology_proposals\": [\n\
\x20   {\n\
\x20     \"kind\": \"entity_label\" or \"edge_type\",\n\
\x20     \"name\": \"<proposed name>\",\n\
\x20     \"description\": \"<what it represents>\",\n\
\x20     \"examples\": [\"<example span>\"]\n\
\x20   }\n\
\x20 ]\n\
}\n\
Use an empty array for any section with no items. Output only this JSON object.\n";
