// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Ontology model: entity labels, edge types, the built-in templates, the merge
//! rule, and the proposal records the LLM emits for new labels or edge types.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A named entity label with a human-readable description.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntityLabelDef {
    pub label: String,
    pub description: String,
}

impl EntityLabelDef {
    /// Build a label def from string-like parts.
    pub fn new(label: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: description.into(),
        }
    }
}

/// A named edge type with a description and optional endpoint label constraints.
/// Empty `source_labels` or `target_labels` means any label is allowed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EdgeTypeDef {
    pub edge_type: String,
    pub description: String,
    pub source_labels: Vec<String>,
    pub target_labels: Vec<String>,
}

impl EdgeTypeDef {
    /// Build an edge-type def. Endpoint constraints default to "any".
    pub fn new(
        edge_type: impl Into<String>,
        description: impl Into<String>,
        source_labels: Vec<String>,
        target_labels: Vec<String>,
    ) -> Self {
        Self {
            edge_type: edge_type.into(),
            description: description.into(),
            source_labels,
            target_labels,
        }
    }
}

/// The active ontology for a collection: the labels and edge types extraction is
/// allowed to produce without an approval step, plus the optional per-collection
/// prompt customization.
///
/// `system_prompt` and `extra_guidance` are extraction *config*, not part of the
/// allowed-types contract: when set on the user extension they customize the
/// system message (see `prompt::build_system_prompt`) but never the JSON shape or
/// the allowed-types listing, both of which stay machine-enforced. Both carry
/// `#[serde(default)]` so ontology sidecars written before this field existed
/// still deserialize to `None`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Ontology {
    pub entity_labels: Vec<EntityLabelDef>,
    pub edge_types: Vec<EdgeTypeDef>,
    /// Optional full override of the default system-prompt framing. `None` keeps
    /// the baked-in `default_framing()`.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Optional domain guidance appended on top of whichever framing is in
    /// effect (default or override). `None` appends nothing.
    #[serde(default)]
    pub extra_guidance: Option<String>,
    /// ADR-012. Opt-in convenience: when true, the merge guarantees a passage-to-
    /// entity `mentions` edge type referencing `@chunk` is present, so GraphRAG
    /// works without the user hand-authoring the `@chunk` relation. Defaults to
    /// false so collections that do not opt in see byte-identical behavior.
    #[serde(default)]
    pub link_passages: bool,
    /// ADR-020. Entity-resolution mode for dedup matching. Defaults to
    /// `Normalized` (the ADR-015 behavior) so existing collections are
    /// byte-identical; `Fuzzy` is opt-in and merges alias / abbreviation / typo
    /// variants conservatively. `#[serde(default)]` keeps sidecars written before
    /// this field existed deserializing to `Normalized`.
    #[serde(default)]
    pub entity_resolution: EntityResolution,
}

/// The passage-to-entity edge type injected when `link_passages` is set.
/// References the special `@chunk` content node via the LLM's `source_ref`.
pub const PASSAGE_LINK_EDGE_TYPE: &str = "mentions";

/// How a new entity name is matched against existing entity nodes of the same
/// label when deduping (ADR-020). Label-scoped in all modes.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EntityResolution {
    /// ADR-015 deterministic normalized exact match (trim, NFKC, lowercase,
    /// quote / trailing-punctuation strip, whitespace collapse). The default, so
    /// collections that do not opt in behave byte-identically to before.
    #[default]
    Normalized,
    /// Opt-in conservative deterministic fuzzy resolution: in addition to the
    /// normalized exact match, merge initials / abbreviations, acronym /
    /// expansion pairs, bounded typos, and strict title-prefixed supersets of the
    /// same head entity. Deliberately conservative to avoid over-merging distinct
    /// entities (see ADR-020). Still label-scoped and fully deterministic.
    Fuzzy,
}

impl Ontology {
    /// True when `l` is a known entity label.
    pub fn has_label(&self, l: &str) -> bool {
        self.entity_labels.iter().any(|e| e.label == l)
    }

    /// True when `t` is a known edge type.
    pub fn has_edge_type(&self, t: &str) -> bool {
        self.edge_types.iter().any(|e| e.edge_type == t)
    }

    /// Union of `base` and `ext` plus every approved proposal. On a name clash the
    /// `ext` definition wins over `base`, and an approved proposal adds a label or
    /// edge type that was not already present.
    ///
    /// The prompt-customization fields (`system_prompt`, `extra_guidance`) are
    /// config the user sets on the extension, so they are carried over from `ext`;
    /// `base` (a built-in template) never carries them. Approved proposals only
    /// widen labels and edge types and never touch them.
    pub fn merge(base: &Ontology, ext: &Ontology, approved: &[OntologyProposal]) -> Ontology {
        let mut entity_labels: Vec<EntityLabelDef> = base.entity_labels.clone();
        let mut edge_types: Vec<EdgeTypeDef> = base.edge_types.clone();

        // Extension wins on a name clash.
        for e in &ext.entity_labels {
            match entity_labels.iter_mut().find(|x| x.label == e.label) {
                Some(slot) => *slot = e.clone(),
                None => entity_labels.push(e.clone()),
            }
        }
        for t in &ext.edge_types {
            match edge_types.iter_mut().find(|x| x.edge_type == t.edge_type) {
                Some(slot) => *slot = t.clone(),
                None => edge_types.push(t.clone()),
            }
        }

        // Approved proposals add anything still missing.
        for p in approved {
            if p.status != ProposalStatus::Approved {
                continue;
            }
            match p.kind {
                ProposalKind::EntityLabel => {
                    if !entity_labels.iter().any(|x| x.label == p.name) {
                        entity_labels.push(EntityLabelDef::new(
                            p.name.clone(),
                            p.description.clone(),
                        ));
                    }
                }
                ProposalKind::EdgeType => {
                    if !edge_types.iter().any(|x| x.edge_type == p.name) {
                        edge_types.push(EdgeTypeDef::new(
                            p.name.clone(),
                            p.description.clone(),
                            Vec::new(),
                            Vec::new(),
                        ));
                    }
                }
            }
        }

        // ADR-012. When link_passages is opted in, guarantee a passage-to-entity
        // edge type referencing @chunk is present so GraphRAG works out of the
        // box. Idempotent and case-insensitive so an existing MENTIONS template
        // edge type is reused rather than duplicated.
        if ext.link_passages
            && !edge_types
                .iter()
                .any(|e| e.edge_type.eq_ignore_ascii_case(PASSAGE_LINK_EDGE_TYPE))
        {
            edge_types.push(EdgeTypeDef::new(
                PASSAGE_LINK_EDGE_TYPE,
                "Links a passage (@chunk) to an entity it mentions",
                Vec::new(),
                Vec::new(),
            ));
        }

        Ontology {
            entity_labels,
            edge_types,
            system_prompt: ext.system_prompt.clone(),
            extra_guidance: ext.extra_guidance.clone(),
            link_passages: ext.link_passages,
            // Resolution mode is user config carried from the extension; a
            // built-in template never sets it, so the default stays Normalized.
            entity_resolution: ext.entity_resolution,
        }
    }
}

// ── Built-in templates ───────────────────────────────────────────────

/// Research-papers template.
pub fn research_papers() -> Ontology {
    Ontology {
        entity_labels: vec![
            EntityLabelDef::new("Paper", "A research paper or article"),
            EntityLabelDef::new("Author", "A person who authored a paper"),
            EntityLabelDef::new("Institution", "A university, lab, or organization"),
            EntityLabelDef::new("Date", "A calendar date such as a publication date"),
            EntityLabelDef::new("Method", "A method, technique, or algorithm"),
        ],
        edge_types: vec![
            EdgeTypeDef::new(
                "AUTHORED_BY",
                "Links a paper to one of its authors",
                vec!["Paper".into()],
                vec!["Author".into()],
            ),
            EdgeTypeDef::new(
                "AFFILIATED_WITH",
                "Links an author to an institution",
                vec!["Author".into()],
                vec!["Institution".into()],
            ),
            EdgeTypeDef::new(
                "CITES",
                "Links a paper to another paper it cites",
                vec!["Paper".into()],
                vec!["Paper".into()],
            ),
            EdgeTypeDef::new(
                "USES_METHOD",
                "Links a paper to a method it uses",
                vec!["Paper".into()],
                vec!["Method".into()],
            ),
            EdgeTypeDef::new(
                "FUNDED_BY",
                "Links a paper to a funding institution",
                vec!["Paper".into()],
                vec!["Institution".into()],
            ),
        ],
        system_prompt: None,
        extra_guidance: None,
        link_passages: false,
        entity_resolution: EntityResolution::Normalized,
    }
}

/// Legal-contracts template.
pub fn legal_contracts() -> Ontology {
    Ontology {
        entity_labels: vec![
            EntityLabelDef::new("Party", "A signatory or party to an agreement"),
            EntityLabelDef::new("Document", "A contract, agreement, or legal document"),
            EntityLabelDef::new("Jurisdiction", "A governing jurisdiction or venue"),
            EntityLabelDef::new("Date", "A calendar date such as an effective date"),
            EntityLabelDef::new("Provision", "A clause, term, or provision"),
        ],
        edge_types: vec![
            EdgeTypeDef::new(
                "PARTY_TO",
                "Links a party to a document it is a party to",
                vec!["Party".into()],
                vec!["Document".into()],
            ),
            EdgeTypeDef::new(
                "GOVERNED_BY",
                "Links a document to its governing jurisdiction",
                vec!["Document".into()],
                vec!["Jurisdiction".into()],
            ),
            EdgeTypeDef::new(
                "REFERENCES",
                "Links a document to another document it references",
                vec!["Document".into()],
                vec!["Document".into()],
            ),
            EdgeTypeDef::new(
                "AMENDS",
                "Links a document to a document it amends",
                vec!["Document".into()],
                vec!["Document".into()],
            ),
            EdgeTypeDef::new(
                "EFFECTIVE_FROM",
                "Links a document to its effective date",
                vec!["Document".into()],
                vec!["Date".into()],
            ),
        ],
        system_prompt: None,
        extra_guidance: None,
        link_passages: false,
        entity_resolution: EntityResolution::Normalized,
    }
}

/// Ecommerce-catalog template.
pub fn ecommerce_catalog() -> Ontology {
    Ontology {
        entity_labels: vec![
            EntityLabelDef::new("Product", "A purchasable product"),
            EntityLabelDef::new("Brand", "A product brand or manufacturer"),
            EntityLabelDef::new("Category", "A product category"),
            EntityLabelDef::new("Customer", "A person who buys products"),
        ],
        edge_types: vec![
            EdgeTypeDef::new(
                "MADE_BY",
                "Links a product to the brand that makes it",
                vec!["Product".into()],
                vec!["Brand".into()],
            ),
            EdgeTypeDef::new(
                "BELONGS_TO",
                "Links a product to a category it belongs to",
                vec!["Product".into()],
                vec!["Category".into()],
            ),
            EdgeTypeDef::new(
                "BOUGHT",
                "Links a customer to a product they bought",
                vec!["Customer".into()],
                vec!["Product".into()],
            ),
        ],
        system_prompt: None,
        extra_guidance: None,
        link_passages: false,
        entity_resolution: EntityResolution::Normalized,
    }
}

/// Support-tickets template.
pub fn support_tickets() -> Ontology {
    Ontology {
        entity_labels: vec![
            EntityLabelDef::new("Ticket", "A support ticket or case"),
            EntityLabelDef::new("Customer", "A customer who raised a ticket"),
            EntityLabelDef::new("Product", "A product a ticket is about"),
            EntityLabelDef::new("Issue", "A problem or issue category"),
            EntityLabelDef::new("Agent", "A support agent who handled a ticket"),
        ],
        edge_types: vec![
            EdgeTypeDef::new(
                "REPORTED_BY",
                "Links a ticket to the customer who reported it",
                vec!["Ticket".into()],
                vec!["Customer".into()],
            ),
            EdgeTypeDef::new(
                "ABOUT_PRODUCT",
                "Links a ticket to the product it is about",
                vec!["Ticket".into()],
                vec!["Product".into()],
            ),
            EdgeTypeDef::new(
                "HAS_ISSUE",
                "Links a ticket to the issue it concerns",
                vec!["Ticket".into()],
                vec!["Issue".into()],
            ),
            EdgeTypeDef::new(
                "HANDLED_BY",
                "Links a ticket to the agent who handled it",
                vec!["Ticket".into()],
                vec!["Agent".into()],
            ),
        ],
        system_prompt: None,
        extra_guidance: None,
        link_passages: false,
        entity_resolution: EntityResolution::Normalized,
    }
}

/// Internal-docs template.
pub fn internal_docs() -> Ontology {
    Ontology {
        entity_labels: vec![
            EntityLabelDef::new("Document", "An internal document or note"),
            EntityLabelDef::new("Person", "A person mentioned or involved"),
            EntityLabelDef::new("Project", "A project or initiative"),
            EntityLabelDef::new("Decision", "A decision recorded in a document"),
        ],
        edge_types: vec![
            EdgeTypeDef::new(
                "AUTHORED_BY",
                "Links a document to its author",
                vec!["Document".into()],
                vec!["Person".into()],
            ),
            EdgeTypeDef::new(
                "MENTIONS",
                "Links a document to a person or project it mentions",
                vec!["Document".into()],
                Vec::new(),
            ),
            EdgeTypeDef::new(
                "DECIDED_ON",
                "Links a document to a decision it records",
                vec!["Document".into()],
                vec!["Decision".into()],
            ),
            EdgeTypeDef::new(
                "REFERENCES",
                "Links a document to another document it references",
                vec!["Document".into()],
                vec!["Document".into()],
            ),
        ],
        system_prompt: None,
        extra_guidance: None,
        link_passages: false,
        entity_resolution: EntityResolution::Normalized,
    }
}

/// Resolve a template by its kebab-case name. Returns `None` for unknown names.
pub fn template_by_name(name: &str) -> Option<Ontology> {
    match name {
        "research-papers" => Some(research_papers()),
        "legal-contracts" => Some(legal_contracts()),
        "ecommerce-catalog" => Some(ecommerce_catalog()),
        "support-tickets" => Some(support_tickets()),
        "internal-docs" => Some(internal_docs()),
        _ => None,
    }
}

// ── Proposals ────────────────────────────────────────────────────────

/// A new label or edge type the LLM suggested inside an extraction result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawProposal {
    pub kind: ProposalKind,
    pub name: String,
    pub description: String,
    pub examples: Vec<String>,
}

/// Whether a proposal targets an entity label or an edge type.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalKind {
    EntityLabel,
    EdgeType,
}

impl ProposalKind {
    /// Stable string tag used in deterministic id derivation.
    pub fn tag(&self) -> &'static str {
        match self {
            ProposalKind::EntityLabel => "entity_label",
            ProposalKind::EdgeType => "edge_type",
        }
    }
}

/// A proposal stored for review, awaiting approval or rejection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OntologyProposal {
    pub id: String,
    pub kind: ProposalKind,
    pub name: String,
    pub description: String,
    pub examples: Vec<String>,
    pub status: ProposalStatus,
    pub source_doc: Option<String>,
    pub source_chunk_id: Option<u64>,
}

impl OntologyProposal {
    /// Build a pending proposal from a raw LLM proposal plus its source chunk.
    /// The id is derived deterministically from (kind, name, source) so this
    /// module set carries no runtime entropy dependency.
    pub fn pending(
        raw: &RawProposal,
        source_doc: Option<String>,
        source_chunk_id: Option<u64>,
    ) -> Self {
        let source = match (&source_doc, source_chunk_id) {
            (Some(doc), Some(chunk)) => format!("{}:{}", doc, chunk),
            (Some(doc), None) => doc.clone(),
            _ => String::new(),
        };
        let id = derive_proposal_id(raw.kind, &raw.name, &source);
        Self {
            id,
            kind: raw.kind,
            name: raw.name.clone(),
            description: raw.description.clone(),
            examples: raw.examples.clone(),
            status: ProposalStatus::Pending,
            source_doc,
            source_chunk_id,
        }
    }
}

/// Lifecycle state of a stored proposal.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
}

/// Derive a stable hex id for a proposal from its kind, name, and source string.
/// Deterministic so repeated extraction of the same chunk does not create a new
/// proposal id for the same suggestion.
pub fn derive_proposal_id(kind: ProposalKind, name: &str, source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(kind.tag().as_bytes());
    hasher.update(b"\0");
    hasher.update(name.as_bytes());
    hasher.update(b"\0");
    hasher.update(source.as_bytes());
    let digest = hasher.finalize();
    hex_encode(&digest)
}

/// Lowercase hex encoding of a byte slice.
fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((b & 0x0f) as u32, 16).unwrap_or('0'));
    }
    out
}
