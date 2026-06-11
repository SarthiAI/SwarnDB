// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Shared bulk-edge parsing and application logic for manual edge import.
//! Both the gRPC and (later) REST surfaces parse a CSV or JSONL payload into
//! rows, then apply the rows against a typed graph store. Imported edges are
//! manual, unverified, and carry an "imported" audit entry. Parsing never
//! touches the store; application validates endpoint ids against a precomputed
//! valid-id set and known edge types per row, collecting per-row errors rather
//! than failing the whole batch.

use std::collections::HashSet;

use vf_graph::{now_millis, GraphStore, NodeId, Provenance, TypedEdge, TypedGraphStore};

/// A per-row failure with its 1-based row index and a human-readable message.
pub struct RowError {
    pub row: u64,
    pub message: String,
}

/// A parsed but not-yet-validated edge row.
pub struct ParsedEdgeRow {
    pub row: u64,
    pub source: u64,
    pub target: u64,
    pub edge_type: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub confidence: Option<f32>,
    // P17. Optional temporal validity window and context. Absent columns/keys
    // default to None (unbounded / no context), so legacy payloads are unchanged.
    pub valid_from: Option<u64>,
    pub valid_until: Option<u64>,
    pub temporal_context: Option<String>,
}

/// Payload format for a bulk import.
pub enum BulkFormat {
    Csv,
    Jsonl,
}

/// Parse a bulk-edge payload into rows plus parse-level row errors. Row indices
/// are 1-based over data rows (CSV header excluded, blank lines excluded).
pub fn parse_bulk_edges(format: BulkFormat, data: &str) -> (Vec<ParsedEdgeRow>, Vec<RowError>) {
    match format {
        BulkFormat::Csv => parse_csv(data),
        BulkFormat::Jsonl => parse_jsonl(data),
    }
}

// ── CSV ───────────────────────────────────────────────────────────────

fn parse_csv(data: &str) -> (Vec<ParsedEdgeRow>, Vec<RowError>) {
    let mut rows = Vec::new();
    let mut errors = Vec::new();

    let mut lines = data.lines();
    // First non-empty line is the header.
    let header_line = loop {
        match lines.next() {
            Some(l) if l.trim().is_empty() => continue,
            Some(l) => break Some(l),
            None => break None,
        }
    };
    let header = match header_line {
        Some(h) => split_csv_line(h),
        None => return (rows, errors),
    };

    // Map required and optional columns by name.
    let idx = |name: &str| header.iter().position(|c| c.trim() == name);
    let (i_source, i_target, i_edge_type) = match (idx("source"), idx("target"), idx("edge_type")) {
        (Some(s), Some(t), Some(e)) => (s, t, e),
        _ => {
            errors.push(RowError {
                row: 0,
                message: "csv header must include source, target, edge_type".to_string(),
            });
            return (rows, errors);
        }
    };
    let i_props = idx("properties");
    let i_conf = idx("confidence");
    // P17. Optional temporal columns; absent = None for every row.
    let i_valid_from = idx("valid_from");
    let i_valid_until = idx("valid_until");
    let i_temporal_context = idx("temporal_context");

    let mut row_no: u64 = 0;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        row_no += 1;
        let fields = split_csv_line(line);
        let get = |i: usize| fields.get(i).map(|s| s.trim()).unwrap_or("");

        let source = match get(i_source).parse::<u64>() {
            Ok(v) => v,
            Err(_) => {
                errors.push(RowError {
                    row: row_no,
                    message: format!("invalid source id '{}'", get(i_source)),
                });
                continue;
            }
        };
        let target = match get(i_target).parse::<u64>() {
            Ok(v) => v,
            Err(_) => {
                errors.push(RowError {
                    row: row_no,
                    message: format!("invalid target id '{}'", get(i_target)),
                });
                continue;
            }
        };
        let edge_type = get(i_edge_type).to_string();
        if edge_type.is_empty() {
            errors.push(RowError {
                row: row_no,
                message: "empty edge_type".to_string(),
            });
            continue;
        }

        let properties = match i_props {
            Some(i) => {
                let raw = fields.get(i).map(|s| s.trim()).unwrap_or("");
                match parse_props_object(raw) {
                    Ok(m) => m,
                    Err(msg) => {
                        errors.push(RowError { row: row_no, message: msg });
                        continue;
                    }
                }
            }
            None => serde_json::Map::new(),
        };

        let confidence = match i_conf {
            Some(i) => {
                let raw = fields.get(i).map(|s| s.trim()).unwrap_or("");
                if raw.is_empty() {
                    None
                } else {
                    match raw.parse::<f32>() {
                        Ok(v) => Some(v),
                        Err(_) => {
                            errors.push(RowError {
                                row: row_no,
                                message: format!("invalid confidence '{raw}'"),
                            });
                            continue;
                        }
                    }
                }
            }
            None => None,
        };

        // P17. Optional temporal columns. An empty cell = None; a malformed
        // unix-millis value is a per-row error (skip the row, don't fail the batch).
        let parse_opt_u64 = |i: Option<usize>| -> Result<Option<u64>, String> {
            match i {
                Some(i) => {
                    let raw = fields.get(i).map(|s| s.trim()).unwrap_or("");
                    if raw.is_empty() {
                        Ok(None)
                    } else {
                        raw.parse::<u64>()
                            .map(Some)
                            .map_err(|_| format!("invalid timestamp '{raw}'"))
                    }
                }
                None => Ok(None),
            }
        };
        let valid_from = match parse_opt_u64(i_valid_from) {
            Ok(v) => v,
            Err(msg) => {
                errors.push(RowError { row: row_no, message: msg });
                continue;
            }
        };
        let valid_until = match parse_opt_u64(i_valid_until) {
            Ok(v) => v,
            Err(msg) => {
                errors.push(RowError { row: row_no, message: msg });
                continue;
            }
        };
        let temporal_context = i_temporal_context.and_then(|i| {
            let raw = fields.get(i).map(|s| s.trim()).unwrap_or("");
            if raw.is_empty() {
                None
            } else {
                Some(raw.to_string())
            }
        });

        rows.push(ParsedEdgeRow {
            row: row_no,
            source,
            target,
            edge_type,
            properties,
            confidence,
            valid_from,
            valid_until,
            temporal_context,
        });
    }

    (rows, errors)
}

/// Split one CSV line on commas, honouring double-quoted fields that may
/// contain commas (e.g. a JSON properties object). Doubled quotes inside a
/// quoted field collapse to one literal quote.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == ',' {
            fields.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    fields.push(cur);
    fields
}

/// Parse a properties cell as a JSON object; empty cell yields an empty map.
fn parse_props_object(raw: &str) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    if raw.is_empty() {
        return Ok(serde_json::Map::new());
    }
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::Object(m)) => Ok(m),
        Ok(_) => Err("properties must be a JSON object".to_string()),
        Err(e) => Err(format!("invalid properties json: {e}")),
    }
}

// ── JSONL ─────────────────────────────────────────────────────────────

fn parse_jsonl(data: &str) -> (Vec<ParsedEdgeRow>, Vec<RowError>) {
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    let mut row_no: u64 = 0;

    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        row_no += 1;
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                errors.push(RowError {
                    row: row_no,
                    message: format!("invalid json: {e}"),
                });
                continue;
            }
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => {
                errors.push(RowError {
                    row: row_no,
                    message: "line is not a json object".to_string(),
                });
                continue;
            }
        };

        let source = match obj.get("source").and_then(|v| v.as_u64()) {
            Some(v) => v,
            None => {
                errors.push(RowError {
                    row: row_no,
                    message: "missing or invalid source".to_string(),
                });
                continue;
            }
        };
        let target = match obj.get("target").and_then(|v| v.as_u64()) {
            Some(v) => v,
            None => {
                errors.push(RowError {
                    row: row_no,
                    message: "missing or invalid target".to_string(),
                });
                continue;
            }
        };
        let edge_type = match obj.get("edge_type").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                errors.push(RowError {
                    row: row_no,
                    message: "missing or empty edge_type".to_string(),
                });
                continue;
            }
        };
        let properties = match obj.get("properties") {
            None | Some(serde_json::Value::Null) => serde_json::Map::new(),
            Some(serde_json::Value::Object(m)) => m.clone(),
            Some(_) => {
                errors.push(RowError {
                    row: row_no,
                    message: "properties must be a json object".to_string(),
                });
                continue;
            }
        };
        let confidence = obj.get("confidence").and_then(|v| v.as_f64()).map(|v| v as f32);

        // P17. Optional temporal keys; absent = None.
        let valid_from = obj.get("valid_from").and_then(|v| v.as_u64());
        let valid_until = obj.get("valid_until").and_then(|v| v.as_u64());
        let temporal_context = obj
            .get("temporal_context")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        rows.push(ParsedEdgeRow {
            row: row_no,
            source,
            target,
            edge_type,
            properties,
            confidence,
            valid_from,
            valid_until,
            temporal_context,
        });
    }

    (rows, errors)
}

// ── Apply ─────────────────────────────────────────────────────────────

/// Validate and write parsed rows into the store. Each row is checked for a
/// known edge type and for source and target ids that are valid endpoints. An
/// id is a valid endpoint if it is either a materialized typed node or an
/// existing plain vector id (under the NodeId == VectorId bridge a plain vector
/// is a valid virtual content node). Validity is decided by the precomputed
/// `valid_node_ids` set the caller supplies, so this function never probes node
/// existence itself. Failures become row errors and are skipped. Surviving rows
/// become manual, unverified edges with an "imported" audit entry and are
/// written in one batch. Returns the count imported plus all row errors.
pub fn apply_bulk_edges(
    store: &mut TypedGraphStore,
    known_edge_types: &HashSet<String>,
    valid_node_ids: &HashSet<u64>,
    rows: Vec<ParsedEdgeRow>,
    actor: Option<String>,
    lsn: u64,
) -> (u64, Vec<RowError>) {
    let mut errors = Vec::new();
    let mut batch: Vec<(TypedEdge, u64)> = Vec::with_capacity(rows.len());

    for row in rows {
        if !known_edge_types.contains(&row.edge_type) {
            errors.push(RowError {
                row: row.row,
                message: format!("unknown edge type '{}'", row.edge_type),
            });
            continue;
        }
        if !valid_node_ids.contains(&row.source) {
            errors.push(RowError {
                row: row.row,
                message: format!("unknown source node id {}", row.source),
            });
            continue;
        }
        if !valid_node_ids.contains(&row.target) {
            errors.push(RowError {
                row: row.row,
                message: format!("unknown target node id {}", row.target),
            });
            continue;
        }

        let created_at = now_millis();
        let edge_type = store.intern(&row.edge_type);
        let mut edge = TypedEdge {
            id: store.alloc_edge_id(),
            source: NodeId(row.source),
            target: NodeId(row.target),
            edge_type,
            properties: row.properties.into_iter().collect(),
            provenance: Provenance::default(),
            confidence: row.confidence.unwrap_or(1.0),
            verified: false,
            is_manual: true,
            created_at,
            history: Vec::new(),
            valid_from: row.valid_from,
            valid_until: row.valid_until,
            temporal_context: row.temporal_context,
        };
        edge.record_audit("imported", actor.clone(), created_at);
        batch.push((edge, lsn));
    }

    if batch.is_empty() {
        return (0, errors);
    }

    let imported = batch.len() as u64;
    if let Err(e) = store.bulk_put_edges(batch) {
        errors.push(RowError {
            row: 0,
            message: format!("bulk write failed: {e}"),
        });
        return (0, errors);
    }
    (imported, errors)
}
