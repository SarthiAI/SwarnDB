// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! ADR-015 entity-name normalization, shared by the graph store index and the
//! extraction writer.

use unicode_normalization::UnicodeNormalization;

/// Deterministic, conservative normalization of an entity name for resolution
/// matching (ADR-015). Pure transform, no LLM and no embedding:
///   1. trim leading/trailing whitespace
///   2. strip surrounding straight/typographic quotes
///   3. strip trailing punctuation (. , ; :)
///   4. collapse internal whitespace runs to a single space
///   5. Unicode NFKC normalization
///   6. lowercase (full Unicode case fold, not ASCII-only)
///
/// Matching-only: it never changes the displayed `name`. Two names that differ
/// only by these dimensions collapse to one entity node.
pub fn normalize_entity_name(name: &str) -> String {
    let trimmed = name.trim();

    // Strip one layer of surrounding quotes if both ends carry one.
    let unquoted = strip_surrounding_quotes(trimmed);

    // Strip trailing sentence punctuation.
    let no_trailing = unquoted.trim_end_matches(['.', ',', ';', ':']).trim_end();

    // Collapse internal whitespace runs.
    let mut collapsed = String::with_capacity(no_trailing.len());
    let mut prev_space = false;
    for ch in no_trailing.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                collapsed.push(' ');
                prev_space = true;
            }
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }

    // NFKC then full Unicode lowercase fold.
    collapsed.nfkc().collect::<String>().to_lowercase()
}

/// Remove a single matched pair of surrounding quotes (straight or typographic).
fn strip_surrounding_quotes(s: &str) -> &str {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return s;
    }
    let first = chars[0];
    let last = chars[chars.len() - 1];
    let is_pair = matches!(
        (first, last),
        ('"', '"') | ('\'', '\'') | ('\u{201c}', '\u{201d}') | ('\u{2018}', '\u{2019}')
    );
    if is_pair {
        // Slice off the first and last char by byte boundaries.
        let start = first.len_utf8();
        let end = s.len() - last.len_utf8();
        if start <= end {
            return s[start..end].trim();
        }
    }
    s
}
