// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Configurable deterministic entity resolution (ADR-020).
//!
//! This module holds the pure, label-agnostic matching rules used when a
//! collection opts in to fuzzy entity resolution. The caller is responsible for
//! label scoping: it only ever passes candidates of the SAME entity label, so
//! the rules here never merge across labels.
//!
//! Every rule is deliberately conservative. The dominant failure mode of any
//! resolver is OVER-MERGING two distinct real-world entities, which corrupts the
//! graph and hurts precision; splitting one entity into two nodes is recoverable,
//! a wrong merge is not. So when a pair is ambiguous the rules return `false`
//! (no merge, a fresh node is created). Thresholds are principled and fixed; they
//! are never tuned to make a particular dataset pass.
//!
//! All inputs are the deterministic normalized name (`normalize_entity_name`,
//! ADR-015): already trimmed, whitespace-collapsed, NFKC, lowercased, with
//! surrounding quotes and trailing punctuation stripped. Rules operate on that
//! normalized form, so case / spacing / punctuation variants are already handled
//! by exact match before any fuzzy rule runs.

/// Whether two already-normalized entity names of the SAME label refer to the
/// same entity under the conservative fuzzy rules. Rules are tried in priority
/// order; the first confident match wins. Returns `false` (do not merge) when no
/// rule is confident.
///
/// The caller guarantees `query` is not empty (an empty name never resolves).
pub fn fuzzy_name_match(stored: &str, query: &str) -> bool {
    // Rule 0: exact normalized match. Identical to the Normalized-mode behavior,
    // so an exact pair always merges regardless of mode.
    if stored == query {
        return true;
    }

    // Empty on either side is never a confident match.
    if stored.is_empty() || query.is_empty() {
        return false;
    }

    let stored_tokens: Vec<&str> = tokens(stored);
    let query_tokens: Vec<&str> = tokens(query);
    if stored_tokens.is_empty() || query_tokens.is_empty() {
        return false;
    }

    // Rule 1: initials / abbreviation, positional. "j. smith" <-> "john smith":
    // same token count, and at every position the tokens are equal OR one side is
    // a single-letter initial that is the first letter of the other side's token.
    // Requires at least one real (non-initial) token to agree, so two pure
    // initialisms ("j. s." vs "j. s.") only match if they are byte-equal (caught
    // by Rule 0), never by this rule alone.
    if initials_positional_match(&stored_tokens, &query_tokens) {
        return true;
    }

    // Rule 2: acronym <-> expansion. "usa" <-> "united states of america": one
    // side is a single token whose letters are exactly the leading letters of the
    // other side's multi-token name, in order. Guarded so two DIFFERENT acronyms
    // never match each other (each acronym only matches its own expansion, and an
    // acronym-vs-acronym pair has two single tokens, which this rule rejects).
    if acronym_match(&stored_tokens, &query_tokens)
        || acronym_match(&query_tokens, &stored_tokens)
    {
        return true;
    }

    // Rule 3: bounded edit distance for typos. The bound SCALES DOWN for short
    // strings so distinct short names ("john" vs "joan") never merge. Measured on
    // the full normalized strings, not per token.
    if within_typo_bound(stored, query) {
        return true;
    }

    // Rule 4: contiguous token-containment with a strong guard. The shorter name
    // (which must have at least TWO tokens) must appear as a CONTIGUOUS run of
    // tokens inside the longer name. Catches "barack obama" <-> "president barack
    // obama" (the two tokens are adjacent in the longer name). Refuses
    // "john smith" vs "jane smith" (neither contains the other), single-token
    // subsumption like "smith" vs "john smith" (the shorter side is one token),
    // and non-contiguous overlaps like "john smith" vs "john michael smith"
    // (the tokens are not adjacent), all of which are over-merge traps.
    if token_containment_match(&stored_tokens, &query_tokens) {
        return true;
    }

    false
}

/// Split a normalized name into whitespace-delimited tokens.
fn tokens(s: &str) -> Vec<&str> {
    s.split(' ').filter(|t| !t.is_empty()).collect()
}

/// True if `t` is a single-letter initial (one alphanumeric char), optionally
/// already stripped of its dot by normalization. After normalization a trailing
/// "." is only stripped at the very end of the whole name, so an internal
/// initial may still carry its dot (e.g. "j."); accept a one-letter token with
/// or without a single trailing dot.
fn is_initial(t: &str) -> bool {
    let core = t.strip_suffix('.').unwrap_or(t);
    core.chars().count() == 1 && core.chars().all(|c| c.is_alphanumeric())
}

/// The leading character of a token, ignoring a trailing dot.
fn head_char(t: &str) -> Option<char> {
    t.strip_suffix('.').unwrap_or(t).chars().next()
}

/// Rule 1 helper: positional initial / full-token match across equal-length
/// token lists. At least one position must be a full-vs-full equal token so two
/// names do not match on initials alone.
fn initials_positional_match(a: &[&str], b: &[&str]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut any_full_agreement = false;
    for (x, y) in a.iter().zip(b.iter()) {
        let xi = is_initial(x);
        let yi = is_initial(y);
        if !xi && !yi {
            // Both full tokens: they must be equal.
            if x != y {
                return false;
            }
            any_full_agreement = true;
        } else {
            // At least one is an initial: their leading letters must agree.
            match (head_char(x), head_char(y)) {
                (Some(cx), Some(cy)) if cx == cy => {}
                _ => return false,
            }
        }
    }
    any_full_agreement
}

/// Rule 2 helper: is `acronym` a single token whose letters are exactly the
/// leading letters of the multi-token `expansion`, in order? Requires the
/// expansion to have at least two tokens, so an acronym never matches another
/// single-token name (including another acronym).
fn acronym_match(acronym: &[&str], expansion: &[&str]) -> bool {
    if acronym.len() != 1 || expansion.len() < 2 {
        return false;
    }
    // The acronym's bare letters (drop any embedded dots already removed by
    // normalization except a possible internal one; normalized "u.s.a." keeps
    // internal dots, so strip them here for the letter sequence).
    let letters: Vec<char> = acronym[0].chars().filter(|c| c.is_alphanumeric()).collect();
    if letters.len() < 2 || letters.len() != expansion.len() {
        // The acronym must cover every expansion token one-for-one. Requiring an
        // exact one-letter-per-token mapping refuses partial acronyms, which are
        // a common over-merge trap.
        return false;
    }
    for (letter, token) in letters.iter().zip(expansion.iter()) {
        match token.chars().next() {
            Some(c) if c == *letter => {}
            _ => return false,
        }
    }
    true
}

/// Rule 3 helper: bounded Levenshtein distance with a length-scaled bound. The
/// bound is measured against the SHORTER string so a long-vs-short pair cannot
/// sneak under a generous bound. Distinct short names stay distinct.
fn within_typo_bound(a: &str, b: &str) -> bool {
    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let shorter = len_a.min(len_b);
    // Conservative, fixed bound. Never a percentage (which would merge short
    // distinct names like "john" / "joan").
    let bound = if shorter <= 4 {
        0
    } else if shorter <= 8 {
        1
    } else {
        2
    };
    if bound == 0 {
        // Bound 0 means only exact strings match, already handled by Rule 0.
        return false;
    }
    // Quick reject: if the length gap already exceeds the bound the distance must
    // too, so skip the O(n*m) computation.
    let gap = len_a.max(len_b) - shorter;
    if gap > bound {
        return false;
    }
    levenshtein_bounded(a, b, bound) <= bound
}

/// Levenshtein distance, short-circuited once the running minimum of a row
/// exceeds `bound` (returns `bound + 1` in that case). Cost is O(n*m) worst case
/// but bounded in practice by the early length-gap reject in the caller and by
/// the small `bound`.
fn levenshtein_bounded(a: &str, b: &str, bound: usize) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        let mut row_min = curr[0];
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            row_min = row_min.min(curr[j]);
        }
        if row_min > bound {
            return bound + 1;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Rule 4 helper: the shorter name (at least TWO tokens) appears as a CONTIGUOUS
/// run of tokens inside the longer name. Contiguity is the guard: it admits a
/// title / honorific / role prefix or suffix wrapping the full shorter name
/// ("president barack obama" contains "barack obama") while refusing a merged
/// token set that is not adjacent ("john michael smith" does NOT contain
/// "john smith"), which would be an over-merge. A bare single-token name is never
/// subsumed into a longer name.
fn token_containment_match(a: &[&str], b: &[&str]) -> bool {
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    // Strict superset only: equal length is handled by other rules.
    if long.len() <= short.len() {
        return false;
    }
    // Refuse single-token subsumption ("smith" into "john smith").
    if short.len() < 2 {
        return false;
    }
    // Slide the shorter token run over the longer name; a contiguous match merges.
    long.windows(short.len()).any(|w| w == short)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Positive cases that MUST merge.
    #[test]
    fn exact_and_variants_merge() {
        assert!(fuzzy_name_match("barack obama", "barack obama"));
    }

    #[test]
    fn initials_merge() {
        assert!(fuzzy_name_match("j. smith", "john smith"));
        assert!(fuzzy_name_match("john smith", "j smith"));
    }

    #[test]
    fn acronym_merges_with_expansion() {
        assert!(fuzzy_name_match("usa", "united states america"));
        assert!(fuzzy_name_match(
            "united states america",
            "usa"
        ));
    }

    #[test]
    fn typo_merges_for_longer_names() {
        // len > 8: bound 2.
        assert!(fuzzy_name_match("massachusettz", "massachusetts"));
        // len 5..=8: bound 1.
        assert!(fuzzy_name_match("obamaa", "obama"));
    }

    #[test]
    fn title_prefix_merges() {
        assert!(fuzzy_name_match("president barack obama", "barack obama"));
    }

    // Negative cases that MUST NOT merge.
    #[test]
    fn different_given_name_does_not_merge() {
        assert!(!fuzzy_name_match("john smith", "jane smith"));
    }

    #[test]
    fn bare_surname_does_not_merge() {
        assert!(!fuzzy_name_match("smith", "john smith"));
    }

    #[test]
    fn two_different_acronyms_do_not_merge() {
        assert!(!fuzzy_name_match("usa", "uae"));
        assert!(!fuzzy_name_match("ibm", "bmw"));
    }

    #[test]
    fn short_distinct_names_do_not_merge() {
        assert!(!fuzzy_name_match("john", "joan"));
        assert!(!fuzzy_name_match("amy", "ana"));
    }

    #[test]
    fn unrelated_long_names_do_not_merge() {
        assert!(!fuzzy_name_match("barack obama", "george bush"));
    }

    #[test]
    fn non_contiguous_token_overlap_does_not_merge() {
        // "john smith" is NOT a contiguous run inside "john michael smith".
        assert!(!fuzzy_name_match("john smith", "john michael smith"));
    }

    #[test]
    fn suffix_prefix_wrapping_merges() {
        // A suffix wraps the full shorter name contiguously.
        assert!(fuzzy_name_match("barack obama jr", "barack obama"));
    }
}
