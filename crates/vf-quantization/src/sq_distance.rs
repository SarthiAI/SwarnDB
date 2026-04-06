// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Distance functions operating on scalar-quantized (u8) vectors.
//!
//! Two flavours are provided for each metric:
//! - **Exact**: dequantize both vectors, then compute the distance in f32.
//! - **Approximate** (`_fast` suffix): operate directly on u8 codes for speed,
//!   trading a small amount of accuracy.

use crate::scalar::ScalarQuantizer;

// ---------------------------------------------------------------------------
// Euclidean (L2) distance
// ---------------------------------------------------------------------------

/// Compute approximate L2 distance by dequantizing both vectors first.
///
/// # Panics
/// Panics if quantized codes have different lengths than the quantizer's dimension.
pub fn sq_euclidean_distance(a: &[u8], b: &[u8], quantizer: &ScalarQuantizer) -> f32 {
    let a_f = quantizer.dequantize(a).expect("dequantize a");
    let b_f = quantizer.dequantize(b).expect("dequantize b");

    a_f.iter()
        .zip(b_f.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Fast approximate L2 distance operating directly on u8 codes.
///
/// The result is scaled by the per-dimension ranges so that it approximates
/// the true L2 distance without full dequantization.
///
/// # Panics
/// Panics if quantized codes have different lengths than the quantizer's dimension.
pub fn sq_euclidean_distance_fast(a: &[u8], b: &[u8], quantizer: &ScalarQuantizer) -> f32 {
    let ranges = quantizer.ranges();
    let scale = 1.0 / 255.0;

    a.iter()
        .zip(b.iter())
        .zip(ranges.iter())
        .map(|((&ca, &cb), &r)| {
            let diff = (ca as f32 - cb as f32) * scale * r;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Dot-product distance
// ---------------------------------------------------------------------------

/// Compute dot-product distance by dequantizing both vectors first.
///
/// Returns the *negative* dot product so that smaller values indicate
/// higher similarity (consistent with distance semantics).
///
/// # Panics
/// Panics if quantized codes have different lengths than the quantizer's dimension.
pub fn sq_dot_product_distance(a: &[u8], b: &[u8], quantizer: &ScalarQuantizer) -> f32 {
    let a_f = quantizer.dequantize(a).expect("dequantize a");
    let b_f = quantizer.dequantize(b).expect("dequantize b");

    let dot: f32 = a_f.iter().zip(b_f.iter()).map(|(x, y)| x * y).sum();
    -dot
}

// ---------------------------------------------------------------------------
// Cosine distance
// ---------------------------------------------------------------------------

/// Compute cosine distance (1 - cosine_similarity) by dequantizing both vectors.
///
/// # Panics
/// Panics if quantized codes have different lengths than the quantizer's dimension.
pub fn sq_cosine_distance(a: &[u8], b: &[u8], quantizer: &ScalarQuantizer) -> f32 {
    let a_f = quantizer.dequantize(a).expect("dequantize a");
    let b_f = quantizer.dequantize(b).expect("dequantize b");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a_f.iter().zip(b_f.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}

// ---------------------------------------------------------------------------
// Asymmetric distance functions (f32 query vs u8 stored code)
// ---------------------------------------------------------------------------

/// Asymmetric L2 (Euclidean) distance: f32 query vs u8 stored code.
///
/// Dequantizes the stored code inline without allocating, then computes
/// distance. This is the primary distance function for HNSW traversal
/// with SQ8.
///
/// # Panics
/// Panics if slice lengths are not all equal.
pub fn sq_asymmetric_euclidean(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    ranges: &[f32],
) -> f32 {
    assert_eq!(query.len(), code.len());
    assert_eq!(query.len(), min_vals.len());
    assert_eq!(query.len(), ranges.len());

    let mut sum = 0.0f32;
    for d in 0..query.len() {
        let dequantized = code[d] as f32 / 255.0 * ranges[d] + min_vals[d];
        let diff = query[d] - dequantized;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Asymmetric dot-product distance: f32 query vs u8 stored code.
///
/// Returns the *negative* dot product so that smaller values indicate
/// higher similarity (consistent with distance semantics).
///
/// # Panics
/// Panics if slice lengths are not all equal.
pub fn sq_asymmetric_dot_product(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    ranges: &[f32],
) -> f32 {
    assert_eq!(query.len(), code.len());
    assert_eq!(query.len(), min_vals.len());
    assert_eq!(query.len(), ranges.len());

    let mut dot = 0.0f32;
    for d in 0..query.len() {
        let dequantized = code[d] as f32 / 255.0 * ranges[d] + min_vals[d];
        dot += query[d] * dequantized;
    }
    -dot
}

/// Asymmetric cosine distance: f32 query vs u8 stored code.
///
/// Returns `1.0 - cosine_similarity`.
///
/// # Panics
/// Panics if slice lengths are not all equal.
pub fn sq_asymmetric_cosine(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    ranges: &[f32],
) -> f32 {
    assert_eq!(query.len(), code.len());
    assert_eq!(query.len(), min_vals.len());
    assert_eq!(query.len(), ranges.len());

    let mut dot = 0.0f32;
    let mut norm_q = 0.0f32;
    let mut norm_c = 0.0f32;
    for d in 0..query.len() {
        let dequantized = code[d] as f32 / 255.0 * ranges[d] + min_vals[d];
        dot += query[d] * dequantized;
        norm_q += query[d] * query[d];
        norm_c += dequantized * dequantized;
    }

    let denom = (norm_q * norm_c).sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}
