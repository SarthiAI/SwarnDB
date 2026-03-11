// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Distance functions operating on scalar-quantized (u8) vectors.
//!
//! Two flavours are provided for each metric:
//! - **Exact**: dequantize both vectors, then compute the distance in f32.
//! - **Approximate** (`_fast` suffix): operate directly on u8 codes for speed,
//!   trading a small amount of accuracy.

use crate::error::QuantizationError;
use crate::scalar::ScalarQuantizer;

// ---------------------------------------------------------------------------
// Euclidean (L2) distance
// ---------------------------------------------------------------------------

/// Compute approximate L2 distance by dequantizing both vectors first.
///
/// Returns an error if the quantizer is not trained or codes have wrong dimension.
pub fn sq_euclidean_distance(
    a: &[u8],
    b: &[u8],
    quantizer: &ScalarQuantizer,
) -> Result<f32, QuantizationError> {
    let a_f = quantizer.dequantize(a)?;
    let b_f = quantizer.dequantize(b)?;

    Ok(a_f
        .iter()
        .zip(b_f.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt())
}

/// Fast approximate L2 distance operating directly on u8 codes.
///
/// The result is scaled by the per-dimension ranges so that it approximates
/// the true L2 distance without full dequantization.
///
/// # Note
/// If `a`, `b`, or `ranges()` have different lengths, `zip` truncates to the
/// shortest — no panic occurs, but the result will be silently incorrect.
/// Callers must ensure all three slices have the same length (the quantizer's
/// trained dimension).
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
/// Returns an error if the quantizer is not trained or codes have wrong dimension.
pub fn sq_dot_product_distance(
    a: &[u8],
    b: &[u8],
    quantizer: &ScalarQuantizer,
) -> Result<f32, QuantizationError> {
    let a_f = quantizer.dequantize(a)?;
    let b_f = quantizer.dequantize(b)?;

    let dot: f32 = a_f.iter().zip(b_f.iter()).map(|(x, y)| x * y).sum();
    Ok(-dot)
}

// ---------------------------------------------------------------------------
// Cosine distance
// ---------------------------------------------------------------------------

/// Compute cosine distance (1 - cosine_similarity) by dequantizing both vectors.
///
/// Returns an error if the quantizer is not trained or codes have wrong dimension.
pub fn sq_cosine_distance(
    a: &[u8],
    b: &[u8],
    quantizer: &ScalarQuantizer,
) -> Result<f32, QuantizationError> {
    let a_f = quantizer.dequantize(a)?;
    let b_f = quantizer.dequantize(b)?;

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
        Ok(1.0)
    } else {
        Ok(1.0 - (dot / denom))
    }
}
