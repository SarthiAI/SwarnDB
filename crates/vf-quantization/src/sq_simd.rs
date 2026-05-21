// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! SIMD-accelerated asymmetric distance kernels for SQ8 scalar quantization.
//!
//! These functions compute distance between an f32 query vector and a u8
//! quantized code vector. They are the hot path during HNSW graph traversal
//! with SQ8 quantization.
//!
//! Three implementations are provided per metric:
//! - **AVX2** (x86_64): processes 8 dimensions per iteration
//! - **NEON** (aarch64/Apple Silicon): processes 8 dimensions per iteration
//! - **Scalar fallback**: portable, always compiled
//!
//! Runtime dispatch selects the fastest available path automatically.
//!
//! The caller must precompute `scales = ranges / 255.0`. Dequantization is:
//! `dequant[d] = code[d] as f32 * scales[d] + min_vals[d]`.

// ==========================================================================
// Public API - runtime dispatch
// ==========================================================================

/// Asymmetric L2 (Euclidean) distance: f32 query vs u8 code, SIMD-accelerated.
///
/// Returns `sqrt(sum((query[d] - dequant[d])^2))`.
pub fn sq_asymmetric_l2_simd(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_asymmetric_l2(query, code, min_vals, scales) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_asymmetric_l2(query, code, min_vals, scales) };
    }
    #[allow(unreachable_code)]
    scalar_asymmetric_l2(query, code, min_vals, scales)
}

/// Asymmetric dot-product distance: f32 query vs u8 code, SIMD-accelerated.
///
/// Returns the *negative* dot product (smaller = more similar).
pub fn sq_asymmetric_dot_simd(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_asymmetric_dot(query, code, min_vals, scales) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_asymmetric_dot(query, code, min_vals, scales) };
    }
    #[allow(unreachable_code)]
    scalar_asymmetric_dot(query, code, min_vals, scales)
}

/// Asymmetric cosine distance: f32 query vs u8 code, SIMD-accelerated.
///
/// Returns `1.0 - cosine_similarity`.
pub fn sq_asymmetric_cosine_simd(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_asymmetric_cosine(query, code, min_vals, scales) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_asymmetric_cosine(query, code, min_vals, scales) };
    }
    #[allow(unreachable_code)]
    scalar_asymmetric_cosine(query, code, min_vals, scales)
}

// ==========================================================================
// Scalar fallback (always compiled, all platforms)
// ==========================================================================

fn scalar_asymmetric_l2(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..query.len() {
        let dequant = code[d] as f32 * scales[d] + min_vals[d];
        let diff = query[d] - dequant;
        sum += diff * diff;
    }
    sum.sqrt()
}

fn scalar_asymmetric_dot(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    let mut dot = 0.0f32;
    for d in 0..query.len() {
        let dequant = code[d] as f32 * scales[d] + min_vals[d];
        dot += query[d] * dequant;
    }
    -dot
}

fn scalar_asymmetric_cosine(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_q = 0.0f32;
    let mut norm_c = 0.0f32;
    for d in 0..query.len() {
        let dequant = code[d] as f32 * scales[d] + min_vals[d];
        dot += query[d] * dequant;
        norm_q += query[d] * query[d];
        norm_c += dequant * dequant;
    }
    let denom = (norm_q * norm_c).sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

// ==========================================================================
// AVX2 implementation (x86_64)
// ==========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_asymmetric_l2(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let dim = query.len();
        let chunks = dim / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;

            // Load 8 u8 values, zero-extend to i32, convert to f32
            let code_bytes = _mm_loadl_epi64(code.as_ptr().add(offset) as *const __m128i);
            let code_i32 = _mm256_cvtepu8_epi32(code_bytes);
            let code_f32 = _mm256_cvtepi32_ps(code_i32);

            // Load scales and min_vals
            let s = _mm256_loadu_ps(scales.as_ptr().add(offset));
            let m = _mm256_loadu_ps(min_vals.as_ptr().add(offset));

            // Dequantize: code * scale + min
            let dequant = _mm256_add_ps(_mm256_mul_ps(code_f32, s), m);

            // Load query, compute diff, accumulate squared diff
            let q = _mm256_loadu_ps(query.as_ptr().add(offset));
            let diff = _mm256_sub_ps(q, dequant);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        }

        // Horizontal sum of acc
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle tail dimensions
        for d in (chunks * 8)..dim {
            let dequant = code[d] as f32 * scales[d] + min_vals[d];
            let diff = query[d] - dequant;
            result += diff * diff;
        }

        result.sqrt()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_asymmetric_dot(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let dim = query.len();
        let chunks = dim / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;

            let code_bytes = _mm_loadl_epi64(code.as_ptr().add(offset) as *const __m128i);
            let code_i32 = _mm256_cvtepu8_epi32(code_bytes);
            let code_f32 = _mm256_cvtepi32_ps(code_i32);

            let s = _mm256_loadu_ps(scales.as_ptr().add(offset));
            let m = _mm256_loadu_ps(min_vals.as_ptr().add(offset));

            // Dequantize: code * scale + min
            let dequant = _mm256_add_ps(_mm256_mul_ps(code_f32, s), m);

            // Accumulate query * dequant
            let q = _mm256_loadu_ps(query.as_ptr().add(offset));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(q, dequant));
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut dot = _mm_cvtss_f32(sum32);

        // Handle tail
        for d in (chunks * 8)..dim {
            let dequant = code[d] as f32 * scales[d] + min_vals[d];
            dot += query[d] * dequant;
        }

        -dot
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_asymmetric_cosine(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let dim = query.len();
        let chunks = dim / 8;
        let mut dot_acc = _mm256_setzero_ps();
        let mut nq_acc = _mm256_setzero_ps();
        let mut nc_acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;

            let code_bytes = _mm_loadl_epi64(code.as_ptr().add(offset) as *const __m128i);
            let code_i32 = _mm256_cvtepu8_epi32(code_bytes);
            let code_f32 = _mm256_cvtepi32_ps(code_i32);

            let s = _mm256_loadu_ps(scales.as_ptr().add(offset));
            let m = _mm256_loadu_ps(min_vals.as_ptr().add(offset));

            let dequant = _mm256_add_ps(_mm256_mul_ps(code_f32, s), m);
            let q = _mm256_loadu_ps(query.as_ptr().add(offset));

            // dot += q * dequant
            dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(q, dequant));
            // norm_q += q * q
            nq_acc = _mm256_add_ps(nq_acc, _mm256_mul_ps(q, q));
            // norm_c += dequant * dequant
            nc_acc = _mm256_add_ps(nc_acc, _mm256_mul_ps(dequant, dequant));
        }

        // Horizontal sums
        let dot = avx2_hsum(dot_acc);
        let norm_q = avx2_hsum(nq_acc);
        let norm_c = avx2_hsum(nc_acc);

        // Tail
        let (mut dot, mut norm_q, mut norm_c) = (dot, norm_q, norm_c);
        for d in (chunks * 8)..dim {
            let dequant = code[d] as f32 * scales[d] + min_vals[d];
            dot += query[d] * dequant;
            norm_q += query[d] * query[d];
            norm_c += dequant * dequant;
        }

        let denom = (norm_q * norm_c).sqrt();
        if denom == 0.0 {
            1.0
        } else {
            1.0 - dot / denom
        }
    }
}

/// AVX2 horizontal sum of 8 f32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_hsum(v: std::arch::x86_64::__m256) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

// ==========================================================================
// NEON implementation (aarch64 / Apple Silicon)
// ==========================================================================

#[cfg(target_arch = "aarch64")]
unsafe fn neon_asymmetric_l2(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    use std::arch::aarch64::*;

    let dim = query.len();
    let chunks = dim / 8;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 8;

            // Load 8 u8 values and widen to f32
            let code_u8 = vld1_u8(code.as_ptr().add(offset));
            let code_u16 = vmovl_u8(code_u8);
            let code_lo_u32 = vmovl_u16(vget_low_u16(code_u16));
            let code_hi_u32 = vmovl_u16(vget_high_u16(code_u16));
            let code_lo_f32 = vcvtq_f32_u32(code_lo_u32);
            let code_hi_f32 = vcvtq_f32_u32(code_hi_u32);

            // Load scales and min_vals
            let s_lo = vld1q_f32(scales.as_ptr().add(offset));
            let s_hi = vld1q_f32(scales.as_ptr().add(offset + 4));
            let m_lo = vld1q_f32(min_vals.as_ptr().add(offset));
            let m_hi = vld1q_f32(min_vals.as_ptr().add(offset + 4));

            // Dequantize: code * scale + min  (using fused multiply-add)
            let dequant_lo = vfmaq_f32(m_lo, code_lo_f32, s_lo);
            let dequant_hi = vfmaq_f32(m_hi, code_hi_f32, s_hi);

            // Load query
            let q_lo = vld1q_f32(query.as_ptr().add(offset));
            let q_hi = vld1q_f32(query.as_ptr().add(offset + 4));

            // Diff and accumulate squared
            let diff_lo = vsubq_f32(q_lo, dequant_lo);
            let diff_hi = vsubq_f32(q_hi, dequant_hi);
            acc0 = vfmaq_f32(acc0, diff_lo, diff_lo);
            acc1 = vfmaq_f32(acc1, diff_hi, diff_hi);
        }

        // Horizontal sum
        let acc = vaddq_f32(acc0, acc1);
        let mut result = vaddvq_f32(acc);

        // Handle tail
        for d in (chunks * 8)..dim {
            let dequant = code[d] as f32 * scales[d] + min_vals[d];
            let diff = query[d] - dequant;
            result += diff * diff;
        }

        result.sqrt()
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_asymmetric_dot(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    use std::arch::aarch64::*;

    let dim = query.len();
    let chunks = dim / 8;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 8;

            let code_u8 = vld1_u8(code.as_ptr().add(offset));
            let code_u16 = vmovl_u8(code_u8);
            let code_lo_u32 = vmovl_u16(vget_low_u16(code_u16));
            let code_hi_u32 = vmovl_u16(vget_high_u16(code_u16));
            let code_lo_f32 = vcvtq_f32_u32(code_lo_u32);
            let code_hi_f32 = vcvtq_f32_u32(code_hi_u32);

            let s_lo = vld1q_f32(scales.as_ptr().add(offset));
            let s_hi = vld1q_f32(scales.as_ptr().add(offset + 4));
            let m_lo = vld1q_f32(min_vals.as_ptr().add(offset));
            let m_hi = vld1q_f32(min_vals.as_ptr().add(offset + 4));

            let dequant_lo = vfmaq_f32(m_lo, code_lo_f32, s_lo);
            let dequant_hi = vfmaq_f32(m_hi, code_hi_f32, s_hi);

            let q_lo = vld1q_f32(query.as_ptr().add(offset));
            let q_hi = vld1q_f32(query.as_ptr().add(offset + 4));

            // Accumulate query * dequant
            acc0 = vfmaq_f32(acc0, q_lo, dequant_lo);
            acc1 = vfmaq_f32(acc1, q_hi, dequant_hi);
        }

        let acc = vaddq_f32(acc0, acc1);
        let mut dot = vaddvq_f32(acc);

        // Tail
        for d in (chunks * 8)..dim {
            let dequant = code[d] as f32 * scales[d] + min_vals[d];
            dot += query[d] * dequant;
        }

        -dot
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_asymmetric_cosine(
    query: &[f32],
    code: &[u8],
    min_vals: &[f32],
    scales: &[f32],
) -> f32 {
    use std::arch::aarch64::*;

    let dim = query.len();
    let chunks = dim / 8;

    unsafe {
        let mut dot_lo = vdupq_n_f32(0.0);
        let mut dot_hi = vdupq_n_f32(0.0);
        let mut nq_lo = vdupq_n_f32(0.0);
        let mut nq_hi = vdupq_n_f32(0.0);
        let mut nc_lo = vdupq_n_f32(0.0);
        let mut nc_hi = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 8;

            let code_u8 = vld1_u8(code.as_ptr().add(offset));
            let code_u16 = vmovl_u8(code_u8);
            let code_lo_u32 = vmovl_u16(vget_low_u16(code_u16));
            let code_hi_u32 = vmovl_u16(vget_high_u16(code_u16));
            let code_lo_f32 = vcvtq_f32_u32(code_lo_u32);
            let code_hi_f32 = vcvtq_f32_u32(code_hi_u32);

            let s_lo = vld1q_f32(scales.as_ptr().add(offset));
            let s_hi = vld1q_f32(scales.as_ptr().add(offset + 4));
            let m_lo = vld1q_f32(min_vals.as_ptr().add(offset));
            let m_hi = vld1q_f32(min_vals.as_ptr().add(offset + 4));

            let dq_lo = vfmaq_f32(m_lo, code_lo_f32, s_lo);
            let dq_hi = vfmaq_f32(m_hi, code_hi_f32, s_hi);

            let q_lo = vld1q_f32(query.as_ptr().add(offset));
            let q_hi = vld1q_f32(query.as_ptr().add(offset + 4));

            // dot += q * dequant
            dot_lo = vfmaq_f32(dot_lo, q_lo, dq_lo);
            dot_hi = vfmaq_f32(dot_hi, q_hi, dq_hi);
            // norm_q += q * q
            nq_lo = vfmaq_f32(nq_lo, q_lo, q_lo);
            nq_hi = vfmaq_f32(nq_hi, q_hi, q_hi);
            // norm_c += dequant * dequant
            nc_lo = vfmaq_f32(nc_lo, dq_lo, dq_lo);
            nc_hi = vfmaq_f32(nc_hi, dq_hi, dq_hi);
        }

        let mut dot = vaddvq_f32(vaddq_f32(dot_lo, dot_hi));
        let mut norm_q = vaddvq_f32(vaddq_f32(nq_lo, nq_hi));
        let mut norm_c = vaddvq_f32(vaddq_f32(nc_lo, nc_hi));

        // Tail
        for d in (chunks * 8)..dim {
            let dequant = code[d] as f32 * scales[d] + min_vals[d];
            dot += query[d] * dequant;
            norm_q += query[d] * query[d];
            norm_c += dequant * dequant;
        }

        let denom = (norm_q * norm_c).sqrt();
        if denom == 0.0 {
            1.0
        } else {
            1.0 - dot / denom
        }
    }
}
