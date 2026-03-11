// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! SIMD-optimized distance computation kernels with runtime CPU feature detection.
//!
//! Provides accelerated implementations of dot product, squared L2 distance,
//! and Manhattan distance using AVX2/SSE4.1 on x86_64 and NEON on aarch64,
//! with automatic scalar fallback.

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Scalar fallback implementations
// ---------------------------------------------------------------------------

fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "scalar_dot_product: mismatched lengths ({} vs {})", a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn scalar_squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "scalar_squared_l2: mismatched lengths ({} vs {})", a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

fn scalar_manhattan(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "scalar_manhattan: mismatched lengths ({} vs {})", a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

/// Fused cosine: compute dot(a,b), ||a||^2, ||b||^2 in a single pass.
fn scalar_fused_cosine(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len(), "scalar_fused_cosine: mismatched lengths ({} vs {})", a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    (dot, norm_a, norm_b)
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot_product(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    // Horizontal sum of 256-bit register
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail
    let tail_start = chunks * 8;
    for i in 0..remainder {
        total += a[tail_start + i] * b[tail_start + i];
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_squared_l2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail
    let tail_start = chunks * 8;
    for i in 0..remainder {
        let diff = a[tail_start + i] - b[tail_start + i];
        total += diff * diff;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_manhattan(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    // Sign mask to compute absolute value: clear the sign bit
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_and_ps(diff, sign_mask);
        sum = _mm256_add_ps(sum, abs_diff);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail
    let tail_start = chunks * 8;
    for i in 0..remainder {
        total += (a[tail_start + i] - b[tail_start + i]).abs();
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_fused_cosine(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum_dot = _mm256_setzero_ps();
    let mut sum_na = _mm256_setzero_ps();
    let mut sum_nb = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum_dot = _mm256_add_ps(sum_dot, _mm256_mul_ps(va, vb));
        sum_na = _mm256_add_ps(sum_na, _mm256_mul_ps(va, va));
        sum_nb = _mm256_add_ps(sum_nb, _mm256_mul_ps(vb, vb));
    }

    // Horizontal sum helper
    #[inline(always)]
    unsafe fn hsum256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }

    let mut dot = hsum256(sum_dot);
    let mut norm_a = hsum256(sum_na);
    let mut norm_b = hsum256(sum_nb);

    // Scalar tail
    let tail_start = chunks * 8;
    for i in 0..remainder {
        let ai = a[tail_start + i];
        let bi = b[tail_start + i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    (dot, norm_a, norm_b)
}

// ---------------------------------------------------------------------------
// x86_64 SSE4.1 kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse41_dot_product(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }

    // Horizontal sum of 128-bit register
    let shuf = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        total += a[tail_start + i] * b[tail_start + i];
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse41_squared_l2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        let diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    // Horizontal sum
    let shuf = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let diff = a[tail_start + i] - b[tail_start + i];
        total += diff * diff;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse41_manhattan(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFu32 as i32));
    let mut sum = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        let diff = _mm_sub_ps(va, vb);
        let abs_diff = _mm_and_ps(diff, sign_mask);
        sum = _mm_add_ps(sum, abs_diff);
    }

    // Horizontal sum
    let shuf = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        total += (a[tail_start + i] - b[tail_start + i]).abs();
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse41_fused_cosine(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_dot = _mm_setzero_ps();
    let mut sum_na = _mm_setzero_ps();
    let mut sum_nb = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        sum_dot = _mm_add_ps(sum_dot, _mm_mul_ps(va, vb));
        sum_na = _mm_add_ps(sum_na, _mm_mul_ps(va, va));
        sum_nb = _mm_add_ps(sum_nb, _mm_mul_ps(vb, vb));
    }

    // Horizontal sum helper
    #[inline(always)]
    unsafe fn hsum128(v: __m128) -> f32 {
        let shuf = _mm_movehdup_ps(v);
        let sums = _mm_add_ps(v, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }

    let mut dot = hsum128(sum_dot);
    let mut norm_a = hsum128(sum_na);
    let mut norm_b = hsum128(sum_nb);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let ai = a[tail_start + i];
        let bi = b[tail_start + i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    (dot, norm_a, norm_b)
}

// ---------------------------------------------------------------------------
// aarch64 NEON kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_product(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            sum = vfmaq_f32(sum, va, vb);
        }
    }

    // Horizontal sum: add pairwise then extract
    let mut total = vaddvq_f32(sum);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        total += a[tail_start + i] * b[tail_start + i];
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_squared_l2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }
    }

    let mut total = vaddvq_f32(sum);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let diff = a[tail_start + i] - b[tail_start + i];
        total += diff * diff;
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_manhattan(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let diff = vsubq_f32(va, vb);
            let abs_diff = vabsq_f32(diff);
            sum = vaddq_f32(sum, abs_diff);
        }
    }

    let mut total = vaddvq_f32(sum);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        total += (a[tail_start + i] - b[tail_start + i]).abs();
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_cosine(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_dot = vdupq_n_f32(0.0);
    let mut sum_na = vdupq_n_f32(0.0);
    let mut sum_nb = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            sum_dot = vfmaq_f32(sum_dot, va, vb);
            sum_na = vfmaq_f32(sum_na, va, va);
            sum_nb = vfmaq_f32(sum_nb, vb, vb);
        }
    }

    let mut dot = vaddvq_f32(sum_dot);
    let mut norm_a = vaddvq_f32(sum_na);
    let mut norm_b = vaddvq_f32(sum_nb);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let ai = a[tail_start + i];
        let bi = b[tail_start + i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    (dot, norm_a, norm_b)
}

// ---------------------------------------------------------------------------
// SimdDispatcher: runtime feature detection + function pointer dispatch
// ---------------------------------------------------------------------------

/// Function pointer type for SIMD-dispatched distance kernels.
type SimdKernelFn = fn(&[f32], &[f32]) -> f32;

/// Function pointer type for fused cosine kernel (returns dot, norm_a_sq, norm_b_sq).
type FusedCosineKernelFn = fn(&[f32], &[f32]) -> (f32, f32, f32);

/// Runtime SIMD dispatcher that detects CPU features once and stores
/// function pointers to the best available implementation.
///
/// Thread-safe (`Send + Sync`) and accessed via a global singleton.
pub struct SimdDispatcher {
    dot_product_fn: SimdKernelFn,
    squared_l2_fn: SimdKernelFn,
    manhattan_fn: SimdKernelFn,
    fused_cosine_fn: FusedCosineKernelFn,
}

// Function pointers are Send + Sync
unsafe impl Send for SimdDispatcher {}
unsafe impl Sync for SimdDispatcher {}

impl SimdDispatcher {
    /// Create a new dispatcher by detecting CPU features at runtime.
    fn detect() -> Self {
        let (dot_fn, sql2_fn, man_fn, fused_cos_fn) = Self::select_kernels();
        SimdDispatcher {
            dot_product_fn: dot_fn,
            squared_l2_fn: sql2_fn,
            manhattan_fn: man_fn,
            fused_cosine_fn: fused_cos_fn,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn select_kernels() -> (SimdKernelFn, SimdKernelFn, SimdKernelFn, FusedCosineKernelFn) {
        if is_x86_feature_detected!("avx2") {
            // Wrap the unsafe target_feature functions in safe function pointers.
            // The runtime detection guarantees AVX2 is available.
            (
                |a, b| unsafe { avx2_dot_product(a, b) },
                |a, b| unsafe { avx2_squared_l2(a, b) },
                |a, b| unsafe { avx2_manhattan(a, b) },
                |a, b| unsafe { avx2_fused_cosine(a, b) },
            )
        } else if is_x86_feature_detected!("sse4.1") {
            (
                |a, b| unsafe { sse41_dot_product(a, b) },
                |a, b| unsafe { sse41_squared_l2(a, b) },
                |a, b| unsafe { sse41_manhattan(a, b) },
                |a, b| unsafe { sse41_fused_cosine(a, b) },
            )
        } else {
            (
                scalar_dot_product as SimdKernelFn,
                scalar_squared_l2 as SimdKernelFn,
                scalar_manhattan as SimdKernelFn,
                scalar_fused_cosine as FusedCosineKernelFn,
            )
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn select_kernels() -> (SimdKernelFn, SimdKernelFn, SimdKernelFn, FusedCosineKernelFn) {
        // NEON is mandatory on aarch64, but we still use runtime detection
        // for consistency and future-proofing.
        if std::arch::is_aarch64_feature_detected!("neon") {
            (
                |a, b| unsafe { neon_dot_product(a, b) },
                |a, b| unsafe { neon_squared_l2(a, b) },
                |a, b| unsafe { neon_manhattan(a, b) },
                |a, b| unsafe { neon_fused_cosine(a, b) },
            )
        } else {
            (
                scalar_dot_product as SimdKernelFn,
                scalar_squared_l2 as SimdKernelFn,
                scalar_manhattan as SimdKernelFn,
                scalar_fused_cosine as FusedCosineKernelFn,
            )
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn select_kernels() -> (SimdKernelFn, SimdKernelFn, SimdKernelFn, FusedCosineKernelFn) {
        (
            scalar_dot_product as SimdKernelFn,
            scalar_squared_l2 as SimdKernelFn,
            scalar_manhattan as SimdKernelFn,
            scalar_fused_cosine as FusedCosineKernelFn,
        )
    }

    /// Compute dot product: sum(a_i * b_i)
    #[inline]
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        (self.dot_product_fn)(a, b)
    }

    /// Compute squared L2 distance: sum((a_i - b_i)^2)
    #[inline]
    pub fn squared_l2(&self, a: &[f32], b: &[f32]) -> f32 {
        (self.squared_l2_fn)(a, b)
    }

    /// Compute Manhattan distance: sum(|a_i - b_i|)
    #[inline]
    pub fn manhattan(&self, a: &[f32], b: &[f32]) -> f32 {
        (self.manhattan_fn)(a, b)
    }

    /// Compute fused cosine: returns (dot_ab, norm_a_sq, norm_b_sq) in single pass.
    #[inline]
    pub fn fused_cosine(&self, a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        (self.fused_cosine_fn)(a, b)
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static DISPATCHER: OnceLock<SimdDispatcher> = OnceLock::new();

/// Get a reference to the global SIMD dispatcher singleton.
/// The dispatcher is initialized on first call with runtime CPU feature detection.
pub fn get_dispatcher() -> &'static SimdDispatcher {
    DISPATCHER.get_or_init(SimdDispatcher::detect)
}

// ---------------------------------------------------------------------------
// Batch computation methods on SimdDispatcher
// ---------------------------------------------------------------------------

impl SimdDispatcher {
    /// Compute dot product from one query to multiple targets.
    /// Keeps the query vector hot in L1 cache while iterating targets.
    #[inline]
    pub fn batch_dot_product(&self, query: &[f32], targets: &[&[f32]], results: &mut [f32]) {
        debug_assert_eq!(targets.len(), results.len(), "batch_dot_product: targets.len()={} != results.len()={}", targets.len(), results.len());
        for (i, target) in targets.iter().enumerate() {
            debug_assert_eq!(query.len(), target.len(), "batch_dot_product: query.len()={} != target[{}].len()={}", query.len(), i, target.len());
            results[i] = (self.dot_product_fn)(query, target);
        }
    }

    /// Compute squared L2 distance from one query to multiple targets.
    #[inline]
    pub fn batch_squared_l2(&self, query: &[f32], targets: &[&[f32]], results: &mut [f32]) {
        debug_assert_eq!(targets.len(), results.len(), "batch_squared_l2: targets.len()={} != results.len()={}", targets.len(), results.len());
        for (i, target) in targets.iter().enumerate() {
            debug_assert_eq!(query.len(), target.len(), "batch_squared_l2: query.len()={} != target[{}].len()={}", query.len(), i, target.len());
            results[i] = (self.squared_l2_fn)(query, target);
        }
    }

    /// Compute Manhattan distance from one query to multiple targets.
    #[inline]
    pub fn batch_manhattan(&self, query: &[f32], targets: &[&[f32]], results: &mut [f32]) {
        debug_assert_eq!(targets.len(), results.len(), "batch_manhattan: targets.len()={} != results.len()={}", targets.len(), results.len());
        for (i, target) in targets.iter().enumerate() {
            debug_assert_eq!(query.len(), target.len(), "batch_manhattan: query.len()={} != target[{}].len()={}", query.len(), i, target.len());
            results[i] = (self.manhattan_fn)(query, target);
        }
    }

    /// Compute fused cosine from one query to multiple targets.
    /// Returns (dot, norm_a_sq, norm_b_sq) triples via output slices.
    #[inline]
    pub fn batch_fused_cosine(
        &self,
        query: &[f32],
        targets: &[&[f32]],
        dots: &mut [f32],
        norms_a: &mut [f32],
        norms_b: &mut [f32],
    ) {
        debug_assert_eq!(targets.len(), dots.len(), "batch_fused_cosine: targets.len()={} != dots.len()={}", targets.len(), dots.len());
        debug_assert_eq!(targets.len(), norms_a.len(), "batch_fused_cosine: targets.len()={} != norms_a.len()={}", targets.len(), norms_a.len());
        debug_assert_eq!(targets.len(), norms_b.len(), "batch_fused_cosine: targets.len()={} != norms_b.len()={}", targets.len(), norms_b.len());
        for (i, target) in targets.iter().enumerate() {
            debug_assert_eq!(query.len(), target.len(), "batch_fused_cosine: query.len()={} != target[{}].len()={}", query.len(), i, target.len());
            let (d, na, nb) = (self.fused_cosine_fn)(query, target);
            dots[i] = d;
            norms_a[i] = na;
            norms_b[i] = nb;
        }
    }
}

// ---------------------------------------------------------------------------
// Public convenience functions
// ---------------------------------------------------------------------------

/// SIMD-accelerated dot product of two f32 slices.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "SIMD vector length mismatch: a.len()={}, b.len()={}", a.len(), b.len());
    get_dispatcher().dot_product(a, b)
}

/// SIMD-accelerated squared L2 distance between two f32 slices.
#[inline]
pub fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "SIMD vector length mismatch: a.len()={}, b.len()={}", a.len(), b.len());
    get_dispatcher().squared_l2(a, b)
}

/// SIMD-accelerated Manhattan distance between two f32 slices.
#[inline]
pub fn manhattan_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "SIMD vector length mismatch: a.len()={}, b.len()={}", a.len(), b.len());
    get_dispatcher().manhattan(a, b)
}

/// SIMD-accelerated fused cosine computation returning (dot, norm_a_sq, norm_b_sq).
/// Computes all three values in a single pass over the data for better cache utilization.
#[inline]
pub fn fused_cosine_f32(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len(), "SIMD vector length mismatch: a.len()={}, b.len()={}", a.len(), b.len());
    get_dispatcher().fused_cosine(a, b)
}

/// Batch SIMD-accelerated dot product: one query vs multiple targets.
#[inline]
pub fn batch_dot_product_f32(query: &[f32], targets: &[&[f32]], results: &mut [f32]) {
    get_dispatcher().batch_dot_product(query, targets, results);
}

/// Batch SIMD-accelerated squared L2: one query vs multiple targets.
#[inline]
pub fn batch_squared_l2_f32(query: &[f32], targets: &[&[f32]], results: &mut [f32]) {
    get_dispatcher().batch_squared_l2(query, targets, results);
}

/// Batch SIMD-accelerated Manhattan: one query vs multiple targets.
#[inline]
pub fn batch_manhattan_f32(query: &[f32], targets: &[&[f32]], results: &mut [f32]) {
    get_dispatcher().batch_manhattan(query, targets, results);
}

/// Batch SIMD-accelerated fused cosine: one query vs multiple targets.
#[inline]
pub fn batch_fused_cosine_f32(
    query: &[f32],
    targets: &[&[f32]],
    dots: &mut [f32],
    norms_a: &mut [f32],
    norms_b: &mut [f32],
) {
    get_dispatcher().batch_fused_cosine(query, targets, dots, norms_a, norms_b);
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!(approx_eq(dot_product_f32(&a, &b), 70.0));
    }

    #[test]
    fn test_squared_l2_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (3^2 + 3^2 + 3^2) = 27
        assert!(approx_eq(squared_l2_f32(&a, &b), 27.0));
    }

    #[test]
    fn test_manhattan_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // |3| + |3| + |3| = 9
        assert!(approx_eq(manhattan_f32(&a, &b), 9.0));
    }

    #[test]
    fn test_dot_product_large_unaligned() {
        // Test with 1537 elements (not a multiple of 4 or 8)
        let a: Vec<f32> = (0..1537).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..1537).map(|i| (i as f32) * 0.002).collect();
        let simd_result = dot_product_f32(&a, &b);
        let scalar_result = scalar_dot_product(&a, &b);
        assert!(
            (simd_result - scalar_result).abs() < 0.1,
            "dot product mismatch: simd={}, scalar={}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_squared_l2_large_unaligned() {
        let a: Vec<f32> = (0..1537).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..1537).map(|i| (i as f32) * 0.002).collect();
        let simd_result = squared_l2_f32(&a, &b);
        let scalar_result = scalar_squared_l2(&a, &b);
        assert!(
            (simd_result - scalar_result).abs() < 0.1,
            "squared_l2 mismatch: simd={}, scalar={}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_manhattan_large_unaligned() {
        let a: Vec<f32> = (0..1537).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..1537).map(|i| (i as f32) * 0.002).collect();
        let simd_result = manhattan_f32(&a, &b);
        let scalar_result = scalar_manhattan(&a, &b);
        assert!(
            (simd_result - scalar_result).abs() < 0.1,
            "manhattan mismatch: simd={}, scalar={}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(approx_eq(dot_product_f32(&a, &b), 0.0));
        assert!(approx_eq(squared_l2_f32(&a, &b), 0.0));
        assert!(approx_eq(manhattan_f32(&a, &b), 0.0));
    }

    #[test]
    fn test_single_element() {
        let a = vec![3.0];
        let b = vec![7.0];
        assert!(approx_eq(dot_product_f32(&a, &b), 21.0));
        assert!(approx_eq(squared_l2_f32(&a, &b), 16.0));
        assert!(approx_eq(manhattan_f32(&a, &b), 4.0));
    }

    #[test]
    fn test_dispatcher_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SimdDispatcher>();
    }
}
