// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

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
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn scalar_squared_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

fn scalar_manhattan(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
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

// ---------------------------------------------------------------------------
// SimdDispatcher: runtime feature detection + function pointer dispatch
// ---------------------------------------------------------------------------

/// Function pointer type for SIMD-dispatched distance kernels.
type SimdKernelFn = fn(&[f32], &[f32]) -> f32;

/// Runtime SIMD dispatcher that detects CPU features once and stores
/// function pointers to the best available implementation.
///
/// Thread-safe (`Send + Sync`) and accessed via a global singleton.
pub struct SimdDispatcher {
    dot_product_fn: SimdKernelFn,
    squared_l2_fn: SimdKernelFn,
    manhattan_fn: SimdKernelFn,
}

// Function pointers are Send + Sync
unsafe impl Send for SimdDispatcher {}
unsafe impl Sync for SimdDispatcher {}

impl SimdDispatcher {
    /// Create a new dispatcher by detecting CPU features at runtime.
    fn detect() -> Self {
        let (dot_fn, sql2_fn, man_fn) = Self::select_kernels();
        SimdDispatcher {
            dot_product_fn: dot_fn,
            squared_l2_fn: sql2_fn,
            manhattan_fn: man_fn,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn select_kernels() -> (SimdKernelFn, SimdKernelFn, SimdKernelFn) {
        if is_x86_feature_detected!("avx2") {
            // Wrap the unsafe target_feature functions in safe function pointers.
            // The runtime detection guarantees AVX2 is available.
            (
                |a, b| unsafe { avx2_dot_product(a, b) },
                |a, b| unsafe { avx2_squared_l2(a, b) },
                |a, b| unsafe { avx2_manhattan(a, b) },
            )
        } else if is_x86_feature_detected!("sse4.1") {
            (
                |a, b| unsafe { sse41_dot_product(a, b) },
                |a, b| unsafe { sse41_squared_l2(a, b) },
                |a, b| unsafe { sse41_manhattan(a, b) },
            )
        } else {
            (
                scalar_dot_product as SimdKernelFn,
                scalar_squared_l2 as SimdKernelFn,
                scalar_manhattan as SimdKernelFn,
            )
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn select_kernels() -> (SimdKernelFn, SimdKernelFn, SimdKernelFn) {
        // NEON is mandatory on aarch64, but we still use runtime detection
        // for consistency and future-proofing.
        if std::arch::is_aarch64_feature_detected!("neon") {
            (
                |a, b| unsafe { neon_dot_product(a, b) },
                |a, b| unsafe { neon_squared_l2(a, b) },
                |a, b| unsafe { neon_manhattan(a, b) },
            )
        } else {
            (
                scalar_dot_product as SimdKernelFn,
                scalar_squared_l2 as SimdKernelFn,
                scalar_manhattan as SimdKernelFn,
            )
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn select_kernels() -> (SimdKernelFn, SimdKernelFn, SimdKernelFn) {
        (
            scalar_dot_product as SimdKernelFn,
            scalar_squared_l2 as SimdKernelFn,
            scalar_manhattan as SimdKernelFn,
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
// Public convenience functions
// ---------------------------------------------------------------------------

/// SIMD-accelerated dot product of two f32 slices.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    get_dispatcher().dot_product(a, b)
}

/// SIMD-accelerated squared L2 distance between two f32 slices.
#[inline]
pub fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    get_dispatcher().squared_l2(a, b)
}

/// SIMD-accelerated Manhattan distance between two f32 slices.
#[inline]
pub fn manhattan_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    get_dispatcher().manhattan(a, b)
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
