// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Hamming distance functions for binary quantized vectors.
//!
//! Hamming distance on binary codes approximates angular (cosine) distance,
//! making it ideal for fast first-pass filtering before exact re-ranking.

/// Compute the Hamming distance between two binary codes packed as u64 slices.
///
/// Uses XOR + popcount which maps to a single hardware instruction on
/// modern CPUs (POPCNT on x86, CNT on ARM).
#[inline]
pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Hamming distances from a query to multiple candidates.
pub fn hamming_distance_batch(query: &[u64], candidates: &[&[u64]]) -> Vec<u32> {
    candidates
        .iter()
        .map(|c| hamming_distance(query, c))
        .collect()
}

/// Normalized Hamming distance: hamming / dimension, in range [0.0, 1.0].
///
/// This normalizes by the number of *actual dimensions* (not the number of
/// packed bits), so trailing padding bits don't affect the result.
#[inline]
pub fn normalized_hamming_distance(a: &[u64], b: &[u64], dimension: usize) -> f32 {
    let h = hamming_distance(a, b);
    h as f32 / dimension as f32
}

/// Estimate cosine distance from Hamming distance.
///
/// Based on the relationship between angular distance and Hamming distance
/// for sign-based random projections:
///   angular_similarity = 1 - (2 * hamming / dimension)
///   cosine_distance = 1 - angular_similarity = 2 * hamming / dimension
///
/// This is an approximation; accuracy depends on data distribution
/// and how well binary quantization preserves angular relationships.
#[inline]
pub fn hamming_to_cosine_estimate(hamming: u32, dimension: usize) -> f32 {
    2.0 * hamming as f32 / dimension as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_identical() {
        let a = vec![0xFFFF_FFFF_FFFF_FFFFu64, 0x0000_0000_0000_0000];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_opposite() {
        let a = vec![0u64];
        let b = vec![u64::MAX];
        assert_eq!(hamming_distance(&a, &b), 64);
    }

    #[test]
    fn test_hamming_known() {
        // 0b0101 vs 0b0110 => XOR = 0b0011 => popcount = 2
        let a = vec![0b0101u64];
        let b = vec![0b0110u64];
        assert_eq!(hamming_distance(&a, &b), 2);
    }

    #[test]
    fn test_hamming_batch() {
        let query = vec![0b1010u64];
        let c1 = vec![0b1010u64]; // dist 0
        let c2 = vec![0b0101u64]; // dist 4
        let c3 = vec![0b1000u64]; // dist 1
        let dists = hamming_distance_batch(&query, &[&c1, &c2, &c3]);
        assert_eq!(dists, vec![0, 4, 1]);
    }

    #[test]
    fn test_normalized_hamming() {
        let a = vec![0b0000u64];
        let b = vec![0b0011u64];
        // 2 bits differ out of 4 dimensions
        let norm = normalized_hamming_distance(&a, &b, 4);
        assert!((norm - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_to_cosine() {
        // hamming=0 => cosine_dist=0.0 (identical)
        assert!((hamming_to_cosine_estimate(0, 128) - 0.0).abs() < 1e-6);
        // hamming=64, dim=128 => cosine_dist=1.0 (orthogonal)
        assert!((hamming_to_cosine_estimate(64, 128) - 1.0).abs() < 1e-6);
        // hamming=128, dim=128 => cosine_dist=2.0 (opposite)
        assert!((hamming_to_cosine_estimate(128, 128) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_word_hamming() {
        // 128-dimensional: 2 u64 words
        let a = vec![0u64, 0u64];
        let b = vec![1u64, 1u64]; // 1 bit set in each word
        assert_eq!(hamming_distance(&a, &b), 2);
    }
}
