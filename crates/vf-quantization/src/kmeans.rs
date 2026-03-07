// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! K-means clustering utility for product quantization codebook training.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rayon::prelude::*;

/// Result of k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// k centroids, each of `subvector_dim` dimensions.
    pub centroids: Vec<Vec<f32>>,
    /// Cluster assignment (0..k-1) for each training vector.
    pub assignments: Vec<u8>,
}

/// Run k-means clustering on the given data.
///
/// # Arguments
/// * `data` — training subvectors (each slice has the same dimensionality)
/// * `k` — number of clusters (typically 256 for PQ so codes fit in u8)
/// * `max_iters` — maximum Lloyd iterations before stopping
/// * `seed` — RNG seed for reproducible k-means++ initialization
///
/// # Panics
/// Panics if `data` is empty or `k` is 0.
pub fn kmeans(data: &[&[f32]], k: usize, max_iters: usize, seed: u64) -> KMeansResult {
    assert!(!data.is_empty(), "kmeans: data must not be empty");
    assert!(k > 0, "kmeans: k must be > 0");
    assert!(k <= 256, "kmeans: k must be <= 256 (u8 assignment type)");

    let n = data.len();
    let dim = data[0].len();

    // If we have fewer points than clusters, duplicate points to fill centroids.
    let effective_k = k.min(n);

    // --- k-means++ initialization ---
    let mut centroids = kmeans_plus_plus_init(data, effective_k, dim, seed);

    // Pad with copies of the last centroid if effective_k < k.
    while centroids.len() < k {
        centroids.push(centroids.last().unwrap().clone());
    }

    let mut assignments = vec![0u8; n];

    for _iter in 0..max_iters {
        // --- Assignment step (parallel) ---
        let new_assignments: Vec<u8> = data
            .par_iter()
            .map(|point| {
                let mut best_idx = 0u8;
                let mut best_dist = f32::MAX;
                for (ci, centroid) in centroids.iter().enumerate() {
                    let d = squared_euclidean(point, centroid);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = ci as u8;
                    }
                }
                best_idx
            })
            .collect();

        // Check convergence: if assignments didn't change, stop early.
        let converged = new_assignments == assignments;
        assignments = new_assignments;

        if converged && _iter > 0 {
            break;
        }

        // --- Update step: recompute centroids ---
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, point) in data.iter().enumerate() {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for (j, &val) in point.iter().enumerate() {
                sums[c][j] += val;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] * inv;
                }
            }
            // Empty clusters keep their previous centroid (avoids NaN).
        }
    }

    KMeansResult {
        centroids,
        assignments,
    }
}

/// K-means++ initialization: pick initial centroids with probability proportional
/// to squared distance from the nearest existing centroid.
fn kmeans_plus_plus_init(
    data: &[&[f32]],
    k: usize,
    dim: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = data.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // Pick first centroid uniformly at random.
    let first_idx = rng.gen_range(0..n);
    centroids.push(data[first_idx].to_vec());

    // Distance from each point to the nearest centroid chosen so far.
    let mut min_dists = vec![f32::MAX; n];

    for _c in 1..k {
        // Update min distances with the most recently added centroid.
        let last_centroid = centroids.last().unwrap();
        let mut total_weight: f64 = 0.0;

        for i in 0..n {
            let d = squared_euclidean(data[i], last_centroid);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
            total_weight += min_dists[i] as f64;
        }

        // Weighted random selection.
        if total_weight <= 0.0 {
            // All points are already centroids; just pick any remaining.
            centroids.push(data[rng.gen_range(0..n)].to_vec());
            continue;
        }

        let threshold = rng.r#gen::<f64>() * total_weight;
        let mut cumulative: f64 = 0.0;
        let mut chosen = n - 1;
        for i in 0..n {
            cumulative += min_dists[i] as f64;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }

        centroids.push(data[chosen].to_vec());
    }

    debug_assert_eq!(centroids.len(), k);
    debug_assert!(centroids.iter().all(|c| c.len() == dim));

    centroids
}

/// Compute squared Euclidean distance between two vectors.
#[inline]
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        // Two clear clusters in 2D.
        let cluster_a: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![1.0 + (i as f32) * 0.01, 1.0])
            .collect();
        let cluster_b: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![10.0 + (i as f32) * 0.01, 10.0])
            .collect();

        let all: Vec<&[f32]> = cluster_a
            .iter()
            .chain(cluster_b.iter())
            .map(|v| v.as_slice())
            .collect();

        let result = kmeans(&all, 2, 100, 42);

        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), 100);

        // All points in cluster_a should share one label, cluster_b another.
        let label_a = result.assignments[0];
        let label_b = result.assignments[50];
        assert_ne!(label_a, label_b);

        for i in 0..50 {
            assert_eq!(result.assignments[i], label_a);
        }
        for i in 50..100 {
            assert_eq!(result.assignments[i], label_b);
        }
    }
}
