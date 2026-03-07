use rand::Rng;
use vf_core::distance::{get_distance_fn, DistanceFunction};
use vf_core::types::{DistanceMetricType, VectorId};

pub struct KMeansConfig {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
    pub metric: DistanceMetricType,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 8,
            max_iterations: 100,
            tolerance: 1e-4,
            metric: DistanceMetricType::Euclidean,
        }
    }
}

pub struct ClusterAssignment {
    pub id: VectorId,
    pub cluster: usize,
    pub distance_to_centroid: f32,
}

pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<ClusterAssignment>,
    pub iterations: usize,
    pub converged: bool,
}

pub struct KMeans {
    config: KMeansConfig,
    distance_fn: Box<dyn DistanceFunction>,
}

impl KMeans {
    pub fn new(config: KMeansConfig) -> Self {
        let distance_fn = get_distance_fn(config.metric);
        Self { config, distance_fn }
    }

    /// Run k-means clustering.
    pub fn cluster(&self, vectors: &[(VectorId, &[f32])]) -> KMeansResult {
        let k = self.config.k.min(vectors.len());
        if k == 0 || vectors.is_empty() {
            return KMeansResult {
                centroids: vec![],
                assignments: vec![],
                iterations: 0,
                converged: true,
            };
        }

        let dim = vectors[0].1.len();
        if vectors.iter().any(|(_, v)| v.len() != dim) {
            return KMeansResult {
                centroids: vec![],
                assignments: vec![],
                iterations: 0,
                converged: false,
            };
        }
        let mut centroids = self.init_centroids(vectors);
        let mut assignments = vec![0usize; vectors.len()];
        let mut converged = false;

        let mut iterations = 0;
        for _ in 0..self.config.max_iterations {
            iterations += 1;

            // Assign each vector to nearest centroid
            for (i, (_, vec)) in vectors.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_dist = f32::INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let dist = self.distance_fn.compute(vec, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, (_, vec)) in vectors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, &val) in vec.iter().enumerate() {
                    new_centroids[c][j] += val;
                }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f32;
                    }
                } else {
                    // Keep old centroid for empty clusters
                    new_centroids[c] = centroids[c].clone();
                }
            }

            // Check convergence
            let max_shift: f32 = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| self.distance_fn.compute(old, new))
                .fold(0.0f32, f32::max);

            centroids = new_centroids;

            if max_shift < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Build final assignments with distances
        let final_assignments = vectors
            .iter()
            .enumerate()
            .map(|(i, (id, vec))| {
                let cluster = assignments[i];
                let distance_to_centroid = self.distance_fn.compute(vec, &centroids[cluster]);
                ClusterAssignment {
                    id: *id,
                    cluster,
                    distance_to_centroid,
                }
            })
            .collect();

        KMeansResult {
            centroids,
            assignments: final_assignments,
            iterations,
            converged,
        }
    }

    /// K-means++ initialization.
    fn init_centroids(&self, vectors: &[(VectorId, &[f32])]) -> Vec<Vec<f32>> {
        let k = self.config.k.min(vectors.len());
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // Pick first centroid randomly
        let first = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first].1.to_vec());

        // Pick remaining centroids with probability proportional to squared distance
        for _ in 1..k {
            let distances: Vec<f32> = vectors
                .iter()
                .map(|(_, vec)| {
                    centroids
                        .iter()
                        .map(|c| self.distance_fn.compute(vec, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            let total: f32 = distances.iter().map(|d| d * d).sum();
            if total == 0.0 {
                // All points are on existing centroids
                centroids.push(vectors[rng.gen_range(0..vectors.len())].1.to_vec());
                continue;
            }

            let threshold = rng.r#gen::<f32>() * total;
            let mut cumulative = 0.0f32;
            let mut chosen = 0;
            for (i, d) in distances.iter().enumerate() {
                cumulative += d * d;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids.push(vectors[chosen].1.to_vec());
        }

        centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![
            (1, vec![0.0, 0.0]),
            (2, vec![0.1, 0.1]),
            (3, vec![10.0, 10.0]),
            (4, vec![10.1, 10.1]),
        ];
        let refs: Vec<(VectorId, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let km = KMeans::new(KMeansConfig {
            k: 2,
            ..Default::default()
        });
        let result = km.cluster(&refs);
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), 4);
        // Vectors 1,2 should be in same cluster; 3,4 in same cluster
        assert_eq!(result.assignments[0].cluster, result.assignments[1].cluster);
        assert_eq!(result.assignments[2].cluster, result.assignments[3].cluster);
        assert_ne!(result.assignments[0].cluster, result.assignments[2].cluster);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![
            (1, vec![1.0, 1.0]),
            (2, vec![2.0, 2.0]),
        ];
        let refs: Vec<(VectorId, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let km = KMeans::new(KMeansConfig {
            k: 1,
            ..Default::default()
        });
        let result = km.cluster(&refs);
        assert_eq!(result.centroids.len(), 1);
        assert!(result.assignments.iter().all(|a| a.cluster == 0));
    }

    #[test]
    fn test_kmeans_empty() {
        let refs: Vec<(VectorId, &[f32])> = vec![];
        let km = KMeans::new(KMeansConfig::default());
        let result = km.cluster(&refs);
        assert!(result.centroids.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_kmeans_k_exceeds_n() {
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![(1, vec![1.0])];
        let refs: Vec<(VectorId, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let km = KMeans::new(KMeansConfig {
            k: 10,
            ..Default::default()
        });
        let result = km.cluster(&refs);
        assert_eq!(result.centroids.len(), 1);
    }
}
