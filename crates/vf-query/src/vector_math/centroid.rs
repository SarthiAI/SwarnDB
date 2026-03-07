pub struct CentroidComputer;

impl CentroidComputer {
    /// Compute unweighted centroid (mean vector).
    pub fn compute(vectors: &[&[f32]]) -> Option<Vec<f32>> {
        if vectors.is_empty() {
            return None;
        }
        let dim = vectors[0].len();
        let n = vectors.len() as f32;
        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            if v.len() != dim {
                return None;
            }
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for c in &mut centroid {
            *c /= n;
        }
        Some(centroid)
    }

    /// Compute weighted centroid.
    pub fn compute_weighted(vectors: &[&[f32]], weights: &[f32]) -> Option<Vec<f32>> {
        if vectors.is_empty() || vectors.len() != weights.len() {
            return None;
        }
        let dim = vectors[0].len();
        let total_weight: f32 = weights.iter().sum();
        if total_weight == 0.0 {
            return None;
        }
        let mut centroid = vec![0.0f32; dim];
        for (v, &w) in vectors.iter().zip(weights.iter()) {
            if v.len() != dim {
                return None;
            }
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val * w;
            }
        }
        for c in &mut centroid {
            *c /= total_weight;
        }
        Some(centroid)
    }

    /// Incremental centroid update: given existing centroid of n vectors, add a new vector.
    pub fn update_incremental(current_centroid: &[f32], n: usize, new_vector: &[f32]) -> Vec<f32> {
        let n_f = n as f32;
        current_centroid
            .iter()
            .zip(new_vector.iter())
            .map(|(&c, &v)| (c * n_f + v) / (n_f + 1.0))
            .collect()
    }

    /// Remove a vector from centroid incrementally.
    pub fn remove_incremental(
        current_centroid: &[f32],
        n: usize,
        removed_vector: &[f32],
    ) -> Option<Vec<f32>> {
        if n <= 1 {
            return None;
        }
        let n_f = n as f32;
        Some(
            current_centroid
                .iter()
                .zip(removed_vector.iter())
                .map(|(&c, &v)| (c * n_f - v) / (n_f - 1.0))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_basic() {
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [3.0, 4.0, 5.0];
        let result = CentroidComputer::compute(&[&v1, &v2]).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_compute_empty() {
        let result = CentroidComputer::compute(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_weighted() {
        let v1 = [0.0, 0.0];
        let v2 = [10.0, 10.0];
        let result = CentroidComputer::compute_weighted(&[&v1, &v2], &[1.0, 3.0]).unwrap();
        assert_eq!(result, vec![7.5, 7.5]);
    }

    #[test]
    fn test_incremental_update() {
        let centroid = [2.0, 3.0];
        let new = [5.0, 6.0];
        let result = CentroidComputer::update_incremental(&centroid, 2, &new);
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn test_incremental_remove() {
        let centroid = [3.0, 4.0];
        let removed = [5.0, 6.0];
        let result = CentroidComputer::remove_incremental(&centroid, 3, &removed).unwrap();
        assert_eq!(result, vec![2.0, 3.0]);
    }

    #[test]
    fn test_remove_last_returns_none() {
        let centroid = [1.0];
        let removed = [1.0];
        assert!(CentroidComputer::remove_incremental(&centroid, 1, &removed).is_none());
    }
}
