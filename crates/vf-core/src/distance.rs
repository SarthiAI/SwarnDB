// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use crate::simd;
use crate::types::DistanceMetricType;

/// Trait for distance/similarity computation between vectors.
/// All implementations must be Send + Sync for concurrent use.
pub trait DistanceFunction: Send + Sync {
    /// Compute distance between two f32 slices of equal length.
    /// Lower values = more similar for distance metrics (euclidean, manhattan).
    /// Higher values = more similar for similarity metrics (cosine, dot product).
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;

    /// Returns the metric type
    fn metric_type(&self) -> DistanceMetricType;

    /// Whether lower scores mean more similar (true for distances, false for similarities)
    fn lower_is_better(&self) -> bool;
}

/// Cosine similarity: dot(a, b) / (||a|| * ||b||)
/// Range: [-1.0, 1.0] where 1.0 = identical direction
/// We return 1.0 - cosine_similarity so that lower = more similar (consistent with distance)
pub struct CosineDistance;

impl DistanceFunction for CosineDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");

        let dispatcher = simd::get_dispatcher();
        let dot = dispatcher.dot_product(a, b);
        let norm_a = dispatcher.dot_product(a, a);
        let norm_b = dispatcher.dot_product(b, b);

        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 {
            return 1.0; // maximum distance for zero vectors
        }

        1.0 - (dot / denom)
    }

    fn metric_type(&self) -> DistanceMetricType {
        DistanceMetricType::Cosine
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Euclidean (L2) distance: sqrt(sum((a_i - b_i)^2))
/// Range: [0.0, inf) where 0.0 = identical
pub struct EuclideanDistance;

impl DistanceFunction for EuclideanDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
        simd::get_dispatcher().squared_l2(a, b).sqrt()
    }

    fn metric_type(&self) -> DistanceMetricType {
        DistanceMetricType::Euclidean
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Squared Euclidean distance: sum((a_i - b_i)^2)
/// Faster than Euclidean (no sqrt), preserves ordering.
/// Used internally where only relative ordering matters.
pub struct SquaredEuclideanDistance;

impl DistanceFunction for SquaredEuclideanDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
        simd::get_dispatcher().squared_l2(a, b)
    }

    fn metric_type(&self) -> DistanceMetricType {
        DistanceMetricType::Euclidean
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Dot product similarity: sum(a_i * b_i)
/// Range: (-inf, inf) where higher = more similar
/// We negate the result so lower = more similar (consistent convention)
pub struct DotProductDistance;

impl DistanceFunction for DotProductDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
        -simd::get_dispatcher().dot_product(a, b) // negate so lower = more similar
    }

    fn metric_type(&self) -> DistanceMetricType {
        DistanceMetricType::DotProduct
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Manhattan (L1) distance: sum(|a_i - b_i|)
/// Range: [0.0, inf) where 0.0 = identical
pub struct ManhattanDistance;

impl DistanceFunction for ManhattanDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
        simd::get_dispatcher().manhattan(a, b)
    }

    fn metric_type(&self) -> DistanceMetricType {
        DistanceMetricType::Manhattan
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Factory function to get the appropriate distance function for a metric type
pub fn get_distance_fn(metric: DistanceMetricType) -> Box<dyn DistanceFunction> {
    match metric {
        DistanceMetricType::Cosine => Box::new(CosineDistance),
        DistanceMetricType::Euclidean => Box::new(EuclideanDistance),
        DistanceMetricType::DotProduct => Box::new(DotProductDistance),
        DistanceMetricType::Manhattan => Box::new(ManhattanDistance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // === Cosine Distance Tests ===

    #[test]
    fn test_cosine_identical_vectors() {
        let d = CosineDistance;
        let a = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(d.compute(&a, &a), 0.0));
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let d = CosineDistance;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(d.compute(&a, &b), 1.0));
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let d = CosineDistance;
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!(approx_eq(d.compute(&a, &b), 2.0));
    }

    #[test]
    fn test_cosine_known_values() {
        // NumPy reference: 1 - np.dot([1,2,3],[4,5,6]) / (np.linalg.norm([1,2,3]) * np.linalg.norm([4,5,6]))
        // = 1 - 32 / (3.7417 * 8.7749) = 1 - 0.97463 = 0.02537
        let d = CosineDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = d.compute(&a, &b);
        assert!((result - 0.025368).abs() < 1e-4, "cosine distance was {}, expected ~0.025368", result);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let d = CosineDistance;
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(d.compute(&a, &b), 1.0));
    }

    // === Euclidean Distance Tests ===

    #[test]
    fn test_euclidean_identical() {
        let d = EuclideanDistance;
        let a = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(d.compute(&a, &a), 0.0));
    }

    #[test]
    fn test_euclidean_known_values() {
        // np.linalg.norm([1-4, 2-5, 3-6]) = np.linalg.norm([-3,-3,-3]) = sqrt(27) = 5.19615
        let d = EuclideanDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = d.compute(&a, &b);
        assert!((result - 5.19615).abs() < 1e-4, "euclidean distance was {}, expected ~5.19615", result);
    }

    #[test]
    fn test_euclidean_unit_distance() {
        let d = EuclideanDistance;
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!(approx_eq(d.compute(&a, &b), 1.0));
    }

    // === Squared Euclidean Tests ===

    #[test]
    fn test_squared_euclidean() {
        let d = SquaredEuclideanDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = d.compute(&a, &b);
        assert!(approx_eq(result, 27.0));
    }

    // === Dot Product Tests ===

    #[test]
    fn test_dot_product_known_values() {
        // np.dot([1,2,3],[4,5,6]) = 32, negated = -32
        let d = DotProductDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!(approx_eq(d.compute(&a, &b), -32.0));
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let d = DotProductDistance;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(d.compute(&a, &b), 0.0));
    }

    // === Manhattan Distance Tests ===

    #[test]
    fn test_manhattan_identical() {
        let d = ManhattanDistance;
        let a = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(d.compute(&a, &a), 0.0));
    }

    #[test]
    fn test_manhattan_known_values() {
        // sum(|1-4| + |2-5| + |3-6|) = 3 + 3 + 3 = 9
        let d = ManhattanDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!(approx_eq(d.compute(&a, &b), 9.0));
    }

    // === Factory Tests ===

    #[test]
    fn test_factory_creates_correct_metric() {
        let cosine = get_distance_fn(DistanceMetricType::Cosine);
        assert_eq!(cosine.metric_type(), DistanceMetricType::Cosine);
        assert!(cosine.lower_is_better());

        let euclidean = get_distance_fn(DistanceMetricType::Euclidean);
        assert_eq!(euclidean.metric_type(), DistanceMetricType::Euclidean);

        let dot = get_distance_fn(DistanceMetricType::DotProduct);
        assert_eq!(dot.metric_type(), DistanceMetricType::DotProduct);

        let manhattan = get_distance_fn(DistanceMetricType::Manhattan);
        assert_eq!(manhattan.metric_type(), DistanceMetricType::Manhattan);
    }

    // === Consistency Tests ===

    #[test]
    fn test_all_metrics_symmetric() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        for metric in [
            DistanceMetricType::Cosine,
            DistanceMetricType::Euclidean,
            DistanceMetricType::DotProduct,
            DistanceMetricType::Manhattan,
        ] {
            let f = get_distance_fn(metric);
            let d_ab = f.compute(&a, &b);
            let d_ba = f.compute(&b, &a);
            assert!(approx_eq(d_ab, d_ba), "{:?} is not symmetric: {} != {}", metric, d_ab, d_ba);
        }
    }

    #[test]
    fn test_all_metrics_self_distance() {
        let a = vec![1.0, 2.0, 3.0];

        let cosine = CosineDistance;
        assert!(approx_eq(cosine.compute(&a, &a), 0.0));

        let euclidean = EuclideanDistance;
        assert!(approx_eq(euclidean.compute(&a, &a), 0.0));

        let manhattan = ManhattanDistance;
        assert!(approx_eq(manhattan.compute(&a, &a), 0.0));

        // Dot product of a with itself = -||a||^2 (negated)
        let dot = DotProductDistance;
        let expected = -(1.0 + 4.0 + 9.0);
        assert!(approx_eq(dot.compute(&a, &a), expected));
    }

    #[test]
    fn test_high_dimensional() {
        // Test with 1536 dimensions (OpenAI embedding size)
        let a: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.002).collect();

        for metric in [
            DistanceMetricType::Cosine,
            DistanceMetricType::Euclidean,
            DistanceMetricType::DotProduct,
            DistanceMetricType::Manhattan,
        ] {
            let f = get_distance_fn(metric);
            let result = f.compute(&a, &b);
            assert!(result.is_finite(), "{:?} returned non-finite for 1536-dim", metric);
        }
    }
}
