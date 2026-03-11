// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use vf_core::distance::DistanceMetric;
use vf_core::types::{DistanceMetricType, ScoredResult, VectorId};

use crate::traits::{IndexError, VectorIndex};

/// Brute-force (flat/linear scan) index.
/// Computes distances to all stored vectors — O(n) per query.
/// Used as a baseline for correctness testing and for small collections.
pub struct BruteForceIndex {
    vectors: RwLock<HashMap<VectorId, Vec<f32>>>,
    dimension: usize,
    distance_fn: DistanceMetric,
}

impl BruteForceIndex {
    /// Create a new brute-force index
    pub fn new(dimension: usize, metric: DistanceMetricType) -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            dimension,
            distance_fn: DistanceMetric::from_metric_type(metric),
        }
    }

    /// Internal search over a set of vectors (caller must hold read lock)
    fn search_internal(
        &self,
        query: &[f32],
        k: usize,
        candidates: impl Iterator<Item = VectorId>,
        vectors: &HashMap<VectorId, Vec<f32>>,
    ) -> Vec<ScoredResult> {
        // Use a max-heap bounded to k elements
        // We want the k smallest distances, so we use a max-heap
        // and pop the largest when we exceed k
        let mut heap: std::collections::BinaryHeap<(OrderedFloat<f32>, VectorId)> =
            std::collections::BinaryHeap::new();

        for id in candidates {
            if let Some(vec) = vectors.get(&id) {
                let dist = self.distance_fn.compute(query, vec);
                let entry = (OrderedFloat(dist), id);

                if heap.len() < k {
                    heap.push(entry);
                } else if let Some(top) = heap.peek() {
                    if entry.0 < top.0 {
                        heap.pop();
                        heap.push(entry);
                    }
                }
            }
        }

        // Convert to sorted results (closest first)
        let mut results: Vec<ScoredResult> = heap
            .into_iter()
            .map(|(dist, id)| ScoredResult::new(id, dist.into_inner()))
            .collect();
        results.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        results
    }
}

impl VectorIndex for BruteForceIndex {
    fn add(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        let mut vectors = self.vectors.write();
        if vectors.contains_key(&id) {
            return Err(IndexError::AlreadyExists(id));
        }
        vectors.insert(id, vector.to_vec());
        Ok(())
    }

    fn remove(&self, id: VectorId) -> Result<(), IndexError> {
        self.vectors
            .write()
            .remove(&id)
            .map(|_| ())
            .ok_or(IndexError::NotFound(id))
    }

    fn search(&self, query: &[f32], k: usize, _ef_search: Option<usize>) -> Result<Vec<ScoredResult>, IndexError> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        let vectors = self.vectors.read();
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let candidates = vectors.keys().copied();
        Ok(self.search_internal(query, k, candidates, &vectors))
    }

    fn search_with_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        _ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        let vectors = self.vectors.read();
        Ok(self.search_internal(query, k, candidates.iter().copied(), &vectors))
    }

    fn len(&self) -> usize {
        self.vectors.read().len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn contains(&self, id: VectorId) -> bool {
        self.vectors.read().contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> BruteForceIndex {
        let index = BruteForceIndex::new(3, DistanceMetricType::Euclidean);
        index.add(0, &[1.0, 0.0, 0.0]).unwrap();
        index.add(1, &[0.0, 1.0, 0.0]).unwrap();
        index.add(2, &[0.0, 0.0, 1.0]).unwrap();
        index.add(3, &[1.0, 1.0, 0.0]).unwrap();
        index.add(4, &[1.0, 1.0, 1.0]).unwrap();
        index
    }

    #[test]
    fn test_add_and_len() {
        let index = make_index();
        assert_eq!(index.len(), 5);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_add_duplicate() {
        let index = BruteForceIndex::new(3, DistanceMetricType::Euclidean);
        index.add(0, &[1.0, 0.0, 0.0]).unwrap();
        let result = index.add(0, &[0.0, 1.0, 0.0]);
        assert!(matches!(result, Err(IndexError::AlreadyExists(0))));
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let index = BruteForceIndex::new(3, DistanceMetricType::Euclidean);
        let result = index.add(0, &[1.0, 0.0]);
        assert!(matches!(
            result,
            Err(IndexError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_remove() {
        let index = make_index();
        assert!(index.contains(0));
        index.remove(0).unwrap();
        assert!(!index.contains(0));
        assert_eq!(index.len(), 4);
    }

    #[test]
    fn test_remove_nonexistent() {
        let index = BruteForceIndex::new(3, DistanceMetricType::Euclidean);
        assert!(matches!(index.remove(999), Err(IndexError::NotFound(999))));
    }

    #[test]
    fn test_search_euclidean() {
        let index = make_index();
        // Query [1, 0, 0] — should find itself (id=0) as nearest
        let results = index.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 0); // exact match
        assert!((results[0].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_cosine() {
        let index = BruteForceIndex::new(3, DistanceMetricType::Cosine);
        index.add(0, &[1.0, 0.0, 0.0]).unwrap();
        index.add(1, &[0.9, 0.1, 0.0]).unwrap();
        index.add(2, &[0.0, 1.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results[0].id, 0); // exact match
        assert_eq!(results[1].id, 1); // close in direction
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let index = make_index();
        let results = index.search(&[1.0, 0.0, 0.0], 100, None).unwrap();
        assert_eq!(results.len(), 5); // only 5 vectors in index
    }

    #[test]
    fn test_search_empty_index() {
        let index = BruteForceIndex::new(3, DistanceMetricType::Euclidean);
        let results = index.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_candidates() {
        let index = make_index();
        // Only search among vectors 0 and 2
        let candidates = vec![0, 2];
        let results = index
            .search_with_candidates(&[1.0, 0.0, 0.0], 5, &candidates, None)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // closer
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let index = make_index();
        let result = index.search(&[1.0, 0.0], 5, None);
        assert!(matches!(
            result,
            Err(IndexError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_results_sorted_by_distance() {
        let index = make_index();
        let results = index.search(&[0.5, 0.5, 0.5], 5, None).unwrap();
        for i in 1..results.len() {
            assert!(
                results[i].score >= results[i - 1].score,
                "results not sorted: {} >= {} at index {}",
                results[i].score,
                results[i - 1].score,
                i
            );
        }
    }

    #[test]
    fn test_contains() {
        let index = make_index();
        assert!(index.contains(0));
        assert!(index.contains(4));
        assert!(!index.contains(99));
    }

    #[test]
    fn test_search_after_remove() {
        let index = make_index();
        index.remove(0).unwrap();
        let results = index.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        // Vector 0 should not appear in results
        for r in &results {
            assert_ne!(r.id, 0);
        }
    }

    #[test]
    fn test_dot_product_metric() {
        let index = BruteForceIndex::new(3, DistanceMetricType::DotProduct);
        index.add(0, &[1.0, 0.0, 0.0]).unwrap();
        index.add(1, &[0.0, 1.0, 0.0]).unwrap();
        index.add(2, &[2.0, 0.0, 0.0]).unwrap();

        // Query [1, 0, 0]: dot products are 1, 0, 2. Negated: -1, 0, -2.
        // Sorted ascending: -2 (id=2), -1 (id=0), 0 (id=1)
        let results = index.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results[0].id, 2); // highest dot product
        assert_eq!(results[1].id, 0);
        assert_eq!(results[2].id, 1);
    }

    #[test]
    fn test_larger_dataset() {
        // Test with 10K vectors, dim=128
        let index = BruteForceIndex::new(128, DistanceMetricType::Euclidean);
        for i in 0..10_000u64 {
            let vec: Vec<f32> = (0..128).map(|d| (i as f32 * 0.01) + (d as f32 * 0.001)).collect();
            index.add(i, &vec).unwrap();
        }
        assert_eq!(index.len(), 10_000);

        let query: Vec<f32> = (0..128).map(|d| 50.0 * 0.01 + d as f32 * 0.001).collect();
        let results = index.search(&query, 10, None).unwrap();
        assert_eq!(results.len(), 10);
        // The closest should be vector 50 (exact match of the query pattern)
        assert_eq!(results[0].id, 50);
    }
}
