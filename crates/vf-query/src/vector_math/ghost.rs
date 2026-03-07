use vf_core::distance::{get_distance_fn, DistanceFunction};
use vf_core::types::{DistanceMetricType, VectorId};

pub struct GhostResult {
    pub id: VectorId,
    pub isolation_score: f32,
}

pub struct GhostDetector {
    threshold: f32,
    distance_fn: Box<dyn DistanceFunction>,
}

impl GhostDetector {
    pub fn new(threshold: f32, metric: DistanceMetricType) -> Self {
        Self {
            threshold,
            distance_fn: get_distance_fn(metric),
        }
    }

    /// Detect ghosts: vectors whose min distance to any centroid > threshold.
    pub fn detect(
        &self,
        vectors: &[(VectorId, &[f32])],
        centroids: &[Vec<f32>],
    ) -> Vec<GhostResult> {
        self.isolation_scores(vectors, centroids)
            .into_iter()
            .filter(|r| r.isolation_score > self.threshold)
            .collect()
    }

    /// Compute isolation scores for all vectors (without threshold filtering).
    pub fn isolation_scores(
        &self,
        vectors: &[(VectorId, &[f32])],
        centroids: &[Vec<f32>],
    ) -> Vec<GhostResult> {
        if centroids.is_empty() {
            return vectors
                .iter()
                .map(|(id, _)| GhostResult {
                    id: *id,
                    isolation_score: f32::INFINITY,
                })
                .collect();
        }
        vectors
            .iter()
            .map(|(id, vec)| {
                let min_dist = centroids
                    .iter()
                    .map(|c| self.distance_fn.compute(vec, c))
                    .fold(f32::INFINITY, f32::min);
                GhostResult {
                    id: *id,
                    isolation_score: min_dist,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_ghost() {
        let detector = GhostDetector::new(5.0, DistanceMetricType::Euclidean);
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 1.0]),
            (3, vec![100.0, 100.0]),
        ];
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let centroids = vec![vec![0.5, 0.5]];
        let ghosts = detector.detect(&refs, &centroids);
        assert_eq!(ghosts.len(), 1);
        assert_eq!(ghosts[0].id, 3);
    }

    #[test]
    fn test_no_ghosts() {
        let detector = GhostDetector::new(100.0, DistanceMetricType::Euclidean);
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![(1, vec![0.0]), (2, vec![1.0])];
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let centroids = vec![vec![0.5]];
        assert!(detector.detect(&refs, &centroids).is_empty());
    }

    #[test]
    fn test_isolation_scores() {
        let detector = GhostDetector::new(0.0, DistanceMetricType::Euclidean);
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![(1, vec![0.0]), (2, vec![10.0])];
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let centroids = vec![vec![0.0], vec![10.0]];
        let scores = detector.isolation_scores(&refs, &centroids);
        assert!(scores[0].isolation_score < 0.01);
        assert!(scores[1].isolation_score < 0.01);
    }

    #[test]
    fn test_empty_centroids() {
        let detector = GhostDetector::new(1.0, DistanceMetricType::Euclidean);
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![(1, vec![0.0])];
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let scores = detector.isolation_scores(&refs, &[]);
        assert_eq!(scores[0].isolation_score, f32::INFINITY);
    }
}
