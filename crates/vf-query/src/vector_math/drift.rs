use vf_core::distance::{get_distance_fn, DistanceFunction};
use vf_core::types::DistanceMetricType;

pub struct DriftReport {
    pub centroid_shift: f32,
    pub mean_distance_window1: f32,
    pub mean_distance_window2: f32,
    pub spread_change: f32,
}

pub struct DriftDetector {
    distance_fn: Box<dyn DistanceFunction>,
}

impl DriftDetector {
    pub fn new(metric: DistanceMetricType) -> Self {
        Self {
            distance_fn: get_distance_fn(metric),
        }
    }

    /// Compare two sets of vectors for distribution drift.
    pub fn detect(&self, window1: &[&[f32]], window2: &[&[f32]]) -> Option<DriftReport> {
        if window1.is_empty() || window2.is_empty() {
            return None;
        }

        let centroid1 = compute_centroid(window1);
        let centroid2 = compute_centroid(window2);

        let centroid_shift = self.distance_fn.compute(&centroid1, &centroid2);

        let mean_dist1 = self.mean_distance_to_centroid(window1, &centroid1);
        let mean_dist2 = self.mean_distance_to_centroid(window2, &centroid2);

        let spread_change = if mean_dist1 > 0.0 {
            mean_dist2 / mean_dist1
        } else {
            1.0
        };

        Some(DriftReport {
            centroid_shift,
            mean_distance_window1: mean_dist1,
            mean_distance_window2: mean_dist2,
            spread_change,
        })
    }

    /// Check if drift exceeds a threshold (based on centroid shift).
    pub fn has_drifted(
        &self,
        window1: &[&[f32]],
        window2: &[&[f32]],
        threshold: f32,
    ) -> Option<bool> {
        self.detect(window1, window2)
            .map(|r| r.centroid_shift > threshold)
    }

    fn mean_distance_to_centroid(&self, vectors: &[&[f32]], centroid: &[f32]) -> f32 {
        if vectors.is_empty() {
            return 0.0;
        }
        let total: f32 = vectors
            .iter()
            .map(|v| self.distance_fn.compute(v, centroid))
            .sum();
        total / vectors.len() as f32
    }
}

fn compute_centroid(vectors: &[&[f32]]) -> Vec<f32> {
    debug_assert!(!vectors.is_empty(), "compute_centroid called with empty input");
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let mut centroid = vec![0.0f32; dim];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for c in &mut centroid {
        *c /= n;
    }
    centroid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_windows() {
        let detector = DriftDetector::new(DistanceMetricType::Euclidean);
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [4.0, 5.0, 6.0];
        let window: Vec<&[f32]> = vec![&v1, &v2];
        let report = detector.detect(&window, &window).unwrap();
        assert!(report.centroid_shift < 1e-6);
        assert!((report.spread_change - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shifted_windows() {
        let detector = DriftDetector::new(DistanceMetricType::Euclidean);
        let w1: Vec<&[f32]> = vec![&[0.0, 0.0], &[1.0, 1.0]];
        let w2: Vec<&[f32]> = vec![&[10.0, 10.0], &[11.0, 11.0]];
        let report = detector.detect(&w1, &w2).unwrap();
        assert!(report.centroid_shift > 10.0);
    }

    #[test]
    fn test_has_drifted() {
        let detector = DriftDetector::new(DistanceMetricType::Euclidean);
        let w1: Vec<&[f32]> = vec![&[0.0], &[1.0]];
        let w2: Vec<&[f32]> = vec![&[100.0], &[101.0]];
        assert!(detector.has_drifted(&w1, &w2, 1.0).unwrap());
        assert!(!detector.has_drifted(&w1, &w2, 1000.0).unwrap());
    }

    #[test]
    fn test_empty_window() {
        let detector = DriftDetector::new(DistanceMetricType::Euclidean);
        let w: Vec<&[f32]> = vec![&[1.0]];
        assert!(detector.detect(&[], &w).is_none());
        assert!(detector.detect(&w, &[]).is_none());
    }
}
