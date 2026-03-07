use vf_core::types::VectorId;

pub struct ConeResult {
    pub id: VectorId,
    pub cosine_similarity: f32,
    pub angle_radians: f32,
}

pub struct ConeSearch;

impl ConeSearch {
    /// Find all vectors within the cone defined by direction and aperture_radians.
    pub fn search(
        direction: &[f32],
        aperture_radians: f32,
        vectors: &[(VectorId, &[f32])],
    ) -> Vec<ConeResult> {
        let cos_aperture = aperture_radians.cos();
        let dir_norm = l2_norm(direction);
        if dir_norm == 0.0 {
            return vec![];
        }

        let mut results: Vec<ConeResult> = vectors
            .iter()
            .filter_map(|(id, vec)| {
                let v_norm = l2_norm(vec);
                if v_norm == 0.0 {
                    return None;
                }
                let cos_sim = dot_product(direction, vec) / (dir_norm * v_norm);
                let cos_sim = cos_sim.clamp(-1.0, 1.0);
                if cos_sim >= cos_aperture {
                    Some(ConeResult {
                        id: *id,
                        cosine_similarity: cos_sim,
                        angle_radians: cos_sim.acos(),
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.cosine_similarity
                .partial_cmp(&a.cosine_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Check if a single vector is within the cone.
    pub fn is_in_cone(direction: &[f32], aperture_radians: f32, vector: &[f32]) -> bool {
        let dir_norm = l2_norm(direction);
        let v_norm = l2_norm(vector);
        if dir_norm == 0.0 || v_norm == 0.0 {
            return false;
        }
        let cos_sim = dot_product(direction, vector) / (dir_norm * v_norm);
        cos_sim >= aperture_radians.cos()
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_cone_search_basic() {
        let direction = [1.0, 0.0];
        let vectors: Vec<(VectorId, Vec<f32>)> = vec![
            (1, vec![1.0, 0.0]),   // 0 degrees
            (2, vec![1.0, 1.0]),   // 45 degrees
            (3, vec![0.0, 1.0]),   // 90 degrees
            (4, vec![-1.0, 0.0]),  // 180 degrees
        ];
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        // 60-degree half-angle cone
        let results = ConeSearch::search(&direction, PI / 3.0, &refs);
        assert_eq!(results.len(), 2); // vectors 1 and 2
        assert_eq!(results[0].id, 1); // closest first
    }

    #[test]
    fn test_is_in_cone() {
        assert!(ConeSearch::is_in_cone(&[1.0, 0.0], PI / 4.0, &[1.0, 0.1]));
        assert!(!ConeSearch::is_in_cone(&[1.0, 0.0], PI / 4.0, &[0.0, 1.0]));
    }

    #[test]
    fn test_zero_direction() {
        let results = ConeSearch::search(&[0.0, 0.0], PI / 4.0, &[(1, &[1.0, 0.0])]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_zero_vector() {
        assert!(!ConeSearch::is_in_cone(&[1.0, 0.0], PI / 2.0, &[0.0, 0.0]));
    }
}
