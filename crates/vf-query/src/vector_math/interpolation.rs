pub struct Interpolator;

impl Interpolator {
    /// Linear interpolation: (1-t)*a + t*b
    pub fn lerp(a: &[f32], b: &[f32], t: f32) -> Option<Vec<f32>> {
        if a.len() != b.len() || !(0.0..=1.0).contains(&t) {
            return None;
        }
        Some(a.iter().zip(b.iter()).map(|(&ai, &bi)| (1.0 - t) * ai + t * bi).collect())
    }

    /// Spherical linear interpolation (vectors should be normalized).
    pub fn slerp(a: &[f32], b: &[f32], t: f32) -> Option<Vec<f32>> {
        if a.len() != b.len() || !(0.0..=1.0).contains(&t) {
            return None;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let dot = dot.clamp(-1.0, 1.0);
        let omega = dot.acos();

        // Fall back to LERP for nearly parallel vectors
        if omega.abs() < 1e-6 {
            return Self::lerp(a, b, t);
        }

        let sin_omega = omega.sin();
        let s0 = ((1.0 - t) * omega).sin() / sin_omega;
        let s1 = (t * omega).sin() / sin_omega;

        Some(
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| s0 * ai + s1 * bi)
                .collect(),
        )
    }

    /// Generate n evenly-spaced interpolated vectors between a and b (inclusive).
    pub fn lerp_sequence(a: &[f32], b: &[f32], n: usize) -> Option<Vec<Vec<f32>>> {
        if a.len() != b.len() {
            return None;
        }
        if n < 2 {
            return if n == 1 { Some(vec![a.to_vec()]) } else { Some(vec![]) };
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            result.push(Self::lerp(a, b, t)?);
        }
        Some(result)
    }

    /// Generate n evenly-spaced SLERP interpolated vectors.
    pub fn slerp_sequence(a: &[f32], b: &[f32], n: usize) -> Option<Vec<Vec<f32>>> {
        if a.len() != b.len() {
            return None;
        }
        if n < 2 {
            return if n == 1 { Some(vec![a.to_vec()]) } else { Some(vec![]) };
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            result.push(Self::slerp(a, b, t)?);
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_endpoints() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_eq!(Interpolator::lerp(&a, &b, 0.0).unwrap(), vec![1.0, 0.0]);
        assert_eq!(Interpolator::lerp(&a, &b, 1.0).unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn test_lerp_midpoint() {
        let a = [0.0, 0.0];
        let b = [2.0, 4.0];
        assert_eq!(Interpolator::lerp(&a, &b, 0.5).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_lerp_invalid_t() {
        let a = [1.0];
        let b = [2.0];
        assert!(Interpolator::lerp(&a, &b, -0.1).is_none());
        assert!(Interpolator::lerp(&a, &b, 1.1).is_none());
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let r0 = Interpolator::slerp(&a, &b, 0.0).unwrap();
        let r1 = Interpolator::slerp(&a, &b, 1.0).unwrap();
        assert!((r0[0] - 1.0).abs() < 1e-5);
        assert!((r1[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_slerp_midpoint_unit_norm() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let mid = Interpolator::slerp(&a, &b, 0.5).unwrap();
        let norm: f32 = mid.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lerp_sequence() {
        let a = [0.0];
        let b = [4.0];
        let seq = Interpolator::lerp_sequence(&a, &b, 5).unwrap();
        assert_eq!(seq.len(), 5);
        assert_eq!(seq[0], vec![0.0]);
        assert_eq!(seq[4], vec![4.0]);
        assert!((seq[2][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = [1.0, 2.0];
        let b = [1.0];
        assert!(Interpolator::lerp(&a, &b, 0.5).is_none());
    }
}
