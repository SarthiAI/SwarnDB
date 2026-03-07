pub struct AnalogyComputer;

impl AnalogyComputer {
    /// Classic analogy: a - b + c (e.g., king - man + woman = queen)
    pub fn analogy(a: &[f32], b: &[f32], c: &[f32]) -> Option<Vec<f32>> {
        if a.len() != b.len() || b.len() != c.len() {
            return None;
        }
        Some(
            a.iter()
                .zip(b.iter())
                .zip(c.iter())
                .map(|((&ai, &bi), &ci)| ai - bi + ci)
                .collect(),
        )
    }

    /// General vector arithmetic: sum of (vector, weight) pairs.
    pub fn arithmetic(terms: &[(&[f32], f32)]) -> Option<Vec<f32>> {
        if terms.is_empty() {
            return None;
        }
        let dim = terms[0].0.len();
        if terms.iter().any(|(v, _)| v.len() != dim) {
            return None;
        }
        let mut result = vec![0.0f32; dim];
        for (vec, weight) in terms {
            for (i, &val) in vec.iter().enumerate() {
                result[i] += val * weight;
            }
        }
        Some(result)
    }

    /// Normalize a vector to unit length in-place.
    pub fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Compute analogy and normalize result.
    pub fn analogy_normalized(a: &[f32], b: &[f32], c: &[f32]) -> Option<Vec<f32>> {
        let mut result = Self::analogy(a, b, c)?;
        Self::normalize(&mut result);
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analogy_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 0.0, 1.0];
        // a - b + c = [0, 2, 4]
        assert_eq!(AnalogyComputer::analogy(&a, &b, &c).unwrap(), vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_arithmetic() {
        let v1 = [1.0, 0.0];
        let v2 = [0.0, 1.0];
        let result = AnalogyComputer::arithmetic(&[(&v1, 1.0), (&v2, -1.0)]).unwrap();
        assert_eq!(result, vec![1.0, -1.0]);
    }

    #[test]
    fn test_normalize() {
        let mut v = [3.0, 4.0];
        AnalogyComputer::normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_analogy_normalized() {
        let a = [1.0, 0.0];
        let b = [0.0, 0.0];
        let c = [0.0, 1.0];
        let result = AnalogyComputer::analogy_normalized(&a, &b, &c).unwrap();
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        assert!(AnalogyComputer::analogy(&[1.0], &[1.0, 2.0], &[1.0]).is_none());
    }
}
