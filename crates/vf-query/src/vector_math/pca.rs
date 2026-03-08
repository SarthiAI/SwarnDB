pub struct PcaResult {
    pub components: Vec<Vec<f32>>,
    pub explained_variance: Vec<f32>,
    pub mean: Vec<f32>,
    pub projected: Vec<Vec<f32>>,
}

pub struct PcaConfig {
    pub n_components: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
}

impl Default for PcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

pub struct Pca {
    config: PcaConfig,
}

impl Pca {
    pub fn new(config: PcaConfig) -> Self {
        Self { config }
    }

    /// Fit PCA on the given vectors and return projected data.
    pub fn fit_transform(&self, vectors: &[&[f32]]) -> Option<PcaResult> {
        if vectors.is_empty() || vectors.len() < 2 {
            return None;
        }
        let dim = vectors[0].len();
        if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
            return None;
        }
        let n_components = self.config.n_components.min(dim);

        // Compute mean
        let mean = compute_mean(vectors);

        // Center data
        let centered: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                v.iter()
                    .zip(mean.iter())
                    .map(|(&vi, &mi)| vi - mi)
                    .collect()
            })
            .collect();

        // Compute covariance matrix
        let cov = compute_covariance(&centered, dim);

        // Power iteration with deflation
        let mut components = Vec::with_capacity(n_components);
        let mut explained_variance = Vec::with_capacity(n_components);
        let mut deflated_cov = cov;

        for _ in 0..n_components {
            let (eigvec, eigval) = power_iteration(
                &deflated_cov,
                dim,
                self.config.max_iterations,
                self.config.tolerance,
            );
            components.push(eigvec.clone());
            explained_variance.push(eigval);
            deflate_matrix(&mut deflated_cov, &eigvec, eigval, dim);
        }

        // Project vectors
        let projected = Self::transform(&components, &mean, vectors);

        Some(PcaResult {
            components,
            explained_variance,
            mean,
            projected,
        })
    }

    /// Project new vectors using already-computed components and mean.
    pub fn transform(components: &[Vec<f32>], mean: &[f32], vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        vectors
            .iter()
            .map(|v| {
                components
                    .iter()
                    .map(|comp| {
                        v.iter()
                            .zip(mean.iter())
                            .zip(comp.iter())
                            .map(|((&vi, &mi), &ci)| (vi - mi) * ci)
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

fn compute_mean(vectors: &[&[f32]]) -> Vec<f32> {
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let mut mean = vec![0.0f32; dim];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            mean[i] += val;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    mean
}

fn compute_covariance(centered: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let n = centered.len() as f32;
    let mut cov = vec![0.0f32; dim * dim];
    for v in centered {
        for i in 0..dim {
            for j in i..dim {
                let val = v[i] * v[j];
                cov[i * dim + j] += val;
                if i != j {
                    cov[j * dim + i] += val;
                }
            }
        }
    }
    for c in &mut cov {
        *c /= n - 1.0;
    }
    cov
}

fn power_iteration(matrix: &[f32], dim: usize, max_iter: usize, tolerance: f32) -> (Vec<f32>, f32) {
    // Deterministic init: use (1, 1, 1, ...) normalized
    let mut v: Vec<f32> = vec![1.0 / (dim as f32).sqrt(); dim];
    let mut eigenvalue = 0.0f32;

    for _ in 0..max_iter {
        // Matrix-vector multiply
        let mut new_v = vec![0.0f32; dim];
        for i in 0..dim {
            for j in 0..dim {
                new_v[i] += matrix[i * dim + j] * v[j];
            }
        }

        // Compute eigenvalue (Rayleigh quotient)
        let new_eigenvalue: f32 = new_v.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();

        // Normalize
        let norm: f32 = new_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut new_v {
                *x /= norm;
            }
        }

        let converged = (new_eigenvalue - eigenvalue).abs() < tolerance;
        eigenvalue = new_eigenvalue;
        v = new_v;

        if converged {
            break;
        }
    }

    (v, eigenvalue)
}

fn deflate_matrix(matrix: &mut [f32], eigvec: &[f32], eigval: f32, dim: usize) {
    for i in 0..dim {
        for j in 0..dim {
            matrix[i * dim + j] -= eigval * eigvec[i] * eigvec[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_2d_to_1d() {
        // Data along y = x line
        let vectors: Vec<&[f32]> = vec![
            &[1.0, 1.0],
            &[2.0, 2.0],
            &[3.0, 3.0],
            &[4.0, 4.0],
            &[5.0, 5.0],
        ];
        let pca = Pca::new(PcaConfig {
            n_components: 1,
            ..Default::default()
        });
        let result = pca.fit_transform(&vectors).unwrap();
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.projected.len(), 5);
        assert_eq!(result.projected[0].len(), 1);
        // First component should be roughly [0.707, 0.707]
        let comp = &result.components[0];
        assert!((comp[0].abs() - comp[1].abs()).abs() < 0.01);
    }

    #[test]
    fn test_pca_preserves_projection_count() {
        let vectors: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]];
        let pca = Pca::new(PcaConfig {
            n_components: 2,
            ..Default::default()
        });
        let result = pca.fit_transform(&vectors).unwrap();
        assert_eq!(result.components.len(), 2);
        for p in &result.projected {
            assert_eq!(p.len(), 2);
        }
    }

    #[test]
    fn test_pca_empty() {
        let pca = Pca::new(PcaConfig::default());
        assert!(pca.fit_transform(&[]).is_none());
    }

    #[test]
    fn test_transform_new_data() {
        let vectors: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[0.0, 0.0]];
        let pca = Pca::new(PcaConfig {
            n_components: 2,
            ..Default::default()
        });
        let result = pca.fit_transform(&vectors).unwrap();
        let new_data: Vec<&[f32]> = vec![&[0.5, 0.5]];
        let projected = Pca::transform(&result.components, &result.mean, &new_data);
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].len(), 2);
    }
}
