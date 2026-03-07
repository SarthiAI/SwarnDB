use vf_core::types::VectorId;

#[derive(Debug, Clone)]
pub struct DiversityResult {
    pub id: VectorId,
    pub relevance_score: f32,
    pub mmr_score: f32,
}

pub struct DiversitySampler;

impl DiversitySampler {
    /// Select k most diverse vectors using MMR.
    /// lambda: 1.0 = pure relevance, 0.0 = pure diversity.
    pub fn mmr(
        query: &[f32],
        candidates: &[(VectorId, &[f32])],
        k: usize,
        lambda: f32,
    ) -> Vec<DiversityResult> {
        let k = k.min(candidates.len());
        if k == 0 || candidates.is_empty() {
            return Vec::new();
        }
        let lambda = lambda.clamp(0.0, 1.0);

        let relevance_scores: Vec<f32> = candidates
            .iter()
            .map(|(_, vec)| cosine_similarity(query, vec))
            .collect();

        let mut selected = Vec::with_capacity(k);
        let mut selected_indices: Vec<usize> = Vec::with_capacity(k);
        let mut is_selected = vec![false; candidates.len()];

        for _ in 0..k {
            let mut best_idx: Option<usize> = None;
            let mut best_mmr = f32::NEG_INFINITY;

            for (i, _) in candidates.iter().enumerate() {
                if is_selected[i] {
                    continue;
                }
                let relevance = relevance_scores[i];
                let max_sim_to_selected = if selected_indices.is_empty() {
                    0.0
                } else {
                    selected_indices
                        .iter()
                        .map(|&s| cosine_similarity(candidates[i].1, candidates[s].1))
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;
                if mmr_score > best_mmr {
                    best_mmr = mmr_score;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                is_selected[idx] = true;
                selected_indices.push(idx);
                selected.push(DiversityResult {
                    id: candidates[idx].0,
                    relevance_score: relevance_scores[idx],
                    mmr_score: best_mmr,
                });
            } else {
                break;
            }
        }

        selected
    }

    /// Select k most diverse vectors from a set (no query — maximize mutual distance).
    pub fn max_diversity(candidates: &[(VectorId, &[f32])], k: usize) -> Vec<VectorId> {
        let k = k.min(candidates.len());
        if k == 0 || candidates.is_empty() {
            return Vec::new();
        }
        if k == 1 {
            return vec![candidates[0].0];
        }

        let n = candidates.len();
        let mut is_selected = vec![false; n];
        let mut selected_indices: Vec<usize> = Vec::with_capacity(k);

        // Find most distant pair
        let mut min_sim = f32::INFINITY;
        let mut pair = (0, 1.min(n - 1));
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(candidates[i].1, candidates[j].1);
                if sim < min_sim {
                    min_sim = sim;
                    pair = (i, j);
                }
            }
        }

        is_selected[pair.0] = true;
        is_selected[pair.1] = true;
        selected_indices.push(pair.0);
        selected_indices.push(pair.1);

        while selected_indices.len() < k {
            let mut best_idx: Option<usize> = None;
            let mut best_max_sim = f32::INFINITY;

            for i in 0..n {
                if is_selected[i] {
                    continue;
                }
                let max_sim = selected_indices
                    .iter()
                    .map(|&s| cosine_similarity(candidates[i].1, candidates[s].1))
                    .fold(f32::NEG_INFINITY, f32::max);
                if max_sim < best_max_sim {
                    best_max_sim = max_sim;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                is_selected[idx] = true;
                selected_indices.push(idx);
            } else {
                break;
            }
        }

        selected_indices.into_iter().map(|i| candidates[i].0).collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
    }

    #[test]
    fn test_mmr_pure_relevance() {
        let query = normalize(&[1.0, 0.0, 0.0]);
        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.7, 0.7, 0.0]);
        let v3 = normalize(&[0.0, 1.0, 0.0]);
        let candidates: Vec<(VectorId, &[f32])> = vec![(1, &v1), (2, &v2), (3, &v3)];
        let results = DiversitySampler::mmr(&query, &candidates, 3, 1.0);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_mmr_empty() {
        let query = [1.0, 0.0];
        let candidates: Vec<(VectorId, &[f32])> = vec![];
        let results = DiversitySampler::mmr(&query, &candidates, 5, 0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_max_diversity() {
        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.99, 0.01, 0.0]);
        let v3 = normalize(&[0.0, 1.0, 0.0]);
        let v4 = normalize(&[0.0, 0.0, 1.0]);
        let candidates: Vec<(VectorId, &[f32])> = vec![(1, &v1), (2, &v2), (3, &v3), (4, &v4)];
        let result = DiversitySampler::max_diversity(&candidates, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_max_diversity_empty() {
        let candidates: Vec<(VectorId, &[f32])> = vec![];
        assert!(DiversitySampler::max_diversity(&candidates, 5).is_empty());
    }
}
