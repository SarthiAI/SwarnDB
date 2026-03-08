// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::time::Instant;

use tonic::{Request, Response, Status};

use vf_core::store::InMemoryVectorStore;
use vf_core::types::{DistanceMetricType, VectorId};
use vf_query::vector_math::*;

use crate::convert::parse_distance_metric;
use crate::proto::swarndb::v1 as proto;
use crate::proto::swarndb::v1::vector_math_service_server::VectorMathService;
use crate::state::AppState;

pub struct VectorMathServiceImpl {
    state: AppState,
}

impl VectorMathServiceImpl {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

// ── Conversion helpers ──────────────────────────────────────────────────

fn vec_to_proto(v: &[f32]) -> proto::Vector {
    proto::Vector { values: v.to_vec() }
}

fn proto_to_vec(v: &proto::Vector) -> Vec<f32> {
    v.values.clone()
}

fn get_vectors_from_store(store: &InMemoryVectorStore, ids: &[u64]) -> Vec<(VectorId, Vec<f32>)> {
    if ids.is_empty() {
        store.iter_vector_data()
    } else {
        ids.iter()
            .filter_map(|&id| store.get_f32_data(id).ok().map(|data| (id, data)))
            .collect()
    }
}

fn resolve_metric(metric: &str) -> DistanceMetricType {
    if metric.is_empty() {
        DistanceMetricType::Euclidean
    } else {
        parse_distance_metric(metric).unwrap_or(DistanceMetricType::Euclidean)
    }
}

// ── Service implementation ──────────────────────────────────────────────

#[tonic::async_trait]
impl VectorMathService for VectorMathServiceImpl {
    async fn detect_ghosts(
        &self,
        request: Request<proto::DetectGhostsRequest>,
    ) -> Result<Response<proto::DetectGhostsResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let owned_vectors = get_vectors_from_store(&collection.store, &[]);
        let vectors: Vec<(VectorId, &[f32])> = owned_vectors
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();

        let centroids: Vec<Vec<f32>> = if req.centroids.is_empty() {
            // Auto-cluster to get centroids
            let auto_k = if req.auto_k == 0 { 8 } else { req.auto_k as usize };
            let metric = resolve_metric(&req.metric);
            let config = KMeansConfig {
                k: auto_k,
                metric,
                ..Default::default()
            };
            let km = KMeans::new(config);
            let result = km.cluster(&vectors);
            result.centroids
        } else {
            req.centroids.iter().map(proto_to_vec).collect()
        };

        let metric = resolve_metric(&req.metric);
        let detector = GhostDetector::new(req.threshold, metric);
        let ghosts = detector.detect(&vectors, &centroids);

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::DetectGhostsResponse {
            ghosts: ghosts
                .into_iter()
                .map(|g| proto::GhostVector {
                    id: g.id,
                    isolation_score: g.isolation_score,
                })
                .collect(),
            compute_time_us,
        }))
    }

    async fn cone_search(
        &self,
        request: Request<proto::ConeSearchRequest>,
    ) -> Result<Response<proto::ConeSearchResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let direction = req
            .direction
            .as_ref()
            .map(proto_to_vec)
            .ok_or_else(|| Status::invalid_argument("direction vector is required"))?;

        let owned_vectors = get_vectors_from_store(&collection.store, &[]);
        let vectors: Vec<(VectorId, &[f32])> = owned_vectors
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();

        let results = ConeSearch::search(&direction, req.aperture_radians, &vectors);

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::ConeSearchResponse {
            results: results
                .into_iter()
                .map(|r| proto::ConeSearchResult {
                    id: r.id,
                    cosine_similarity: r.cosine_similarity,
                    angle_radians: r.angle_radians,
                })
                .collect(),
            compute_time_us,
        }))
    }

    async fn compute_centroid(
        &self,
        request: Request<proto::ComputeCentroidRequest>,
    ) -> Result<Response<proto::ComputeCentroidResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let owned_vectors = get_vectors_from_store(&collection.store, &req.vector_ids);
        let vec_slices: Vec<&[f32]> = owned_vectors.iter().map(|(_, v)| v.as_slice()).collect();

        if vec_slices.is_empty() {
            return Err(Status::not_found("no vectors found"));
        }

        let centroid = if !req.weights.is_empty() {
            CentroidComputer::compute_weighted(&vec_slices, &req.weights)
                .ok_or_else(|| Status::invalid_argument("weighted centroid computation failed (dimension or weight mismatch)"))?
        } else {
            CentroidComputer::compute(&vec_slices)
                .ok_or_else(|| Status::internal("centroid computation failed"))?
        };

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::ComputeCentroidResponse {
            centroid: Some(vec_to_proto(&centroid)),
            compute_time_us,
        }))
    }

    async fn interpolate(
        &self,
        request: Request<proto::InterpolateRequest>,
    ) -> Result<Response<proto::InterpolateResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let a = req
            .a
            .as_ref()
            .map(proto_to_vec)
            .ok_or_else(|| Status::invalid_argument("vector 'a' is required"))?;
        let b = req
            .b
            .as_ref()
            .map(proto_to_vec)
            .ok_or_else(|| Status::invalid_argument("vector 'b' is required"))?;

        let method = if req.method.is_empty() { "lerp" } else { &req.method };

        let results = if req.sequence_count > 0 {
            let n = req.sequence_count as usize;
            match method {
                "slerp" => Interpolator::slerp_sequence(&a, &b, n)
                    .ok_or_else(|| Status::invalid_argument("slerp sequence failed (dimension mismatch or invalid input)"))?,
                _ => Interpolator::lerp_sequence(&a, &b, n)
                    .ok_or_else(|| Status::invalid_argument("lerp sequence failed (dimension mismatch or invalid input)"))?,
            }
        } else {
            let t = req.t;
            let result = match method {
                "slerp" => Interpolator::slerp(&a, &b, t)
                    .ok_or_else(|| Status::invalid_argument("slerp failed (dimension mismatch or t out of range)"))?,
                _ => Interpolator::lerp(&a, &b, t)
                    .ok_or_else(|| Status::invalid_argument("lerp failed (dimension mismatch or t out of range)"))?,
            };
            vec![result]
        };

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::InterpolateResponse {
            results: results.iter().map(|v| vec_to_proto(v)).collect(),
            compute_time_us,
        }))
    }

    async fn detect_drift(
        &self,
        request: Request<proto::DetectDriftRequest>,
    ) -> Result<Response<proto::DetectDriftResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let owned_w1 = get_vectors_from_store(&collection.store, &req.window1_ids);
        let owned_w2 = get_vectors_from_store(&collection.store, &req.window2_ids);

        let w1_slices: Vec<&[f32]> = owned_w1.iter().map(|(_, v)| v.as_slice()).collect();
        let w2_slices: Vec<&[f32]> = owned_w2.iter().map(|(_, v)| v.as_slice()).collect();

        let metric = resolve_metric(&req.metric);
        let detector = DriftDetector::new(metric);

        let report = detector
            .detect(&w1_slices, &w2_slices)
            .ok_or_else(|| Status::invalid_argument("drift detection failed (empty windows)"))?;

        let has_drifted = if req.threshold > 0.0 {
            report.centroid_shift > req.threshold
        } else {
            false
        };

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::DetectDriftResponse {
            centroid_shift: report.centroid_shift,
            mean_distance_window1: report.mean_distance_window1,
            mean_distance_window2: report.mean_distance_window2,
            spread_change: report.spread_change,
            has_drifted,
            compute_time_us,
        }))
    }

    async fn cluster(
        &self,
        request: Request<proto::ClusterRequest>,
    ) -> Result<Response<proto::ClusterResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let owned_vectors = get_vectors_from_store(&collection.store, &[]);
        let vectors: Vec<(VectorId, &[f32])> = owned_vectors
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();

        let metric = resolve_metric(&req.metric);
        let config = KMeansConfig {
            k: req.k as usize,
            max_iterations: if req.max_iterations == 0 { 100 } else { req.max_iterations as usize },
            tolerance: if req.tolerance == 0.0 { 1e-4 } else { req.tolerance },
            metric,
        };

        let km = KMeans::new(config);
        let result = km.cluster(&vectors);

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::ClusterResponse {
            centroids: result.centroids.iter().map(|c| vec_to_proto(c)).collect(),
            assignments: result
                .assignments
                .into_iter()
                .map(|a| proto::ClusterAssignmentProto {
                    id: a.id,
                    cluster: a.cluster as u32,
                    distance_to_centroid: a.distance_to_centroid,
                })
                .collect(),
            iterations: result.iterations as u32,
            converged: result.converged,
            compute_time_us,
        }))
    }

    async fn reduce_dimensions(
        &self,
        request: Request<proto::ReduceDimensionsRequest>,
    ) -> Result<Response<proto::ReduceDimensionsResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let owned_vectors = get_vectors_from_store(&collection.store, &req.vector_ids);
        let vec_slices: Vec<&[f32]> = owned_vectors.iter().map(|(_, v)| v.as_slice()).collect();

        let n_components = if req.n_components == 0 { 2 } else { req.n_components as usize };
        let pca = Pca::new(PcaConfig {
            n_components,
            ..Default::default()
        });

        let result = pca
            .fit_transform(&vec_slices)
            .ok_or_else(|| Status::invalid_argument("PCA failed (need at least 2 vectors with matching dimensions)"))?;

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::ReduceDimensionsResponse {
            components: result.components.iter().map(|c| vec_to_proto(c)).collect(),
            explained_variance: result.explained_variance,
            mean: Some(vec_to_proto(&result.mean)),
            projected: result.projected.iter().map(|p| vec_to_proto(p)).collect(),
            compute_time_us,
        }))
    }

    async fn compute_analogy(
        &self,
        request: Request<proto::ComputeAnalogyRequest>,
    ) -> Result<Response<proto::ComputeAnalogyResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let mut result = if !req.terms.is_empty() {
            // General arithmetic mode
            let terms: Vec<(Vec<f32>, f32)> = req
                .terms
                .iter()
                .map(|t| {
                    let vec = t
                        .vector
                        .as_ref()
                        .map(proto_to_vec)
                        .ok_or_else(|| Status::invalid_argument("arithmetic term missing vector"))?;
                    Ok((vec, t.weight))
                })
                .collect::<Result<Vec<_>, Status>>()?;

            let term_refs: Vec<(&[f32], f32)> = terms
                .iter()
                .map(|(v, w)| (v.as_slice(), *w))
                .collect();

            AnalogyComputer::arithmetic(&term_refs)
                .ok_or_else(|| Status::invalid_argument("arithmetic computation failed (dimension mismatch or empty terms)"))?
        } else {
            // Classic analogy mode: a - b + c
            let a = req
                .a
                .as_ref()
                .map(proto_to_vec)
                .ok_or_else(|| Status::invalid_argument("vector 'a' is required"))?;
            let b = req
                .b
                .as_ref()
                .map(proto_to_vec)
                .ok_or_else(|| Status::invalid_argument("vector 'b' is required"))?;
            let c = req
                .c
                .as_ref()
                .map(proto_to_vec)
                .ok_or_else(|| Status::invalid_argument("vector 'c' is required"))?;

            AnalogyComputer::analogy(&a, &b, &c)
                .ok_or_else(|| Status::invalid_argument("analogy computation failed (dimension mismatch)"))?
        };

        if req.normalize {
            AnalogyComputer::normalize(&mut result);
        }

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::ComputeAnalogyResponse {
            result: Some(vec_to_proto(&result)),
            compute_time_us,
        }))
    }

    async fn diversity_sample(
        &self,
        request: Request<proto::DiversitySampleRequest>,
    ) -> Result<Response<proto::DiversitySampleResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let query = req
            .query
            .as_ref()
            .map(proto_to_vec)
            .ok_or_else(|| Status::invalid_argument("query vector is required"))?;

        let owned_candidates = get_vectors_from_store(&collection.store, &req.candidate_ids);
        let candidates: Vec<(VectorId, &[f32])> = owned_candidates
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();

        let k = req.k as usize;
        let lambda = req.lambda;

        let results = DiversitySampler::mmr(&query, &candidates, k, lambda);

        let compute_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(proto::DiversitySampleResponse {
            results: results
                .into_iter()
                .map(|r| proto::DiversitySampleResult {
                    id: r.id,
                    relevance_score: r.relevance_score,
                    mmr_score: r.mmr_score,
                })
                .collect(),
            compute_time_us,
        }))
    }
}
