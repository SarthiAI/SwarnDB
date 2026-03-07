// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Concurrency benchmark suite: QPS, build throughput, distance functions, DashMap store.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use vf_core::simd::{dot_product_f32, manhattan_f32, squared_l2_f32};
use vf_core::store::{InMemoryVectorStore, VectorRecord};
use vf_core::types::DistanceMetricType;
use vf_core::vector::VectorData;
use vf_index::hnsw::{HnswIndex, HnswParams};
use vf_index::traits::VectorIndex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

fn build_hnsw_index(vectors: &[Vec<f32>], dim: usize) -> HnswIndex {
    let params = HnswParams::new(16, 100, 50);
    let index = HnswIndex::new(dim, DistanceMetricType::Cosine, params);
    for (i, v) in vectors.iter().enumerate() {
        index.add(i as u64, v).unwrap();
    }
    index
}

// ---------------------------------------------------------------------------
// (a) HNSW Search QPS — single and multi-threaded
// ---------------------------------------------------------------------------

fn bench_hnsw_search_qps(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let k = 10;

    let vectors = random_vectors(n, dim, 42);
    let index = Arc::new(build_hnsw_index(&vectors, dim));

    // Pre-generate query vectors (one per thread-count scenario)
    let queries = random_vectors(64, dim, 99);

    let mut group = c.benchmark_group("hnsw_search_qps");
    group.sample_size(30);

    for thread_count in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &tc| {
                let idx = Arc::clone(&index);
                let qs = &queries;
                b.iter(|| {
                    std::thread::scope(|s| {
                        let handles: Vec<_> = (0..tc)
                            .map(|t| {
                                let idx = Arc::clone(&idx);
                                s.spawn(move || {
                                    // Each thread does multiple searches
                                    for i in 0..8 {
                                        let q = &qs[(t * 8 + i) % qs.len()];
                                        let _ = black_box(idx.search(q, k));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// (b) HNSW Build Throughput — sequential vs parallel
// ---------------------------------------------------------------------------

fn bench_hnsw_build_throughput(c: &mut Criterion) {
    let dim = 128;
    let n_base = 2_000;
    let n_insert = 1_000;

    let base_vectors = random_vectors(n_base, dim, 10);
    let insert_vectors = random_vectors(n_insert, dim, 20);

    let mut group = c.benchmark_group("hnsw_build_throughput");
    group.sample_size(10);

    // Sequential bulk_add
    group.bench_function("bulk_add_sequential", |b| {
        let insert_refs: Vec<(u64, &[f32])> = insert_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| ((n_base + i) as u64, v.as_slice()))
            .collect();
        b.iter(|| {
            let index = build_hnsw_index(&base_vectors, dim);
            black_box(index.bulk_add(&insert_refs).unwrap());
        });
    });

    // Parallel build_parallel
    group.bench_function("build_parallel", |b| {
        let insert_refs: Vec<(u64, &[f32])> = insert_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| ((n_base + i) as u64, v.as_slice()))
            .collect();
        b.iter(|| {
            let index = build_hnsw_index(&base_vectors, dim);
            black_box(index.build_parallel(&insert_refs).unwrap());
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// (c) Distance Function Throughput — SIMD dispatch validation
// ---------------------------------------------------------------------------

fn bench_distance_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_functions");

    for dim in [128, 768] {
        let a = random_vectors(1, dim, 1).into_iter().next().unwrap();
        let b = random_vectors(1, dim, 2).into_iter().next().unwrap();

        group.bench_with_input(
            BenchmarkId::new("dot_product", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(dot_product_f32(black_box(&a), black_box(&b))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("squared_l2", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(squared_l2_f32(black_box(&a), black_box(&b))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("manhattan", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(manhattan_f32(black_box(&a), black_box(&b))));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// (d) DashMap Store Throughput — concurrent reads and writes
// ---------------------------------------------------------------------------

fn bench_dashmap_store(c: &mut Criterion) {
    let dim = 128;
    let n_preload = 5_000;
    let vectors = random_vectors(n_preload, dim, 77);

    let mut group = c.benchmark_group("dashmap_store");
    group.sample_size(30);

    // Concurrent reads
    group.bench_function("concurrent_reads_4t", |b| {
        let store = Arc::new(InMemoryVectorStore::new(dim));
        for (i, v) in vectors.iter().enumerate() {
            store
                .insert(VectorRecord::new(
                    i as u64,
                    VectorData::F32(v.clone()),
                    None,
                ))
                .unwrap();
        }
        b.iter(|| {
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..4)
                    .map(|t| {
                        let st = Arc::clone(&store);
                        s.spawn(move || {
                            for i in 0..100 {
                                let id = ((t * 100 + i) % n_preload) as u64;
                                let _ = black_box(st.get(id));
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    });

    // Concurrent writes
    group.bench_function("concurrent_writes_4t", |b| {
        let write_vectors = random_vectors(400, dim, 88);
        b.iter(|| {
            let store = Arc::new(InMemoryVectorStore::new(dim));
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..4)
                    .map(|t| {
                        let st = Arc::clone(&store);
                        let vecs = &write_vectors;
                        s.spawn(move || {
                            for i in 0..100 {
                                let id = (t * 100 + i) as u64;
                                let v = &vecs[(t * 100 + i) % vecs.len()];
                                let _ = st.insert(VectorRecord::new(
                                    id,
                                    VectorData::F32(v.clone()),
                                    None,
                                ));
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    });

    // Mixed reads + writes
    group.bench_function("mixed_rw_4t", |b| {
        let store = Arc::new(InMemoryVectorStore::new(dim));
        for (i, v) in vectors.iter().enumerate() {
            store
                .insert(VectorRecord::new(
                    i as u64,
                    VectorData::F32(v.clone()),
                    None,
                ))
                .unwrap();
        }
        let extra_vectors = random_vectors(200, dim, 55);
        b.iter(|| {
            std::thread::scope(|s| {
                // 2 reader threads
                for t in 0..2 {
                    let st = Arc::clone(&store);
                    s.spawn(move || {
                        for i in 0..100 {
                            let id = ((t * 100 + i) % n_preload) as u64;
                            let _ = black_box(st.get(id));
                        }
                    });
                }
                // 2 writer threads
                for t in 0..2 {
                    let st = Arc::clone(&store);
                    let vecs = &extra_vectors;
                    s.spawn(move || {
                        for i in 0..100 {
                            let id = (n_preload + t * 100 + i) as u64;
                            let v = &vecs[(t * 100 + i) % vecs.len()];
                            let _ = st.insert(VectorRecord::new(
                                id,
                                VectorData::F32(v.clone()),
                                None,
                            ));
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_hnsw_search_qps,
    bench_hnsw_build_throughput,
    bench_distance_functions,
    bench_dashmap_store,
);
criterion_main!(benches);
