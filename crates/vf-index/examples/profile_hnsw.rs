// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Profiling binary for HNSW index.
//!
//! Usage:
//!   cargo run --release --example profile_hnsw -p vf-index
//!   cargo flamegraph --example profile_hnsw -p vf-index

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vf_core::types::DistanceMetricType;
use vf_index::hnsw::{HnswIndex, HnswParams};
use vf_index::traits::VectorIndex;

const NUM_VECTORS: usize = 10_000;
const DIMENSION: usize = 128;
const NUM_QUERIES: usize = 1_000;
const K: usize = 10;

fn generate_random_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let params = HnswParams::new(16, 200, 50, 100_000, 24).expect("valid HNSW params");
    let index = HnswIndex::new(DIMENSION, DistanceMetricType::Euclidean, params);

    // --- Insert phase ---
    println!(
        "Inserting {} vectors of dimension {}...",
        NUM_VECTORS, DIMENSION
    );
    let insert_start = Instant::now();
    for i in 0..NUM_VECTORS {
        let vector = generate_random_vector(&mut rng, DIMENSION);
        index.add(i as u64, &vector).expect("insert failed");
    }
    let insert_elapsed = insert_start.elapsed();
    println!(
        "Insert complete: {:.2?} ({:.1} vectors/sec)",
        insert_elapsed,
        NUM_VECTORS as f64 / insert_elapsed.as_secs_f64()
    );

    // --- Search phase ---
    println!("Running {} search queries (k={})...", NUM_QUERIES, K);
    let queries: Vec<Vec<f32>> = (0..NUM_QUERIES)
        .map(|_| generate_random_vector(&mut rng, DIMENSION))
        .collect();

    let search_start = Instant::now();
    let mut total_results = 0usize;
    for query in &queries {
        let results = index.search(query, K, None).expect("search failed");
        total_results += results.len();
    }
    let search_elapsed = search_start.elapsed();
    println!(
        "Search complete: {:.2?} ({:.1} queries/sec, avg {:.1} results/query)",
        search_elapsed,
        NUM_QUERIES as f64 / search_elapsed.as_secs_f64(),
        total_results as f64 / NUM_QUERIES as f64
    );

    // --- Summary ---
    println!("\n--- Summary ---");
    println!("Index size: {} vectors", index.len());
    println!("Total insert time: {:.2?}", insert_elapsed);
    println!("Total search time: {:.2?}", search_elapsed);
    println!(
        "Avg insert latency: {:.2?}",
        insert_elapsed / NUM_VECTORS as u32
    );
    println!(
        "Avg search latency: {:.2?}",
        search_elapsed / NUM_QUERIES as u32
    );
}
