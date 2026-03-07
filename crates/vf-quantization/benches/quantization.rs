// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Quantization benchmark suite: encode/decode throughput, distance computation,
//! and memory comparison across SQ, PQ, and BQ quantizers.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use vf_quantization::binary::BinaryQuantizer;
use vf_quantization::bq_distance::hamming_distance;
use vf_quantization::pq_distance::PqDistanceTable;
use vf_quantization::product::ProductQuantizer;
use vf_quantization::scalar::ScalarQuantizer;
use vf_quantization::sq_distance::{sq_euclidean_distance, sq_euclidean_distance_fast};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn trained_sq(dim: usize, vectors: &[Vec<f32>]) -> ScalarQuantizer {
    let mut sq = ScalarQuantizer::new(dim);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    sq.train(&refs).unwrap();
    sq
}

fn trained_pq(dim: usize, m: usize, vectors: &[Vec<f32>]) -> ProductQuantizer {
    let mut pq = ProductQuantizer::new(dim, m).unwrap();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    pq.train(&refs, 20).unwrap();
    pq
}

fn trained_bq(dim: usize, vectors: &[Vec<f32>]) -> BinaryQuantizer {
    let mut bq = BinaryQuantizer::new(dim);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    bq.train(&refs).unwrap();
    bq
}

// ---------------------------------------------------------------------------
// (1) Quantization Throughput — encode/decode speed
// ---------------------------------------------------------------------------

fn bench_quantization_throughput(c: &mut Criterion) {
    let n = 10_000;
    let train_n = 1_000; // smaller training set for PQ (k-means is slow)

    let mut group = c.benchmark_group("quantization_throughput");
    group.sample_size(20);

    // SQ encode 128-dim
    {
        let dim = 128;
        let vectors = random_vectors(n, dim, 42);
        let sq = trained_sq(dim, &vectors);
        group.bench_function(BenchmarkId::new("sq_encode", "128d_10k"), |b| {
            b.iter(|| {
                for v in &vectors {
                    black_box(sq.quantize(v).unwrap());
                }
            });
        });
    }

    // SQ encode 1536-dim
    {
        let dim = 1536;
        let vectors = random_vectors(n, dim, 43);
        let sq = trained_sq(dim, &vectors);
        group.bench_function(BenchmarkId::new("sq_encode", "1536d_10k"), |b| {
            b.iter(|| {
                for v in &vectors {
                    black_box(sq.quantize(v).unwrap());
                }
            });
        });
    }

    // PQ encode 128-dim, M=16
    {
        let dim = 128;
        let m = 16;
        let vectors = random_vectors(n, dim, 44);
        let train_vecs = random_vectors(train_n, dim, 45);
        let pq = trained_pq(dim, m, &train_vecs);
        group.bench_function(BenchmarkId::new("pq_encode", "128d_M16_10k"), |b| {
            b.iter(|| {
                for v in &vectors {
                    black_box(pq.encode(v).unwrap());
                }
            });
        });
    }

    // BQ encode 128-dim
    {
        let dim = 128;
        let vectors = random_vectors(n, dim, 46);
        let bq = trained_bq(dim, &vectors);
        group.bench_function(BenchmarkId::new("bq_encode", "128d_10k"), |b| {
            b.iter(|| {
                for v in &vectors {
                    black_box(bq.quantize(v).unwrap());
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// (2) Distance Computation — quantized vs full-precision
// ---------------------------------------------------------------------------

fn bench_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_computation");
    group.sample_size(50);

    for &dim in &[128usize, 768] {
        let vectors = random_vectors(2, dim, 50 + dim as u64);

        // -- Full f32 Euclidean baseline --
        {
            let a = &vectors[0];
            let b = &vectors[1];
            group.bench_function(
                BenchmarkId::new("f32_euclidean", format!("{}d", dim)),
                |bench| {
                    bench.iter(|| {
                        let dist: f32 = black_box(a)
                            .iter()
                            .zip(black_box(b).iter())
                            .map(|(&x, &y)| {
                                let d = x - y;
                                d * d
                            })
                            .sum::<f32>()
                            .sqrt();
                        black_box(dist);
                    });
                },
            );
        }

        // -- SQ Euclidean (dequantize path) --
        {
            let sq = trained_sq(dim, &vectors);
            let a_q = sq.quantize(&vectors[0]).unwrap();
            let b_q = sq.quantize(&vectors[1]).unwrap();
            group.bench_function(
                BenchmarkId::new("sq_euclidean_deq", format!("{}d", dim)),
                |bench| {
                    bench.iter(|| {
                        black_box(sq_euclidean_distance(
                            black_box(&a_q),
                            black_box(&b_q),
                            &sq,
                        ));
                    });
                },
            );
        }

        // -- SQ Euclidean (fast u8 path) --
        {
            let sq = trained_sq(dim, &vectors);
            let a_q = sq.quantize(&vectors[0]).unwrap();
            let b_q = sq.quantize(&vectors[1]).unwrap();
            group.bench_function(
                BenchmarkId::new("sq_euclidean_fast", format!("{}d", dim)),
                |bench| {
                    bench.iter(|| {
                        black_box(sq_euclidean_distance_fast(
                            black_box(&a_q),
                            black_box(&b_q),
                            &sq,
                        ));
                    });
                },
            );
        }

        // -- PQ distance table build + lookup --
        // Dimension must be divisible by M; pick M accordingly
        {
            let m = if dim == 128 { 16 } else { 32 };
            let train_vecs = random_vectors(500, dim, 60 + dim as u64);
            let pq = trained_pq(dim, m, &train_vecs);
            let query = &vectors[0];
            let code = pq.encode(&vectors[1]).unwrap();

            group.bench_function(
                BenchmarkId::new("pq_table_build", format!("{}d_M{}", dim, m)),
                |bench| {
                    bench.iter(|| {
                        black_box(PqDistanceTable::build_euclidean(black_box(query), &pq));
                    });
                },
            );

            let table = PqDistanceTable::build_euclidean(query, &pq);
            group.bench_function(
                BenchmarkId::new("pq_table_lookup", format!("{}d_M{}", dim, m)),
                |bench| {
                    bench.iter(|| {
                        black_box(table.distance(black_box(&code)));
                    });
                },
            );
        }

        // -- BQ Hamming distance --
        {
            let bq = trained_bq(dim, &vectors);
            let a_bq = bq.quantize(&vectors[0]).unwrap();
            let b_bq = bq.quantize(&vectors[1]).unwrap();
            group.bench_function(
                BenchmarkId::new("bq_hamming", format!("{}d", dim)),
                |bench| {
                    bench.iter(|| {
                        black_box(hamming_distance(black_box(&a_bq), black_box(&b_bq)));
                    });
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// (3) Memory Comparison — formatted table of bytes-per-vector
// ---------------------------------------------------------------------------

fn bench_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_comparison");

    // Print the memory table once, use a trivial benchmark so criterion runs it.
    group.bench_function("print_table", |b| {
        b.iter(|| {
            let dims = [128, 256, 512, 768, 1024, 1536];

            let mut output = String::new();
            output.push_str(&format!(
                "\n{:<8} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
                "Dim", "F32", "SQ(u8)", "PQ(M=16)", "PQ(M=32)", "BQ"
            ));
            output.push_str(&"-".repeat(62));
            output.push('\n');

            for &dim in &dims {
                let f32_bytes = dim * 4;
                let sq_bytes = dim; // 1 byte per dimension
                let pq16_bytes = 16; // M=16 codes
                let pq32_bytes = 32; // M=32 codes
                let bq_bytes = (dim + 7) / 8; // ceil(dim/8)

                output.push_str(&format!(
                    "{:<8} {:>8} B {:>8} B {:>8} B {:>8} B {:>8} B\n",
                    dim, f32_bytes, sq_bytes, pq16_bytes, pq32_bytes, bq_bytes
                ));
            }

            // Compression ratios for 128-dim
            output.push_str("\nCompression ratios (128-dim vs F32 baseline of 512 B):\n");
            output.push_str(&format!("  SQ:      {:.1}x\n", 512.0 / 128.0));
            output.push_str(&format!("  PQ(M=16): {:.1}x\n", 512.0 / 16.0));
            output.push_str(&format!("  PQ(M=32): {:.1}x\n", 512.0 / 32.0));
            output.push_str(&format!("  BQ:      {:.1}x\n", 512.0 / 16.0));

            black_box(output);
        });
    });

    group.finish();

    // Also print to stderr so it's visible in benchmark output
    let dims = [128, 256, 512, 768, 1024, 1536];
    eprintln!(
        "\n{:<8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Dim", "F32", "SQ(u8)", "PQ(M=16)", "PQ(M=32)", "BQ"
    );
    eprintln!("{}", "-".repeat(62));
    for &dim in &dims {
        let f32_bytes = dim * 4;
        let sq_bytes = dim;
        let pq16_bytes = 16;
        let pq32_bytes = 32;
        let bq_bytes = (dim + 7) / 8;
        eprintln!(
            "{:<8} {:>8} B {:>8} B {:>8} B {:>8} B {:>8} B",
            dim, f32_bytes, sq_bytes, pq16_bytes, pq32_bytes, bq_bytes
        );
    }
    eprintln!("\nCompression ratios (128-dim vs F32 baseline of 512 B):");
    eprintln!("  SQ:       {:.1}x", 512.0 / 128.0);
    eprintln!("  PQ(M=16): {:.1}x", 512.0 / 16.0);
    eprintln!("  PQ(M=32): {:.1}x", 512.0 / 32.0);
    eprintln!("  BQ:       {:.1}x", 512.0 / 16.0);
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_quantization_throughput,
    bench_distance_computation,
    bench_memory_comparison,
);
criterion_main!(benches);
