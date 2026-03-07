// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! ANN-benchmarks compatible benchmark suite for SwarnDB HNSW index.
//!
//! Supports standard ann-benchmarks datasets: SIFT1M, GIST1M, GloVe-200.
//! Measures recall@k, QPS, build time, and memory usage.
//!
//! Usage:
//!   cargo run --release --example ann_benchmark -p vf-index -- --dataset sift1m --path /data/sift
//!   cargo run --release --example ann_benchmark -p vf-index -- --dataset gist1m --path /data/gist
//!   cargo run --release --example ann_benchmark -p vf-index -- --dataset glove200 --path /data/glove
//!   cargo run --release --example ann_benchmark -p vf-index -- --all --path /data

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use vf_core::types::DistanceMetricType;
use vf_index::hnsw::{HnswIndex, HnswParams};
use vf_index::traits::VectorIndex;

// ---------------------------------------------------------------------------
// Dataset configuration
// ---------------------------------------------------------------------------

/// Configuration for a standard ANN benchmark dataset.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Dataset name (e.g. "sift1m").
    pub name: &'static str,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Number of training (base) vectors.
    pub train_count: usize,
    /// Number of query vectors.
    pub query_count: usize,
    /// Number of neighbors to retrieve.
    pub k: usize,
    /// Distance metric.
    pub metric: DistanceMetricType,
    /// Base vectors filename.
    pub base_file: &'static str,
    /// Query vectors filename.
    pub query_file: &'static str,
    /// Ground truth filename.
    pub groundtruth_file: &'static str,
}

/// SIFT1M: 128-dimensional, L2 distance, 1M base vectors, 10K queries.
pub const SIFT1M: DatasetConfig = DatasetConfig {
    name: "sift1m",
    dimension: 128,
    train_count: 1_000_000,
    query_count: 10_000,
    k: 100,
    metric: DistanceMetricType::Euclidean,
    base_file: "sift_base.fvecs",
    query_file: "sift_query.fvecs",
    groundtruth_file: "sift_groundtruth.ivecs",
};

/// GIST1M: 960-dimensional, L2 distance, 1M base vectors, 1K queries.
pub const GIST1M: DatasetConfig = DatasetConfig {
    name: "gist1m",
    dimension: 960,
    train_count: 1_000_000,
    query_count: 1_000,
    k: 100,
    metric: DistanceMetricType::Euclidean,
    base_file: "gist_base.fvecs",
    query_file: "gist_query.fvecs",
    groundtruth_file: "gist_groundtruth.ivecs",
};

/// GloVe-200: 200-dimensional, cosine similarity, 1.2M base vectors, 10K queries.
pub const GLOVE200: DatasetConfig = DatasetConfig {
    name: "glove200",
    dimension: 200,
    train_count: 1_183_514,
    query_count: 10_000,
    k: 100,
    metric: DistanceMetricType::Cosine,
    base_file: "glove_base.fvecs",
    query_file: "glove_query.fvecs",
    groundtruth_file: "glove_groundtruth.ivecs",
};

/// All supported datasets.
pub const ALL_DATASETS: &[DatasetConfig] = &[SIFT1M, GIST1M, GLOVE200];

// ---------------------------------------------------------------------------
// HNSW parameter configurations to sweep
// ---------------------------------------------------------------------------

/// A named set of HNSW build/search parameters.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub label: String,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

/// Default parameter sweep: varying M and ef_search for recall/speed trade-off.
fn default_param_sweep() -> Vec<HnswConfig> {
    let mut configs = Vec::new();
    for &m in &[16, 32] {
        for &ef_c in &[128, 200] {
            for &ef_s in &[50, 100, 200, 400] {
                configs.push(HnswConfig {
                    label: format!("M={m}_efC={ef_c}_efS={ef_s}"),
                    m,
                    ef_construction: ef_c,
                    ef_search: ef_s,
                });
            }
        }
    }
    configs
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Dataset name.
    pub dataset: String,
    /// Recall@k value (0.0 .. 1.0).
    pub recall_at_k: f64,
    /// Queries per second.
    pub qps: f64,
    /// Index build time in seconds.
    pub build_time_secs: f64,
    /// Approximate memory usage in bytes (estimated from vector data + graph).
    pub memory_bytes: usize,
    /// HNSW parameter description.
    pub index_params: String,
}

// ---------------------------------------------------------------------------
// File I/O: fvecs / ivecs (standard ann-benchmarks format)
// ---------------------------------------------------------------------------

/// Load vectors from an fvecs file (4-byte dimension header + dim * 4-byte floats per vector).
pub fn load_fvecs(path: &Path) -> io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut vectors = Vec::new();

    let mut consumed = 0;
    while consumed < file_len {
        // Read dimension (4 bytes, little-endian i32)
        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)?;
        let dim = i32::from_le_bytes(dim_buf) as usize;
        consumed += 4;

        // Read dim floats
        let mut float_buf = vec![0u8; dim * 4];
        reader.read_exact(&mut float_buf)?;
        consumed += dim * 4;

        let vec: Vec<f32> = float_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vectors.push(vec);
    }

    Ok(vectors)
}

/// Load integer vectors from an ivecs file (4-byte dimension header + dim * 4-byte ints).
pub fn load_ivecs(path: &Path) -> io::Result<Vec<Vec<i32>>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut vectors = Vec::new();

    let mut consumed = 0;
    while consumed < file_len {
        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)?;
        let dim = i32::from_le_bytes(dim_buf) as usize;
        consumed += 4;

        let mut int_buf = vec![0u8; dim * 4];
        reader.read_exact(&mut int_buf)?;
        consumed += dim * 4;

        let vec: Vec<i32> = int_buf
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vectors.push(vec);
    }

    Ok(vectors)
}

// ---------------------------------------------------------------------------
// Recall computation
// ---------------------------------------------------------------------------

/// Compute recall@k: fraction of true k-nearest neighbors found in the results.
pub fn compute_recall(results: &[Vec<u64>], ground_truth: &[Vec<i32>], k: usize) -> f64 {
    assert_eq!(results.len(), ground_truth.len(), "query count mismatch");
    if results.is_empty() {
        return 0.0;
    }

    let mut total_hits = 0usize;
    let mut total_expected = 0usize;

    for (res, gt) in results.iter().zip(ground_truth.iter()) {
        let k_actual = k.min(gt.len());
        let gt_set: std::collections::HashSet<u64> = gt[..k_actual]
            .iter()
            .map(|&id| id as u64)
            .collect();

        let found = res.iter().take(k_actual).filter(|id| gt_set.contains(id)).count();
        total_hits += found;
        total_expected += k_actual;
    }

    total_hits as f64 / total_expected as f64
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Estimate memory used by the index (vector data + overhead per node).
/// This is approximate; real memory includes allocator overhead and graph adjacency.
fn estimate_memory(num_vectors: usize, dimension: usize, m: usize) -> usize {
    let vector_data = num_vectors * dimension * std::mem::size_of::<f32>();
    // Each node has ~M neighbors per layer; average ~1.5 layers; each neighbor = 8 bytes (u64).
    let graph_overhead = num_vectors * m * 3 * std::mem::size_of::<u64>();
    // HashMap/metadata overhead: ~64 bytes per node estimate.
    let metadata = num_vectors * 64;
    vector_data + graph_overhead + metadata
}

// ---------------------------------------------------------------------------
// Run a single benchmark
// ---------------------------------------------------------------------------

/// Run a full benchmark for a given dataset config and HNSW parameters.
pub fn run_benchmark(
    config: &DatasetConfig,
    data_path: &Path,
    hnsw_config: &HnswConfig,
) -> Result<BenchmarkResult, String> {
    let base_path = data_path.join(config.base_file);
    let query_path = data_path.join(config.query_file);
    let gt_path = data_path.join(config.groundtruth_file);

    // --- Load data ---
    println!(
        "  Loading {} base vectors from {} ...",
        config.name,
        base_path.display()
    );
    let base_vectors = load_fvecs(&base_path)
        .map_err(|e| format!("Failed to load base vectors: {e}"))?;
    if base_vectors.is_empty() {
        return Err("No base vectors loaded".into());
    }
    println!("    Loaded {} base vectors (dim={})", base_vectors.len(), base_vectors[0].len());

    println!("  Loading query vectors from {} ...", query_path.display());
    let query_vectors = load_fvecs(&query_path)
        .map_err(|e| format!("Failed to load query vectors: {e}"))?;
    println!("    Loaded {} query vectors", query_vectors.len());

    println!("  Loading ground truth from {} ...", gt_path.display());
    let ground_truth = load_ivecs(&gt_path)
        .map_err(|e| format!("Failed to load ground truth: {e}"))?;
    println!("    Loaded {} ground truth entries", ground_truth.len());

    // Validate dimensions
    if base_vectors[0].len() != config.dimension {
        return Err(format!(
            "Dimension mismatch: expected {}, got {}",
            config.dimension,
            base_vectors[0].len()
        ));
    }

    // --- Build index ---
    println!(
        "  Building HNSW index [{}] ...",
        hnsw_config.label
    );
    let params = HnswParams::new(
        hnsw_config.m,
        hnsw_config.ef_construction,
        hnsw_config.ef_search,
    );
    let index = HnswIndex::new(config.dimension, config.metric.clone(), params);

    let build_start = Instant::now();
    for (i, vec) in base_vectors.iter().enumerate() {
        index
            .add(i as u64, vec)
            .map_err(|e| format!("Failed to add vector {i}: {e}"))?;
        if (i + 1) % 100_000 == 0 {
            println!("    Inserted {}/{}", i + 1, base_vectors.len());
        }
    }
    let build_time = build_start.elapsed();
    println!("    Build time: {:.2}s", build_time.as_secs_f64());

    // --- Search ---
    println!("  Running {} queries (k={}) ...", query_vectors.len(), config.k);
    let mut all_results: Vec<Vec<u64>> = Vec::with_capacity(query_vectors.len());

    let search_start = Instant::now();
    for query in &query_vectors {
        let results = index
            .search(query, config.k)
            .map_err(|e| format!("Search failed: {e}"))?;
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        all_results.push(ids);
    }
    let search_time = search_start.elapsed();

    let qps = query_vectors.len() as f64 / search_time.as_secs_f64();
    println!("    Search time: {:.2}s ({:.0} QPS)", search_time.as_secs_f64(), qps);

    // --- Recall ---
    let recall = compute_recall(&all_results, &ground_truth, config.k);
    println!("    Recall@{}: {:.4}", config.k, recall);

    // --- Memory estimate ---
    let memory = estimate_memory(base_vectors.len(), config.dimension, hnsw_config.m);

    Ok(BenchmarkResult {
        dataset: config.name.to_string(),
        recall_at_k: recall,
        qps,
        build_time_secs: build_time.as_secs_f64(),
        memory_bytes: memory,
        index_params: hnsw_config.label.clone(),
    })
}

// ---------------------------------------------------------------------------
// Result formatting
// ---------------------------------------------------------------------------

fn print_results_table(results: &[BenchmarkResult]) {
    println!();
    println!("{}", "=".repeat(110));
    println!(
        "{:<12} {:<28} {:>10} {:>10} {:>12} {:>14} {:>14}",
        "Dataset", "Params", "Recall@K", "QPS", "Build(s)", "Memory(MB)", "Memory(GB)"
    );
    println!("{}", "-".repeat(110));

    for r in results {
        let mem_mb = r.memory_bytes as f64 / (1024.0 * 1024.0);
        let mem_gb = r.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{:<12} {:<28} {:>10.4} {:>10.0} {:>12.2} {:>14.1} {:>14.3}",
            r.dataset, r.index_params, r.recall_at_k, r.qps, r.build_time_secs, mem_mb, mem_gb,
        );
    }

    println!("{}", "=".repeat(110));
    println!();
}

fn print_csv_header() {
    println!("dataset,params,recall_at_k,qps,build_time_secs,memory_bytes");
}

fn print_csv_row(r: &BenchmarkResult) {
    println!(
        "{},{},{:.6},{:.2},{:.2},{}",
        r.dataset, r.index_params, r.recall_at_k, r.qps, r.build_time_secs, r.memory_bytes,
    );
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct CliArgs {
    dataset: Option<String>,
    data_path: PathBuf,
    run_all: bool,
    csv_output: bool,
    param_config: Option<(usize, usize, usize)>, // (M, ef_construction, ef_search)
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut dataset = None;
    let mut data_path = PathBuf::from(".");
    let mut run_all = false;
    let mut csv_output = false;
    let mut param_config = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" | "-d" => {
                i += 1;
                if i >= args.len() {
                    return Err("--dataset requires a value".into());
                }
                dataset = Some(args[i].to_lowercase());
            }
            "--path" | "-p" => {
                i += 1;
                if i >= args.len() {
                    return Err("--path requires a value".into());
                }
                data_path = PathBuf::from(&args[i]);
            }
            "--all" | "-a" => {
                run_all = true;
            }
            "--csv" => {
                csv_output = true;
            }
            "--params" => {
                // Format: M,ef_construction,ef_search
                i += 1;
                if i >= args.len() {
                    return Err("--params requires M,efC,efS".into());
                }
                let parts: Vec<&str> = args[i].split(',').collect();
                if parts.len() != 3 {
                    return Err("--params format: M,ef_construction,ef_search".into());
                }
                let m: usize = parts[0].parse().map_err(|_| "Invalid M")?;
                let ef_c: usize = parts[1].parse().map_err(|_| "Invalid ef_construction")?;
                let ef_s: usize = parts[2].parse().map_err(|_| "Invalid ef_search")?;
                param_config = Some((m, ef_c, ef_s));
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(format!("Unknown argument: {other}"));
            }
        }
        i += 1;
    }

    if dataset.is_none() && !run_all {
        return Err("Specify --dataset <name> or --all".into());
    }

    Ok(CliArgs {
        dataset,
        data_path,
        run_all,
        csv_output,
        param_config,
    })
}

fn print_usage() {
    println!("SwarnDB ANN Benchmark Suite");
    println!();
    println!("Usage:");
    println!("  ann_benchmark --dataset <name> --path <dir> [--params M,efC,efS] [--csv]");
    println!("  ann_benchmark --all --path <dir> [--csv]");
    println!();
    println!("Datasets: sift1m, gist1m, glove200");
    println!();
    println!("Options:");
    println!("  --dataset, -d   Dataset name (sift1m, gist1m, glove200)");
    println!("  --path, -p      Path to dataset directory containing .fvecs/.ivecs files");
    println!("  --all, -a       Run all datasets (expects subdirs per dataset)");
    println!("  --params        Single HNSW config: M,ef_construction,ef_search");
    println!("  --csv           Output results in CSV format");
    println!("  --help, -h      Print this help");
    println!();
    println!("Expected file layout:");
    println!("  <path>/sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs");
    println!("  <path>/gist_base.fvecs, gist_query.fvecs, gist_groundtruth.ivecs");
    println!("  <path>/glove_base.fvecs, glove_query.fvecs, glove_groundtruth.ivecs");
}

fn get_dataset_config(name: &str) -> Option<DatasetConfig> {
    match name {
        "sift1m" | "sift" => Some(SIFT1M),
        "gist1m" | "gist" => Some(GIST1M),
        "glove200" | "glove" => Some(GLOVE200),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
    };

    // Build parameter list
    let hnsw_configs: Vec<HnswConfig> = if let Some((m, ef_c, ef_s)) = args.param_config {
        vec![HnswConfig {
            label: format!("M={m}_efC={ef_c}_efS={ef_s}"),
            m,
            ef_construction: ef_c,
            ef_search: ef_s,
        }]
    } else {
        default_param_sweep()
    };

    // Collect datasets to benchmark
    let datasets: Vec<DatasetConfig> = if args.run_all {
        ALL_DATASETS.to_vec()
    } else if let Some(ref name) = args.dataset {
        match get_dataset_config(name) {
            Some(cfg) => vec![cfg],
            None => {
                eprintln!("Unknown dataset: {name}");
                eprintln!("Available: sift1m, gist1m, glove200");
                std::process::exit(1);
            }
        }
    } else {
        unreachable!()
    };

    println!("SwarnDB ANN Benchmark Suite");
    println!("===========================");
    println!("Datasets: {}", datasets.iter().map(|d| d.name).collect::<Vec<_>>().join(", "));
    println!("Parameter configs: {}", hnsw_configs.len());
    println!("Data path: {}", args.data_path.display());
    println!();

    if args.csv_output {
        print_csv_header();
    }

    let mut all_results = Vec::new();

    for dataset in &datasets {
        println!("--- {} ---", dataset.name.to_uppercase());

        for hnsw_cfg in &hnsw_configs {
            match run_benchmark(dataset, &args.data_path, hnsw_cfg) {
                Ok(result) => {
                    if args.csv_output {
                        print_csv_row(&result);
                    }
                    all_results.push(result);
                }
                Err(e) => {
                    eprintln!("  FAILED [{}]: {e}", hnsw_cfg.label);
                }
            }
        }
        println!();
    }

    if !args.csv_output && !all_results.is_empty() {
        print_results_table(&all_results);
    }

    if all_results.is_empty() {
        eprintln!("No benchmarks completed successfully.");
        std::process::exit(1);
    }

    println!(
        "Completed {} benchmark runs across {} dataset(s).",
        all_results.len(),
        datasets.len()
    );
}

// ---------------------------------------------------------------------------
// Unit tests for recall computation and file format helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_perfect() {
        let results = vec![vec![0, 1, 2, 3, 4]];
        let gt = vec![vec![0, 1, 2, 3, 4]];
        let recall = compute_recall(&results, &gt, 5);
        assert!((recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_partial() {
        let results = vec![vec![0, 1, 5, 6, 7]];
        let gt = vec![vec![0, 1, 2, 3, 4]];
        let recall = compute_recall(&results, &gt, 5);
        assert!((recall - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_recall_empty() {
        let results: Vec<Vec<u64>> = vec![];
        let gt: Vec<Vec<i32>> = vec![];
        let recall = compute_recall(&results, &gt, 5);
        assert!((recall - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_k_smaller_than_gt() {
        let results = vec![vec![0, 1, 2]];
        let gt = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
        let recall = compute_recall(&results, &gt, 3);
        assert!((recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dataset_configs() {
        assert_eq!(SIFT1M.dimension, 128);
        assert_eq!(GIST1M.dimension, 960);
        assert_eq!(GLOVE200.dimension, 200);
    }

    #[test]
    fn test_memory_estimate() {
        let mem = estimate_memory(1000, 128, 16);
        assert!(mem > 0);
        // Vector data alone: 1000 * 128 * 4 = 512KB
        assert!(mem >= 1000 * 128 * 4);
    }
}
