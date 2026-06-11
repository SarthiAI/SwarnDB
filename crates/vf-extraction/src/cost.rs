// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Token estimation and provider pricing. A heuristic estimator counts tokens by
//! a chars-per-token ratio, and a pricing table prices known models per 1k tokens.

use serde::{Deserialize, Serialize};

use crate::error::ExtractionError;

/// Counts the tokens in a piece of text.
pub trait TokenEstimator: Send + Sync {
    /// Estimated token count for `text`.
    fn count(&self, text: &str) -> u64;
}

/// A simple chars-per-token heuristic estimator. The default ratio is 4.0.
pub struct HeuristicEstimator {
    pub chars_per_token: f32,
}

impl HeuristicEstimator {
    /// Build an estimator with an explicit chars-per-token ratio.
    pub fn new(chars_per_token: f32) -> Self {
        let ratio = if chars_per_token > 0.0 {
            chars_per_token
        } else {
            4.0
        };
        Self {
            chars_per_token: ratio,
        }
    }
}

impl Default for HeuristicEstimator {
    fn default() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }
}

impl TokenEstimator for HeuristicEstimator {
    fn count(&self, text: &str) -> u64 {
        let chars = text.chars().count() as f64;
        let ratio = self.chars_per_token as f64;
        // Round up so a partial token still counts as a whole token.
        (chars / ratio).ceil() as u64
    }
}

/// Per-1k-token pricing for one model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPrice {
    pub model: String,
    pub input_usd_per_1k: f64,
    pub output_usd_per_1k: f64,
}

/// A table of model prices, with a built-in default set and a JSON override.
pub struct PricingTable {
    rows: Vec<ModelPrice>,
}

impl PricingTable {
    /// The built-in pricing for a few known OpenAI models. Values are realistic
    /// per-1k-token list prices and are easy to override from JSON.
    pub fn builtin() -> Self {
        Self {
            rows: vec![
                ModelPrice {
                    model: "gpt-4o-mini".to_string(),
                    input_usd_per_1k: 0.00015,
                    output_usd_per_1k: 0.0006,
                },
                ModelPrice {
                    model: "gpt-4o".to_string(),
                    input_usd_per_1k: 0.0025,
                    output_usd_per_1k: 0.01,
                },
                ModelPrice {
                    model: "gpt-4-turbo".to_string(),
                    input_usd_per_1k: 0.01,
                    output_usd_per_1k: 0.03,
                },
                ModelPrice {
                    model: "gpt-3.5-turbo".to_string(),
                    input_usd_per_1k: 0.0005,
                    output_usd_per_1k: 0.0015,
                },
            ],
        }
    }

    /// Load a pricing table from a JSON file holding an array of `ModelPrice`.
    pub fn from_json_path(path: &str) -> Result<Self, ExtractionError> {
        let bytes = std::fs::read(path).map_err(|e| ExtractionError::Io(e.to_string()))?;
        let rows: Vec<ModelPrice> =
            serde_json::from_slice(&bytes).map_err(|e| ExtractionError::Config(e.to_string()))?;
        Ok(Self { rows })
    }

    /// Construct directly from a set of rows.
    pub fn from_rows(rows: Vec<ModelPrice>) -> Self {
        Self { rows }
    }

    /// Look up the price for a model id. Returns `None` for unknown models.
    pub fn price(&self, model: &str) -> Option<&ModelPrice> {
        self.rows.iter().find(|p| p.model == model)
    }

    /// All known rows.
    pub fn rows(&self) -> &[ModelPrice] {
        &self.rows
    }
}

impl Default for PricingTable {
    fn default() -> Self {
        Self::builtin()
    }
}
