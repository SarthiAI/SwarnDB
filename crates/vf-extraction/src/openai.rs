// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! An adapter for any OpenAI-compatible chat-completions endpoint. It posts the
//! system and user prompts in JSON mode, parses the model's JSON content into an
//! extraction result, reads token usage from the response when present, and
//! retries a bounded number of times on rate limiting.

use std::sync::Arc;
use std::time::Duration;

use metrics::counter;
use serde::{Deserialize, Serialize};

use crate::adapter::{
    ChunkContent, CostEstimate, ExtractionAdapter, ExtractionResult, PromptVersion,
};
use crate::config::LlmConfig;
use crate::cost::{PricingTable, TokenEstimator};
use crate::error::ExtractionError;
use crate::ontology::Ontology;
use crate::prompt::{build_extraction_prompt, build_system_prompt};

/// Maximum number of retries on a 429 before failing.
const MAX_RETRIES: u32 = 3;
/// Base backoff in milliseconds; doubles per attempt and is capped.
const BACKOFF_BASE_MS: u64 = 500;
/// Cap on a single backoff sleep in milliseconds.
const BACKOFF_CAP_MS: u64 = 8_000;
/// Fixed prompt overhead in tokens added to each chunk for cost estimation.
const PROMPT_OVERHEAD_TOKENS: u64 = 400;
/// Assumed output-to-input token ratio for cost estimation.
const OUTPUT_RATIO: f64 = 0.4;
/// Provider-safe upper bound when raising `max_tokens` for a truncation retry,
/// so a length-truncated response cannot demand an unbounded output budget.
const MAX_OUTPUT_TOKENS_CAP: u32 = 8192;
/// Fallback output budget used for the truncation retry when `max_tokens` is
/// unset (0), so the bump-and-retry has a sane base to double from.
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// An adapter over an OpenAI-compatible `/chat/completions` endpoint.
pub struct OpenAICompatibleAdapter {
    client: reqwest::Client,
    base_url: String,
    api_key: zeroize::Zeroizing<String>,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
    #[allow(dead_code)]
    timeout: Duration,
    estimator: Arc<dyn TokenEstimator>,
    pricing: Arc<PricingTable>,
    provider_id: String,
}

impl OpenAICompatibleAdapter {
    /// Build an adapter from an unsealed LLM config, sharing a token estimator
    /// and a pricing table.
    pub fn new(
        config: &LlmConfig,
        estimator: Arc<dyn TokenEstimator>,
        pricing: Arc<PricingTable>,
    ) -> Result<Self, ExtractionError> {
        if config.base_url.trim().is_empty() {
            return Err(ExtractionError::Config("base_url is empty".to_string()));
        }
        if config.model_name.trim().is_empty() {
            return Err(ExtractionError::Config("model_name is empty".to_string()));
        }
        let timeout = Duration::from_secs(config.timeout_seconds.max(1));
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| ExtractionError::Config(format!("http client build failed: {}", e)))?;

        let provider_id = provider_id_from_url(&config.base_url);

        Ok(Self {
            client,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key: config.api_key.clone(),
            model_name: config.model_name.clone(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            timeout,
            estimator,
            pricing,
            provider_id,
        })
    }

    /// The completions endpoint url.
    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

/// Derive a stable provider id from the base url host (falls back to the raw url).
fn provider_id_from_url(base_url: &str) -> String {
    let without_scheme = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url);
    let host = without_scheme
        .split(['/', ':'])
        .next()
        .unwrap_or(without_scheme);
    if host.is_empty() {
        base_url.to_string()
    } else {
        host.to_string()
    }
}

// ── Request / response wire types ────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    temperature: f32,
    max_tokens: u32,
    response_format: ResponseFormat,
    messages: Vec<ChatMessage<'a>>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChoiceMessage,
    /// Why the model stopped. `"length"` means the output was truncated against
    /// the `max_tokens` budget, which drives the bump-and-retry below.
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Compute the exponential backoff for a given attempt, capped.
fn backoff_for(attempt: u32) -> Duration {
    let factor = 1u64 << attempt.min(20);
    let ms = BACKOFF_BASE_MS.saturating_mul(factor).min(BACKOFF_CAP_MS);
    Duration::from_millis(ms)
}

#[async_trait::async_trait]
impl ExtractionAdapter for OpenAICompatibleAdapter {
    async fn extract(
        &self,
        chunk: &ChunkContent,
        ontology: &Ontology,
        _prompt_version: PromptVersion,
    ) -> Result<ExtractionResult, ExtractionError> {
        let user_prompt = build_extraction_prompt(&chunk.text, ontology);
        // The system message: the per-collection custom framing and/or domain
        // guidance when set, always wrapped by the non-negotiable machine
        // contract. Built once per call; it does not vary across retry attempts.
        // link_passages (ADR-017) appends the proven passage-linking guidance so
        // the LLM actually emits @chunk -> entity "mentions" edges, not just the
        // edge type ADR-012 injects into the ontology.
        let system_message = build_system_prompt(
            ontology.system_prompt.as_deref(),
            ontology.extra_guidance.as_deref(),
            ontology.link_passages,
        );

        // One LLM call per extract invocation (the 429/transport backoff retries
        // and a single truncation retry are all part of the same logical call).
        counter!("swarndb_extraction_llm_calls_total", "provider" => self.provider_id.clone())
            .increment(1);

        let endpoint = self.endpoint();
        // The output budget for this attempt. Starts at the configured value and
        // may be raised once for a truncation retry. A 0/unset config means the
        // provider default applies, so we treat it as `DEFAULT_MAX_TOKENS` purely
        // as the base to double from on a bump.
        let mut max_tokens = self.max_tokens;
        // Allow at most one extra attempt, and only on a length-truncation.
        let mut truncation_retry_used = false;

        loop {
            let body = ChatRequest {
                model: &self.model_name,
                temperature: self.temperature,
                max_tokens,
                response_format: ResponseFormat {
                    kind: "json_object",
                },
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: &system_message,
                    },
                    ChatMessage {
                        role: "user",
                        content: &user_prompt,
                    },
                ],
            };

            // Send with the existing bounded 429/transport backoff loop nested
            // inside this attempt.
            let mut attempt: u32 = 0;
            let response = loop {
                let result = self
                    .client
                    .post(&endpoint)
                    .bearer_auth(self.api_key.as_str())
                    .json(&body)
                    .send()
                    .await;

                match result {
                    Ok(resp) => {
                        let status = resp.status();
                        // Retry on rate limiting with bounded exponential backoff.
                        if status.as_u16() == 429 && attempt < MAX_RETRIES {
                            let wait = retry_after(&resp).unwrap_or_else(|| backoff_for(attempt));
                            attempt += 1;
                            tokio::time::sleep(wait).await;
                            continue;
                        }
                        if !status.is_success() {
                            counter!("swarndb_extraction_llm_errors_total", "provider" => self.provider_id.clone())
                                .increment(1);
                            let detail = resp.text().await.unwrap_or_default();
                            return Err(ExtractionError::Llm(format!(
                                "http {}: {}",
                                status.as_u16(),
                                truncate(&detail, 512)
                            )));
                        }
                        break resp;
                    }
                    Err(e) => {
                        // Retry transient transport errors a few times as well.
                        if attempt < MAX_RETRIES {
                            let wait = backoff_for(attempt);
                            attempt += 1;
                            tokio::time::sleep(wait).await;
                            continue;
                        }
                        counter!("swarndb_extraction_llm_errors_total", "provider" => self.provider_id.clone())
                            .increment(1);
                        return Err(ExtractionError::Llm(e.to_string()));
                    }
                }
            };

            let parsed: ChatResponse = match response.json().await {
                Ok(p) => p,
                Err(e) => {
                    counter!("swarndb_extraction_llm_errors_total", "provider" => self.provider_id.clone())
                        .increment(1);
                    return Err(ExtractionError::Parse(format!("response envelope: {}", e)));
                }
            };

            let choice = match parsed.choices.first() {
                Some(c) => c,
                None => {
                    counter!("swarndb_extraction_llm_errors_total", "provider" => self.provider_id.clone())
                        .increment(1);
                    return Err(ExtractionError::Parse("response had no choices".to_string()));
                }
            };
            let content = choice.message.content.clone();
            // Truncation signal: the provider's own `finish_reason`, with the
            // completion-token budget as a secondary signal when usage is present.
            let truncated = choice.finish_reason.as_deref() == Some("length")
                || parsed
                    .usage
                    .as_ref()
                    .map(|u| max_tokens > 0 && u.completion_tokens >= max_tokens)
                    .unwrap_or(false);

            let mut result: ExtractionResult = match serde_json::from_str(content.trim()) {
                Ok(r) => r,
                Err(e) => {
                    // A length-truncated response can be salvaged by one retry
                    // with a raised budget; a genuinely malformed (non-truncated)
                    // body cannot, so it fails immediately as before.
                    if truncated && !truncation_retry_used {
                        truncation_retry_used = true;
                        let base = if max_tokens == 0 {
                            DEFAULT_MAX_TOKENS
                        } else {
                            max_tokens
                        };
                        let bumped = base.saturating_mul(2).min(MAX_OUTPUT_TOKENS_CAP);
                        // Only retry if the bump actually raises the budget;
                        // otherwise we would just truncate again at the cap.
                        if bumped > max_tokens {
                            counter!("swarndb_extraction_truncation_retries_total", "provider" => self.provider_id.clone())
                                .increment(1);
                            tracing::warn!(
                                provider = %self.provider_id,
                                from_max_tokens = max_tokens,
                                to_max_tokens = bumped,
                                "llm output truncated at max_tokens; retrying once with a raised budget"
                            );
                            max_tokens = bumped;
                            continue;
                        }
                    }
                    counter!("swarndb_extraction_llm_errors_total", "provider" => self.provider_id.clone())
                        .increment(1);
                    return Err(ExtractionError::Parse(format!("content json: {}", e)));
                }
            };

            // Prefer the provider-reported usage; otherwise estimate and mark the
            // result so post-job actuals stay honest about their source.
            match parsed.usage {
                Some(usage) => {
                    result.input_tokens = usage.prompt_tokens;
                    result.output_tokens = usage.completion_tokens;
                    result.usage_reported = true;
                }
                None => {
                    result.input_tokens = self.estimator.count(&user_prompt) as u32;
                    result.output_tokens = self.estimator.count(&content) as u32;
                    result.usage_reported = false;
                }
            }

            counter!("swarndb_extraction_llm_input_tokens_total")
                .increment(result.input_tokens as u64);
            counter!("swarndb_extraction_llm_output_tokens_total")
                .increment(result.output_tokens as u64);

            return Ok(result);
        }
    }

    fn estimate_cost(&self, chunks: &[ChunkContent]) -> CostEstimate {
        let mut input_tokens: u64 = 0;
        for chunk in chunks {
            input_tokens = input_tokens
                .saturating_add(self.estimator.count(&chunk.text))
                .saturating_add(PROMPT_OVERHEAD_TOKENS);
        }

        // Output tokens assumed as a ratio of input, capped per chunk at max_tokens.
        let per_chunk_cap = (chunks.len() as u64).saturating_mul(self.max_tokens as u64);
        let mut output_tokens = ((input_tokens as f64) * OUTPUT_RATIO).ceil() as u64;
        if per_chunk_cap > 0 {
            output_tokens = output_tokens.min(per_chunk_cap);
        }

        match self.pricing.price(&self.model_name) {
            Some(price) => {
                let cost = (input_tokens as f64 / 1000.0) * price.input_usd_per_1k
                    + (output_tokens as f64 / 1000.0) * price.output_usd_per_1k;
                CostEstimate {
                    chunks: chunks.len(),
                    estimated_input_tokens: input_tokens,
                    estimated_output_tokens: output_tokens,
                    estimated_cost_usd: cost,
                    model: self.model_name.clone(),
                    pricing_known: true,
                }
            }
            None => CostEstimate {
                chunks: chunks.len(),
                estimated_input_tokens: input_tokens,
                estimated_output_tokens: output_tokens,
                estimated_cost_usd: 0.0,
                model: self.model_name.clone(),
                pricing_known: false,
            },
        }
    }

    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_name
    }
}

/// Parse a `Retry-After` header (delta-seconds form) into a duration.
fn retry_after(resp: &reqwest::Response) -> Option<Duration> {
    let value = resp.headers().get(reqwest::header::RETRY_AFTER)?;
    let secs: u64 = value.to_str().ok()?.trim().parse().ok()?;
    Some(Duration::from_secs(secs.min(BACKOFF_CAP_MS / 1000)))
}

/// Truncate a string to at most `max` bytes on a char boundary, for error text.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}
