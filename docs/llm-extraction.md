# LLM Extraction

LLM extraction turns plain text into a typed graph. You hand SwarnDB chunks of text; it asks a language model to pull out entities and the relationships between them, and writes those as typed nodes and edges into a [hybrid collection](graph-first-class.md). Vector similarity tells you what is alike; extraction tells you what is connected and why.

A few ground rules up front:

- Extraction is **hybrid mode only**. Every extraction call is rejected on a `vector_only` or `auto_similarity` collection.
- It is **bring-your-own-key (BYOK)**. SwarnDB never ships an LLM or an LLM key. You point it at your own provider and your own model.
- The extractor writes the graph. Your curation (verify, reject, manual edges) always wins over it; see [Provenance and trust](graph-first-class.md#5-provenance-and-trust).

---

## 1. LLM configuration

Each hybrid collection has its own LLM config. You set it once, and the api key is **write-only**: you send it on set, but no read ever returns it.

```python
client.extraction.set_llm_config(
    "docs_graph",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",            # write-only, never returned
    model_name="openai/gpt-4o-mini",
    temperature=0.0,
    max_tokens=2048,
    timeout_seconds=30,
)

info = client.extraction.get_llm_config("docs_graph")
# info.base_url, info.model_name, info.temperature, info.max_tokens,
# info.timeout_seconds, info.api_key_set  (never the key itself)

# Rotate just the key, keeping everything else
client.extraction.rotate_llm_config("docs_graph", new_api_key="sk-or-...")
```

Config fields:

| Field | Description |
|-------|-------------|
| `base_url` | The OpenAI-compatible endpoint of your provider. |
| `api_key` | Your provider key. Write-only; sealed at rest (see below). |
| `model_name` | The model id to call. |
| `temperature` | Sampling temperature; `0.0` for the most deterministic extraction. |
| `max_tokens` | Cap on output tokens per call. |
| `timeout_seconds` | Per-call timeout to the provider. |

### How the key is protected

When you set or rotate a key, SwarnDB seals it at rest under a server master key before writing it to disk. The master key is supplied through the `SWARNDB_MASTER_KEY` environment variable, which must be the base64 encoding of 32 random bytes.

If `SWARNDB_MASTER_KEY` is not set (or is invalid), the server cannot store keys safely and **extraction config cannot be saved**. Set it before configuring extraction. See [Configuration](configuration.md) for the variable and the rest of the extraction settings.

### OpenAI-compatible providers, including OpenRouter

Any OpenAI-compatible chat endpoint works. For OpenRouter, set:

- `base_url=https://openrouter.ai/api/v1`
- `model_name` to an OpenRouter model id that supports structured JSON output, for example `openai/gpt-4o-mini`.

The model must be able to return JSON, since extraction asks for a structured result.

---

## 2. Ontology

The **ontology** is the vocabulary the extractor is allowed to use: which entity labels exist and which edge types connect them. It keeps the extracted graph consistent instead of letting the model invent a new label every time.

### Built-in templates

Five templates ship with SwarnDB. Pick one as a starting point with `base_template`:

| Template name | Domain |
|---------------|--------|
| `research-papers` | Papers, authors, citations, topics. |
| `legal-contracts` | Parties, clauses, obligations, dates. |
| `ecommerce-catalog` | Products, categories, brands, attributes. |
| `support-tickets` | Tickets, customers, products, issues. |
| `internal-docs` | People, teams, projects, documents. |

```python
client.extraction.set_ontology("docs_graph", base_template="research-papers")
```

### Custom extensions

Add your own entity labels and edge types on top of (or instead of) a template:

```python
from swarndb import EntityLabel, EdgeType

client.extraction.set_ontology(
    "docs_graph",
    base_template="research-papers",
    entity_labels=[EntityLabel(label="Dataset", description="A named dataset")],
    edge_types=[EdgeType(
        edge_type="USES_DATASET",
        description="A paper uses a dataset",
        source_labels=["Paper"],
        target_labels=["Dataset"],
    )],
    replace=False,   # True replaces the template entirely with your extension
)

ontology = client.extraction.get_ontology("docs_graph")
```

Leaving `source_labels` or `target_labels` empty allows any label at that end of the edge.

### Customizing the extraction prompt

By default SwarnDB frames the extraction task with a generic instruction: read the passage, pull out the entities and the typed relationships between them. That works without any tuning, but you know your domain and the model often does not. Two optional knobs on `set_ontology` let you shape that framing. Both are per collection and both are optional.

- **`extra_guidance`** is a short hint that is **added on top of** whichever framing is in effect. Reach for this first. The generic task stays in place and your hint is appended to it, so you can teach the model the few things it cannot guess from the text alone: how a term is used in your data, a date format, which mention counts as which party.
- **`system_prompt`** is a **full override** of the generic framing. Reach for this when a hint is not enough and you want to describe the whole task in your own words, for example to give the model a role or a richer set of rules. It replaces the built-in instruction entirely. You can still pass `extra_guidance` alongside it, and the guidance is appended on top of your override.

Whichever you use, SwarnDB always keeps the machine contract intact. The list of allowed entity labels and edge types from your ontology, and the exact JSON output schema, are always added to the prompt, and a fixed contract footer is always appended last: output only the JSON object, use only the allowed labels and edge types (or propose a new one rather than forcing a bad fit), cite the exact source span and a confidence for everything, and do not invent facts. So a custom prompt can change the task framing and add domain knowledge, but it **cannot** break parsing or step outside your ontology. The worst a bad prompt can do is give you weaker extractions, never a job that fails to parse.

```python
# Add-on guidance: keep the default framing, just teach the model your domain.
client.extraction.set_ontology(
    "contracts_graph",
    base_template="legal-contracts",
    extra_guidance=(
        "Treat 'the Company' as the first party named in the agreement. "
        "Dates are written DD/MM/YYYY."
    ),
)

# Full override: describe the whole task in your own words. The JSON schema,
# your ontology's allowed types, and the contract footer are still enforced.
client.extraction.set_ontology(
    "contracts_graph",
    base_template="legal-contracts",
    system_prompt=(
        "You are a contracts analyst. Read the clause and extract the parties, "
        "the governing law, effective dates, and the obligations between parties."
    ),
)

# Read either value back; an unset knob comes back as None.
ontology = client.extraction.get_ontology("contracts_graph")
print(ontology.system_prompt)   # your override, or None for the default
print(ontology.extra_guidance)  # your hint, or None
```

An empty or whitespace-only value means "use the default", so clearing a knob restores the built-in framing. Both values are saved with the ontology and read back through `get_ontology`. Because the prompt is part of how a chunk is extracted, changing either knob recomputes the [extraction cache](#the-extraction-cache): the next run re-extracts under the new prompt instead of serving the old cached result.

### LLM-proposed additions

While extracting, the model may run into something the ontology does not cover and **propose** a new entity label or edge type rather than silently dropping it. Proposals are held for you to review, so the ontology only grows when you say so.

```python
proposals = client.extraction.list_proposals("docs_graph")
for p in proposals:
    print(p.id, p.kind, p.name, p.description, p.status)

client.extraction.approve_proposal("docs_graph", proposal_id)   # adds it to the ontology
client.extraction.reject_proposal("docs_graph", proposal_id)    # discards it
```

---

## 3. Running extraction

### Preview the cost first

Extraction calls a paid model, so check the estimate before you commit:

```python
from swarndb import Chunk

chunks = [
    Chunk(doc_id="paper-1", chunk_id=0, text="...", embedding=embed("...")),
    Chunk(doc_id="paper-1", chunk_id=1, text="...", embedding=embed("...")),
]

estimate = client.extraction.cost_preview("docs_graph", chunks)
print(estimate.chunks, estimate.estimated_input_tokens,
      estimate.estimated_output_tokens, estimate.estimated_cost_usd,
      estimate.model, estimate.pricing_known)
```

`pricing_known` tells you whether SwarnDB had a price for the model; if not, the token counts are still accurate but the dollar figure is a best effort.

### Read the actual cost after a job

`cost_preview` is an estimate you read before the work. After a job runs, the same job status carries the **actuals**: what the provider actually reported, distinct from the pre-job guess. So you can compare what you were quoted against what you really spent.

```python
status = client.extraction.extraction_status("docs_graph", job_id)
# status.actual_input_tokens, status.actual_output_tokens
# status.actual_cost_usd
# status.usage_provider_reported
```

`usage_provider_reported` is the honesty flag. It stays `True` only while every priced call reported its own token usage. The first time a call's tokens had to be estimated instead, it flips to `False` for the rest of the job, so you always know whether the actuals are fully provider-reported or partly estimated.

Two more things stay honest rather than guessing:

- **Cache hits cost nothing.** A chunk served from the [extraction cache](#the-extraction-cache) never calls the provider, so it adds zero tokens and zero cost to the actuals. The actuals reflect only the calls that actually ran. That is why the actual figures can land well below the pre-job estimate, which counted every chunk.
- **An unknown model reports zero cost, never a made-up number.** If SwarnDB has no price for your model, `actual_cost_usd` stays at zero rather than inventing a figure. The token counts are still real; only the dollar amount is withheld. This mirrors `pricing_known` on the estimate side.

### Start a job and poll it

Extraction runs as an asynchronous job so a large batch does not block your request:

```python
job_id = client.extraction.start_extraction("docs_graph", chunks)

status = client.extraction.extraction_status("docs_graph", job_id)
# status.state: "queued" | "running" | "completed" | "completed_with_errors" | "failed" | "cancelled"
# status.total_chunks, status.processed_chunks,
# status.entities_written, status.edges_written,
# status.cache_hits, status.cache_misses, status.error
# status.failed_chunks, status.chunk_errors   (see Resilience and partial success below)

client.extraction.cancel_extraction("docs_graph", job_id)   # stop a running job
```

A chunk is a unit of extraction: a `doc_id`, a `chunk_id`, the `text`, and an optional `embedding` (used to deduplicate entities when present). You can pass a `Chunk` dataclass or a plain dict with the same keys.

### The extraction cache

Calling the model is the expensive part, so results are cached by the triple `(chunk_hash + model + prompt_version)`. If you re-run extraction on the same text with the same model and prompt version, it is served from cache instead of calling the provider again. The `cache_hits` and `cache_misses` counters on the job status show how much you saved.

### Resilience and partial success

A large ingestion job processes many chunks, and over thousands of LLM calls the occasional chunk will come back with a bad or cut-off reply. SwarnDB isolates that to the single chunk: one chunk failing no longer fails the whole job. The job keeps going, extracts from every other chunk, and tells you afterwards exactly which chunks did not make it. So a one-in-a-thousand hiccup costs you one chunk, not the entire run.

This shows up in the job's final state. There are three terminal outcomes, and you read them as:

| State | What it means |
|-------|---------------|
| `completed` | Every chunk succeeded. Nothing failed. |
| `completed_with_errors` | The job finished and extracted from most chunks, but one or more chunks failed. The graph holds everything that succeeded. |
| `failed` | Every chunk failed, so nothing was extracted. This usually points at a setup problem (a bad api key, an unreachable provider, a misconfigured model). |

When the state is `completed_with_errors`, two fields on the status tell you what went wrong:

- `failed_chunks` is the true total number of chunks that failed.
- `chunk_errors` is a sample of those failures so you can see what happened, each with the `doc_id`, the `chunk_id`, and the `error` message. The sample is capped at 100 entries, so a run with thousands of failures stays small in memory while `failed_chunks` still reports the real total.

```python
status = client.extraction.extraction_status("docs_graph", job_id)

if status.state == "completed_with_errors":
    print(f"{status.failed_chunks} chunk(s) failed; showing up to {len(status.chunk_errors)}:")
    for e in status.chunk_errors:
        print(f"  doc={e.doc_id} chunk={e.chunk_id}: {e.error}")
    # Re-run just the failed chunks once your provider settles, if you want them.
```

A common reason a single chunk's reply comes back unusable is **truncation**: the model hit its output token limit mid-answer and the JSON was cut off. SwarnDB handles this for you. When it sees a reply was cut short for length, it automatically retries that one chunk a single time with a higher output budget (it doubles `max_tokens`, capped at 8192). Most truncations heal on that retry with no action from you, so they never reach `chunk_errors` at all. If you watch metrics, each automatic retry is counted by `swarndb_extraction_truncation_retries_total`.

---

## 4. Re-extraction and document updates

Documents change. SwarnDB lets you re-extract only what changed and keeps your curation intact.

### The replace policy

When a chunk is re-extracted, the **unverified auto-edges** that came from that chunk are replaced with the new extraction. What is left untouched:

- **Manual edges** you created by hand.
- **Verified edges** you confirmed with `verify_edge`.
- **Rejected patterns**: anything you rejected with `reject_edge` is not recreated from the same source.

So an extraction pipeline can run continuously over a curated graph without ever stomping on human decisions.

### Diff a document

Before re-extracting, see what actually changed:

```python
diffs = client.extraction.diff_document("docs_graph", doc_id="paper-1", chunks=new_chunks)
for d in diffs:
    print(d.chunk_id, d.action)   # "unchanged" | "changed" | "new" | "deleted"
```

### Re-extract a document

```python
summary = client.extraction.reextract_document("docs_graph", doc_id="paper-1", chunks=new_chunks)
# summary.job_id (for the changed + new chunks; empty if nothing was enqueued)
# summary.unchanged, summary.changed, summary.added, summary.deleted
# summary.edges_deleted, summary.nodes_deleted
```

`reextract_document` re-extracts only the changed and new chunks, and cleans up after the deleted ones: it removes the auto-edges that came from chunks that no longer exist and any content nodes that are now orphaned. The summary tells you exactly what moved.

### Re-extract a whole corpus, resumably

`reextract_document` handles one document. When you need to re-extract many documents (a whole corpus, or any set of them), looping over `reextract_document` from your own code is fragile: a single hiccup partway through loses all the progress you made, and there is no way to pick up where you left off. The corpus re-extraction API gives you one managed run over the whole set that survives a restart or a cancellation and resumes without redoing the work that already finished.

You hand it a set of documents, each as a `(doc_id, chunks)` pair (or a dict with `doc_id` / `chunks`), and it drives them through the exact same per-document re-extract path one at a time. Progress is written to disk after each document, so a crash loses at most the one document that was in flight. A document supplied with no chunks is treated as a full deletion, the same as the per-document path.

```python
docs = [
    ("paper-1", new_chunks_1),
    ("paper-2", new_chunks_2),
    # ... as many as you supply
]

corpus_job_id = client.extraction.start_corpus_reextraction("docs_graph", docs)

status = client.extraction.corpus_reextraction_status("docs_graph", corpus_job_id)
# status.state: "queued" | "running" | "completed"
#             | "completed_with_errors" | "failed" | "cancelled"
# status.total_documents, status.processed_documents,
# status.failed_documents, status.skipped_documents,
# status.changed_chunks, status.added_chunks, status.deleted_chunks,
# status.edges_deleted, status.nodes_deleted,
# status.entities_written, status.edges_written, status.error
# status.documents: per-document progress, each with .doc_id, .state,
#   .job_id, .changed, .added, .deleted

client.extraction.cancel_corpus_reextraction("docs_graph", corpus_job_id)  # stop the run
```

The run proceeds in the background; poll `corpus_reextraction_status` for progress. Pass `doc_ids` to restrict the run to a subset of the documents you supplied; leaving it empty runs every supplied document.

#### Resuming an interrupted run

The `corpus_job_id` doubles as the **resume token**. If a run is interrupted (a crash, a restart, a cancellation), start it again with the same documents and pass the prior id as `resume_token`. The run continues the same job and skips every document it already completed, so you never pay to re-extract finished work:

```python
# A fresh run mints a new corpus_job_id.
corpus_job_id = client.extraction.start_corpus_reextraction("docs_graph", docs)

# ... the server restarts mid-run ...

# Resume: same documents, prior id as the resume token. Completed documents
# are skipped; the reported counts stay cumulative across the resume.
client.extraction.start_corpus_reextraction(
    "docs_graph", docs, resume_token=corpus_job_id
)
```

The counts you read back after a resume are cumulative across the whole run, not just the documents the resumed pass touched. Skipped documents are counted in `skipped_documents` and show up in `status.documents` with a `skipped` state.

#### Partial success across the corpus

Partial success works the same way it does for a single job (see [Resilience and partial success](#resilience-and-partial-success)), lifted to the document level. A single document that finishes with some failed chunks does not fail the whole corpus run: it pushes the master state to `completed_with_errors`, never to `failed`. The master run only reports `failed` when not a single document succeeded across the whole run, which usually points at a setup problem rather than a bad document. Everything that succeeded is in the graph either way. Because a resumed run counts the documents an earlier pass already completed as successes, a resume that re-attempts only documents that keep failing still reports `completed_with_errors`, not a false `failed`.

#### An honest limitation: you supply the document set

SwarnDB does not enumerate the documents in a collection for you. There is no server-side "every document in this collection" call, so "re-extract the whole corpus" really means "re-extract every document you hand me". You decide which `(doc_id, chunks)` pairs make up the corpus and supply them. This is stated plainly here, in the proto comments, and in the SDK docstrings rather than hidden behind the API: the run is only as complete as the set you provide.

---

## 5. A note on embeddings

SwarnDB stores the vectors **you** provide. It does not generate embeddings for you. You bring your own embedding model, of any kind, and pass the resulting vectors when you insert and (optionally) on each extraction chunk.

This keeps the two concerns separate: embeddings are yours, and the extraction LLM is yours. Embeddings and the LLM are independent providers, since OpenRouter does not offer a general embeddings API. In the bundled example apps, embeddings come from OpenAI (`text-embedding-3-small`, 1536-dim) via `EMBED_API_KEY` / `EMBED_BASE_URL`, and the extraction LLM is reached through OpenRouter via `LLM_API_KEY` / `LLM_BASE_URL`. Neither choice is baked into SwarnDB: it stores whatever vectors you provide, so you can bring your own embeddings from any model or provider.

---

## See also

- [Typed Graph: Overview](graph-first-class.md): the typed graph, manual curation, and the hybrid query engine that extraction feeds.
- [Configuration](configuration.md): `SWARNDB_MASTER_KEY` and the extraction worker, cache, and pricing settings.
- [API Reference](api-reference.md): the ExtractionService RPCs and REST routes.
- [Python SDK](python-sdk.md): the full `ExtractionAPI` method reference (sync and async).
