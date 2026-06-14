# Typed Graph: Overview

This is the short overview and the place to start for the graph in SwarnDB. It explains the two graph surfaces SwarnDB offers, helps you decide which one you want, and describes what the typed graph gives you. The complete how-to and API reference for the typed graph lives in the [Typed Graph: Complete Guide](graph-guide.md).

SwarnDB started as a vector database that could connect similar vectors automatically (see [Virtual Graph](virtual-graph.md)). It now also offers a collection where the graph is an explicit, typed structure you build and curate, sitting next to your vectors instead of being derived from them. You choose how much graph you want when you create a collection.

---

## Two graph surfaces: which one do I use?

SwarnDB has two distinct graph surfaces. They are different tools for different jobs, and a collection's mode decides which one is active.

- **The virtual graph (SwarnDB's automatic similarity graph).** Edges are derived automatically from vector proximity: any two vectors whose similarity passes a threshold become connected, with no curation on your part. You read this graph with `get_related` and `traverse`, and you can enrich a search with it. Pick the virtual graph when you want "related vectors" and multi-hop discovery driven purely by similarity, with zero setup. It is the automatic option, turned on by the `auto_similarity` mode. The full treatment is in [Virtual Graph](virtual-graph.md).

- **The typed graph (the explicit, first-class graph).** You store typed nodes (content and entity) and typed, directed edges that carry provenance, confidence, and an audit trail. Edges come from manual curation, bulk import, or optional LLM extraction, not from similarity. You then run composable hybrid queries that mix vector search with graph traversal in one plan. Pick the typed graph when you have real relationships (entities, typed edges, properties, who-said-so) and want to combine them with vector search. It is turned on by the `hybrid` mode. The full how-to is in the [Typed Graph: Complete Guide](graph-guide.md).

In short: reach for the virtual graph when "similar to" is the only relationship you need and you want it for free; reach for the typed graph when relationships are explicit facts you curate and want to query alongside your vectors. Vector-only users use neither and see no change.

---

## Collection modes

A collection is created in one of three modes. The mode is fixed at creation time and decides which graph surface is available.

| Mode | What it gives you | Pick it when |
|------|-------------------|--------------|
| `vector_only` | Pure vector search and metadata filtering. No graph at all. | You only need nearest-neighbor search and filters. |
| `auto_similarity` | The virtual graph (SwarnDB's automatic similarity graph): edges appear automatically between vectors that exceed a similarity threshold. | You want the "related vectors" and multi-hop traversal features driven purely by similarity, with no manual curation. |
| `hybrid` | The typed graph of nodes and edges alongside your vectors, plus the composable hybrid query engine and optional LLM extraction. | You want real relationships (entities, typed edges, properties, provenance) and want to combine vector search with graph traversal in one query. |

The mode is passed as a keyword when you create the collection (`mode="vector_only"`, `mode="auto_similarity"`, or `mode="hybrid"`). When `mode` is omitted, the server defaults to vector-only.

### Backward compatibility

Existing collections keep working exactly as they did. In plain terms:

1. A collection created before modes existed (no mode recorded) resolves to `auto_similarity`. It keeps its virtual graph, and `get_related`, `traverse`, and the `include_graph` option on search behave as before.
2. Vector-only behavior is unchanged. If you do not ask for a graph, nothing about search, insert, or storage changes.
3. The graph RPCs and REST routes that existed before are frozen. Old clients keep calling them and get the same answers.
4. The typed-graph and extraction features are additive. They appear only on `hybrid` collections and never alter how the other two modes work.
5. Upgrading the server does not change the mode or behavior of any existing collection.

These are the high-level commitments of the upgrade contract; nothing you already run needs to change.

---

## What the typed graph gives you

In a `hybrid` collection the graph is an explicit, first-class structure you own. This is a conceptual map of what it offers; the field-by-field reference and runnable examples for every piece are in the [Typed Graph: Complete Guide](graph-guide.md).

- **Typed nodes and edges with provenance.** Nodes are either content (a piece of source material, usually carrying an embedding) or entity (a thing such as a person, company, or product). Edges are directed and labelled (for example `CITES`, `WORKS_AT`, `AUTHORED_BY`) and each one records where it came from, a confidence score, whether a human verified it, and an audit trail of how it changed. The key idea that keeps this hybrid and not just a separate graph database: a vector and its content node share the same id, so you attach a typed edge directly to a vector you already have, with no separate mapping to keep in sync.

- **Composable hybrid queries.** A chainable query builder mixes vector similarity and graph traversal in one plan: seed from a vector search or from explicit nodes, walk edges (single hop, k-hop, shortest path), combine result sets, filter by properties, and finish by returning nodes, edges, or paths. One query can move from "vectors similar to this" to "and authored by a verified person" without leaving the engine.

- **Filter-then-search correctness.** When you need the top results among only the nodes that satisfy a condition, you scope the candidate set with the graph first (a filtered scan or a traversal), then rank that exact set by similarity to your query. Because the graph has fixed the candidate set, the ranking is exact over exactly those nodes rather than ranking the whole collection and hoping the matches surface. This is one of the validated wins of the typed graph.

- **Quality-aware and temporal traversal.** Traversal can weight an edge by how much you trust it (its confidence, an explicit weight you store, or recency), so stronger relationships steer the walk and the ranking. Edges can also carry a validity window and a context label, so one graph can hold facts that changed over time and you can ask what the graph looked like at a chosen instant. Both are opt-in: a query that does not ask for them is unchanged.

- **Vector math over a graph-built frontier.** Once the graph has scoped a set of nodes, you can run a vector operation exactly over that set: rank by similarity, by analogy, by diversity, by closeness to a centroid or an interpolated point, and more. Because the frontier is fixed by the graph, each operation runs exactly over those nodes rather than as an approximate search.

- **Optional LLM extraction.** You can let your own LLM read text chunks and propose typed entities and edges to populate the graph, with a human-in-the-loop curation model (verify, reject, manual edges) that always wins over the extractor. This is opt-in and `hybrid`-only; see [LLM Extraction](llm-extraction.md).

These capabilities compose: extraction or manual curation builds the typed graph, the hybrid query builder reads across vectors and structure at once, and the curation operations keep the graph trustworthy over time. For the complete how-to (every method, field, predicate, and query step) see the [Typed Graph: Complete Guide](graph-guide.md).

---

## Where to go next

- [Typed Graph: Complete Guide](graph-guide.md): the complete how-to and API reference for the typed graph. Typed nodes and edges, the full hybrid query builder, predicates, curation, quality-aware and temporal traversal, vector math over a frontier, and async patterns.
- [LLM Extraction](llm-extraction.md): turn text chunks into typed entities and edges using your own LLM (hybrid mode only).
- [Virtual Graph](virtual-graph.md): the virtual graph (SwarnDB's automatic similarity graph) in depth, used by `auto_similarity` collections.
- [Bulk Ingestion](bulk-ingestion.md): load vectors at scale and control when the index is built.
- [API Reference](api-reference.md): the GraphService RPCs and REST routes behind these methods.
- [Python SDK](python-sdk.md): the full method-by-method SDK reference.
