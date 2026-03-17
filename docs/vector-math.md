# Vector Math Operations

SwarnDB ships with 15+ vector math operations built directly into the engine. These operations run server-side, meaning you do not need external libraries, post-processing pipelines, or round-trip data transfers to perform advanced vector analytics.

All operations are available through both the REST API and gRPC. They work on stored vectors in collections (referenced by ID) or on raw vector inputs provided in the request.

---

## Ghost Vector Detection

Ghost vectors are vectors that sit far from any cluster in your collection. They are isolated, disconnected, and often indicate data quality problems, under-represented concepts, or genuine outliers worth investigating.

### How It Works

The ghost detector computes an isolation score for each vector by measuring its distance to the nearest cluster centroid. Vectors whose isolation score exceeds the threshold are flagged as ghosts.

You can provide your own centroids or let SwarnDB auto-compute them using k-means clustering (controlled by the `auto_k` parameter, default 8).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.0 | Minimum isolation score to classify a vector as a ghost |
| `centroids` | array of vectors | null | Explicit centroid vectors. If omitted, auto-computed via k-means |
| `auto_k` | int | 8 | Number of clusters for auto-centroid computation |
| `metric` | string | "euclidean" | Distance metric for isolation scoring |

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/math/ghosts`

```bash
curl -X POST http://localhost:8080/api/v1/collections/products/math/ghosts \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 2.5,
    "auto_k": 10,
    "metric": "euclidean"
  }'
```

**Response**:

```json
{
  "ghosts": [
    { "id": 4821, "isolation_score": 4.12 },
    { "id": 917, "isolation_score": 3.87 },
    { "id": 2204, "isolation_score": 2.91 }
  ],
  "compute_time_us": 12340
}
```

### Python SDK

```python
from swarndb import SwarnDBClient

client = SwarnDBClient("localhost:50051")

ghosts = client.math.detect_ghosts(
    "products",
    threshold=2.5,
    auto_k=10,
    metric="euclidean",
)

for ghost in ghosts:
    print(f"Vector {ghost.id}: isolation_score = {ghost.isolation_score:.2f}")
```

### Interpreting Results

Higher `isolation_score` means the vector is farther from its nearest centroid, making it more isolated. A vector with score 0.5 is well within a cluster; a vector with score 5.0 is far from any cluster center.

**Use cases**: data quality auditing (find mis-embedded or corrupted vectors), outlier detection in sensor data, identifying under-represented concepts in a knowledge base.

---

## Cone Search

Cone search finds all vectors that fall within an angular cone defined by a direction vector and an aperture angle. Think of it as pointing a flashlight in a direction and finding everything the beam hits.

### How It Works

For each vector in the collection, SwarnDB computes the angle between it and the direction vector. Vectors whose angle is less than or equal to the aperture (in radians) are included in the results.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `direction` | vector (array of floats) | Unit direction vector defining the cone axis |
| `aperture_radians` | float | Half-angle of the cone in radians |

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/math/cone`

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/cone \
  -H "Content-Type: application/json" \
  -d '{
    "direction": [0.5, 0.5, 0.5, 0.5],
    "aperture_radians": 0.3
  }'
```

**Response**:

```json
{
  "results": [
    { "id": 12, "cosine_similarity": 0.98, "angle_radians": 0.05 },
    { "id": 87, "cosine_similarity": 0.95, "angle_radians": 0.18 },
    { "id": 203, "cosine_similarity": 0.91, "angle_radians": 0.27 }
  ],
  "compute_time_us": 5420
}
```

### Python SDK

```python
results = client.math.cone_search(
    "documents",
    direction=[0.5, 0.5, 0.5, 0.5],
    aperture_radians=0.3,
)

for r in results:
    print(f"Vector {r.id}: similarity={r.cosine_similarity:.3f}, "
          f"angle={r.angle_radians:.3f} rad")
```

**Use cases**: directional similarity search, finding vectors that point in roughly the same direction as a concept, filtering by angular proximity rather than distance.

---

## SLERP Interpolation

Spherical Linear Interpolation (SLERP) creates smooth transitions between two vectors by interpolating along the great circle on the unit sphere. This is the mathematically correct way to blend high-dimensional embeddings.

### SLERP vs. LERP

| Method | How it works | Best for |
|--------|-------------|----------|
| LERP | Linear blend: `(1-t)*a + t*b` | Quick approximations, low dimensions |
| SLERP | Great-circle path on the sphere | Embedding transitions, constant angular velocity |

LERP produces a straight line through the vector space, which can cut through the interior of the sphere. SLERP follows the surface, maintaining the norm and producing more meaningful intermediate representations.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a` | vector | required | Start vector |
| `b` | vector | required | End vector |
| `t` | float | 0.0 | Interpolation parameter (0.0 = a, 1.0 = b) |
| `method` | string | "lerp" | `"slerp"` or `"lerp"` |
| `sequence_count` | int | 0 | If > 0, generates a sequence of evenly spaced interpolations |

### REST API

**Endpoint**: `POST /api/v1/math/interpolate`

**Single interpolation**:

```bash
curl -X POST http://localhost:8080/api/v1/math/interpolate \
  -H "Content-Type: application/json" \
  -d '{
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0, 0.0],
    "t": 0.5,
    "method": "slerp"
  }'
```

**Sequence generation** (5 evenly spaced points from a to b):

```bash
curl -X POST http://localhost:8080/api/v1/math/interpolate \
  -H "Content-Type: application/json" \
  -d '{
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0, 0.0],
    "method": "slerp",
    "sequence_count": 5
  }'
```

**Response**:

```json
{
  "results": [
    [1.0, 0.0, 0.0, 0.0],
    [0.81, 0.31, 0.0, 0.0],
    [0.59, 0.59, 0.0, 0.0],
    [0.31, 0.81, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
  ],
  "compute_time_us": 42
}
```

### Python SDK

```python
# Single interpolation at t=0.5
midpoint = client.math.interpolate(
    a=[1.0, 0.0, 0.0, 0.0],
    b=[0.0, 1.0, 0.0, 0.0],
    t=0.5,
    method="slerp",
)

# Generate a sequence of 5 interpolated vectors
sequence = client.math.interpolate_sequence(
    a=[1.0, 0.0, 0.0, 0.0],
    b=[0.0, 1.0, 0.0, 0.0],
    n=5,
    method="slerp",
)
```

**Use cases**: smooth concept transitions (e.g., blending "formal" and "casual" tone embeddings), generating intermediate training data, animation of embedding trajectories.

---

## Centroid Computation

Computes the geometric center of a set of vectors. The centroid represents the "average direction" of the group, which is useful for summarizing clusters or computing representative vectors.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_ids` | array of ints | [] (all vectors) | IDs of vectors to include. If empty, uses all vectors in the collection |
| `weights` | array of floats | [] (unweighted) | Per-vector weights for a weighted centroid |

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/math/centroid`

```bash
# Unweighted centroid of specific vectors
curl -X POST http://localhost:8080/api/v1/collections/articles/math/centroid \
  -H "Content-Type: application/json" \
  -d '{
    "vector_ids": [1, 2, 3, 4, 5]
  }'

# Weighted centroid (give more influence to certain vectors)
curl -X POST http://localhost:8080/api/v1/collections/articles/math/centroid \
  -H "Content-Type: application/json" \
  -d '{
    "vector_ids": [1, 2, 3],
    "weights": [0.5, 0.3, 0.2]
  }'
```

**Response**:

```json
{
  "centroid": [0.23, 0.45, 0.12, 0.67],
  "compute_time_us": 189
}
```

### Python SDK

```python
# Centroid of all vectors in the collection
centroid = client.math.centroid("articles")

# Centroid of specific vectors
centroid = client.math.centroid("articles", vector_ids=[1, 2, 3, 4, 5])

# Weighted centroid
centroid = client.math.centroid(
    "articles",
    vector_ids=[1, 2, 3],
    weights=[0.5, 0.3, 0.2],
)
```

**Use cases**: finding the "average" of a category, computing cluster centers manually, creating representative embeddings for groups of documents.

---

## Vector Drift Detection

Drift detection measures how the distribution of vectors changes between two time windows. This is critical for monitoring embedding model health: if your production embeddings start drifting from your training distribution, search quality degrades silently.

### How It Works

SwarnDB computes four metrics to characterize drift:

| Metric | What it measures |
|--------|-----------------|
| `centroid_shift` | Distance between the centroids of the two windows |
| `mean_distance_window1` | Average distance from vectors to centroid in window 1 |
| `mean_distance_window2` | Average distance from vectors to centroid in window 2 |
| `spread_change` | Difference in spread (dispersion) between the two windows |

If you provide a `threshold`, the response includes a boolean `has_drifted` flag indicating whether `centroid_shift` exceeds it.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window1_ids` | array of ints | required | Vector IDs for the baseline window |
| `window2_ids` | array of ints | required | Vector IDs for the comparison window |
| `metric` | string | "euclidean" | Distance metric |
| `threshold` | float | null | If set, `has_drifted` is true when centroid_shift exceeds this |

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/math/drift`

```bash
curl -X POST http://localhost:8080/api/v1/collections/embeddings/math/drift \
  -H "Content-Type: application/json" \
  -d '{
    "window1_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "window2_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "metric": "euclidean",
    "threshold": 0.5
  }'
```

**Response**:

```json
{
  "centroid_shift": 0.72,
  "mean_distance_window1": 1.23,
  "mean_distance_window2": 1.89,
  "spread_change": 0.66,
  "has_drifted": true,
  "compute_time_us": 3450
}
```

### Python SDK

```python
report = client.math.detect_drift(
    "embeddings",
    window1_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    window2_ids=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    metric="euclidean",
    threshold=0.5,
)

print(f"Centroid shift: {report.centroid_shift:.3f}")
print(f"Spread change: {report.spread_change:.3f}")
print(f"Drifted: {report.has_drifted}")
```

**Use cases**: monitoring embedding model degradation over time, detecting distribution shift in production data, triggering re-indexing or model retraining when drift exceeds acceptable bounds.

---

## K-Means Clustering

Partitions all vectors in a collection into k clusters using iterative centroid assignment and refinement. This is the standard k-means algorithm, running entirely server-side.

### How It Works

1. Initialize k random centroids.
2. Assign each vector to its nearest centroid.
3. Recompute centroids as the mean of assigned vectors.
4. Repeat until convergence (centroid movement falls below tolerance) or max iterations is reached.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of clusters |
| `max_iterations` | int | 100 | Maximum iteration count |
| `tolerance` | float | 1e-4 | Convergence tolerance (stop when centroid movement is below this) |
| `metric` | string | "euclidean" | Distance metric for assignment |

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/math/cluster`

```bash
curl -X POST http://localhost:8080/api/v1/collections/products/math/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "k": 5,
    "max_iterations": 200,
    "tolerance": 1e-5,
    "metric": "euclidean"
  }'
```

**Response**:

```json
{
  "centroids": [
    [0.12, 0.45, 0.67, 0.23],
    [0.89, 0.11, 0.34, 0.56],
    [0.34, 0.78, 0.12, 0.45],
    [0.56, 0.23, 0.89, 0.11],
    [0.67, 0.56, 0.45, 0.78]
  ],
  "assignments": [
    { "id": 1, "cluster": 0, "distance_to_centroid": 0.12 },
    { "id": 2, "cluster": 2, "distance_to_centroid": 0.34 },
    { "id": 3, "cluster": 0, "distance_to_centroid": 0.18 }
  ],
  "iterations": 47,
  "converged": true,
  "compute_time_us": 89200
}
```

### Python SDK

```python
result = client.math.cluster("products", k=5, max_iterations=200)

print(f"Converged in {result.iterations} iterations: {result.converged}")
print(f"Found {len(result.centroids)} clusters")

for assignment in result.assignments:
    print(f"Vector {assignment.id} -> cluster {assignment.cluster} "
          f"(distance: {assignment.distance_to_centroid:.3f})")
```

**Use cases**: automatic categorization of unlabeled data, topic modeling across document collections, data segmentation for targeted analysis.

---

## PCA (Dimensionality Reduction)

Reduces vector dimensions using Principal Component Analysis. PCA finds the axes of maximum variance in your data and projects vectors onto those axes, preserving as much information as possible in fewer dimensions.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 2 | Number of principal components to keep |
| `vector_ids` | array of ints | [] (all) | Subset of vector IDs. If empty, uses all vectors |

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/math/pca`

```bash
curl -X POST http://localhost:8080/api/v1/collections/embeddings/math/pca \
  -H "Content-Type: application/json" \
  -d '{
    "n_components": 3,
    "vector_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  }'
```

**Response**:

```json
{
  "components": [
    [0.45, 0.32, -0.12, 0.78],
    [0.23, -0.67, 0.56, 0.11],
    [-0.34, 0.12, 0.78, 0.45]
  ],
  "explained_variance": [0.52, 0.28, 0.12],
  "mean": [0.34, 0.21, 0.45, 0.56],
  "projected": [
    [1.23, -0.45, 0.12],
    [0.87, 0.34, -0.67],
    [-0.56, 1.12, 0.23]
  ],
  "compute_time_us": 15600
}
```

### Python SDK

```python
result = client.math.reduce_dimensions(
    "embeddings",
    n_components=3,
    vector_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)

print(f"Explained variance: {result.explained_variance}")
# e.g., [0.52, 0.28, 0.12] means PC1 captures 52% of variance

# Use projected vectors for visualization
for i, projected in enumerate(result.projected):
    print(f"Vector {i}: {projected}")
```

### Understanding the Response

- **components**: The principal component vectors (eigenvectors). Each one defines a direction of maximum variance.
- **explained_variance**: How much variance each component captures. Sum these to see total variance explained.
- **mean**: The mean vector subtracted before PCA. Useful for projecting new vectors.
- **projected**: The input vectors projected onto the principal components. These are the reduced-dimension representations.

**Use cases**: reducing 1536-dimensional embeddings to 2D or 3D for visualization, feature analysis to understand which dimensions carry the most information, compression for downstream processing.

---

## Analogy Completion

Vector analogies exploit the algebraic structure of embedding spaces. The classic example: "king" minus "man" plus "woman" equals "queen". SwarnDB provides this as a built-in operation.

### How It Works

The standard analogy formula is: **result = B - A + C**

This computes "A is to B as C is to ?". The result vector is the answer.

For more general arithmetic, SwarnDB also supports a `terms` array where each term is a vector with a weight. The result is the weighted sum of all terms.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | vector | The "is to" vector |
| `b` | vector | The "as" vector |
| `c` | vector | The query vector |
| `normalize` | bool | Whether to L2-normalize the result (default: false) |
| `terms` | array of {vector, weight} | Alternative: arbitrary weighted vector arithmetic |

When `terms` is provided, it takes precedence over `a`, `b`, `c`.

### REST API

**Endpoint**: `POST /api/v1/math/analogy`

**Classic analogy** (A is to B as C is to ?):

```bash
curl -X POST http://localhost:8080/api/v1/math/analogy \
  -H "Content-Type: application/json" \
  -d '{
    "a": [0.2, 0.8, 0.1, 0.5],
    "b": [0.7, 0.3, 0.6, 0.4],
    "c": [0.1, 0.9, 0.2, 0.6],
    "normalize": true
  }'
```

**Weighted arithmetic** (custom vector combination):

```bash
curl -X POST http://localhost:8080/api/v1/math/analogy \
  -H "Content-Type: application/json" \
  -d '{
    "terms": [
      { "vector": [0.5, 0.5, 0.0, 0.0], "weight": 0.7 },
      { "vector": [0.0, 0.0, 0.5, 0.5], "weight": 0.3 }
    ],
    "normalize": true
  }'
```

**Response**:

```json
{
  "result": [0.58, 0.34, 0.52, 0.42],
  "compute_time_us": 12
}
```

### Python SDK

```python
# Classic analogy: A is to B as C is to ?
result = client.math.analogy(
    a=[0.2, 0.8, 0.1, 0.5],
    b=[0.7, 0.3, 0.6, 0.4],
    c=[0.1, 0.9, 0.2, 0.6],
    normalize=True,
)

# Weighted sum of vectors
result = client.math.weighted_sum(
    vectors=[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]],
    weights=[0.7, 0.3],
    normalize=True,
)
```

**Use cases**: word analogy tasks, transferring relationships between domains (e.g., "product A is the premium version of product B; what is the premium version of product C?"), concept arithmetic for creative exploration.

---

## Maximal Marginal Relevance (MMR)

MMR is a diversity-aware selection algorithm. It iteratively picks vectors that are both relevant to your query and different from the vectors already selected. This eliminates the redundancy problem where top-k results all say essentially the same thing.

### How It Works

At each step, MMR scores every remaining candidate using:

```text
MMR(v) = lambda * relevance(v, query) - (1 - lambda) * max_similarity(v, selected)
```

The vector with the highest MMR score is added to the selected set. This continues until k vectors are selected.

### The Lambda Parameter

| Lambda | Behavior |
|--------|----------|
| 1.0 | Pure relevance (identical to standard top-k search) |
| 0.7 | Slight diversity bias |
| 0.5 | Equal balance between relevance and diversity |
| 0.3 | Strong diversity bias |
| 0.0 | Pure diversity (maximum spread, ignoring relevance) |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | vector | required | The query vector |
| `k` | int | required | Number of vectors to select |
| `lambda` | float | required | Relevance vs. diversity trade-off (0.0 to 1.0) |
| `candidate_ids` | array of ints | [] (all) | Subset of candidate vector IDs. If empty, considers all vectors |

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/math/diversity`

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/math/diversity \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.3, 0.7, 0.1, 0.5],
    "k": 5,
    "lambda": 0.6,
    "candidate_ids": []
  }'
```

**Response**:

```json
{
  "results": [
    { "id": 42, "relevance_score": 0.95, "mmr_score": 0.95 },
    { "id": 17, "relevance_score": 0.91, "mmr_score": 0.72 },
    { "id": 205, "relevance_score": 0.88, "mmr_score": 0.65 },
    { "id": 83, "relevance_score": 0.85, "mmr_score": 0.58 },
    { "id": 112, "relevance_score": 0.79, "mmr_score": 0.51 }
  ],
  "compute_time_us": 8920
}
```

### Python SDK

```python
results = client.math.diversity_sample(
    "articles",
    query=[0.3, 0.7, 0.1, 0.5],
    k=5,
    lambda_=0.6,
)

for r in results:
    print(f"Vector {r.id}: relevance={r.relevance_score:.3f}, "
          f"mmr={r.mmr_score:.3f}")
```

Note: The Python SDK uses `lambda_` (with underscore) because `lambda` is a reserved keyword in Python.

**Use cases**: diverse search results for user-facing applications, avoiding redundancy in recommendation lists, selecting a representative subset from a large candidate pool.

---

## Summary Table

| Operation | What It Does | Endpoint |
|-----------|-------------|----------|
| Ghost Detection | Finds isolated vectors far from clusters | `POST /api/v1/collections/{col}/math/ghosts` |
| Cone Search | Finds vectors within an angular cone | `POST /api/v1/collections/{col}/math/cone` |
| SLERP/LERP | Interpolates between two vectors | `POST /api/v1/math/interpolate` |
| Centroid | Computes (weighted) geometric center | `POST /api/v1/collections/{col}/math/centroid` |
| Drift Detection | Measures distribution shift between windows | `POST /api/v1/collections/{col}/math/drift` |
| K-Means | Partitions vectors into k clusters | `POST /api/v1/collections/{col}/math/cluster` |
| PCA | Reduces dimensions via principal components | `POST /api/v1/collections/{col}/math/pca` |
| Analogy | Vector arithmetic (A:B :: C:?) | `POST /api/v1/math/analogy` |
| MMR Diversity | Diversity-aware sampling | `POST /api/v1/collections/{col}/math/diversity` |

> **Note**: Interpolation and Analogy are collection-independent operations. They work on raw vectors provided in the request, not on stored vectors. All other operations require a collection name and operate on stored vectors.
