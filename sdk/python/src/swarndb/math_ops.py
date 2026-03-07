"""SwarnDB vector math operations.

This module provides server-side vector math: ghost detection, cone search,
centroid computation, SLERP/LERP interpolation, drift detection, k-means
clustering, PCA dimensionality reduction, analogy solving, and MMR diversity
sampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ._helpers import _to_proto_vector
from ._proto import common_pb2, vector_math_pb2
from .types import (
    ClusterAssignment,
    ClusterResult,
    ConeSearchResult,
    DiversityResult,
    DriftReport,
    GhostVector,
    PCAResult,
)

if TYPE_CHECKING:
    from .client import SwarnDBClient


class MathAPI:
    """Pythonic wrapper around the VectorMathService gRPC API."""

    def __init__(self, client: SwarnDBClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # DetectGhosts
    # ------------------------------------------------------------------

    def detect_ghosts(
        self,
        collection: str,
        threshold: float,
        *,
        centroids: Optional[List[List[float]]] = None,
        auto_k: int = 8,
        metric: str = "euclidean",
    ) -> List[GhostVector]:
        """Detect isolated 'ghost' vectors far from any cluster centroid.

        Args:
            collection: Collection name.
            threshold: Distance threshold above which a vector is a ghost.
            centroids: Optional explicit centroid vectors. If omitted the
                server auto-computes centroids using ``auto_k``.
            auto_k: Number of centroids to auto-compute when ``centroids``
                is not provided.
            metric: Distance metric (e.g. ``"euclidean"``).

        Returns:
            List of GhostVector(id, isolation_score).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        proto_centroids = (
            [_to_proto_vector(c) for c in centroids] if centroids else []
        )
        request = vector_math_pb2.DetectGhostsRequest(
            collection=collection,
            threshold=threshold,
            centroids=proto_centroids,
            auto_k=auto_k,
            metric=metric,
        )
        response = self._client._call(
            self._client._vector_math_stub.DetectGhosts, request
        )
        return [
            GhostVector(
                id=g.id,
                isolation_score=g.isolation_score,
            )
            for g in response.ghosts
        ]

    # ------------------------------------------------------------------
    # ConeSearch
    # ------------------------------------------------------------------

    def cone_search(
        self,
        collection: str,
        direction: List[float],
        aperture_radians: float,
    ) -> List[ConeSearchResult]:
        """Find vectors within an angular cone around a direction vector.

        Args:
            collection: Collection name.
            direction: Unit direction vector defining the cone axis.
            aperture_radians: Half-angle of the cone in radians.

        Returns:
            List of ConeSearchResult(id, cosine_similarity, angle_radians).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ConeSearchRequest(
            collection=collection,
            direction=_to_proto_vector(direction),
            aperture_radians=aperture_radians,
        )
        response = self._client._call(
            self._client._vector_math_stub.ConeSearch, request
        )
        return [
            ConeSearchResult(
                id=r.id,
                cosine_similarity=r.cosine_similarity,
                angle_radians=r.angle_radians,
            )
            for r in response.results
        ]

    # ------------------------------------------------------------------
    # ComputeCentroid
    # ------------------------------------------------------------------

    def centroid(
        self,
        collection: str,
        *,
        vector_ids: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute the (optionally weighted) centroid of vectors.

        Args:
            collection: Collection name.
            vector_ids: IDs of vectors to include. If omitted, uses all
                vectors in the collection.
            weights: Optional per-vector weights for a weighted centroid.

        Returns:
            Centroid as a list of floats.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ComputeCentroidRequest(
            collection=collection,
            vector_ids=vector_ids or [],
            weights=weights or [],
        )
        response = self._client._call(
            self._client._vector_math_stub.ComputeCentroid, request
        )
        return list(response.centroid.values)

    # ------------------------------------------------------------------
    # Interpolate (single)
    # ------------------------------------------------------------------

    def interpolate(
        self,
        a: List[float],
        b: List[float],
        t: float = 0.5,
        *,
        method: str = "lerp",
    ) -> List[float]:
        """Interpolate between two vectors at parameter t.

        Args:
            a: Start vector.
            b: End vector.
            t: Interpolation parameter in [0, 1].
            method: ``"lerp"`` for linear or ``"slerp"`` for spherical.

        Returns:
            Interpolated vector as a list of floats.

        Raises:
            MathError: If the operation fails.
        """
        request = vector_math_pb2.InterpolateRequest(
            a=_to_proto_vector(a),
            b=_to_proto_vector(b),
            t=t,
            method=method,
            sequence_count=0,
        )
        response = self._client._call(
            self._client._vector_math_stub.Interpolate, request
        )
        return list(response.results[0].values)

    # ------------------------------------------------------------------
    # Interpolate (sequence)
    # ------------------------------------------------------------------

    def interpolate_sequence(
        self,
        a: List[float],
        b: List[float],
        n: int,
        *,
        method: str = "lerp",
    ) -> List[List[float]]:
        """Generate a sequence of n interpolated vectors between a and b.

        Args:
            a: Start vector.
            b: End vector.
            n: Number of interpolation steps.
            method: ``"lerp"`` for linear or ``"slerp"`` for spherical.

        Returns:
            List of interpolated vectors.

        Raises:
            MathError: If the operation fails.
        """
        request = vector_math_pb2.InterpolateRequest(
            a=_to_proto_vector(a),
            b=_to_proto_vector(b),
            t=0.0,
            method=method,
            sequence_count=n,
        )
        response = self._client._call(
            self._client._vector_math_stub.Interpolate, request
        )
        return [list(v.values) for v in response.results]

    # ------------------------------------------------------------------
    # DetectDrift
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        collection: str,
        window1_ids: List[int],
        window2_ids: List[int],
        *,
        metric: str = "euclidean",
        threshold: float = 0.0,
    ) -> DriftReport:
        """Detect distribution drift between two temporal windows of vectors.

        Args:
            collection: Collection name.
            window1_ids: Vector IDs for the first (baseline) window.
            window2_ids: Vector IDs for the second (comparison) window.
            metric: Distance metric (e.g. ``"euclidean"``).
            threshold: Drift threshold; if centroid shift exceeds this
                the report marks ``has_drifted`` as True.

        Returns:
            DriftReport with centroid shift, spread change, and drift flag.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.DetectDriftRequest(
            collection=collection,
            window1_ids=window1_ids,
            window2_ids=window2_ids,
            metric=metric,
            threshold=threshold,
        )
        response = self._client._call(
            self._client._vector_math_stub.DetectDrift, request
        )
        return DriftReport(
            centroid_shift=response.centroid_shift,
            mean_distance_window1=response.mean_distance_window1,
            mean_distance_window2=response.mean_distance_window2,
            spread_change=response.spread_change,
            has_drifted=response.has_drifted,
        )

    # ------------------------------------------------------------------
    # Cluster
    # ------------------------------------------------------------------

    def cluster(
        self,
        collection: str,
        k: int,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        metric: str = "euclidean",
    ) -> ClusterResult:
        """Run k-means clustering on vectors in a collection.

        Args:
            collection: Collection name.
            k: Number of clusters.
            max_iterations: Maximum iteration count.
            tolerance: Convergence tolerance.
            metric: Distance metric (e.g. ``"euclidean"``).

        Returns:
            ClusterResult with centroids, assignments, iteration count,
            and convergence flag.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ClusterRequest(
            collection=collection,
            k=k,
            max_iterations=max_iterations,
            tolerance=tolerance,
            metric=metric,
        )
        response = self._client._call(
            self._client._vector_math_stub.Cluster, request
        )
        return ClusterResult(
            centroids=[list(c.values) for c in response.centroids],
            assignments=[
                ClusterAssignment(
                    id=a.id,
                    cluster=a.cluster,
                    distance_to_centroid=a.distance_to_centroid,
                )
                for a in response.assignments
            ],
            iterations=response.iterations,
            converged=response.converged,
        )

    # ------------------------------------------------------------------
    # ReduceDimensions (PCA)
    # ------------------------------------------------------------------

    def reduce_dimensions(
        self,
        collection: str,
        n_components: int,
        *,
        vector_ids: Optional[List[int]] = None,
    ) -> PCAResult:
        """Perform PCA dimensionality reduction on collection vectors.

        Args:
            collection: Collection name.
            n_components: Number of principal components to keep.
            vector_ids: Optional subset of vector IDs. If omitted, uses
                all vectors in the collection.

        Returns:
            PCAResult with components, explained variance, mean, and
            projected vectors.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ReduceDimensionsRequest(
            collection=collection,
            n_components=n_components,
            vector_ids=vector_ids or [],
        )
        response = self._client._call(
            self._client._vector_math_stub.ReduceDimensions, request
        )
        return PCAResult(
            components=[list(c.values) for c in response.components],
            explained_variance=list(response.explained_variance),
            mean=list(response.mean.values),
            projected=[list(p.values) for p in response.projected],
        )

    # ------------------------------------------------------------------
    # ComputeAnalogy
    # ------------------------------------------------------------------

    def analogy(
        self,
        a: List[float],
        b: List[float],
        c: List[float],
        *,
        normalize: bool = True,
    ) -> List[float]:
        """Compute vector analogy: a - b + c.

        Args:
            a: First vector (the "is to" side).
            b: Second vector (the "as" side).
            c: Third vector (the query side).
            normalize: Whether to L2-normalize the result.

        Returns:
            Result vector as a list of floats.

        Raises:
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ComputeAnalogyRequest(
            a=_to_proto_vector(a),
            b=_to_proto_vector(b),
            c=_to_proto_vector(c),
            normalize=normalize,
        )
        response = self._client._call(
            self._client._vector_math_stub.ComputeAnalogy, request
        )
        return list(response.result.values)

    # ------------------------------------------------------------------
    # Weighted sum (via ComputeAnalogy with terms)
    # ------------------------------------------------------------------

    def weighted_sum(
        self,
        vectors: List[List[float]],
        weights: List[float],
        *,
        normalize: bool = True,
    ) -> List[float]:
        """Compute a weighted sum of vectors using the analogy endpoint.

        Uses the ``terms`` field of ComputeAnalogyRequest to send
        arbitrary (vector, weight) pairs. The ``a``, ``b``, ``c`` fields
        are set to zero-vectors of matching dimension and ignored by the
        server when terms are present.

        Args:
            vectors: List of vectors to combine.
            weights: Corresponding weight for each vector.
            normalize: Whether to L2-normalize the result.

        Returns:
            Result vector as a list of floats.

        Raises:
            ValueError: If vectors and weights have different lengths.
            MathError: If the operation fails.
        """
        if len(vectors) != len(weights):
            raise ValueError(
                f"vectors and weights must have the same length, "
                f"got {len(vectors)} and {len(weights)}"
            )
        dim = len(vectors[0]) if vectors else 0
        zero = [0.0] * dim
        terms = [
            vector_math_pb2.ArithmeticTerm(
                vector=_to_proto_vector(v),
                weight=w,
            )
            for v, w in zip(vectors, weights)
        ]
        request = vector_math_pb2.ComputeAnalogyRequest(
            a=_to_proto_vector(zero),
            b=_to_proto_vector(zero),
            c=_to_proto_vector(zero),
            normalize=normalize,
            terms=terms,
        )
        response = self._client._call(
            self._client._vector_math_stub.ComputeAnalogy, request
        )
        return list(response.result.values)

    # ------------------------------------------------------------------
    # DiversitySample (MMR)
    # ------------------------------------------------------------------

    def diversity_sample(
        self,
        collection: str,
        query: List[float],
        k: int,
        *,
        lambda_: float = 0.5,
        candidate_ids: Optional[List[int]] = None,
    ) -> List[DiversityResult]:
        """Maximal Marginal Relevance diversity sampling.

        Selects k vectors that balance relevance to the query with
        diversity among the selected set.

        Args:
            collection: Collection name.
            query: Query vector.
            k: Number of vectors to select.
            lambda_: Trade-off parameter in [0, 1]. Higher values favour
                relevance; lower values favour diversity.
            candidate_ids: Optional subset of candidate vector IDs to
                consider. If omitted, considers all vectors.

        Returns:
            List of DiversityResult(id, relevance_score, mmr_score).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        # proto field is named "lambda" which is a Python keyword;
        # protobuf Python codegen maps it via keyword argument syntax.
        request = vector_math_pb2.DiversitySampleRequest(
            collection=collection,
            query=_to_proto_vector(query),
            k=k,
            candidate_ids=candidate_ids or [],
            **{"lambda": lambda_},
        )
        response = self._client._call(
            self._client._vector_math_stub.DiversitySample, request
        )
        return [
            DiversityResult(
                id=r.id,
                relevance_score=r.relevance_score,
                mmr_score=r.mmr_score,
            )
            for r in response.results
        ]
