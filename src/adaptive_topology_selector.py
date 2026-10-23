"""Label-free selector for exact versus scalable topology relations.

The selector intentionally does not inspect a dataset name or ground-truth
label.  A very small exact graph is always used because it is both cheap and
fully observed.  At larger (but still safe) scales, exact relations are used
only when a raw multi-view co-association partition agrees with the learned
view partitions by a statistically meaningful margin over the learned common
partition.  Otherwise the scalable semantic memory is retained.

This module contains only the deterministic decision rule.  Expensive
diagnostic construction remains separate so it can be cached and audited.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from topology_policy import estimated_exact_graph_megabytes


SELECTOR_VERSION = "resource_structure_resolution_v3"


def _deterministic_subset(sample_count, maximum, seed=0):
    count = int(sample_count)
    maximum = max(1, int(maximum))
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    generator = np.random.RandomState(int(seed))
    return np.sort(generator.choice(count, size=maximum, replace=False)).astype(
        np.int64
    )


def _partition(features, cluster_count, seed=0, n_init=1):
    features = (
        features.astype(np.float32)
        if sparse.issparse(features)
        else np.asarray(features, dtype=np.float32)
    )
    return KMeans(
        n_clusters=int(cluster_count),
        random_state=int(seed),
        n_init=max(1, int(n_init)),
        max_iter=500,
    ).fit_predict(features)


def compute_backend_structure_diagnostics(
    raw_views,
    learned_common,
    learned_views,
    cluster_count,
    *,
    maximum_samples=5000,
    seed=0,
    n_init=1,
):
    """Measure raw/learned structure agreement without ground-truth labels.

    One deterministic KMeans partition is built per raw and learned view on a
    fixed subset.  The raw co-association partition is obtained by clustering
    a sparse one-hot membership embedding, avoiding an ``N x N`` matrix and
    the expensive multi-run diagnostic used in the research audit script.
    """
    learned_common = np.asarray(learned_common, dtype=np.float32)
    learned_views = [np.asarray(view, dtype=np.float32) for view in learned_views]
    raw_views = [np.asarray(view, dtype=np.float32) for view in raw_views]
    if not raw_views or len(raw_views) != len(learned_views):
        raise ValueError("raw_views and learned_views must have equal nonzero length")
    sample_count = int(learned_common.shape[0])
    if any(view.shape[0] != sample_count for view in raw_views + learned_views):
        raise ValueError("all representations must share the sample dimension")
    sample_indices = _deterministic_subset(
        sample_count, maximum_samples, seed=seed
    )
    if sample_indices.size < int(cluster_count):
        raise ValueError("diagnostic subset is smaller than the cluster count")

    learned_partitions = [
        _partition(
            normalize(view[sample_indices]),
            cluster_count,
            seed=seed,
            n_init=n_init,
        )
        for view in learned_views
    ]
    common_partition = _partition(
        normalize(learned_common[sample_indices]),
        cluster_count,
        seed=seed,
        n_init=n_init,
    )

    raw_partitions = []
    for view in raw_views:
        raw_sample = view[sample_indices]
        # Row normalization makes Euclidean KMeans equivalent to cosine
        # structure while avoiding a dataset-specific feature scale.
        raw_partitions.append(
            _partition(
                normalize(raw_sample),
                cluster_count,
                seed=seed,
                n_init=n_init,
            )
        )

    rows = np.tile(
        np.arange(sample_indices.size, dtype=np.int64), len(raw_partitions)
    )
    columns = np.concatenate(
        [
            np.asarray(partition, dtype=np.int64)
            + view_index * int(cluster_count)
            for view_index, partition in enumerate(raw_partitions)
        ]
    )
    values = np.full(rows.size, 1.0 / math.sqrt(len(raw_partitions)), np.float32)
    membership = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(sample_indices.size, len(raw_partitions) * int(cluster_count)),
        dtype=np.float32,
    ).tocsr()
    raw_coassociation_partition = _partition(
        membership,
        cluster_count,
        seed=seed,
        n_init=n_init,
    )

    learned_score = float(
        np.mean(
            [
                adjusted_rand_score(common_partition, partition)
                for partition in learned_partitions
            ]
        )
    )
    raw_to_learned = float(
        np.mean(
            [
                adjusted_rand_score(raw_coassociation_partition, partition)
                for partition in learned_partitions
            ]
        )
    )
    raw_view_to_learned = [
        float(
            np.mean(
                [
                    adjusted_rand_score(raw_partition, learned_partition)
                    for learned_partition in learned_partitions
                ]
            )
        )
        for raw_partition in raw_partitions
    ]
    raw_internal = float(
        np.mean(
            [
                adjusted_rand_score(raw_coassociation_partition, partition)
                for partition in raw_partitions
            ]
        )
    )
    return {
        "version": SELECTOR_VERSION,
        "sample_count": int(sample_indices.size),
        "learned_view_agreement": learned_score,
        "raw_coassociation_learned_view_agreement": raw_to_learned,
        "raw_view_learned_view_agreements": raw_view_to_learned,
        "best_raw_view_learned_view_agreement": float(
            max(raw_view_to_learned)
        ),
        "raw_coassociation_raw_view_agreement": raw_internal,
    }


def structural_margin(sample_count: int, floor: float = 0.01) -> float:
    """Return a conservative finite-sample margin for ARI agreement scores."""
    count = int(sample_count)
    if count < 2:
        raise ValueError("sample_count must be at least two")
    if float(floor) < 0.0:
        raise ValueError("floor must be non-negative")
    return float(max(float(floor), 1.0 / math.sqrt(float(count))))


def select_topology_backend(
    sample_count: int,
    view_count: int,
    learned_view_agreement: float,
    raw_coassociation_learned_view_agreement: float,
    *,
    cluster_count: Optional[int] = None,
    raw_coassociation_raw_view_agreement: float = 0.0,
    best_raw_view_learned_view_agreement: Optional[float] = None,
    compact_exact_budget_mb: float = 64.0,
    maximum_safe_exact_mb: float = 4096.0,
    agreement_floor: float = 0.01,
) -> dict:
    """Choose one relation provider without dataset names or labels.

    Returns an auditable dictionary including the reason and all quantities
    used in the decision.  ``maximum_safe_exact_mb`` is a hard
    dense-equivalent search guard.  Exact KNN is still searched in chunks and
    stored in ``O(Nk)`` sparse form; the guard limits quadratic search work
    rather than claiming that a dense graph is retained during training.
    """
    compact_budget = float(compact_exact_budget_mb)
    safe_budget = float(maximum_safe_exact_mb)
    if compact_budget <= 0.0 or safe_budget < compact_budget:
        raise ValueError("invalid exact-graph resource budgets")

    learned = float(learned_view_agreement)
    raw = float(raw_coassociation_learned_view_agreement)
    best_raw = (
        raw
        if best_raw_view_learned_view_agreement is None
        else float(best_raw_view_learned_view_agreement)
    )
    raw_internal = float(raw_coassociation_raw_view_agreement)
    if (
        not math.isfinite(learned)
        or not math.isfinite(raw)
        or not math.isfinite(best_raw)
        or not math.isfinite(raw_internal)
    ):
        raise ValueError("agreement scores must be finite")
    estimate = estimated_exact_graph_megabytes(sample_count, view_count)
    clusters = None if cluster_count is None else int(cluster_count)
    if clusters is not None and clusters <= 1:
        raise ValueError("cluster_count must be greater than one")
    # A task is fine-grained when the number of semantic regions is at least
    # the expected samples per region, i.e. K >= N/K.  This criterion uses
    # only the public cluster count and sample count.  It captures the regime
    # where a complete local relation is statistically valuable without a
    # dataset-name exception or an additional tuned threshold.
    fine_grained = bool(
        clusters is not None and clusters * clusters >= int(sample_count)
    )
    margin = structural_margin(sample_count, floor=agreement_floor)
    raw_candidate = max(raw, best_raw)
    structural_gain = raw_candidate - learned

    if estimate <= compact_budget:
        route = "exact_graph"
        reason = "compact_exact_graph"
    elif estimate > safe_budget:
        route = "scalable_global_memory"
        reason = "exact_graph_exceeds_safe_budget"
    elif fine_grained:
        route = "exact_graph"
        reason = "fine_grained_clusters_need_complete_local_relations"
    elif structural_gain > margin:
        route = "exact_graph"
        reason = "raw_structure_significantly_improves_learned_view_agreement"
    else:
        route = "scalable_global_memory"
        reason = "learned_semantic_structure_is_not_weaker"

    return {
        "selector_version": SELECTOR_VERSION,
        "route": route,
        "scalable": route == "scalable_global_memory",
        "reason": reason,
        "estimated_exact_graph_mb": float(estimate),
        "compact_exact_budget_mb": compact_budget,
        "maximum_safe_exact_mb": safe_budget,
        "cluster_count": clusters,
        "fine_grained_resolution": fine_grained,
        "learned_view_agreement": learned,
        "raw_coassociation_learned_view_agreement": raw,
        "raw_coassociation_raw_view_agreement": raw_internal,
        "best_raw_view_learned_view_agreement": best_raw,
        "raw_candidate_learned_view_agreement": raw_candidate,
        "structural_gain": float(structural_gain),
        "required_structural_margin": float(margin),
    }
