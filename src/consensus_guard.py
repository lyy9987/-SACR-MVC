"""Label-free consensus guard for a single unified MVC readout.

The guard does not choose between different clustering algorithms.  It only
controls the dataset-level prior inside the same reliability-weighted
consensus equation.  A view receives a sparse prior only when it is both
intrinsically stable/separable and clearly stronger than the runner-up;
otherwise the uniform consensus is retained to protect complementary views.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def fixed_subsample(sample_count, maximum=5000, seed=0):
    sample_count = int(sample_count)
    maximum = int(maximum)
    if sample_count <= maximum:
        return np.arange(sample_count, dtype=np.int64)
    generator = np.random.RandomState(int(seed))
    return np.sort(
        generator.choice(sample_count, size=maximum, replace=False)
    ).astype(np.int64)


def pairwise_partition_stability(partitions):
    scores = []
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            scores.append(
                adjusted_rand_score(partitions[left], partitions[right])
            )
    return float(np.mean(scores)) if scores else 1.0


def clusterability(representation, cluster_count, indices, runs=3):
    values = np.asarray(representation, dtype=np.float64)[indices]
    cluster_count = int(cluster_count)
    partitions = [
        KMeans(
            n_clusters=cluster_count,
            random_state=seed,
            n_init=1,
            max_iter=500,
        ).fit_predict(values)
        for seed in range(int(runs))
    ]
    stable = KMeans(
        n_clusters=cluster_count,
        random_state=0,
        n_init=10,
        max_iter=900,
    ).fit_predict(values)
    stability = pairwise_partition_stability(partitions)
    try:
        silhouette = float(silhouette_score(values, stable, metric="cosine"))
    except ValueError:
        silhouette = -1.0
    quality = max(0.0, stability) * max(0.0, silhouette)
    return {
        "stability_ari": float(stability),
        "cosine_silhouette": float(silhouette),
        "quality": float(quality),
    }


def consensus_guarded_gate(
    representations,
    cluster_count,
    view_count,
    *,
    max_samples=5000,
    stability_runs=3,
    dominance_margin=0.005,
    dominance_scale=0.005,
    minimum_quality=0.90,
    quality_scale=0.02,
):
    """Return one continuous gate and its label-free audit diagnostics."""
    view_count = int(view_count)
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    common = np.asarray(representations["learned_common"])
    indices = fixed_subsample(common.shape[0], max_samples, seed=0)
    diagnostics = {
        "version": "consensus_guard_v1",
        "consensus": clusterability(
            common, cluster_count, indices, stability_runs
        ),
        "views": [],
        "subsample_count": int(indices.size),
    }
    for view_index in range(view_count):
        diagnostics["views"].append(
            clusterability(
                representations["learned_view%d" % (view_index + 1)],
                cluster_count,
                indices,
                stability_runs,
            )
        )
    qualities = np.asarray(
        [item["quality"] for item in diagnostics["views"]],
        dtype=np.float64,
    )
    order = np.argsort(-qualities, kind="stable")
    best_view = int(order[0])
    runner_up = float(qualities[order[1]]) if view_count > 1 else 0.0
    raw_margin = float(qualities[best_view] - runner_up)
    dominance_strength = float(
        np.clip(
            (raw_margin - float(dominance_margin))
            / max(float(dominance_scale), 1e-8),
            0.0,
            1.0,
        )
    )
    quality_strength = float(
        np.clip(
            (float(qualities[best_view]) - float(minimum_quality))
            / max(float(quality_scale), 1e-8),
            0.0,
            1.0,
        )
    )
    strength = float(dominance_strength * quality_strength)
    gate = np.full(view_count, (1.0 - strength) / float(view_count))
    gate[best_view] += strength
    diagnostics.update(
        {
            "best_view": best_view + 1,
            "runner_up_view": int(order[1]) + 1 if view_count > 1 else 0,
            "best_runner_up_margin": raw_margin,
            "dominance_strength": dominance_strength,
            "quality_strength": quality_strength,
            "gate_strength": strength,
            "gate": gate.tolist(),
            "parameters": {
                "dominance_margin": float(dominance_margin),
                "dominance_scale": float(dominance_scale),
                "minimum_quality": float(minimum_quality),
                "quality_scale": float(quality_scale),
                "max_samples": int(max_samples),
                "stability_runs": int(stability_runs),
            },
        }
    )
    return gate, diagnostics
