# -*- coding: utf-8 -*-
"""Label-free readout utilities for the adaptive-small-g release.

The active publication route in ``train_final.py`` uses the learned
reliability-weighted consensus, the consensus guard, a symmetric residual only
when the common representation is insufficient, and one KMeans head.  The
partition helpers below are retained only for backward-compatible diagnostics;
they are not called by this release's active route.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader


def l2_normalize(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + eps)


def zscore_normalize(values, eps=1e-12):
    values = np.array(values, dtype=np.float64, copy=True)
    values -= values.mean(axis=0, keepdims=True)
    variance = np.einsum("ij,ij->j", values, values, optimize=True)
    variance /= max(1, values.shape[0])
    values /= np.sqrt(variance.reshape(1, -1) + eps)
    return values


def reduce_and_normalize(values, max_dim=256, seed=0):
    """Standardize, optionally reduce, and row-normalize a representation."""
    values = zscore_normalize(values)
    if values.shape[1] > int(max_dim):
        values = PCA(
            n_components=int(max_dim), random_state=int(seed)
        ).fit_transform(values)
    return l2_normalize(values)


def _collect_learned_blocks(
    model,
    dataset,
    batch_size=256,
    seed=0,
    inference_reliability_power=2.0,
):
    """Collect common and view representations without accessing labels."""
    was_training = bool(model.training)
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        generator=generator,
    )
    common_batches = []
    view_batches = None
    try:
        with torch.no_grad():
            for xs, _, sample_indices in loader:
                _, view_features = model(xs)
                reliability = None
                if hasattr(model, "sample_reliability"):
                    reliability = model.sample_reliability[sample_indices]
                common = model.fusion(
                    view_features,
                    reliability,
                    reliability_power=float(inference_reliability_power),
                )
                common_batches.append(common.detach().cpu())
                if view_batches is None:
                    view_batches = [[] for _ in view_features]
                for view_index, features in enumerate(view_features):
                    view_batches[view_index].append(features.detach().cpu())
    finally:
        model.train(was_training)

    if not common_batches:
        raise ValueError("cannot extract a representation from an empty dataset")
    common = torch.cat(common_batches, dim=0).numpy().astype(np.float64)
    views = [
        torch.cat(batches, dim=0).numpy().astype(np.float64)
        for batches in view_batches
    ]
    return common, views


def extract_shared_private_clustering_representation(
    model,
    dataset,
    batch_size=256,
    max_dim=None,
    seed=0,
):
    """DMCAR-style clustering code: mean shared plus every private block."""
    if model.shared_private_scale <= 0.0 or (
        model.shared_residual is None
        and getattr(model, "context_moe", None) is None
    ):
        raise ValueError("shared-private branches must be enabled")
    was_training = bool(model.training)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    shared_batches = []
    private_batches = None
    try:
        with torch.no_grad():
            for xs, _, _ in loader:
                _, shared = model(xs)
                private = model.last_private_features
                shared_batches.append(
                    torch.stack(shared, dim=0).mean(dim=0).cpu()
                )
                if private_batches is None:
                    private_batches = [[] for _ in private]
                for view, values in enumerate(private):
                    private_batches[view].append(values.detach().cpu())
    finally:
        model.train(was_training)
    mean_shared = torch.cat(shared_batches, dim=0).numpy()
    private_views = [
        torch.cat(items, dim=0).numpy() for items in private_batches
    ]
    del max_dim, seed
    # Follow the declared DMCAR clustering equation literally: concatenate
    # the mean shared code with every private code.  Do not z-score tiny
    # private residuals or compress the concatenation, because either step
    # would change their intended relative mass.  One final row normalisation
    # matches this project's single cosine-KMeans protocol.
    representation = np.concatenate(
        [mean_shared] + private_views, axis=1
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(representation, axis=1, keepdims=True)
    return representation / np.maximum(norms, np.finfo(np.float32).eps)


def extract_learned_consensus(
    model,
    dataset,
    batch_size=256,
    max_dim=256,
    seed=0,
    inference_reliability_power=2.0,
):
    """Extract the sole final representation without accessing class labels."""
    common, _ = _collect_learned_blocks(
        model,
        dataset,
        batch_size=batch_size,
        seed=seed,
        inference_reliability_power=inference_reliability_power,
    )
    return reduce_and_normalize(common, max_dim=max_dim, seed=seed)


def extract_learned_representations(
    model,
    dataset,
    batch_size=256,
    max_raw_dim=256,
    seed=0,
    inference_reliability_power=2.0,
    anchor_view_index=None,
    anchor_expert_mass=None,
):
    """Compatibility wrapper used by the label-free pretraining selector.

    The legacy anchor arguments are accepted so old cache code can call this
    function, but the unified protocol never injects an expert or raw branch
    into the learned consensus.
    """
    del anchor_view_index, anchor_expert_mass
    common, learned_views = _collect_learned_blocks(
        model,
        dataset,
        batch_size=batch_size,
        seed=seed,
        inference_reliability_power=inference_reliability_power,
    )
    representations = {
        "learned_common": reduce_and_normalize(
            common, max_dim=max_raw_dim, seed=seed
        )
    }
    for view_index, features in enumerate(learned_views, start=1):
        representations["learned_view%d" % view_index] = reduce_and_normalize(
            features, max_dim=max_raw_dim, seed=seed
        )
    return representations


def guarded_complementary_consensus(
    representations,
    *,
    gate_strength,
    consensus_quality,
    view_count,
    residual_weight=0.10,
    activation_quality=0.80,
    activation_scale=0.05,
    max_dim=256,
    seed=0,
):
    """Compose one consensus with a label-free complementary residual.

    The residual is suppressed when either a dominant expert has already been
    selected or the consensus is intrinsically stable and separable.  All view
    blocks enter symmetrically when the residual is active.
    """
    view_count = int(view_count)
    sufficiency_deficit = float(
        np.clip(
            (float(activation_quality) - float(consensus_quality))
            / max(float(activation_scale), 1e-8),
            0.0,
            1.0,
        )
    )
    effective = float(
        np.clip(float(residual_weight), 0.0, 1.0)
        * (1.0 - float(np.clip(gate_strength, 0.0, 1.0)))
        * sufficiency_deficit
    )
    common = np.asarray(representations["learned_common"], dtype=np.float64)
    if effective <= 1e-12:
        return common, effective, sufficiency_deficit
    blocks = [np.sqrt(1.0 - effective) * common]
    blocks.extend(
        np.sqrt(effective / float(view_count))
        * np.asarray(
            representations["learned_view%d" % (index + 1)],
            dtype=np.float64,
        )
        for index in range(view_count)
    )
    fused = reduce_and_normalize(
        np.concatenate(blocks, axis=1), max_dim=max_dim, seed=seed
    )
    return fused, effective, sufficiency_deficit


def weighted_partition_embedding(partitions, weights, cluster_count):
    """Create a sparse, label-permutation-invariant partition embedding."""
    if not partitions or len(partitions) != len(weights):
        raise ValueError("partitions and weights must have equal nonzero length")
    cluster_count = int(cluster_count)
    if cluster_count <= 1:
        raise ValueError("cluster_count must be greater than one")
    sample_count = int(np.asarray(partitions[0]).size)
    blocks = []
    for partition, weight in zip(partitions, weights):
        labels = np.asarray(partition, dtype=np.int64).reshape(-1)
        weight = float(weight)
        if labels.size != sample_count:
            raise ValueError("all partitions must have the same sample count")
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError("partition weights must be finite and non-negative")
        if labels.min(initial=0) < 0 or labels.max(initial=0) >= cluster_count:
            raise ValueError("partition label is outside [0,K)")
        rows = np.arange(sample_count, dtype=np.int64)
        values = np.full(sample_count, np.sqrt(weight), dtype=np.float32)
        blocks.append(
            sparse.csr_matrix(
                (values, (rows, labels)),
                shape=(sample_count, cluster_count),
                dtype=np.float32,
            )
        )
    return sparse.hstack(blocks, format="csr")


def reliability_partition_consensus(
    representations,
    cluster_count,
    view_gate,
    *,
    view_mass=0.50,
    partition_seed=0,
    partition_n_init=20,
    final_seed=0,
    final_n_init=20,
):
    """Cluster every dataset with one reliability-weighted partition head.

    The common partition has unit mass.  The declared total view mass is
    distributed by the label-free view gate.  Since a permutation of cluster
    identifiers only permutes columns inside one sparse block, no label
    alignment or dataset-specific recovery algorithm is required.
    """
    cluster_count = int(cluster_count)
    view_gate = np.asarray(view_gate, dtype=np.float64).reshape(-1)
    if view_gate.size <= 0 or np.any(view_gate < 0.0):
        raise ValueError("view_gate must contain non-negative view weights")
    view_gate = view_gate / max(float(view_gate.sum()), 1e-12)
    view_mass = float(view_mass)
    if view_mass < 0.0 or not np.isfinite(view_mass):
        raise ValueError("view_mass must be finite and non-negative")

    names = ["learned_common"] + [
        "learned_view%d" % (index + 1) for index in range(view_gate.size)
    ]
    missing = [name for name in names if name not in representations]
    if missing:
        raise ValueError("missing learned representations: %s" % missing)
    partitions = [
        kmeans_clustering(
            representations[name],
            cluster_count,
            seed=int(partition_seed),
            n_init=int(partition_n_init),
        )
        for name in names
    ]
    weights = [1.0] + (view_mass * view_gate).tolist()
    embedding = weighted_partition_embedding(
        partitions, weights, cluster_count
    )
    prediction = KMeans(
        n_clusters=cluster_count,
        random_state=int(final_seed),
        n_init=int(final_n_init),
        max_iter=900,
    ).fit_predict(embedding)
    return prediction.astype(np.int64), {
        "partition_weights": weights,
        "partition_embedding_dim": int(embedding.shape[1]),
        "view_mass": view_mass,
        "partition_n_init": int(partition_n_init),
        "final_n_init": int(final_n_init),
    }


def kmeans_clustering(representation, num_clusters, seed=0, n_init=200):
    """Apply the one final cosine-KMeans readout used on every dataset."""
    representation = l2_normalize(representation)
    clusterer = KMeans(
        n_clusters=int(num_clusters),
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=900,
    )
    return clusterer.fit_predict(representation).astype(np.int64)


def agreement_weighted_multigranular_consensus(
    representations,
    num_clusters,
    *,
    granularities=(1.0, 2.0),
    base_seeds=(0, 17),
    base_n_init=20,
    final_n_init=200,
    max_dim=128,
    seed=0,
):
    """One label-free multi-view ensemble head shared by every dataset.

    A fixed set of KMeans partitions is produced for the learned common code
    and every learned view at the same granularities and seeds.  A base
    partition receives the mean non-negative ARI that it has with all other
    base partitions.  Its weighted one-hot incidence block is then compressed
    with truncated SVD and clustered once more.  ARI is invariant to a
    permutation of cluster identifiers, so neither label alignment nor
    ground-truth classes enter the weighting rule.

    This is deliberately a single equation: no dataset name, metric result,
    raw-input branch, or candidate selector is accepted by the function.
    """
    if not representations:
        raise ValueError("representations must be non-empty")
    cluster_count = int(num_clusters)
    if cluster_count < 2:
        raise ValueError("num_clusters must be at least two")
    factors = tuple(float(value) for value in granularities)
    seeds = tuple(int(value) for value in base_seeds)
    if not factors or any(value <= 0.0 for value in factors):
        raise ValueError("granularities must be positive")
    if not seeds:
        raise ValueError("base_seeds must be non-empty")

    names = tuple(sorted(representations))
    sample_count = None
    prepared = {}
    for name in names:
        values = l2_normalize(representations[name])
        if values.ndim != 2:
            raise ValueError("%s must have shape [N,D]" % name)
        if sample_count is None:
            sample_count = int(values.shape[0])
        elif int(values.shape[0]) != sample_count:
            raise ValueError("all representations must share N")
        prepared[name] = values
    if sample_count is None or sample_count < 3:
        raise ValueError("at least three samples are required")

    partitions = []
    metadata = []
    for name in names:
        values = prepared[name]
        for factor in factors:
            base_cluster_count = min(
                sample_count - 1,
                max(2, int(round(cluster_count * factor))),
            )
            for base_seed in seeds:
                labels = KMeans(
                    n_clusters=base_cluster_count,
                    random_state=base_seed,
                    n_init=int(base_n_init),
                    max_iter=600,
                ).fit_predict(values).astype(np.int64)
                partitions.append(labels)
                metadata.append(
                    {
                        "representation": name,
                        "granularity": factor,
                        "clusters": base_cluster_count,
                        "seed": base_seed,
                    }
                )

    agreement = np.eye(len(partitions), dtype=np.float64)
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            value = max(
                0.0,
                float(adjusted_rand_score(partitions[left], partitions[right])),
            )
            agreement[left, right] = value
            agreement[right, left] = value
    if len(partitions) == 1:
        weights = np.ones(1, dtype=np.float64)
    else:
        weights = (agreement.sum(axis=1) - 1.0) / float(
            len(partitions) - 1
        )
        # A small common floor prevents one legitimately complementary view
        # from disappearing merely because its partition differs from the
        # majority.  The rule and coefficient are identical on every dataset.
        weights = 0.05 + np.maximum(weights, 0.0)
        weights /= weights.sum()

    rows = np.arange(sample_count, dtype=np.int64)
    incidence_blocks = []
    for labels, weight, item in zip(partitions, weights, metadata):
        item["agreement_weight"] = float(weight)
        incidence_blocks.append(
            sparse.csr_matrix(
                (
                    np.full(sample_count, np.sqrt(weight), dtype=np.float64),
                    (rows, labels),
                ),
                shape=(sample_count, int(item["clusters"])),
            )
        )
    incidence = sparse.hstack(incidence_blocks, format="csr")
    target_dim = min(
        int(max_dim),
        max(1, sample_count - 1),
        max(1, incidence.shape[1] - 1),
    )
    if incidence.shape[1] <= target_dim:
        embedding = incidence.toarray()
        explained_variance = 1.0
    else:
        reducer = TruncatedSVD(
            n_components=target_dim,
            algorithm="randomized",
            n_iter=7,
            random_state=int(seed),
        )
        embedding = reducer.fit_transform(incidence)
        explained_variance = float(
            np.sum(reducer.explained_variance_ratio_)
        )
    embedding = l2_normalize(embedding)
    prediction = kmeans_clustering(
        embedding,
        cluster_count,
        seed=int(seed),
        n_init=int(final_n_init),
    )
    diagnostics = {
        "version": "agreement_weighted_multigranular_v1",
        "representations": list(names),
        "granularities": list(factors),
        "base_seeds": list(seeds),
        "base_partition_count": len(partitions),
        "incidence_columns": int(incidence.shape[1]),
        "embedding_dim": int(embedding.shape[1]),
        "explained_variance": explained_variance,
        "mean_pairwise_ari": float(
            agreement[np.triu_indices(len(partitions), 1)].mean()
            if len(partitions) > 1
            else 1.0
        ),
        "partitions": metadata,
    }
    return prediction, embedding, diagnostics


def reliability_weighted_similarity_network_fusion(
    learned_views,
    num_clusters,
    *,
    view_weights=None,
    neighbour_count=10,
    diffusion_steps=10,
    local_scale=0.5,
    seed=0,
):
    """Fuse learned-view neighbourhoods with one cross-diffusion equation.

    Each view first produces a locally scaled cosine-affinity matrix and a
    row-stochastic KNN transition matrix.  At every diffusion step a view is
    updated from the reliability-weighted mean of *the other views*, avoiding
    a self-copy shortcut.  The function accepts neither labels nor a dataset
    identifier and applies the same neighbourhood/diffusion rule everywhere.

    The dense implementation is intentionally used only for compact-data
    diagnostics.  A sparse backend is required before enabling this route on
    large datasets.
    """
    if len(learned_views) < 2:
        raise ValueError("at least two learned views are required")
    prepared = [l2_normalize(values) for values in learned_views]
    sample_count = int(prepared[0].shape[0])
    if sample_count < 3 or any(
        values.ndim != 2 or int(values.shape[0]) != sample_count
        for values in prepared
    ):
        raise ValueError("learned views must share a valid [N,D] shape")
    neighbour_count = min(
        max(2, int(neighbour_count)), max(2, sample_count - 1)
    )
    diffusion_steps = max(1, int(diffusion_steps))
    local_scale = max(float(local_scale), 1e-6)
    if view_weights is None:
        weights = np.full(len(prepared), 1.0 / float(len(prepared)))
    else:
        weights = np.asarray(view_weights, dtype=np.float64).reshape(-1)
        if weights.size != len(prepared) or np.any(weights < 0.0):
            raise ValueError("view_weights must be non-negative with length V")
        weights = weights / max(float(weights.sum()), 1e-12)

    full_transitions = []
    local_transitions = []
    for values in prepared:
        distance = np.maximum(0.0, 1.0 - np.matmul(values, values.T))
        np.fill_diagonal(distance, 0.0)
        ordered = np.argpartition(
            distance, kth=neighbour_count, axis=1
        )[:, : neighbour_count + 1]
        local_mean = np.take_along_axis(distance, ordered, axis=1).mean(
            axis=1
        )
        denominator = local_scale * (
            local_mean[:, None]
            + local_mean[None, :]
            + distance
        ) / 3.0
        affinity = np.exp(-np.square(distance) / np.maximum(denominator, 1e-12))
        np.fill_diagonal(affinity, 0.0)
        affinity = 0.5 * (affinity + affinity.T)
        transition = affinity / np.maximum(
            affinity.sum(axis=1, keepdims=True), 1e-12
        )
        transition = 0.5 * transition
        np.fill_diagonal(transition, 0.5)

        rows = np.repeat(np.arange(sample_count), ordered.shape[1])
        columns = ordered.reshape(-1)
        local_values = affinity[
            np.arange(sample_count)[:, None], ordered
        ].reshape(-1)
        local = sparse.csr_matrix(
            (local_values, (rows, columns)),
            shape=(sample_count, sample_count),
        )
        row_sum = np.asarray(local.sum(axis=1)).reshape(-1)
        local = sparse.diags(1.0 / np.maximum(row_sum, 1e-12)).dot(
            local
        )
        full_transitions.append(transition)
        local_transitions.append(local)

    for _ in range(diffusion_steps):
        updated = []
        for index, local in enumerate(local_transitions):
            other_indices = [
                item for item in range(len(prepared)) if item != index
            ]
            other_weights = weights[other_indices]
            other_weights = other_weights / max(
                float(other_weights.sum()), 1e-12
            )
            other = sum(
                weight * full_transitions[item]
                for weight, item in zip(other_weights, other_indices)
            )
            # S is sparse with K+1 entries per row.  Computing
            # S @ P @ S.T as two sparse-dense products keeps this diagnostic
            # O(V*T*K*N^2), rather than an unnecessary dense O(V*T*N^3).
            left = local.dot(other)
            propagated = local.dot(left.T).T
            propagated = 0.5 * (propagated + propagated.T)
            propagated = propagated / np.maximum(
                propagated.sum(axis=1, keepdims=True), 1e-12
            )
            propagated *= 0.5
            np.fill_diagonal(propagated, 0.5)
            updated.append(propagated)
        full_transitions = updated

    fused = sum(
        weight * transition
        for weight, transition in zip(weights, full_transitions)
    )
    fused = np.maximum(0.0, 0.5 * (fused + fused.T))
    prediction = SpectralClustering(
        n_clusters=int(num_clusters),
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=50,
        random_state=int(seed),
    ).fit_predict(fused)
    return prediction.astype(np.int64), fused, {
        "version": "reliability_weighted_snf_v1",
        "view_count": len(prepared),
        "view_weights": weights.tolist(),
        "neighbour_count": int(neighbour_count),
        "diffusion_steps": int(diffusion_steps),
        "local_scale": float(local_scale),
    }
