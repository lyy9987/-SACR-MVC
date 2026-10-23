"""Label-free structural anchor construction for pseudo-label warm-up."""

import itertools

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize


def _as_numpy(view):
    if hasattr(view, "detach"):
        view = view.detach().cpu().numpy()
    return np.asarray(view, dtype=np.float64)


def structural_representation(view, max_dim=256):
    """Return the deterministic normalized raw-view representation used by teachers."""
    x = _as_numpy(view)
    if x.shape[1] > max_dim:
        x = PCA(n_components=max_dim, random_state=0).fit_transform(x)
    return normalize(x)


def view_structure_score(
    view, cluster_count, stability_runs=5, return_partitions=False
):
    """Measure cluster stability and compactness without ground-truth labels."""
    x = structural_representation(view)
    predictions = []
    silhouettes = []
    sample_size = min(2000, len(x))
    for seed in range(int(stability_runs)):
        prediction = KMeans(
            n_clusters=int(cluster_count),
            random_state=seed,
            n_init=1,
            max_iter=600,
        ).fit_predict(x)
        predictions.append(prediction)
        silhouettes.append(
            silhouette_score(
                x,
                prediction,
                metric="cosine",
                sample_size=sample_size,
                random_state=0,
            )
        )
    pairwise_stability = [
        adjusted_rand_score(predictions[first], predictions[second])
        for first, second in itertools.combinations(range(len(predictions)), 2)
    ]
    stability = float(np.mean(pairwise_stability)) if pairwise_stability else 1.0
    compactness = float(np.mean(silhouettes))
    result = (stability + compactness, stability, compactness)
    return result + (predictions,) if return_partitions else result


def build_structural_anchor(
    views,
    cluster_count,
    knn_k=5,
    stability_runs=5,
    spectral_max_samples=5000,
    mode="single_view",
):
    """Obtain deterministic, label-free structural warm-up labels.

    ``single_view`` preserves the audited SACR-MVC v1 teacher.  ``coassociation``
    is the v2 default: it ensembles every view's multi-seed partitions in a
    permutation-invariant sparse sample/partition embedding.  It never
    materialises an N x N co-association matrix and therefore follows the same
    operator for two-view and many-view data.
    """
    mode = str(mode).lower()
    if mode not in {"single_view", "coassociation"}:
        raise ValueError("mode must be 'single_view' or 'coassociation'")
    diagnostics = []
    view_partitions = []
    view_partition_ensembles = []
    diagnostic_views = []
    for index, view in enumerate(views):
        score, stability, compactness, partitions = view_structure_score(
            view,
            cluster_count,
            stability_runs=stability_runs,
            return_partitions=True,
        )
        diagnostics.append(
            {
                "view": index,
                "score": score,
                "stability": stability,
                "compactness": compactness,
            }
        )
        diagnostic_x = structural_representation(view)
        diagnostic_views.append(diagnostic_x)
        view_partition_ensembles.append(partitions)
        view_partitions.append(
            KMeans(
                n_clusters=int(cluster_count),
                random_state=0,
                n_init=20,
                max_iter=900,
            ).fit_predict(diagnostic_x)
        )

    if mode == "coassociation":
        quality = np.asarray(
            [item["stability"] + item["compactness"] for item in diagnostics],
            dtype=np.float64,
        )
        logits = (quality - quality.max()) / 0.25
        weights = np.exp(logits)
        weights = weights / max(weights.sum(), 1e-12)
        # A probability floor prevents an early internal score from removing a
        # complementary view before representation learning has started.
        floor = min(0.05, 0.5 / float(len(views)))
        weights = (1.0 - floor * len(views)) * weights + floor
        selected = int(np.argmax(weights))
        for index, value in enumerate(weights):
            diagnostics[index]["teacher_weight"] = float(value)
            diagnostics[index]["centrality"] = None

        sample_count = len(view_partitions[0])
        run_count = sum(len(items) for items in view_partition_ensembles)
        row_indices = []
        column_indices = []
        values = []
        offset = 0
        rows = np.arange(sample_count, dtype=np.int64)
        for weight, partitions in zip(weights, view_partition_ensembles):
            partition_weight = np.sqrt(float(weight) / float(len(partitions)))
            for partition in partitions:
                row_indices.append(rows)
                column_indices.append(
                    offset + np.asarray(partition, dtype=np.int64)
                )
                values.append(
                    np.full(sample_count, partition_weight, dtype=np.float32)
                )
                offset += int(cluster_count)
        membership = sparse.coo_matrix(
            (
                np.concatenate(values),
                (np.concatenate(row_indices), np.concatenate(column_indices)),
            ),
            shape=(sample_count, run_count * int(cluster_count)),
            dtype=np.float32,
        ).tocsr()
        components = min(128, membership.shape[1] - 1, sample_count - 1)
        if components >= 2:
            embedding = TruncatedSVD(
                n_components=int(components), random_state=0
            ).fit_transform(membership)
        else:
            embedding = membership.toarray()
        embedding = normalize(embedding)
        labels = KMeans(
            n_clusters=int(cluster_count),
            random_state=0,
            n_init=20,
            max_iter=900,
        ).fit_predict(embedding)
        operator = "coassociation_svd_kmeans"
        return (
            labels.astype(np.int64),
            selected,
            operator,
            "stability_weighted_coassociation",
            diagnostics,
        )

    # Legacy single-view teacher: with at least three views, semantic evidence
    # shared by other modalities is used instead of internal stability.
    # is more trustworthy than a single view's internal compactness.  Adjusted
    # Rand agreement is permutation invariant and uses no ground-truth labels.
    # The two-view case is symmetric, so it falls back to the stability score.
    if len(views) >= 3:
        agreement = np.eye(len(views), dtype=np.float64)
        for first, second in itertools.combinations(range(len(views)), 2):
            value = adjusted_rand_score(
                view_partitions[first], view_partitions[second]
            )
            agreement[first, second] = value
            agreement[second, first] = value
        centrality = (agreement.sum(axis=1) - 1.0) / float(len(views) - 1)
        for index, value in enumerate(centrality):
            diagnostics[index]["centrality"] = float(value)
        selected = int(np.argmax(centrality))
        selector = "cross_view_centrality"
    else:
        selected = max(diagnostics, key=lambda item: item["score"])["view"]
        for item in diagnostics:
            item["centrality"] = None
        selector = "internal_stability"
    raw = normalize(_as_numpy(views[selected]))
    if len(raw) <= int(spectral_max_samples):
        graph = kneighbors_graph(
            raw,
            n_neighbors=min(int(knn_k), len(raw) - 1),
            mode="connectivity",
            metric="cosine",
            include_self=True,
            n_jobs=-1,
        )
        graph = graph.maximum(graph.T).astype(np.float64)
        labels = SpectralClustering(
            n_clusters=int(cluster_count),
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=0,
        ).fit_predict(graph)
        operator = "spectral"
    else:
        raw = structural_representation(raw)
        labels = KMeans(
            n_clusters=int(cluster_count),
            random_state=0,
            n_init=20,
            max_iter=900,
        ).fit_predict(raw)
        operator = "kmeans"

    return labels.astype(np.int64), selected, operator, selector, diagnostics
