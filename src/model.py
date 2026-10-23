import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from dataloader import get_all_multiview_data, get_multiview_data
from global_memory import (
    build_sparse_global_topology,
    ema_normalized_bank,
    sparse_global_topology_loss,
)
from unified_relation import build_unified_multiview_topology


def normalized_view_dominance(view_quality):
    """Return zero for uniform quality and one for a single-view monopoly."""
    quality = torch.as_tensor(view_quality).detach().float().reshape(-1)
    if quality.numel() < 2 or torch.any(quality < 0.0):
        raise ValueError("view_quality must contain at least two non-negative values")
    quality = quality / quality.sum().clamp_min(1e-8)
    view_count = float(quality.numel())
    return float(
        ((view_count * quality.max() - 1.0) / (view_count - 1.0))
        .clamp(0.0, 1.0)
        .item()
    )


def adaptive_semantic_projector_scale(
    view_quality,
    maximum_scale=1.0,
    dominance_floor=0.10,
    dominance_ceiling=0.20,
):
    """Continuously activate semantic capacity only for a dominant view.

    The statistic is normalized for the number of views.  It is exactly zero
    for equal-quality complementary views and approaches one only when one
    view carries nearly all reliability mass.  No dataset identity, label, or
    downstream metric is accepted by this function.
    """
    floor = float(dominance_floor)
    ceiling = float(dominance_ceiling)
    maximum = float(maximum_scale)
    if maximum < 0.0 or not np.isfinite(maximum):
        raise ValueError("maximum_scale must be finite and non-negative")
    if not 0.0 <= floor < ceiling <= 1.0:
        raise ValueError("dominance bounds must satisfy 0 <= floor < ceiling <= 1")
    dominance = normalized_view_dominance(view_quality)
    strength = float(np.clip((dominance - floor) / (ceiling - floor), 0.0, 1.0))
    return maximum * strength, dominance


class SparseKNNRelation:
    """Exact symmetric KNN relation stored as sorted sparse edge keys.

    Training consumes only a ``B x B`` subgraph, so materialising one dense
    ``N x N`` matrix per view is unnecessary.  This storage backend preserves
    the same symmetric edge set in ``O(Nk)`` memory and reconstructs the
    requested batch relation on demand.
    """

    def __init__(self, sample_count, edge_keys, *, device=None):
        self.sample_count = int(sample_count)
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")
        keys = torch.as_tensor(edge_keys, dtype=torch.long, device=device)
        if keys.ndim != 1:
            raise ValueError("edge_keys must be one-dimensional")
        self.edge_keys = torch.unique(keys, sorted=True)

    def batch_subgraph(self, sample_indices):
        sample_indices = torch.as_tensor(
            sample_indices, dtype=torch.long, device=self.edge_keys.device
        ).flatten()
        pair_keys = (
            sample_indices[:, None] * self.sample_count
            + sample_indices[None, :]
        )
        flat = pair_keys.reshape(-1)
        locations = torch.searchsorted(self.edge_keys, flat)
        valid = locations < self.edge_keys.numel()
        matched = torch.zeros_like(valid)
        matched[valid] = self.edge_keys[locations[valid]].eq(flat[valid])
        return matched.reshape(pair_keys.shape).to(torch.int32)

    @property
    def device(self):
        return self.edge_keys.device


class NeighborAwareBatchSampler:
    """Pack a deterministic matching of global KNN neighbours into batches.

    Every sample appears exactly once per epoch.  Pairs are formed from the
    union of the view-specific KNN graphs, prioritising edges supported by
    more views.  Packing never splits a pair, so both endpoints retain at
    least one true global neighbour inside the mini-batch without changing
    the epoch sample budget.
    """

    def __init__(self, view_graphs, batch_size, seed=0):
        if not view_graphs:
            raise ValueError("neighbor-aware batching requires view graphs")
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("neighbor-aware batch size must be at least two")
        first = view_graphs[0]
        self.sample_count = int(
            first.sample_count
            if isinstance(first, SparseKNNRelation)
            else first.size(0)
        )
        votes = [dict() for _ in range(self.sample_count)]
        for graph in view_graphs:
            graph_samples = int(
                graph.sample_count
                if isinstance(graph, SparseKNNRelation)
                else graph.size(0)
            )
            if graph_samples != self.sample_count:
                raise ValueError("all view graphs must share sample count")
            if isinstance(graph, SparseKNNRelation):
                edge_keys = graph.edge_keys.detach().cpu()
                rows = torch.div(
                    edge_keys, self.sample_count, rounding_mode="floor"
                ).numpy()
                columns = edge_keys.remainder(self.sample_count).numpy()
            else:
                rows_tensor, columns_tensor = torch.nonzero(
                    graph.detach(), as_tuple=True
                )
                rows = rows_tensor.cpu().numpy()
                columns = columns_tensor.cpu().numpy()
            for row, column in zip(rows.tolist(), columns.tolist()):
                if row == column:
                    continue
                votes[row][column] = votes[row].get(column, 0) + 1
        if any(not row_votes for row_votes in votes):
            raise ValueError("global KNN union contains an isolated sample")

        rng = np.random.RandomState(int(seed))
        permutation = rng.permutation(self.sample_count).tolist()
        available = np.ones(self.sample_count, dtype=np.bool_)
        groups = []
        paired_samples = 0
        for anchor in permutation:
            if not available[anchor]:
                continue
            available[anchor] = False
            # Random numbers only break equal-vote ties; multi-view consensus
            # support remains the primary ordering criterion.
            candidates = list(votes[anchor])
            tie_break = rng.random_sample(len(candidates))
            ordered = [
                candidate
                for _, candidate in sorted(
                    zip(tie_break, candidates),
                    key=lambda item: (
                        -votes[anchor][item[1]], item[0], item[1]
                    ),
                )
            ]
            partner = next(
                (candidate for candidate in ordered if available[candidate]),
                None,
            )
            if partner is None:
                groups.append([int(anchor)])
            else:
                available[partner] = False
                groups.append([int(anchor), int(partner)])
                paired_samples += 2
        rng.shuffle(groups)

        batches = []
        current = []
        for group in groups:
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        flattened = [index for batch in batches for index in batch]
        if sorted(flattened) != list(range(self.sample_count)):
            raise RuntimeError("neighbor-aware sampler lost or repeated samples")
        self.batches = batches
        self.paired_fraction = float(paired_samples) / float(self.sample_count)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


@torch.no_grad()
def get_sparse_knn_relation(data, k, chunk_size=256):
    """Compute exact Euclidean KNN in chunks and retain only sparse edges."""
    if data.ndim != 2:
        raise ValueError("data must have shape [N,D]")
    sample_count = int(data.size(0))
    neighbour_count = min(int(k), max(0, sample_count - 1))
    if neighbour_count <= 0:
        raise ValueError("k must provide at least one non-self neighbour")
    chunk_size = max(1, int(chunk_size))
    features = data.detach().float()
    squared_norm = (features * features).sum(dim=1)
    rows = []
    columns = []
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        query = features[start:stop]
        distance = (
            (query * query).sum(dim=1, keepdim=True)
            + squared_norm[None, :]
            - 2.0 * torch.mm(query, features.t())
        ).clamp_min_(0.0)
        local_rows = torch.arange(start, stop, device=features.device)
        distance[
            torch.arange(stop - start, device=features.device), local_rows
        ] = torch.inf
        neighbours = distance.topk(
            neighbour_count, dim=1, largest=False, sorted=False
        ).indices
        rows.append(local_rows[:, None].expand_as(neighbours).reshape(-1))
        columns.append(neighbours.reshape(-1))
        del distance
    row = torch.cat(rows)
    column = torch.cat(columns)
    forward = row * sample_count + column
    reverse = column * sample_count + row
    relation = SparseKNNRelation(
        sample_count,
        torch.cat([forward, reverse]),
        device=features.device,
    )
    if relation.edge_keys.numel() < sample_count * neighbour_count:
        raise RuntimeError("sparse symmetric KNN relation lost directed edges")
    return relation


def get_knn_graph(data, k):
    num_samples = data.size(0)
    graph = torch.zeros(num_samples, num_samples, dtype=torch.int32, device=data.device)
    neighbour_count = min(int(k), max(0, num_samples - 1))

    for i in range(num_samples):
        distance = torch.sum((data - data[i]) ** 2, dim=1)
        # Request one extra item because the first index is the sample itself.
        _, small_indices = torch.topk(
            distance, neighbour_count + 1, largest=False
        )
        graph[i, small_indices[1:]] = 1

    result_graph = torch.max(graph, graph.t())
    return result_graph


def get_W(mv_data, k, storage="auto", dense_budget_mb=64.0, chunk_size=256):
    W = []
    mv_data_loader, num_views, num_samples, _ = get_all_multiview_data(mv_data)
    storage = str(storage).lower()
    if storage not in {"auto", "dense", "sparse"}:
        raise ValueError("storage must be one of: auto, dense, sparse")
    estimated_dense_mb = (
        float(num_views) * float(num_samples) * float(num_samples) * 4.0
        / (1024.0 * 1024.0)
    )
    resolved_storage = storage
    if storage == "auto":
        resolved_storage = (
            "dense" if estimated_dense_mb <= float(dense_budget_mb) else "sparse"
        )
    print(
        "Exact KNN relation storage: %s estimated_dense_mb=%.1f budget_mb=%.1f"
        % (resolved_storage, estimated_dense_mb, float(dense_budget_mb))
    )
    for _, (sub_data_views, _, _) in enumerate(mv_data_loader):
        for i in range(num_views):
            result_graph = (
                get_knn_graph(sub_data_views[i], k)
                if resolved_storage == "dense"
                else get_sparse_knn_relation(
                    sub_data_views[i], k, chunk_size=chunk_size
                )
            )
            W.append(result_graph)
    return W


def batch_relation_graph(relation, sample_indices):
    """Return a dense batch graph from either exact storage backend."""
    if isinstance(relation, SparseKNNRelation):
        return relation.batch_subgraph(sample_indices)
    return relation[sample_indices][:, sample_indices]


@torch.no_grad()
def install_exact_multiview_memory_positives(model, view_graphs):
    """Install a sparse exact-global positive table from full raw-view KNNs."""
    if not view_graphs:
        raise ValueError("at least one full-view graph is required")
    first_graph = view_graphs[0]
    sample_count = int(
        first_graph.sample_count
        if isinstance(first_graph, SparseKNNRelation)
        else first_graph.size(0)
    )
    positive_k = int(model.fixed_memory_positive_indices.size(1))
    if sample_count != int(model.num_samples) or sample_count <= positive_k:
        raise ValueError("full-view graphs do not match the memory dimensions")

    votes = torch.zeros(
        sample_count,
        sample_count,
        dtype=torch.float32,
        device=first_graph.device,
    )
    for graph in view_graphs:
        if isinstance(graph, SparseKNNRelation):
            row = torch.div(
                graph.edge_keys, sample_count, rounding_mode="floor"
            )
            column = graph.edge_keys.remainder(sample_count)
            votes[row, column] += 1.0
        else:
            if tuple(graph.shape) != (sample_count, sample_count):
                raise ValueError("all full-view graphs must have shape [N,N]")
            votes.add_(graph.float())
    votes.div_(float(len(view_graphs)))
    votes.fill_diagonal_(0.0)
    values, indices = votes.topk(positive_k, dim=1, largest=True)
    if torch.any(values.sum(dim=1) <= 0.0):
        raise ValueError("exact multi-view KNN topology contains an empty row")
    weights = values.clamp_min(1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    model.fixed_memory_positive_indices.copy_(indices)
    model.fixed_memory_positive_weights.copy_(weights)
    model.fixed_memory_positive_enabled.fill_(True)
    print(
        "Exact multiview memory positives installed: views=%d positive_k=%d"
        % (len(view_graphs), positive_k)
    )


def route_graph_by_sample_reliability(
    own_graph,
    reference_graph,
    view_reliability,
    num_views,
    routing_strength=1.0,
):
    """Continuously route neighbourhood evidence at sample-pair level.

    ``view_reliability`` sums to one across views.  A uniformly reliable view
    therefore keeps its complete graph, while a below-uniform view mixes its
    own neighbourhood with the graph of the globally most reliable view.  The
    geometric pair confidence makes an edge conservative whenever either end
    point is unreliable.  Binary graphs are recovered exactly for full
    confidence, preserving the binary-graph objective at that endpoint.
    """
    routing_strength = float(routing_strength)
    if not 0.0 <= routing_strength <= 1.0:
        raise ValueError("routing_strength must lie in [0,1]")
    retention = (float(num_views) * view_reliability.float()).clamp(0.0, 1.0)
    pair_retention = torch.sqrt(
        retention.unsqueeze(1) * retention.unsqueeze(0)
    )
    # Strength zero is bit-for-bit the original per-view graph; strength one
    # is the complete reliability route.  Intermediate dominance produces a
    # continuous curriculum instead of a dataset-name switch.
    pair_retention = 1.0 - routing_strength * (1.0 - pair_retention)
    routed = (
        pair_retention * own_graph.float()
        + (1.0 - pair_retention) * reference_graph.float()
    )
    return 0.5 * (routed + routed.t())


def _normalize_01(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def _safe_one_hot(labels, num_clusters):
    out = torch.zeros(labels.size(0), num_clusters, dtype=torch.float32, device=labels.device)
    out[torch.arange(labels.size(0), device=labels.device), labels] = 1.0
    return out


def _normalised_entropy(probability):
    probability = probability.clamp_min(1e-8)
    entropy = -(probability * probability.log()).sum(dim=1)
    return entropy / max(float(np.log(probability.size(1))), 1e-8)


def _js_divergence(first, second):
    first = first.clamp_min(1e-8)
    second = second.clamp_min(1e-8)
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - midpoint.log())).sum(dim=1)
        + (second * (second.log() - midpoint.log())).sum(dim=1)
    ) / float(np.log(2.0))


def _robust_view_scale(values):
    lower = torch.quantile(values, 0.05, dim=0, keepdim=True)
    upper = torch.quantile(values, 0.95, dim=0, keepdim=True)
    return ((values - lower) / (upper - lower).clamp_min(1e-8)).clamp(0.0, 1.0)


@torch.no_grad()
def compute_counterfactual_reliability(
    model,
    zs,
    common_z,
    pseudo_labels,
    reconstruction_errors,
    assignment_temperature=0.20,
    reliability_temperature=0.20,
    conflict_weight=0.50,
    reconstruction_weight=0.25,
    floor=0.05,
    mode="marginal",
    total_floor=0.10,
):
    """Estimate each sample-view's marginal structural contribution.

    A view is useful when removing it makes the consensus assignment more
    uncertain.  It is discounted when its own assignment conflicts with the
    leave-one-view-out consensus or when it reconstructs the sample poorly.
    Every assignment is evaluated against centres sharing the same pseudo-label
    index, so probabilities from different spaces remain semantically aligned.
    """
    cluster_count = model.num_clusters
    common_centres = model.compute_centers(common_z, pseudo_labels)
    common_probability = F.softmax(
        torch.mm(F.normalize(common_z, dim=1), common_centres.t())
        / float(assignment_temperature),
        dim=1,
    )
    common_entropy = _normalised_entropy(common_probability)
    mode = str(mode).lower()
    if mode not in {"marginal", "hybrid"}:
        raise ValueError("reliability mode must be 'marginal' or 'hybrid'")
    scores = []
    utilities = []
    conflicts = []
    certainties = []
    common_agreements = []
    scaled_reconstruction = _robust_view_scale(reconstruction_errors)

    for view in range(model.num_views):
        mask = torch.ones(
            (common_z.size(0), model.num_views), device=common_z.device
        )
        mask[:, view] = 0.0
        mask = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
        leave_one_out = model.fusion(zs, mask)
        leave_centres = model.compute_centers(leave_one_out, pseudo_labels)
        leave_probability = F.softmax(
            torch.mm(F.normalize(leave_one_out, dim=1), leave_centres.t())
            / float(assignment_temperature),
            dim=1,
        )

        view_centres = model.compute_centers(zs[view], pseudo_labels)
        view_probability = F.softmax(
            torch.mm(F.normalize(zs[view], dim=1), view_centres.t())
            / float(assignment_temperature),
            dim=1,
        )
        utility = _normalised_entropy(leave_probability) - common_entropy
        conflict = _js_divergence(view_probability, leave_probability)
        certainty = 1.0 - _normalised_entropy(view_probability)
        common_agreement = 1.0 - _js_divergence(
            view_probability, common_probability
        )
        reconstruction_quality = 1.0 - scaled_reconstruction[:, view]
        if mode == "hybrid":
            # One calibrated sample-view score combines cooperative marginal
            # value, absolute certainty, consensus agreement and data quality.
            # Unlike JS(view, leave-one-out), the last three terms remain
            # discriminative when exactly two views are present.
            score = (
                0.35 * utility
                + 0.25 * certainty
                + 0.25 * common_agreement
                + 0.15 * reconstruction_quality
            )
        else:
            score = (
                utility
                - float(conflict_weight) * conflict
                - float(reconstruction_weight) * scaled_reconstruction[:, view]
            )
        scores.append(score)
        utilities.append(utility)
        conflicts.append(conflict)
        certainties.append(certainty)
        common_agreements.append(common_agreement)

    score_matrix = torch.stack(scores, dim=1)
    temperature = 0.30 if mode == "hybrid" else float(reliability_temperature)
    reliability = F.softmax(score_matrix / temperature, dim=1)
    if mode == "hybrid" and total_floor > 0.0:
        reliability = (
            (1.0 - float(total_floor)) * reliability
            + float(total_floor) / float(model.num_views)
        )
    elif floor > 0.0:
        reliability = (1.0 - floor * model.num_views) * reliability + floor
    diagnostics = {
        "counterfactual_utility": torch.stack(utilities, dim=1),
        "assignment_conflict": torch.stack(conflicts, dim=1),
        "view_certainty": torch.stack(certainties, dim=1),
        "common_agreement": torch.stack(common_agreements, dim=1),
        "reconstruction_evidence": scaled_reconstruction,
        "reliability_score": score_matrix,
    }
    return reliability, diagnostics


@torch.no_grad()
def estimate_pretrained_view_quality(
    model,
    dataset,
    batch_size,
    reliability_mode="marginal",
):
    """Estimate one fixed, label-free view-quality vector after pretraining.

    The estimate combines reconstruction evidence with each view's marginal
    contribution to a consensus partition.  It is computed once, before the
    clustering stage, so an adaptive capacity decision cannot oscillate with
    the pseudo labels it subsequently helps to learn.
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    was_training = bool(model.training)
    model.eval()
    view_batches = None
    reconstruction_batches = []
    try:
        for xs, _, _ in loader:
            xrs, zs = model(xs)
            if view_batches is None:
                view_batches = [[] for _ in zs]
            for view, values in enumerate(zs):
                view_batches[view].append(values.detach())
            reconstruction_batches.append(
                torch.stack(
                    [
                        (source - reconstruction).square().mean(dim=1)
                        for source, reconstruction in zip(xs, xrs)
                    ],
                    dim=1,
                )
            )
        if not view_batches:
            raise ValueError("cannot estimate view quality from an empty dataset")
        view_features = [torch.cat(items, dim=0) for items in view_batches]
        reconstruction_errors = torch.cat(reconstruction_batches, dim=0)
        common_features = model.fusion(view_features)
        pseudo_labels = model.clustering(common_features)
        reliability, diagnostics = compute_counterfactual_reliability(
            model,
            view_features,
            common_features,
            pseudo_labels,
            reconstruction_errors,
            mode=reliability_mode,
        )
        quality = model.robust_view_quality(reliability)
        return quality.detach(), {
            key: value.detach() for key, value in diagnostics.items()
        }
    finally:
        model.train(was_training)


def compute_trust_scores(
    model,
    common_z,
    zs,
    psedo_labels,
    trust_alpha=0.6,
    trust_tau=0.5,
    trust_geometry="cosine",
):
    geometry = str(trust_geometry).lower()
    if geometry == "cosine":
        metric_z = F.normalize(common_z, dim=1)
        centers = model.compute_centers(metric_z, psedo_labels)
        dist_mat = 1.0 - torch.mm(metric_z, centers.t()).clamp(-1.0, 1.0)
    elif geometry == "legacy_euclidean":
        centers = model.compute_centers(common_z, psedo_labels)
        dist_mat = torch.sum(
            (common_z.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2
        )
    else:
        raise ValueError("trust_geometry must be 'cosine' or 'legacy_euclidean'")
    sorted_dist, sorted_idx = torch.sort(dist_mat, dim=1)
    d1 = sorted_dist[:, 0]
    d2 = sorted_dist[:, 1]
    margin = d2 - d1
    margin_norm = _normalize_01(margin)

    common_norm = F.normalize(common_z, dim=1)
    cons_list = []
    for z in zs:
        z_norm = F.normalize(z, dim=1)
        cos = F.cosine_similarity(z_norm, common_norm, dim=1)
        cons_list.append((cos + 1.0) / 2.0)
    consistency = torch.stack(cons_list, dim=1).mean(dim=1)
    consistency = torch.clamp(consistency, 0.0, 1.0)

    # margin 小、跨视图一致性差 => 不确定性高 => trust 低。
    uncertainty = trust_alpha * (1.0 - margin_norm) + (1.0 - trust_alpha) * (1.0 - consistency)
    trust = torch.exp(-uncertainty / trust_tau)
    trust = torch.clamp(trust, 0.05, 1.0)

    return trust.detach(), margin_norm.detach(), consistency.detach(), sorted_idx[:, :2].detach(), sorted_dist[:, :2].detach(), centers.detach()


def build_confidence_adaptive_targets(
    psedo_labels,
    top2_idx,
    top2_dist,
    num_clusters,
    trust,
    soft_proto_temperature=0.6,
    min_soft_weight=0.10,
    max_soft_weight=0.70,
):
    """
    用“全体样本自适应软目标”替代旧版“只取低 trust 的固定比例边界样本”。
    - 高 trust：接近硬伪标签，稳定整体簇结构；
    - 低 trust：top-2 原型软分配，避免错误硬标签把边界样本拉错；
    - 所有样本都参与全局优化，因此目标从 Boundary-10/20/30 回到 Full dataset。
    """
    N = psedo_labels.size(0)
    device = psedo_labels.device
    hard_targets = _safe_one_hot(psedo_labels, num_clusters)

    top2_targets = torch.zeros((N, num_clusters), dtype=torch.float32, device=device)
    p2 = F.softmax(-top2_dist / soft_proto_temperature, dim=1)
    rows = torch.arange(N, device=device)
    top2_targets[rows, top2_idx[:, 0]] = p2[:, 0]
    top2_targets[rows, top2_idx[:, 1]] = p2[:, 1]

    # trust 越低，软目标权重越大；trust 越高，硬伪标签权重越大。
    soft_weight = (1.0 - trust).clamp(min_soft_weight, max_soft_weight).view(-1, 1)
    targets = (1.0 - soft_weight) * hard_targets + soft_weight * top2_targets
    targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
    return targets


@torch.no_grad()
def align_cluster_channels(reference_targets, current_labels):
    """Match current cluster channels to the preceding teacher channels.

    KMeans cluster identifiers are arbitrary.  A temporal teacher therefore
    needs a label-free maximum-overlap assignment before it can average two
    consecutive target matrices.  The matching uses only the two predicted
    partitions; ground-truth labels never enter this function.
    """
    if reference_targets.ndim != 2:
        raise ValueError("reference_targets must have shape [N,K]")
    current_labels = current_labels.long().flatten()
    sample_count, cluster_count = reference_targets.shape
    if current_labels.numel() != sample_count:
        raise ValueError("current_labels must match reference sample count")
    previous_labels = reference_targets.argmax(dim=1)
    pair_index = current_labels * cluster_count + previous_labels
    overlap = torch.bincount(
        pair_index,
        minlength=cluster_count * cluster_count,
    ).reshape(cluster_count, cluster_count)
    current_index, previous_index = linear_sum_assignment(
        -overlap.detach().cpu().numpy()
    )
    mapping = torch.empty(
        cluster_count,
        dtype=torch.long,
        device=current_labels.device,
    )
    mapping[
        torch.as_tensor(current_index, device=current_labels.device)
    ] = torch.as_tensor(previous_index, device=current_labels.device)
    return mapping


@torch.no_grad()
def update_aligned_temporal_teacher(
    model,
    current_labels,
    current_targets,
    current_centers,
    momentum=0.0,
    auxiliary_targets=None,
):
    """EMA pseudo targets/prototypes after label-free cluster alignment.

    ``momentum=0`` is the exact released endpoint.  Positive momentum creates
    an online teacher without a second encoder, dataset-specific thresholds or
    hard sample filtering.
    """
    momentum = float(momentum)
    if momentum <= 0.0:
        return current_targets, current_centers, auxiliary_targets
    if not 0.0 < momentum < 1.0:
        raise ValueError("pseudo_teacher_momentum must lie in [0, 1)")

    initialized = int(model.pseudo_teacher_updates.item()) > 0
    if not initialized:
        model.pseudo_teacher_updates.add_(1)
        return current_targets, current_centers, auxiliary_targets

    mapping = align_cluster_channels(model.soft_targets, current_labels)
    aligned_targets = torch.zeros_like(current_targets)
    aligned_targets[:, mapping] = current_targets
    aligned_centers = torch.zeros_like(current_centers)
    aligned_centers[mapping] = current_centers

    teacher_targets = (
        momentum * model.soft_targets
        + (1.0 - momentum) * aligned_targets
    )
    teacher_targets = teacher_targets / teacher_targets.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    teacher_centers = F.normalize(
        momentum * model.prototype_centers
        + (1.0 - momentum) * aligned_centers,
        dim=1,
    )

    aligned_auxiliary = auxiliary_targets
    if auxiliary_targets is not None:
        aligned_auxiliary = torch.zeros_like(auxiliary_targets)
        aligned_auxiliary[:, mapping] = auxiliary_targets
    model.pseudo_teacher_updates.add_(1)
    return teacher_targets, teacher_centers, aligned_auxiliary


@torch.no_grad()
def build_reliability_barycentric_targets(
    model,
    zs,
    pseudo_labels,
    reliability,
    assignment_temperature=0.20,
    reliability_power=2.0,
):
    """Build one product-of-experts target in aligned cluster coordinates.

    Every view centre is indexed by the same current pseudo partition.  The
    reliability-weighted logarithmic pool therefore combines probabilities in
    a shared semantic coordinate system and stays invariant to a view's own
    arbitrary KMeans label permutation.  The target is detached by design.
    """
    weights = model.effective_reliability(
        reliability, power=float(reliability_power)
    ).to(zs[0])
    log_probability = None
    temperature = max(float(assignment_temperature), 1e-6)
    for view, features in enumerate(zs):
        centres = model.compute_centers(features, pseudo_labels)
        probability = F.softmax(
            torch.mm(F.normalize(features, dim=1), centres.t()) / temperature,
            dim=1,
        ).clamp_min(1e-8)
        contribution = weights[:, view : view + 1] * probability.log()
        log_probability = (
            contribution
            if log_probability is None
            else log_probability + contribution
        )
    targets = F.softmax(log_probability, dim=1)
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)


def update_pseudo_trust_and_soft_targets(
    model,
    dataset,
    batch_size,
    trust_alpha=0.6,
    trust_tau=0.5,
    trust_geometry="cosine",
    soft_proto_temperature=0.5,
    enable_counterfactual_reliability=False,
    forced_pseudo_labels=None,
    reliability_mode="marginal",
    reliability_ema=0.80,
    epoch=None,
    enable_global_memory=False,
    global_memory_momentum=0.90,
    global_memory_refresh=5,
    global_memory_positive_k=9,
    global_memory_negative_k=64,
    global_memory_candidate_clusters=2,
    global_memory_chunk_size=512,
    global_memory_raw_weight=0.50,
    global_relation_mode="legacy_semantic",
    global_relation_consensus_weight=0.50,
    global_relation_semantic_floor=0.25,
    global_relation_raw_local_weight=0.50,
    barycentric_target_weight=0.0,
    barycentric_assignment_temperature=0.20,
    barycentric_reliability_power=2.0,
    global_gate_temperature=1e6,
    pseudo_teacher_momentum=0.0,
):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    commonZ_list = []
    z_list_per_view = None
    reconstruction_error_list = []

    for _, (xs, _, _) in enumerate(loader):
        with torch.no_grad():
            xrs, zs = model(xs)
            commonz = model.fusion(zs)
            commonZ_list.append(commonz)
            reconstruction_error_list.append(
                torch.stack(
                    [((x - xr) ** 2).mean(dim=1) for x, xr in zip(xs, xrs)],
                    dim=1,
                )
            )
            if z_list_per_view is None:
                z_list_per_view = [[] for _ in range(len(zs))]
            for i, z in enumerate(zs):
                z_list_per_view[i].append(z)

    commonZ = torch.cat(commonZ_list, dim=0)
    full_zs = [torch.cat(zs_i, dim=0) for zs_i in z_list_per_view]
    reconstruction_errors = torch.cat(reconstruction_error_list, dim=0)

    psedo_labels = (
        forced_pseudo_labels.to(commonZ.device).long()
        if forced_pseudo_labels is not None
        else model.clustering(commonZ)
    )
    if enable_counterfactual_reliability:
        reliability, diagnostics = compute_counterfactual_reliability(
            model,
            full_zs,
            commonZ,
            psedo_labels,
            reconstruction_errors,
            mode=reliability_mode,
        )
        # Both marginal and hybrid counterfactual estimates are pseudo-label
        # dependent and can jump when KMeans refreshes.  Apply the same EMA to
        # either estimator so the routing semantics match the public
        # ``reliability_ema`` configuration.
        if int(model.reliability_updates.item()) > 0:
            reliability = (
                float(reliability_ema) * model.sample_reliability
                + (1.0 - float(reliability_ema)) * reliability
            )
            reliability = reliability / reliability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        model.reliability_updates.add_(1)
        model.sample_reliability.copy_(reliability)
        model.update_global_view_gate(
            model.sample_reliability,
            temperature=global_gate_temperature,
        )
        commonZ = model.fusion(full_zs, model.sample_reliability).detach()
        if forced_pseudo_labels is None:
            psedo_labels = model.clustering(commonZ)
        model.reliability_diagnostics = {
            key: value.detach() for key, value in diagnostics.items()
        }
    else:
        model.sample_reliability.fill_(1.0 / float(model.num_views))
        model.global_view_gate.fill_(1.0 / float(model.num_views))
    trust, margin_scores, consistency_scores, top2_idx, top2_dist, centers = compute_trust_scores(
        model,
        commonZ,
        full_zs,
        psedo_labels,
        trust_alpha=trust_alpha,
        trust_tau=trust_tau,
        trust_geometry=trust_geometry,
    )

    from confidence_intervention import replace_trust, replace_targets
    confidence_mode = os.environ.get("SACR_CONFIDENCE_MODE", "none")
    trust = replace_trust(trust, confidence_mode)
    soft_targets = build_confidence_adaptive_targets(
        psedo_labels,
        top2_idx,
        top2_dist,
        model.num_clusters,
        trust,
        soft_proto_temperature=soft_proto_temperature,
    )
    barycentric_targets = soft_targets
    barycentric_weight = float(barycentric_target_weight)
    if enable_counterfactual_reliability and barycentric_weight > 0.0:
        barycentric_targets = build_reliability_barycentric_targets(
            model,
            full_zs,
            psedo_labels,
            model.sample_reliability,
            assignment_temperature=barycentric_assignment_temperature,
            reliability_power=barycentric_reliability_power,
        )
        barycentric_weight = max(0.0, min(1.0, barycentric_weight))
        soft_targets = (
            (1.0 - barycentric_weight) * soft_targets
            + barycentric_weight * barycentric_targets
        )
        soft_targets = soft_targets / soft_targets.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

    soft_targets, centers, barycentric_targets = (
        update_aligned_temporal_teacher(
            model,
            psedo_labels,
            soft_targets,
            centers,
            momentum=pseudo_teacher_momentum,
            auxiliary_targets=barycentric_targets,
        )
    )

    soft_targets = replace_targets(soft_targets, psedo_labels, confidence_mode)
    barycentric_targets = replace_targets(barycentric_targets, psedo_labels, confidence_mode)
    if confidence_mode == "off":
        assert bool(torch.all(trust == 1)), "confidence not fully neutralized"
        assert bool(torch.all(soft_targets.max(dim=1).values == 1)), "targets not hard"
    model.audit_confidence_mode = confidence_mode
    if not getattr(model, "audit_confidence_announced", False):
        print("Confidence intervention: %s; off means t=1 and hard pseudo-targets" % confidence_mode, flush=True)
        model.audit_confidence_announced = True
    model.psedo_labels = psedo_labels
    model.trust_scores = trust
    model.margin_scores = margin_scores
    model.consistency_scores = consistency_scores
    model.soft_targets = soft_targets
    model.barycentric_targets = barycentric_targets
    model.prototype_centers = centers
    if enable_global_memory:
        update_global_topology_memory(
            model,
            commonZ,
            soft_targets,
            trust,
            epoch=epoch,
            view_features=full_zs,
            sample_reliability=model.sample_reliability,
            momentum=global_memory_momentum,
            refresh_interval=global_memory_refresh,
            positive_k=global_memory_positive_k,
            negative_k=global_memory_negative_k,
            candidate_clusters=global_memory_candidate_clusters,
            chunk_size=global_memory_chunk_size,
            raw_weight=global_memory_raw_weight,
            relation_mode=global_relation_mode,
            relation_consensus_weight=global_relation_consensus_weight,
            relation_semantic_floor=global_relation_semantic_floor,
            relation_raw_local_weight=global_relation_raw_local_weight,
        )


@torch.no_grad()
def update_global_topology_memory(
    model,
    common_features,
    soft_targets,
    trust_scores,
    epoch,
    momentum=0.90,
    refresh_interval=5,
    positive_k=9,
    negative_k=64,
    candidate_clusters=2,
    chunk_size=512,
    raw_weight=0.50,
    relation_mode="legacy_semantic",
    relation_consensus_weight=0.50,
    relation_semantic_floor=0.25,
    relation_raw_local_weight=0.50,
    view_features=None,
    sample_reliability=None,
):
    """Update the stop-gradient consensus bank and periodic sparse topology."""
    if epoch is None:
        raise ValueError("epoch is required when global memory is enabled")
    if model.memory_positive_indices.size(1) != int(positive_k):
        raise ValueError("Network memory_positive_k does not match configuration")
    if model.memory_negative_indices.size(1) != int(negative_k):
        raise ValueError("Network memory_negative_k does not match configuration")

    initialized = bool(model.memory_initialized.item())
    updated_bank = ema_normalized_bank(
        model.global_feature_bank,
        common_features,
        momentum=float(momentum),
        initialized=initialized,
    )
    model.global_feature_bank.copy_(updated_bank)
    for view_index, features in enumerate(view_features or []):
        updated_view_bank = ema_normalized_bank(
            model.global_view_feature_banks[view_index],
            features,
            momentum=float(momentum),
            initialized=initialized,
        )
        model.global_view_feature_banks[view_index].copy_(updated_view_bank)
    model.memory_initialized.fill_(True)

    last_refresh = int(model.memory_last_refresh.item())
    should_refresh = (
        not initialized
        or last_refresh < 0
        or int(epoch) - last_refresh >= max(1, int(refresh_interval))
    )
    if not should_refresh:
        return

    relation_mode = str(relation_mode).lower()
    if relation_mode not in {"legacy_semantic", "multiview_consensus"}:
        raise ValueError(
            "global_relation_mode must be legacy_semantic or multiview_consensus"
        )
    if relation_mode == "multiview_consensus":
        if view_features is None or sample_reliability is None:
            raise ValueError(
                "multiview_consensus requires view_features and sample_reliability"
            )
        topology = build_unified_multiview_topology(
            model.global_feature_bank,
            view_features,
            soft_targets,
            trust_scores,
            sample_reliability,
            raw_view_features=(
                list(model.raw_view_relation_bank)
                if bool(model.raw_view_relation_initialized.item())
                else None
            ),
            raw_candidate_indices=(
                model.raw_view_candidate_indices
                if bool(model.raw_view_candidate_initialized.item())
                else None
            ),
            positive_k=int(positive_k),
            negative_k=int(negative_k),
            candidate_clusters=int(candidate_clusters),
            chunk_size=int(chunk_size),
            seed=int(epoch),
            consensus_weight=float(relation_consensus_weight),
            semantic_floor=float(relation_semantic_floor),
            raw_local_weight=float(relation_raw_local_weight),
        )
    else:
        topology = build_sparse_global_topology(
            model.global_feature_bank,
            soft_targets,
            trust_scores,
            positive_k=int(positive_k),
            negative_k=int(negative_k),
            candidate_clusters=int(candidate_clusters),
            chunk_size=int(chunk_size),
            seed=int(epoch),
            auxiliary_features=(
                model.raw_topology_bank
                if bool(model.raw_topology_initialized.item())
                else None
            ),
            auxiliary_weight=(
                float(raw_weight)
                if bool(model.raw_topology_initialized.item())
                else 0.0
            ),
        )
    if (
        relation_mode == "legacy_semantic"
        and bool(model.fixed_memory_positive_enabled.item())
    ):
        model.memory_positive_indices.copy_(
            model.fixed_memory_positive_indices
        )
        model.memory_positive_weights.copy_(
            model.fixed_memory_positive_weights
        )
        positive_source = "exact_multiview_knn"
    else:
        model.memory_positive_indices.copy_(topology["positive_indices"])
        model.memory_positive_weights.copy_(topology["positive_weights"])
        positive_source = (
            "multiview_relation_consensus"
            if relation_mode == "multiview_consensus"
            else "semantic_cluster_routing"
        )
    model.memory_negative_indices.copy_(topology["negative_indices"])
    model.memory_negative_weights.copy_(topology["negative_weights"])
    model.memory_last_refresh.fill_(int(epoch))
    print(
        "Sparse global topology refreshed: epoch=%d positive_k=%d negative_k=%d source=%s"
        % (
            int(epoch),
            int(positive_k),
            int(negative_k),
            positive_source,
        )
    )


def _ramp_value(epoch, warmup, ramp_epochs=30):
    if epoch <= warmup:
        return 0.0
    return float(min(1.0, (epoch - warmup) / max(1, ramp_epochs)))




def build_high_order_reliable_graph(W, sample_idx=None, steps=2, keep_topk=20, mix=0.55):
    """
    BGCR-v38：由多视图 KNN 图构造高阶可靠关系图。
    - 多视图共同邻接越多，基础可靠性越高；
    - 随机游走扩散用于发现“表面不近但全局可达”的潜在同簇样本；
    - 只保留每行 top-k，避免图过密导致错误正对传播。
    返回当前 batch 的 [B, B] 软可靠图，值域 [0,1]。
    """
    if W is None or len(W) == 0:
        return None
    device = W[0].device
    A = None
    for wi in W:
        wi = wi.float().to(device)
        A = wi if A is None else A + wi
    A = A / float(len(W))
    A = torch.maximum(A, A.t())
    A.fill_diagonal_(1.0)
    P = A / (A.sum(dim=1, keepdim=True) + 1e-8)
    H = P
    for _ in range(max(1, int(steps)) - 1):
        H = torch.mm(H, P)
    R = float(mix) * A + (1.0 - float(mix)) * H
    R = torch.maximum(R, R.t())
    R.fill_diagonal_(0.0)
    if keep_topk is not None and int(keep_topk) > 0 and int(keep_topk) < R.size(1):
        vals, idx = torch.topk(R, k=int(keep_topk), dim=1, largest=True)
        R2 = torch.zeros_like(R)
        R2.scatter_(1, idx, vals)
        R = torch.maximum(R2, R2.t())
    R = R / (R.max() + 1e-8)
    if sample_idx is not None:
        return R[sample_idx][:, sample_idx]
    return R


def pre_train(
    model,
    mv_data,
    batch_size,
    epochs,
    optimizer,
    epoch_callback=None,
):
    """Reconstruction pretraining with an optional label-free epoch hook."""
    model.train()
    mv_data_loader, num_views, _, _ = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.
        for _, (sub_data_views, _, _) in enumerate(mv_data_loader):
            xrs, _ = model(sub_data_views)
            loss_list = []
            for idx in range(num_views):
                loss_list.append(criterion(sub_data_views[idx], xrs[idx]))
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss))

        if epoch_callback is not None:
            epoch_callback(model, optimizer, epoch, total_loss)


@torch.no_grad()
def collect_multilevel_predictions(model, dataset, batch_size):
    """Collect auxiliary projections/probabilities without reading labels."""
    if not model.enable_multilevel_heads:
        raise RuntimeError("multilevel heads are disabled")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    projection_batches = None
    probability_batches = None
    was_training = bool(model.training)
    model.eval()
    try:
        for xs, _, _ in loader:
            _, features = model(xs)
            projections, logits = model.multilevel_outputs(features)
            if projection_batches is None:
                projection_batches = [[] for _ in projections]
                probability_batches = [[] for _ in logits]
            for view, values in enumerate(projections):
                projection_batches[view].append(values.detach().cpu())
            for view, values in enumerate(logits):
                probability_batches[view].append(
                    F.softmax(values, dim=1).detach().cpu()
                )
    finally:
        model.train(was_training)
    projections = [torch.cat(items, dim=0) for items in projection_batches]
    probabilities = [torch.cat(items, dim=0) for items in probability_batches]
    return projections, probabilities


def build_multilevel_pseudo_partitions(
    model, dataset, batch_size, n_init=100
):
    """Cluster every instance-projection view under one fixed protocol."""
    projections, _ = collect_multilevel_predictions(
        model, dataset, batch_size
    )
    partitions = []
    for values in projections:
        partition = KMeans(
            n_clusters=int(model.num_clusters),
            random_state=0,
            n_init=int(n_init),
            max_iter=900,
        ).fit_predict(F.normalize(values.float(), dim=1).numpy())
        partitions.append(
            torch.as_tensor(
                partition,
                dtype=torch.long,
                device=model.weights.device,
            )
        )
    return partitions


@torch.no_grad()
def align_pseudo_partition_to_head(pseudo_labels, head_labels, cluster_count):
    """Map a predicted pseudo partition into the current head coordinates."""
    pseudo_labels = pseudo_labels.long().flatten()
    head_labels = head_labels.long().flatten()
    cluster_count = int(cluster_count)
    if pseudo_labels.shape != head_labels.shape:
        raise ValueError("pseudo/head labels must have equal shape")
    pair = pseudo_labels * cluster_count + head_labels
    overlap = torch.bincount(
        pair, minlength=cluster_count * cluster_count
    ).reshape(cluster_count, cluster_count)
    pseudo_index, head_index = linear_sum_assignment(
        -overlap.detach().cpu().numpy()
    )
    mapping = torch.empty(
        cluster_count, dtype=torch.long, device=pseudo_labels.device
    )
    mapping[
        torch.as_tensor(pseudo_index, device=mapping.device)
    ] = torch.as_tensor(head_index, device=mapping.device)
    return mapping[pseudo_labels]


def multilevel_pseudo_tune(
    model,
    dataset,
    batch_size,
    epochs,
    optimizer,
    pseudo_n_init=100,
):
    """MFLVC-style label-free pseudo-assignment head refinement."""
    epochs = int(epochs)
    if epochs <= 0:
        return
    pseudo_partitions = build_multilevel_pseudo_partitions(
        model, dataset, batch_size, n_init=pseudo_n_init
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    for epoch in range(1, epochs + 1):
        _, full_probabilities = collect_multilevel_predictions(
            model, dataset, batch_size
        )
        aligned_targets = [
            align_pseudo_partition_to_head(
                pseudo,
                probability.argmax(dim=1).to(pseudo.device),
                model.num_clusters,
            )
            for pseudo, probability in zip(
                pseudo_partitions, full_probabilities
            )
        ]
        model.train()
        total_loss = 0.0
        for xs, _, sample_indices in loader:
            _, features = model(xs)
            _, logits = model.multilevel_outputs(features)
            loss = torch.stack(
                [
                    F.cross_entropy(
                        values,
                        aligned_targets[view][sample_indices],
                    )
                    for view, values in enumerate(logits)
                ]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch % 10 == 0 or epoch == epochs:
            print(
                "Multi-level tune, epoch {} loss:{:.7f}".format(
                    epoch, total_loss
                )
            )


def leave_one_out_cross_view_reconstruction_loss(
    model,
    target_views,
    view_features,
    reliability=None,
):
    """Reconstruct every view from a consensus that excludes that view.

    The target encoder is absent from its own predictor, preventing an
    identity shortcut.  All target decoders and all source encoders share the
    same objective.  Optional reliability only weights the remaining source
    views and is renormalised after exclusion.
    """
    view_count = len(view_features)
    if view_count < 2 or len(target_views) != view_count:
        raise ValueError("cross-view reconstruction requires aligned views")
    batch_size, feature_dim = view_features[0].shape
    if any(item.shape != (batch_size, feature_dim) for item in view_features):
        raise ValueError("all latent views must share shape [B,D]")
    if reliability is None:
        reliability = torch.full(
            (batch_size, view_count),
            1.0 / float(view_count),
            dtype=view_features[0].dtype,
            device=view_features[0].device,
        )
    else:
        reliability = reliability.to(view_features[0])
        if reliability.shape != (batch_size, view_count):
            raise ValueError("reliability must have shape [B,V]")
    stacked = torch.stack(view_features, dim=1)
    losses = []
    for target in range(view_count):
        source_mass = reliability.clone()
        source_mass[:, target] = 0.0
        source_mass = source_mass / source_mass.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        source_consensus = (
            stacked * source_mass.unsqueeze(2)
        ).sum(dim=1)
        prediction = model.decoders[target](source_consensus)
        losses.append(F.mse_loss(prediction, target_views[target]))
    return torch.stack(losses).mean()


def contrastive_train(
    model,
    mv_data,
    mvc_loss,
    batch_size,
    epoch,
    W,
    alpha,
    beta,
    trust_warmup_epochs,
    reliability_warmup_epochs,
    trust_alpha,
    trust_tau,
    trust_geometry,
    trust_pair_lambda,
    soft_proto_temperature,
    proto_loss_weight,
    optimizer,
    assign_loss_weight=0.05,
    assignment_confidence_power=0.0,
    pseudo_teacher_momentum=0.0,
    cross_view_correlation_weight=0.0,
    cross_view_redundancy_weight=1.0,
    canonical_correlation_weight=0.0,
    canonical_correlation_regularization=1e-3,
    instance_contrast_weight=0.0,
    instance_contrast_temperature=0.20,
    instance_false_negative_floor=0.05,
    cross_view_reconstruction_weight=0.0,
    shared_private_orthogonality_weight=0.0,
    multilevel_instance_weight=0.0,
    multilevel_cluster_weight=0.0,
    multilevel_instance_temperature=0.5,
    multilevel_cluster_temperature=1.0,
    balance_loss_weight=0.005,
    sharpen_loss_weight=0.002,
    ramp_epochs=30,
    graph_routing="none",
    adaptive_reliability_routing=False,
    reliability_routing_dominance_floor=0.10,
    reliability_routing_dominance_ceiling=0.20,
    adaptive_assignment_confidence=False,
    assignment_confidence_max_power=2.0,
    structural_anchor_epochs=0,
    reliability_mode="marginal",
    reliability_ema=0.80,
    assignment_temperature=0.20,
    feature_alignment_scale=1.0,
    enable_global_memory=False,
    global_topology_weight=0.05,
    global_memory_temperature=0.20,
    global_memory_warmup_epochs=None,
    global_memory_ramp_epochs=10,
    global_memory_momentum=0.90,
    global_memory_refresh=5,
    global_memory_positive_k=9,
    global_memory_negative_k=64,
    global_memory_candidate_clusters=2,
    global_memory_chunk_size=512,
    global_memory_raw_weight=0.50,
    global_relation_mode="legacy_semantic",
    global_relation_consensus_weight=0.50,
    global_relation_semantic_floor=0.25,
    global_relation_raw_local_weight=0.50,
    global_relation_loss_objective="relation_energy",
    global_memory_self_weight=0.0,
    global_memory_query="views",
    relation_barycenter_weight=0.0,
    relation_barycenter_temperature=0.50,
    use_bgcr_v38=False,
    reliable_graph_weight=0.0,
    reliable_graph_k=20,
    reliable_graph_steps=2,
    cluster_contrast_weight=0.0,
    cluster_contrast_temperature=0.5,
    barycentric_target_weight=0.0,
    barycentric_assignment_temperature=0.20,
    barycentric_reliability_power=2.0,
    global_gate_temperature=1e6,
    adaptive_semantic_projector=False,
    semantic_projector_max_scale=0.0,
    semantic_projector_dominance_floor=0.10,
    semantic_projector_dominance_ceiling=0.20,
    adaptive_shared_private=False,
    shared_private_max_scale=0.0,
    shared_private_dominance_floor=0.10,
    shared_private_dominance_ceiling=0.20,
    normalized_multi_positive=False,
    neighbor_aware_batching=False,
    context_moe_alignment_weight=0.0,
    context_moe_alignment_temperature=0.5,
):
    model.train()
    global_memory_query = str(global_memory_query).lower()
    if global_memory_query not in {"views", "consensus"}:
        raise ValueError("global_memory_query must be 'views' or 'consensus'")
    if bool(neighbor_aware_batching):
        if W is None:
            raise ValueError("neighbor-aware batching requires exact view graphs")
        batch_sampler = NeighborAwareBatchSampler(
            W, batch_size, seed=int(epoch)
        )
        mv_data_loader = torch.utils.data.DataLoader(
            mv_data, batch_sampler=batch_sampler
        )
        num_views = len(mv_data.data_views)
        if epoch == 1 or epoch % 10 == 0:
            print(
                "Neighbor-aware batching: epoch=%d paired_fraction=%.4f "
                "batches=%d"
                % (epoch, batch_sampler.paired_fraction, len(batch_sampler))
            )
    else:
        mv_data_loader, num_views, _, _ = get_multiview_data(
            mv_data, batch_size
        )
    criterion = torch.nn.MSELoss()
    total_loss = 0.

    reliability_active = epoch > reliability_warmup_epochs
    use_trust = epoch > trust_warmup_epochs
    update_pseudo_trust_and_soft_targets(
        model,
        mv_data,
        batch_size,
        trust_alpha=trust_alpha,
        trust_tau=trust_tau,
        trust_geometry=trust_geometry,
        soft_proto_temperature=soft_proto_temperature,
        enable_counterfactual_reliability=reliability_active,
        forced_pseudo_labels=(
            model.structural_anchor_labels
            if epoch <= int(structural_anchor_epochs)
            and hasattr(model, "structural_anchor_labels")
            else None
        ),
        reliability_mode=reliability_mode,
        reliability_ema=reliability_ema,
        epoch=epoch,
        enable_global_memory=enable_global_memory,
        global_memory_momentum=global_memory_momentum,
        global_memory_refresh=global_memory_refresh,
        global_memory_positive_k=global_memory_positive_k,
        global_memory_negative_k=global_memory_negative_k,
        global_memory_candidate_clusters=global_memory_candidate_clusters,
        global_memory_chunk_size=global_memory_chunk_size,
        global_memory_raw_weight=global_memory_raw_weight,
        global_relation_mode=global_relation_mode,
        global_relation_consensus_weight=global_relation_consensus_weight,
        global_relation_semantic_floor=global_relation_semantic_floor,
        global_relation_raw_local_weight=global_relation_raw_local_weight,
        barycentric_target_weight=barycentric_target_weight,
        barycentric_assignment_temperature=barycentric_assignment_temperature,
        barycentric_reliability_power=barycentric_reliability_power,
        global_gate_temperature=global_gate_temperature,
        pseudo_teacher_momentum=pseudo_teacher_momentum,
    )
    if bool(adaptive_semantic_projector):
        if reliability_active:
            _, current_projector_dominance = (
                adaptive_semantic_projector_scale(
                    model.global_view_quality,
                    maximum_scale=semantic_projector_max_scale,
                    dominance_floor=semantic_projector_dominance_floor,
                    dominance_ceiling=semantic_projector_dominance_ceiling,
                )
            )
            # Treat dominance as curriculum evidence, not as a rapidly
            # changing architecture knob.  Once a strong-view regime has
            # been observed, retaining that evidence prevents the projector
            # from shrinking merely because it succeeds in bringing the
            # weaker views closer to the dominant view.
            projector_dominance = max(
                float(getattr(model, "semantic_projector_dominance", 0.0)),
                float(current_projector_dominance),
            )
            projector_scale, _ = adaptive_semantic_projector_scale(
                [
                    (1.0 + projector_dominance) / 2.0,
                    (1.0 - projector_dominance) / 2.0,
                ],
                maximum_scale=semantic_projector_max_scale,
                dominance_floor=semantic_projector_dominance_floor,
                dominance_ceiling=semantic_projector_dominance_ceiling,
            )
        else:
            projector_scale, projector_dominance = 0.0, 0.0
        model.semantic_projector_scale = float(projector_scale)
        model.semantic_projector_dominance = float(projector_dominance)
        if epoch == reliability_warmup_epochs + 1 or epoch % 10 == 0:
            print(
                "Adaptive semantic projector: epoch=%d dominance=%.6f scale=%.6f"
                % (epoch, projector_dominance, projector_scale)
            )
    if bool(adaptive_shared_private) and epoch == 1:
        print(
            "Latched adaptive shared-private residual: dominance=%.6f scale=%.6f"
            % (
                float(getattr(model, "shared_private_dominance", 0.0)),
                float(model.shared_private_scale),
            )
        )
    if bool(adaptive_shared_private) and reliability_active:
        candidate_scale, candidate_dominance = adaptive_semantic_projector_scale(
            model.global_view_quality,
            maximum_scale=shared_private_max_scale,
            dominance_floor=shared_private_dominance_floor,
            dominance_ceiling=shared_private_dominance_ceiling,
        )
        # Capacity is a one-way curriculum decision.  Once reliable evidence
        # exposes cross-view heterogeneity, retain the maximum observed scale;
        # never contract the representation because the new branch succeeds
        # in reducing that very heterogeneity.
        model.shared_private_scale = max(
            float(model.shared_private_scale), float(candidate_scale)
        )
        model.shared_private_dominance = max(
            float(getattr(model, "shared_private_dominance", 0.0)),
            float(candidate_dominance),
        )
        if epoch == reliability_warmup_epochs + 1 or epoch % 10 == 0:
            print(
                "Latched adaptive shared-private residual: epoch=%d "
                "dominance=%.6f scale=%.6f"
                % (
                    epoch,
                    float(model.shared_private_dominance),
                    float(model.shared_private_scale),
                )
            )
    graph_routing = str(graph_routing).lower()
    if graph_routing not in {"none", "hard", "soft"}:
        raise ValueError("graph_routing must be one of: none, hard, soft")
    routing_strength = 1.0 if graph_routing != "none" else 0.0
    routing_dominance = 0.0
    if reliability_active and (
        bool(adaptive_reliability_routing)
        or bool(adaptive_assignment_confidence)
    ):
        routing_strength, routing_dominance = adaptive_semantic_projector_scale(
            model.global_view_quality,
            maximum_scale=1.0,
            dominance_floor=reliability_routing_dominance_floor,
            dominance_ceiling=reliability_routing_dominance_ceiling,
        )
    if bool(adaptive_reliability_routing):
        graph_routing = "soft"
    effective_assignment_confidence_power = float(
        assignment_confidence_power
    )
    if bool(adaptive_assignment_confidence):
        effective_assignment_confidence_power = (
            float(assignment_confidence_max_power) * routing_strength
        )
    model.reliability_routing_strength = float(routing_strength)
    model.reliability_routing_dominance = float(routing_dominance)
    model.effective_assignment_confidence_power = float(
        effective_assignment_confidence_power
    )
    if reliability_active and (
        epoch == reliability_warmup_epochs + 1 or epoch % 10 == 0
    ) and (
        bool(adaptive_reliability_routing)
        or bool(adaptive_assignment_confidence)
    ):
        print(
            "Adaptive reliable routing: epoch=%d dominance=%.6f "
            "routing_strength=%.6f confidence_power=%.6f"
            % (
                epoch,
                routing_dominance,
                routing_strength,
                effective_assignment_confidence_power,
            )
        )
    routed_graph_view = (
        int(model.sample_reliability.mean(dim=0).argmax().item())
        if reliability_active
        and graph_routing != "none"
        and routing_strength > 0.0
        else None
    )
    ramp = _ramp_value(epoch, trust_warmup_epochs, ramp_epochs=ramp_epochs)
    if epoch == trust_warmup_epochs + 1 and float(assign_loss_weight) > 0.0:
        print("Confidence-adaptive view assignment enabled.")
    full_reliable_graph = None
    if use_bgcr_v38 and use_trust and float(reliable_graph_weight) > 0:
        if (not hasattr(model, 'bgcr_reliable_graph')) or model.bgcr_reliable_graph is None:
            print("BGCR-v38 high-order reliable graph enabled.")
            model.bgcr_reliable_graph = build_high_order_reliable_graph(
                W, steps=reliable_graph_steps, keep_topk=reliable_graph_k, mix=0.55
            ).detach()
        full_reliable_graph = model.bgcr_reliable_graph

    for _, (sub_data_views, _, sample_idx) in enumerate(mv_data_loader):
        batch_psedo_label = model.psedo_labels[sample_idx]
        y_matrix = (batch_psedo_label.view(-1, 1) == batch_psedo_label.view(1, -1)).int()
        batch_trust = model.trust_scores[sample_idx] if use_trust else None
        batch_soft_targets = model.soft_targets[sample_idx]
        prototype_centers = model.prototype_centers.detach()
        batch_reliability = None
        if reliability_active:
            batch_reliability = model.effective_reliability(
                model.sample_reliability[sample_idx]
            )

        xrs, zs = model(sub_data_views)
        multilevel_projections = None
        multilevel_logits = None
        canonical_projections = None
        if model.enable_multilevel_heads:
            multilevel_projections, multilevel_logits = (
                model.multilevel_outputs(zs)
            )
        if model.enable_canonical_correlation_head:
            canonical_projections = model.canonical_correlation_outputs(zs)
        common_z = model.fusion(
            zs,
            batch_reliability,
            reliability_power=(1.0 if batch_reliability is not None else None),
        )
        q_centers = model.compute_centers(common_z, batch_psedo_label)

        loss_list = []
        for i in range(num_views):
            view_scale = (
                float(num_views) * batch_reliability[:, i].mean()
                if batch_reliability is not None
                else torch.tensor(1.0, device=common_z.device)
            )
            w = None
            if float(feature_alignment_scale) > 0.0:
                own_w = batch_relation_graph(W[i], sample_idx)
                if routed_graph_view is not None:
                    reliable_w = batch_relation_graph(
                        W[routed_graph_view], sample_idx
                    )
                    if graph_routing == "hard":
                        keep_own = (
                            batch_reliability[:, i] >= (1.0 / float(num_views))
                        ).view(-1, 1)
                        w = torch.where(keep_own, own_w, reliable_w)
                        w = torch.maximum(w, w.t())
                    else:
                        w = route_graph_by_sample_reliability(
                            own_w,
                            reliable_w,
                            batch_reliability[:, i],
                            num_views,
                            routing_strength=routing_strength,
                        )
                else:
                    w = own_w
            k_centers = model.compute_centers(zs[i], batch_psedo_label)
            loss_list.append(criterion(sub_data_views[i], xrs[i]))
            loss_list.append(view_scale * alpha * mvc_loss.compute_cluster_loss(q_centers, k_centers, batch_psedo_label))
            view_trust = batch_trust
            if batch_trust is not None and batch_reliability is not None:
                view_trust = batch_trust * (
                    0.10 + 0.90 * (batch_reliability[:, i] * float(num_views)).clamp(max=1.0)
                )
            if float(feature_alignment_scale) > 0.0:
                loss_list.append(view_scale * beta * float(feature_alignment_scale) * mvc_loss.feature_loss(
                    zs[i], common_z, w, y_matrix,
                    trust_scores=view_trust,
                    trust_pair_lambda=trust_pair_lambda,
                    reliability_graph=(full_reliable_graph[sample_idx][:, sample_idx] if full_reliable_graph is not None else None),
                    graph_lambda=(ramp * reliable_graph_weight if use_bgcr_v38 else 0.0),
                    normalized_multi_positive=normalized_multi_positive,
                ))
            if enable_global_memory and global_memory_query == "views":
                memory_gate = 0.25 + 0.75 * model.trust_scores[sample_idx]
                if batch_reliability is not None:
                    memory_gate = memory_gate * (
                        0.10
                        + 0.90
                        * (batch_reliability[:, i] * float(num_views)).clamp(max=1.0)
                    )
                memory_ramp = _ramp_value(
                    epoch,
                    (
                        trust_warmup_epochs
                        if global_memory_warmup_epochs is None
                        else global_memory_warmup_epochs
                    ),
                    ramp_epochs=global_memory_ramp_epochs,
                )
                loss_list.append(
                    memory_ramp
                    * float(global_topology_weight)
                    * sparse_global_topology_loss(
                        zs[i],
                        sample_idx,
                        model.global_feature_bank,
                        model.memory_positive_indices,
                        model.memory_positive_weights,
                        model.memory_negative_indices,
                        model.memory_negative_weights,
                        temperature=global_memory_temperature,
                        query_weights=memory_gate,
                        self_positive_weight=global_memory_self_weight,
                        objective=global_relation_loss_objective,
                        negative_feature_bank=(
                            model.global_view_feature_banks[i]
                            if global_relation_mode == "multiview_consensus"
                            else None
                        ),
                    )
                )

            if float(assign_loss_weight) > 0.0:
                view_logits = torch.mm(
                    F.normalize(zs[i], dim=1),
                    F.normalize(prototype_centers, dim=1).t(),
                ) / max(float(assignment_temperature), 1e-6)
                loss_list.append(
                    view_scale
                    * ramp
                    * assign_loss_weight
                    * mvc_loss.assignment_alignment_loss(
                        view_logits,
                        batch_soft_targets,
                        trust_scores=view_trust,
                        confidence_power=effective_assignment_confidence_power,
                        confidence_blend=(
                            routing_strength
                            if bool(adaptive_assignment_confidence)
                            else 1.0
                        ),
                    )
                )

        if enable_global_memory and global_memory_query == "consensus":
            memory_ramp = _ramp_value(
                epoch,
                (
                    trust_warmup_epochs
                    if global_memory_warmup_epochs is None
                    else global_memory_warmup_epochs
                ),
                ramp_epochs=global_memory_ramp_epochs,
            )
            loss_list.append(
                memory_ramp
                * float(global_topology_weight)
                * sparse_global_topology_loss(
                    common_z,
                    sample_idx,
                    model.global_feature_bank,
                    model.memory_positive_indices,
                    model.memory_positive_weights,
                    model.memory_negative_indices,
                    model.memory_negative_weights,
                    temperature=global_memory_temperature,
                    query_weights=0.25
                    + 0.75 * model.trust_scores[sample_idx],
                    self_positive_weight=global_memory_self_weight,
                    objective=global_relation_loss_objective,
                )
            )

        if float(relation_barycenter_weight) > 0.0:
            relation_ramp = _ramp_value(
                epoch,
                trust_warmup_epochs,
                ramp_epochs=ramp_epochs,
            )
            loss_list.append(
                relation_ramp
                * float(relation_barycenter_weight)
                * mvc_loss.reliability_relation_barycenter_loss(
                    zs,
                    common_z,
                    reliability=batch_reliability,
                    trust_scores=batch_trust,
                    temperature=relation_barycenter_temperature,
                )
            )

        if float(cross_view_correlation_weight) > 0.0:
            loss_list.append(
                ramp
                * float(cross_view_correlation_weight)
                * mvc_loss.leave_one_out_global_correlation_loss(
                    zs,
                    reliability=batch_reliability,
                    redundancy_weight=cross_view_redundancy_weight,
                )
            )

        if float(canonical_correlation_weight) > 0.0:
            if canonical_projections is None:
                raise RuntimeError(
                    "canonical_correlation_weight requires enabled CCA heads"
                )
            loss_list.append(
                ramp
                * float(canonical_correlation_weight)
                * mvc_loss.pairwise_canonical_max_correlation_loss(
                    canonical_projections,
                    reliability=batch_reliability,
                    regularization=canonical_correlation_regularization,
                )
            )

        if float(instance_contrast_weight) > 0.0:
            loss_list.append(
                ramp
                * float(instance_contrast_weight)
                * mvc_loss.leave_one_out_instance_contrastive_loss(
                    zs,
                    reliability=batch_reliability,
                    soft_targets=(batch_soft_targets if use_trust else None),
                    trust_scores=batch_trust,
                    temperature=instance_contrast_temperature,
                    false_negative_floor=instance_false_negative_floor,
                )
            )

        if float(cross_view_reconstruction_weight) > 0.0:
            loss_list.append(
                float(cross_view_reconstruction_weight)
                * leave_one_out_cross_view_reconstruction_loss(
                    model,
                    sub_data_views,
                    zs,
                    reliability=batch_reliability,
                )
            )

        if float(shared_private_orthogonality_weight) > 0.0:
            orthogonality_loss = (
                mvc_loss.shared_private_sample_cosine_loss(
                    zs,
                    model.last_private_features,
                )
                if getattr(model, "context_moe", None) is not None
                else mvc_loss.shared_private_orthogonality_loss(
                    zs,
                    model.last_private_features,
                )
            )
            loss_list.append(
                ramp
                * float(shared_private_orthogonality_weight)
                * orthogonality_loss
            )

        if float(context_moe_alignment_weight) > 0.0:
            if getattr(model, "context_moe", None) is None:
                raise RuntimeError(
                    "context_moe_alignment_weight requires context-aware MoE"
                )
            loss_list.append(
                float(context_moe_alignment_weight)
                * mvc_loss.multilevel_instance_contrastive_loss(
                    zs,
                    temperature=context_moe_alignment_temperature,
                )
            )

        if float(multilevel_instance_weight) > 0.0:
            if multilevel_projections is None:
                raise RuntimeError(
                    "multilevel_instance_weight requires enabled heads"
                )
            loss_list.append(
                float(multilevel_instance_weight)
                * mvc_loss.multilevel_instance_contrastive_loss(
                    multilevel_projections,
                    temperature=multilevel_instance_temperature,
                )
            )

        if float(multilevel_cluster_weight) > 0.0:
            if multilevel_logits is None:
                raise RuntimeError(
                    "multilevel_cluster_weight requires enabled heads"
                )
            loss_list.append(
                float(multilevel_cluster_weight)
                * mvc_loss.multilevel_cluster_contrastive_loss(
                    multilevel_logits,
                    temperature=multilevel_cluster_temperature,
                )
            )

        needs_common_logits = any(
            float(weight) > 0.0
            for weight in (
                cluster_contrast_weight,
                proto_loss_weight,
                balance_loss_weight,
                sharpen_loss_weight,
            )
        )
        if needs_common_logits:
            common_logits = torch.mm(
                F.normalize(common_z, dim=1),
                F.normalize(prototype_centers, dim=1).t(),
            ) / max(float(assignment_temperature), 1e-6)
            if use_bgcr_v38 and float(cluster_contrast_weight) > 0.0:
                for i in range(num_views):
                    view_logits_i = torch.mm(
                        F.normalize(zs[i], dim=1),
                        F.normalize(prototype_centers, dim=1).t(),
                    )
                    loss_list.append(
                        ramp
                        * cluster_contrast_weight
                        * mvc_loss.cluster_assignment_contrastive_loss(
                            view_logits_i,
                            common_logits,
                            temperature=cluster_contrast_temperature,
                        )
                    )
            if float(proto_loss_weight) > 0.0:
                loss_list.append(
                    ramp
                    * proto_loss_weight
                    * mvc_loss.prototype_consistency_loss(
                        common_logits,
                        batch_soft_targets,
                        trust_scores=batch_trust,
                    )
                )
            if float(balance_loss_weight) > 0.0:
                loss_list.append(
                    ramp
                    * balance_loss_weight
                    * mvc_loss.cluster_balance_loss(common_logits)
                )
            if float(sharpen_loss_weight) > 0.0:
                loss_list.append(
                    ramp
                    * sharpen_loss_weight
                    * mvc_loss.target_sharpen_loss(
                        common_logits,
                        trust_scores=batch_trust,
                    )
                )

        loss = sum(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print('Contrastive_train, epoch {} loss:{:.7f} ramp:{:.3f}'.format(epoch, total_loss, ramp))
