"""Sparse global topology memory for mini-batch multi-view clustering.

The original large-dataset route constructs a KNN graph inside every shuffled
mini-batch.  Such a graph is not an approximation of the dataset graph: two
true neighbours can only be connected when they happen to occur in the same
batch.  This module keeps stop-gradient consensus features for *all* samples,
builds a sparse teacher topology periodically, and lets a mini-batch query
that topology in O(B (K+M) D) work.

The functions are intentionally independent of ``model.py`` and ``loss.py``
so they can be unit-tested and integrated without changing the baseline.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def ema_normalized_bank(
    previous: torch.Tensor,
    current: torch.Tensor,
    momentum: float = 0.90,
    initialized: bool = True,
) -> torch.Tensor:
    """Return an L2-normalized stop-gradient EMA feature bank.

    Args:
        previous: Existing bank with shape ``[N, D]``.
        current: Current full-dataset consensus features, shape ``[N, D]``.
        momentum: Teacher momentum.  ``0.90`` is the recommended default.
        initialized: If false, initialize directly from ``current``.
    """
    if previous.shape != current.shape:
        raise ValueError("previous and current feature banks must have equal shape")
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("momentum must lie in [0, 1)")
    current = F.normalize(current.detach().float(), dim=1)
    if not initialized:
        return current
    previous = F.normalize(previous.detach().float(), dim=1)
    return F.normalize(
        float(momentum) * previous + (1.0 - float(momentum)) * current,
        dim=1,
    )


def _validate_teacher_inputs(
    features: torch.Tensor,
    targets: torch.Tensor,
    trust: torch.Tensor,
) -> Tuple[int, int, int]:
    if features.ndim != 2 or targets.ndim != 2 or trust.ndim != 1:
        raise ValueError("expected features [N,D], targets [N,C], trust [N]")
    sample_count, feature_dim = features.shape
    if targets.size(0) != sample_count or trust.numel() != sample_count:
        raise ValueError("teacher inputs must share the same sample dimension")
    if sample_count < 2 or targets.size(1) < 2:
        raise ValueError("global topology requires at least 2 samples and 2 clusters")
    return sample_count, feature_dim, targets.size(1)


@torch.no_grad()
def build_sparse_global_topology(
    features: torch.Tensor,
    targets: torch.Tensor,
    trust: torch.Tensor,
    positive_k: int = 9,
    negative_k: int = 64,
    candidate_clusters: int = 2,
    chunk_size: int = 512,
    seed: int = 0,
    auxiliary_features: Optional[torch.Tensor] = None,
    auxiliary_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Build a cluster-routed sparse graph from the global teacher bank.

    Positive candidates are drawn from the teacher's top-r semantic clusters
    (r=2 by default).  Candidates are ranked by

        a_ij = relu(cos(m_i,m_j)) * <q_i,q_j> * sqrt(t_i t_j),

    where ``m`` is the EMA consensus feature, ``q`` the soft cluster target and
    ``t`` sample trust.  Top-2 routing permits a boundary sample to connect
    across a currently hard pseudo-label, avoiding a purely self-confirming
    within-cluster graph.

    Negatives are sampled globally from semantically dissimilar *semi-hard*
    clusters: among nonempty clusters other than the hard assignment, the
    router favours high feature similarity while penalising target
    probability.  Their mass is

        b_ij = (1 - <q_i,q_j>) * sqrt(t_i t_j).

    All returned tensors are sparse row lists rather than an ``N x N`` matrix.
    Computation is exact within routed candidate pools and chunked to cap peak
    memory.  With balanced clusters it evaluates approximately
    ``r*N^2/C`` feature pairs instead of ``N^2``.
    """
    sample_count, _, cluster_count = _validate_teacher_inputs(
        features, targets, trust
    )
    positive_k = min(max(1, int(positive_k)), sample_count - 1)
    negative_k = max(1, int(negative_k))
    candidate_clusters = min(max(1, int(candidate_clusters)), cluster_count)
    chunk_size = max(1, int(chunk_size))

    device = features.device
    dtype = features.dtype
    features = F.normalize(features.detach().float(), dim=1)
    auxiliary_weight = float(auxiliary_weight)
    if not 0.0 <= auxiliary_weight <= 1.0:
        raise ValueError("auxiliary_weight must lie in [0, 1]")
    if auxiliary_features is not None:
        if auxiliary_features.shape != features.shape:
            raise ValueError("auxiliary_features must have shape [N,D]")
        auxiliary_features = F.normalize(
            auxiliary_features.detach().to(features).float(), dim=1
        )
    else:
        auxiliary_weight = 0.0
    targets = targets.detach().float().clamp_min(0.0)
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
    trust = trust.detach().float().clamp(0.0, 1.0)

    hard_cluster = targets.argmax(dim=1)
    routed_clusters = targets.topk(candidate_clusters, dim=1).indices
    best_scores = torch.full(
        (sample_count, positive_k), -torch.inf, device=device
    )
    best_indices = torch.zeros(
        (sample_count, positive_k), dtype=torch.long, device=device
    )

    # Each query appears in at most ``candidate_clusters`` loops.  The dense
    # temporary is only [chunk_size, size_of_one_cluster].
    for cluster in range(cluster_count):
        candidate_index = torch.nonzero(
            hard_cluster.eq(cluster), as_tuple=False
        ).flatten()
        query_index = torch.nonzero(
            routed_clusters.eq(cluster).any(dim=1), as_tuple=False
        ).flatten()
        if candidate_index.numel() == 0 or query_index.numel() == 0:
            continue
        candidate_feature = features[candidate_index]
        candidate_auxiliary = (
            auxiliary_features[candidate_index]
            if auxiliary_features is not None
            else None
        )
        candidate_target = targets[candidate_index]
        candidate_trust = trust[candidate_index]

        for start in range(0, query_index.numel(), chunk_size):
            row_index = query_index[start : start + chunk_size]
            similarity = torch.mm(features[row_index], candidate_feature.t())
            if candidate_auxiliary is not None and auxiliary_weight > 0.0:
                raw_similarity = torch.mm(
                    auxiliary_features[row_index], candidate_auxiliary.t()
                )
                similarity = (
                    (1.0 - auxiliary_weight) * similarity
                    + auxiliary_weight * raw_similarity
                )
            semantic = torch.mm(targets[row_index], candidate_target.t())
            pair_trust = torch.sqrt(
                trust[row_index, None] * candidate_trust[None, :]
            )
            score = similarity.clamp_min(0.0) * semantic * pair_trust
            self_mask = row_index[:, None].eq(candidate_index[None, :])
            score.masked_fill_(self_mask, -torch.inf)

            local_k = min(positive_k, candidate_index.numel())
            local_score, local_column = score.topk(local_k, dim=1)
            local_index = candidate_index[local_column]
            if local_k < positive_k:
                padding = positive_k - local_k
                local_score = torch.cat(
                    [
                        local_score,
                        torch.full(
                            (row_index.numel(), padding),
                            -torch.inf,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                local_index = torch.cat(
                    [
                        local_index,
                        torch.zeros(
                            (row_index.numel(), padding),
                            dtype=torch.long,
                            device=device,
                        ),
                    ],
                    dim=1,
                )

            merged_score = torch.cat([best_scores[row_index], local_score], dim=1)
            merged_index = torch.cat([best_indices[row_index], local_index], dim=1)
            new_score, order = merged_score.topk(positive_k, dim=1)
            best_scores[row_index] = new_score
            best_indices[row_index] = merged_index.gather(1, order)

    # Degenerate empty/singleton clusters are handled without inventing a
    # self-edge: use a deterministic cyclic non-self sample with tiny mass.
    invalid = ~torch.isfinite(best_scores)
    fully_degenerate = invalid.all(dim=1)
    if invalid.any():
        row = torch.arange(sample_count, device=device)[:, None]
        fallback = (row + 1 + torch.arange(positive_k, device=device)[None, :])
        fallback = fallback.remainder(sample_count)
        best_indices = torch.where(invalid, fallback, best_indices)
        best_scores = torch.where(invalid, torch.full_like(best_scores, 1e-8), best_scores)

    positive_weights = best_scores.clamp_min(1e-8)
    positive_weights = positive_weights / positive_weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    # If an entire row had no valid non-self semantic neighbour, retain the
    # cyclic edge only as a genuinely tiny numerical fallback.  Normalising a
    # row of 1e-8 values to unit mass would otherwise turn the fallback into a
    # full-strength invented positive relation.
    positive_weights[fully_degenerate] *= 1e-6

    # Vectorized, deterministic semi-hard negative routing.  The previous
    # implementation selected the smallest target probabilities.  With sparse
    # top-2 targets that created a massive zero-probability tie and repeatedly
    # routed every K=100 query to the same few low-index clusters.  Cluster
    # prototypes provide a meaningful ordering over all semantically unlikely
    # clusters; a tiny cyclic jitter only resolves exact numerical ties.
    member_lists = [
        torch.nonzero(hard_cluster.eq(cluster), as_tuple=False).flatten()
        for cluster in range(cluster_count)
    ]
    nonempty = torch.as_tensor(
        [members.numel() > 0 for members in member_lists],
        dtype=torch.bool,
        device=device,
    )
    if int(nonempty.sum().item()) >= 2:
        negative_cluster_count = min(4, int(nonempty.sum().item()) - 1)
        cluster_sum = torch.zeros(
            (cluster_count, features.size(1)),
            dtype=features.dtype,
            device=device,
        )
        cluster_sum.index_add_(0, hard_cluster, features)
        cluster_prototype = F.normalize(cluster_sum, dim=1)
        route_score = torch.mm(features, cluster_prototype.t())
        route_score = route_score + (1.0 - targets)
        route_score[:, ~nonempty] = -torch.inf
        route_score.scatter_(1, hard_cluster[:, None], -torch.inf)
        row_id = torch.arange(sample_count, device=device)[:, None]
        cluster_id = torch.arange(cluster_count, device=device)[None, :]
        cyclic_rank = (
            cluster_id - (row_id + int(seed)).remainder(cluster_count)
        ).remainder(cluster_count)
        route_score = route_score - cyclic_rank.to(route_score) * 1e-7
        negative_clusters = route_score.topk(
            negative_cluster_count, dim=1, largest=True
        ).indices
    else:
        # A completely collapsed teacher has no semantically safe negative
        # cluster.  Keep a global fallback so the operation remains finite;
        # its low semantic mass makes collapse visible rather than silently
        # treating arbitrary same-cluster samples as certain negatives.
        negative_cluster_count = 1
        negative_clusters = hard_cluster[:, None]
    # An empty predicted cluster falls back to the full sample index below.
    # Reserve N slots in that case; sizing only from non-empty clusters can
    # overflow when a transiently empty cluster is selected as a fallback.
    has_empty_cluster = any(members.numel() == 0 for members in member_lists)
    largest_cluster = (
        sample_count
        if has_empty_cluster
        else max(max(1, members.numel()) for members in member_lists)
    )
    member_table = torch.zeros(
        (cluster_count, largest_cluster), dtype=torch.long, device=device
    )
    member_count = torch.ones(cluster_count, dtype=torch.long, device=device)
    all_index = torch.arange(sample_count, device=device)
    for cluster, members in enumerate(member_lists):
        if members.numel() == 0:
            # Empty predicted clusters cannot supply a negative.  A complete
            # fallback table keeps indexing valid; semantic weights will gate
            # uncertain pairs below.
            members = all_index
        member_table[cluster, : members.numel()] = members
        member_count[cluster] = members.numel()

    row = torch.arange(sample_count, device=device, dtype=torch.long)[:, None]
    slot = torch.arange(negative_k, device=device, dtype=torch.long)[None, :]
    selected_cluster = negative_clusters[
        row.expand(-1, negative_k),
        slot.remainder(negative_cluster_count).expand(sample_count, -1),
    ]
    counts = member_count[selected_cluster]
    # Integer constants are small enough to be exact in int64 for practical N.
    offset = (
        row * 1103515245 + slot * 12345 + int(seed) * 2654435761
    ).remainder(counts)
    negative_indices = member_table[selected_cluster, offset]

    negative_semantic = (
        targets[:, None, :]
        * targets[negative_indices]
    ).sum(dim=2)
    negative_pair_trust = torch.sqrt(
        trust[:, None] * trust[negative_indices]
    )
    negative_weights = (
        (1.0 - negative_semantic).clamp_min(0.0) * negative_pair_trust
    ).clamp_min(1e-8)

    return {
        "positive_indices": best_indices,
        "positive_weights": positive_weights.to(dtype),
        "negative_indices": negative_indices,
        "negative_weights": negative_weights.to(dtype),
        "hard_cluster": hard_cluster,
    }


def sparse_global_topology_loss(
    query_features: torch.Tensor,
    sample_indices: torch.Tensor,
    feature_bank: torch.Tensor,
    positive_indices: torch.Tensor,
    positive_weights: torch.Tensor,
    negative_indices: torch.Tensor,
    negative_weights: torch.Tensor,
    temperature: float = 0.20,
    query_weights: Optional[torch.Tensor] = None,
    self_positive_weight: float = 0.0,
    objective: str = "infonce",
    negative_feature_bank: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-positive InfoNCE against stop-gradient global memory keys.

    Shapes:
        ``query_features``: ``[B,D]``;
        ``sample_indices``: ``[B]`` global dataset indices;
        ``feature_bank``: ``[N,D]``;
        positive rows: ``[N,K]``; negative rows: ``[N,M]``.

    The loss for sample i is

        -log (sum_j a_ij exp(z_i^T m_j/tau))
             / (sum_j a_ij exp(.) + sum_l b_il exp(z_i^T m_l/tau)).

    Memory keys, topology indices and edge masses are detached teacher state;
    gradients therefore flow only through the current mini-batch query.
    """
    if query_features.ndim != 2 or sample_indices.ndim != 1:
        raise ValueError("expected query_features [B,D] and sample_indices [B]")
    if query_features.size(0) != sample_indices.numel():
        raise ValueError("batch dimensions do not match")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    objective = str(objective).lower()
    if objective not in {"infonce", "relation_energy"}:
        raise ValueError("objective must be infonce or relation_energy")

    sample_indices = sample_indices.to(
        device=query_features.device, dtype=torch.long
    )
    bank = F.normalize(
        feature_bank.detach().to(query_features).float(), dim=1
    )
    query = F.normalize(query_features.float(), dim=1)
    pos_index = positive_indices[sample_indices].detach()
    neg_index = negative_indices[sample_indices].detach()
    pos_key = bank[pos_index]
    neg_key = bank[neg_index]

    pos_logit = torch.einsum("bd,bkd->bk", query, pos_key) / float(temperature)
    neg_logit = torch.einsum("bd,bmd->bm", query, neg_key) / float(temperature)
    extra_neg_logit = None
    if negative_feature_bank is not None:
        negative_bank = F.normalize(
            negative_feature_bank.detach().to(query_features).float(), dim=1
        )
        if negative_bank.shape != bank.shape:
            raise ValueError("negative_feature_bank must match feature_bank")
        extra_neg_key = negative_bank[neg_index]
        extra_neg_logit = torch.einsum(
            "bd,bmd->bm", query, extra_neg_key
        ) / float(temperature)
    pos_mass = positive_weights[sample_indices].detach().to(pos_logit).clamp_min(1e-8)
    neg_mass = negative_weights[sample_indices].detach().to(neg_logit).clamp_min(1e-8)
    if float(self_positive_weight) > 0.0:
        self_key = bank[sample_indices]
        self_logit = (query * self_key).sum(dim=1, keepdim=True) / float(
            temperature
        )
        self_mass = torch.full_like(
            self_logit, float(self_positive_weight)
        ).clamp_min(1e-8)
        pos_logit = torch.cat([self_logit, pos_logit], dim=1)
        pos_mass = torch.cat([self_mass, pos_mass], dim=1)
    pos_term = pos_logit + pos_mass.log()
    neg_term = neg_logit + neg_mass.log()
    if extra_neg_logit is not None:
        neg_term = torch.cat(
            [neg_term, extra_neg_logit + neg_mass.log()], dim=1
        )
    if objective == "relation_energy":
        # Use the relation-energy objective, now
        # evaluated against a global sparse memory instead of only those
        # neighbours that happen to enter the current mini-batch.
        positive_energy = -(
            pos_mass * pos_logit
        ).sum(dim=1) / pos_mass.sum(dim=1).clamp_min(1e-8)
        negative_energy = torch.logsumexp(neg_term, dim=1)
        per_sample = positive_energy + negative_energy
    else:
        log_numerator = torch.logsumexp(pos_term, dim=1)
        log_denominator = torch.logsumexp(
            torch.cat([pos_term, neg_term], dim=1), dim=1
        )
        per_sample = log_denominator - log_numerator

    if query_weights is None:
        return per_sample.mean()
    query_weights = query_weights.to(per_sample).detach().clamp_min(0.0)
    if query_weights.shape != per_sample.shape:
        raise ValueError("query_weights must have shape [B]")
    return (per_sample * query_weights).sum() / query_weights.sum().clamp_min(1e-8)




class SparseGlobalTopologyMemory(nn.Module):
    """Checkpointable state wrapper around the pure topology functions.

    The module has no trainable parameters.  Attaching it as
    ``model.global_topology = SparseGlobalTopologyMemory(...)`` registers all
    teacher state in ``state_dict`` and moves it with ``model.to(device)``.
    """

    def __init__(
        self,
        num_samples: int,
        feature_dim: int,
        num_clusters: int,
        positive_k: int = 9,
        negative_k: int = 64,
        momentum: float = 0.90,
        refresh_interval: int = 5,
        candidate_clusters: int = 2,
        chunk_size: int = 512,
    ):
        super().__init__()
        if num_samples <= positive_k:
            raise ValueError("num_samples must be larger than positive_k")
        self.num_samples = int(num_samples)
        self.feature_dim = int(feature_dim)
        self.num_clusters = int(num_clusters)
        self.positive_k = int(positive_k)
        self.negative_k = int(negative_k)
        self.momentum = float(momentum)
        self.refresh_interval = int(refresh_interval)
        self.candidate_clusters = int(candidate_clusters)
        self.chunk_size = int(chunk_size)

        self.register_buffer("feature_bank", torch.zeros(num_samples, feature_dim))
        self.register_buffer("target_bank", torch.zeros(num_samples, num_clusters))
        self.register_buffer("trust_bank", torch.zeros(num_samples))
        self.register_buffer(
            "positive_indices", torch.zeros(num_samples, positive_k, dtype=torch.long)
        )
        self.register_buffer("positive_weights", torch.zeros(num_samples, positive_k))
        self.register_buffer(
            "negative_indices", torch.zeros(num_samples, negative_k, dtype=torch.long)
        )
        self.register_buffer("negative_weights", torch.zeros(num_samples, negative_k))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_refresh_epoch", torch.full((), -1, dtype=torch.long))

    @torch.no_grad()
    def update_teacher(
        self,
        current_features: torch.Tensor,
        current_targets: torch.Tensor,
        current_trust: torch.Tensor,
        epoch: int,
        force_graph_refresh: bool = False,
        seed: int = 0,
    ) -> bool:
        """Update the EMA bank and refresh sparse edges when due.

        Returns true iff neighbour indices were rebuilt.  The bank and current
        semantic confidence are updated every call; expensive neighbour search
        occurs on the first call and then every ``refresh_interval`` epochs.
        """
        if current_features.shape != self.feature_bank.shape:
            raise ValueError("current_features has the wrong [N,D] shape")
        if current_targets.shape != self.target_bank.shape:
            raise ValueError("current_targets has the wrong [N,C] shape")
        if current_trust.shape != self.trust_bank.shape:
            raise ValueError("current_trust has the wrong [N] shape")
        initialized = int(self.num_updates.item()) > 0
        self.feature_bank.copy_(
            ema_normalized_bank(
                self.feature_bank,
                current_features,
                momentum=self.momentum,
                initialized=initialized,
            )
        )
        normalized_targets = current_targets.detach().to(self.target_bank).clamp_min(0.0)
        normalized_targets = normalized_targets / normalized_targets.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        self.target_bank.copy_(normalized_targets)
        self.trust_bank.copy_(current_trust.detach().to(self.trust_bank).clamp(0.0, 1.0))

        last = int(self.last_refresh_epoch.item())
        due = (
            bool(force_graph_refresh)
            or not initialized
            or last < 0
            or int(epoch) - last >= self.refresh_interval
        )
        if due:
            topology = build_sparse_global_topology(
                self.feature_bank,
                self.target_bank,
                self.trust_bank,
                positive_k=self.positive_k,
                negative_k=self.negative_k,
                candidate_clusters=self.candidate_clusters,
                chunk_size=self.chunk_size,
                seed=seed,
            )
            self.positive_indices.copy_(topology["positive_indices"])
            self.positive_weights.copy_(topology["positive_weights"])
            self.negative_indices.copy_(topology["negative_indices"])
            self.negative_weights.copy_(topology["negative_weights"])
            self.last_refresh_epoch.fill_(int(epoch))
        self.num_updates.add_(1)
        return due

    def forward(
        self,
        query_features: torch.Tensor,
        sample_indices: torch.Tensor,
        temperature: float = 0.20,
        query_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if int(self.num_updates.item()) == 0:
            raise RuntimeError("global topology memory must be updated before use")
        return sparse_global_topology_loss(
            query_features,
            sample_indices,
            self.feature_bank,
            self.positive_indices,
            self.positive_weights,
            self.negative_indices,
            self.negative_weights,
            temperature=temperature,
            query_weights=query_weights,
        )
