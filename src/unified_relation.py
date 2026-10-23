"""Unified multi-view relation consensus for every dataset scale.

The old implementation optimized an in-batch raw KNN loss on compact data and
a memory-bank semantic loss on larger data.  This module removes that
mathematical branch.  Every dataset now builds the same sparse relation target:

* candidate positives are proposed by the consensus and every learned view;
* one edge score combines consensus similarity, reliability-weighted local
  view similarity, soft-cluster overlap, and pair trust;
* negatives and the final multi-positive InfoNCE use the same global memory.

Only the chunk size used while searching candidates may change with scale.
That is a computational backend choice, not a different learning objective.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from global_memory import build_sparse_global_topology


@torch.no_grad()
def build_chunked_cosine_knn(
    features: torch.Tensor, k: int, chunk_size: int = 512
) -> torch.Tensor:
    """Return exact non-self cosine KNN indices without an ``N x N`` graph."""
    if features.ndim != 2 or features.size(0) < 2:
        raise ValueError("features must have shape [N,D] with N >= 2")
    sample_count = int(features.size(0))
    k = min(max(1, int(k)), sample_count - 1)
    chunk_size = max(1, int(chunk_size))
    features = F.normalize(features.detach().float(), dim=1)
    output = torch.empty(
        (sample_count, k), dtype=torch.long, device=features.device
    )
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        similarity = torch.mm(features[start:stop], features.t())
        rows = torch.arange(start, stop, device=features.device)
        similarity[torch.arange(stop - start, device=features.device), rows] = -torch.inf
        output[start:stop] = similarity.topk(
            k, dim=1, largest=True, sorted=True
        ).indices
    return output


@torch.no_grad()
def build_unified_multiview_topology(
    consensus_features: torch.Tensor,
    view_features: Sequence[torch.Tensor],
    targets: torch.Tensor,
    trust: torch.Tensor,
    reliability: torch.Tensor,
    *,
    raw_view_features: Sequence[torch.Tensor] | None = None,
    raw_candidate_indices: torch.Tensor | None = None,
    positive_k: int = 9,
    negative_k: int = 64,
    candidate_clusters: int = 2,
    chunk_size: int = 512,
    seed: int = 0,
    consensus_weight: float = 0.50,
    semantic_floor: float = 0.25,
    raw_local_weight: float = 0.50,
) -> Dict[str, torch.Tensor]:
    """Build one sparse relation graph from consensus and all learned views.

    For a candidate pair ``(i,j)``, the positive mass is proportional to

    ``relu(eta*cos(zc_i,zc_j) + (1-eta)*sum_v pi_ijv*cos(zv_i,zv_j))``
    ``* (semantic_floor + (1-semantic_floor)*<q_i,q_j>) * sqrt(t_i*t_j)``,

    where ``pi_ijv`` is the normalized geometric mean of the two endpoints'
    sample-view reliabilities.  Candidate retrieval is cluster-routed and
    chunked for every scale; it never materializes an ``N x N`` graph.
    """
    if consensus_features.ndim != 2:
        raise ValueError("consensus_features must have shape [N,D]")
    if not view_features:
        raise ValueError("at least one learned view is required")
    sample_count, feature_dim = consensus_features.shape
    view_count = len(view_features)
    if any(tuple(view.shape) != (sample_count, feature_dim) for view in view_features):
        raise ValueError("every learned view must match consensus shape [N,D]")
    if tuple(reliability.shape) != (sample_count, view_count):
        raise ValueError("reliability must have shape [N,V]")
    if targets.ndim != 2 or targets.size(0) != sample_count:
        raise ValueError("targets must have shape [N,K]")
    if trust.ndim != 1 or trust.numel() != sample_count:
        raise ValueError("trust must have shape [N]")
    positive_k = min(max(1, int(positive_k)), sample_count - 1)
    chunk_size = max(1, int(chunk_size))
    consensus_weight = float(consensus_weight)
    semantic_floor = float(semantic_floor)
    raw_local_weight = float(raw_local_weight)
    if not 0.0 <= consensus_weight <= 1.0:
        raise ValueError("consensus_weight must lie in [0,1]")
    if not 0.0 <= semantic_floor <= 1.0:
        raise ValueError("semantic_floor must lie in [0,1]")
    if not 0.0 <= raw_local_weight <= 1.0:
        raise ValueError("raw_local_weight must lie in [0,1]")

    device = consensus_features.device
    output_dtype = consensus_features.dtype
    consensus = F.normalize(consensus_features.detach().float(), dim=1)
    views = [F.normalize(view.detach().to(device).float(), dim=1) for view in view_features]
    raw_views = []
    if raw_view_features is not None:
        if len(raw_view_features) != view_count:
            raise ValueError("raw_view_features must contain V tensors")
        if any(tuple(view.shape) != (sample_count, feature_dim) for view in raw_view_features):
            raise ValueError("every raw relation view must match [N,D]")
        raw_views = [
            F.normalize(view.detach().to(device).float(), dim=1)
            for view in raw_view_features
        ]
        if raw_candidate_indices is not None:
            if (
                raw_candidate_indices.ndim != 3
                or raw_candidate_indices.size(0) != view_count
                or raw_candidate_indices.size(1) != sample_count
            ):
                raise ValueError("raw_candidate_indices must have shape [V,N,k]")
            raw_candidate_indices = raw_candidate_indices.detach().to(
                device=device, dtype=torch.long
            )
    else:
        raw_local_weight = 0.0
    targets = targets.detach().to(device).float().clamp_min(0.0)
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
    trust = trust.detach().to(device).float().clamp(0.0, 1.0)
    reliability = reliability.detach().to(device).float().clamp_min(1e-8)
    reliability = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # The consensus proposal also supplies the semantically safe negatives.
    consensus_topology = build_sparse_global_topology(
        consensus,
        targets,
        trust,
        positive_k=positive_k,
        negative_k=int(negative_k),
        candidate_clusters=int(candidate_clusters),
        chunk_size=chunk_size,
        seed=int(seed),
    )
    proposal_indices = [consensus_topology["positive_indices"]]

    # Every view uses the same proposal operator.  Only one negative is built
    # here because final negatives come from the consensus proposal above.
    for view_index, view in enumerate(views):
        view_trust = trust * (
            0.10
            + 0.90
            * (reliability[:, view_index] * float(view_count)).clamp(max=1.0)
        )
        proposal = build_sparse_global_topology(
            view,
            targets,
            view_trust,
            positive_k=positive_k,
            negative_k=1,
            candidate_clusters=int(candidate_clusters),
            chunk_size=chunk_size,
            seed=int(seed) + 104729 * (view_index + 1),
        )
        proposal_indices.append(proposal["positive_indices"])
        if raw_views and raw_candidate_indices is not None:
            proposal_indices.append(raw_candidate_indices[view_index])
        elif raw_views:
            raw_proposal = build_sparse_global_topology(
                raw_views[view_index],
                targets,
                view_trust,
                positive_k=positive_k,
                negative_k=1,
                candidate_clusters=int(candidate_clusters),
                chunk_size=chunk_size,
                seed=int(seed) + 130363 * (view_index + 1),
            )
            proposal_indices.append(raw_proposal["positive_indices"])

    candidates = torch.cat(proposal_indices, dim=1)
    candidate_count = int(candidates.size(1))
    final_indices = torch.zeros(
        (sample_count, positive_k), dtype=torch.long, device=device
    )
    final_scores = torch.zeros(
        (sample_count, positive_k), dtype=torch.float32, device=device
    )

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        rows = torch.arange(start, stop, device=device)
        row_candidates = candidates[rows]
        sorted_candidates, _ = row_candidates.sort(dim=1)

        # The score below depends only on the pair, not on the proposal source.
        # After sorting, retain one copy of each duplicate candidate.
        duplicate = torch.zeros_like(sorted_candidates, dtype=torch.bool)
        duplicate[:, 1:] = sorted_candidates[:, 1:].eq(
            sorted_candidates[:, :-1]
        )
        duplicate |= sorted_candidates.eq(rows[:, None])

        consensus_similarity = torch.einsum(
            "bd,bmd->bm", consensus[rows], consensus[sorted_candidates]
        )
        semantic_overlap = torch.einsum(
            "bk,bmk->bm", targets[rows], targets[sorted_candidates]
        ).clamp(0.0, 1.0)
        pair_trust = torch.sqrt(
            trust[rows, None] * trust[sorted_candidates]
        )

        local_numerator = torch.zeros_like(consensus_similarity)
        raw_numerator = torch.zeros_like(consensus_similarity)
        local_denominator = torch.zeros_like(consensus_similarity)
        for view_index, view in enumerate(views):
            view_similarity = torch.einsum(
                "bd,bmd->bm", view[rows], view[sorted_candidates]
            )
            pair_reliability = torch.sqrt(
                reliability[rows, view_index, None]
                * reliability[sorted_candidates, view_index]
            )
            local_numerator.add_(
                pair_reliability * view_similarity.clamp_min(0.0)
            )
            if raw_views:
                raw_similarity = torch.einsum(
                    "bd,bmd->bm",
                    raw_views[view_index][rows],
                    raw_views[view_index][sorted_candidates],
                )
                raw_numerator.add_(
                    pair_reliability * raw_similarity.clamp_min(0.0)
                )
            local_denominator.add_(pair_reliability)
        local_similarity = local_numerator / local_denominator.clamp_min(1e-8)
        if raw_views:
            raw_similarity = raw_numerator / local_denominator.clamp_min(1e-8)
            local_similarity = (
                (1.0 - raw_local_weight) * local_similarity
                + raw_local_weight * raw_similarity
            )

        relation_similarity = (
            consensus_weight * consensus_similarity.clamp_min(0.0)
            + (1.0 - consensus_weight) * local_similarity
        )
        semantic_mass = semantic_floor + (1.0 - semantic_floor) * semantic_overlap
        score = relation_similarity * semantic_mass * pair_trust
        score.masked_fill_(duplicate, -torch.inf)
        selected_score, selected_column = score.topk(
            positive_k, dim=1, largest=True, sorted=True
        )
        selected_index = sorted_candidates.gather(1, selected_column)

        invalid = ~torch.isfinite(selected_score)
        if invalid.any():
            fallback = (
                rows[:, None]
                + 1
                + torch.arange(positive_k, device=device)[None, :]
            ).remainder(sample_count)
            selected_index = torch.where(invalid, fallback, selected_index)
            selected_score = torch.where(
                invalid, torch.full_like(selected_score, 1e-8), selected_score
            )
        selected_score = selected_score.clamp_min(1e-8)
        selected_score = selected_score / selected_score.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        final_indices[rows] = selected_index
        final_scores[rows] = selected_score

    return {
        "positive_indices": final_indices,
        "positive_weights": final_scores.to(output_dtype),
        "negative_indices": consensus_topology["negative_indices"],
        "negative_weights": consensus_topology["negative_weights"],
        "hard_cluster": consensus_topology["hard_cluster"],
        "candidate_count": torch.tensor(candidate_count, device=device),
    }
