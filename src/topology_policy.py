# -*- coding: utf-8 -*-
"""Dataset-name-free policy for exact versus scalable topology training."""

from __future__ import annotations


def estimated_exact_graph_megabytes(
    sample_count,
    view_count,
    *,
    element_bytes=4,
    safety_factor=1.25,
):
    """Estimate persistent full-view adjacency storage including headroom."""
    samples = int(sample_count)
    views = int(view_count)
    if samples <= 0 or views <= 0:
        raise ValueError("sample_count and view_count must be positive")
    if int(element_bytes) <= 0 or float(safety_factor) < 1.0:
        raise ValueError("invalid exact-graph storage assumptions")
    byte_count = (
        samples
        * samples
        * views
        * int(element_bytes)
        * float(safety_factor)
    )
    return float(byte_count / (1024.0 * 1024.0))


def topology_route(sample_count, view_count, exact_graph_budget_mb):
    """Return a deterministic route based only on estimated memory cost."""
    budget = float(exact_graph_budget_mb)
    if budget <= 0.0:
        raise ValueError("exact_graph_budget_mb must be positive")
    estimate = estimated_exact_graph_megabytes(sample_count, view_count)
    scalable = estimate > budget
    return {
        "route": "scalable_global_memory" if scalable else "exact_graph",
        "scalable": scalable,
        "estimated_exact_graph_mb": estimate,
        "exact_graph_budget_mb": budget,
    }
