# -*- coding: utf-8 -*-
"""SACR-MVC training with one label-isolated evaluation protocol."""

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from adaptive_topology_selector import (
    compute_backend_structure_diagnostics,
    select_topology_backend,
)
from consensus_guard import consensus_guarded_gate
from dataloader import MultiviewData
from dataset_metadata import canonical_dataset_name, known_cluster_count
from network import Network
from loss import Loss
from metric import evaluate
from model import (
    adaptive_semantic_projector_scale,
    collect_multilevel_predictions,
    contrastive_train,
    estimate_pretrained_view_quality,
    get_W,
    install_exact_multiview_memory_positives,
    multilevel_pseudo_tune,
    pre_train,
)
from pretrain_selection import (
    SELECTOR_VERSION,
    CrossViewPretrainingSelector,
    capture_rng_state,
    move_optimizer_state_to_parameter_devices,
    restore_rng_state,
)
from structural_anchor import build_structural_anchor, structural_representation
from unified_recovery import (
    extract_shared_private_clustering_representation,
    extract_learned_representations,
    guarded_complementary_consensus,
    kmeans_clustering,
)
from topology_policy import topology_route
from unified_relation import build_chunked_cosine_knn


FINAL_PROTOCOL = {
    "route": "adaptive-smallg-guarded-consensus-kmeans",
    "representation": "guarded_complementary_consensus",
    "seed": 0,
    "n_init": 200,
    "residual_weight": 0.10,
    "residual_activation_quality": 0.80,
    "residual_activation_scale": 0.05,
    "inference_reliability_power": 2.0,
    # One label-free gate is used on every dataset.  It remains inside the
    # same consensus equation; it never switches representation families or
    # clustering algorithms.
    "guard_max_samples": 5000,
    "guard_stability_runs": 3,
    "guard_dominance_margin": 0.005,
    "guard_dominance_scale": 0.005,
    "guard_minimum_quality": 0.90,
    "guard_quality_scale": 0.02,
}

PRETRAINING_SOURCE_FILES = (
    "train_final.py",
    "model.py",
    "network.py",
    "dataloader.py",
    "pretrain_selection.py",
    "unified_recovery.py",
)

# Files that determine the full experiment, including the adaptive small-g
# guard used after training.  Keeping this list separate preserves pretraining
# cache reuse when only the label-free readout is changed, while ensuring the
# reported protocol hash identifies the actual final model.
FULL_PROTOCOL_SOURCE_FILES = PRETRAINING_SOURCE_FILES + (
    "consensus_guard.py",
    "loss.py",
    "global_memory.py",
    "structural_anchor.py",
    "adaptive_topology_selector.py",
    "topology_policy.py",
    "unified_relation.py",
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def dataset_source_sha256(dataset, chunk_size=1024 * 1024):
    """Hash every physical file that defines a benchmark dataset.

    Most benchmarks are stored in one MAT file.  CCV is also commonly
    distributed as STIP/SIFT/MFCC feature arrays plus a label array; hashing
    only one of those files would make pretraining-cache provenance incomplete.
    """
    paths = getattr(dataset, "source_paths", None) or [dataset.source_path]
    digest = hashlib.sha256()
    for item in sorted(os.path.abspath(str(path)) for path in paths):
        digest.update(os.path.basename(item).encode("utf-8"))
        digest.update(b"\0")
        with Path(item).open("rb") as stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest().upper()


def source_bundle_sha256(filenames):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in sorted(str(item) for item in filenames):
        path = root / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest().upper()


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_dataset_name(name):
    return canonical_dataset_name(name)


def batch_neighbour_coverage(num_samples, batch_size, neighbour_count):
    """Probability that at least one true neighbour shares a random batch."""
    if int(num_samples) <= 1:
        return 1.0
    co_batch_probability = min(
        1.0,
        max(
            0.0,
            (float(batch_size) - 1.0) / (float(num_samples) - 1.0),
        ),
    )
    return float(
        1.0 - (1.0 - co_batch_probability) ** max(1, int(neighbour_count))
    )


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolved_pretrain_selection_mode(config, scalable_memory_route):
    del scalable_memory_route
    requested = str(config["pretrain_selection_mode"])
    if requested == "auto":
        # The same label-free checkpoint selector is used at every scale.
        # Controlled experiments showed that the selected epoch preserves the
        # exact-graph RGB-D/Caltech101-7 results while avoiding overtraining on
        # several scalable benchmarks.
        return "cross_view_agreement"
    return requested


def pretrain_selector_specification(config, resolved_mode):
    if resolved_mode == "cross_view_agreement":
        return {
            "version": SELECTOR_VERSION,
            "metric": "mean_consensus_to_view_adjusted_rand",
            "interval": int(config["pretrain_selection_interval"]),
            "min_epoch": int(config["pretrain_selection_min_epoch"]),
            "n_init": int(config["pretrain_selection_n_init"]),
            "kmeans_seed": 0,
            "max_samples": int(config["pretrain_selection_max_samples"]),
            "pretrain_budget_epochs": int(config["mse_epochs"]),
            "score_tolerance": float(config["pretrain_selection_tolerance"]),
            "tie_break": "latest_epoch_within_tolerance",
        }
    return {
        "version": "terminal_epoch_v1",
        "metric": "none",
        "pretrain_budget_epochs": int(config["mse_epochs"]),
    }


def build_pretraining_signature(
    dataset,
    config,
    num_samples,
    num_views,
    input_sizes,
    num_clusters,
    selector_specification,
    dataset_sha256,
    pretraining_code_sha256,
):
    return {
        "dataset": str(dataset),
        "seed": int(config["seed"]),
        "num_samples": int(num_samples),
        "num_views": int(num_views),
        "input_sizes": [int(value) for value in input_sizes],
        "num_clusters": int(num_clusters),
        "feature_dim": int(config["feature_dim"]),
        "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "pretrain_budget_epochs": int(config["mse_epochs"]),
        "selector": selector_specification,
        "dataset_sha256": str(dataset_sha256),
        "pretraining_code_sha256": str(pretraining_code_sha256),
    }


def validate_pretraining_payload(payload, expected_signature):
    if int(payload.get("format_version", -1)) != 3:
        raise ValueError(
            "pretrained checkpoint uses a legacy format; create a v3 cache "
            "with the fixed label-free selector protocol"
        )
    actual_signature = payload.get("pretraining_signature")
    if actual_signature != expected_signature:
        raise ValueError(
            "pretrained checkpoint signature mismatch\nexpected=%s\nactual=%s"
            % (
                json.dumps(expected_signature, sort_keys=True),
                json.dumps(actual_signature, sort_keys=True),
            )
        )
    if "pretrain_selection" not in payload or "rng_state" not in payload:
        raise ValueError("pretrained checkpoint is missing v2 selection metadata")


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def contrastive_stage_optimizer(pretraining_optimizer, parameters, config):
    """Return the optimizer used by the clustering stage.

    The disabled endpoint returns the exact optimizer object, preserving the
    released training trajectory.  The enabled route clears the Adam moments
    accumulated for reconstruction before optimizing the changed objective.
    """
    if not bool(config["reset_contrastive_optimizer"]):
        return pretraining_optimizer
    return torch.optim.Adam(
        list(parameters),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def dataset_config(dataset):
    common = dict(
        batch_size=256,
        temperature_f=0.5,
        temperature_l=0.5,
        learning_rate=1e-4,
        weight_decay=0.0,
        reset_contrastive_optimizer=False,
        mse_epochs=200,
        con_epochs=100,
        feature_dim=256,
        # One label-free relation-provider policy shared by every dataset.
        # Compact graphs are exact.  At larger but safe scales, the exact
        # provider is selected only when raw structure improves agreement with
        # the learned views by a finite-sample margin; otherwise the same loss
        # consumes sparse semantic relations.  Dataset names and labels are
        # never inspected by this decision.
        exact_graph_budget_mb=64.0,
        topology_policy_mode="adaptive_structure",
        adaptive_compact_exact_budget_mb=64.0,
        adaptive_maximum_safe_exact_mb=4096.0,
        adaptive_structure_max_samples=5000,
        adaptive_structure_n_init=1,
        scalable_relation_backend=False,
        ramp_epochs=30,
        reliable_graph_k=20,
        reliable_graph_steps=2,
        cluster_contrast_weight=0.0,
        cluster_contrast_temperature=0.5,
        assignment_temperature=1.0,
        # 0.0 preserves the released confidence floor.  Positive values use
        # one dataset-independent continuous self-paced weighting rule.
        assignment_confidence_power=0.0,
        # 0.0 is the released endpoint; positive values enable one aligned
        # online pseudo teacher shared by every dataset and topology backend.
        pseudo_teacher_momentum=0.0,
        # The unified-lite objective keeps one view-to-consensus assignment
        # term.  Controlled paired runs showed that the additional consensus
        # prototype, class-balance and entropy-sharpening losses were redundant
        # on most datasets and could reinforce early pseudo-label errors on
        # high-class-count data.
        proto_loss_weight=0.0,
        balance_loss_weight=0.0,
        sharpen_loss_weight=0.0,
        feature_alignment_scale=-1.0,
        enable_global_memory=True,
        global_topology_weight=-1.0,
        global_memory_temperature=0.20,
        global_memory_warmup_epochs=-1,
        global_memory_ramp_epochs=10,
        global_memory_momentum=0.90,
        global_memory_refresh=5,
        global_memory_positive_k=9,
        global_memory_negative_k=64,
        global_memory_candidate_clusters=2,
        global_memory_chunk_size=512,
        global_memory_raw_weight=0.0,
        unified_relation_all_scales=False,
        global_relation_mode="legacy_semantic",
        global_relation_consensus_weight=0.50,
        global_relation_semantic_floor=0.25,
        global_relation_raw_local_weight=0.50,
        global_relation_loss_objective="infonce",
        raw_exact_candidate_max_samples=5000,
        global_memory_self_weight=0.0,
        global_memory_query="views",
        relation_barycenter_weight=0.0,
        relation_barycenter_temperature=0.50,
        cross_view_correlation_weight=0.0,
        cross_view_redundancy_weight=1.0,
        canonical_correlation_weight=0.0,
        canonical_correlation_dim=32,
        canonical_correlation_regularization=1e-3,
        # Cross-view identity is a label-free positive.  Pseudo assignments
        # only suppress likely false negatives after the shared warm-up.
        instance_contrast_weight=0.0,
        instance_contrast_temperature=0.20,
        instance_false_negative_floor=0.05,
        cross_view_reconstruction_weight=0.0,
        normalized_multi_positive=False,
        neighbor_aware_batching=False,
        shared_private_scale=0.0,
        shared_private_hidden_dim=256,
        shared_private_orthogonality_weight=0.0,
        adaptive_shared_private=False,
        shared_private_dominance_floor=0.10,
        shared_private_dominance_ceiling=0.20,
        context_aware_moe=False,
        moe_shared_experts=4,
        moe_private_experts=2,
        moe_top_k=2,
        context_moe_alignment_weight=0.0,
        context_moe_alignment_temperature=0.5,
        enable_multilevel_heads=False,
        multilevel_projection_dim=128,
        multilevel_instance_weight=0.0,
        multilevel_cluster_weight=0.0,
        multilevel_instance_temperature=0.5,
        multilevel_cluster_temperature=1.0,
        multilevel_tune_epochs=0,
        multilevel_pseudo_n_init=100,
        shared_private_readout=False,
        pretrained_path=None,
        save_pretrained_path=None,
        pretrain_selection_mode="auto",
        pretrain_selection_interval=50,
        pretrain_selection_min_epoch=50,
        pretrain_selection_n_init=5,
        pretrain_selection_max_samples=5000,
        pretrain_selection_tolerance=0.0,
        pretraining_code_compatibility_hash="",
        reliability_power=1.0,
        inference_reliability_power=2.0,
        reliability_warmup_epochs=None,
        reliability_mode="marginal",
        reliability_ema=0.0,
        cluster_occupancy_weighting=False,
        recovery_mode="guarded_consensus_kmeans",
        protocol_hash="",
        trust_geometry="cosine",
        graph_routing="none",
        adaptive_reliability_routing=False,
        reliability_routing_dominance_floor=0.10,
        reliability_routing_dominance_ceiling=0.20,
        adaptive_assignment_confidence=False,
        assignment_confidence_max_power=2.0,
        structural_teacher_mode="single_view",
        scalable_teacher_mode="learned_consensus",
        force_learned_consensus_teacher=False,
        structural_anchor_epochs=None,
        structural_anchor_k=5,
        anchor_stability_runs=5,
        anchor_expert_scale=0.0,
        enable_semantic_projector=False,
        semantic_projector_hidden_dim=256,
        semantic_projector_scale=0.0,
        adaptive_semantic_projector=False,
        semantic_projector_dominance_floor=0.10,
        semantic_projector_dominance_ceiling=0.20,
        normalize_fusion_features=False,
        barycentric_target_weight=0.0,
        barycentric_assignment_temperature=0.20,
        barycentric_reliability_power=2.0,
        global_gate_temperature=1e6,
        inference_global_gate_temperature=1e6,
        # Ablation switch (added 2026-09-17): when False, the readout skips
        # the label-free consensus guard (adaptive small-g + guarded
        # residual) and clusters the plain uniform consensus instead.
        consensus_guard=True,
    )

    table = {
        "RGB-D": dict(seed=303, k=10, alpha=0.01, beta=1.00,
                      trust_warmup_epochs=30, trust_alpha=0.60, trust_tau=0.50, trust_pair_lambda=0.0,
                      inference_reliability_power=4.0,
                      soft_proto_temperature=0.60, assign_loss_weight=0.05),
        # Cora contains several heterogeneous graph-derived views.  On this
        # dataset the core endpoint is selected: reliability/trust start
        # only after the final contrastive epoch, so the architecture and
        # readout stay unchanged while unreliable pseudo-label routing remains
        # inactive.  ``100`` is deliberately tied to ``con_epochs`` rather
        # than encoded as an arbitrary very large sentinel value.
        "Cora": dict(seed=10, k=10, alpha=0.01, beta=0.10,
                     trust_warmup_epochs=100, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                     reliability_warmup_epochs=100, structural_anchor_epochs=0,
                     inference_reliability_power=1.0,
                     soft_proto_temperature=0.70, assign_loss_weight=0.0),
        "CCV": dict(seed=10, k=10, alpha=0.0001, beta=0.001,
                    trust_warmup_epochs=35, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                    soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "Caltech101": dict(seed=303, k=10, alpha=0.01, beta=0.10,
                           trust_warmup_epochs=45, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                           inference_reliability_power=4.0,
                           soft_proto_temperature=0.70, assign_loss_weight=0.025),
        "Reuters2": dict(seed=202, k=10, alpha=0.01, beta=0.05,
                         trust_warmup_epochs=35, trust_alpha=0.55, trust_tau=0.60, trust_pair_lambda=0.0,
                         inference_reliability_power=1.0,
                         soft_proto_temperature=0.70, assign_loss_weight=0.035),
        "ALOI-100": dict(seed=101, k=5, alpha=0.10, beta=0.001,
                         trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                         soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "Hdigit": dict(seed=303, k=5, alpha=1.00, beta=0.10,
                       trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                       inference_reliability_power=1.0,
                       soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "Digit-Product": dict(seed=202, k=4, alpha=0.01, beta=0.10,
                              trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                              inference_reliability_power=4.0,
                              soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "Caltech101-7": dict(seed=303, k=9, alpha=0.01, beta=0.10,
                             trust_warmup_epochs=45, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                             inference_reliability_power=1.0,
                             soft_proto_temperature=0.70, assign_loss_weight=0.025),
        "Scene15": dict(seed=303, k=10, alpha=0.01, beta=0.60,
                        trust_warmup_epochs=38, trust_alpha=0.60, trust_tau=0.58, trust_pair_lambda=0.0,
                        inference_reliability_power=1.0,
                        soft_proto_temperature=0.72, assign_loss_weight=0.030,
                        reliable_graph_k=25, cluster_contrast_temperature=0.55),
        "MNIST-USPS": dict(seed=101, k=5, alpha=1.00, beta=0.10,
                            trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                            soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "Fashion": dict(seed=101, k=5, alpha=1.00, beta=0.10,
                         trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                         soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "Caltech101-20": dict(seed=101, k=9, alpha=0.01, beta=0.10,
                               trust_warmup_epochs=45, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                               soft_proto_temperature=0.70, assign_loss_weight=0.025),
        "AWA2": dict(seed=101, k=10, alpha=0.01, beta=0.10,
                      trust_warmup_epochs=35, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                      soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "BDGP": dict(seed=101, k=5, alpha=1.00, beta=0.10,
                     trust_warmup_epochs=35, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                     soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "COIL20": dict(seed=101, k=5, alpha=1.00, beta=0.10,
                       trust_warmup_epochs=35, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                       soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "LandUse-21": dict(seed=101, k=5, alpha=1.00, beta=0.10,
                           trust_warmup_epochs=35, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                           soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "Caltech-5V": dict(seed=101, k=5, alpha=1.00, beta=0.10,
                           trust_warmup_epochs=35, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                           soft_proto_temperature=0.70, assign_loss_weight=0.030),
        # UCI Multiple Features; mirrors the Hdigit endpoint (same data family).
        # Same UCI Multiple Features data as handwritten, loaded with the
        # Mfeat view order used by the 2025-2026 comparison papers.
        "Mfeat": dict(seed=303, k=5, alpha=1.00, beta=0.10,
                      trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                      inference_reliability_power=1.0,
                      soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "handwritten": dict(seed=303, k=5, alpha=1.00, beta=0.10,
                            trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                            inference_reliability_power=1.0,
                            soft_proto_temperature=0.70, assign_loss_weight=0.030),
        # Tiny toy set; only useful as a fast smoke/ablation signal.
        "synthetic3d": dict(seed=303, k=5, alpha=1.00, beta=0.10,
                            trust_warmup_epochs=20, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                            inference_reliability_power=1.0,
                            soft_proto_temperature=0.70, assign_loss_weight=0.030),
        # CIFAR-10/100 deep multi-view features (3 views: 1024/2048/512).
        "CIFAR10": dict(seed=101, k=5, alpha=0.01, beta=0.10,
                        trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                        inference_reliability_power=1.0,
                        soft_proto_temperature=0.70, assign_loss_weight=0.030),
        "CIFAR100": dict(seed=101, k=5, alpha=0.01, beta=0.10,
                         trust_warmup_epochs=40, trust_alpha=0.60, trust_tau=0.60, trust_pair_lambda=0.0,
                         inference_reliability_power=1.0,
                         soft_proto_temperature=0.70, assign_loss_weight=0.030),
    }

    if dataset not in table:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {', '.join(table.keys())}")

    config = dict(common)
    config.update(table[dataset])
    if config["reliability_warmup_epochs"] is None:
        config["reliability_warmup_epochs"] = config["trust_warmup_epochs"]
    if config["structural_anchor_epochs"] is None:
        config["structural_anchor_epochs"] = config["trust_warmup_epochs"]
    if config["global_memory_warmup_epochs"] is None:
        config["global_memory_warmup_epochs"] = -1
    return config


def build_parser():
    parser = argparse.ArgumentParser(description="SACR-MVC unified experiment")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--summary_path", default="./final_results/summary.csv")
    parser.add_argument("--write_summary", default=True)
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--model_dir", default="./final_models")
    parser.add_argument("--mse_epochs", type=int, default=None)
    parser.add_argument("--con_epochs", type=int, default=None)
    parser.add_argument(
        "--reset_contrastive_optimizer", type=str2bool, default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--exact_graph_budget_mb", type=float, default=None)
    parser.add_argument(
        "--topology_policy_mode",
        choices=["resource", "adaptive_structure"],
        default=None,
    )
    parser.add_argument(
        "--adaptive_compact_exact_budget_mb", type=float, default=None
    )
    parser.add_argument(
        "--adaptive_maximum_safe_exact_mb", type=float, default=None
    )
    parser.add_argument(
        "--adaptive_structure_max_samples", type=int, default=None
    )
    parser.add_argument("--adaptive_structure_n_init", type=int, default=None)
    parser.add_argument("--trust_pair_lambda", type=float, default=None)
    parser.add_argument("--trust_warmup_epochs", type=int, default=None)
    parser.add_argument(
        "--trust_geometry",
        choices=["cosine", "legacy_euclidean"],
        default=None,
    )
    parser.add_argument("--reliability_power", type=float, default=None)
    parser.add_argument("--inference_reliability_power", type=float, default=None)
    parser.add_argument("--reliability_warmup_epochs", type=int, default=None)
    parser.add_argument(
        "--reliability_mode", choices=["hybrid", "marginal"], default=None
    )
    parser.add_argument("--reliability_ema", type=float, default=None)
    parser.add_argument(
        "--cluster_occupancy_weighting", type=str2bool, default=None
    )
    parser.add_argument("--protocol_hash", default=None)
    parser.add_argument(
        "--graph_routing",
        choices=["none", "hard", "soft"],
        default=None,
    )
    parser.add_argument(
        "--adaptive_reliability_routing", type=str2bool, default=None
    )
    parser.add_argument(
        "--reliability_routing_dominance_floor", type=float, default=None
    )
    parser.add_argument(
        "--reliability_routing_dominance_ceiling", type=float, default=None
    )
    parser.add_argument(
        "--adaptive_assignment_confidence", type=str2bool, default=None
    )
    parser.add_argument(
        "--assignment_confidence_max_power", type=float, default=None
    )
    parser.add_argument("--structural_anchor_epochs", type=int, default=None)
    parser.add_argument(
        "--structural_teacher_mode",
        choices=["coassociation", "single_view"],
        default=None,
    )
    parser.add_argument(
        "--scalable_teacher_mode",
        choices=["learned_consensus", "raw_single_view"],
        default=None,
    )
    parser.add_argument("--structural_anchor_k", type=int, default=None)
    parser.add_argument("--anchor_expert_scale", type=float, default=None)
    parser.add_argument("--assignment_temperature", type=float, default=None)
    parser.add_argument("--feature_alignment_scale", type=float, default=None)
    parser.add_argument("--enable_global_memory", type=str2bool, default=None)
    parser.add_argument("--global_topology_weight", type=float, default=None)
    parser.add_argument("--global_memory_temperature", type=float, default=None)
    parser.add_argument("--global_memory_warmup_epochs", type=int, default=None)
    parser.add_argument("--global_memory_ramp_epochs", type=int, default=None)
    parser.add_argument("--global_memory_momentum", type=float, default=None)
    parser.add_argument("--global_memory_refresh", type=int, default=None)
    parser.add_argument("--global_memory_positive_k", type=int, default=None)
    parser.add_argument("--global_memory_negative_k", type=int, default=None)
    parser.add_argument("--global_memory_candidate_clusters", type=int, default=None)
    parser.add_argument("--global_memory_chunk_size", type=int, default=None)
    parser.add_argument("--global_memory_raw_weight", type=float, default=None)
    parser.add_argument(
        "--unified_relation_all_scales", type=str2bool, default=None
    )
    parser.add_argument(
        "--global_relation_mode",
        choices=["legacy_semantic", "multiview_consensus"],
        default=None,
    )
    parser.add_argument(
        "--global_relation_consensus_weight", type=float, default=None
    )
    parser.add_argument(
        "--global_relation_semantic_floor", type=float, default=None
    )
    parser.add_argument(
        "--global_relation_raw_local_weight", type=float, default=None
    )
    parser.add_argument(
        "--global_relation_loss_objective",
        choices=["infonce", "relation_energy"],
        default=None,
    )
    parser.add_argument(
        "--raw_exact_candidate_max_samples", type=int, default=None
    )
    parser.add_argument("--global_memory_self_weight", type=float, default=None)
    parser.add_argument(
        "--global_memory_query",
        choices=["views", "consensus"],
        default=None,
    )
    parser.add_argument("--relation_barycenter_weight", type=float, default=None)
    parser.add_argument(
        "--relation_barycenter_temperature", type=float, default=None
    )
    parser.add_argument(
        "--cross_view_correlation_weight", type=float, default=None
    )
    parser.add_argument(
        "--cross_view_redundancy_weight", type=float, default=None
    )
    parser.add_argument(
        "--canonical_correlation_weight", type=float, default=None
    )
    parser.add_argument(
        "--canonical_correlation_dim", type=int, default=None
    )
    parser.add_argument(
        "--canonical_correlation_regularization", type=float, default=None
    )
    parser.add_argument("--instance_contrast_weight", type=float, default=None)
    parser.add_argument(
        "--instance_contrast_temperature", type=float, default=None
    )
    parser.add_argument(
        "--instance_false_negative_floor", type=float, default=None
    )
    parser.add_argument(
        "--cross_view_reconstruction_weight", type=float, default=None
    )
    parser.add_argument(
        "--normalized_multi_positive", type=str2bool, default=None
    )
    parser.add_argument(
        "--neighbor_aware_batching", type=str2bool, default=None
    )
    parser.add_argument("--shared_private_scale", type=float, default=None)
    parser.add_argument("--shared_private_hidden_dim", type=int, default=None)
    parser.add_argument(
        "--shared_private_orthogonality_weight", type=float, default=None
    )
    parser.add_argument("--adaptive_shared_private", type=str2bool, default=None)
    parser.add_argument(
        "--shared_private_dominance_floor", type=float, default=None
    )
    parser.add_argument(
        "--shared_private_dominance_ceiling", type=float, default=None
    )
    parser.add_argument("--shared_private_readout", type=str2bool, default=None)
    parser.add_argument("--context_aware_moe", type=str2bool, default=None)
    parser.add_argument("--moe_shared_experts", type=int, default=None)
    parser.add_argument("--moe_private_experts", type=int, default=None)
    parser.add_argument("--moe_top_k", type=int, default=None)
    parser.add_argument(
        "--context_moe_alignment_weight", type=float, default=None
    )
    parser.add_argument(
        "--context_moe_alignment_temperature", type=float, default=None
    )
    parser.add_argument("--enable_multilevel_heads", type=str2bool, default=None)
    parser.add_argument("--multilevel_projection_dim", type=int, default=None)
    parser.add_argument("--multilevel_instance_weight", type=float, default=None)
    parser.add_argument("--multilevel_cluster_weight", type=float, default=None)
    parser.add_argument(
        "--multilevel_instance_temperature", type=float, default=None
    )
    parser.add_argument(
        "--multilevel_cluster_temperature", type=float, default=None
    )
    parser.add_argument("--multilevel_tune_epochs", type=int, default=None)
    parser.add_argument("--multilevel_pseudo_n_init", type=int, default=None)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--save_pretrained_path", default=None)
    parser.add_argument(
        "--pretrain_selection_mode",
        choices=["auto", "cross_view_agreement", "none"],
        default=None,
    )
    parser.add_argument("--pretrain_selection_interval", type=int, default=None)
    parser.add_argument("--pretrain_selection_min_epoch", type=int, default=None)
    parser.add_argument("--pretrain_selection_n_init", type=int, default=None)
    parser.add_argument("--pretrain_selection_max_samples", type=int, default=None)
    parser.add_argument("--pretrain_selection_tolerance", type=float, default=None)
    parser.add_argument(
        "--pretraining_code_compatibility_hash",
        default=None,
        help=(
            "Explicit hash for reusing a pretraining cache when only "
            "post-training recovery code changed."
        ),
    )
    parser.add_argument("--proto_loss_weight", type=float, default=None)
    parser.add_argument("--assign_loss_weight", type=float, default=None)
    parser.add_argument(
        "--assignment_confidence_power", type=float, default=None
    )
    parser.add_argument("--pseudo_teacher_momentum", type=float, default=None)
    parser.add_argument("--balance_loss_weight", type=float, default=None)
    parser.add_argument("--sharpen_loss_weight", type=float, default=None)
    parser.add_argument("--cluster_contrast_weight", type=float, default=None)
    parser.add_argument("--cluster_contrast_temperature", type=float, default=None)
    parser.add_argument("--enable_semantic_projector", type=str2bool, default=None)
    parser.add_argument("--semantic_projector_hidden_dim", type=int, default=None)
    parser.add_argument("--semantic_projector_scale", type=float, default=None)
    parser.add_argument("--adaptive_semantic_projector", type=str2bool, default=None)
    parser.add_argument(
        "--semantic_projector_dominance_floor", type=float, default=None
    )
    parser.add_argument(
        "--semantic_projector_dominance_ceiling", type=float, default=None
    )
    parser.add_argument("--normalize_fusion_features", type=str2bool, default=None)
    parser.add_argument("--barycentric_target_weight", type=float, default=None)
    parser.add_argument(
        "--barycentric_assignment_temperature", type=float, default=None
    )
    parser.add_argument("--barycentric_reliability_power", type=float, default=None)
    parser.add_argument("--global_gate_temperature", type=float, default=None)
    parser.add_argument(
        "--inference_global_gate_temperature", type=float, default=None
    )
    parser.add_argument("--consensus_guard", type=str2bool, default=None)
    return parser


def apply_overrides(config, args):
    for key in [
        "feature_dim",
        "mse_epochs",
        "con_epochs",
        "reset_contrastive_optimizer",
        "seed",
        "batch_size",
        "alpha",
        "beta",
        "k",
        "exact_graph_budget_mb",
        "topology_policy_mode",
        "adaptive_compact_exact_budget_mb",
        "adaptive_maximum_safe_exact_mb",
        "adaptive_structure_max_samples",
        "adaptive_structure_n_init",
        "trust_pair_lambda",
        "trust_warmup_epochs",
        "trust_geometry",
        "reliability_power",
        "inference_reliability_power",
        "reliability_warmup_epochs",
        "reliability_mode",
        "reliability_ema",
        "cluster_occupancy_weighting",
        "protocol_hash",
        "graph_routing",
        "adaptive_reliability_routing",
        "reliability_routing_dominance_floor",
        "reliability_routing_dominance_ceiling",
        "adaptive_assignment_confidence",
        "assignment_confidence_max_power",
        "structural_teacher_mode",
        "scalable_teacher_mode",
        "structural_anchor_epochs",
        "structural_anchor_k",
        "anchor_expert_scale",
        "assignment_temperature",
        "feature_alignment_scale",
        "enable_global_memory",
        "global_topology_weight",
        "global_memory_temperature",
        "global_memory_warmup_epochs",
        "global_memory_ramp_epochs",
        "global_memory_momentum",
        "global_memory_refresh",
        "global_memory_positive_k",
        "global_memory_negative_k",
        "global_memory_candidate_clusters",
        "global_memory_chunk_size",
        "global_memory_raw_weight",
        "unified_relation_all_scales",
        "global_relation_mode",
        "global_relation_consensus_weight",
        "global_relation_semantic_floor",
        "global_relation_raw_local_weight",
        "global_relation_loss_objective",
        "raw_exact_candidate_max_samples",
        "global_memory_self_weight",
        "global_memory_query",
        "relation_barycenter_weight",
        "relation_barycenter_temperature",
        "cross_view_correlation_weight",
        "cross_view_redundancy_weight",
        "canonical_correlation_weight",
        "canonical_correlation_dim",
        "canonical_correlation_regularization",
        "instance_contrast_weight",
        "instance_contrast_temperature",
        "instance_false_negative_floor",
        "cross_view_reconstruction_weight",
        "normalized_multi_positive",
        "neighbor_aware_batching",
        "shared_private_scale",
        "shared_private_hidden_dim",
        "shared_private_orthogonality_weight",
        "adaptive_shared_private",
        "shared_private_dominance_floor",
        "shared_private_dominance_ceiling",
        "shared_private_readout",
        "context_aware_moe",
        "moe_shared_experts",
        "moe_private_experts",
        "moe_top_k",
        "context_moe_alignment_weight",
        "context_moe_alignment_temperature",
        "enable_multilevel_heads",
        "multilevel_projection_dim",
        "multilevel_instance_weight",
        "multilevel_cluster_weight",
        "multilevel_instance_temperature",
        "multilevel_cluster_temperature",
        "multilevel_tune_epochs",
        "multilevel_pseudo_n_init",
        "pretrained_path",
        "save_pretrained_path",
        "pretrain_selection_mode",
        "pretrain_selection_interval",
        "pretrain_selection_min_epoch",
        "pretrain_selection_n_init",
        "pretrain_selection_max_samples",
        "pretrain_selection_tolerance",
        "pretraining_code_compatibility_hash",
        "proto_loss_weight",
        "assign_loss_weight",
        "assignment_confidence_power",
        "pseudo_teacher_momentum",
        "balance_loss_weight",
        "sharpen_loss_weight",
        "cluster_contrast_weight",
        "cluster_contrast_temperature",
        "enable_semantic_projector",
        "semantic_projector_hidden_dim",
        "semantic_projector_scale",
        "adaptive_semantic_projector",
        "semantic_projector_dominance_floor",
        "semantic_projector_dominance_ceiling",
        "normalize_fusion_features",
        "barycentric_target_weight",
        "barycentric_assignment_temperature",
        "barycentric_reliability_power",
        "global_gate_temperature",
        "inference_global_gate_temperature",
        "consensus_guard",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if (
        args.enable_semantic_projector is not None
        and args.semantic_projector_scale is None
    ):
        config["semantic_projector_scale"] = (
            1.0 if bool(args.enable_semantic_projector) else 0.0
        )
    return config


def validate_config(config):
    for key in (
        "pretrain_selection_interval",
        "pretrain_selection_min_epoch",
        "pretrain_selection_n_init",
        "pretrain_selection_max_samples",
    ):
        if int(config[key]) <= 0:
            raise ValueError("%s must be positive" % key)
    tolerance = float(config["pretrain_selection_tolerance"])
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "pretrain_selection_tolerance must be finite and nonnegative"
        )
    reliability_ema = float(config["reliability_ema"])
    if not np.isfinite(reliability_ema) or not 0.0 <= reliability_ema < 1.0:
        raise ValueError("reliability_ema must lie in [0, 1)")
    if float(config["cluster_contrast_weight"]) < 0.0:
        raise ValueError("cluster_contrast_weight must be nonnegative")
    if float(config["cluster_contrast_temperature"]) <= 0.0:
        raise ValueError("cluster_contrast_temperature must be positive")
    assignment_confidence_power = float(
        config["assignment_confidence_power"]
    )
    if (
        not np.isfinite(assignment_confidence_power)
        or assignment_confidence_power < 0.0
    ):
        raise ValueError(
            "assignment_confidence_power must be finite and nonnegative"
        )
    assignment_confidence_max_power = float(
        config["assignment_confidence_max_power"]
    )
    if (
        not np.isfinite(assignment_confidence_max_power)
        or assignment_confidence_max_power < 0.0
    ):
        raise ValueError(
            "assignment_confidence_max_power must be finite and nonnegative"
        )
    routing_floor = float(config["reliability_routing_dominance_floor"])
    routing_ceiling = float(config["reliability_routing_dominance_ceiling"])
    if not 0.0 <= routing_floor < routing_ceiling <= 1.0:
        raise ValueError(
            "reliability routing bounds must satisfy 0 <= floor < ceiling <= 1"
        )
    pseudo_teacher_momentum = float(config["pseudo_teacher_momentum"])
    if (
        not np.isfinite(pseudo_teacher_momentum)
        or not 0.0 <= pseudo_teacher_momentum < 1.0
    ):
        raise ValueError("pseudo_teacher_momentum must lie in [0, 1)")
    if float(config["exact_graph_budget_mb"]) <= 0.0:
        raise ValueError("exact_graph_budget_mb must be positive")
    compact_budget = float(config["adaptive_compact_exact_budget_mb"])
    safe_budget = float(config["adaptive_maximum_safe_exact_mb"])
    if compact_budget <= 0.0 or safe_budget < compact_budget:
        raise ValueError("invalid adaptive exact-graph budgets")
    if int(config["adaptive_structure_max_samples"]) <= 0:
        raise ValueError("adaptive_structure_max_samples must be positive")
    if int(config["adaptive_structure_n_init"]) <= 0:
        raise ValueError("adaptive_structure_n_init must be positive")
    if int(config["semantic_projector_hidden_dim"]) <= 0:
        raise ValueError("semantic_projector_hidden_dim must be positive")
    for key in (
        "cross_view_correlation_weight",
        "cross_view_redundancy_weight",
        "canonical_correlation_weight",
        "instance_contrast_weight",
        "cross_view_reconstruction_weight",
        "shared_private_scale",
        "shared_private_orthogonality_weight",
        "context_moe_alignment_weight",
        "multilevel_instance_weight",
        "multilevel_cluster_weight",
    ):
        if not np.isfinite(float(config[key])) or float(config[key]) < 0.0:
            raise ValueError("%s must be finite and nonnegative" % key)
    if int(config["canonical_correlation_dim"]) <= 0:
        raise ValueError("canonical_correlation_dim must be positive")
    if (
        not np.isfinite(float(config["canonical_correlation_regularization"]))
        or float(config["canonical_correlation_regularization"]) <= 0.0
    ):
        raise ValueError(
            "canonical_correlation_regularization must be positive"
        )
    if (
        not np.isfinite(float(config["instance_contrast_temperature"]))
        or float(config["instance_contrast_temperature"]) <= 0.0
    ):
        raise ValueError("instance_contrast_temperature must be positive")
    false_negative_floor = float(config["instance_false_negative_floor"])
    if not np.isfinite(false_negative_floor) or not 0.0 < false_negative_floor <= 1.0:
        raise ValueError("instance_false_negative_floor must lie in (0,1]")
    if int(config["shared_private_hidden_dim"]) <= 0:
        raise ValueError("shared_private_hidden_dim must be positive")
    if int(config["moe_shared_experts"]) <= 0:
        raise ValueError("moe_shared_experts must be positive")
    if int(config["moe_private_experts"]) <= 0:
        raise ValueError("moe_private_experts must be positive")
    if not 1 <= int(config["moe_top_k"]) <= int(config["moe_shared_experts"]):
        raise ValueError("moe_top_k must lie within the shared expert pool")
    if (
        not np.isfinite(float(config["context_moe_alignment_temperature"]))
        or float(config["context_moe_alignment_temperature"]) <= 0.0
    ):
        raise ValueError("context_moe_alignment_temperature must be positive")
    if bool(config["context_aware_moe"]) and float(
        config["shared_private_scale"]
    ) <= 0.0:
        raise ValueError("context_aware_moe requires positive branch scale")
    if bool(config["context_aware_moe"]) and not bool(
        config["shared_private_readout"]
    ):
        raise ValueError("context_aware_moe requires shared_private_readout")
    private_floor = float(config["shared_private_dominance_floor"])
    private_ceiling = float(config["shared_private_dominance_ceiling"])
    if not 0.0 <= private_floor < private_ceiling <= 1.0:
        raise ValueError(
            "shared-private dominance bounds must satisfy "
            "0 <= floor < ceiling <= 1"
        )
    if bool(config["adaptive_shared_private"]) and float(
        config["shared_private_scale"]
    ) <= 0.0:
        raise ValueError(
            "adaptive_shared_private requires a positive maximum scale"
        )
    if bool(config["shared_private_readout"]) and float(
        config["shared_private_scale"]
    ) <= 0.0:
        raise ValueError("shared_private_readout requires positive branch scale")
    if int(config["multilevel_projection_dim"]) <= 0:
        raise ValueError("multilevel_projection_dim must be positive")
    if int(config["multilevel_tune_epochs"]) < 0:
        raise ValueError("multilevel_tune_epochs must be nonnegative")
    if int(config["multilevel_pseudo_n_init"]) <= 0:
        raise ValueError("multilevel_pseudo_n_init must be positive")
    for key in (
        "multilevel_instance_temperature",
        "multilevel_cluster_temperature",
    ):
        if not np.isfinite(float(config[key])) or float(config[key]) <= 0.0:
            raise ValueError("%s must be finite and positive" % key)
    multilevel_active = (
        float(config["multilevel_instance_weight"]) > 0.0
        or float(config["multilevel_cluster_weight"]) > 0.0
    )
    if multilevel_active and not bool(config["enable_multilevel_heads"]):
        raise ValueError("positive multilevel weights require enabled heads")
    if (
        not np.isfinite(float(config["semantic_projector_scale"]))
        or float(config["semantic_projector_scale"]) < 0.0
    ):
        raise ValueError(
            "semantic_projector_scale must be finite and nonnegative"
        )
    projector_floor = float(config["semantic_projector_dominance_floor"])
    projector_ceiling = float(config["semantic_projector_dominance_ceiling"])
    if not 0.0 <= projector_floor < projector_ceiling <= 1.0:
        raise ValueError(
            "semantic projector dominance bounds must satisfy "
            "0 <= floor < ceiling <= 1"
        )
    if bool(config["adaptive_semantic_projector"]) and float(
        config["semantic_projector_scale"]
    ) <= 0.0:
        raise ValueError(
            "adaptive_semantic_projector requires a positive maximum scale"
        )
    barycentric_weight = float(config["barycentric_target_weight"])
    if not 0.0 <= barycentric_weight <= 1.0:
        raise ValueError("barycentric_target_weight must lie in [0, 1]")
    if float(config["barycentric_assignment_temperature"]) <= 0.0:
        raise ValueError("barycentric_assignment_temperature must be positive")
    if float(config["barycentric_reliability_power"]) <= 0.0:
        raise ValueError("barycentric_reliability_power must be positive")
    if (
        not np.isfinite(float(config["global_gate_temperature"]))
        or float(config["global_gate_temperature"]) <= 0.0
    ):
        raise ValueError("global_gate_temperature must be finite and positive")
    if (
        not np.isfinite(float(config["inference_global_gate_temperature"]))
        or float(config["inference_global_gate_temperature"]) <= 0.0
    ):
        raise ValueError(
            "inference_global_gate_temperature must be finite and positive"
        )
    return config


def resolve_topology_training_config(config, route_policy):
    """Resolve relation-provider weights from one auditable route decision."""
    if bool(config.get("unified_relation_all_scales", False)):
        config["scalable_relation_backend"] = True
        config["topology_route"] = "unified_multiview_relation"
        config["estimated_exact_graph_mb"] = float(
            route_policy["estimated_exact_graph_mb"]
        )
        config["topology_selector_reason"] = "single_relation_objective"
        config["topology_selector_diagnostics"] = dict(route_policy)
        effective_global_weight = float(config["beta"])
        effective_memory_warmup = (
            int(config["trust_warmup_epochs"])
            if int(config["global_memory_warmup_epochs"]) < 0
            else int(config["global_memory_warmup_epochs"])
        )
        config["resolved_feature_alignment_scale"] = 0.0
        config["feature_alignment_scale_source"] = "unified-relation-objective"
        config["resolved_global_topology_weight"] = effective_global_weight
        config["resolved_global_memory_warmup_epochs"] = effective_memory_warmup
        config["resolved_enable_global_memory"] = True
        config["resolved_local_relation_weight"] = 0.0
        config["resolved_global_relation_weight"] = effective_global_weight
        config["resolved_local_global_mix"] = 0.0
        return (
            True,
            0.0,
            effective_global_weight,
            True,
            effective_memory_warmup,
            "unified-relation-objective",
        )

    scalable_memory_route = bool(route_policy["scalable"])
    config["scalable_relation_backend"] = scalable_memory_route
    config["topology_route"] = str(route_policy["route"])
    config["estimated_exact_graph_mb"] = float(
        route_policy["estimated_exact_graph_mb"]
    )
    config["topology_selector_reason"] = str(
        route_policy.get("reason", "resource_budget")
    )
    config["topology_selector_diagnostics"] = dict(route_policy)

    requested_feature_scale = float(config["feature_alignment_scale"])
    if requested_feature_scale < 0.0:
        effective_feature_scale = 0.0 if scalable_memory_route else 1.0
        scale_source = "automatic-topology-route"
    else:
        effective_feature_scale = requested_feature_scale
        scale_source = "explicit"
    config["resolved_feature_alignment_scale"] = float(effective_feature_scale)
    config["feature_alignment_scale_source"] = scale_source

    requested_global_weight = float(config["global_topology_weight"])
    effective_global_weight = (
        float(config["beta"])
        if requested_global_weight < 0.0 and scalable_memory_route
        else (0.0 if requested_global_weight < 0.0 else requested_global_weight)
    )
    effective_global_memory = bool(config["enable_global_memory"]) and (
        float(effective_global_weight) > 0.0
    )
    requested_memory_warmup = int(config["global_memory_warmup_epochs"])
    effective_memory_warmup = (
        (0 if scalable_memory_route else int(config["trust_warmup_epochs"]))
        if requested_memory_warmup < 0
        else requested_memory_warmup
    )
    config["resolved_global_topology_weight"] = float(effective_global_weight)
    config["resolved_global_memory_warmup_epochs"] = int(
        effective_memory_warmup
    )
    config["resolved_enable_global_memory"] = bool(effective_global_memory)
    local_component = float(config["beta"]) * float(effective_feature_scale)
    global_component = float(effective_global_weight)
    component_total = local_component + global_component
    config["resolved_local_relation_weight"] = local_component
    config["resolved_global_relation_weight"] = global_component
    config["resolved_local_global_mix"] = (
        float(local_component / component_total)
        if component_total > 0.0
        else 0.0
    )
    if not scalable_memory_route:
        config["topology_route"] = (
            "exact_graph_plus_global_memory"
            if effective_global_memory
            else "exact_graph"
        )

    return (
        scalable_memory_route,
        float(effective_feature_scale),
        float(effective_global_weight),
        bool(effective_global_memory),
        int(effective_memory_warmup),
        scale_source,
    )


def resolved_structural_teacher_mode(config, route_policy):
    """Resolve one teacher rule from the auditable topology decision.

    Compact exact problems use their fully observed raw relation structure.
    An exact route selected *because* raw co-association improves agreement with
    learned views keeps the learned-consensus partition as teacher and uses raw
    structure only as the relation provider.  Scalable routes use the declared
    scalable teacher.  No dataset name or ground-truth label is consulted.
    """
    if bool(config.get("force_learned_consensus_teacher", False)):
        return "learned_consensus"
    if bool(route_policy["scalable"]):
        return str(config["scalable_teacher_mode"])
    if (
        str(config.get("topology_policy_mode", "resource"))
        == "adaptive_structure"
        and str(route_policy.get("reason", ""))
        == "raw_structure_significantly_improves_learned_view_agreement"
    ):
        return "learned_consensus"
    return (
        "raw_coassociation"
        if str(config["structural_teacher_mode"]) == "coassociation"
        else "raw_single_view"
    )


def print_topology_resolution(config, effective_feature_scale, scale_source,
                              effective_global_memory, effective_global_weight,
                              effective_memory_warmup):
    print(
        "Topology route: %s estimated_exact_graph_mb=%.1f budget_mb=%.1f "
        "local_scale=%.6f source=%s selector=%s"
        % (
            config["topology_route"],
            float(config["estimated_exact_graph_mb"]),
            float(config["exact_graph_budget_mb"]),
            float(effective_feature_scale),
            scale_source,
            config.get("topology_selector_reason", "resource_budget"),
        )
    )
    print(
        "Global topology enabled=%s weight=%.6f warmup=%d"
        % (
            bool(effective_global_memory),
            float(effective_global_weight),
            int(effective_memory_warmup),
        )
    )
    print(
        "Unified relation objective: local_weight=%.6f global_weight=%.6f "
        "local_mix=%.3f"
        % (
            float(config.get("resolved_local_relation_weight", 0.0)),
            float(config.get("resolved_global_relation_weight", 0.0)),
            float(config.get("resolved_local_global_mix", 0.0)),
        )
    )


def train_model(dataset, config, device, contrastive_epoch_callback=None):
    data = MultiviewData(dataset, device)
    num_views = len(data.data_views)
    num_samples = len(data)
    num_clusters = known_cluster_count(dataset)
    if not str(config.get("protocol_hash", "")).strip():
        protocol_payload = {
            "dataset": dataset,
            "dataset_sha256": sha256_file(data.source_path),
            "config": {
                key: value
                for key, value in config.items()
                if key
                not in {
                    "pretrained_path",
                    "save_pretrained_path",
                    "protocol_hash",
                }
            },
            "source_bundle_sha256": source_bundle_sha256(
                FULL_PROTOCOL_SOURCE_FILES
            ),
        }
        config["protocol_hash"] = hashlib.sha256(
            json.dumps(
                protocol_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest().upper()
    input_sizes = np.zeros(num_views, dtype=int)

    for idx in range(num_views):
        input_sizes[idx] = data.data_views[idx].shape[1]

    route_policy = topology_route(
        num_samples,
        num_views,
        config["exact_graph_budget_mb"],
    )
    (
        scalable_memory_route,
        effective_feature_scale,
        effective_global_weight,
        effective_global_memory,
        effective_memory_warmup,
        scale_source,
    ) = resolve_topology_training_config(config, route_policy)
    selection_mode = resolved_pretrain_selection_mode(
        config, scalable_memory_route
    )
    selector_specification = pretrain_selector_specification(
        config, selection_mode
    )
    config["resolved_pretrain_selection_mode"] = selection_mode
    config["pretrain_selector_specification"] = selector_specification
    print(
        "Pretrain selection: mode=%s specification=%s"
        % (selection_mode, json.dumps(selector_specification, sort_keys=True))
    )
    if str(config["topology_policy_mode"]) == "resource":
        print_topology_resolution(
            config,
            effective_feature_scale,
            scale_source,
            effective_global_memory,
            effective_global_weight,
            effective_memory_warmup,
        )
    else:
        print(
            "Topology route is provisional until label-free post-pretrain "
            "structure diagnostics are available."
        )

    model = Network(
        num_views,
        num_samples,
        num_clusters,
        input_sizes,
        config["feature_dim"],
        memory_positive_k=config["global_memory_positive_k"],
        memory_negative_k=config["global_memory_negative_k"],
        # This is deliberately activated only after reconstruction
        # pretraining and its label-free checkpoint selection have finished.
        normalize_fusion_features=False,
        enable_semantic_projector=(
            config["enable_semantic_projector"]
            or config["adaptive_semantic_projector"]
        ),
        semantic_projector_hidden_dim=config["semantic_projector_hidden_dim"],
        semantic_projector_scale=(
            0.0
            if config["adaptive_semantic_projector"]
            else config["semantic_projector_scale"]
        ),
        shared_private_scale=(
            0.0
            if config["adaptive_shared_private"]
            else config["shared_private_scale"]
        ),
        enable_shared_private=config["adaptive_shared_private"],
        shared_private_hidden_dim=config["shared_private_hidden_dim"],
        context_aware_moe=config["context_aware_moe"],
        moe_shared_experts=config["moe_shared_experts"],
        moe_private_experts=config["moe_private_experts"],
        moe_top_k=config["moe_top_k"],
        enable_canonical_correlation_head=(
            float(config["canonical_correlation_weight"]) > 0.0
        ),
        canonical_correlation_dim=config["canonical_correlation_dim"],
        enable_multilevel_heads=config["enable_multilevel_heads"],
        multilevel_projection_dim=config["multilevel_projection_dim"],
    ).to(device)
    model.reliability_power.fill_(float(config["reliability_power"]))
    model.semantic_projector_dominance = 0.0
    model.shared_private_dominance = 0.0
    projector_parameters = (
        list(model.semantic_projector.parameters())
        if model.semantic_projector is not None
        else []
    )
    projector_parameter_ids = {id(parameter) for parameter in projector_parameters}
    shared_private_parameters = (
        list(model.shared_residual.parameters())
        if model.shared_residual is not None
        else []
    )
    for branch in model.private_residuals:
        shared_private_parameters.extend(branch.parameters())
    context_moe_parameters = (
        list(model.context_moe.parameters())
        if model.context_moe is not None
        else []
    )
    shared_private_parameters.extend(context_moe_parameters)
    shared_private_parameter_ids = {
        id(parameter) for parameter in shared_private_parameters
    }
    pretraining_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in projector_parameter_ids
        and id(parameter) not in shared_private_parameter_ids
    ]
    optimizer = torch.optim.Adam(
        pretraining_parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = Loss(
        config["batch_size"],
        num_clusters,
        config["temperature_l"],
        config["temperature_f"],
        cluster_occupancy_weighting=config[
            "cluster_occupancy_weighting"
        ],
    ).to(device)

    current_pretraining_code_sha256 = source_bundle_sha256(
        PRETRAINING_SOURCE_FILES
    )
    compatibility_hash = str(
        config.get("pretraining_code_compatibility_hash", "")
    ).strip().upper()
    if compatibility_hash:
        if (
            len(compatibility_hash) != 64
            or any(char not in "0123456789ABCDEF" for char in compatibility_hash)
        ):
            raise ValueError(
                "pretraining_code_compatibility_hash must be 64 hex characters"
            )
        resolved_pretraining_code_sha256 = compatibility_hash
        print(
            "Pretraining cache compatibility override: expected=%s current=%s"
            % (
                resolved_pretraining_code_sha256,
                current_pretraining_code_sha256,
            )
        )
    else:
        resolved_pretraining_code_sha256 = (
            current_pretraining_code_sha256
        )
    config["resolved_pretraining_code_sha256"] = (
        resolved_pretraining_code_sha256
    )

    pretraining_signature = build_pretraining_signature(
        dataset,
        config,
        num_samples,
        num_views,
        input_sizes,
        num_clusters,
        selector_specification,
        dataset_source_sha256(data),
        resolved_pretraining_code_sha256,
    )

    pretrained_path = config.get("pretrained_path")
    if pretrained_path:
        payload = torch.load(
            str(pretrained_path), map_location="cpu", weights_only=False
        )
        validate_pretraining_payload(payload, pretraining_signature)
        incompatible = model.load_state_dict(
            payload["model_state_dict"], strict=False
        )
        allowed_missing = {
            "barycentric_targets",
            "global_view_gate",
            "global_view_quality",
        }
        if model.semantic_projector is not None:
            allowed_missing.update(
                "semantic_projector.%s" % name
                for name, _ in model.semantic_projector.named_parameters()
            )
            allowed_missing.update(
                "semantic_projector.%s" % name
                for name, _ in model.semantic_projector.named_buffers()
            )
        if model.shared_residual is not None:
            allowed_missing.update(
                "shared_residual.%s" % name
                for name, _ in model.shared_residual.named_parameters()
            )
            allowed_missing.update(
                "shared_residual.%s" % name
                for name, _ in model.shared_residual.named_buffers()
            )
        for index, branch in enumerate(model.private_residuals):
            allowed_missing.update(
                "private_residuals.%d.%s" % (index, name)
                for name, _ in branch.named_parameters()
            )
            allowed_missing.update(
                "private_residuals.%d.%s" % (index, name)
                for name, _ in branch.named_buffers()
            )
        if model.context_moe is not None:
            allowed_missing.update(
                "context_moe.%s" % name
                for name, _ in model.context_moe.named_parameters()
            )
            allowed_missing.update(
                "context_moe.%s" % name
                for name, _ in model.context_moe.named_buffers()
            )
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise ValueError(
                "pretrained model state mismatch: missing=%s unexpected=%s"
                % (
                    sorted(unexpected_missing),
                    sorted(incompatible.unexpected_keys),
                )
            )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        move_optimizer_state_to_parameter_devices(optimizer)
        restore_rng_state(payload["rng_state"])
        selection_metadata = payload["pretrain_selection"]
        print("Pretrained encoder loaded from: %s" % pretrained_path)
    else:
        selector = None
        if selection_mode == "cross_view_agreement":
            selector = CrossViewPretrainingSelector(
                data,
                config["batch_size"],
                num_clusters,
                config["mse_epochs"],
                interval=config["pretrain_selection_interval"],
                min_epoch=config["pretrain_selection_min_epoch"],
                n_init=config["pretrain_selection_n_init"],
                kmeans_seed=0,
                max_samples=config["pretrain_selection_max_samples"],
                feature_dim=config["feature_dim"],
                score_tolerance=config["pretrain_selection_tolerance"],
            )
            if selector.specification() != selector_specification:
                raise RuntimeError("pretrain selector specification drift")

        pre_train(
            model,
            data,
            config["batch_size"],
            config["mse_epochs"],
            optimizer,
            epoch_callback=selector,
        )
        if selector is not None:
            selection_metadata = selector.restore_best(model, optimizer)
        else:
            selection_metadata = {
                "mode": "terminal_epoch",
                "selected_epoch": int(config["mse_epochs"]),
                "selected_score": None,
                "selected_pairwise_view_mean": None,
                "specification": selector_specification,
                "history": [],
            }

        save_pretrained_path = config.get("save_pretrained_path")
        if save_pretrained_path:
            path = Path(save_pretrained_path)
            atomic_torch_save(
                {
                    "format_version": 3,
                    "stage": "autoencoder_pretraining",
                    "dataset": dataset,
                    "seed": int(config["seed"]),
                    "pretraining_signature": pretraining_signature,
                    "pretrain_selection": selection_metadata,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": capture_rng_state(),
                },
                path,
            )
            print("Pretrained encoder saved to: %s" % path)

    if bool(config["adaptive_shared_private"]):
        rng_before_capacity_estimate = capture_rng_state()
        try:
            pretrained_view_quality, capacity_diagnostics = (
                estimate_pretrained_view_quality(
                    model,
                    data,
                    config["batch_size"],
                    reliability_mode=config["reliability_mode"],
                )
            )
            capacity_scale, capacity_dominance = (
                adaptive_semantic_projector_scale(
                    pretrained_view_quality,
                    maximum_scale=config["shared_private_scale"],
                    dominance_floor=config[
                        "shared_private_dominance_floor"
                    ],
                    dominance_ceiling=config[
                        "shared_private_dominance_ceiling"
                    ],
                )
            )
            model.shared_private_scale = float(capacity_scale)
            model.shared_private_dominance = float(capacity_dominance)
            model.pretrained_view_quality = (
                pretrained_view_quality.detach().cpu()
            )
            model.pretrained_capacity_diagnostics = capacity_diagnostics
            print(
                "Pretrained adaptive capacity: view_quality=%s "
                "dominance=%.6f shared_private_scale=%.6f"
                % (
                    [
                        round(float(value), 6)
                        for value in pretrained_view_quality.detach().cpu()
                    ],
                    float(capacity_dominance),
                    float(capacity_scale),
                )
            )
        finally:
            restore_rng_state(rng_before_capacity_estimate)

    previous_optimizer = optimizer
    optimizer = contrastive_stage_optimizer(
        previous_optimizer, pretraining_parameters, config
    )
    if optimizer is not previous_optimizer:
        print(
            "Contrastive optimizer: fresh Adam state (reconstruction moments cleared)"
        )
    else:
        print("Contrastive optimizer: inherited reconstruction Adam state")

    if projector_parameters:
        optimizer.add_param_group(
            {
                "params": projector_parameters,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            }
        )
        print(
            "Shared semantic projector: scale=%.6f adaptive=%s "
            "hidden_dim=%d parameters=%d"
            % (
                float(model.semantic_projector_scale),
                bool(config["adaptive_semantic_projector"]),
                int(config["semantic_projector_hidden_dim"]),
                sum(parameter.numel() for parameter in projector_parameters),
            )
        )

    if shared_private_parameters:
        optimizer.add_param_group(
            {
                "params": shared_private_parameters,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            }
        )

    canonical_correlation_parameters = (
        model.initialize_canonical_correlation_heads(
            seed=int(config["seed"]) + 2603
        )
    )
    if canonical_correlation_parameters:
        optimizer.add_param_group(
            {
                "params": canonical_correlation_parameters,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            }
        )
        print(
            "Canonical max-correlation heads: dim=%d parameters=%d"
            % (
                int(model.canonical_correlation_dim),
                sum(
                    parameter.numel()
                    for parameter in canonical_correlation_parameters
                ),
            )
        )

    multilevel_parameters = model.initialize_multilevel_heads(
        seed=int(config["seed"]) + 1701
    )
    if multilevel_parameters:
        optimizer.add_param_group(
            {
                "params": multilevel_parameters,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            }
        )
        print(
            "Multi-level heads: projection_dim=%d parameters=%d"
            % (
                int(config["multilevel_projection_dim"]),
                sum(parameter.numel() for parameter in multilevel_parameters),
            )
        )
        print(
            "Shared-private residual branches: scale=%.6f hidden_dim=%d parameters=%d"
            % (
                float(config["shared_private_scale"]),
                int(config["shared_private_hidden_dim"]),
                sum(parameter.numel() for parameter in shared_private_parameters),
            )
        )

    config["selected_pretrain_epoch"] = int(
        selection_metadata["selected_epoch"]
    )
    config["selected_pretrain_score"] = selection_metadata.get(
        "selected_score"
    )
    config["pretrain_selection_history"] = selection_metadata.get(
        "history", []
    )
    model.normalize_fusion_features = bool(
        config["normalize_fusion_features"]
    )
    print(
        "Fusion geometry: %s"
        % (
            "per-view-l2-normalized-convex-consensus"
            if model.normalize_fusion_features
            else "released-unnormalized-consensus"
        )
    )
    print(
        "Pretrain checkpoint selected: epoch=%d score=%s mode=%s"
        % (
            config["selected_pretrain_epoch"],
            (
                "none"
                if config["selected_pretrain_score"] is None
                else "%.6f" % float(config["selected_pretrain_score"])
            ),
            selection_metadata["mode"],
        )
    )

    pretrained_representations = None
    if str(config["topology_policy_mode"]) == "adaptive_structure":
        rng_before_diagnostics = capture_rng_state()
        was_training = bool(model.training)
        try:
            pretrained_representations = extract_learned_representations(
                model,
                data,
                batch_size=config["batch_size"],
                max_raw_dim=config["feature_dim"],
                seed=0,
                inference_reliability_power=1.0,
            )
            learned_view_names = sorted(
                (
                    name
                    for name in pretrained_representations
                    if name.startswith("learned_view")
                ),
                key=lambda name: int(name.replace("learned_view", "")),
            )
            structure_diagnostics = compute_backend_structure_diagnostics(
                [view.detach().cpu().numpy() for view in data.data_views],
                pretrained_representations["learned_common"],
                [pretrained_representations[name] for name in learned_view_names],
                num_clusters,
                maximum_samples=config["adaptive_structure_max_samples"],
                seed=0,
                n_init=config["adaptive_structure_n_init"],
            )
            route_policy = select_topology_backend(
                num_samples,
                num_views,
                structure_diagnostics["learned_view_agreement"],
                structure_diagnostics[
                    "raw_coassociation_learned_view_agreement"
                ],
                cluster_count=num_clusters,
                raw_coassociation_raw_view_agreement=structure_diagnostics[
                    "raw_coassociation_raw_view_agreement"
                ],
                best_raw_view_learned_view_agreement=structure_diagnostics[
                    "best_raw_view_learned_view_agreement"
                ],
                compact_exact_budget_mb=config[
                    "adaptive_compact_exact_budget_mb"
                ],
                maximum_safe_exact_mb=config[
                    "adaptive_maximum_safe_exact_mb"
                ],
            )
            route_policy["structure_diagnostics"] = structure_diagnostics
        finally:
            model.train(was_training)
            restore_rng_state(rng_before_diagnostics)

        (
            scalable_memory_route,
            effective_feature_scale,
            effective_global_weight,
            effective_global_memory,
            effective_memory_warmup,
            scale_source,
        ) = resolve_topology_training_config(config, route_policy)
        print(
            "Adaptive topology diagnostics: %s"
            % json.dumps(structure_diagnostics, sort_keys=True)
        )
        print_topology_resolution(
            config,
            effective_feature_scale,
            scale_source,
            effective_global_memory,
            effective_global_weight,
            effective_memory_warmup,
        )

    if int(config["structural_anchor_epochs"]) > 0:
        teacher_mode = resolved_structural_teacher_mode(config, route_policy)
        config["resolved_structural_teacher_mode"] = teacher_mode
        if teacher_mode == "learned_consensus":
            if pretrained_representations is None:
                pretrained_representations = extract_learned_representations(
                    model,
                    data,
                    batch_size=config["batch_size"],
                    max_raw_dim=config["feature_dim"],
                    seed=0,
                    inference_reliability_power=1.0,
                )
            anchor = kmeans_clustering(
                pretrained_representations["learned_common"],
                num_clusters,
                seed=0,
                n_init=20,
            )
            selected_view = -1
            operator = "pretrained_consensus_kmeans"
            selector = "%s_learned_teacher" % str(
                route_policy.get("reason", "topology_route")
            )
            diagnostics = []
        else:
            anchor, selected_view, operator, selector, diagnostics = build_structural_anchor(
                data.data_views,
                num_clusters,
                knn_k=config["structural_anchor_k"],
                stability_runs=config["anchor_stability_runs"],
                mode=(
                    "coassociation"
                    if teacher_mode == "raw_coassociation"
                    else "single_view"
                ),
            )
            if scalable_memory_route:
                selector = "scalable_raw_%s" % selector
        config["resolved_teacher_source"] = (
            "learned_consensus"
            if selected_view < 0
            else "raw_view_%d" % (selected_view + 1)
        )
        model.structural_anchor_labels = torch.as_tensor(
            anchor, dtype=torch.long, device=device
        )
        model.structural_anchor_diagnostics = diagnostics
        model.anchor_view_index.fill_(int(selected_view))
        centrality = (
            diagnostics[selected_view].get("centrality")
            if selected_view >= 0 and diagnostics
            else None
        )
        model.anchor_expert_mass.fill_(
            0.0
            if centrality is None
            or config["structural_teacher_mode"] == "coassociation"
            else max(
                0.0,
                min(
                    1.0,
                    float(centrality) * float(config["anchor_expert_scale"]),
                ),
            )
        )
        print(
            "Structural anchor enabled: view=%d selector=%s operator=%s "
            "expert_mass=%.4f scores=%s"
            % (
                selected_view + 1,
                selector,
                operator,
                float(model.anchor_expert_mass.item()),
                [round(item["score"], 4) for item in diagnostics],
            )
        )
        if (
            scalable_memory_route
            and float(config["global_memory_raw_weight"]) > 0.0
        ):
            # A single selected raw view reintroduces view bias and leaves the
            # learned-consensus teacher with no raw prior at all.  Build one
            # deterministic, label-free multiview raw scaffold instead: each
            # view is normalised/PCA-reduced first, then their concatenation is
            # reduced to the memory dimension.
            raw_blocks = [
                structural_representation(
                    view, max_dim=config["feature_dim"]
                )
                for view in data.data_views
            ]
            raw_topology = structural_representation(
                np.concatenate(raw_blocks, axis=1),
                max_dim=config["feature_dim"],
            )
            raw_topology = torch.as_tensor(
                raw_topology, dtype=torch.float32, device=device
            )
            if raw_topology.size(1) < int(config["feature_dim"]):
                padding = torch.zeros(
                    raw_topology.size(0),
                    int(config["feature_dim"]) - raw_topology.size(1),
                    dtype=raw_topology.dtype,
                    device=device,
                )
                raw_topology = torch.cat([raw_topology, padding], dim=1)
            model.raw_topology_bank.copy_(raw_topology)
            model.raw_topology_initialized.fill_(True)
            print(
                "Raw global topology prior: multiview_concat weight=%.3f"
                % float(config["global_memory_raw_weight"])
            )

    if bool(config.get("unified_relation_all_scales", False)):
        raw_relation_views = []
        for view in data.data_views:
            raw_relation = structural_representation(
                view, max_dim=config["feature_dim"]
            )
            raw_relation = torch.as_tensor(
                raw_relation, dtype=torch.float32, device=device
            )
            if raw_relation.size(1) < int(config["feature_dim"]):
                padding = torch.zeros(
                    raw_relation.size(0),
                    int(config["feature_dim"]) - raw_relation.size(1),
                    dtype=raw_relation.dtype,
                    device=device,
                )
                raw_relation = torch.cat([raw_relation, padding], dim=1)
            raw_relation_views.append(raw_relation)
        model.raw_view_relation_bank.copy_(
            torch.stack(raw_relation_views, dim=0)
        )
        model.raw_view_relation_initialized.fill_(True)
        if num_samples <= int(config["raw_exact_candidate_max_samples"]):
            raw_candidate_indices = torch.stack(
                [
                    build_chunked_cosine_knn(
                        raw_relation,
                        config["global_memory_positive_k"],
                        chunk_size=config["global_memory_chunk_size"],
                    )
                    for raw_relation in raw_relation_views
                ],
                dim=0,
            )
            model.raw_view_candidate_indices.copy_(raw_candidate_indices)
            model.raw_view_candidate_initialized.fill_(True)
        print(
            "Unified raw-view relation bank installed: views=%d dim=%d "
            "candidate_search=%s"
            % (
                num_views,
                int(config["feature_dim"]),
                (
                    "exact_chunked_knn"
                    if bool(model.raw_view_candidate_initialized.item())
                    else "cluster_routed_knn"
                ),
            )
        )

    # Every scale now enters the same optimizer loop.  ``graph`` is only a
    # relation provider: compact problems pass exact multi-view neighbours;
    # scalable problems pass ``None`` and consume sparse global-memory
    # neighbours inside the same objective.  This removes the former pair of
    # small/large training functions without changing either validated route.
    graph = None
    if not config["scalable_relation_backend"]:
        graph = (
            get_W(data, k=config["k"])
            if float(effective_feature_scale) > 0.0
            or effective_global_memory
            else None
        )
        if effective_global_memory:
            install_exact_multiview_memory_positives(model, graph)

    for epoch in range(1, config["con_epochs"] + 1):
        contrastive_train(
            model, data, loss_fn, config["batch_size"], epoch, graph,
            config["alpha"], config["beta"],
            config["trust_warmup_epochs"], config["reliability_warmup_epochs"],
            config["trust_alpha"], config["trust_tau"], config["trust_geometry"],
            config["trust_pair_lambda"], config["soft_proto_temperature"],
            config["proto_loss_weight"], optimizer,
            assign_loss_weight=config["assign_loss_weight"],
            assignment_confidence_power=config[
                "assignment_confidence_power"
            ],
            pseudo_teacher_momentum=config["pseudo_teacher_momentum"],
            balance_loss_weight=config["balance_loss_weight"],
            sharpen_loss_weight=config["sharpen_loss_weight"],
            ramp_epochs=config["ramp_epochs"],
            graph_routing=config["graph_routing"],
            adaptive_reliability_routing=config[
                "adaptive_reliability_routing"
            ],
            reliability_routing_dominance_floor=config[
                "reliability_routing_dominance_floor"
            ],
            reliability_routing_dominance_ceiling=config[
                "reliability_routing_dominance_ceiling"
            ],
            adaptive_assignment_confidence=config[
                "adaptive_assignment_confidence"
            ],
            assignment_confidence_max_power=config[
                "assignment_confidence_max_power"
            ],
            structural_anchor_epochs=config["structural_anchor_epochs"],
            reliability_mode=config["reliability_mode"],
            reliability_ema=config["reliability_ema"],
            assignment_temperature=config["assignment_temperature"],
            feature_alignment_scale=effective_feature_scale,
            enable_global_memory=effective_global_memory,
            global_topology_weight=effective_global_weight,
            global_memory_temperature=config["global_memory_temperature"],
            global_memory_warmup_epochs=effective_memory_warmup,
            global_memory_ramp_epochs=config["global_memory_ramp_epochs"],
            global_memory_momentum=config["global_memory_momentum"],
            global_memory_refresh=config["global_memory_refresh"],
            global_memory_positive_k=config["global_memory_positive_k"],
            global_memory_negative_k=config["global_memory_negative_k"],
            global_memory_candidate_clusters=config["global_memory_candidate_clusters"],
            global_memory_chunk_size=config["global_memory_chunk_size"],
            global_memory_raw_weight=config["global_memory_raw_weight"],
            global_relation_mode=config["global_relation_mode"],
            global_relation_consensus_weight=config[
                "global_relation_consensus_weight"
            ],
            global_relation_semantic_floor=config[
                "global_relation_semantic_floor"
            ],
            global_relation_raw_local_weight=config[
                "global_relation_raw_local_weight"
            ],
            global_relation_loss_objective=config[
                "global_relation_loss_objective"
            ],
            global_memory_self_weight=config["global_memory_self_weight"],
            global_memory_query=config["global_memory_query"],
            relation_barycenter_weight=config["relation_barycenter_weight"],
            relation_barycenter_temperature=config[
                "relation_barycenter_temperature"
            ],
            cross_view_correlation_weight=config[
                "cross_view_correlation_weight"
            ],
            cross_view_redundancy_weight=config[
                "cross_view_redundancy_weight"
            ],
            canonical_correlation_weight=config[
                "canonical_correlation_weight"
            ],
            canonical_correlation_regularization=config[
                "canonical_correlation_regularization"
            ],
            instance_contrast_weight=config["instance_contrast_weight"],
            instance_contrast_temperature=config[
                "instance_contrast_temperature"
            ],
            instance_false_negative_floor=config[
                "instance_false_negative_floor"
            ],
            cross_view_reconstruction_weight=config[
                "cross_view_reconstruction_weight"
            ],
            shared_private_orthogonality_weight=config[
                "shared_private_orthogonality_weight"
            ],
            multilevel_instance_weight=config["multilevel_instance_weight"],
            multilevel_cluster_weight=config["multilevel_cluster_weight"],
            multilevel_instance_temperature=config[
                "multilevel_instance_temperature"
            ],
            multilevel_cluster_temperature=config[
                "multilevel_cluster_temperature"
            ],
            use_bgcr_v38=(float(config["cluster_contrast_weight"]) > 0.0),
            reliable_graph_weight=0.0,
            reliable_graph_k=config["reliable_graph_k"],
            reliable_graph_steps=config["reliable_graph_steps"],
            cluster_contrast_weight=config["cluster_contrast_weight"],
            cluster_contrast_temperature=config["cluster_contrast_temperature"],
            barycentric_target_weight=config["barycentric_target_weight"],
            barycentric_assignment_temperature=config[
                "barycentric_assignment_temperature"
            ],
            barycentric_reliability_power=config[
                "barycentric_reliability_power"
            ],
            global_gate_temperature=config["global_gate_temperature"],
            adaptive_semantic_projector=config[
                "adaptive_semantic_projector"
            ],
            semantic_projector_max_scale=config[
                "semantic_projector_scale"
            ],
            semantic_projector_dominance_floor=config[
                "semantic_projector_dominance_floor"
            ],
            semantic_projector_dominance_ceiling=config[
                "semantic_projector_dominance_ceiling"
            ],
            adaptive_shared_private=config["adaptive_shared_private"],
            shared_private_max_scale=config["shared_private_scale"],
            shared_private_dominance_floor=config[
                "shared_private_dominance_floor"
            ],
            shared_private_dominance_ceiling=config[
                "shared_private_dominance_ceiling"
            ],
            normalized_multi_positive=config["normalized_multi_positive"],
            neighbor_aware_batching=config["neighbor_aware_batching"],
            context_moe_alignment_weight=config[
                "context_moe_alignment_weight"
            ],
            context_moe_alignment_temperature=config[
                "context_moe_alignment_temperature"
            ],
        )
        if contrastive_epoch_callback is not None:
            contrastive_epoch_callback(
                model,
                optimizer,
                epoch,
                data,
                config,
            )

    multilevel_pseudo_tune(
        model,
        data,
        config["batch_size"],
        config["multilevel_tune_epochs"],
        optimizer,
        pseudo_n_init=config["multilevel_pseudo_n_init"],
    )

    return model, data


def fixed_route_prediction(model, data, config, dataset):
    """Freeze the publication-route prediction without accessing labels."""
    del dataset
    route = dict(FINAL_PROTOCOL)
    if bool(config["shared_private_readout"]):
        representation = extract_shared_private_clustering_representation(
            model,
            data,
            batch_size=config["batch_size"],
            max_dim=config["feature_dim"],
            seed=0,
        )
        prediction = kmeans_clustering(
            representation, model.num_clusters, seed=0, n_init=200
        )
        route.update(
            {
                "route": "unified-shared-private-kmeans",
                "representation": "mean_shared_plus_all_private",
                "representation_formula": (
                    "concat(mean(shared_views),private_view_1,...,private_view_V)"
                ),
                "recovery_mode": "single_cosine_kmeans",
                "n_init": 200,
                "protocol_hash": str(config.get("protocol_hash", "")),
                "shared_private_scale": float(config["shared_private_scale"]),
                "shared_private_orthogonality_weight": float(
                    config["shared_private_orthogonality_weight"]
                ),
                "context_aware_moe": bool(config["context_aware_moe"]),
                "moe_shared_experts": int(config["moe_shared_experts"]),
                "moe_private_experts": int(config["moe_private_experts"]),
                "moe_top_k": int(config["moe_top_k"]),
                "context_moe_alignment_weight": float(
                    config["context_moe_alignment_weight"]
                ),
                "instance_contrast_weight": float(
                    config["instance_contrast_weight"]
                ),
                "pretrain_selection_mode": str(
                    config.get("resolved_pretrain_selection_mode", "none")
                ),
                "selected_pretrain_epoch": int(
                    config.get("selected_pretrain_epoch", config["mse_epochs"])
                ),
                "selected_pretrain_score": config.get(
                    "selected_pretrain_score"
                ),
            }
        )
        return route, prediction.copy()
    if (
        bool(config["enable_multilevel_heads"])
        and int(config["multilevel_tune_epochs"]) > 0
    ):
        _, probabilities = collect_multilevel_predictions(
            model, data, config["batch_size"]
        )
        mean_probability = torch.stack(probabilities, dim=0).mean(dim=0)
        prediction = mean_probability.argmax(dim=1).numpy().astype(np.int64)
        route.update(
            {
                "route": "unified-multilevel-mean-cluster-head",
                "representation": "mean_multiview_cluster_probability",
                "representation_formula": (
                    "separate-reconstruction-instance-cluster-spaces"
                ),
                "recovery_mode": "mean_cluster_probability_argmax",
                "n_init": 0,
                "protocol_hash": str(config.get("protocol_hash", "")),
                "multilevel_instance_weight": float(
                    config["multilevel_instance_weight"]
                ),
                "multilevel_cluster_weight": float(
                    config["multilevel_cluster_weight"]
                ),
                "multilevel_tune_epochs": int(
                    config["multilevel_tune_epochs"]
                ),
                "pretrain_selection_mode": str(
                    config.get("resolved_pretrain_selection_mode", "none")
                ),
                "selected_pretrain_epoch": int(
                    config.get("selected_pretrain_epoch", config["mse_epochs"])
                ),
                "selected_pretrain_score": config.get(
                    "selected_pretrain_score"
                ),
            }
        )
        return route, prediction.copy()
    route["route"] = "adaptive-smallg-guarded-consensus-kmeans"
    route["inference_reliability_power"] = float(
        config["inference_reliability_power"]
    )
    route["trust_geometry"] = str(config["trust_geometry"])
    route["recovery_mode"] = "guarded_consensus_kmeans"
    route["reliability_ema"] = float(config["reliability_ema"])
    route["graph_routing"] = (
        "adaptive_soft"
        if bool(config["adaptive_reliability_routing"])
        else str(config["graph_routing"])
    )
    route["adaptive_reliability_routing"] = bool(
        config["adaptive_reliability_routing"]
    )
    route["reliability_routing_dominance_floor"] = float(
        config["reliability_routing_dominance_floor"]
    )
    route["reliability_routing_dominance_ceiling"] = float(
        config["reliability_routing_dominance_ceiling"]
    )
    route["reliability_routing_strength"] = float(
        getattr(model, "reliability_routing_strength", 0.0)
    )
    route["reliability_routing_dominance"] = float(
        getattr(model, "reliability_routing_dominance", 0.0)
    )
    route["adaptive_assignment_confidence"] = bool(
        config["adaptive_assignment_confidence"]
    )
    route["effective_assignment_confidence_power"] = float(
        getattr(
            model,
            "effective_assignment_confidence_power",
            config["assignment_confidence_power"],
        )
    )
    route["assignment_confidence_power"] = float(
        config["assignment_confidence_power"]
    )
    route["pseudo_teacher_momentum"] = float(
        config["pseudo_teacher_momentum"]
    )
    route["cluster_occupancy_weighting"] = bool(
        config["cluster_occupancy_weighting"]
    )
    route["semantic_projector_scale"] = float(
        model.semantic_projector_scale
    )
    route["semantic_projector_max_scale"] = float(
        config["semantic_projector_scale"]
    )
    route["adaptive_semantic_projector"] = bool(
        config["adaptive_semantic_projector"]
    )
    route["semantic_projector_dominance"] = float(
        getattr(model, "semantic_projector_dominance", 0.0)
    )
    route["normalize_fusion_features"] = bool(
        getattr(model, "normalize_fusion_features", False)
    )
    route["cross_view_correlation_weight"] = float(
        config["cross_view_correlation_weight"]
    )
    route["canonical_correlation_weight"] = float(
        config["canonical_correlation_weight"]
    )
    route["canonical_correlation_dim"] = int(
        min(
            config["canonical_correlation_dim"],
            model.feature_dim,
            model.num_clusters,
        )
    )
    route["canonical_correlation_regularization"] = float(
        config["canonical_correlation_regularization"]
    )
    route["instance_contrast_weight"] = float(
        config["instance_contrast_weight"]
    )
    route["cross_view_reconstruction_weight"] = float(
        config["cross_view_reconstruction_weight"]
    )
    route["normalized_multi_positive"] = bool(
        config["normalized_multi_positive"]
    )
    route["neighbor_aware_batching"] = bool(
        config["neighbor_aware_batching"]
    )
    route["shared_private_scale"] = float(model.shared_private_scale)
    route["shared_private_max_scale"] = float(config["shared_private_scale"])
    route["adaptive_shared_private"] = bool(
        config["adaptive_shared_private"]
    )
    route["shared_private_dominance"] = float(
        getattr(model, "shared_private_dominance", 0.0)
    )
    route["shared_private_orthogonality_weight"] = float(
        config["shared_private_orthogonality_weight"]
    )
    route["context_aware_moe"] = bool(config["context_aware_moe"])
    route["moe_shared_experts"] = int(config["moe_shared_experts"])
    route["moe_private_experts"] = int(config["moe_private_experts"])
    route["moe_top_k"] = int(config["moe_top_k"])
    route["context_moe_alignment_weight"] = float(
        config["context_moe_alignment_weight"]
    )
    route["multilevel_instance_weight"] = float(
        config["multilevel_instance_weight"]
    )
    route["multilevel_cluster_weight"] = float(
        config["multilevel_cluster_weight"]
    )
    route["global_gate_temperature"] = float(
        config["global_gate_temperature"]
    )
    route["inference_global_gate_temperature"] = float(
        config["inference_global_gate_temperature"]
    )
    # Evaluate the experts before labels are read.  The same label-free guard
    # either protects the uniform consensus or continuously sharpens it toward
    # a demonstrably stable, dominant expert.
    model.global_view_gate.fill_(1.0 / float(model.num_views))
    ungated_representations = extract_learned_representations(
        model,
        data,
        batch_size=config["batch_size"],
        max_raw_dim=256,
        seed=0,
        inference_reliability_power=route["inference_reliability_power"],
    )
    num_clusters = int(model.num_clusters)
    if bool(config["consensus_guard"]):
        gate, guard = consensus_guarded_gate(
            ungated_representations,
            model.num_clusters,
            model.num_views,
            max_samples=route["guard_max_samples"],
            stability_runs=route["guard_stability_runs"],
            dominance_margin=route["guard_dominance_margin"],
            dominance_scale=route["guard_dominance_scale"],
            minimum_quality=route["guard_minimum_quality"],
            quality_scale=route["guard_quality_scale"],
        )
        gate_tensor = torch.as_tensor(
            gate,
            device=model.global_view_gate.device,
            dtype=model.global_view_gate.dtype,
        )
        model.global_view_gate.copy_(gate_tensor)
        qualities = torch.as_tensor(
            [item["quality"] for item in guard["views"]],
            device=model.global_view_quality.device,
            dtype=model.global_view_quality.dtype,
        ).clamp_min(1e-8)
        model.global_view_quality.copy_(qualities / qualities.sum())
        route["gate_rule"] = "label-free-consensus-guard-v1"
        route["consensus_guard"] = json.dumps(guard, sort_keys=True)
        route["global_view_gate"] = json.dumps(
            model.global_view_gate.detach().cpu().tolist()
        )
        route["global_view_quality"] = json.dumps(
            model.global_view_quality.detach().cpu().tolist()
        )
        route["protocol_hash"] = str(config.get("protocol_hash", ""))
        guarded_representations = extract_learned_representations(
            model,
            data,
            batch_size=config["batch_size"],
            max_raw_dim=256,
            seed=0,
            inference_reliability_power=route["inference_reliability_power"],
        )
        representation, effective_residual, sufficiency_deficit = (
            guarded_complementary_consensus(
                guarded_representations,
                gate_strength=float(guard["gate_strength"]),
                consensus_quality=float(guard["consensus"]["quality"]),
                view_count=model.num_views,
                residual_weight=float(route["residual_weight"]),
                activation_quality=float(route["residual_activation_quality"]),
                activation_scale=float(route["residual_activation_scale"]),
                max_dim=256,
                seed=int(route["seed"]),
            )
        )
        route["representation"] = "guarded_complementary_consensus"
        route["representation_formula"] = (
            "adaptive-smallg-common-plus-label-free-symmetric-residual"
        )
    else:
        # Ablation baseline: keep the uniform 1/V gate and cluster the plain
        # reliability-weighted consensus without adaptive small-g or residual.
        route["gate_rule"] = "uniform-consensus-baseline"
        route["consensus_guard"] = ""
        route["global_view_gate"] = json.dumps(
            model.global_view_gate.detach().cpu().tolist()
        )
        route["global_view_quality"] = json.dumps(
            model.global_view_quality.detach().cpu().tolist()
        )
        route["protocol_hash"] = str(config.get("protocol_hash", ""))
        representation = ungated_representations["learned_common"]
        effective_residual = 0.0
        sufficiency_deficit = 0.0
        route["representation"] = "uniform_consensus_only"
        route["representation_formula"] = "uniform-consensus-only"
    forced_n_init = os.environ.get("SACR_FORCE_FINAL_N_INIT", "").strip()
    route["n_init"] = int(forced_n_init) if forced_n_init else (20 if effective_residual > 0.0 else 200)
    if route["n_init"] < 1:
        raise ValueError("final n_init must be positive")
    pred = kmeans_clustering(
        representation,
        num_clusters,
        seed=int(route["seed"]),
        n_init=int(route["n_init"]),
    )
    route["partition_diagnostics"] = ""
    route["effective_residual_weight"] = float(effective_residual)
    route["consensus_sufficiency_deficit"] = float(sufficiency_deficit)
    route["anchor_view"] = int(model.anchor_view_index.item()) + 1
    route["anchor_expert_mass"] = float(model.anchor_expert_mass.item())
    route["feature_alignment_scale"] = float(
        config.get("resolved_feature_alignment_scale", config["feature_alignment_scale"])
    )
    route["global_memory"] = bool(
        config.get("resolved_enable_global_memory", config["enable_global_memory"])
    )
    route["global_topology_weight"] = float(
        config.get("resolved_global_topology_weight", config["global_topology_weight"])
    )
    route["local_relation_weight"] = float(
        config.get("resolved_local_relation_weight", 0.0)
    )
    route["global_relation_weight"] = float(
        config.get("resolved_global_relation_weight", 0.0)
    )
    route["local_global_mix"] = float(
        config.get("resolved_local_global_mix", 0.0)
    )
    route["topology_route"] = str(config.get("topology_route", ""))
    route["estimated_exact_graph_mb"] = float(
        config.get("estimated_exact_graph_mb", 0.0)
    )
    route["exact_graph_budget_mb"] = float(
        config.get("exact_graph_budget_mb", 0.0)
    )
    route["topology_policy_mode"] = str(
        config.get("topology_policy_mode", "resource")
    )
    route["topology_selector_reason"] = str(
        config.get("topology_selector_reason", "resource_budget")
    )
    route["topology_selector_diagnostics"] = json.dumps(
        config.get("topology_selector_diagnostics", {}), sort_keys=True
    )
    route["pretrain_selection_mode"] = str(
        config.get("resolved_pretrain_selection_mode", "none")
    )
    route["selected_pretrain_epoch"] = int(
        config.get("selected_pretrain_epoch", config["mse_epochs"])
    )
    route["selected_pretrain_score"] = config.get("selected_pretrain_score")
    route["teacher_source"] = str(
        config.get("resolved_teacher_source", "none")
    )
    route["global_memory_query"] = str(config["global_memory_query"])
    route["reset_contrastive_optimizer"] = bool(
        config["reset_contrastive_optimizer"]
    )

    return route, np.asarray(pred).copy()


def fixed_route_clustering(model, data, config, dataset):
    """Evaluate the one publication route after its predictions are fixed."""
    route, pred = fixed_route_prediction(model, data, config, dataset)
    # Ground-truth assignments are accessed only after predictions are fixed.
    route["prediction_sha256"] = hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()
    labels = np.asarray(data.labels).reshape(-1)
    nmi, ari, acc, pur, f_score = evaluate(labels.copy(), pred.copy())

    metrics = {
        "ACC": float(acc),
        "NMI": float(nmi),
        "PUR": float(pur),
        "ARI": float(ari),
        "Fscore": float(f_score),
    }

    return route, metrics


def append_summary(path, dataset, route, metrics, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "method": "SACR-MVC",
        "seed": int(seed),
        "route": route["route"],
        "protocol_hash": route.get("protocol_hash", ""),
        "representation": route.get("representation", ""),
        "representation_formula": route.get("representation_formula", ""),
        "gate_rule": route.get("gate_rule", ""),
        "consensus_guard": route.get("consensus_guard", ""),
        "partition_diagnostics": route.get("partition_diagnostics", ""),
        "recovery_mode": route.get("recovery_mode", ""),
        "kmeans_n_init": route.get("n_init", ""),
        "trust_geometry": route.get("trust_geometry", ""),
        "anchor_view": route.get("anchor_view", ""),
        "anchor_expert_mass": f"{route.get('anchor_expert_mass', 0.0):.4f}",
        "feature_alignment_scale": f"{route.get('feature_alignment_scale', 0.0):.6f}",
        "global_memory": str(bool(route.get("global_memory", False))),
        "global_topology_weight": f"{route.get('global_topology_weight', 0.0):.6f}",
        "local_relation_weight": f"{route.get('local_relation_weight', 0.0):.6f}",
        "global_relation_weight": f"{route.get('global_relation_weight', 0.0):.6f}",
        "local_global_mix": f"{route.get('local_global_mix', 0.0):.6f}",
        "topology_route": route.get("topology_route", ""),
        "estimated_exact_graph_mb": f"{route.get('estimated_exact_graph_mb', 0.0):.3f}",
        "exact_graph_budget_mb": f"{route.get('exact_graph_budget_mb', 0.0):.3f}",
        "topology_policy_mode": route.get("topology_policy_mode", ""),
        "topology_selector_reason": route.get("topology_selector_reason", ""),
        "topology_selector_diagnostics": route.get(
            "topology_selector_diagnostics", ""
        ),
        "pretrain_selection_mode": route.get("pretrain_selection_mode", ""),
        "selected_pretrain_epoch": route.get("selected_pretrain_epoch", ""),
        "selected_pretrain_score": (
            ""
            if route.get("selected_pretrain_score") is None
            else f"{float(route['selected_pretrain_score']):.6f}"
        ),
        "teacher_source": route.get("teacher_source", ""),
        "global_memory_query": route.get("global_memory_query", ""),
        "reset_contrastive_optimizer": str(
            bool(route.get("reset_contrastive_optimizer", False))
        ),
        "reliability_ema": f"{route.get('reliability_ema', 0.0):.4f}",
        "graph_routing": route.get("graph_routing", "none"),
        "adaptive_reliability_routing": str(
            bool(route.get("adaptive_reliability_routing", False))
        ),
        "reliability_routing_dominance_floor": (
            f"{float(route.get('reliability_routing_dominance_floor', 0.0)):.6f}"
        ),
        "reliability_routing_dominance_ceiling": (
            f"{float(route.get('reliability_routing_dominance_ceiling', 0.0)):.6f}"
        ),
        "reliability_routing_strength": (
            f"{float(route.get('reliability_routing_strength', 0.0)):.6f}"
        ),
        "reliability_routing_dominance": (
            f"{float(route.get('reliability_routing_dominance', 0.0)):.6f}"
        ),
        "adaptive_assignment_confidence": str(
            bool(route.get("adaptive_assignment_confidence", False))
        ),
        "effective_assignment_confidence_power": (
            f"{float(route.get('effective_assignment_confidence_power', 0.0)):.6f}"
        ),
        "assignment_confidence_power": (
            f"{float(route.get('assignment_confidence_power', 0.0)):.4f}"
        ),
        "pseudo_teacher_momentum": (
            f"{float(route.get('pseudo_teacher_momentum', 0.0)):.4f}"
        ),
        "cluster_occupancy_weighting": str(
            bool(route.get("cluster_occupancy_weighting", False))
        ),
        "semantic_projector_scale": (
            f"{float(route.get('semantic_projector_scale', 0.0)):.6f}"
        ),
        "semantic_projector_max_scale": (
            f"{float(route.get('semantic_projector_max_scale', 0.0)):.6f}"
        ),
        "adaptive_semantic_projector": str(
            bool(route.get("adaptive_semantic_projector", False))
        ),
        "semantic_projector_dominance": (
            f"{float(route.get('semantic_projector_dominance', 0.0)):.6f}"
        ),
        "normalize_fusion_features": str(
            bool(route.get("normalize_fusion_features", False))
        ),
        "cross_view_correlation_weight": (
            f"{float(route.get('cross_view_correlation_weight', 0.0)):.6f}"
        ),
        "canonical_correlation_weight": (
            f"{float(route.get('canonical_correlation_weight', 0.0)):.6f}"
        ),
        "canonical_correlation_dim": int(
            route.get("canonical_correlation_dim", 0)
        ),
        "canonical_correlation_regularization": (
            f"{float(route.get('canonical_correlation_regularization', 0.0)):.8g}"
        ),
        "instance_contrast_weight": (
            f"{float(route.get('instance_contrast_weight', 0.0)):.6f}"
        ),
        "cross_view_reconstruction_weight": (
            f"{float(route.get('cross_view_reconstruction_weight', 0.0)):.6f}"
        ),
        "normalized_multi_positive": str(
            bool(route.get("normalized_multi_positive", False))
        ),
        "neighbor_aware_batching": str(
            bool(route.get("neighbor_aware_batching", False))
        ),
        "shared_private_scale": (
            f"{float(route.get('shared_private_scale', 0.0)):.6f}"
        ),
        "shared_private_max_scale": (
            f"{float(route.get('shared_private_max_scale', 0.0)):.6f}"
        ),
        "adaptive_shared_private": str(
            bool(route.get("adaptive_shared_private", False))
        ),
        "shared_private_dominance": (
            f"{float(route.get('shared_private_dominance', 0.0)):.6f}"
        ),
        "shared_private_orthogonality_weight": (
            f"{float(route.get('shared_private_orthogonality_weight', 0.0)):.6f}"
        ),
        "context_aware_moe": str(
            bool(route.get("context_aware_moe", False))
        ),
        "moe_shared_experts": int(route.get("moe_shared_experts", 0)),
        "moe_private_experts": int(route.get("moe_private_experts", 0)),
        "moe_top_k": int(route.get("moe_top_k", 0)),
        "context_moe_alignment_weight": (
            f"{float(route.get('context_moe_alignment_weight', 0.0)):.6f}"
        ),
        "multilevel_instance_weight": (
            f"{float(route.get('multilevel_instance_weight', 0.0)):.6f}"
        ),
        "multilevel_cluster_weight": (
            f"{float(route.get('multilevel_cluster_weight', 0.0)):.6f}"
        ),
        "multilevel_tune_epochs": int(
            route.get("multilevel_tune_epochs", 0)
        ),
        "global_gate_temperature": (
            f"{float(route.get('global_gate_temperature', 1e6)):.8g}"
        ),
        "inference_global_gate_temperature": (
            f"{float(route.get('inference_global_gate_temperature', 1e6)):.8g}"
        ),
        "inference_reliability_power": (
            f"{float(route.get('inference_reliability_power', 1.0)):.6f}"
        ),
        "global_view_gate": route.get("global_view_gate", ""),
        "global_view_quality": route.get("global_view_quality", ""),
        "ACC": f"{metrics['ACC']:.4f}",
        "NMI": f"{metrics['NMI']:.4f}",
        "PUR": f"{metrics['PUR']:.4f}",
        "ARI": f"{metrics['ARI']:.4f}",
        "Fscore": f"{metrics['Fscore']:.4f}",
    }

    fields = list(row.keys())
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    config = validate_config(
        apply_overrides(dataset_config(dataset), args)
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    route = dict(FINAL_PROTOCOL)

    print("=" * 80)
    print("SACR-MVC UNIFIED RUN")
    print(f"Dataset      : {dataset}")
    print(f"Device       : {device}")
    print(f"Fixed route  : {route['route']}")
    print(
        "Recovery     : adaptive small-g + guarded residual + "
        "KMeans(seed=0, n_init=%d or 20 when residual is active)"
        % int(route["n_init"])
    )
    print("Evaluation   : labels are accessed only after predictions are fixed")
    print("Cluster count: K is treated as known benchmark metadata")
    print("Configuration: %s" % json.dumps(config, sort_keys=True))
    print("=" * 80)

    model, data = train_model(dataset, config, device)

    print("=" * 80)
    print("FINAL CLUSTERING")
    print("Protocol     : one fixed label-free route, no candidate search")
    print(f"Route        : {route['route']}")
    print("=" * 80)

    route, metrics = fixed_route_clustering(model, data, config, dataset)

    print("=" * 80)
    print("FINAL RESULT")
    print(f"Dataset      : {dataset}")
    print(f"Method       : SACR-MVC")
    print(f"Route        : {route['route']}")
    print(
        "ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F-score = {:.4f}".format(
            metrics["ACC"],
            metrics["NMI"],
            metrics["PUR"],
            metrics["ARI"],
            metrics["Fscore"],
        )
    )
    print("=" * 80)

    if str2bool(args.write_summary):
        append_summary(args.summary_path, dataset, route, metrics, config["seed"])
        print(f"Summary saved to: {args.summary_path}")
    payload = {"status": "completed", "protocol_version": "frozen_20260923_v1",
               "dataset": dataset, "seed": config["seed"], "metrics": metrics,
               "configuration": config, "route": route,
               "confidence_mode": os.environ.get("SACR_CONFIDENCE_MODE", "none"),
               "source_paths": list(data.source_paths)}
    print("RESULT_JSON " + json.dumps(payload, default=str, ensure_ascii=False), flush=True)


    if str2bool(args.save_model):
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{dataset}.pth"
        torch.save(
            {
                "format_version": 1,
                "method": "SACR-MVC",
                "dataset": dataset,
                "model_state_dict": model.state_dict(),
                "config": config,
                "final_protocol": route,
                "metrics": metrics,
            },
            str(model_path),
        )
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
