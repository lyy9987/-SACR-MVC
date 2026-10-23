# -*- coding: utf-8 -*-
"""Label-free checkpoint selection for multi-view reconstruction pretraining."""

import copy
import random

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from unified_recovery import (
    extract_learned_representations,
    kmeans_clustering,
)


SELECTOR_VERSION = "cross_view_ari_v2"


def capture_rng_state():
    """Capture every RNG that can affect the subsequent training trajectory."""
    numpy_state = np.random.get_state()
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state().cpu().clone(),
        "cuda": (
            [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def clone_to_cpu(value):
    """Recursively detach tensor state so a best checkpoint is not a live view."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def move_optimizer_state_to_parameter_devices(optimizer):
    def move(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item, device) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state, parameter.device)


def deterministic_subset(num_samples, max_samples, seed=0):
    count = int(num_samples)
    maximum = max(1, int(max_samples))
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    generator = np.random.RandomState(int(seed))
    return np.sort(generator.choice(count, size=maximum, replace=False)).astype(
        np.int64
    )


def cross_view_partition_score(
    representations,
    num_clusters,
    n_init=5,
    kmeans_seed=0,
    max_samples=5000,
):
    """Compare only predicted partitions; no ground-truth labels are accepted."""
    common = np.asarray(representations["learned_common"])
    view_names = sorted(
        (name for name in representations if name.startswith("learned_view")),
        key=lambda name: int(name.replace("learned_view", "")),
    )
    if not view_names:
        raise ValueError("at least one learned view representation is required")
    if not np.isfinite(common).all():
        raise ValueError("learned consensus contains non-finite values")

    sample_indices = deterministic_subset(
        common.shape[0], max_samples=max_samples, seed=kmeans_seed
    )
    common_sample = common[sample_indices]
    if common_sample.shape[0] < int(num_clusters):
        raise ValueError("selector subset is smaller than the cluster count")

    common_partition = kmeans_clustering(
        common_sample,
        num_clusters,
        seed=kmeans_seed,
        n_init=n_init,
    )
    if np.unique(common_partition).size != int(num_clusters):
        raise ValueError("consensus partition contains empty clusters")

    view_partitions = []
    consensus_agreements = []
    for name in view_names:
        view = np.asarray(representations[name])
        if view.shape[0] != common.shape[0] or not np.isfinite(view).all():
            raise ValueError("invalid learned view representation: %s" % name)
        partition = kmeans_clustering(
            view[sample_indices],
            num_clusters,
            seed=kmeans_seed,
            n_init=n_init,
        )
        if np.unique(partition).size != int(num_clusters):
            raise ValueError("%s partition contains empty clusters" % name)
        view_partitions.append(partition)
        consensus_agreements.append(
            float(adjusted_rand_score(common_partition, partition))
        )

    pairwise_agreements = [
        float(adjusted_rand_score(view_partitions[first], view_partitions[second]))
        for first in range(len(view_partitions))
        for second in range(first + 1, len(view_partitions))
    ]
    counts = np.bincount(common_partition, minlength=int(num_clusters)).astype(
        np.float64
    )
    probabilities = counts[counts > 0] / max(1.0, counts.sum())
    normalized_entropy = float(
        -(probabilities * np.log(probabilities)).sum()
        / max(np.log(max(2, int(num_clusters))), 1e-12)
    )
    return {
        "score": float(np.mean(consensus_agreements)),
        "consensus_to_views": consensus_agreements,
        "pairwise_view_mean": (
            float(np.mean(pairwise_agreements)) if pairwise_agreements else 1.0
        ),
        "pairwise_views": pairwise_agreements,
        "partition_entropy": normalized_entropy,
        "sample_count": int(sample_indices.size),
    }


class CrossViewPretrainingSelector:
    """Select a reconstruction checkpoint by label-free partition agreement."""

    def __init__(
        self,
        dataset,
        batch_size,
        num_clusters,
        total_epochs,
        interval=50,
        min_epoch=50,
        n_init=5,
        kmeans_seed=0,
        max_samples=5000,
        feature_dim=256,
        score_tolerance=0.0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_clusters = int(num_clusters)
        self.total_epochs = int(total_epochs)
        self.interval = max(1, int(interval))
        self.min_epoch = max(1, int(min_epoch))
        self.n_init = max(1, int(n_init))
        self.kmeans_seed = int(kmeans_seed)
        self.max_samples = max(1, int(max_samples))
        self.feature_dim = int(feature_dim)
        self.score_tolerance = max(0.0, float(score_tolerance))
        self.history = []
        self._best = None
        self._best_observed_score = -float("inf")

    def specification(self):
        return {
            "version": SELECTOR_VERSION,
            "metric": "mean_consensus_to_view_adjusted_rand",
            "interval": self.interval,
            "min_epoch": self.min_epoch,
            "n_init": self.n_init,
            "kmeans_seed": self.kmeans_seed,
            "max_samples": self.max_samples,
            "pretrain_budget_epochs": self.total_epochs,
            "score_tolerance": self.score_tolerance,
            "tie_break": "latest_epoch_within_tolerance",
        }

    def should_evaluate(self, epoch):
        epoch = int(epoch)
        return epoch == self.total_epochs or (
            epoch >= self.min_epoch and epoch % self.interval == 0
        )

    def __call__(self, model, optimizer, epoch, total_loss):
        if not self.should_evaluate(epoch):
            return

        rng_before_evaluation = capture_rng_state()
        was_training = bool(model.training)
        try:
            representations = extract_learned_representations(
                model,
                self.dataset,
                batch_size=self.batch_size,
                max_raw_dim=self.feature_dim,
                seed=self.kmeans_seed,
                inference_reliability_power=1.0,
            )
            diagnostics = cross_view_partition_score(
                representations,
                self.num_clusters,
                n_init=self.n_init,
                kmeans_seed=self.kmeans_seed,
                max_samples=self.max_samples,
            )
            record = {
                "epoch": int(epoch),
                "reconstruction_loss": float(total_loss),
                "valid": True,
                **diagnostics,
            }
        except Exception as error:
            record = {
                "epoch": int(epoch),
                "reconstruction_loss": float(total_loss),
                "valid": False,
                "error": "%s: %s" % (type(error).__name__, error),
            }
        finally:
            model.train(was_training)
            restore_rng_state(rng_before_evaluation)

        self.history.append(record)
        if not record["valid"]:
            print(
                "Pretrain selector epoch=%d INVALID error=%s"
                % (int(epoch), record["error"])
            )
            return

        print(
            "Pretrain selector epoch=%d score=%.6f pair=%.6f entropy=%.6f "
            "views=%s samples=%d"
            % (
                int(epoch),
                record["score"],
                record["pairwise_view_mean"],
                record["partition_entropy"],
                [round(value, 6) for value in record["consensus_to_views"]],
                record["sample_count"],
            )
        )

        if record["score"] > self._best_observed_score:
            self._best_observed_score = float(record["score"])
        # Cross-view ARI estimates from a fixed subsample can differ by a few
        # thousandths.  Within a predeclared tolerance, prefer the later
        # checkpoint so a negligible agreement fluctuation does not cause
        # aggressive reconstruction under-training.
        eligible = (
            record["score"]
            >= self._best_observed_score - self.score_tolerance - 1e-12
        )
        if eligible:
            self._best = {
                "epoch": int(epoch),
                "score": float(record["score"]),
                "pairwise_view_mean": float(record["pairwise_view_mean"]),
                "best_observed_score": float(self._best_observed_score),
                "model_state_dict": clone_to_cpu(model.state_dict()),
                "optimizer_state_dict": clone_to_cpu(optimizer.state_dict()),
                "rng_state": clone_to_cpu(rng_before_evaluation),
            }

    def restore_best(self, model, optimizer):
        if self._best is None:
            raise RuntimeError(
                "all label-free pretraining checkpoint candidates were invalid"
            )
        model.load_state_dict(self._best["model_state_dict"], strict=True)
        optimizer.load_state_dict(self._best["optimizer_state_dict"])
        move_optimizer_state_to_parameter_devices(optimizer)
        restore_rng_state(self._best["rng_state"])
        model.train()
        metadata = {
            "mode": "cross_view_agreement",
            "selected_epoch": int(self._best["epoch"]),
            "selected_score": float(self._best["score"]),
            "selected_pairwise_view_mean": float(
                self._best["pairwise_view_mean"]
            ),
            "best_observed_score": float(self._best["best_observed_score"]),
            "specification": self.specification(),
            "history": copy.deepcopy(self.history),
        }
        self._best = None
        return metadata
