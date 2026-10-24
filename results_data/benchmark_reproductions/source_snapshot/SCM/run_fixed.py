"""One fixed SCM/IJCAI'24 transfer protocol across all six frozen inputs.

Original network, loss, SCM() optimization function and final valid(eval_z)
are executed. Adapter changes are recorded before fitting. No score-dependent
epoch, seed, hyperparameter or readout selection is performed.
"""
from pathlib import Path
import argparse
import csv
import difflib
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
import types

for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "2"

import numpy as np
import scipy
import scipy.io
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
from torch.utils.data import Dataset
from threadpoolctl import threadpool_limits

ROOT = Path(r"F:\SACR_MVC_project")
HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / "official"
DATASETS = {
    "RGB-D": ("RGB-D.mat", 1449, [2048, 300], 13),
    "ALOI-100": ("ALOI-100.mat", 10800, [77, 13, 64, 125], 100),
    "Hdigit": ("Hdigit.mat", 10000, [784, 256], 10),
    "Mfeat": ("Mfeat.mat", 2000, [216, 76, 64, 6, 240, 47], 10),
    "CIFAR10": ("cifar10.mat", 50000, [512, 2048, 1024], 10),
    "CIFAR100": ("cifar100.mat", 50000, [512, 2048, 1024], 100),
}
PROTOCOL = {
    "method": "SCM", "variant": "SCM (known K); no REC/ETC variants",
    "publication_year": 2024, "venue": "IJCAI",
    "official_repository": "https://github.com/SubmissionsIn/SCM",
    "official_commit": "f23dbc25a7dda62651f12d6cba53f53aa72548dc",
    "training_seed": 1, "training_runs": 1, "iterations": 1000,
    "iteration_definition": "One first random minibatch from a freshly shuffled DataLoader, exactly as official SCM(); not one full epoch",
    "batch_size": 256, "drop_last": True, "learning_rate": 0.0003,
    "weight_decay": 0.0, "optimizer": "Adam", "temperature_f": 0.5,
    "feature_dim": 256, "high_feature_dim": 128,
    "miss_rate": 0.25, "noise_rate": 0.25, "Gaussian_noise": 0.4,
    "augmentation": "Original native missing-view mask and Gaussian perturbation, used for SCM training only; complete frozen views retained",
    "transfer_budget": "1000 optimization iterations from the public MNIST-USPS preset, fixed uniformly before any target score. Not a claim that the paper prescribes 1000 for all datasets.",
    "target_preprocessing": "float32, view-level global min-max to [0,1], all views and all rows in original order; follows the author's published scale_normalize_matrix helper with a constant-view zero guard",
    "readout": "Original metric.valid(eval_z=True), final normalized z, known-K squared-Euclidean KMeans",
    "KMeans_n_init": 10, "KMeans_random_state": 1, "KMeans_max_iter": 300,
    "KMeans_tol": 1e-4, "KMeans_init": "k-means++", "KMeans_algorithm": "lloyd",
    "readout_note": "n_init=10 fixes the effective sklearn1.3 default explicitly; random_state fixes the otherwise unspecified RNG. No repeated label-based readout selection.",
    "NMI_normalization": "arithmetic", "checkpoint": "final only",
    "four_metrics_same_saved_prediction": True,
    "all_six_results_retained_regardless_of_rank": True,
    "post_score_tuning_or_seed_retries": False,
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def arr_sha(a, dtype):
    return hashlib.sha256(np.asarray(a, dtype=dtype).tobytes(order="C")).hexdigest()

def independent_metrics(y, p):
    _, yy = np.unique(y, return_inverse=True)
    _, pp = np.unique(p, return_inverse=True)
    table = np.zeros((int(yy.max()) + 1, int(pp.max()) + 1), dtype=np.int64)
    np.add.at(table, (yy, pp), 1)
    r, c = linear_sum_assignment(table.max() - table)
    return dict(ACC=float(table[r, c].sum() / len(y)),
                NMI=float(normalized_mutual_info_score(y, p, average_method="arithmetic")),
                PUR=float(table.max(axis=0).sum() / len(y)),
                ARI=float(adjusted_rand_score(y, p)))

class FrozenViews(Dataset):
    def __init__(self, dataset):
        name, n, dims, classes = DATASETS[dataset]
        source = ROOT / "release/frozen_20260923_v1/data" / name
        start = time.perf_counter()
        raw = scipy.io.loadmat(source)
        field = "X" if "X" in raw else "data"
        label_field = "Y" if "Y" in raw else "truelabel"
        label = raw[label_field]
        while label.dtype == object:
            label = label.ravel()[0]
        label = np.asarray(label).reshape(-1)
        _, self.y = np.unique(label, return_inverse=True)
        self.y = self.y.astype(np.int64)
        assert len(self.y) == n and len(np.unique(self.y)) == classes
        cells = raw[field].reshape(-1)
        assert len(cells) == len(dims)
        self.views, view_records = [], []
        for v, d in enumerate(dims):
            matrix = cells[v]
            if matrix.shape == (d, n):
                matrix = matrix.T
            assert matrix.shape == (n, d), (dataset, v, matrix.shape)
            raw_hash = arr_sha(matrix, "<f8")
            x = np.array(matrix, dtype=np.float32, order="C", copy=True)
            lo, hi = float(x.min()), float(x.max())
            if hi > lo:
                x -= lo
                x /= hi - lo
            else:
                x.fill(0)
            assert np.isfinite(x).all()
            cells[v] = None
            del matrix
            self.views.append(x)
            view_records.append(dict(view=v, shape=[n, d], raw_ordered_float64_sha256=raw_hash,
                                     transformed_float32_sha256=arr_sha(x, "<f4"),
                                     global_min=lo, global_max=hi))
        del raw, cells
        gc.collect()
        self.manifest = dict(dataset=dataset, input_file=str(source), input_sha256=sha(source),
                             N=n, V=len(dims), K=classes, dimensions=dims,
                             labels_ordered_sha256=arr_sha(self.y, "<i8"),
                             sample_order_sha256=arr_sha(np.arange(n), "<i8"),
                             views=view_records, loading_seconds=time.perf_counter() - start)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return [torch.from_numpy(x[i]) for x in self.views], self.y[i], torch.tensor(i, dtype=torch.long)

def patched_source():
    original = (OFFICIAL / "train.py").read_text(encoding="utf-8")
    current = original
    changes = [
        ("from dataloader import load_data", "from target_scm_loader import load_data"),
        ('parser.add_argument("--con_iterations", default=50)', 'parser.add_argument("--con_iterations", default=50, type=int)'),
        ("from TSNE import TSNE_PLOT as ttsne", "# Unused optional TSNE visualization import omitted by adapter"),
        ("tsne = True  # True / False", "tsne = False  # No optional visualization during fixed comparison"),
        ("args = parser.parse_args()", "args = parser.parse_args()\n# Unused ETC-only function signature defaults for new datasets; SCM does not use these.\nargs.gamma, args.alpha, args.beta = 0.02, 0.5, 0.5\nseed = 1"),
    ]
    for old, new in changes:
        assert current.count(old) == 1, old
        current = current.replace(old, new, 1)
    return original, current

def main(dataset):
    target = HERE / "runs" / dataset.replace("-", "_")
    target.mkdir(parents=True, exist_ok=True)
    assert not (target / "result.json").exists(), "Never overwrite a completed run or retry based on score"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), "Fixed GPU run expected; don't silently change execution"
    original, source = patched_source()
    (target / "executed_train.py").write_text(source, encoding="utf-8")
    (target / "adapter.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile="official/train.py", tofile="executed_train.py")), encoding="utf-8")
    config = dict(PROTOCOL, dataset=dataset, python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), runner_sha256=sha(__file__),
                  official_files={p.name: sha(p) for p in OFFICIAL.glob("*.py")},
                  executed_train_sha256=sha(target / "executed_train.py"))
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(dict(phase="predeclared_before_fit", dataset=dataset, protocol=PROTOCOL)), flush=True)
    frozen = FrozenViews(dataset)
    (target / "input_manifest.json").write_text(json.dumps(frozen.manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(phase="loaded_frozen_input", **frozen.manifest)), flush=True)
    loader_module = types.ModuleType("target_scm_loader")
    def load_data(_name):
        _, n, dims, k = DATASETS[dataset]
        return frozen, None, len(dims), n, k, sum(dims)
    loader_module.load_data = load_data
    sys.modules["target_scm_loader"] = loader_module
    sys.path.insert(0, str(OFFICIAL))
    import metric as native_metric
    captures = {}
    class SavedKMeans(SKKMeans):
        def __init__(self, n_clusters):
            super().__init__(n_clusters=n_clusters, n_init=10, random_state=1,
                             max_iter=300, tol=1e-4, init="k-means++", algorithm="lloyd")
        def fit_predict(self, X, y=None, sample_weight=None):
            pred = super().fit_predict(X, y=y, sample_weight=sample_weight)
            captures["pred"] = np.asarray(pred).copy()
            captures["z"] = np.asarray(X).copy()
            captures["inertia"] = float(self.inertia_)
            captures["n_iter"] = int(self.n_iter_)
            np.savez_compressed(target / "final_prediction.npz", y_true=frozen.y,
                                y_pred=pred, z=X, sample_order_index_zero_based=np.arange(len(frozen)), seed=1)
            return pred
    native_metric.KMeans = SavedKMeans
    saved_valid = native_metric.valid
    def final_valid(*args, **kwargs):
        assert kwargs.get("eval_z") is True and not kwargs.get("eval_q", False)
        assert not captures, "Only one final readout is allowed"
        result = saved_valid(*args, **kwargs)
        captures["native_metrics"] = dict(zip(["ACC", "NMI", "ARI", "PUR"], map(float, result)))
        return result
    native_metric.valid = final_valid
    sys.argv = [str(target / "executed_train.py"), "--dataset", "TARGET_" + dataset,
                "--mode", "SCM", "--con_iterations", "1000", "--batch_size", "256"]
    scope = dict(__name__="__main__", __file__=str(target / "executed_train.py"))
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        exec(compile(source, str(target / "executed_train.py"), "exec"), scope)
    elapsed = time.perf_counter() - start
    assert "pred" in captures and len(captures["pred"]) == len(frozen)
    assert np.isfinite(captures["z"]).all()
    check = independent_metrics(frozen.y.copy(), captures["pred"])
    assert max(abs(check[m] - captures["native_metrics"][m]) for m in check) < 1e-12
    torch.save(scope["model"].state_dict(), target / "final_checkpoint.pt")
    result = dict(method="SCM", dataset=dataset, metrics=check, native_metrics=captures["native_metrics"],
                  protocol=PROTOCOL, elapsed_seconds=elapsed,
                  statistic_type="one_fixed_seed_final_native_KMeans_complete_partition",
                  no_score_based_checkpoint_or_seed_selection=True,
                  KMeans_inertia=captures["inertia"], KMeans_n_iter=captures["n_iter"],
                  final_prediction_sha256=sha(target / "final_prediction.npz"),
                  final_checkpoint_sha256=sha(target / "final_checkpoint.pt"),
                  input_manifest_sha256=sha(target / "input_manifest.json"), config_sha256=sha(target / "config.json"))
    (target / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPLETE_RESULT_JSON " + json.dumps(result), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(DATASETS))
    args = p.parse_args()
    main(args.dataset)
