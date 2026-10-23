from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np
from torch.utils.data import Dataset
import torch, os, scipy.io as sio
from scipy import sparse

from dataset_metadata import (
    DATASET_FILES,
    canonical_dataset_name,
    dataset_file_candidates,
    known_cluster_count,
)

try:
    import hdf5storage
except Exception:
    hdf5storage = None

def _candidate_paths(filename, path="datasets/"):
    candidates = []
    dataset_root = os.environ.get("SACR_DATASET_ROOT", "").strip()
    if dataset_root:
        candidates.append(os.path.join(dataset_root, filename))
    candidates.extend([
        os.path.join(path, filename),
        os.path.join("datasets", filename),
        os.path.join("data", filename),
        os.path.join("..", "data", filename),
        os.path.join("..", "datasets", filename),
    ])
    return candidates

def _find_existing_file(filename, path="datasets/"):
    for p in _candidate_paths(filename, path):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find %s. Checked: %s" % (filename, " | ".join(_candidate_paths(filename, path))))

def _load_mat_auto(filename, path="datasets/"):
    file_path = _find_existing_file(filename, path)
    loaders = [sio.loadmat] if hdf5storage is None else [hdf5storage.loadmat, sio.loadmat]
    last_err = None
    for loader in loaders:
        try:
            return loader(file_path)
        except Exception as e:
            last_err = e
    raise last_err

def _to_dense_float32(x):
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    if x.dtype == np.object_ and x.size == 1:
        x = np.asarray(x.item())
    return x.astype(np.float32)

def _clean_numeric_view(v):
    v = _to_dense_float32(v)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v


def _effective_preprocess(default_norm="standard"):
    """Label-free preprocessing overrides via environment variables.

    SACR_PREPROCESS_NORM : standard | minmax | robust | l2 | none
    SACR_MAX_VIEW_DIM    : int (0 = no reduction)
    SACR_REDUCER         : svd | pca | none
    SACR_PREPROCESS_SEED : int
    """
    norm = os.environ.get("SACR_PREPROCESS_NORM", default_norm).strip().lower()
    if norm not in {"standard", "minmax", "robust", "l2", "none"}:
        raise ValueError("Unsupported SACR_PREPROCESS_NORM=%s" % norm)
    try:
        max_dim = max(0, int(os.environ.get("SACR_MAX_VIEW_DIM", "0") or 0))
    except Exception:
        max_dim = 0
    reducer = os.environ.get("SACR_REDUCER", "none").strip().lower()
    if reducer not in {"svd", "pca", "none"}:
        raise ValueError("Unsupported SACR_REDUCER=%s" % reducer)
    seed = int(os.environ.get("SACR_PREPROCESS_SEED", "303") or 303)
    return norm, max_dim, reducer, seed


def _reduce_view(v, max_dim, reducer="none", seed=303):
    v = _clean_numeric_view(v)
    if max_dim <= 0 or reducer == "none" or v.shape[1] <= max_dim:
        return v
    n_comp = min(int(max_dim), int(v.shape[1] - 1), int(v.shape[0] - 1))
    if n_comp < 2:
        return v
    if reducer == "pca":
        model = PCA(n_components=n_comp, random_state=seed)
    else:
        model = TruncatedSVD(n_components=n_comp, random_state=seed)
    return np.asarray(model.fit_transform(v), dtype=np.float32)


def _normalize_views(views, mode="standard"):
    norm, max_dim, reducer, seed = _effective_preprocess(mode)
    out = []
    for v in views:
        v = _reduce_view(v, max_dim=max_dim, reducer=reducer, seed=seed)
        if norm == "standard":
            v = StandardScaler().fit_transform(v)
        elif norm == "minmax":
            v = MinMaxScaler().fit_transform(v)
        elif norm == "robust":
            v = RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(v)
        elif norm == "l2":
            v = Normalizer(norm="l2").fit_transform(v)
        out.append(np.asarray(v, dtype=np.float32))
    return out

def _extract_labels(mat):
    for key in [
        "Y", "y", "gt", "truth", "truelabel", "label", "labels", "gnd"
    ]:
        if key in mat:
            raw = np.asarray(mat[key])
            if raw.dtype == np.object_:
                candidates = [
                    np.asarray(item).squeeze()
                    for item in raw.reshape(-1)
                    if np.asarray(item).size > 0
                ]
                if not candidates:
                    continue
                # Hdigit stores one identical label vector per view.
                y = candidates[0]
                for candidate in candidates[1:]:
                    if candidate.shape == y.shape and not np.array_equal(
                        candidate, y
                    ):
                        raise ValueError(
                            "View-specific label vectors disagree in key %s" % key
                        )
            else:
                y = raw.squeeze()
            if y.size == 0:
                continue
            y = y.astype(np.int32)
            if y.min() == 1:
                y = y - 1
            return y
    raise KeyError("Cannot find labels key in mat file. Existing keys: %s" % list(mat.keys()))

def _extract_two_views_from_cell(cell_like):
    arr = np.asarray(cell_like)
    if arr.dtype != np.object_:
        return None
    flat = [x for x in arr.reshape(-1) if x is not None]
    if len(flat) < 2:
        return None
    return [_to_dense_float32(flat[0]), _to_dense_float32(flat[1])]


def _extract_all_views_from_cell(cell_like):
    """Extract all views from MATLAB cell/object arrays.

    Supports common multi-view .mat formats:
    - X is a 1 x V or V x 1 cell array, each cell is [n_samples, dim]
    - data / fea / data_views is stored in the same style
    """
    arr = np.asarray(cell_like)
    if arr.dtype != np.object_:
        return None
    flat = [x for x in arr.reshape(-1) if x is not None]
    views = []
    for x in flat:
        try:
            v = _to_dense_float32(x)
            if v.ndim == 2 and min(v.shape) > 1:
                # Some Caltech files store features as [dim, n_samples].
                # If one axis is much larger than the other, sample axis is usually the larger one.
                views.append(v)
        except Exception:
            pass
    return views if len(views) >= 2 else None

def _extract_all_views(mat):
    """Extract all views, not only two views.

    This is needed for Caltech101-7 / Caltech101-20 style datasets,
    which usually contain six hand-crafted visual feature views.
    """
    for key in ["X", "data", "data_views", "fea", "x"]:
        if key in mat:
            views = _extract_all_views_from_cell(mat[key])
            if views is not None:
                return views

    # Fallback: named view keys.
    named = []
    for key in ["X1","X2","X3","X4","X5","X6","x1","x2","x3","x4","x5","x6",
                "view1","view2","view3","view4","view5","view6",
                "View1","View2","View3","View4","View5","View6",
                "fea1","fea2","fea3","fea4","fea5","fea6"]:
        if key in mat:
            named.append(_to_dense_float32(mat[key]))
    if len(named) >= 2:
        return named

    # Last fallback: all numeric matrices with the same sample number.
    numeric = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        try:
            arr = _to_dense_float32(v)
        except Exception:
            continue
        if arr.ndim == 2 and min(arr.shape) > 1:
            numeric.append((k, arr))
    if len(numeric) >= 2:
        # Select the most common sample dimension.
        candidates = []
        for _, arr in numeric:
            candidates.extend([arr.shape[0], arr.shape[1]])
        sample_dim = max(set(candidates), key=candidates.count)
        views = []
        for _, arr in numeric:
            if arr.shape[0] == sample_dim:
                views.append(arr)
            elif arr.shape[1] == sample_dim:
                views.append(arr.T)
        if len(views) >= 2:
            return views

    raise KeyError("Cannot extract multi-view features from mat file. Available keys: %s" % list(mat.keys()))

def load_generic_multiview_dataset(filename, path="datasets/", norm="standard"):
    mat = _load_mat_auto(filename, path)
    views = _extract_all_views(mat)
    labels = _extract_labels(mat)
    # Make sure every view is [n_samples, dim].
    n = len(labels)
    fixed = []
    for v in views:
        v = _to_dense_float32(v)
        if v.shape[0] == n and v.ndim > 2:
            # Image-based MVC mirrors may store a view as [N,H,W,C].  The
            # shared MLP consumes vector features, so flatten only the
            # non-sample axes while preserving the sample order through this
            # deterministic conversion.
            v = v.reshape(n, -1)
        elif v.ndim == 2 and v.shape[0] != n and v.shape[1] == n:
            v = v.T
        if v.ndim != 2 or v.shape[0] != n:
            raise ValueError(
                "Cannot align view shape %s with %d labels in %s"
                % (tuple(v.shape), n, filename)
            )
        fixed.append(v)
    views = _normalize_views(fixed, mode=norm)
    return views, labels


def _extract_two_views(mat):
    for key in ["X", "data", "data_views", "fea", "x"]:
        if key in mat:
            views = _extract_two_views_from_cell(mat[key])
            if views is not None:
                return views

    for k1, k2 in [
        ("X1", "X2"), ("x1", "x2"),
        ("view1", "view2"), ("View1", "View2"),
        ("fea1", "fea2"), ("F1", "F2"),
        ("Xa", "Xb"), ("A", "B"),
        ("img_fea", "txt_fea"), ("rgb", "depth")
    ]:
        if k1 in mat and k2 in mat:
            return [_to_dense_float32(mat[k1]), _to_dense_float32(mat[k2])]

    numeric_keys = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.dtype != np.object_ and arr.ndim == 2 and min(arr.shape) > 1:
            numeric_keys.append(k)

    if len(numeric_keys) >= 2:
        return [_to_dense_float32(mat[numeric_keys[0]]), _to_dense_float32(mat[numeric_keys[1]])]

    raise KeyError("Cannot extract two views from mat file. Available keys: %s" % list(mat.keys()))

def load_generic_two_view_dataset(filename, path="datasets/", norm="standard"):
    mat = _load_mat_auto(filename, path)
    views = _extract_two_views(mat)
    labels = _extract_labels(mat)
    views = _normalize_views(views, mode=norm)
    return views, labels

class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/"):
        self.data_views = []
        self.source_paths = []
        alias = {
            "Scene-15": "Scene15",
            "Caltech-101": "Caltech101",
            "Caltech101-7": "Caltech101-7",
            "Caltech-7": "Caltech101-7",
            "Caltech101_7": "Caltech101-7",
            "AWA_Deep_2view": "AWA2",
            "AWA2": "AWA2",
            "Reuters_dim10_2view": "Reuters2",
            "Reuters2": "Reuters2",
        }
        db = canonical_dataset_name(alias.get(db, db))
        self.name = db
        source_name = DATASET_FILES.get(db)
        if db in DATASET_FILES:
            source_name = next(
                (
                    filename
                    for filename in dataset_file_candidates(db)
                    if any(
                        os.path.exists(candidate)
                        for candidate in _candidate_paths(filename, path)
                    )
                ),
                source_name,
            )
        ccv_split_names = ("STIP.npy", "SIFT.npy", "MFCC.npy", "label.npy")
        ccv_split_paths = [
            next(
                (
                    os.path.abspath(candidate)
                    for candidate in _candidate_paths(filename, path)
                    if os.path.exists(candidate)
                ),
                None,
            )
            for filename in ccv_split_names
        ]
        use_ccv_split = db == "CCV" and all(ccv_split_paths)
        if use_ccv_split:
            # Several public mirrors distribute CCV as three feature arrays
            # plus one label array instead of a monolithic CCV.mat.  Treat the
            # four files as one dataset source and let the training manifest
            # hash every component (see ``dataset_source_sha256``).
            self.source_paths = list(ccv_split_paths)
            self.source_path = self.source_paths[0]
        else:
            self.source_path = (
                os.path.abspath(_find_existing_file(source_name, path))
                if source_name is not None
                else None
            )
            if self.source_path is not None:
                self.source_paths = [self.source_path]

        if db == "RGB-D":
            mat = _load_mat_auto("RGB-D.mat", path)
            X_data = mat["X"]; self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(_to_dense_float32(X_data[0, idx]))
            self.data_views = _normalize_views(self.data_views, mode="minmax")
            self.labels = _extract_labels(mat)

        elif db == "CCV":
            if use_ccv_split:
                feature_paths = self.source_paths[:3]
                label_path = self.source_paths[3]
                self.data_views = [
                    np.asarray(np.load(filename, mmap_mode="r"), dtype=np.float32)
                    for filename in feature_paths
                ]
                self.num_views = len(self.data_views)
                self.data_views = _normalize_views(
                    self.data_views, mode="minmax"
                )
                self.labels = (
                    np.asarray(np.load(label_path, mmap_mode="r"))
                    .reshape(-1)
                    .astype(np.int64)
                )
            else:
                mat = _load_mat_auto("CCV.mat", path)
                X_data = mat["X"]; self.num_views = X_data.shape[1]
                for idx in range(self.num_views):
                    self.data_views.append(_to_dense_float32(X_data[0, idx]))
                self.data_views = _normalize_views(self.data_views, mode="minmax")
                self.labels = _extract_labels(mat)

        elif db == "Cora":
            # Public Cora mirrors use either an X cell array or four named
            # matrices (content/inbound/outbound/cites).  The generic loader
            # resolves orientation against the common 2,708-sample axis and
            # preserves all four standard views.
            self.data_views, self.labels = load_generic_multiview_dataset(
                "Cora.mat", path, norm="minmax"
            )
            self.num_views = len(self.data_views)

        elif db == "Digit-Product":
            mat = _load_mat_auto("Digit-Product.mat", path)
            X_data = mat["X"]; self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(_to_dense_float32(X_data[0, idx]))
            self.data_views = _normalize_views(self.data_views, mode="standard")
            self.labels = _extract_labels(mat)

        elif db == "ALOI-100":
            mat = _load_mat_auto("ALOI-100.mat", path)
            X_data = mat["X"]; self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(_to_dense_float32(X_data[0, idx]))
            self.data_views = _normalize_views(self.data_views, mode="minmax")
            self.labels = _extract_labels(mat)

        elif db == "Hdigit":
            # Prefer the X/Y feature layout
            # when it is available.  The alternative public copy uses
            # ``data`` / ``truelabel`` and transposed matrices; the same generic
            # loader handles both without changing sample order.
            official = any(
                os.path.exists(candidate)
                for candidate in _candidate_paths("Hdigit_official.mat", path)
            )
            self.data_views, self.labels = load_generic_multiview_dataset(
                "Hdigit_official.mat" if official else "Hdigit.mat",
                path,
                norm="minmax",
            )
            self.num_views = len(self.data_views)

        elif db == "Scene15":
            # Scene15 contains three standard feature views (20/59/40 dims).
            # The inherited loader silently discarded the third view, making
            # comparisons with standard multi-view protocols invalid.
            self.data_views, self.labels = load_generic_multiview_dataset(
                "Scene_15.mat", path, norm="standard"
            )
            self.num_views = len(self.data_views)


        elif db == "Caltech101-7":
            # Standard multi-view benchmark: usually 1474 samples, 7 classes, 6 views.
            # File name in the downloaded repository: Caltech101-7.mat
            self.data_views, self.labels = load_generic_multiview_dataset("Caltech101-7.mat", path, norm="standard")
            self.num_views = len(self.data_views)

        elif db == "Caltech101":
            # 这个文件是特殊格式，必须转置成 [8677, dim]
            mat = _load_mat_auto("2view-caltech101-8677sample.mat", path)
            X = mat["X"][0]
            self.num_views = 2
            self.data_views.append(_to_dense_float32(X[0]).T)
            self.data_views.append(_to_dense_float32(X[1]).T)
            self.data_views = _normalize_views(self.data_views, mode="standard")
            self.labels = np.asarray(mat["gt"]).squeeze().astype(np.int32) - 1

        elif db == "AWA2":
            self.data_views, self.labels = load_generic_two_view_dataset("AWA_Deep_2view.mat", path, norm="standard")
            self.num_views = 2

        elif db == "Reuters2":
            self.data_views, self.labels = load_generic_two_view_dataset("Reuters_dim10_2view.mat", path, norm="standard")
            self.num_views = 2

        elif db == "MNIST-USPS":
            self.data_views, self.labels = load_generic_multiview_dataset(
                "MNIST_USPS.mat", path, norm="minmax"
            )
            self.num_views = len(self.data_views)

        elif db == "Fashion":
            self.data_views, self.labels = load_generic_multiview_dataset(
                "Fashion.mat", path, norm="minmax"
            )
            self.num_views = len(self.data_views)

        elif db == "Caltech101-20":
            self.data_views, self.labels = load_generic_multiview_dataset(
                "Caltech101-20.mat", path, norm="standard"
            )
            self.num_views = len(self.data_views)

        elif db == "BDGP":
            self.data_views, self.labels = load_generic_multiview_dataset(
                "BDGP.mat", path, norm="minmax"
            )
            self.num_views = len(self.data_views)

        elif db == "COIL20":
            self.data_views, self.labels = load_generic_multiview_dataset(
                "COIL20.mat", path, norm="minmax"
            )
            self.num_views = len(self.data_views)

        elif db == "LandUse-21":
            self.data_views, self.labels = load_generic_multiview_dataset(
                "LandUse-21.mat", path, norm="standard"
            )
            self.num_views = len(self.data_views)

        elif db == "Caltech-5V":
            self.data_views, self.labels = load_generic_multiview_dataset(
                "Caltech-5V.mat", path, norm="standard"
            )
            self.num_views = len(self.data_views)

        elif db == "Mfeat":
            # Same UCI Multiple Features data as handwritten (2000 samples,
            # 6 views, 10 classes), loaded with the view order stored in the
            # user ZIP; min-max matches the comparison papers' preprocessing.
            self.data_views, self.labels = load_generic_multiview_dataset(
                "Mfeat.mat", path, norm="minmax"
            )
            self.num_views = len(self.data_views)

        elif db == "handwritten":
            # UCI Multiple Features (2000 samples, 6 views). Same underlying
            # data as Hdigit/Mfeat; loaded with the generic multi-view reader.
            self.data_views, self.labels = load_generic_multiview_dataset(
                "handwritten.mat", path, norm="minmax"
            )
            self.num_views = len(self.data_views)

        elif db == "synthetic3d":
            # Toy 3-view, 3-dim, 3-class synthetic set (600 samples).
            self.data_views, self.labels = load_generic_multiview_dataset(
                "synthetic3d.mat", path, norm="standard"
            )
            self.num_views = len(self.data_views)

        elif db == "CIFAR10":
            # 3 deep feature views (1024/2048/512), 10 classes, 50k samples.
            self.data_views, self.labels = load_generic_multiview_dataset(
                "cifar10.mat", path, norm="standard"
            )
            self.num_views = len(self.data_views)

        elif db == "CIFAR100":
            # Same feature layout as CIFAR10 with 100 classes.
            self.data_views, self.labels = load_generic_multiview_dataset(
                "cifar100.mat", path, norm="standard"
            )
            self.num_views = len(self.data_views)

        else:
            raise NotImplementedError(f"Unsupported dataset: {db}")

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    def __len__(self):
        return int(self.data_views[0].shape[0])

    def __getitem__(self, index):
        # Do not place ground-truth assignments in any training/evaluation
        # mini-batch. Metrics read ``labels`` only after predictions exist.
        return [self.data_views[v][index] for v in range(self.num_views)], -1, index

def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data)
    num_clusters = known_cluster_count(mv_data.name)
    mv_data_loader = torch.utils.data.DataLoader(mv_data, batch_size=batch_size, shuffle=True, drop_last=False)
    return mv_data_loader, num_views, num_samples, num_clusters

def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data)
    num_clusters = known_cluster_count(mv_data.name)
    mv_data_loader = torch.utils.data.DataLoader(mv_data, batch_size=num_samples, shuffle=False, drop_last=False)
    return mv_data_loader, num_views, num_samples, num_clusters
