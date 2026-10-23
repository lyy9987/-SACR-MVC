"""Public benchmark metadata known before unsupervised training."""


ALIASES = {
    "Scene-15": "Scene15",
    "Caltech-101": "Caltech101",
    "Caltech101_7": "Caltech101-7",
    "Caltech-7": "Caltech101-7",
    "AWA_Deep_2view": "AWA2",
    "Reuters_dim10_2view": "Reuters2",
    "ALOI": "ALOI-100",
    "ALOI100": "ALOI-100",
    "MNIST_USPS": "MNIST-USPS",
    "Caltech101_20": "Caltech101-20",
    "COIL-20": "COIL20",
    "LandUse21": "LandUse-21",
    "Caltech5V": "Caltech-5V",
    # Verified 2026-09-18 by direct view-by-view comparison: Mfeat.mat and
    # handwritten.mat are the same UCI Multiple Features data, so Mfeat is
    # kept as its own entry (its own view order) instead of an alias.
}

KNOWN_CLUSTER_COUNTS = {
    "RGB-D": 13,
    "Caltech101": 101,
    "Caltech101-7": 7,
    "Scene15": 15,
    "Reuters2": 6,
    "AWA2": 50,
    "ALOI-100": 100,
    "Hdigit": 10,
    "Digit-Product": 10,
    "CCV": 20,
    "Cora": 7,
    "MNIST-USPS": 10,
    "Fashion": 10,
    "Caltech101-20": 20,
    "BDGP": 5,
    "COIL20": 20,
    "LandUse-21": 21,
    "Caltech-5V": 7,
    "Mfeat": 10,
    "handwritten": 10,
    "synthetic3d": 3,
    "CIFAR10": 10,
    "CIFAR100": 100,
}

DATASET_FILES = {
    "RGB-D": "RGB-D.mat",
    "Cora": "Cora.mat",
    "CCV": "CCV.mat",
    "Caltech101": "2view-caltech101-8677sample.mat",
    "Reuters2": "Reuters_dim10_2view.mat",
    "ALOI-100": "ALOI-100.mat",
    "Hdigit": "Hdigit_official.mat",
    "Digit-Product": "Digit-Product.mat",
    "Caltech101-7": "Caltech101-7.mat",
    "Scene15": "Scene_15.mat",
    "AWA2": "AWA_Deep_2view.mat",
    "MNIST-USPS": "MNIST_USPS.mat",
    "Fashion": "Fashion.mat",
    "Caltech101-20": "Caltech101-20.mat",
    "BDGP": "BDGP.mat",
    "COIL20": "COIL20.mat",
    "LandUse-21": "LandUse-21.mat",
    "Caltech-5V": "Caltech-5V.mat",
    "Mfeat": "Mfeat.mat",
    "handwritten": "handwritten.mat",
    "synthetic3d": "synthetic3d.mat",
    "CIFAR10": "cifar10.mat",
    "CIFAR100": "cifar100.mat",
}

DATASET_FILE_FALLBACKS = {
    # Upstream feature packages also use the name ``Hdigit_official.mat``.
    # Some public mirrors provide the same benchmark under ``Hdigit.mat``
    # with different MATLAB field names and matrix orientation.
    "Hdigit": ("Hdigit.mat",),
}


def canonical_dataset_name(name):
    value = str(name).strip()
    return ALIASES.get(value, value)


def known_cluster_count(name):
    canonical = canonical_dataset_name(name)
    if canonical not in KNOWN_CLUSTER_COUNTS:
        raise KeyError("No known cluster-count metadata for %s" % canonical)
    return int(KNOWN_CLUSTER_COUNTS[canonical])


def dataset_file_candidates(name):
    canonical = canonical_dataset_name(name)
    if canonical not in DATASET_FILES:
        raise KeyError("No dataset-file metadata for %s" % canonical)
    return (
        DATASET_FILES[canonical],
        *DATASET_FILE_FALLBACKS.get(canonical, ()),
    )
