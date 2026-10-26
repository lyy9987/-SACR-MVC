# Benchmark feature files

This implementation consumes precomputed multi-view feature matrices in MATLAB files. Download the feature datasets from the upstream multi-view-clustering repositories, then put the six files into a single folder. Pass that folder to `run.py --data-root`.

## Upstream access points

- RGB-D: the [DCMVC RGB-D feature file](https://github.com/tweety1028/DCMVC/blob/main/datasets/RGB-D.mat) matches the recorded input by Git blob identity.
- Mfeat: the [UCI Multiple Features dataset](https://archive.ics.uci.edu/dataset/72/multiple+features) supplies the six feature files. In order, `Mfeat.mat` stores `mfeat-fac`, `mfeat-fou`, `mfeat-kar`, `mfeat-mor`, `mfeat-pix`, and `mfeat-zer`. The six numeric matrices and labels were checked against the UCI archive; the MATLAB container is a packaging of these features.

- ALOI-100 and Hdigit: the [DCMVC official repository](https://github.com/tweety1028/DCMVC#datasets--trained-models) explicitly provides these datasets through [Google Drive](https://drive.google.com/drive/folders/108M1L8fqFk4ZcViZWqQbDe3a2d-uGcXd?usp=drive_link) and [Baidu Cloud](https://pan.baidu.com/s/10vzfz623i4NMx-HslacObQ), password `data`.
- Shared multi-view feature collection: [GCFAggMVC data README](https://github.com/Galaxy922/GCFAggMVC/blob/main/data/README.txt) links [this Baidu Cloud dataset package](https://pan.baidu.com/s/1XNWW8UqTcPMkw9NpiKqvOQ), password `data`. The [GDCN repository](https://github.com/HackerHyper/GDCN#3datasets) links the same package.
- The [SCMVC repository](https://github.com/SongwuJob/SCMVC#3datasets) provides [another dataset-package entry](https://pan.baidu.com/s/18If7bx2ZOVZhyijtzycjXA), password `data`, and attributes its datasets to MFLVC and GCFAggMVC. Its [loader](https://github.com/SongwuJob/SCMVC/blob/main/dataloader.py) documents `cifar10.mat` and `cifar100.mat` as three-view precomputed features.

These are upstream package access points, checked on 2026-10-20. The cloud archives were not re-downloaded and checked byte-for-byte for this release. Use the file hashes and layouts below to identify the actual inputs used for the reported runs. The raw-image CIFAR/RGB-D websites are not substitutes for these feature files. The precise upstream extraction network and pretraining recipe for the CIFAR feature views have not been established from the available package documentation.

## File layout and preprocessing

`N × d` below denotes the matrix after the loader aligns the sample axis; the listed feature dimensions preserve the stored view order.

| Dataset | Filename | N | K | Raw dimensions per view | MATLAB feature and label fields |
|---|---|---:|---:|---|---|
| RGB-D | `RGB-D.mat` | 1,449 | 13 | 2,048; 300 | `X` 1×2 cell (`N × d`); `Y` |
| Mfeat | `Mfeat.mat` | 2,000 | 10 | 216; 76; 64; 6; 240; 47 | `X` 6×1 cell (`N × d`); `Y` |
| Hdigit | `Hdigit.mat` | 10,000 | 10 | 784; 256 | `data` 1×2 cell (`d × N`); `truelabel` 1×2 identical label cells |
| ALOI-100 | `ALOI-100.mat` | 10,800 | 100 | 77; 13; 64; 125 | `X` 1×4 cell (`N × d`); `Y` |
| CIFAR10 | `cifar10.mat` | 50,000 | 10 | 512; 2,048; 1,024 | `data` 3×1 cell (`d × N`); `truelabel` |
| CIFAR100 | `cifar100.mat` | 50,000 | 100 | 512; 2,048; 1,024 | `data` 3×1 cell (`d × N`); `truelabel` |

Input preprocessing and clustering readout follow the supplied configurations for each reported local result. The dimensions above describe the original stored feature views.

The loader accepts `Hdigit_official.mat` or `Hdigit.mat`, preferring the former if both are present. For exact input identity, use the file that matches the `Hdigit.mat` hash below. Generic loaders transpose `d × N` matrices to `N × d` and map one-based labels to zero-based labels. All stored feature views are used; the CIFAR `MissingStatus` and `index` fields are not used to mask or reorder features.

## SHA-256 of the recorded input files

| File | SHA-256 |
|---|---|
| `RGB-D.mat` | `b41e6f68bd1a1d4b632b0c1ecf9ecf2314c7f26519b16ffeb4ec25f27ef6ac88` |
| `Mfeat.mat` | `f5ad03f0fbdea89e1af38b6a6e2941be8d6bc85ec641b4cdcf884d04c7c94bda` |
| `Hdigit.mat` | `af2f600a05137ab12697a169397864d6be7c5bee40ba4ce25db7f4074f0296ce` |
| `ALOI-100.mat` | `e20bb24d6473a14e84aba93a92cfd19c4ddbb8ea3fa41618a63049f533da8323` |
| `cifar10.mat` | `9b8ef1e354f54382a26b1c5b11812d4ad94e63c5da78c56df19e8195398e60fb` |
| `cifar100.mat` | `d693a211fd03adc9ce3a0e64208a19a23fd69582739f13686b9edcee90265c82` |

MATLAB serialization can differ between mirrors. A different file hash alone does not establish feature inequality; the matrices, view order and labels must also be checked when using a reserialized copy. Original benchmark feature files and model weights are not included. Supporting result data, saved representations, and class metadata are provided in `../results_data/`.
