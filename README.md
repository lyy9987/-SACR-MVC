# SACR-MVC

Training and evaluation code for **SACR-MVC: Contrastive multi-view clustering with structure anchoring and reliable consensus**.
中文说明见 [README_中文.md](README_中文.md).

Repository: [SACR-MVC](https://github.com/lyy9987/-SACR-MVC).

Supporting numerical results, retained experiment receipts, and RGB-D visualization inputs are in [`results_data`](results_data/README.md). Run `python results_data/scripts/verify_results.py` to check the packaged values.


## Environment

The recorded experiments used Python 3.8.20, PyTorch 2.4.1 and CUDA 11.8. Create an isolated environment, install the matching PyTorch build, then install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

See [data/README.md](data/README.md) for upstream access links, expected MATLAB fields, view order, file hashes and preprocessing. Put the six feature files in one folder and pass its path with `--data-root`.

## Train

```bash
python run.py --list-configs
python run.py --dataset RGB-D --data-root /path/to/data --save-model
python run.py --dataset CIFAR10 --data-root /path/to/data --save-model
```

`--protocol table` (default) selects the supplied main-table setting for each dataset. Inspect the command and settings without starting training:

```bash
python run.py --dataset CIFAR10 --dry-run
```

The supplied three-seed configurations are accessible with `--protocol frozen`:

```bash
python run.py --dataset RGB-D --protocol frozen --seed 303 --data-root /path/to/data
python run.py --dataset RGB-D --protocol frozen --seed 303 --variant no_anchor --data-root /path/to/data
```

The supported leave-one-out options are `no_memory`, `no_anchor`, `no_pretrain_selection` and `no_guard`. Use the paired frozen configurations for the CIFAR10 ablations. Mfeat's preset has zero contrastive epochs. A disabled component can be inactive in a given dataset route.

Use `--gpu 0` to select a CUDA device or `--gpu cpu` to run on CPU. Outputs are written under `outputs/` (override with `--output-dir`), with a CSV summary and an optional final checkpoint. The training program writes the final metrics in a `RESULT_JSON` line. The data loader treats the benchmark cluster count as known metadata; labels are used by the final evaluation routine.

## Evaluate a saved checkpoint

```bash
python evaluate.py --dataset RGB-D --data-root /path/to/data --checkpoint outputs/RGB-D_table_303_Full/models/RGB-D.pth
```

Use the same dataset files and protocol as for training. For a frozen-run checkpoint, also specify `--protocol frozen --seed SEED`.

## Source acknowledgments

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source acknowledgments and third-party licenses.
