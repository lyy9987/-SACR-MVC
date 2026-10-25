# SACR-MVC supplementary material

Code, parameters and selected records for Tables 3-5, Figures 4-9 and the manuscript strategy comparisons.

Install with `python -m pip install -r requirements.txt`. The experiment environment used Python 3.8.20, PyTorch 2.4.1 and CUDA 11.8. [data/README.md](data/README.md) describes access, layouts and preprocessing for six feature datasets. Feature files and model checkpoints are not bundled.

```bash
python run.py --list-configs
python run.py --dataset RGB-D --protocol table --dry-run
python run.py --dataset RGB-D --data-root /path/to/data --save-model
python evaluate.py --dataset RGB-D --data-root /path/to/data --checkpoint outputs/RGB-D_table_303_Full/models/RGB-D.pth
python run_ablation.py --dataset ALOI-100 --variant no_dcd --dry-run
```

Dataset names: RGB-D, Mfeat, Hdigit, ALOI-100, CIFAR10, CIFAR100. Only main-table presets are supplied: use `--protocol table`. Mfeat uses 200 reconstruction epochs and zero contrastive epochs. Use `--gpu cpu` for CPU execution. Table 4 overrides for ALOI-100 and Hdigit are in `configs/ablation_modules.json`. CIFAR10 uses OMP_NUM_THREADS=4; the other three numerical-thread settings remain inherited.

`results_data/tables/` and `figures/` contain the reported values. `results_data/record_index.json` maps result IDs to source records. `results_data/selected_runs.json` indexes 44 saved main-table partitions and their configurations. These comprise retained original partitions, reported single runs, preserved-checkpoint readouts and later fixed-configuration replays; later partitions are not historical training-time predictions. Selected receipts and source excerpts retain the reported metrics and actual configurations. Source-specific seed, epoch and selection rules are recorded with each entry.

Main-table sources are prioritized as the original paper, traceable secondary-paper transcriptions when an original entry is unavailable, and retained local results when neither published source supplies a usable entry. A dash means that the metric was unreported, a verifiable local result was unavailable, or the source metric was not directly comparable. Sources use their own settings; a common preprocessing, training or restart protocol is not established. Entry sources, statistical selection and NMI definitions or exclusions are recorded in `results_data/tables/entry_sources.csv`. Shared/recomputed NMI uses arithmetic-mean normalization.

Replot Figure 6 without training: `python results_data/scripts/reproduce_fig6_rgbd.py /path/to/output`. Saved representations and ordered labels are in `results_data/visualization/`.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records source citations and license scope. The MIT license covers the torch_clustering component. Existing copyright and author headers remain in source files.

Run `python results_data/scripts/reproduce_fig6_rgbd.py --help` to inspect the output-directory argument without loading arrays or running t-SNE.
