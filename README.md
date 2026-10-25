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

Dataset names: RGB-D, Mfeat, Hdigit, ALOI-100, CIFAR10, CIFAR100. Only main-table presets are supplied: use `--protocol table`. Mfeat uses 200 reconstruction epochs and zero contrastive epochs. Use `--gpu cpu` for CPU execution. Table 4 overrides for ALOI-100 and Hdigit are in `configs/ablation_modules.json`.

`results_data/tables/` and `figures/` contain reported values. `results_data/record_index.json` maps result IDs to selected records. `results_data/selected_runs.json` describes 17 main-table partitions and corresponding configurations. All 44 own/DCMVC selected receipts are directly referenced by manuscript CSVs. Full metric vectors and actual configurations are retained; unrelated diagnostic histories and unselected run metrics are omitted. Selected source excerpts retain original metric-bearing text.

## Main-table construction and missing entries

Main-table results are compiled using the following source priority: verifiable values from the method's original paper; values explicitly transcribed in a traceable secondary paper when a usable original-paper entry is unavailable; and retained local reproduction results when neither published source provides a usable value for the relevant benchmark. The source category and supporting evidence for each entry are indexed in `results_data/tables/entry_sources.csv`.

A dash indicates that a verifiable, directly comparable result is unavailable. This can occur because the source does not report the metric, a reliable local result is unavailable, or the source uses a metric definition that does not support direct comparison. Missing values are not estimated or filled using another metric.

Published values and local results use source-specific settings; a common all-method preprocessing, tuning, training or restart protocol is not established. Statistical selection and NMI limitations remain in `results_data/tables/entry_sources.csv`. Five own main entries rely on recorded metrics rather than saved final predictions. Mfeat runtime code does not establish bitwise equivalence with its historical environment. Best-ACC metadata describe original selection; omitted candidate metrics remain local and are not reverified by this supplement.

Replot Figure 6 without training: `python results_data/scripts/reproduce_fig6_rgbd.py /path/to/output`. Saved representations and ordered labels are in `results_data/visualization/`.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records source citations and license scope. The MIT license covers the torch_clustering component. Existing copyright and author headers remain in source files.

Run `python results_data/scripts/reproduce_fig6_rgbd.py --help` to inspect the output-directory argument without loading arrays or running t-SNE.
