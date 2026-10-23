# SACR-MVC supporting experimental data

This package supports the manuscript **SACR-MVC: Contrastive multi-view clustering with structure anchoring and reliable consensus**. It contains the numerical results used in Tables 3–6 and Figs. 4–9, retained result receipts, and the saved RGB-D representations used for Fig. 6. The implementation and experiment configurations are in the parent directory of this supplement.

## Files and units

- `tables/main_comparison.csv`: the displayed comparison values in Tables 3–4, on a 0–1 scale. SACR-MVC rows are checked against the retained receipts. Other comparison values are explicitly labelled as reported-table transcriptions; they are not new measurements or raw experiment logs.
- `tables/ablation_modules.csv`: Table 5, with full-precision retained metrics on a 0–1 scale. The manuscript rounds these values to four decimal places.
- `tables/parameter_onefactor.csv`: Table 6. ACC and NMI are percentages; delta ACC is in percentage points relative to the corresponding main setting.
- `tables/strategy_checks.csv`: the three additional ablation comparisons reported in the text.
- `tables/baseline_receipt_index.csv`: origins of six directly located GCFAgg/CAFE result records whose method, dataset, and displayed metrics match the comparison table. The original JSON lines are retained in `receipts/baseline_selected.jsonl`.
- `figures/`: numerical values for the ACC comparison, Fowlkes–Mallows comparison, loss-weight grids, and neighborhood-size analysis. Percentages are calculated as 100 times the corresponding receipt metrics.
- `receipts/`: the selected original JSON lines from the frozen result file, the promoted CIFAR10 `RESULT_JSON` line, and the two retained standalone neighborhood-study receipts. Selected lines preserve the original fields, including historical local file paths; these paths document provenance and are not required to run the verification script.
- `visualization/`: the unmodified `RGB-D_stage0.npz` and `RGB-D_finetune.npz` stage files, plus class-label metadata in original row order. These contain four 1449 × 256 float32 arrays: stage0, stage25, stage50, and stage100. Stage0 is the representation after pretraining.
- `SOURCE_MANIFEST.json`: file origins, source hashes, extraction types, and package boundaries.
- `CHECKSUMS.sha256`: checksums of the packaged files.

## Verification and visualization

Run `python scripts/verify_results.py` to check the included numerical CSVs against their result receipts and verify file checksums. Python 3 is sufficient; this check uses the standard library.

Run `python scripts/reproduce_fig6_rgbd.py` to recompute the four RGB-D t-SNE panels from the saved representations. This optional script requires NumPy, Matplotlib, and scikit-learn. It uses the original settings: perplexity 30, learning rate 200, PCA initialization, 2000 iterations, random state 0, and the same fixed sample indices across panels. Labels are used for coloring. The code does not train or modify the clustering model. The saved stage representations and fixed settings provide the inputs for recomputing the visualization; coordinates can vary with library versions.

The CIFAR10 main result and F-score use the promoted r5 run with seed 2424. The other main SACR-MVC entries use the corresponding retained frozen Full runs. F-score entries use each method's own retained representative run; the DCMVC RGB-D, ALOI-100, and Hdigit F-scores are not the literature rows in Table 3.

## Source data and rights

Original benchmark features are obtained from the repositories and archives cited in the manuscript and described in S1. They are not redistributed here. Class metadata and learned representations retain any applicable benchmark-source restrictions. No additional rights to third-party benchmark data are granted by this supplement.
