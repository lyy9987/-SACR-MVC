# Supporting numerical data

This material supports Table 3 (ten baselines and SACR-MVC on six datasets), Tables 4–5 and Figs. 4–9. Entry-specific sources and protocols accompany the results. The SACR-MVC training code and configurations retain the verified experiment-source bytes.

## Result sources and selection

`tables/main_comparison.csv` contains 66 method–dataset entries in 0–1 units. `tables/entry_sources.csv` identifies each original-paper result, secondary report, local record or saved prediction. Original-paper tables take priority; identified secondary reports are used when those tables do not provide the value, followed by documented local reproduction. Published entries retain their reported statistics. Multiple retained local candidates are selected by ACC, with all available metrics taken from the same run or logged epoch. Missing metrics are left blank. Metrics are displayed to four decimal places; percentage-point differences use unrounded records when available.

`evidence/source_statistics_and_candidates.json` records three retained GCFAgg candidates for each of RGB-D, ALOI-100 and Mfeat, and clarifies the published CAFE and GDR-CGDR statistics. CAFE reports the best result over five random seeds. GDR-CGDR repeats each parameter setting ten times, summarizes mean/standard deviation and reports the best setting; its displayed literature values are not single-run maxima. The two marked published CGDR NMI values remain in `NMI_source_reported` and are excluded from the arithmetic-NMI ranking; the source-normalization details remain in the entry index.

`tables/additional_literature_reference.csv` retains additional published results with their sources. `figures/fig4_acc.csv` matches Table 3; the remaining table and figure data accompany the ablation and parameter analyses.

## Configurations and retained evidence

The six SACR-MVC presets are in `configs/`. They specify the actual pretraining and clustering-stage durations, seed, preprocessing and readout settings. Main SACR-MVC entries are supported by atomic result records linked to these presets. RGB-D additionally provides a partition matching its historical prediction-array hash. Exact source hashes and the Mfeat pretraining-source reconstruction are documented in `SOURCE_MANIFEST.json` and `evidence/own_Mfeat_source_boundary.json`.

`benchmark_reproductions/run_index.json` links configurations and result records to 20 GDR-CGDR, five TMCN/Mfeat, six SCM and five RAV partitions. `selected_runs.json` identifies the five best-ACC complete-run selections. The package contains these small label/prediction arrays; feature matrices, embeddings, model weights and full console logs remain in the local archive.

## Baseline settings

GDR-CGDR uses the author implementation with the documented class-to-one-hot correction, bounded-memory anchors and sparse graph construction. Settings are KM, k=20, Maxiter=30 and h=10/10/12/12 for RGB-D/Mfeat/CIFAR10/CIFAR100. Independent mathematical compatibility functions are identified separately from author code.

TMCN/Mfeat uses seeds 0–4, 200 pretraining and 100 fine-tuning epochs, followed by KMeans n_init=100. Its adapter includes the documented pure-PyTorch scan compatibility implementation. TMCN/RGB-D retains its complete terminal evaluation at epoch 300.

SCM uses the author network, loss and optimization function with a fixed target-data adapter: all views and samples, per-view global min-max, seed 1, 1000 minibatch updates, batch 256, Adam lr=0.0003, hidden/projection dimensions 256/128 and native noise/missing settings 0.25/0.25/0.4. This transfers the public MNIST-USPS preset uniformly to the six targets. Terminal normalized z is clustered with known K, n_init=10, seed=1 and max_iter=300.

SCMVC/RGB-D and Mfeat use the complete logged vectors selected by ACC at epochs 247 and 210. RAV uses its saved native argmax partition. Individual record/prediction coverage is identified in `entry_sources.csv`.

## Verification and access

Run `python results_data/scripts/verify_results.py` from the package root for source/table consistency, complete-run selection, atomic-record checks and all file checksums. With NumPy/SciPy, `verify_saved_predictions.py` independently recomputes four metrics from the 37 supplied partitions; it does not load a model or fit clustering. Optional `--data-root` checks the six canonical MAT hashes and ordered labels. Data-access instructions are in `data/README.md`; code sources and component licences are in `THIRD_PARTY_NOTICES.md`. Historical paths in evidence records are provenance locators, not runtime dependencies.
