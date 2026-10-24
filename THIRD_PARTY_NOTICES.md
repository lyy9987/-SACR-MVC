# Source acknowledgments

The implementation builds on the DCMVC codebase:

- Jinrong Cui, Yuting Li, Han Huang and Jie Wen, *Dual Contrast-Driven Deep Multi-View Clustering*, IEEE Transactions on Image Processing, 33, 4753–4764, 2024. DOI: https://doi.org/10.1109/TIP.2024.3444269
- Source: https://github.com/tweety1028/DCMVC

At the review on 23 October 2026, no explicit project license was found in the DCMVC repository root or README. The citation above records the implementation source; it does not grant a license to that upstream code. The component license below must not be interpreted as a license for the entire project.

`src/torch_clustering/` contains the PyTorch K-Means subset originating from Zhizhong Huang's implementation:

- Source: https://github.com/Hzzone/torch_clustering
- Zhizhong Huang, Jie Chen, Junping Zhang and Hongming Shan, *Learning Representation for Clustering via Prototype Scattering and Positive Sampling*, IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Existing author headers are retained. The package initializer exposes only the K-Means class used by SACR-MVC. The K-Means implementation files are retained from the experiment source.
- The upstream MIT license is reproduced in `third_party/torch_clustering_LICENSE`; it applies to this third-party component, not to the entire SACR-MVC project.

Data remain subject to the terms of their respective providers. Upstream dataset links are listed in `data/README.md`.


## Included baseline provenance snapshots

Small baseline source snapshots and adapters in `results_data/benchmark_reproductions/source_snapshot/` are included to identify the implementations behind the numerical evidence. Their original headers are retained. `results_data/SOURCE_MANIFEST.json` and `benchmark_reproductions/run_index.json` record exact file hashes and available upstream Git blob identifiers.

- **GDR-CGDR:** the authors' implementation of *Scalable Graph Discrete Reconstruction for Efficient Multi-View Clustering*, https://github.com/MarathonZhenyuMa/2026-TKDE-GDR. Original files are separated from the independent corrected-onehot, bounded-memory/sparse and missing-signature compatibility adapters. The latter are local adaptations, not official MathWorks toolbox distributions.
- **TMCN:** the authors' *Trusted Mamba Contrastive Network for Multi-View Clustering* implementation, https://github.com/HackerHyper/TMCN, author snapshot `0198dc9a3eb6441b9aa7b92bcae8f5d47367c21c`. The retained target adapter and pure-PyTorch scan compatibility code are identified separately from author network/loss/metric bytes.
- **SCM:** Caixuan Luo, Jie Xu, Yazhou Ren, Junbo Ma and Xiaofeng Zhu, *Simple Contrastive Multi-View Clustering with Data-Level Fusion*, IJCAI 2024, pp.4697–4705, https://doi.org/10.24963/ijcai.2024/519. Original source: https://github.com/SubmissionsIn/SCM, commit `f23dbc25a7dda62651f12d6cba53f53aa72548dc`. The original `readme.txt` and all seven source-file Git blob hashes are retained. A separate adapter documents target loading, deterministic RNG settings, explicit sklearn defaults and prediction capture. No explicit LICENSE file was present in the checked SCM repository tree; a citation or component license elsewhere in this package must not be read as a license grant for SCM.

Additional literature/reference rows cite their actual papers or author tables in the source CSVs. Numerical inclusion does not imply that those external implementations are part of the SACR-MVC training code or have been redistributed in full.
