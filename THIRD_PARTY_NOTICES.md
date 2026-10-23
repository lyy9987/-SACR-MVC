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
