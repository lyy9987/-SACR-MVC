# 实验数值与来源

本材料对应表3（六个数据集上的十种基线及SACR-MVC）、表4–5和图4–9。结果附带逐项来源与实验协议。SACR-MVC训练代码和配置保持已核验实验源码的字节内容。

## 来源与结果选择

`tables/main_comparison.csv`包含66个方法—数据集条目，采用0–1数值；`tables/entry_sources.csv`提供逐项来源索引。

`evidence/source_statistics_and_candidates.json`记录来源材料、保留候选及核验资料，逐项原始值和引用保留在对应索引文件。

`tables/additional_literature_reference.csv`保留附带来源的其他文献结果。图4数据与表3一致；其余表格和图件数据对应消融及参数分析。

## 配置与结果证据

`configs/`中六个SACR-MVC预设记录真实预训练和聚类阶段时长、种子、预处理及读出设置。主表条目由与预设关联的完整原子结果记录支持；RGB-D另外提供与历史预测数组哈希一致的划分。精确源码哈希及Mfeat预训练源码重构见`SOURCE_MANIFEST.json`和`evidence/own_Mfeat_source_boundary.json`。

`benchmark_reproductions/run_index.json`关联20组GDR-CGDR、五组TMCN/Mfeat、六组SCM及五组RAV的配置、结果记录与划分；`selected_runs.json`记录五组最高ACC完整运行选择。包内保留小型标签/预测数组，特征矩阵、嵌入、模型权重和完整控制台日志留在本地档案。

## 基线设置

GDR-CGDR使用作者实现，并注明类别到one-hot映射修正、有界内存anchor及稀疏图构造。固定KM、k=20、Maxiter=30，RGB-D/Mfeat/CIFAR10/CIFAR100的h为10/10/12/12；独立数学兼容函数与作者代码分别标明。

TMCN/Mfeat采用种子0–4、200轮预训练和100轮微调，最终KMeans n_init=100；适配器包含注明的纯PyTorch scan兼容实现。TMCN/RGB-D保留epoch300的完整终轮评测。

SCM采用作者网络、损失与优化函数，通过固定目标接口使用全部视图和样本：逐视图全局min-max、seed1、1000次小批次更新、batch256、Adam学习率0.0003、隐藏/投影维数256/128及原生noise/missing参数0.25/0.25/0.4。该预算由公开MNIST-USPS预设统一迁移到六目标。最终归一化z采用已知K、n_init=10、seed1和max_iter300读出。

SCMVC在RGB-D与Mfeat上分别选择ACC最高的完整日志向量epoch247和210。RAV采用保存的原生argmax划分。每个条目的记录与预测证据类型见`entry_sources.csv`。

## 核验与访问

包根目录运行`python results_data/scripts/verify_results.py`可检查来源/表格、完整运行选择、原子记录及全部文件校验和。安装NumPy/SciPy后，`verify_saved_predictions.py`从37份已提供划分独立重算四项指标，不加载模型或重新聚类；可选`--data-root`核对六个MAT哈希与样本标签顺序。数据入口见`data/README.md`，代码来源与组件许可见`THIRD_PARTY_NOTICES.md`。证据中的历史路径仅作来源定位，不是运行依赖。
