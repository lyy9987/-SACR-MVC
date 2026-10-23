# SACR-MVC 支撑实验结果数据

本包收录 **SACR-MVC：基于结构锚定与可靠共识的对比多视图聚类** 当前稿件表3—表6、图4—图9所用数值、对应的已保存实验回执，以及图6实际使用的 RGB-D 四阶段表示。算法实现和实验配置见本目录的上一级。

`tables/` 是主表、消融、单参数表和正文三项策略对照；`figures/` 是各定量图的数值；`receipts/` 保存真实结果文件中的选定原始 JSON 行与两个独立近邻实验回执；`visualization/` 保存图6实际使用的 epoch0、25、50、100阶段表示及绘图所需类别元数据。epoch0 是预训练后的状态。每个来源的文件路径、行号和哈希见 `SOURCE_MANIFEST.json`。

主表的外部对比方法行明确标为论文转录值，并未冒称原始实验回执。本文方法的 CIFAR10 采用 r5、seed2424的最终记录，其余采用对应冻结 Full 回执。主表0—1刻度、图中百分数、参数表百分点差值分别列明。当前图6对应 `RGB-D_stage0.npz` 与 `RGB-D_finetune.npz`，与早期 `RGB-D_two_phase.npz` 区分。

另外保留6条能直接定位并与当前表格显示数值一致的 GCFAgg、CAFE 结果原始 JSON 行，出处见 `tables/baseline_receipt_index.csv`。

运行 `python scripts/verify_results.py` 可核对数值与校验和；该脚本仅需 Python 标准库。需要复绘图6时，运行 `python scripts/reproduce_fig6_rgbd.py`，另需 NumPy、Matplotlib、scikit-learn。脚本只使用已保存表示，不启动训练；不同库版本可能使 t-SNE 二维坐标有所变化。

原始特征仍由正文及上级数据说明中列出的来源提供，本目录保留支撑结果的数值、回执和可视化输入。来源数据适用的使用条件继续有效。
