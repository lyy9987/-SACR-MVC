# SACR-MVC 使用说明

本包提供 **SACR-MVC：基于结构锚定与可靠共识的对比多视图聚类** 的训练、预训练选择、模型保存和聚类评估代码，包含六个基准数据集的配置。

代码仓库：[SACR-MVC](https://github.com/lyy9987/-SACR-MVC)。

## 环境

原实验环境为 Python 3.8.20、PyTorch 2.4.1、CUDA 11.8。建议创建独立环境，安装匹配的 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 数据准备

上游数据包入口、MATLAB 字段、视图顺序、输入文件哈希与预处理见 [data/README.md](data/README.md)。将六个特征文件放到同一目录，通过 `--data-root` 指定。

## 训练

```bash
python run.py --list-configs
python run.py --dataset RGB-D --data-root F:/datasets --save-model
python run.py --dataset CIFAR10 --data-root F:/datasets --save-model
```

默认的 `--protocol table` 使用对应数据集主表配置。`--dry-run` 仅显示运行命令和环境设置：

```bash
python run.py --dataset CIFAR10 --dry-run
```

三种子配置通过 `--protocol frozen` 选择：

```bash
python run.py --dataset RGB-D --protocol frozen --seed 303 --data-root F:/datasets
python run.py --dataset RGB-D --protocol frozen --seed 303 --variant no_anchor --data-root F:/datasets
```

支持的留一消融选项为 `no_memory`、`no_anchor`、`no_pretrain_selection` 和 `no_guard`。CIFAR10 消融使用配对的 frozen 配置。Mfeat 配置的对比训练轮数为 0；具体模块是否参与计算由对应数据集的训练路线决定。

`--gpu 0` 指定 CUDA 设备，`--gpu cpu` 使用 CPU。运行输出默认放在 `outputs/` 中，可用 `--output-dir` 指定其他位置。每次运行输出 CSV 汇总，使用 `--save-model` 时另存最终模型；控制台 `RESULT_JSON` 行包含最终指标。聚类数来自公开基准元信息，最终评估例程读取真实标签计算指标。

## 评估已保存模型

```bash
python evaluate.py --dataset RGB-D --data-root F:/datasets --checkpoint outputs/RGB-D_table_303_Full/models/RGB-D.pth
```

评估时使用与训练相同的数据文件和配置。评估 frozen 模型时，另加 `--protocol frozen --seed 种子值`。

## 实现出处

来源说明和第三方许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

S1数值证据对应统一表3（含SCM的十种基线与SACR-MVC）、表4–5和图4–9。逐项来源和可执行核验见results_data/README_中文说明.md。
