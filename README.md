# PCDformer-Reproduction-Improvement
Reproduction and improvement of PCDformer for multivariate long-term time-series forecasting.
# PCDformer 复现与改进实验

本仓库用于整理赵云波老师 NICLab 预推免阅读报告考核中的代码、实验日志和结果文件。对应阅读论文为 **Multivariate Time-Series Modeling and Forecasting With Parallelized Convolution and Decomposed Sparse-Transformer**。本文围绕多变量长序列时间序列预测任务，对 PCDformer 的主要思想进行复现，并在复现基础上进行了结构改进、对比实验和消融实验。

## 一、项目说明

PCDformer 主要面向多变量长序列预测任务。原论文认为，传统 Transformer 类模型通常将每个时间步的多变量观测映射为一个 token，这容易造成不同变量之间的信息耦合，也会使模型受到无关变量干扰。为缓解这一问题，PCDformer 采用变量并行处理、时间序列分解和稀疏自注意力机制，使模型能够分别提取不同变量的时间依赖，并进一步建模变量之间的重要关联。

本仓库的工作主要包括三个部分：

1. 基于 Autoformer 框架完成 PCDformer 思想的结构对齐复现
2. 在复现版本上加入改进模块，提升多变量预测的稳定性
3. 保存对比实验和消融实验日志，便于检查实验过程和结果来源

## 二、论文信息

论文题目：Multivariate Time-Series Modeling and Forecasting With Parallelized Convolution and Decomposed Sparse-Transformer
作者：Shusen Ma, Yun-Bo Zhao, Yu Kang, Peng Bai
期刊：IEEE Transactions on Artificial Intelligence
年份：2024
任务类型：多变量长序列时间序列预测
评价指标：MSE，MAE

## 三、仓库结构

当前仓库主要包含以下内容：

```text
PCDformer-Reproduction-Improvement
├── PCDformer-Autoformer-paper-aligned
│   └── 用于结构对齐和实验配置的基础文件
├── PCDformer复现代码
│   └── 基于 Autoformer 框架整理得到的 PCDformer 复现版本
├── PCDformer改进的代码
│   └── 在复现代码基础上加入改进方法后的版本
├── 两份消融代码运行的日志结果
│   └── 不同模块单独加入时的消融实验日志
├── 完整的改进代码与复现代码运行的日志结果
│   └── 原始复现版本和完整改进版本的对比实验日志
├── .gitignore
└── README.md
```

其中，代码文件用于说明复现与改进的具体实现过程，日志文件用于保留实验运行记录和最终指标，便于后续核对。

## 四、复现思路

原论文 PCDformer 的完整官方实现未直接开源，因此本项目选择基于 Autoformer 代码框架进行结构对齐复现。这样处理主要有两个原因。

第一，PCDformer 与 Autoformer 在整体形式上具有一定相似性，二者都属于面向长序列预测的 encoder decoder 类结构，并且都利用了时间序列分解思想。Autoformer 的代码结构较清晰，数据读取、训练流程、评价指标和实验配置都比较完整，适合作为复现基础。

第二，阅读报告的重点不是简单运行已有代码，而是理解论文方法并尽量还原其核心思想。因此，本项目重点复现 PCDformer 的变量并行处理、时间模式分解和变量关系建模思想，并在同一训练框架下进行实验对比，使复现版本和改进版本具有可比性。

## 五、改进思路

在复现过程中，我主要关注两个问题。

第一个问题是高维多变量场景下变量相关性建模的冗余。对于 Traffic 等变量数量较多的数据集，如果直接在所有变量之间建立注意力关系，模型需要处理大量弱相关甚至无关的变量交互。改进版本通过引入更紧凑的变量路由思想，使变量信息先汇聚到少量核心表示，再由核心表示反馈给各变量，从而减少冗余变量关系带来的干扰。

第二个问题是不同数据集之间的尺度变化和趋势变化。时间序列数据通常存在不同变量尺度差异明显、局部趋势不断变化等问题。如果模型直接学习原始序列，容易受到分布变化影响。改进版本加入了更稳定的归一化和残差适配思想，使模型在预测时能够更好地处理尺度差异和趋势项变化。

整体上，改进方法的目标不是单纯增加模型复杂度，而是在原论文思想基础上增强变量交互的选择性和预测结果的稳定性。

## 六、实验环境

本项目实验主要在如下环境下完成：

```text
GPU：NVIDIA RTX 3080 20GB
CPU：AMD EPYC 7601 32-Core Processor
CPU 核心数：8 核
实例内存：15GB
CUDA 版本：12.4
显卡驱动版本：550.163.01
系统盘：20GB
数据盘：50GB SSD
```
具体 Python 依赖以代码文件夹中的实际环境为准。建议使用 PyTorch 环境运行，并根据报错补充安装对应依赖。

## 七、运行说明

不同代码版本的运行方式略有差异，建议进入对应代码文件夹后查看脚本或参数文件。运行流程如下：

```bash
cd PCDformer复现代码
python run.py
```

或进入改进版本代码文件夹：

```bash
cd PCDformer改进的代码
python run.py
```
实际运行前需要确认数据集路径、预测步长、输入长度、batch size 等参数是否与实验设置一致。

## 八、数据集说明

由于时间序列数据集文件较大，本仓库不直接上传完整数据集。复现实验时需要按照代码中的数据路径自行放置数据文件。

常见数据集包括：

```text
ETTh1
ETTm1
Electricity
Traffic
Weather
```

建议将数据集统一放在代码指定的数据目录下，并保证文件名与配置文件一致。
