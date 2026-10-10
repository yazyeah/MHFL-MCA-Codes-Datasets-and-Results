<a id="top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mhfl-mca-banner-dark-v3.png" />
    <img src="assets/mhfl-mca-banner-v3.png" width="850" alt="MHFL-MCA" />
  </picture>
</p>

<p align="center">
  <a href="https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results/stargazers"><img src="https://img.shields.io/github/stars/yazyeah/MHFL-MCA-Codes-Datasets-and-Results?style=social" alt="GitHub stars" /></a>
  <a href="LICENSE_SCOPE.md"><img src="https://img.shields.io/badge/license-MIT-green" alt="Code license: MIT" /></a>
  <a href="https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results/commits"><img src="https://img.shields.io/github/last-commit/yazyeah/MHFL-MCA-Codes-Datasets-and-Results" alt="Last commit" /></a>
  <a href="https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results/pulls"><img src="https://img.shields.io/badge/PRs-welcome-blue" alt="PRs welcome" /></a>
  <a href="ENVIRONMENT.md"><img src="https://img.shields.io/badge/TensorFlow-2.7-orange?logo=tensorflow&logoColor=white" alt="TensorFlow 2.7" /></a>
  <a href="https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results/actions/workflows/repository-audit.yml"><img src="https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results/actions/workflows/repository-audit.yml/badge.svg" alt="Repository audit" /></a>
  <a href="https://deepwiki.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results"><img src="https://img.shields.io/badge/DeepWiki-Ask%20DeepWiki-435b75" alt="Ask DeepWiki" /></a>
</p>

<p align="center">
  中文 | <a href="README_en.md">English</a>
</p>

- MHFL-MCA 面向小样本、变载荷和噪声干扰条件下的多模态轴承故障诊断。
- 仓库包含论文对应的 TensorFlow/Keras 实现、对比方法、实验结果和补充实验。
- UO 实验使用振动与声学信号，KAIST 实验使用振动与电机电流信号；两路输入均为 2048 点的一维片段。
- 对比实验包含五种深度学习方法；补充实验包括消融、模态缺失和极低样本分析。
- 数据集与实验图表由 Git LFS 管理，支持按实验分别下载。

> 作者自有代码和文档采用 MIT 协议，数据遵循各自的上游许可。模型权重清单和校验值已收录，权重文件尚未发布。

<p align="center">
  <a href="https://doi.org/10.1016/j.aei.2026.105277">论文</a> |
  <a href="#method">方法框架</a> |
  <a href="#quick-start">快速开始</a> |
  <a href="#datasets">数据与实验</a> |
  <a href="#results">实验结果</a> |
  <a href="https://deepwiki.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results">DeepWiki</a>
</p>

---

# 📌 项目介绍

振动、声学和电流信号的统计特征不同，采用相同的编码结构再拼接特征，可能丢失模态间的互补信息。MHFL-MCA 用模态专属卷积分支提取特征，通过双向交叉注意力完成模态间交互，并按样本调整融合权重，输出健康状态分类结果。

网络包括模态专属异构编码器（MSHEM）、交叉注意力交互模块（CAIM）、自适应模态重加权模块（AMRM）和融合分类模块（FCB）。方法与实验设置详见论文：

> [Multimodal heterogeneous feature learning with modality-cross-attention for robust fault diagnosis of machines under complex conditions](https://doi.org/10.1016/j.aei.2026.105277)<br>
> Yuang Zhou, Tianci Zhang, Tongyang Pan, Jingsong Xie, Jinglong Chen<br>
> *Advanced Engineering Informatics*, 77, 105277, 2027.

<a id="method"></a>

## 方法框架

框架由异构编码、双向交互、模态重加权和分类四部分组成。模态 1 为振动，模态 2 为声学（UO）或电流（KAIST）。

<p align="center">
  <img src="MHFL-MCA%20Supplementary%20Experiments/figures/MHFL-MCA_architecture.jpeg" width="800" alt="MHFL-MCA network architecture" />
</p>

### 1. 异构特征编码（MSHEM）

对样本 $i$，两路输入为 $\mathbf{x}_i^{(1)},\mathbf{x}_i^{(2)}\in\mathbb{R}^{2048}$。独立参数化的编码器提取模态特征：

```math
F_i^{(m)}=E_m(\mathbf{x}_i^{(m)};\theta_m)
\in\mathbb{R}^{T_m\times C_m},\qquad m\in\{1,2\}.
```

$T_m$ 和 $C_m$ 分别为特征图的时间长度和通道数。振动分支采用卷积核大小 16、LeakyReLU 和平均池化；声学或电流分支采用卷积核大小 8、GELU 和最大池化。

### 2. 双向交叉注意力（CAIM）

方向 $a\leftarrow b$ 表示模态 $a$ 提供查询，模态 $b$ 提供键和值，其中 $a,b\in\{1,2\}$ 且 $a\ne b$。两个方向分别使用独立的投影参数：

```math
\begin{aligned}
Q_i^{a\leftarrow b}&=F_i^{(a)}W_q^{a\leftarrow b},\\
K_i^{a\leftarrow b}&=F_i^{(b)}W_k^{a\leftarrow b},\qquad
V_i^{a\leftarrow b}=F_i^{(b)}W_v^{a\leftarrow b},\\
G_i^{a\leftarrow b}&=\mathrm{softmax}_{\mathrm{row}}
\!\left(Q_i^{a\leftarrow b}(K_i^{a\leftarrow b})^\top\right)
V_i^{a\leftarrow b},\qquad a\ne b.
\end{aligned}
```

Softmax 沿键的时间维度计算，使每个查询位置聚合另一模态的信息。两路交互输出 $G_i^{1\leftarrow2}$ 和 $G_i^{2\leftarrow1}$ 的形状分别为 $T_1\times D$ 和 $T_2\times D$，其中 $D$ 为注意力投影维度。

### 3. 模态重加权与融合（AMRM）

交互特征展平后，分别生成 Sigmoid 标量评分，再通过二输出全连接层和 Softmax 得到样本级模态权重：

```math
\begin{aligned}
g_i^{(1)}&=\mathrm{vec}(G_i^{1\leftarrow2}),\qquad
g_i^{(2)}=\mathrm{vec}(G_i^{2\leftarrow1}),\\
s_i^{(m)}&=\sigma(w_m^\top g_i^{(m)}+b_m),\\
\boldsymbol{\alpha}_i&=\mathrm{softmax}
\!\left(W_s[s_i^{(1)};s_i^{(2)}]+b_s\right).
\end{aligned}
```

权重满足 $\alpha_i^{(1)}+\alpha_i^{(2)}=1$，融合向量由两路加权特征拼接而成：

```math
z_i=[\,\alpha_i^{(1)}g_i^{(1)};\ \alpha_i^{(2)}g_i^{(2)}\,]
\in\mathbb{R}^{(T_1+T_2)D}.
```

$[\,\cdot\,;\,\cdot\,]$ 表示向量拼接，$\sigma$ 表示 Sigmoid。每个样本的权重由两路交互特征共同决定。

### 4. 健康状态分类（FCB）

融合向量经过 256 维隐藏层和 LeakyReLU，再由 Softmax 输出 $K$ 类健康状态的概率。基础监督目标为交叉熵：

```math
\begin{aligned}
p_i&=\mathrm{softmax}\!\left(
W_c\mathrm{LeakyReLU}(W_fz_i+b_f)+b_c\right),\\
\mathcal{L}_{\mathrm{CE}}&=-\frac{1}{M}
\sum_{i=1}^{M}\sum_{k=1}^{K}y_{ik}\log p_{ik}.
\end{aligned}
```

$M$ 为批次样本数，$y_{ik}$ 为独热标签的第 $k$ 个分量，$p_{ik}$ 为预测概率。KAIST 的噪声稳定训练还包括 KL 一致性约束，训练设置见 [MHCNN_KAIST.py](Codes/Codes%20for%20KAIST/MHCNN_KAIST.py)。上述计算对应主实现中的 `CrossAttention` 和 `create_model`：[UO](Codes/Codes%20for%20UO/MHCNN_UO.py) | [KAIST](Codes/Codes%20for%20KAIST/MHCNN_KAIST.py)。

代码模块、数据流程和实验入口的在线索引：[DeepWiki](https://deepwiki.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results)。

---

<a id="quick-start"></a>

# 📌 快速开始

代码和数据下载依赖 Git 与 [Git LFS](https://git-lfs.com/)，Python 环境由 Conda 管理。命令使用 Windows PowerShell，环境创建在 Conda PowerShell 中执行。

## 第 0 步：获取代码

克隆命令跳过 LFS 自动下载，数据和历史结果可按实验单独获取。

```powershell
git lfs install
$env:GIT_LFS_SKIP_SMUDGE = '1'
git -c core.longpaths=true clone https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results.git
Set-Location .\MHFL-MCA-Codes-Datasets-and-Results
git config core.longpaths true
Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
```

## 第 1 步：创建环境

实验环境使用 Python 3.9.20 和 TensorFlow/Keras 2.7.0，完整依赖见 [requirements-lock.txt](requirements-lock.txt)。

```powershell
conda env create -f environment.yml
conda activate mhfl-mca-revision
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pip check
python tools\verify_repository_contract.py
```

`verify_repository_contract.py` 检查仓库文件和实验记录，不运行模型训练。环境与 GPU 配置见 [ENVIRONMENT.md](ENVIRONMENT.md) | [HARDWARE.md](HARDWARE.md)。

## 第 2 步：下载数据

UO 和 KAIST 数据的下载命令分别为：

```powershell
# UO：实验使用的 MATLAB 数据
git lfs pull --include="Datasets/UO/3_MatLab_Raw_Data/**" --exclude=""

# KAIST：振动和电流数据
git lfs pull --include="Datasets/KAIST/**" --exclude=""
```

数据下载和文件核验方法见 [DATA.md](DATA.md)。几百字节的 `.mat` 或 `.tdms` 文件通常是 LFS 指针，实际数据尚未下载。

## 第 3 步：运行实验

主实验脚本沿用早期的 `MHCNN` 文件名，实现的方法为 MHFL-MCA。运行前需将脚本中的 `YOUR_*` 数据、输出和临时目录替换为实际路径。

```powershell
# UO 主实验
python ".\Codes\Codes for UO\MHCNN_UO.py"

# KAIST 主实验
python ".\Codes\Codes for KAIST\MHCNN_KAIST.py"
```

补充实验采用独立配置和运行入口，具体命令见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | [补充实验说明](MHFL-MCA%20Supplementary%20Experiments/README.md)。

<details>
<summary>KAIST 通道选择、模型权重与复现说明</summary>

原始 KAIST 脚本与补充实验采用不同的通道选择规则。原始脚本在振动数组至少有 5 列时选择零基列 `1`，电流 U 相识别失败时选择最大方差通道；两种规则分别由 `_pick_vib_1d` 和 `read_current_1d_from_tdms` 实现。复现结果应按各自的实验协议比较。

补充实验固定选择振动零基列 `0`，以及 TDMS `Log` 组的 `cDAQ9185-1F486B5Mod2/ai0` 电流通道。补充实验的环境变量不会控制原始脚本。

权重清单、SHA-256 和训练重建方法见 [MODEL_ZOO.md](MODEL_ZOO.md)。历史初始化和随机流未完整记录，重新训练不保证得到逐字节相同的权重。

网络结构图和补充图表的下载命令：

```powershell
git lfs pull --include="MHFL-MCA Supplementary Experiments/**" --exclude=""
```

</details>

---

<a id="datasets"></a>

# 📌 数据与实验

| 数据集 | 使用的信号 | 实验设置 | 包含的文件 |
| :--- | :--- | :--- | :--- |
| UO / UORED-VAFCLS | 振动 + 声学 | 7 类健康状态，小样本诊断 | 60 条记录，分别提供 CSV、XLSX 和 MAT 表示。 |
| KAIST | 振动 + 电流 | 5 类健康状态；0 Nm 训练/验证，2 和 4 Nm 测试 | 15 个振动 MAT 文件和 15 个电流 TDMS 文件。 |

官方数据来源：[UO](https://data.mendeley.com/datasets/y2px5tg92h) | [KAIST v6](https://data.mendeley.com/datasets/ztmf3m7h5x/6)。KAIST 实验使用其中的选定子集；UO 文件的精确上游版本尚未确认（`UNCONFIRMED`）。数据许可和署名要求见 [DATA_LICENSES.md](DATA_LICENSES.md)。

UO MATLAB 数据约 0.282 GiB，KAIST 数据约 2.027 GiB；全部 LFS 对象（含历史结果）约 8.858 GiB。

```text
Codes/                              UO、KAIST 主实验与对比方法
Datasets/                           原始数据（Git LFS）
Results/                            主实验结果与可视化
MHFL-MCA Supplementary Experiments/
├── Codes/Revision Experiments/     补充实验代码与配置
├── Results/Revision Experiments/   补充实验结果
├── Provenance/                     数据来源与核验记录
└── figures/                        网络结构图
```

主实验包含 MRCFN、CFFN、CDTFAFN、MSF-DFormer 和 KDCNN-DF 五种对比方法。代码与结果：[UO](Codes/Codes%20for%20UO/) | [KAIST](Codes/Codes%20for%20KAIST/) | [Results](Results/)。

| 补充实验 | 内容 |
| :--- | :--- |
| [02](MHFL-MCA%20Supplementary%20Experiments/Results/Revision%20Experiments/full_20260807_r1/02_Modality_Weights_and_Missing/README.md) | 模态权重与模态缺失分析 |
| [03](MHFL-MCA%20Supplementary%20Experiments/Results/Revision%20Experiments/full_20260807_r1/03_Efficiency_Profile/README.md) | 参数量、计算量与推理效率 |
| [04](MHFL-MCA%20Supplementary%20Experiments/Results/Revision%20Experiments/full_20260807_r1/04_Additional_Controlled_Ablation/README.md) | 控制变量消融 |
| [05](MHFL-MCA%20Supplementary%20Experiments/Results/Revision%20Experiments/full_20260807_r1/05_Extreme_Lowshot_CAIM/README.md) | 极低样本与交叉注意力分析 |
| [06](MHFL-MCA%20Supplementary%20Experiments/Results/Revision%20Experiments/full_20260807_r1/06_Traditional_Baselines/README.md) | 传统 TF-SVM 对比 |

---

<a id="results"></a>

# 📈 实验结果

每类 5 个训练样本（N = 5）时的主文准确率，单位为 %：

| 方法 | UO | KAIST：0 → 2 Nm | KAIST：0 → 4 Nm |
| :--- | ---: | ---: | ---: |
| MRCFN | 65.21 ± 7.88 | 59.29 ± 4.46 | 53.84 ± 5.68 |
| CFFN | 76.83 ± 5.79 | 58.29 ± 4.18 | 52.80 ± 5.36 |
| CDTFAFN | 62.37 ± 7.60 | 65.32 ± 3.36 | 57.74 ± 5.08 |
| MSF-DFormer | 72.87 ± 6.54 | 74.55 ± 4.31 | 78.62 ± 4.66 |
| KDCNN-DF | 75.32 ± 2.82 | 64.27 ± 4.38 | 57.36 ± 5.44 |
| **MHFL-MCA** | **92.79 ± 2.88** | **97.39 ± 1.95** | **91.56 ± 8.32** |

每个指标在 10 次运行中去除最高和最低值，报告剩余 8 次的均值 ± 标准差。UO 使用原始振动与声学片段；KAIST 在 0 Nm 训练，在 2/4 Nm、0 dB 基线噪声条件下测试。

结果对应主文 Tables 5/9/10：[结果与协议](MHFL-MCA%20Supplementary%20Experiments/Codes/Revision%20Experiments/configs/main_manuscript_deep_reference.json) | [数值核验](MHFL-MCA%20Supplementary%20Experiments/Provenance/audits/ARTIFACT_VALUE_VERIFICATION.md)。

<details>
<summary>统计口径与实验 05 的协议</summary>

主文参考结果仅包含聚合值，没有逐种子记录，不能用于配对检验或统计显著性判断。

实验 05 中，Full 模型在 N = 5 和 N = 10 时的准确率来自主文，其余 Full 结果及去除 CAIM 的结果来自受控扩展。两种协议的结果分别统计，不合并为一个估计。来源与协议见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

</details>

---

# 🎓 引用

使用该方法或实验结果时，请引用：

```bibtex
@article{zhou2027mhflmca,
  title   = {Multimodal heterogeneous feature learning with modality-cross-attention for robust fault diagnosis of machines under complex conditions},
  author  = {Zhou, Yuang and Zhang, Tianci and Pan, Tongyang and Xie, Jingsong and Chen, Jinglong},
  journal = {Advanced Engineering Informatics},
  volume  = {77},
  pages   = {105277},
  year    = {2027},
  doi     = {10.1016/j.aei.2026.105277}
}
```

数据使用需同时引用对应数据集和数据论文，署名要求见 [DATA_LICENSES.md](DATA_LICENSES.md)。感谢 University of Ottawa 和 KAIST 数据集作者提供公开数据。

---

# 许可与贡献

作者自有代码和文档采用 [MIT](LICENSE) 协议。数据和其他第三方材料的适用范围见 [DATA_LICENSES.md](DATA_LICENSES.md) | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | [LICENSE_SCOPE.md](LICENSE_SCOPE.md)。

复现问题可提交至 [Issue](https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results/issues)，并附上环境版本、执行命令和日志。文档修正、数据加载和验证工具的改进可通过 [PR](https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results/pulls) 提交。

<p align="center">
  <a href="README_en.md">English</a> | <a href="#top">回到顶部</a>
</p>
