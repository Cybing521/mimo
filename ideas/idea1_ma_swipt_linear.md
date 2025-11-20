# Idea 1: Joint Optimization for MA-MIMO SWIPT (Linear EH)

## 1. 核心概念 (Core Concept)
将可移动天线 (MA) 引入传统的 SWIPT 系统，但首先基于**线性能量收集模型**进行研究。这是最直接的结合方案，作为基准线 (Baseline)。

## 2. 问题描述 (Problem Statement)
- **现状**: 现有的 SWIPT 研究多基于固定位置天线 (FPA)。虽然已有关于“流体天线 (FAS) + SWIPT”的初期研究，但多集中在单用户场景，且 FAS 与 MA 建模有所不同。
- **缺失**: 基于 Ma et al. (2023) 的 Movable Antenna 模型的多用户 MIMO SWIPT 系统性分析尚属空白。
- **局限**: 当信道条件不佳（如深衰落）时，单纯依靠波束成形很难同时满足高 SIR 和高能量接收。
- **机会**: MA 提供了额外的空间自由度。通过移动天线，可以找到信道增益更强的“热点”用于能量传输，同时利用空间零陷 (Spatial Nulling) 降低对 ID 接收机的干扰。

## 3. 方案设计 (Methodology)

### 3.1 系统模型
- **发射机**: 配备 $N$ 根可移动天线。
- **接收机**: 单个/多个 ID 用户 + 单个/多个 EH 用户 (线性模型 $E = \eta P_{in}$)。
- **信道**: MA 信道模型 (基于 `core/mimo_core.py`)。

### 3.2 优化问题
$$
\begin{aligned}
\max_{\mathbf{t}, \mathbf{Q}} \quad & R(\mathbf{t}, \mathbf{Q}) \\
\text{s.t.} \quad & E(\mathbf{t}, \mathbf{Q}) \ge E_{th} \\
& \text{MA 移动区域约束}
\end{aligned}
$$

### 3.3 算法思路
- **交替优化 (AO)**:
  1. 固定位置 $\mathbf{t}$，优化协方差矩阵 $\mathbf{Q}$ (凸优化/SDR)。
  2. 固定 $\mathbf{Q}$，利用梯度下降或逐次凸逼近 (SCA) 优化位置 $\mathbf{t}$。

## 4. 预期贡献
- 展示 MA 相比 FPA 在 Rate-Energy (R-E) Region 上的显著扩展。
- 证明 MA 可以降低对发射功率的需求。

## 5. 可行性分析
- **代码复用**: 80% 代码可复用。需将 `swipt_core.py` 中的约束加入 `mimo_core.py` 的位置优化循环中。

