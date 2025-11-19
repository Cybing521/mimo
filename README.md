# MIMO 可移动天线容量特性仿真

本项目复现了 Ma 等人 (2023) 的论文《MIMO Capacity Characterization for Movable Antenna》中的仿真结果，重点关注**Proposed 方案**（收发两端联合位置优化）在不同信噪比和区域大小下的信道容量特性。

## 📚 项目概述

本项目实现了一个基于交替优化（Alternating Optimization）的 MIMO 系统信道容量最大化算法。通过联合优化发送功率分配矩阵 ($Q$) 和收发天线位置 ($r, t$)，在满足最小距离约束的前提下最大化信道容量。

### 核心特性
- **完整复现**: 实现了论文中的 Algorithm 2（Proposed 方案），包括收发两端的联合优化。
- **性能优化**: 相比原始 MATLAB 代码，Python 版本引入了多进程并行加速、稳健的优化器和数值稳定性改进。
- **严格验证**: 参数设置（SNR, Lt, Lr）已严格对齐论文标准，算法逻辑已通过维度和流程验证。

---

## 🚀 快速开始

### 1. 环境准备
确保安装 Python 3.8+，然后安装依赖：

```bash
# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行仿真

本项目使用 `universal_simulation.py` 通用脚本来复现论文中的所有图表，支持自定义扫描参数。详细的实验参数说明请参考 [Experiment Parameters](docs/experiment_parameters.md)。

#### 基本用法
```bash
python universal_simulation.py --sweep_param [参数名] --range [起始] [结束] [步长] [其他固定参数...]
```

#### 2.1 复现 Fig. 7: Achievable Rate vs SNR (不同散射体数)
```bash
# Lt=Lr=3
python universal_simulation.py --sweep_param SNR --range -15 16 5 --A 3 --Lt 3 --Lr 3 --N 4 --M 4 --modes Proposed RMA TMA FPA

# Lt=Lr=6
python universal_simulation.py --sweep_param SNR --range -15 16 5 --A 3 --Lt 6 --Lr 6 --N 4 --M 4 --modes Proposed RMA TMA FPA
```

#### 2.2 复现 Fig. 8: Achievable Rate vs M(N)
```bash
# 固定 SNR=5dB, A=4λ, Lt=Lr=5
python universal_simulation.py --sweep_param antennas --range 2 9 1 --SNR 5 --A 4 --Lt 5 --Lr 5 --modes Proposed RMA TMA FPA
```

#### 2.3 复现 Fig. 9: Achievable Rate vs SNR (不同架构)
```bash
# 对比 MA-FD 和 FPA-FD (全数字架构)
python universal_simulation.py --sweep_param SNR --range -15 16 5 --A 4 --N 6 --M 6 --Lt 5 --Lr 5 --modes MA-FD FPA-FD
```

#### 2.4 复现 Fig. 5 & 6: Achievable Rate vs Region Size
```bash
# Fig. 6: High SNR (25dB)
python universal_simulation.py --sweep_param A --range 1 5 1 --SNR 25 --Lt 10 --Lr 15 --modes Proposed RMA
```

### 3. 参数详解
| 参数 | 说明 | 示例 |
|---|---|---|
| `--sweep_param` | **必选**。要扫描变化的参数 (X轴)。可选: `SNR`, `antennas`, `A`, `Lt`, `Lr` | `--sweep_param SNR` |
| `--range` | **必选**。扫描范围 (起始 结束 步长)。注意：不包含结束值 (Python range 习惯)。 | `--range -15 16 5` |
| `--modes` | 要对比的模式列表。 | `--modes Proposed RMA` |
| `--trials` | 每个数据点的蒙特卡洛试验次数。 | `--trials 50` |
| `--cores` | 并行核数。 | `--cores 4` |
| `--N` / `--M` | 天线数量 (若不扫描则固定)。 | `--N 4 --M 4` |
| `--Lt` / `--Lr` | 散射体数量 (若不扫描则固定)。 | `--Lt 5` |
| `--A` | 区域大小 (波长倍数)。 | `--A 4` |
| `--SNR` | 信噪比 (dB)。 | `--SNR 10` |

### 4. 输出结果
脚本运行结束后，结果将自动保存至 `results/` 目录：
- **PNG**: 自动生成的折线图。
- **JSON**: 包含完整数据点的元数据文件。

---

## 📐 数学原理与参数说明

### 系统模型
- **信道模型**: $H_r = F^H \Sigma G$
    - $G$: 发送端阵列响应 ($L_t \times N$)
    - $\Sigma$: 散射矩阵 ($L_r \times L_t$)，Rician 衰落
    - $F$: 接收端阵列响应 ($L_r \times M$)
- **容量公式**: $C = \log_2 \det(I_M + \frac{1}{\sigma^2} H_r Q H_r^H)$

### 算法流程 (Algorithm 2)
1. **初始化**: 随机生成满足约束的天线位置。
2. **交替优化循环**:
    - **步骤 A**: 固定位置，优化功率 $Q$ (凸优化/注水算法)。
    - **步骤 B**: 固定 $Q$ 和 $t$，优化接收位置 $r$ (SCA/梯度下降)。
    - **步骤 C**: 固定 $Q$ 和 $r$，优化发送位置 $t$ (SCA/梯度下降)。
3. **收敛判定**: 当容量相对增量小于阈值 ($\epsilon=10^{-3}$) 时停止。

---

## 📂 文件结构

**通用脚本**:
- `universal_simulation.py`: **核心仿真脚本**。集成了所有功能的通用仿真入口。

**核心模块**:
- `mimo_optimized.py`: **核心算法实现** (MIMOSystem 类)。包含所有系统建模、信道生成和优化算法逻辑。此文件为`universal_simulation.py`提供底层支持，不可删除。

**文档**:
- `docs/`: 包含详细的参数说明和图表总结。
    - `experiment_parameters.md`: 实验参数详细配置。
    - `figures_summary.md`: 论文图表复现说明。

**其他**:
- `requirements.txt`: 项目依赖。
- `results/`: 仿真结果输出目录。
- `papers/`: 存放相关研究论文 PDF。

---

## 📝 引用
如果您使用本代码进行研究，请引用原始论文：
Ma, W., et al. "MIMO Capacity Characterization for Movable Antenna." *IEEE Transactions on Wireless Communications*, 2023.
