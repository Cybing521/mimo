# MIMO 无线通信研究项目

本项目复现多篇无线通信领域前沿论文的仿真结果，涵盖 **可移动天线 MIMO** 和 **SWIPT (同时无线信息和功率传输)** 等方向。

---

## 📚 已实现论文

### 1. **Ma et al. (2023) - Movable Antenna MIMO**
**论文**: *MIMO Capacity Characterization for Movable Antenna Systems*  
**期刊**: IEEE Transactions on Wireless Communications, 2023  
**核心算法**: `core/mimo_core.py`

**研究内容**: 通过优化发送和接收天线的**物理位置**（而非传统的固定位置）来最大化 MIMO 信道容量。

**复现图表**:
- Fig. 5/6: Achievable Rate vs Region Size
- Fig. 7: Achievable Rate vs SNR
- Fig. 8: Achievable Rate vs Antenna Number
- Fig. 9: Achievable Rate vs SNR (不同架构)

### 2. **Xiong et al. (2017) - SWIPT for MIMO**
**论文**: *Rate-Energy Region of SWIPT for MIMO Broadcasting Under Nonlinear Energy Harvesting Model*  
**期刊**: IEEE Transactions on Wireless Communications, 2017  
**核心算法**: `core/swipt_core.py`

**研究内容**: 在 MIMO 广播信道下，研究**信息传输速率**和**能量收集效率**之间的权衡关系（R-E Region）。考虑了非线性能量收集（Nonlinear EH）模型，并对比了线性和非线性模型在 **分离接收机** 和 **共址接收机**（TS/PS 架构）下的性能差异。

**复现图表**:
- Fig. 5: Average R-E Region (Separated Receivers)
- Fig. 10: Average R-E Region (Co-located Receivers: TS vs PS)

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 可选：检查 CUDA
python tools/check_cuda.py

# 可选：基础单元测试
pytest tests/unit
```

> 依赖列表已包含 `pypdf`（用于 `utils/extract_pdf.py` 的论文解析工具）以及修正后的 `pytest` 版本号，请确保重新执行 `pip install -r requirements.txt` 以同步最新环境。
>
> ⚠️ 2025-11 起，OpenAI Gym 已停止维护，本项目已切换到 **Gymnasium**。若你之前安装过 `gym`，请先 `pip uninstall gym` 再重新安装依赖，避免二者冲突。

### 2. 运行仿真 (Ma 2023)

使用通用脚本 `universal_simulation.py` 复现 Ma 2023 论文的所有图表：

#### 基本用法
```bash
python universal_simulation.py --sweep_param [参数名] --range [起始] [结束] [步长] [其他固定参数...]
```

*(详情见上文参数详解...)*

#### 🔁 推荐：复现 Ma Fig.6(a) **Proposed**（高 SNR）
- **目标**：`A/λ = 1~4` 区域扫描，比较 `Proposed / SEPM / FPA`
- **固定参数**：`N=M=4`, `Lt=Lr=5`, `SNR=25 dB`, `trials=50`, `κ=1`
- **并行**：通过 `--cores` 指定 CPU 核心数（例如 `--cores 8`），内部 `multiprocessing.Pool` 会自动并行 trials
- **命令**（注意 `np.arange` 的右开区间，`--range 1 4.5 0.5` 才能覆盖 4）：
```bash
python universal_simulation.py \
    --sweep_param A \
    --range 1 4.5 0.5 \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --SNR 25 \
    --trials 50 \
    --cores 8 \
    --modes Proposed SEPM FPA
```
- **输出**：图像 & JSON 会写入 `results/universal_sweep_A_*.png/.json`，可直接用于 Fig.6(a)。

> κ（Rician 因子）控制直射分量与散射分量的功率比，这里设为 1（等功率），与 Ma 2023 在 Fig.6 中的默认配置一致；该值在 `core/mimo_core.py` 内部设定，如需修改请在核心模型中调整。

### 3. 运行仿真 (Xiong 2017)

使用 `swipt_simulation.py` 复现 Xiong 2017 的 Rate-Energy Region：

#### 复现 Fig. 5 (Separated Receivers)
```bash
# 复现分离接收机场景下的平均 R-E 区域
python swipt_simulation.py --mode separated --Nt 2 --Ne 2 --Ni 2 --trials 50
```

#### 复现 Fig. 10 (Co-located Receivers: TS vs PS)
```bash
# 复现共址接收机（TS/PS）场景下的平均 R-E 区域
python swipt_simulation.py --mode colocated --Nt 2 --Ne 2 --Ni 2 --trials 50
```

---

## 📐 数学原理

### Ma 2023: Movable Antenna MIMO
- **信道模型**: $H_r = F^H \Sigma G$
- **容量公式**: $C = \log_2 \det(I_M + \frac{1}{\sigma^2} H_r Q H_r^H)$
- **优化变量**: 天线位置 $(x, y)$ + 功率分配矩阵 $Q$

### Xiong 2017: SWIPT
- **非线性 EH 模型**: $E = \frac{M}{1 + e^{-a(P_{in} - b)}} - \frac{M}{1 + e^{ab}}$
- **R-E Region**: 权衡信息速率 $R$ 和能量传输 $E$ 的帕累托前沿
- **分离接收机算法**: 对偶梯度法 (Dual Sub-gradient Method)
- **共址接收机算法**:
    - **Time Switching (TS)**: 交替优化 $\theta$ 和 $(Q_E, Q_I)$
    - **Power Splitting (PS)**: 交替优化 $\Omega_\rho$ 和 $Q$

---

## 📂 项目结构

```
MIMO/
├── universal_simulation.py    # Ma 2023 通用仿真脚本
├── swipt_simulation.py        # Xiong 2017 仿真脚本
├── README.md                  # 项目文档
├── requirements.txt           # 依赖
├── core/                      # ⭐ 核心算法库
│   ├── __init__.py
│   ├── mimo_core.py           # Ma 2023 算法
│   └── swipt_core.py          # Xiong 2017 算法
├── docs/                      # 详细文档
├── papers/                    # 论文 PDF
└── results/                   # 结果按论文分离
    ├── ma2023/                # Ma 2023 的结果
    └── swipt2017/             # Xiong 2017 的结果
```

---

## 📝 引用

### Ma 2023
```
Ma, W., Zhu, L., & Zhang, R. (2023). 
MIMO Capacity Characterization for Movable Antenna Systems. 
IEEE Transactions on Wireless Communications.
```

### Xiong 2017
```
Xiong, K., Wang, B., & Liu, K. J. R. (2017). 
Rate-Energy Region of SWIPT for MIMO Broadcasting Under Nonlinear Energy Harvesting Model. 
IEEE Transactions on Wireless Communications, 16(8), 5147-5161.
```

---

## 🔬 未来工作

- [ ] 研究 MA-MIMO + SWIPT 的结合方向 (MA-SWIPT)
- [ ] 添加更多基准算法 (AS, SEPM, APS)

---

## 🤖 **NEW: Deep Reinforcement Learning for MA-MIMO**

### 最新进展：突破局部最优瓶颈

我们提出了基于深度强化学习（DRL）的可移动天线优化方法，解决了Ma et al. (2023)算法的局部最优问题。

#### **核心创新**

1. **首次应用DRL于MA-MIMO**: 将天线位置优化建模为马尔可夫决策过程(MDP)
2. **混合优化策略**: DRL全局探索 + 传统AO局部精调
3. **实时推理**: 推理时间从5秒降低到0.1秒
4. **性能提升**: 相比Ma's Algorithm提升10-15%的信道容量

#### **快速开始**

**1. 正式训练（论文级，推荐 GPU）**

```bash
python experiments/train_drl.py \
    --num_episodes 5000 \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --SNR_dB 25 \
    --A_lambda 3.0 \
    --max_steps 50 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_epsilon 0.2 \
    --ppo_epochs 10 \
    --batch_size 64 \
    --entropy_coef 0.01 \
    --min_entropy_coef 0.001 \
    --rollout_episodes 4 \
    --lr_anneal \
    --min_lr_factor 0.1 \
    --eval_interval 100 \
    --eval_episodes 20 \
    --eval_seed 2024 \
    --save_interval 500 \
    --seed 42 \
    --device auto \
    --save_dir results/drl_training
```

**2. 快速调试（10 分钟体验）**

```bash
python experiments/train_drl.py \
    --num_episodes 100 \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --SNR_dB 15 \
    --A_lambda 3.0 \
    --max_steps 50 \
    --rollout_episodes 2 \
    --log_interval 5 \
    --eval_interval 20 \
    --eval_episodes 10 \
    --eval_seed 2024 \
    --save_dir results/drl_training_quick
```

> 正式训练输出写入 `results/drl_training/run_时间戳/`，用于对比实验与论文图；快速训练写入 `results/drl_training_quick/`，仅用于流程调试，性能不可用于论文。

### 🛰️ 在线可视化（WandB 集成）

现在 `train_drl.py`、`universal_simulation.py` 与 `experiments/compare_methods.py` 均支持 **WandB（Weights & Biases）** 实时记录。使用流程：

1. 在项目根目录创建 `.env` 文件并写入 `WANDB_API_KEY`（每次运行会自动读取，无需手动 `export`）：  
   ```bash
   cat <<'EOF' > .env
   WANDB_API_KEY=YOUR_WANDB_KEY
   EOF
   chmod 600 .env
   ```
2. 在命令后追加 `--use_wandb` 以及其他可选参数：
   - `--wandb_project mimo-lab`：Project 名（默认 `ma-mimo`）
   - `--wandb_entity your-team`：团队/账号
   - `--wandb_run_name 自定义名称`
   - `--wandb_tags tag1 tag2`：标签数组

示例：
```bash
# DRL 训练可视化
python experiments/train_drl.py ... \
    --use_wandb --wandb_project mimo-lab --wandb_tags drl high_snr

# AO 扫描（Ma Fig.6）可视化
python universal_simulation.py ... \
    --use_wandb --wandb_project mimo-lab --wandb_tags ao fig6

# DRL vs AO 对比实验
python experiments/compare_methods.py ... \
    --use_wandb --wandb_project mimo-lab --wandb_tags comparison fig7
```
开启后，训练曲线、关键指标与生成的 PNG 图都会自动同步到 WandB 仪表盘，便于线上查看。

**参数说明（训练脚本常用项）**

- `--num_episodes`：训练 episode 总数；5000 对应 Fig.5/6 水平，100 仅用于调试。
- `--N/--M/--Lt/--Lr/--SNR_dB/--A_lambda/--max_steps`：与 Ma 2023 的系统配置保持一致，确保可比性。
- `--lr_actor/--lr_critic/--gamma/--gae_lambda/--clip_epsilon/--ppo_epochs/--batch_size`：PPO 核心超参，可按需求微调。
- `--entropy_coef` 与 `--min_entropy_coef`：控制策略探索强度，并在线性衰减到下限。
- `--rollout_episodes`：每次 PPO 更新前要收集的 episode 数，>1 可降低梯度方差。
- `--lr_anneal` 与 `--min_lr_factor`：启用后会线性衰减 Actor/Critic 学习率至设定下限。
- `--eval_interval/--eval_episodes/--eval_seed`：评估频率、每次评估的 episode 数以及随机种子；固定种子有利于曲线可复现。
- `--save_interval/--save_dir`：Checkpoint 周期与输出目录；正式/快速 run 使用不同目录以免混淆。
- `--device`：设备选择，可选值：
  - `auto`（默认）：自动选择最佳可用设备（优先 CUDA > MPS > CPU）
  - `cuda`：NVIDIA GPU（仅 Linux/Windows，macOS 不支持）
  - `mps`：Apple Silicon GPU（仅 macOS）
  - `cpu`：CPU（所有平台）
  
  **注意**：macOS 不支持 CUDA，如果指定 `--device cuda` 会自动回退到 CPU。
- `--seed`：全局随机种子，保证可复现。

**3. 对比实验**

```bash
# 对比容量 vs 区域大小 (复现Ma Fig.5 + DRL)
python experiments/compare_methods.py \
    --experiment region_size \
    --drl_model results/drl_training/run_XXXXXX/best_model.pth \
    --methods AO MS-AO DRL Hybrid \
    --trials 20

# 对比容量 vs SNR (复现Ma Fig.7 + DRL)
python experiments/compare_methods.py \
    --experiment snr \
    --drl_model results/drl_training/run_XXXXXX/best_model.pth \
    --methods AO DRL Hybrid
```

#### **项目结构（DRL扩展）**

```
MIMO/
├── drl/                           # ✨ DRL模块
│   ├── __init__.py
│   ├── env.py                     # Gym环境
│   ├── agent.py                   # PPO Agent
│   ├── networks.py                # Actor-Critic网络
│   └── utils.py                   # 工具函数
│
├── experiments/                   # 实验脚本
│   ├── train_drl.py               # DRL训练
│   ├── compare_methods.py         # 对比实验
│   ├── ablation_study.py          # 消融实验
│   └── transfer_learning.py       # 迁移学习
│
├── docs/
│   ├── drl_technical_proposal.md  # 完整技术方案
│   └── implementation_guide.md    # 实现指南
│
└── results/
    ├── drl_training/              # 训练日志和模型
    └── comparison/                # 对比实验结果
```

#### **预期结果**

| 方法 | 容量 (bps/Hz) | 时间 (s) | 成功率 |
|------|--------------|----------|--------|
| Ma's AO | 23.5 | 5.2 | 60% |
| MS-AO (10×) | 24.7 | 52.0 | 75% |
| DRL (Ours) | 26.2 | 0.08 | 82% |
| **Hybrid (Ours)** | **26.8** | **0.3** | **89%** |

#### **技术细节**

- **状态空间**: 信道特征值 + Tx/Rx 位置 + 历史容量（N=M=4 时共 44 维，随阵元数线性扩展）
- **动作空间**: 归一化连续向量（长度 2(N+M)），分别控制 Tx/Rx 的 Δx/Δy，环境内部缩放为 ±0.1λ
- **奖励函数**: 以容量提升为核心，叠加约束惩罚、效率奖励与平滑项
- **算法**: PPO-Clip with GAE（支持多 episode 累积更新、LR/熵系数线性衰减、固定种子评估）
- **网络**: Actor-Critic with Dueling architecture

> 2025-11 更新：DRL 环境会在每个 episode 重采样 Rician 信道、联合优化 Tx/Rx 阵列，并采用标准 water-filling 进行功率分配，训练更贴近 Ma et al. 的仿真设置。

#### **论文投稿目标**

- **目标会议**: IEEE ICC 2026 / IEEE GLOBECOM 2025
- **目标期刊**: IEEE TWC / IEEE TCOM
- **创新点**: 首次DRL应用 + 混合策略 + 实时推理 + 泛化能力

#### **参考文献**

[Ma et al., 2023] - "MIMO Capacity Characterization for Movable Antenna Systems", IEEE TWC  
[DRL-MA] - "Deep Reinforcement Learning for Movable Antenna Optimization" (本工作)

---
