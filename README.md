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
```

### 2. 运行仿真 (Ma 2023)

使用通用脚本 `universal_simulation.py` 复现 Ma 2023 论文的所有图表：

#### 基本用法
```bash
python universal_simulation.py --sweep_param [参数名] --range [起始] [结束] [步长] [其他固定参数...]
```

*(详情见上文参数详解...)*

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
