# 论文图表复现参数总结

本文档详细说明论文中所有仿真图表的参数配置和内容。

## Fig. 5: Performance of MA-enabled MIMO (Low-SNR)

### 参数配置
- **SNR**: -15 dB (低信噪比)
- **天线**: N=4, M=4
- **散射体**: Lt=5, Lr=5
- **区域大小**: A/λ = 1 ~ 3
- **κ** (Rician 因子): 1

### 子图内容
- **(a) Achievable Rate vs Region Size**
  - Y 轴范围: 0.4 ~ 1.2 bps/Hz
  - 关键结论: 低 SNR 下，SEPM 与 Proposed 性能接近
  
- **(b) Strongest Eigenchannel Power vs Region Size**
  - Y 轴范围: 10 ~ 40
  - 公式: $\max(\text{diag}(\tilde{\Lambda}))^2$

### 复现命令
```bash
python mimo_comparison.py --snr -15 --trials 50 --cores 4
```

---

## Fig. 6: Performance of MA-enabled MIMO (High-SNR)

### 参数配置
- **SNR**: 25 dB (高信噪比)
- **天线**: N=4, M=4
- **散射体**: Lt=5, Lr=5
- **区域大小**: A/λ = 1 ~ 4
- **κ**: 1

### 子图内容
- **(a) Achievable Rate vs Region Size**
  - **Y 轴范围: 18 ~ 30 bps/Hz**
  - **Proposed 在 A/λ=4 时: ~28-29 bps/Hz** ✓
  - 关键结论: 高 SNR 下，Proposed 显著优于 SEPM
  
- **(b) Channel Total Power vs Region Size**
  - **Y 轴范围: 15 ~ 40**
  - **SEPM 达到 ~38-40** (但 Capacity 反而最低！)
  - 公式: $\|\mathbf{H}\|_F^2 = \text{trace}(\mathbf{H}\mathbf{H}^H)$
  
- **(c) Condition Number vs Region Size**
  - Y 轴范围: 10^1 ~ 10^3 (对数刻度)
  - 公式: $\zeta = \lambda_{max} / \lambda_{min}$

### 复现命令
```bash
python mimo_comparison.py --snr 25 --trials 50 --cores 4
```

### ⚠️ 重要说明
**不要混淆 Fig. 6(b) 的 "Channel Total Power" (Y 轴到 40) 和 Fig. 6(a) 的 "Achievable Rate" (Y 轴到 30)！**

---

## Fig. 7: Achievable Rate vs SNR

### 参数配置
- **SNR 范围**: -15 ~ 15 dB
- **天线**: N=4, M=4
- **区域大小**: A = 3λ (固定)
- **κ**: 1
- **两种配置**:
  - **(a)**: Lt=3, Lr=3
  - **(b)**: Lt=6, Lr=6

### 关键观察
- SNR < -5 dB 时，Proposed ≈ SEPM
- 散射体越多 (Lr=6)，容量越高

### 复现命令
```bash
python reproduce_fig7.py --trials 50 --cores 4
```

---

## Fig. 8: Achievable Rate vs M(N)

### 参数配置
- **SNR**: **5 dB** (注意不是 25 dB！)
- **天线数量**: M=N = 2, 3, 4, 5, 6, 7, 8
- **散射体**: Lt=5, Lr=5
- **区域大小**: A = 4λ (固定)
- **κ**: 1

### 关键观察
- 即使 M(N) > Lt=5，容量仍会增加
- 原因: 总信道功率增益随天线数增加

### 复现命令
```bash
python reproduce_fig8.py --trials 50 --cores 4
```

---

## Fig. 9: Achievable Rate vs SNR (不同架构)

### 参数配置
- **SNR 范围**: -15 ~ 15 dB
- **天线**: **M=6, N=6** (注意是 6x6！)
- **散射体**: Lt=5, Lr=5
- **区域大小**: A = 4λ
- **κ**: 1
- **RF Chains**: K=3

### 对比架构
- **MA-FD**: Movable Antenna + Fully Digital
- **MA-AP**: Movable Antenna + Analog Precoding
- **MA-HP**: Movable Antenna + Hybrid Precoding
- **FPA-FD/AP/HP**: Fixed Position Antenna + 对应架构

### 关键结论
- MA 系统在所有架构下均优于 FPA
- 即使使用 AP/HP 减少 RF 链，MA 仍能获得增益

### 复现命令
```bash
python reproduce_fig9.py --trials 50 --cores 4
```
*注: 当前脚本仅实现 FD 架构对比，AP/HP 需额外实现预编码矩阵设计。*

---

## 参数对比总结表

| 图表 | SNR (dB) | M, N | Lt, Lr | A/λ | 横轴 | 纵轴 (Proposed 最大值) |
|---|---|---|---|---|---|---|
| **Fig. 5(a)** | -15 | 4, 4 | 5, 5 | 1~3 | A/λ | ~1.1 bps/Hz |
| **Fig. 5(b)** | -15 | 4, 4 | 5, 5 | 1~3 | A/λ | ~35 (Power) |
| **Fig. 6(a)** | **25** | 4, 4 | 5, 5 | 1~4 | A/λ | **~29 bps/Hz** ✓ |
| **Fig. 6(b)** | 25 | 4, 4 | 5, 5 | 1~4 | A/λ | ~18 (Total Power) |
| **Fig. 6(c)** | 25 | 4, 4 | 5, 5 | 1~4 | A/λ | ~6 (Cond. Num.) |
| **Fig. 7(a)** | -15~15 | 4, 4 | **3, 3** | 3 | SNR | ~15 bps/Hz |
| **Fig. 7(b)** | -15~15 | 4, 4 | **6, 6** | 3 | SNR | ~18 bps/Hz |
| **Fig. 8** | **5** | **2~8** | 5, 5 | 4 | M(N) | ~14 bps/Hz (M=8) |
| **Fig. 9** | -15~15 | **6, 6** | 5, 5 | 4 | SNR | ~20 bps/Hz |


