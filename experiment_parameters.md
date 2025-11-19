# 实验参数配置说明文档

本文档基于 Ma et al. (2023) 论文，详细说明 **Fig. 5 (低信噪比)** 与 **Fig. 6 (高信噪比)** 两个实验的参数差异。

## 1. 公共参数 (Common Parameters)
两个实验共享的基础系统设置：

*   **天线数量**: $N = 4$ (发射端), $M = 4$ (接收端)
*   **多径数量**: $L_t = 5$ (发射路径), $L_r = 5$ (接收路径)
*   **信道模型**: Geometry-based Channel Model
    *   LoS/NLoS 功率比 ($\kappa$): 1
    *   AoD/AoA 分布: Uniform $[0, \pi]$
*   **最小天线间距**: $D = \lambda / 2$
*   **载波波长**: $\lambda = 1$ (归一化单位)
*   **蒙特卡洛次数**: 建议 $\ge 50$ (论文使用 $4 \times 10^4$，快速验证可减少)

---

## 2. 实验差异对比 (Experiment Differences)

| 参数项 | **Fig. 5 (Low-SNR Regime)** | **Fig. 6 (High-SNR Regime)** |
| :--- | :--- | :--- |
| **信噪比 (SNR)** | **-15 dB** (极低信噪比) | **25 dB** (高信噪比) |
| **X轴变量** | Normalized region size ($A/\lambda$) | Normalized region size ($A/\lambda$) |
| **X轴范围** | 1.0 ~ 3.0 (步长 0.2 或 0.5) | 1.0 ~ 4.0 (步长 0.5) |
| **Y轴指标** | Achievable Rate (bps/Hz) | Achievable Rate (bps/Hz) |
| **Y轴数值范围** | 约 0.4 ~ 1.2 bps/Hz | 约 18 ~ 30 bps/Hz |
| **主要特征** | 所有方案差异较小，SEPM 方案效果极佳 | Proposed 方案显著优于其他，SEPM 效果最差 |
| **对比方案** | Proposed, FPA, RMA, TMA, SEPM | Proposed, FPA, RMA, TMA, SEPM |

## 3. 核心观察点 (Key Observations)

### Fig. 5 (Low-SNR, -15 dB)
*   **现象**: 在低信噪比下，系统容量主要受限于接收信号能量。
*   **结论**: 最强特征信道功率最大化 (SEPM) 算法几乎能达到与复杂 AO 算法 (Proposed) 相同的性能，但计算复杂度更低。

### Fig. 6 (High-SNR, 25 dB)
*   **现象**: 在高信噪比下，系统容量受限于信道矩阵的秩和条件数 (Condition Number)。
*   **结论**: SEPM 算法虽然最大化了最强路径能量，但导致信道条件数变差（Rank-1倾向），在高 SNR 下反而性能垫底。Proposed 算法通过平衡多条数据流的增益，获得了最高容量。

## 4. 如何运行复现
使用 `mimo_comparison.py` 脚本时，请修改 `SNR_dB` 变量：

```python
# 复现 Fig. 5
SNR_dB = -15 
A_lambda_values = np.arange(1, 3.1, 0.5)

# 复现 Fig. 6
SNR_dB = 25
A_lambda_values = np.arange(1, 4.1, 0.5)
```

