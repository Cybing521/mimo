# DRL训练问题排查指南

## 问题1：容量值偏低（23.55 vs 期望29.4）

### 可能原因分析

#### 1. **配置参数不一致**
- **训练时**: `SNR_dB=25.0`, `A_lambda=3.0`
- **对比时**: 必须使用**完全相同**的参数
- **检查**: 确认训练配置和测试配置一致

#### 2. **训练未完全收敛**
- **当前状态**: 500 episodes，容量23.55
- **建议**: 
  - 继续训练到5000 episodes
  - 观察评估容量是否持续上升
  - 如果评估容量在后期不再提升，说明已收敛

#### 3. **max_steps可能不够**
- **当前**: `max_steps=50`
- **问题**: 50步可能不足以让DRL找到最优位置
- **建议**: 
  - 尝试增加到 `max_steps=100`
  - 观察容量是否提升

#### 4. **奖励函数权重需要微调**
- **当前权重**:
  - `w_capacity=2.0`
  - `w_improvement=10.0`
- **建议**: 如果容量提升缓慢，可以：
  - 进一步增加 `w_improvement` 到 15.0
  - 或者增加 `w_capacity` 到 3.0

#### 5. **学习率可能过大或过小**
- **当前**: `lr_actor=3e-4`, `lr_critic=3e-4`
- **建议**: 
  - 如果训练不稳定，降低到 `2e-4`
  - 如果收敛太慢，可以尝试 `5e-4`

### 诊断步骤

1. **检查训练曲线**:
   ```bash
   # 查看训练曲线图
   open results/drl_training/run_20251129_185802/training_curves.png
   ```
   - 评估容量是否还在上升？
   - 训练容量是否稳定？

2. **运行快速对比**:
   ```bash
   python experiments/quick_compare.py \
       --drl_model results/drl_training/run_20251129_185802/best_model.pth \
       --SNR_dB 25.0 \
       --A_lambda 3.0 \
       --trials 20
   ```

3. **检查初始容量**:
   - DRL从随机初始化开始
   - AO可能从更好的初始化开始
   - 这可能导致差距

## 问题2：如何使用对比脚本

### 方法1：快速对比脚本（推荐）

```bash
# 基本用法
python experiments/quick_compare.py \
    --drl_model results/drl_training/run_20251129_185802/best_model.pth \
    --SNR_dB 25.0 \
    --A_lambda 3.0 \
    --trials 20

# 完整参数
python experiments/quick_compare.py \
    --drl_model results/drl_training/run_20251129_185802/best_model.pth \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --SNR_dB 25.0 \
    --A_lambda 3.0 \
    --trials 20 \
    --max_steps 50 \
    --device auto
```

**输出示例**:
```
======================================================================
对比结果
======================================================================

方法            平均容量(bps/Hz)    标准差          平均时间(s)    
----------------------------------------------------------------------
AO (迭代算法)       29.40 ± 2.15     0.2345
DRL               23.55 ± 3.21     0.0234
----------------------------------------------------------------------

容量差距: 5.85 bps/Hz (19.9%)
速度提升: 10.0x (DRL更快)
```

### 方法2：完整对比脚本（多A_lambda值）

```bash
python experiments/compare_methods.py \
    --experiment region_size \
    --drl_model results/drl_training/run_20251129_185802/best_model.pth \
    --methods AO DRL \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --trials 20 \
    --device auto
```

这会生成对比图，显示不同A_lambda下的容量对比。

### 方法3：单点对比（相同A_lambda）

```bash
# 使用compare_methods.py的内部方法
python -c "
from experiments.compare_methods import MethodComparator
import numpy as np

comparator = MethodComparator(
    drl_model_path='results/drl_training/run_20251129_185802/best_model.pth',
    device='auto'
)

# 单次对比
result = comparator.run_comparison(
    methods=['AO', 'DRL'],
    N=4, M=4, Lt=5, Lr=5,
    SNR_dB=25.0,
    A_lambda=3.0
)

print('AO容量:', result['AO']['capacity'])
print('DRL容量:', result['DRL']['capacity'])
print('差距:', result['AO']['capacity'] - result['DRL']['capacity'])
"
```

## 改进建议

### 1. 增加训练时间
```bash
python experiments/train_drl.py \
    --num_episodes 10000 \  # 增加到10000
    --SNR_dB 25.0 \
    --A_lambda 3.0 \
    --max_steps 100 \  # 增加到100步
    # ... 其他参数
```

### 2. 调整奖励函数
如果容量提升缓慢，可以修改 `drl/env.py` 中的奖励权重：
```python
self.reward_config = {
    'w_capacity': 3.0,      # 从2.0增加到3.0
    'w_improvement': 15.0,  # 从10.0增加到15.0
    # ...
}
```

### 3. 使用预训练初始化
考虑使用AO算法的结果作为DRL的初始位置，而不是完全随机初始化。

### 4. 检查SNR设置
- 训练时使用 `SNR_dB=25.0`
- 但之前讨论的是 `SNR_dB=15.0`
- 不同SNR下容量范围不同，需要确认期望值

## 配置检查清单

在运行对比前，确认以下参数一致：

- [ ] `SNR_dB`: 训练时 vs 测试时
- [ ] `A_lambda`: 训练时 vs 测试时  
- [ ] `N, M, Lt, Lr`: 训练时 vs 测试时
- [ ] `max_steps`: DRL测试时的步数
- [ ] 模型文件路径正确

## 常见问题

**Q: 为什么DRL容量总是低于AO？**
A: 这是正常的，因为：
1. AO是迭代优化，理论上能找到局部最优
2. DRL是学习策略，可能陷入次优解
3. 但DRL的优势是速度快（10-100倍）

**Q: 如何判断训练是否收敛？**
A: 观察评估容量曲线：
- 如果最后1000个episodes评估容量不再提升 → 已收敛
- 如果还在上升 → 继续训练

**Q: 容量差距多少算正常？**
A: 
- 差距 < 2 bps/Hz: 非常好
- 差距 2-5 bps/Hz: 可接受
- 差距 > 5 bps/Hz: 需要改进

