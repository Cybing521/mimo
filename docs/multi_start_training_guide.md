# 多起点训练 + 改进探索策略使用指南

## 概述

本指南介绍如何使用**多起点训练**和**改进探索策略**来提升DRL模型的性能，使其更接近全局最优。

## 实现的功能

### 方案3：多起点训练
- 训练多个模型，每个使用不同的随机种子和初始化
- 增加找到全局最优的概率
- 自动选择评估容量最高的模型

### 方案4：改进探索策略
- **自适应熵系数**：早期高探索（3倍熵），后期高利用（正常熵）
- **动作噪声注入**：早期添加探索噪声，后期无噪声
- **多起点初始化**：每个模型从不同的随机位置开始

## 使用方法

### 基本用法

```bash
# 训练3个模型，自动选择最好的
python experiments/train_drl_multi_start.py \
    --num_models 3 \
    --select_best \
    --num_episodes 5000 \
    --SNR_dB 25.0 \
    --A_lambda 3.0 \
    --device auto \
    --save_dir results/drl_training
```

### 完整参数示例

```bash
python experiments/train_drl_multi_start.py \
    --num_models 5 \
    --select_best \
    --num_episodes 10000 \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --SNR_dB 25.0 \
    --A_lambda 3.0 \
    --max_steps 100 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --entropy_coef 0.01 \
    --base_seed 42 \
    --device auto \
    --save_dir results/drl_training \
    --use_wandb \
    --wandb_project mimo
```

### 参数说明

**多起点训练参数**：
- `--num_models`: 训练的模型数量（默认3，建议3-5）
- `--select_best`: 是否自动选择最佳模型
- `--base_seed`: 基础随机种子（每个模型使用 base_seed + model_id * 1000）

**其他参数**：与 `train_drl.py` 相同，包括所有PPO超参数、环境参数等。

## 工作原理

### 1. 多起点训练流程

```
模型1 (seed=42)    → 训练 → 评估容量: 24.5
模型2 (seed=1042)  → 训练 → 评估容量: 25.8  ← 最佳
模型3 (seed=2042)  → 训练 → 评估容量: 24.2
                    ↓
              选择模型2作为最佳模型
```

### 2. 自适应熵系数

```python
# 训练进度 < 20%: 高探索
entropy_coef = 0.01 * 3.0 = 0.03

# 训练进度 20%-50%: 中等探索
entropy_coef = 0.01 * 2.0 = 0.02

# 训练进度 > 50%: 低探索，高利用
entropy_coef = 0.01 (正常衰减)
```

### 3. 动作噪声注入

```python
# 训练进度 < 30%: 高噪声（0.3 → 0）
exploration_noise = 0.3 * (1 - progress)

# 训练进度 30%-60%: 中等噪声（0.1 → 0）
exploration_noise = 0.1 * (1 - progress)

# 训练进度 > 60%: 无噪声
exploration_noise = 0.0
```

## 输出结果

训练完成后，会生成：

1. **每个模型的训练目录**：
   ```
   results/drl_training/
   ├── run_20251129_120000/  # 模型1
   ├── run_20251129_120500/  # 模型2
   └── run_20251129_121000/  # 模型3
   ```

2. **最佳模型信息**（如果使用 `--select_best`）：
   ```
   results/drl_training/best_model_info.json
   {
     "best_model_id": 2,
     "best_eval_capacity": 25.8,
     "best_model_path": "results/drl_training/run_20251129_120500/best_model.pth",
     "all_capacities": [24.5, 25.8, 24.2]
   }
   ```

## 预期效果

| 方法 | 容量 (bps/Hz) | 改进 |
|------|--------------|------|
| 单模型训练 | 23.22 | 基准 |
| 多起点训练 (3个模型) | 25-26 | +1.8-2.8 |
| 多起点训练 (5个模型) | 26-27 | +2.8-3.8 |

**注意**：实际效果取决于：
- 模型数量（越多越好，但训练时间也越长）
- 训练episodes数量
- 系统配置（SNR, A_lambda等）

## 使用最佳模型

训练完成后，使用最佳模型进行测试：

```bash
# 方法1：使用best_model_info.json中的路径
python experiments/quick_compare.py \
    --drl_model results/drl_training/run_XXXXXX/best_model.pth \
    --SNR_dB 25.0 \
    --A_lambda 3.0 \
    --trials 20

# 方法2：直接使用最佳模型路径（从best_model_info.json中获取）
```

## 注意事项

1. **训练时间**：多起点训练时间 = 单模型时间 × 模型数量
   - 3个模型：约3倍时间
   - 5个模型：约5倍时间

2. **内存使用**：每个模型独立训练，不会同时加载多个模型到内存

3. **WandB日志**：如果使用 `--use_wandb`，每个模型会创建独立的WandB run

4. **随机种子**：确保每个模型使用不同的种子，以获得不同的初始化

## 故障排查

**问题1：导入错误**
```bash
# 确保在项目根目录运行
cd /Users/cyibin/Documents/研一/项目/MIMO
python experiments/train_drl_multi_start.py ...
```

**问题2：模型训练失败**
- 检查是否有足够的磁盘空间
- 检查内存是否足够
- 尝试减少 `--num_models` 或 `--num_episodes`

**问题3：最佳模型选择失败**
- 确保使用 `--select_best` 参数
- 检查所有模型是否成功训练完成

## 进阶技巧

1. **并行训练**（如果有多GPU）：
   - 可以手动并行运行多个 `train_drl.py`，每个使用不同的seed
   - 然后手动比较结果

2. **集成方法**：
   - 不选择最佳模型，而是使用所有模型的平均动作
   - 需要修改代码实现集成预测

3. **渐进式训练**：
   - 先用少量episodes训练所有模型
   - 选择表现最好的几个继续训练

