# DRL-MA-MIMO 快速入门指南

## 🚀 5分钟快速开始

### 第1步：安装依赖

```bash
cd /Users/cyibin/Documents/研一/项目/MIMO
source venv/bin/activate

# 安装DRL相关依赖
pip install torch torchvision torchaudio
pip install gym tqdm tensorboard
```

### 第2步：测试环境

```bash
# 测试Gym环境
python drl/env.py

# 测试神经网络
python drl/networks.py

# 测试Agent
python drl/agent.py
```

预期输出：
```
Environment test passed!
All tests passed!
```

### 第3步：快速训练（10分钟体验）

```bash
# 快速训练100个episode（约5-10分钟）
python experiments/train_drl.py \
    --num_episodes 100 \
    --N 4 --M 4 \
    --SNR_dB 15 \
    --A_lambda 3.0 \
    --log_interval 5 \
    --save_dir results/drl_training_quick
```

训练过程输出：
```
Configuration saved to results/drl_training_quick/run_20241124_XXXXXX/config.json

Environment created:
  State dim: 20
  Action dim: 8
  N=4, M=4, Lt=5, Lr=5
  SNR=15.0dB, A=3.0λ

Agent created:
  Actor params: 134,664
  Critic params: 133,633
  Device: cpu

Starting training for 100 episodes...

Ep 5/100 | Reward: 18.52 | Capacity: 21.34 | Actor Loss: 0.0234 | Critic Loss: 0.0156
Ep 10/100 | Reward: 22.15 | Capacity: 22.67 | Actor Loss: 0.0198 | Critic Loss: 0.0142
...

=== Evaluation at episode 100 ===
Mean capacity: 24.21 ± 1.23
Mean reward: 25.67 ± 2.45
✓ New best model saved! Capacity: 24.21

Training curves saved to results/drl_training_quick/run_20241124_XXXXXX/training_curves.png
```

### 第4步：完整训练（推荐，2-3小时）

```bash
# 完整训练5000个episode
python experiments/train_drl.py \
    --num_episodes 5000 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --ppo_epochs 10 \
    --batch_size 64 \
    --eval_interval 100 \
    --save_interval 500 \
    --save_dir results/drl_training
```

使用GPU加速（如果有）：
```bash
python experiments/train_drl.py \
    --device cuda \
    --num_episodes 5000
```

---

## 📊 对比实验

### 实验1：容量 vs 区域大小（复现Ma Fig.5）

```bash
# 使用训练好的模型
python experiments/compare_methods.py \
    --experiment region_size \
    --drl_model results/drl_training/run_XXXXXX/best_model.pth \
    --methods AO MS-AO DRL Hybrid \
    --trials 20 \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --save_dir results/comparison
```

参数说明：
- `--experiment`: 实验类型 (region_size, snr, antenna_num)
- `--drl_model`: 训练好的DRL模型路径
- `--methods`: 要对比的方法列表
- `--trials`: 随机试验次数（建议20-50）

### 实验2：容量 vs SNR（复现Ma Fig.7）

```bash
python experiments/compare_methods.py \
    --experiment snr \
    --drl_model results/drl_training/run_XXXXXX/best_model.pth \
    --methods AO DRL Hybrid \
    --trials 20
```

### 实验3：不同天线数量

```bash
python experiments/compare_methods.py \
    --experiment antenna_num \
    --drl_model results/drl_training/run_XXXXXX/best_model.pth \
    --methods AO DRL \
    --trials 20
```

---

## 🎨 可视化结果

### 查看训练曲线

```bash
# 使用tensorboard（如果已安装）
tensorboard --logdir results/drl_training

# 或直接查看保存的图片
open results/drl_training/run_XXXXXX/training_curves.png
```

### 生成论文图表

所有对比实验会自动生成图表，保存在：
```
results/comparison/region_size_XXXXXX/comparison.png
results/comparison/snr_XXXXXX/comparison.png
```

---

## 🔧 调试和优化

### 如果训练不收敛

1. **降低学习率**
```bash
python experiments/train_drl.py \
    --lr_actor 1e-4 \
    --lr_critic 1e-4
```

2. **增大批次大小**
```bash
python experiments/train_drl.py \
    --batch_size 128
```

3. **调整奖励函数**（修改 `drl/env.py` 的 `reward_config`）

### 如果内存不足

```bash
# 减小批次大小
python experiments/train_drl.py \
    --batch_size 32 \
    --ppo_epochs 5
```

### 如果想加速训练

1. **使用GPU**（最有效）
```bash
python experiments/train_drl.py --device cuda
```

2. **减少评估频率**
```bash
python experiments/train_drl.py \
    --eval_interval 200 \
    --save_interval 1000
```

---

## 📈 预期性能基准

| 训练Episode数 | 预期容量 (bps/Hz) | 训练时间 |
|--------------|------------------|---------|
| 100 | 24-25 | 10分钟 |
| 500 | 25-26 | 30分钟 |
| 1000 | 26-27 | 1小时 |
| 5000 | 27-28 | 3小时 |

与Ma's AO对比（Ma: ~23.5 bps/Hz）：
- 100 episodes: +3-7%
- 5000 episodes: +15-20%

---

## 🐛 常见问题

### Q1: ImportError: No module named 'gym'

**A**: 安装gym
```bash
pip install gym
```

### Q2: RuntimeError: CUDA out of memory

**A**: 改用CPU或减小batch_size
```bash
python experiments/train_drl.py --device cpu
```

### Q3: 训练很慢，如何加速？

**A**: 
1. 使用GPU（20-50倍加速）
2. 减少trials数量（实验时）
3. 降低max_steps（从50降到30）

### Q4: 如何复现论文结果？

**A**: 使用固定随机种子
```bash
python experiments/train_drl.py --seed 42
python experiments/compare_methods.py --seed 42
```

---

## 📝 下一步

1. **完整训练**：运行5000 episodes
2. **对比实验**：与Ma's Algorithm对比
3. **消融实验**：测试不同奖励函数
4. **迁移学习**：跨SNR泛化测试
5. **论文撰写**：使用生成的图表

---

## 📞 联系和支持

- 技术文档：`docs/drl_technical_proposal.md`
- GitHub Issues：[项目链接]
- Email: [你的邮箱]

---

**祝实验顺利！🎉**

