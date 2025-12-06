# MIMO RL 优化策略分析与指南

## 1. 当前问题分析 (Problem Analysis)
目前的 RL agent (PPO/SAC/TD3/DDPG) 在 MIMO 容量优化任务上表现不如传统的交替优化 (AO) 算法。
- **AO Baseline**: ~29.27 bps/Hz
- **RL Agents**: ~25.00 bps/Hz
- **差距**: ~4.27 bps/Hz

### 原因推测
1.  **局部最优 (Local Optima)**: 问题是非凸的，RL 可能陷入了次优的局部极值。
2.  **奖励函数 (Reward Function)**: 之前的奖励函数可能过于关注“步长”或“效率”，导致 agent 不敢大幅探索。
3.  **探索不足 (Insufficient Exploration)**: 在高维连续空间中，单纯的高斯噪声探索效率低下。

---

## 2. 优化策略 (Optimization Strategies)

### 2.1 奖励函数重构 (Reward Shaping) - **已实施**
*   **目标**: 引导 agent 关注绝对容量，而非细微的步进。
*   **方案**:
    *   增加 `w_capacity` 权重。
    *   引入 **Soft Penalties** (软惩罚) 代替硬截断，允许 agent 偶尔探索边界外区域但给予惩罚。
    *   减少 `w_efficiency`，鼓励探索。

### 2.2 混合优化 (Hybrid Approach) - **已实施**
*   **原理**: RL 擅长全局搜索（找大概位置），AO 擅长局部微调（找精确峰值）。
*   **方案**:
    1.  **RL Initialization**: 使用训练好的 RL agent 输出一个初始位置 $(t_{init}, r_{init})$。
    2.  **AO Fine-tuning**: 以 $(t_{init}, r_{init})$ 为起点，运行 AO 算法进行微调。
*   **预期**: 结合两者优势，突破 29 bps/Hz。

### 2.3 高级 RL 算法 (Advanced RL) - **已实施**
*   **SAC (Soft Actor-Critic)**: 最大化熵，鼓励探索，防止过早收敛。
*   **TD3 (Twin Delayed DDPG)**: 减少 Q 值高估，提高训练稳定性。

---

## 3. 实验结果与分析 (Phase 1 Results)

经过一系列实验（Reward Shaping, Hybrid, Extended Training, Imitation Learning, GNN），我们发现所有基于梯度的 RL 方法都收敛在 **25-26 bps/Hz**，而 AO 算法能达到 **~29 bps/Hz**。

### 3.1 核心瓶颈：欺骗性优化地貌 (Deceptive Landscape)
*   **局部最优陷阱**：问题空间中存在大量“平庸”的局部最优解（~25 bps/Hz）。这些解的吸引域（Basin of Attraction）很大，随机初始化或简单的探索很容易落入其中。
*   **狭窄的全局最优**：AO 找到的解（~29 bps/Hz）可能位于一个非常狭窄的峰值上。RL 的随机梯度估计很难精确地指引 agent 进入这个狭窄区域。
*   **精度问题**：连续控制的 RL (如 SAC/PPO) 在高精度微调方面不如利用二阶信息或凸优化结构的 AO 算法。

---

## 4. 下一步优化策略 (Phase 2 Proposals)

针对上述分析，我认为单纯调整网络结构或奖励函数已经无法解决问题。我们需要改变**学习的方式**。

### 策略一：课程学习 (Curriculum Learning)
*   **原理**：先学简单的，再学难的。
*   **实施**：
    1.  **降维**：先在 N=2, M=2 的简单场景下训练 GNN。此时搜索空间小，RL 更容易找到全局最优。
    2.  **迁移**：利用 GNN 的可扩展性，将训练好的策略迁移到 N=4, M=4 场景作为初始化。
    3.  **微调**：在 N=4 的场景下继续训练。
*   **预期**：N=2 的经验能教会 agent 某种“通用模式”（如：天线应该成对出现，还是分散？），从而在 N=4 时避开平庸解。

### 策略二：由粗到细搜索 (Coarse-to-Fine Search)
*   **原理**：将连续空间搜索转化为“离散区域选择 + 连续微调”。
*   **实施**：
    1.  **粗粒度离散化**：将移动区域划分为 $K \times K$ 的网格。
    2.  **DQN 决策**：训练一个 DQN agent 决定每个天线应该放在哪个格子里（离散动作）。
    3.  **AO 微调**：在 DQN 选定的格子里，运行 AO 算法进行连续微调。
*   **预期**：DQN 负责跳出局部最优（全局搜索），AO 负责高精度爬山。

### 策略三：进化策略 (Evolutionary Strategies, ES)
*   **原理**：放弃梯度，使用基于种群的黑盒优化（如 CMA-ES）。
*   **实施**：维护一个参数分布，通过采样-评估-更新来寻找最优策略。
*   **预期**：ES 对局部极值有更强的鲁棒性，适合这种非凸、多模态的优化问题。

**建议优先尝试：策略一（课程学习）**，因为我们已经有了 GNN 基础，这是最自然的延伸。
