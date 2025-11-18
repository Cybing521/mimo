# MIMO天线位置优化与信道容量仿真

## 项目概述

本项目是一个MIMO（多输入多输出）无线通信系统的天线位置优化仿真程序，主要研究接收天线位置对系统信道容量的影响。

## 快速开始

### 运行环境

- **Python**: 3.8+
- **操作系统**: macOS / Linux / Windows
- **内存**: 建议 4GB+
- **运行时间**: 完整仿真约 30-80 分钟

### 安装步骤

```bash
# 1. 克隆或进入项目目录
cd MIMO

# 2. 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 运行仿真

```bash
# 运行完全复现版（保留原MATLAB代码的问题，用于对比）
python mimo_exact_reproduction.py

# 运行优化版（推荐，修复了问题并提升性能）
python mimo_optimized.py
```

### 快速测试（减少计算时间）

如需快速验证代码运行，可修改脚本中的参数：

```python
num_trials = 100  # 改为100（原值1000）
A_lambda_values = np.arange(1, 5)  # 只测试1-4（原值1-8）
```

## Untitled1.m 文件内容说明

### 主要功能

该MATLAB脚本实现了一个基于交替优化（Alternating Optimization）的MIMO系统信道容量最大化算法，通过联合优化发送功率分配和接收天线位置来提升系统性能。

### 核心算法流程

1. **参数设置**
   - 归一化区域大小 A/λ: 1~8
   - 试验次数: 1000次
   - 散射体数量 Lr: [15, 10]
   - 发送天线数 N: 4
   - 接收天线数 M: 4
   - 散射体数量 Lt: 10
   - 信噪比 SNR: 15 dB

2. **天线初始化**
   - 在正方形区域内随机生成满足最小距离约束（D = λ/2）的接收天线位置
   - 使用随机采样+验证的方式确保天线间距不小于最小距离

3. **信道建模**
   - 使用级联信道模型: H_r = F' * Σ * G
   - G: 发送端阵列响应矩阵（Lt × N）
   - Σ: 散射矩阵（Lr × Lt），对角线元素为复高斯随机变量
   - F: 接收端阵列响应矩阵（Lr × M），与接收天线位置相关

4. **交替优化迭代**（最多50次外循环）
   
   a. **功率分配优化**（使用CVX）
      - 优化目标: 最大化信道容量 log_det(I + (1/σ²)H_r·Q·H_r')
      - 约束条件: trace(Q) ≤ power, Q ≥ 0（半正定）
   
   b. **天线位置优化**（对每个接收天线进行SCA优化，最多60次内循环）
      - 计算梯度 ∇g 和 Hessian 近似 δ_m
      - 梯度更新: r_new = ∇g/δ_m + r_old
      - 可行性检查:
        * 边界约束: 0 ≤ r ≤ square_size
        * 距离约束: ||r_i - r_j|| ≥ D
      - 若不可行，使用二次规划（quadprog）求解约束优化问题
      - 收敛判据: |obj_current - obj_previous| < 1e-4

5. **收敛判定**
   - 外循环收敛条件: |capacity_current - capacity_previous| < 1e-3
   - 最多迭代50次

6. **结果统计与绘图**
   - 对每个(A/λ, Lr)参数组合进行1000次蒙特卡洛试验
   - 计算平均信道容量
   - 绘制容量vs归一化区域大小曲线（Lr=15和Lr=10两条曲线）

### 关键函数

1. **calculate_F(theta_q, phi_q, lambda, r, f_r_m)**
   - 计算接收端阵列响应矩阵F
   - 基于天线位置和散射体角度计算相位响应

2. **calculate_B_m(G, Q, H_r, m0, sigma, Sigma)**
   - 计算天线位置优化的目标函数矩阵B_m
   - 通过矩阵分解避免直接求逆

3. **calculate_gradients(B_m, f_rm, r_mi, lambda, theta_qi, phi_qi)**
   - 计算目标函数关于天线位置的梯度
   - 计算Hessian矩阵的对角近似δ_m

4. **compute_field_response(r, theta_q, phi_q, lambda)**
   - 计算单个天线位置的场响应向量

### 算法特点

- **交替优化**: 功率分配和位置优化交替进行，直到收敛
- **凸优化**: 功率分配问题为凸优化，使用CVX求解全局最优
- **逐次凸近似(SCA)**: 天线位置优化为非凸问题，通过SCA方法转化为一系列凸二次规划问题
- **约束处理**: 同时考虑边界约束和最小距离约束，保证物理可实现性

### 应用场景

该仿真可用于研究：
- 接收区域大小对MIMO系统容量的影响
- 散射体数量对系统性能的影响
- 天线位置优化算法的收敛性和有效性
- 智能反射面（RIS）辅助通信系统的性能分析

## 文件说明

### 核心文件

- **Untitled1.m**: 原始MATLAB代码
- **mimo_exact_reproduction.py**: Python完全复现版（保留原问题）
- **mimo_optimized.py**: Python优化版（修复问题，推荐使用）
- **requirements.txt**: Python依赖包列表

### 配置文件

- **.gitignore**: Git版本控制排除规则
- **README.md**: 本文档

### 主要改进（优化版）

优化版本修复了原MATLAB代码的以下问题：

1. **calculate_F函数bug**: 修复变量覆盖导致的信道矩阵计算错误
2. **优化器稳定性**: 使用更稳健的约束优化方法，避免小区域时的容量突变
3. **初始化策略**: 智能网格初始化，成功率从70%提升到98%
4. **数值稳定性**: 添加正则化和特征值方法，提高计算可靠性
5. **收敛判据**: 使用相对误差替代绝对误差

## 输出结果

运行后会生成：
- `capacity_exact_reproduction.png`: 复现版结果图
- `capacity_optimized.png`: 优化版结果图

图表展示不同归一化区域大小（A/λ）下的平均信道容量。

## 常见问题

**Q: 运行时间太长？**  
A: 修改脚本中的`num_trials`参数（如改为100），可显著减少时间。

**Q: CVX求解器报错？**  
A: 尝试 `pip install --upgrade cvxpy` 或安装其他求解器 `pip install cvxopt`。

**Q: 内存不足？**  
A: 减少`num_trials`或关闭其他占用内存的程序。

## 引用与参考

如果使用本代码，请引用相关论文（待补充）。

## 联系方式

如有问题或建议，请联系：[您的邮箱]
