# 基于图神经网络 (GNN) 的 MIMO 天线位置优化方案

## 1. 为什么选择 GNN？

在可移动天线 (MA) 系统中，天线阵列本质上是一个**点集 (Point Set)** 或 **图 (Graph)**。传统的多层感知机 (MLP) 并不适合处理这种数据结构。

### 传统 MLP 的局限性
1.  **排列敏感 (Permutation Sensitive)**：
    *   对于 MLP，输入向量 `[x1, y1, x2, y2]` 和 `[x2, y2, x1, y1]` 是完全不同的。
    *   然而物理上，这只是交换了两个天线的编号，系统性能完全一样。
    *   MLP 需要浪费大量训练数据来学习这种“对称性”。
2.  **缺乏局部交互**：
    *   天线之间的耦合（互耦效应、空间相关性）主要取决于**相对距离**。
    *   MLP 难以捕捉这种局部的几何关系。
3.  **维度固定**：
    *   训练好的 MLP 只能处理固定数量的天线（如 N=4）。如果变成 N=8，必须重新设计网络并从头训练。

### GNN 的优势
1.  **排列不变性 (Permutation Invariance)**：
    *   GNN 对节点的输入顺序不敏感。无论怎么编号，输出的策略都是一致的。
    *   这极大地提高了**样本效率 (Sample Efficiency)**。
2.  **几何感知 (Geometric Awareness)**：
    *   GNN 通过边 (Edge) 传递信息，天然地利用了天线之间的相对位置信息。
    *   非常适合处理“保持最小间距”这类几何约束。
3.  **可扩展性 (Scalability)**：
    *   同一个 GNN 模型可以处理任意数量的天线。

---

## 2. 拟定网络架构

我们将设计一个 **Actor-Critic** 架构，其中 Actor 和 Critic 都使用 GNN。

### 2.1 图的构建 (Graph Construction)
*   **节点 (Nodes)**：每个天线作为一个节点。
    *   节点特征 $h_i$：`[x, y, type]` (type: 0 for Tx, 1 for Rx)。
    *   还可以加入局部信道特征（如果能分解到天线粒度）。
*   **边 (Edges)**：全连接图 (Fully Connected) 或 K-近邻图 (KNN)。
    *   边特征 $e_{ij}$：`[相对距离, 相对角度]`。

### 2.2 GNN 层 (Graph Layer)
我们将使用 **Graph Attention Network (GAT)** 或 **Transformer** 变体，因为注意力机制可以动态衡量不同天线之间的相互影响权重。

$$ h_i' = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right) $$

其中 $\alpha_{ij}$ 是注意力系数，由节点特征和边特征共同决定。

### 2.3 整体流程
1.  **Encoder**：将全局信道状态（特征值等）编码为全局上下文向量 $C$。
2.  **GNN Backbone**：处理天线节点特征，进行多轮消息传递，更新每个节点的嵌入向量 $h_i$。
3.  **Actor Head**：
    *   输入：节点嵌入 $h_i$ + 全局上下文 $C$。
    *   输出：每个节点的动作 $\Delta x_i, \Delta y_i$。
4.  **Critic Head**：
    *   输入：所有节点嵌入的聚合（如 Mean Pooling）+ 全局上下文 $C$。
    *   输出：状态价值 $V(s)$。

---

## 3. 实施计划

1.  **环境准备**：安装 `torch_geometric` (PyG) 或手动实现简单的 GNN 层（为了减少依赖，建议手动实现简单的 Attention 层）。
2.  **代码实现**：
    *   创建 `drl/gnn_agent.py`。
    *   实现 `GNNActor` 和 `GNNCritic`。
3.  **实验验证**：
    *   对比 GNN-PPO 与 MLP-PPO 的收敛速度和最终性能。
    *   验证 GNN 的泛化能力（在 N=4 上训练，在 N=8 上测试）。

## 4. 预期结果
*   **收敛更快**：由于利用了对称性，GNN 应该能更快找到有效策略。
*   **性能更高**：更好的几何感知能力有助于找到更优的天线排布。
