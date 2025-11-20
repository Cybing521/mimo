# Idea 2: Secrecy Energy Efficiency in MA-SWIPT

## 1. 核心概念 (Core Concept)
研究 **物理层安全 (Physical Layer Security)** 在 MA-SWIPT 系统中的应用。引入窃听者 (Eve)，利用 MA 的位置灵活性进行“位置抗干扰”。

## 2. 问题描述 (Problem Statement)
- **现状**: FAS/MA 的物理层安全研究刚起步，但专门针对 **SWIPT 场景** 的安全性研究极少。
- **缺失**: SWIPT 场景下的特有冲突（强能量信号易泄露 vs 安全性需求）在 MA 系统中未被充分探索。
- **痛点**: 为了给 EH 用户充够电，必须发送强信号，这增加了信号泄露给 Eve 的风险。
- **机会**: MA 可以移动到某个位置，使得该位置对合法用户 (Bob) 的信道增益最大，同时对 Eve 的信道处于深衰落（即利用多径效应制造天然的“屏蔽”）。

## 3. 方案设计 (Methodology)

### 3.1 系统模型
- **节点**: Alice (MA Tx), Bob (ID Rx), Charlie (EH Rx), Eve (Eavesdropper).
- **目标**: 最大化保密速率 (Secrecy Rate) $C_s = [C_{Bob} - C_{Eve}]^+$，同时满足 $E_{Charlie} \ge E_{th}$。

### 3.2 算法思路
- **人工噪声 (Artificial Noise, AN)**: 利用 MA 生成指向 Eve 的人工噪声，同时向 Bob/Charlie 发送有用信号。
- **位置优化**: 寻找使得 $h_{Bob}$ 强、$h_{Charlie}$ 强、但 $h_{Eve}$ 弱的天线位置配置。

## 4. 预期结果
- 相比固定天线，MA 可以显著提高保密容量。
- 即使不使用大量人工噪声，仅靠位置优化也能实现较好的安全性。

## 5. 可行性分析
- 需要新增 Eve 的信道生成逻辑。
- 目标函数从简单的 Capacity 改为 Secrecy Rate。

