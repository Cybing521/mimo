"""
PPO 智能体实现
==============

用于 MA-MIMO 优化任务的 Proximal Policy Optimization (PPO) 智能体。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from .networks import ActorNetwork, CriticNetwork


class PPOAgent:
    """
    Proximal Policy Optimization Agent.

    对强化学习新手的速览：
    - Actor 负责输出动作分布；Critic 负责估计当前状态的“好坏”。
    - 训练时我们使用 PPO-Clip（限制策略变化的比率）+ GAE（更平滑的优势估计），
      以避免策略每次更新跨步过大导致崩溃。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
    ):
        """
        Initialize PPO Agent
        
        Args:
            state_dim: 状态向量长度（来自环境的观测维度）。
            action_dim: 动作向量长度（环境动作空间的维度）。
            lr_actor: Actor 学习率，控制策略网络每次更新幅度。
            lr_critic: Critic 学习率，控制价值网络更新幅度。
            gamma: 折扣因子 γ，决定未来奖励的重要性（越接近 1 越看重远期）。
            gae_lambda: GAE 的 λ，调节优势估计平滑度（0.95 常用）。
            clip_epsilon: PPO clip 的 ε，限制新旧策略概率比，防止更新过猛。
            ppo_epochs: 每收集一批轨迹后，重复优化 PPO 损失的轮数。
            batch_size: 每次梯度更新所用的小批量样本数。
            entropy_coef: 熵正则系数，值越大越鼓励探索（策略分布更发散）。
            value_loss_coef: Value Loss 权重，用于平衡 critic 相对 actor 的梯度。
            max_grad_norm: 最大梯度范数，超过则裁剪以避免梯度爆炸。
            device: 计算设备（'cpu' 或 'cuda'），传给 torch.device。
        """
        self.device = torch.device(device)
        
        # Networks（Actor 学策略，Critic 学状态价值）
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Hyperparameters（PPO/GAE 的核心控制项）
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Replay buffer（用 list 暂存一个 rollout，更新后立即清空）
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Statistics（滑动窗口，方便日志记录）
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'ratio': deque(maxlen=100),
            'ratio_mean': deque(maxlen=100),
        }
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        从当前策略中采样动作。
        
        Args:
            state: 环境给出的状态观测。
            deterministic: True 时使用均值动作（评估），False 时随机采样（训练）。
        
        Returns:
            action: 要施加到环境的动作向量（np.ndarray）。
            log_prob: 该动作在旧策略下的对数概率（用于 PPO 比值）。
            value: Critic 估计的状态价值。
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
        
        action_np = action.cpu().numpy()[0]
        log_prob_np = log_prob.cpu().item() if log_prob is not None else 0.0
        value_np = value.cpu().item()
        
        return action_np, log_prob_np, value_np
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """
        将一次交互 (s, a, r, V, logπ, done) 写入缓冲区，供更新阶段统一处理。
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(
        self, 
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算广义优势估计（GAE）。
        
        Args:
            next_value: 轨迹终点下一状态的价值，用于 bootstrapping。
        
        Returns:
            advantages: 每个时间步的优势估计 A_t。
            returns: 对应的回报（用于训练 Critic）。
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        使用 PPO-Clip 对策略与价值网络进行一次批量更新。
        
        Args:
            next_state: rollout 结束时的状态，用于计算 bootstrapping 价值。
        
        Returns:
            记录本次更新中 actor/critic loss、熵等统计量的字典。
        """
        if len(self.states) == 0:
            return {}
        
        # Compute advantages and returns（next_state 用来做 bootstrapping）
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor).cpu().item()
        
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages（标准化后更易训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO update
        dataset_size = states.size(0)
        
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'ratio_mean': 0.0,
        }
        
        for epoch in range(self.ppo_epochs):
            # Generate random indices
            indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Sample mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Evaluate actions（新策略给出 log prob 和熵）
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_states, batch_actions
                )
                values = self.critic(batch_states)
                
                # Ratio and surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs.unsqueeze(1))
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * batch_advantages.unsqueeze(1)
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns.unsqueeze(1))
                
                # Entropy loss（鼓励探索，熵高=更平）
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    actor_loss 
                    + self.value_loss_coef * value_loss 
                    + self.entropy_coef * entropy_loss
                )
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Update statistics
                stats['actor_loss'] += actor_loss.item()
                stats['critic_loss'] += value_loss.item()
                stats['entropy'] += -entropy_loss.item()
                stats['ratio_mean'] += ratio.mean().item()
        
        # Average statistics
        num_updates = self.ppo_epochs * (dataset_size // self.batch_size + 1)
        for key in stats:
            stats[key] /= num_updates
            self.training_stats[key].append(stats[key])
        
        # Clear buffer
        self.clear_buffer()
        
        return stats
    
    def clear_buffer(self):
        """清空当前 rollout 数据。"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save(self, filepath: str):
        """
        保存智能体参数。
        
        Args:
            filepath: checkpoint 写入路径。
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """
        加载已保存的智能体。
        
        Args:
            filepath: checkpoint 文件路径。
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Agent loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """
        返回近期训练指标（滑动平均），便于日志或可视化。
        """
        stats = {}
        for key, values in self.training_stats.items():
            if len(values) > 0:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
        return stats


# Test agent
if __name__ == "__main__":
    print("Testing PPO Agent...")
    
    # Parameters
    state_dim = 20
    action_dim = 8
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu'
    )
    
    # Test action selection
    print("\n1. Testing action selection...")
    state = np.random.randn(state_dim)
    action, log_prob, value = agent.select_action(state)
    print(f"   Action shape: {action.shape}")
    print(f"   Log prob: {log_prob:.4f}")
    print(f"   Value: {value:.4f}")
    
    # Test storing transitions
    print("\n2. Testing transition storage...")
    for _ in range(100):
        state = np.random.randn(state_dim)
        action, log_prob, value = agent.select_action(state)
        reward = np.random.randn()
        done = False
        agent.store_transition(state, action, reward, value, log_prob, done)
    print(f"   Stored {len(agent.states)} transitions")
    
    # Test update
    print("\n3. Testing PPO update...")
    next_state = np.random.randn(state_dim)
    stats = agent.update(next_state)
    print("   Training stats:")
    for key, value in stats.items():
        print(f"      {key}: {value:.4f}")
    
    # Test save/load
    print("\n4. Testing save/load...")
    agent.save('/tmp/test_agent.pth')
    agent.load('/tmp/test_agent.pth')
    print("   Save/load successful!")
    
    print("\nAll tests passed!")

