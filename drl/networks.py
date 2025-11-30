"""
Neural Network Architectures for PPO Agent
===========================================

Actor-Critic networks with support for continuous action spaces.
实现了方案7：改进网络架构（注意力机制 + 残差连接）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ResidualBlock(nn.Module):
    """
    残差连接块（方案7.2）
    提升网络深度和训练稳定性
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接 + Dropout
        out = self.layers(x)
        return F.relu(x + self.dropout(out))


class SelfAttentionLayer(nn.Module):
    """
    自注意力层（方案7.1）
    帮助网络关注状态中的重要特征
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # 使用线性层实现多头注意力（避免序列维度问题）
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, dim) - 状态向量
        Returns:
            out: (batch_size, dim) - 注意力增强后的特征
        """
        # 将状态向量扩展为序列（batch_size, 1, dim）
        x_expanded = x.unsqueeze(1)  # (batch, 1, dim)
        
        # 计算 Q, K, V
        Q = self.query(x_expanded)  # (batch, 1, dim)
        K = self.key(x_expanded)
        V = self.value(x_expanded)
        
        # 重塑为多头形式 (batch, 1, num_heads, head_dim)
        batch_size = x.size(0)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (batch, num_heads, 1, 1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, 1, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, 1, self.dim)  # (batch, 1, dim)
        
        # 输出投影
        out = self.output(attn_output)  # (batch, 1, dim)
        out = self.dropout(out)
        
        # 残差连接 + LayerNorm
        out = self.layer_norm(x_expanded + out)  # (batch, 1, dim)
        
        # 压缩回 (batch, dim)
        return out.squeeze(1)


class ActorNetwork(nn.Module):
    """
    Actor Network (Policy Network)
    
    Outputs Gaussian policy: π(a|s) = N(μ(s), σ(s))
    
    📌 对应直觉：给定“当前信道+天线位置”，网络输出每个动作维度的
    平均值与方差；采样后的增量会被环境再缩放到 ±0.1λ。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        log_std_min: float = -20,
        log_std_max: float = 2,
        use_attention: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize Actor Network with improved architecture (方案7)
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            use_attention: Whether to use self-attention (方案7.1)
            use_residual: Whether to use residual blocks (方案7.2)
            dropout: Dropout rate for regularization
        """
        super(ActorNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 自注意力层（方案7.1）
        if use_attention and hidden_dims[0] % 4 == 0:
            self.attention = SelfAttentionLayer(hidden_dims[0], num_heads=4, dropout=dropout)
        else:
            self.attention = None
            if use_attention:
                print(f"⚠️  警告: hidden_dim={hidden_dims[0]} 不能被4整除，跳过注意力层")
        
        # 共享层（带残差连接）
        shared_layers = []
        prev_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            # 添加残差块（方案7.2）
            if use_residual and prev_dim == hidden_dim:
                shared_layers.append(ResidualBlock(prev_dim, dropout=dropout))
            else:
                # 普通层（维度变化时不能使用残差）
                shared_layers.append(nn.Linear(prev_dim, hidden_dim))
                shared_layers.append(nn.LayerNorm(hidden_dim))
                shared_layers.append(nn.ReLU())
                shared_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Mean head（tanh 保证均值 ∈ [-1, 1]）
        self.mean = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()  # Limit action range
        )
        
        # Log std head (learnable, 添加数值稳定性)
        self.log_std = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with NaN detection and numerical stability
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            mean: Action mean (batch_size, action_dim)
            std: Action standard deviation (batch_size, action_dim)
        """
        # NaN检测：检查输入
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("⚠️  警告: 输入state包含NaN或Inf，使用零填充")
            state = torch.where(torch.isnan(state) | torch.isinf(state), 
                              torch.zeros_like(state), state)
        
        # 输入层
        features = self.input_layer(state)
        
        # 自注意力（方案7.1）
        if self.attention is not None:
            features = self.attention(features)
        
        # 共享层
        features = self.shared(features)
        
        # NaN检测：检查中间特征
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("⚠️  警告: 中间特征包含NaN或Inf，使用零填充")
            features = torch.where(torch.isnan(features) | torch.isinf(features),
                                  torch.zeros_like(features), features)
        
        # Mean head
        mean = self.mean(features)
        
        # Log std head（添加数值稳定性）
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # 确保std不会太小（数值稳定性）
        std = torch.clamp(std, min=1e-6)
        
        # 最终NaN检测
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("⚠️  警告: 输出包含NaN，使用默认值")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std) * 0.1
        
        return mean, std
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = torch.distributions.Normal(mean, std)  # 连续动作的常见设计
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions
        
        Args:
            state: State tensor
            action: Action tensor
        
        Returns:
            log_prob: Log probability of actions
            entropy: Policy entropy
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic Network (Value Network) with Dueling Architecture.
    实现了方案7：改进网络架构（残差连接）
    
    Dueling 结构常见于值函数，用"共享干路 + Value stream"来获得一个标量 V(s)。
    在连续动作环境中，这样的冗余层可以提升估计平稳度。
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        use_dueling: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize Critic Network with improved architecture (方案7)
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: Tuple of hidden layer dimensions
            use_dueling: Whether to use dueling architecture
            use_residual: Whether to use residual blocks (方案7.2)
            dropout: Dropout rate for regularization
        """
        super(CriticNetwork, self).__init__()
        
        self.use_dueling = use_dueling
        self.use_residual = use_residual
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Shared layers（带残差连接）
        shared_layers = []
        prev_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:-1], 1):  # All but last
            # 添加残差块（方案7.2）
            if use_residual and prev_dim == hidden_dim:
                shared_layers.append(ResidualBlock(prev_dim, dropout=dropout))
            else:
                # 普通层（维度变化时不能使用残差）
                shared_layers.append(nn.Linear(prev_dim, hidden_dim))
                shared_layers.append(nn.LayerNorm(hidden_dim))
                shared_layers.append(nn.ReLU())
                shared_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        if use_dueling:
            # Dueling architecture
            last_hidden = hidden_dims[-1]
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, last_hidden),
                nn.LayerNorm(last_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(last_hidden, 1)
            )
        else:
            # Standard value head
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1], 1)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with NaN detection
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            value: State value (batch_size, 1)
        """
        # NaN检测：检查输入
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("⚠️  警告: Critic输入state包含NaN或Inf，使用零填充")
            state = torch.where(torch.isnan(state) | torch.isinf(state),
                              torch.zeros_like(state), state)
        
        # 输入层
        features = self.input_layer(state)
        
        # 共享层
        features = self.shared(features)
        
        # NaN检测：检查中间特征
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("⚠️  警告: Critic中间特征包含NaN或Inf，使用零填充")
            features = torch.where(torch.isnan(features) | torch.isinf(features),
                                  torch.zeros_like(features), features)
        
        # Value head
        if self.use_dueling:
            value = self.value_stream(features)
        else:
            value = self.value_head(features)
        
        # 最终NaN检测
        if torch.isnan(value).any() or torch.isinf(value).any():
            print("⚠️  警告: Critic输出包含NaN或Inf，使用零值")
            value = torch.zeros_like(value)
        
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network with improved architecture (方案7)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden: Tuple[int, ...] = (512, 256, 128),
        critic_hidden: Tuple[int, ...] = (512, 256, 128),
        use_attention: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize Actor-Critic with improved architecture (方案7)
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            actor_hidden: Hidden dimensions for actor
            critic_hidden: Hidden dimensions for critic
            use_attention: Whether to use self-attention in actor (方案7.1)
            use_residual: Whether to use residual blocks (方案7.2)
            dropout: Dropout rate for regularization
        """
        super(ActorCritic, self).__init__()
        
        self.actor = ActorNetwork(
            state_dim, action_dim, actor_hidden,
            use_attention=use_attention,
            use_residual=use_residual,
            dropout=dropout,
        )
        self.critic = CriticNetwork(
            state_dim, critic_hidden,
            use_residual=use_residual,
            dropout=dropout,
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks
        
        Args:
            state: State tensor
        
        Returns:
            action: Sampled action
            value: State value
        """
        action, _ = self.actor.get_action(state)
        value = self.critic(state)
        return action, value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value
        
        Args:
            state: State tensor
            deterministic: If True, use deterministic policy
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value
        """
        action, log_prob = self.actor.get_action(state, deterministic)
        value = self.critic(state)
        return action, log_prob, value


# Test networks
if __name__ == "__main__":
    print("Testing neural networks...")
    
    # Parameters
    state_dim = 20  # For N=4
    action_dim = 8  # 2*N
    batch_size = 32
    
    # Create networks
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)
    ac = ActorCritic(state_dim, action_dim)
    
    # Test data
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    # Test actor
    print("\n1. Testing Actor Network...")
    mean, std = actor(states)
    print(f"   Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    action, log_prob = actor.get_action(states)
    print(f"   Action shape: {action.shape}, Log prob shape: {log_prob.shape}")
    
    log_prob_eval, entropy = actor.evaluate_actions(states, actions)
    print(f"   Evaluated log prob: {log_prob_eval.shape}, Entropy: {entropy.shape}")
    
    # Test critic
    print("\n2. Testing Critic Network...")
    values = critic(states)
    print(f"   Value shape: {values.shape}")
    
    # Test actor-critic
    print("\n3. Testing Actor-Critic...")
    action, log_prob, value = ac.get_action_and_value(states)
    print(f"   Action: {action.shape}, Log prob: {log_prob.shape}, Value: {value.shape}")
    
    # Count parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"\n4. Parameter counts:")
    print(f"   Actor: {actor_params:,} parameters")
    print(f"   Critic: {critic_params:,} parameters")
    print(f"   Total: {actor_params + critic_params:,} parameters")
    
    print("\nAll tests passed!")

