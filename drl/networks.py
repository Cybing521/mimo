"""
Neural Network Architectures for PPO Agent
===========================================

Actor-Critic networks with support for continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ActorNetwork(nn.Module):
    """
    Actor Network (Policy Network)
    
    Outputs Gaussian policy: π(a|s) = N(μ(s), σ(s))
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """
        Initialize Actor Network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(ActorNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Mean head
        self.mean = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()  # Limit action range
        )
        
        # Log std head (learnable)
        self.log_std = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            mean: Action mean (batch_size, action_dim)
            std: Action standard deviation (batch_size, action_dim)
        """
        features = self.shared(state)
        
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
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
            dist = torch.distributions.Normal(mean, std)
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
    Critic Network (Value Network) with Dueling Architecture
    
    Outputs state value: V(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        use_dueling: bool = True,
    ):
        """
        Initialize Critic Network
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: Tuple of hidden layer dimensions
            use_dueling: Whether to use dueling architecture
        """
        super(CriticNetwork, self).__init__()
        
        self.use_dueling = use_dueling
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:  # All but last
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling architecture
            last_hidden = hidden_dims[-1]
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, last_hidden),
                nn.ReLU(),
                nn.Linear(last_hidden, 1)
            )
        else:
            # Standard value head
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            value: State value (batch_size, 1)
        """
        features = self.shared(state)
        
        if self.use_dueling:
            value = self.value_stream(features)
        else:
            value = self.value_head(features)
        
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden: Tuple[int, ...] = (512, 256, 128),
        critic_hidden: Tuple[int, ...] = (512, 256, 128),
    ):
        """
        Initialize Actor-Critic
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            actor_hidden: Hidden dimensions for actor
            critic_hidden: Hidden dimensions for critic
        """
        super(ActorCritic, self).__init__()
        
        self.actor = ActorNetwork(state_dim, action_dim, actor_hidden)
        self.critic = CriticNetwork(state_dim, critic_hidden)
    
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

