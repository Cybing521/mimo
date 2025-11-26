"""
Deep Reinforcement Learning for Movable Antenna Optimization
=============================================================

This package implements DRL-based optimization for MA-MIMO systems.

Modules:
- env: Gym environment for MA-MIMO
- agent: PPO agent implementation
- networks: Actor-Critic neural networks
- state_encoder: State space encoder
- utils: Utility functions
"""

from .env import MAMIMOEnv
from .agent import PPOAgent
from .networks import ActorNetwork, CriticNetwork

__all__ = [
    'MAMIMOEnv',
    'PPOAgent',
    'ActorNetwork',
    'CriticNetwork',
]

__version__ = '1.0.0'

