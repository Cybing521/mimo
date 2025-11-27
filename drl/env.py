"""
Gym Environment for Movable Antenna MIMO System
================================================

This module implements a custom Gym environment for training DRL agents
to optimize antenna positions in MA-MIMO systems.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mimo_core import MIMOSystem


class MAMIMOEnv(gym.Env):
    """
    Custom Gym Environment for MA-MIMO Optimization.

    🌐 状态空间（agent 看到的观测）：
        - 4 个主特征值：近似表征信道的“通道等级”。
        - 3 个统计量：信道功率、条件数、相位方差，帮助 agent 估计信道质量。
        - 天线位置信息：Tx 和 Rx 各自的坐标，被展平成 2(N+M) 维。
        - 5 条容量历史：短期性能回顾，可让 agent 感知走势。
        → 总维度 4 + 3 + 2(N+M) + 5

    🕹 动作空间：
        - 输入为 [-1, 1] 区间的标准化位移。
        - 内部再缩放到 ±0.1 λ，使一步移动不过大，便于稳定训练。

    🏆 奖励设计：
        - 以容量及容量提升为主奖励（更高/更快提升更好）。
        - 对违反约束（越界、天线太近）给予惩罚。
        - 额外加入效率奖励与平滑惩罚，防止“抽搐式”移动。
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        N: int = 4,
        M: int = 4,
        Lt: int = 5,
        Lr: int = 5,
        SNR_dB: float = 15.0,
        A_lambda: float = 3.0,
        max_steps: int = 50,
        reward_config: Optional[Dict] = None,
    ):
        """
        Initialize the MA-MIMO environment
        
        Args:
            N: Number of transmit antennas
            M: Number of receive antennas
            Lt: Number of transmit paths
            Lr: Number of receive paths
            SNR_dB: Signal-to-noise ratio in dB
            A_lambda: Normalized region size (in wavelengths)
            max_steps: Maximum steps per episode
            reward_config: Configuration for reward function
        """
        super(MAMIMOEnv, self).__init__()
        
        # System parameters（环境配置，决定信道和阵列规模）
        self.N = N
        self.M = M
        self.Lt = Lt
        self.Lr = Lr
        self.SNR_dB = SNR_dB
        self.A_lambda = A_lambda
        self.max_steps = max_steps
        
        # Create MIMO system（复用论文实现，保证与仿真设置一致）
        self.mimo_system = MIMOSystem(
            N=N, M=M, Lt=Lt, Lr=Lr, 
            SNR_dB=SNR_dB, lambda_val=1.0
        )
        
        self.square_size = A_lambda * self.mimo_system.lambda_val
        self.D = self.mimo_system.D  # Minimum distance
        
        # Reward configuration
        self.reward_config = reward_config or {
            'w_capacity': 0.5,
            'w_improvement': 3.0,
            'w_distance_penalty': 5.0,
            'w_boundary_penalty': 5.0,
            'w_efficiency': 0.1,
            'w_smooth': 0.5,
        }
        
        # Action scaling (normalized actions ∈ [-1, 1])
        self.max_delta = 0.1 * self.mimo_system.lambda_val
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * (N + M),),  # control Tx and Rx arrays jointly
            dtype=np.float32
        )
        
        # State space dimension（这里不直接依赖 gym.spaces.Dict，方便 PPO 处理向量）
        state_dim = 4 + 3 + 2*(N + M) + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.capacity_history = []
        self.capacity_normalizer = 30.0  # empirical upper bound
        
        # Current state（训练过程中持续更新）
        self.t = None  # Transmit antenna positions (2, N)
        self.r = None  # Receive antenna positions (2, M)
        self.Q = None  # Power allocation matrix
        self.H_r = None  # Channel matrix
        self.current_capacity = 0.0
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial state observation
        """
        # Initialize antenna positions using smart initialization
        try:
            self.t = self.mimo_system.initialize_antennas_smart(
                self.N, self.square_size
            )
            self.r = self.mimo_system.initialize_antennas_smart(
                self.M, self.square_size
            )
        except RuntimeError:
            # Fallback to random initialization
            self.t = np.random.rand(2, self.N) * self.square_size
            self.r = np.random.rand(2, self.M) * self.square_size
        
        # Reset tracking variables
        self.current_step = 0
        self.capacity_history = []
        
        # Reset per-episode channel parameters（每个 episode 抽样新的 Rician 信道）
        self._initialize_channel_params()
        
        # Compute initial channel and capacity
        self._update_channel()
        
        # Return initial state
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Position increments [Δx₁, Δy₁, ..., ΔxN, ΔyN]
        
        Returns:
            observation: Next state
            reward: Reward signal
            done: Whether episode is finished
            info: Additional information
        """
        # Store previous capacity and positions for smoothness penalties
        prev_capacity = self.current_capacity
        prev_t = self.t.copy()
        prev_r = self.r.copy()
        
        # Split normalized actions into Tx/Rx components
        action = np.clip(action, -1.0, 1.0)
        tx_action = action[:2 * self.N].reshape(2, self.N) * self.max_delta
        rx_action = action[2 * self.N:].reshape(2, self.M) * self.max_delta
        
        # Update antenna positions with projection
        new_t = self.t + tx_action
        new_r = self.r + rx_action
        
        # Project to feasible region（软投影，避免动作“闯红灯”）
        new_t = self._project_to_feasible_region(new_t, self.t, is_transmit=True)
        new_r = self._project_to_feasible_region(new_r, self.r, is_transmit=False)
        self.t = new_t
        self.r = new_r
        
        # Update channel and compute new capacity
        self._update_channel()
        
        # Compute reward（包含奖励与惩罚的混合信号）
        reward = self._compute_reward(prev_capacity, prev_t, prev_r)
        
        # Update tracking
        self.current_step += 1
        self.capacity_history.append(self.current_capacity)
        if len(self.capacity_history) > 5:
            self.capacity_history.pop(0)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'capacity': self.current_capacity,
            'step': self.current_step,
            'constraint_violations': self._count_violations(),
        }
        
        return self._get_state(), reward, done, info
    
    def _initialize_channel_params(self):
        """Sample per-episode channel parameters"""
        self._theta_p = np.random.rand(self.Lt) * np.pi
        self._phi_p = np.random.rand(self.Lt) * np.pi
        self._theta_q = np.random.rand(self.Lr) * np.pi
        self._phi_q = np.random.rand(self.Lr) * np.pi
        
        # Rician channel core
        self._Sigma = np.zeros((self.Lr, self.Lt), dtype=complex)
        kappa = 1.0
        self._Sigma[0, 0] = (np.random.randn() + 1j*np.random.randn()) * \
                           np.sqrt(kappa/(kappa+1)/2)
        for i in range(1, min(self.Lr, self.Lt)):
            self._Sigma[i, i] = (np.random.randn() + 1j*np.random.randn()) * \
                               np.sqrt(1/((kappa+1)*(self.Lr-1))/2)
    
    def _update_channel(self):
        """Update channel matrix and capacity"""
        # Compute channel matrix
        F = self.mimo_system.calculate_F(self._theta_q, self._phi_q, self.r)
        G = self.mimo_system.calculate_G(self._theta_p, self._phi_p, self.t)
        self.H_r = F.T.conj() @ self._Sigma @ G
        
        # Optimize power allocation Q using water-filling
        H_HH = self.H_r @ self.H_r.T.conj()
        eigvals, eigvecs = np.linalg.eigh(H_HH)
        eigvals = np.maximum(eigvals, 0)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        p_alloc = self._water_filling(eigvals, self.mimo_system.power, self.mimo_system.sigma)
        self.Q = eigvecs @ np.diag(p_alloc) @ eigvecs.T.conj()
        
        # Compute capacity
        H_rQH = self.H_r @ self.Q @ self.H_r.T.conj()
        eigvals_cap = np.linalg.eigvalsh(
            np.eye(self.M) + (1/self.mimo_system.sigma) * H_rQH
        )
        self.current_capacity = np.sum(np.log2(np.maximum(eigvals_cap, 1e-10)))
    
    def _water_filling(self, gains: np.ndarray, total_power: float, noise_var: float) -> np.ndarray:
        """Classic water-filling over eigenmodes"""
        alloc = np.zeros_like(gains, dtype=float)
        if gains.size == 0 or total_power <= 0:
            return alloc
        
        positive = gains > 1e-10
        if not np.any(positive):
            return alloc
        
        gains_pos = gains[positive]
        inv = noise_var / (gains_pos + 1e-12)
        sort_idx = np.argsort(inv)
        inv_sorted = inv[sort_idx]
        n = len(inv_sorted)
        
        active = n
        water_level = 0.0
        for k in range(1, n + 1):
            mu = (total_power + np.sum(inv_sorted[:k])) / k
            if k == n or mu <= inv_sorted[k]:
                active = k
                water_level = mu
                break
        
        powers_sorted = np.maximum(water_level - inv_sorted[:active], 0.0)
        alloc_pos = np.zeros_like(gains_pos)
        alloc_pos[sort_idx[:active]] = powers_sorted
        
        alloc[positive] = alloc_pos
        return alloc
    
    def _get_state(self) -> np.ndarray:
        """
        Encode current state
        
        Returns:
            State vector with dimension (4 + 3 + 2(N+M) + 5)
        """
        # 1. Channel eigenvalues (4 dims)
        H_HH = self.H_r @ self.H_r.T.conj()
        full_eigvals = np.linalg.eigvalsh(H_HH)
        eigvals = np.sort(full_eigvals)[::-1]
        if eigvals.size > 0:
            eigvals = eigvals / (np.max(eigvals) + 1e-8)  # Normalize
        else:
            eigvals = np.zeros(4, dtype=np.float32)
        if eigvals.size < 4:
            padded = np.zeros(4, dtype=np.float32)
            padded[:eigvals.size] = eigvals
            eigvals = padded
        else:
            eigvals = eigvals[:4]
        
        # 2. Channel features (3 dims)
        channel_power = np.linalg.norm(self.H_r, 'fro')
        cond_number = (np.max(full_eigvals) + 1e-8) / (np.min(full_eigvals) + 1e-8)
        phase_variance = np.var(np.angle(self.H_r))
        
        channel_features = np.array([
            channel_power / 10.0,  # Normalize
            np.log10(cond_number + 1) / 5.0,
            phase_variance / np.pi
        ])
        
        # 3. Antenna positions (2(N+M) dims)
        tx_positions = self.t.flatten() / self.square_size
        rx_positions = self.r.flatten() / self.square_size
        
        # 4. Capacity history (5 dims, padded with current capacity)
        history = self.capacity_history[-5:] if self.capacity_history else []
        history_padded = history + [self.current_capacity] * (5 - len(history))
        history_normalized = np.array(history_padded) / self.capacity_normalizer
        
        # Concatenate all features
        state = np.concatenate([
            eigvals,
            channel_features,
            tx_positions,
            rx_positions,
            history_normalized,
        ]).astype(np.float32)
        
        return state
    
    def _compute_reward(self, prev_capacity: float, prev_t: np.ndarray, prev_r: np.ndarray) -> float:
        """
        Compute reward signal
        
        Args:
            prev_capacity: Capacity in previous step
        
        Returns:
            Total reward
        """
        cfg = self.reward_config
        
        # 1. Capacity reward (normalized and centered)
        normalized_capacity = (self.current_capacity / self.capacity_normalizer) - 0.5
        r_capacity = cfg['w_capacity'] * normalized_capacity
        
        # 2. Improvement reward (tanh to keep bounded)
        improvement = self.current_capacity - prev_capacity
        r_improvement = cfg['w_improvement'] * np.tanh(improvement)
        
        # 3. Constraint penalties
        distance_violations, boundary_violations = self._constraint_stats()
        r_constraint = (
            -cfg['w_distance_penalty'] * distance_violations
            -cfg['w_boundary_penalty'] * boundary_violations
        )
        # 4. Efficiency bonus (capacity per unit power)
        total_power = np.sum(np.abs(np.diag(self.H_r @ self.H_r.T.conj())))
        r_efficiency = cfg['w_efficiency'] * (self.current_capacity / (total_power + 1e-8))
        
        # 5. Smoothness penalty to avoid thrashing
        delta_t = np.linalg.norm(self.t - prev_t, axis=0).mean()
        delta_r = np.linalg.norm(self.r - prev_r, axis=0).mean()
        r_smooth = -cfg['w_smooth'] * (delta_t + delta_r) / (self.max_delta + 1e-8)
        
        # Total reward
        reward = r_capacity + r_improvement + r_constraint + r_efficiency + r_smooth
        
        return float(reward)
    
    def _project_to_feasible_region(
        self, 
        new_positions: np.ndarray, 
        old_positions: np.ndarray,
        is_transmit: bool = True
    ) -> np.ndarray:
        """
        Project antenna positions to feasible region
        
        Args:
            new_positions: Proposed new positions (2, N or M)
            old_positions: Current positions
            is_transmit: Whether these are transmit antennas
        
        Returns:
            Feasible positions
        """
        num_antennas = new_positions.shape[1]
        feasible = new_positions.copy()
        
        # Clip to region boundaries
        feasible = np.clip(feasible, 0, self.square_size)
        
        # Check minimum distance constraints
        max_iterations = 10
        for _ in range(max_iterations):
            violations = []
            for i in range(num_antennas):
                for j in range(i+1, num_antennas):
                    dist = np.linalg.norm(feasible[:, i] - feasible[:, j])
                    if dist < self.D:
                        # Push apart
                        direction = (feasible[:, i] - feasible[:, j]) / (dist + 1e-8)
                        push = (self.D - dist) / 2
                        feasible[:, i] += direction * push
                        feasible[:, j] -= direction * push
                        violations.append((i, j))
            
            # Re-clip after pushing
            feasible = np.clip(feasible, 0, self.square_size)
            
            if len(violations) == 0:
                break
        
        return feasible
    
    def _constraint_stats(self) -> Tuple[int, int]:
        """Return (distance violations, boundary violations)"""
        distance_violations = self._pairwise_violations(self.t) + self._pairwise_violations(self.r)
        boundary_violations = 0
        if np.any(self.t < 0) or np.any(self.t > self.square_size):
            boundary_violations += self.N
        if np.any(self.r < 0) or np.any(self.r > self.square_size):
            boundary_violations += self.M
        return distance_violations, boundary_violations
    
    def _count_violations(self) -> int:
        """Count number of constraint violations"""
        distance, boundary = self._constraint_stats()
        return distance + boundary
    
    def _pairwise_violations(self, positions: np.ndarray) -> int:
        """Count pairwise minimum-distance violations for a given array"""
        violations = 0
        num_antennas = positions.shape[1]
        for i in range(num_antennas):
            for j in range(i+1, num_antennas):
                dist = np.linalg.norm(positions[:, i] - positions[:, j])
                if dist < self.D:
                    violations += 1
        return violations
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Capacity: {self.current_capacity:.2f}")
    
    def close(self):
        """Clean up resources"""
        pass


# Test the environment
if __name__ == "__main__":
    # Create environment
    env = MAMIMOEnv(N=4, M=4, Lt=5, Lr=5, SNR_dB=15, A_lambda=3.0)
    
    # Test reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial capacity: {env.current_capacity:.2f}")
    
    # Test step
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(f"After step - Capacity: {info['capacity']:.2f}, Reward: {reward:.2f}")
    
    # Run a short episode
    state = env.reset()
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: Capacity={info['capacity']:.2f}, Reward={reward:.2f}")
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print("Environment test passed!")

