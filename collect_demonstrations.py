"""
Collect Expert Demonstrations (AO) for Imitation Learning
=========================================================

This script runs the AO algorithm to generate a dataset of (state, action) pairs.
The dataset will be used to pre-train the RL agent via Behavior Cloning.
"""

import sys
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mimo_core import MIMOSystem
from drl.env import MAMIMOEnv

def collect_demonstrations(num_episodes=1000, save_path='data/expert_demos.pkl'):
    """
    Collect expert demonstrations from AO algorithm.
    
    Args:
        num_episodes: Number of episodes to run
        save_path: Path to save the dataset
    """
    print(f"Collecting {num_episodes} expert demonstrations...")
    
    # Environment parameters (must match training config)
    N, M = 4, 4
    Lt, Lr = 5, 5
    SNR_dB = 25.0
    A_lambda = 3.0
    
    env = MAMIMOEnv(N=N, M=M, Lt=Lt, Lr=Lr, SNR_dB=SNR_dB, A_lambda=A_lambda)
    mimo_system = MIMOSystem(N, M, Lt, Lr, SNR_dB)
    
    dataset = []
    
    for i in tqdm(range(num_episodes)):
        # 1. Reset environment to get a new channel state
        state = env.reset(init_seed=i)
        
        # 2. Extract channel parameters from environment
        channel_params = env.get_channel_params()
        
        # 3. Run AO algorithm to find optimal positions
        # We use 'Proposed' mode (AO)
        result = mimo_system.run_optimization(
            A_lambda, 
            mode='Proposed', 
            channel_params=channel_params
        )
        
        # 4. Extract optimal actions
        # The environment expects action in range [-1, 1] mapped to [0, square_size]
        # We need to normalize the AO output positions to [-1, 1]
        
        # AO returns positions in [0, square_size]
        # Action mapping in env: action = (pos / square_size) * 2 - 1
        square_size = A_lambda * mimo_system.lambda_val
        
        # Get optimal positions from AO
        t_opt = result['t'] # Shape (2, N)
        r_opt = result['r'] # Shape (2, M)
        
        # Normalize positions to action space [-1, 1]
        # Action mapping: action = (pos / square_size) * 2 - 1
        t_action = (t_opt / square_size) * 2 - 1
        r_action = (r_opt / square_size) * 2 - 1
        
        # Flatten to match action space
        # Action vector: [t_x1, t_y1, ..., t_xN, t_yN, r_x1, r_y1, ..., r_xM, r_yM]
        action = np.concatenate([t_action.flatten('F'), r_action.flatten('F')])
        
        # Store (state, action) pair
        dataset.append((state, action))
        
    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"Saved {len(dataset)} demonstrations to {save_path}")

if __name__ == "__main__":
    collect_demonstrations(num_episodes=1000)
