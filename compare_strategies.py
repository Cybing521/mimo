import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drl.env import MAMIMOEnv
from drl.agent import PPOAgent
from drl.sac_agent import SACAgent
from drl.td3_agent import TD3Agent
from drl.ddpg_agent import DDPGAgent

def train_agent(agent_name, agent_class, env, max_episodes=50, max_steps=50, seed=42):
    print(f"\nTraining {agent_name}...")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if agent_name == 'PPO':
        agent = agent_class(state_dim, action_dim, device='cpu')
    else:
        agent = agent_class(state_dim, action_dim, batch_size=64, device='cpu')
    
    rewards = []
    capacities = []
    
    for episode in range(max_episodes):
        state = env.reset(init_seed=seed+episode)
        episode_reward = 0
        episode_capacity = 0
        
        for step in range(max_steps):
            # Select action
            if agent_name == 'PPO':
                action, log_prob, value = agent.select_action(state)
            else:
                # Add exploration noise for off-policy agents during training
                if agent_name == 'TD3':
                    action = agent.select_action(state, noise=0.1)
                elif agent_name == 'DDPG':
                    action = agent.select_action(state, noise=0.1)
                else: # SAC
                    action = agent.select_action(state)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            if agent_name == 'PPO':
                agent.store_transition(state, action, reward, value, log_prob, done)
            else:
                agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent
            if agent_name == 'PPO':
                if (step + 1) % 20 == 0 or done: # Update periodically
                    agent.update(next_state)
            else:
                agent.update()
                
            state = next_state
            episode_reward += reward
            episode_capacity = max(episode_capacity, info['capacity'])
            
            if done:
                break
        
        rewards.append(episode_reward)
        capacities.append(episode_capacity)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{max_episodes}, Reward: {episode_reward:.2f}, Max Capacity: {episode_capacity:.2f}")
            
    return rewards, capacities

def main():
    # Environment config
    env_config = {
        'N': 4, 'M': 4,
        'Lt': 5, 'Lr': 5,
        'SNR_dB': 15,
        'A_lambda': 3.0,
        'max_steps': 50
    }
    
    env = MAMIMOEnv(**env_config)
    
    # Agents to compare
    agents = {
        'PPO': PPOAgent,
        'SAC': SACAgent,
        'TD3': TD3Agent,
        'DDPG': DDPGAgent
    }
    
    results = {}
    
    # Train each agent
    for name, agent_class in agents.items():
        rewards, capacities = train_agent(name, agent_class, env, max_episodes=20) # Short run for verification
        results[name] = {
            'rewards': rewards,
            'capacities': capacities
        }
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['rewards'], label=name)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['capacities'], label=name)
    plt.title('Max Capacity per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Capacity (bps/Hz)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("\nComparison complete. Results saved to comparison_results.png")

if __name__ == "__main__":
    main()
