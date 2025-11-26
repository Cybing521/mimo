"""
Training Script for DRL-MA-MIMO
================================

Train PPO agent for movable antenna optimization.
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.env import MAMIMOEnv
from drl.agent import PPOAgent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DRL agent for MA-MIMO')
    
    # Environment parameters
    parser.add_argument('--N', type=int, default=4, help='Number of transmit antennas')
    parser.add_argument('--M', type=int, default=4, help='Number of receive antennas')
    parser.add_argument('--Lt', type=int, default=5, help='Number of transmit paths')
    parser.add_argument('--Lr', type=int, default=5, help='Number of receive paths')
    parser.add_argument('--SNR_dB', type=float, default=15.0, help='SNR in dB')
    parser.add_argument('--A_lambda', type=float, default=3.0, help='Normalized region size')
    parser.add_argument('--max_steps', type=int, default=50, help='Max steps per episode')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate every N episodes')
    parser.add_argument('--save_interval', type=int, default=500, help='Save model every N episodes')
    parser.add_argument('--save_dir', type=str, default='results/drl_training', help='Save directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(env, agent, num_episodes=10):
    """
    Evaluate agent performance
    
    Args:
        env: Environment
        agent: Trained agent
        num_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary of evaluation metrics
    """
    capacities = []
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        capacities.append(info['capacity'])
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'mean_capacity': np.mean(capacities),
        'std_capacity': np.std(capacities),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
    }


def train(args):
    """Main training loop"""
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}")
    
    # Create environment
    env = MAMIMOEnv(
        N=args.N,
        M=args.M,
        Lt=args.Lt,
        Lr=args.Lr,
        SNR_dB=args.SNR_dB,
        A_lambda=args.A_lambda,
        max_steps=args.max_steps,
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"\nEnvironment created:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  N={args.N}, M={args.M}, Lt={args.Lt}, Lr={args.Lr}")
    print(f"  SNR={args.SNR_dB}dB, A={args.A_lambda}λ")
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )
    
    print(f"\nAgent created:")
    print(f"  Actor params: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in agent.critic.parameters()):,}")
    print(f"  Device: {args.device}")
    
    # Training loop
    print(f"\nStarting training for {args.num_episodes} episodes...")
    
    episode_rewards = []
    episode_capacities = []
    eval_capacities = []
    eval_episodes = []
    
    best_eval_capacity = -np.inf
    
    progress_bar = tqdm(range(args.num_episodes), desc='Training')
    
    for episode in progress_bar:
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Collect experience
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
        
        # Update agent
        update_stats = agent.update(state)
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_capacities.append(info['capacity'])
        
        # Logging
        if (episode + 1) % args.log_interval == 0:
            mean_reward = np.mean(episode_rewards[-args.log_interval:])
            mean_capacity = np.mean(episode_capacities[-args.log_interval:])
            
            log_str = f"Ep {episode+1}/{args.num_episodes} | "
            log_str += f"Reward: {mean_reward:.2f} | "
            log_str += f"Capacity: {mean_capacity:.2f} | "
            
            if update_stats:
                log_str += f"Actor Loss: {update_stats['actor_loss']:.4f} | "
                log_str += f"Critic Loss: {update_stats['critic_loss']:.4f}"
            
            progress_bar.set_description(log_str)
        
        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_agent(env, agent, num_episodes=10)
            eval_capacities.append(eval_metrics['mean_capacity'])
            eval_episodes.append(episode + 1)
            
            print(f"\n=== Evaluation at episode {episode+1} ===")
            print(f"Mean capacity: {eval_metrics['mean_capacity']:.2f} ± {eval_metrics['std_capacity']:.2f}")
            print(f"Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            
            # Save best model
            if eval_metrics['mean_capacity'] > best_eval_capacity:
                best_eval_capacity = eval_metrics['mean_capacity']
                best_model_path = os.path.join(run_dir, 'best_model.pth')
                agent.save(best_model_path)
                print(f"✓ New best model saved! Capacity: {best_eval_capacity:.2f}")
        
        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_ep{episode+1}.pth')
            agent.save(checkpoint_path)
    
    # Final save
    final_model_path = os.path.join(run_dir, 'final_model.pth')
    agent.save(final_model_path)
    
    # Save training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(episode_capacities)
    plt.xlabel('Episode')
    plt.ylabel('Capacity (bps/Hz)')
    plt.title('Training Capacity')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(eval_episodes, eval_capacities, 'o-')
    plt.xlabel('Episode')
    plt.ylabel('Eval Capacity (bps/Hz)')
    plt.title('Evaluation Capacity')
    plt.grid(True)
    
    plt.tight_layout()
    fig_path = os.path.join(run_dir, 'training_curves.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {fig_path}")
    
    # Save metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_capacities': episode_capacities,
        'eval_episodes': eval_episodes,
        'eval_capacities': eval_capacities,
        'best_eval_capacity': float(best_eval_capacity),
    }
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best evaluation capacity: {best_eval_capacity:.2f} bps/Hz")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*60}")
    
    return agent, run_dir


if __name__ == "__main__":
    args = parse_args()
    agent, run_dir = train(args)

