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
from typing import Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.env import MAMIMOEnv
from drl.agent import PPOAgent
from utils.wandb_utils import (
    init_wandb,
    log_image,
    log_line_series,
    log_metrics,
    ensure_wandb_api_key,
)


def get_device(device_preference: str = 'cpu') -> str:
    """
    智能设备选择：根据用户偏好和系统可用性自动选择最佳设备。
    
    Args:
        device_preference: 用户偏好的设备 ('cpu', 'cuda', 'mps', 'auto')
    
    Returns:
        实际可用的设备字符串
    """
    if device_preference == 'auto':
        # 自动选择：优先 CUDA > MPS > CPU
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    elif device_preference == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            print("⚠️  警告: CUDA 不可用，回退到 CPU")
            print("   提示: macOS 不支持 CUDA，请使用 'cpu' 或 'mps'（Apple Silicon）")
            return 'cpu'
    
    elif device_preference == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            print("⚠️  警告: MPS 不可用，回退到 CPU")
            return 'cpu'
    
    else:  # 'cpu' 或其他
        return 'cpu'


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DRL agent for MA-MIMO')
    
    # ===== 环境/仿真规模相关超参 =====
    # Environment parameters
    parser.add_argument('--N', type=int, default=4, help='Number of transmit antennas')
    parser.add_argument('--M', type=int, default=4, help='Number of receive antennas')
    parser.add_argument('--Lt', type=int, default=5, help='Number of transmit paths')
    parser.add_argument('--Lr', type=int, default=5, help='Number of receive paths')
    parser.add_argument('--SNR_dB', type=float, default=15.0, help='SNR in dB')
    parser.add_argument('--A_lambda', type=float, default=3.0, help='Normalized region size')
    parser.add_argument('--max_steps', type=int, default=50, help='Max steps per episode')
    
    # ===== PPO 训练核心超参 =====
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
    parser.add_argument('--min_entropy_coef', type=float, default=0.001,
                        help='Minimum entropy coefficient after decay')
    parser.add_argument('--rollout_episodes', type=int, default=4,
                        help='Number of episodes to collect before one PPO update')
    parser.add_argument('--lr_anneal', action='store_true',
                        help='Linearly anneal actor/critic learning rates')
    parser.add_argument('--min_lr_factor', type=float, default=0.1,
                        help='Lower bound as a fraction of initial LR when annealing')
    
    # ===== 日志与模型保存频率控制 =====
    # Logging
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate every N episodes')
    parser.add_argument('--save_interval', type=int, default=500, help='Save model every N episodes')
    parser.add_argument('--save_dir', type=str, default='results/drl_training', help='Save directory')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Episodes for each evaluation')
    parser.add_argument('--eval_seed', type=int, default=2024,
                        help='Seed used during evaluation to reduce variance')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['cpu', 'cuda', 'mps', 'auto'], 
                       help='Device (auto: 自动选择最佳可用设备)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # WandB logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='ma-mimo', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (team) name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom WandB run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'], help='WandB run mode')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='Optional WandB tags')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    # Gym、NumPy、PyTorch 都需要分别设定随机种子以保持实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(env, agent, num_episodes=10, seed: Optional[int] = None):
    """
    Evaluate agent performance
    
    采用确定性策略在相同环境上跑多次 episode，统计容量/奖励/长度等指标，
    用于监控训练过程中策略性能是否真正提升。
    
    Args:
        env: Environment
        agent: Trained agent
        num_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary of evaluation metrics
    """
    original_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
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
    
    eval_metrics = {
        'mean_capacity': np.mean(capacities),
        'std_capacity': np.std(capacities),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
    }
    np.random.set_state(original_state)
    return eval_metrics


def train(args):
    """Main training loop"""
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories（每次运行独立输出，方便对比）
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}")
    
    # Initialize WandB（可选在线可视化）
    ensure_wandb_api_key()
    wandb_run = init_wandb(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or f"drl_train_{timestamp}",
        mode=args.wandb_mode,
        config=vars(args),
        tags=args.wandb_tags,
        run_dir=run_dir,
    )
    
    # Create environment（训练与评估共用同一个 env，保持配置一致）
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
    
    # 智能设备选择
    actual_device = get_device(args.device)
    if actual_device != args.device:
        print(f"设备选择: {args.device} -> {actual_device}")
    else:
        print(f"使用设备: {actual_device}")
    
    # Create agent（所有超参由命令行指定，便于做 ablation）
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
        device=actual_device,
    )
    
    print(f"\nAgent created:")
    print(f"  Actor params: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in agent.critic.parameters()):,}")
    print(f"  Device: {args.device}")
    
    # Training loop（标准 PPO：收集一整条 episode → update → log/eval/save）
    print(f"\nStarting training for {args.num_episodes} episodes...")
    
    episode_rewards = []
    episode_capacities = []
    eval_capacities = []
    eval_episodes = []
    
    # 记录最优评估容量，用于“只保留最好模型”策略
    best_eval_capacity = -np.inf
    episodes_since_update = 0
    update_stats = {}
    
    progress_bar = tqdm(range(args.num_episodes), desc='Training')
    
    for episode in progress_bar:
        # ===== 改进探索策略：计算探索噪声 =====
        # 早期添加更多噪声，后期减少噪声
        if episode < args.num_episodes * 0.3:  # 前30%：高探索噪声
            exploration_noise = 0.3 * (1.0 - episode / (args.num_episodes * 0.3))
        elif episode < args.num_episodes * 0.6:  # 30%-60%：中等噪声
            exploration_noise = 0.1 * (1.0 - (episode - args.num_episodes * 0.3) / (args.num_episodes * 0.3))
        else:  # 后40%：无噪声
            exploration_noise = 0.0
        
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Collect experience（运行一整条 episode 的轨迹）
        while not done:
            action, log_prob, value = agent.select_action(
                state, 
                deterministic=False,
                exploration_noise=exploration_noise
            )
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
        
        episodes_since_update += 1
        should_update = (episodes_since_update >= args.rollout_episodes)
        is_last_episode = (episode == args.num_episodes - 1)
        
        if should_update or is_last_episode:
            # ===== 阶段 1：采样完成，进入 PPO 更新 =====
            update_stats = agent.update(state)
            episodes_since_update = 0
            
            # 自适应学习率与熵衰减（线性）
            progress = (episode + 1) / args.num_episodes
            decay_factor = max(args.min_lr_factor, 1.0 - progress) if args.lr_anneal else 1.0
            if args.lr_anneal:
                new_actor_lr = args.lr_actor * decay_factor
                new_critic_lr = args.lr_critic * decay_factor
                for param_group in agent.actor_optimizer.param_groups:
                    param_group['lr'] = new_actor_lr
                for param_group in agent.critic_optimizer.param_groups:
                    param_group['lr'] = new_critic_lr
            
            # ===== 改进探索策略：自适应熵系数 =====
            # 早期高探索（高熵），后期高利用（低熵）
            if episode < args.num_episodes * 0.2:  # 前20%：高探索
                adaptive_entropy = args.entropy_coef * 3.0  # 3倍熵系数
            elif episode < args.num_episodes * 0.5:  # 20%-50%：中等探索
                adaptive_entropy = args.entropy_coef * 2.0  # 2倍熵系数
            else:  # 后50%：低探索，高利用
                adaptive_entropy = args.min_entropy_coef + \
                    (args.entropy_coef - args.min_entropy_coef) * decay_factor
            
            agent.entropy_coef = adaptive_entropy
        
        # ===== 阶段 2：纪录训练指标，便于后续绘图 =====
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_capacities.append(info['capacity'])
        
        # Logging（按 log_interval 打印平滑指标）
        if (episode + 1) % args.log_interval == 0:
            mean_reward = np.mean(episode_rewards[-args.log_interval:])
            mean_capacity = np.mean(episode_capacities[-args.log_interval:])
            
            log_str = f"Ep {episode+1}/{args.num_episodes} | "
            log_str += f"Reward: {mean_reward:.2f} | "
            log_str += f"Capacity: {mean_capacity:.2f} | "
            
            if update_stats:
                log_str += f"Actor Loss: {update_stats['actor_loss']:.4f} | "
                log_str += f"Critic Loss: {update_stats['critic_loss']:.4f}"
            
            # 使用 refresh=False 避免重复输出，只在需要时刷新进度条
            progress_bar.set_description(log_str, refresh=True)
            
            log_payload = {
                'train/episode': episode + 1,
                'train/reward': mean_reward,
                'train/capacity': mean_capacity,
            }
            if update_stats:
                log_payload['loss/actor'] = update_stats['actor_loss']
                log_payload['loss/critic'] = update_stats['critic_loss']
            log_metrics(wandb_run, log_payload, step=episode + 1)
        
        # ===== 阶段 3：定期进行评估（deterministic policy），并保存最好模型 =====
        # Evaluation（deterministic 模式评估策略质量）
        if (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_agent(
                env, agent,
                num_episodes=args.eval_episodes,
                seed=args.eval_seed
            )
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
            
            log_metrics(
                wandb_run,
                {
                    'eval/episode': episode + 1,
                    'eval/mean_capacity': eval_metrics['mean_capacity'],
                    'eval/std_capacity': eval_metrics['std_capacity'],
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/std_reward': eval_metrics['std_reward'],
                },
                step=episode + 1,
            )
        
        # ===== 阶段 4：常规 checkpoint，防止训练中断导致结果丢失 =====
        # Save checkpoint（长时间训练时可恢复）
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_ep{episode+1}.pth')
            agent.save(checkpoint_path)
    
    # Final save
    final_model_path = os.path.join(run_dir, 'final_model.pth')
    agent.save(final_model_path)
    
    # Save training curves（生成用户问题中展示的三联图）
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

    # WandB logging（上传曲线与关键指标）
    if wandb_run is not None:
        wandb_run.summary['best_eval_capacity'] = best_eval_capacity
        log_image(
            wandb_run,
            key='plots/training_curves',
            image_path=fig_path,
            caption='DRL training curves',
            step=args.num_episodes,
        )
        episodes_axis = list(range(1, len(episode_rewards) + 1))
        log_line_series(
            wandb_run,
            episodes_axis,
            {
                'reward': episode_rewards,
                'capacity': episode_capacities,
            },
            title='Training reward & capacity',
            x_name='episode',
            key='plots/training_series',
        )
        if eval_capacities:
            log_line_series(
                wandb_run,
                eval_episodes,
                {'eval_capacity': eval_capacities},
                title='Evaluation capacity',
                x_name='episode',
                key='plots/eval_series',
            )
        log_metrics(
            wandb_run,
            {
                'final/best_eval_capacity': best_eval_capacity,
                'final/run_dir': run_dir,
            },
            step=args.num_episodes,
        )
        wandb_run.finish()
    
    return agent, run_dir


if __name__ == "__main__":
    args = parse_args()
    agent, run_dir = train(args)

