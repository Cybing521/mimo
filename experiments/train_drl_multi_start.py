"""
多起点训练脚本：方案3 + 方案4
============================

实现：
1. 多起点训练：训练多个模型，每个使用不同的随机种子和初始化
2. 改进探索策略：自适应熵系数 + 动作噪声注入
3. 选择最佳模型：评估所有模型，选择性能最好的
"""

import sys
import os
import argparse
import numpy as np
import torch
from datetime import datetime
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from train_drl module (using importlib for relative import)
import importlib.util
train_drl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_drl.py')
spec = importlib.util.spec_from_file_location("train_drl", train_drl_path)
train_drl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_drl)

train = train_drl.train
parse_args = train_drl.parse_args
evaluate_agent = train_drl.evaluate_agent
set_seed = train_drl.set_seed

from drl.env import MAMIMOEnv
from drl.agent import PPOAgent


def train_single_model(
    model_id: int,
    base_args: argparse.Namespace,
    base_seed: int = 42,
) -> tuple:
    """
    训练单个模型（使用不同的随机种子）
    
    Args:
        model_id: 模型ID（0, 1, 2, ...）
        base_args: 基础参数
        base_seed: 基础随机种子
    
    Returns:
        (agent, run_dir, eval_capacity)
    """
    print(f"\n{'='*70}")
    print(f"训练模型 {model_id + 1}/{base_args.num_models}")
    print(f"{'='*70}")
    
    # 为每个模型使用不同的随机种子
    model_seed = base_seed + model_id * 1000
    
    # 创建参数副本并修改种子
    args = argparse.Namespace(**vars(base_args))
    args.seed = model_seed
    args.wandb_run_name = f"{base_args.wandb_run_name or 'drl_multi'}_model{model_id+1}"
    
    # 训练模型
    agent, run_dir = train(args)
    
    # 评估模型
    env = MAMIMOEnv(
        N=args.N, M=args.M, Lt=args.Lt, Lr=args.Lr,
        SNR_dB=args.SNR_dB, A_lambda=args.A_lambda,
        max_steps=args.max_steps,
    )
    
    eval_metrics = evaluate_agent(
        env, agent,
        num_episodes=args.eval_episodes,
        seed=args.eval_seed
    )
    
    eval_capacity = eval_metrics['mean_capacity']
    print(f"\n模型 {model_id + 1} 评估容量: {eval_capacity:.2f} ± {eval_metrics['std_capacity']:.2f}")
    
    return agent, run_dir, eval_capacity


def main():
    """多起点训练主函数"""
    parser = argparse.ArgumentParser(
        description='Multi-start DRL training with improved exploration',
        parents=[parse_args()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
多起点训练说明：
  1. 训练多个模型，每个使用不同的随机种子和初始化
  2. 每个模型都使用改进的探索策略（自适应熵 + 噪声注入）
  3. 最后选择评估容量最高的模型作为最佳模型

示例：
  python experiments/train_drl_multi_start.py \\
      --num_models 3 \\
      --select_best \\
      --num_episodes 5000 \\
      --SNR_dB 25.0 \\
      --A_lambda 3.0
        """
    )
    
    # 多起点训练参数
    parser.add_argument('--num_models', type=int, default=3,
                        help='Number of models to train (default: 3)')
    parser.add_argument('--select_best', action='store_true',
                        help='Select and save best model after training')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed (each model uses base_seed + model_id * 1000)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("多起点训练 + 改进探索策略")
    print("="*70)
    print(f"\n配置:")
    print(f"  模型数量: {args.num_models}")
    print(f"  每个模型训练: {args.num_episodes} episodes")
    print(f"  基础随机种子: {args.base_seed}")
    print(f"  系统参数: N={args.N}, M={args.M}, Lt={args.Lt}, Lr={args.Lr}")
    print(f"  SNR={args.SNR_dB}dB, A/λ={args.A_lambda}")
    print(f"\n改进的探索策略:")
    print(f"  ✓ 自适应熵系数（早期高探索，后期高利用）")
    print(f"  ✓ 动作噪声注入（早期高噪声，后期无噪声）")
    print(f"  ✓ 多起点初始化（每个模型不同的随机种子）")
    
    # 训练所有模型
    models = []
    eval_capacities = []
    
    for model_id in range(args.num_models):
        agent, run_dir, eval_capacity = train_single_model(
            model_id=model_id,
            base_args=args,
            base_seed=args.base_seed,
        )
        models.append({
            'agent': agent,
            'run_dir': run_dir,
            'model_id': model_id,
            'eval_capacity': eval_capacity,
        })
        eval_capacities.append(eval_capacity)
    
    # 选择最佳模型
    if args.select_best:
        best_idx = np.argmax(eval_capacities)
        best_model = models[best_idx]
        
        print(f"\n{'='*70}")
        print("模型选择结果")
        print(f"{'='*70}")
        print(f"\n所有模型的评估容量:")
        for i, (model, capacity) in enumerate(zip(models, eval_capacities)):
            marker = " ← 最佳" if i == best_idx else ""
            print(f"  模型 {i+1}: {capacity:.2f} bps/Hz{marker}")
        
        print(f"\n✓ 选择模型 {best_idx + 1} 作为最佳模型")
        print(f"  评估容量: {best_model['eval_capacity']:.2f} bps/Hz")
        print(f"  模型路径: {best_model['run_dir']}")
        
        # 保存最佳模型信息
        best_info_path = os.path.join(
            os.path.dirname(best_model['run_dir']),
            'best_model_info.json'
        )
        best_info = {
            'best_model_id': best_idx + 1,
            'best_eval_capacity': float(best_model['eval_capacity']),
            'best_model_path': os.path.join(best_model['run_dir'], 'best_model.pth'),
            'all_capacities': [float(c) for c in eval_capacities],
            'training_config': vars(args),
        }
        
        with open(best_info_path, 'w') as f:
            json.dump(best_info, f, indent=4)
        
        print(f"\n最佳模型信息已保存到: {best_info_path}")
        print(f"最佳模型路径: {os.path.join(best_model['run_dir'], 'best_model.pth')}")
    
    print(f"\n{'='*70}")
    print("多起点训练完成！")
    print(f"{'='*70}")
    print(f"\n所有模型目录:")
    for model in models:
        print(f"  模型 {model['model_id']+1}: {model['run_dir']}")
    
    return models, eval_capacities


if __name__ == "__main__":
    models, capacities = main()

