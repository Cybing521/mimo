"""
快速对比脚本：在相同配置下比较AO和DRL的容量
==============================================

用法示例：
    python experiments/quick_compare.py \
        --drl_model results/drl_training/run_20251129_185802/best_model.pth \
        --A_lambda 3.0 \
        --SNR_dB 25.0 \
        --trials 10
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mimo_core import MIMOSystem
from drl.env import MAMIMOEnv
from drl.agent import PPOAgent
from experiments.compare_methods import MethodComparator
import torch


def get_device(device_preference: str = 'cpu') -> str:
    """智能设备选择"""
    if device_preference == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    elif device_preference == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_preference == 'mps':
        return 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
    else:
        return 'cpu'


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='快速对比AO和DRL在相同配置下的容量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认参数（A_lambda=3.0, SNR_dB=25.0）
  python experiments/quick_compare.py \\
      --drl_model results/drl_training/run_20251129_185802/best_model.pth

  # 指定A_lambda和SNR
  python experiments/quick_compare.py \\
      --drl_model results/drl_training/run_20251129_185802/best_model.pth \\
      --A_lambda 3.0 \\
      --SNR_dB 15.0 \\
      --trials 20
        """
    )
    
    # 必需参数
    parser.add_argument('--drl_model', type=str, required=True,
                       help='训练好的DRL模型路径')
    
    # 系统参数（必须与训练时一致！）
    parser.add_argument('--N', type=int, default=4, help='发射天线数')
    parser.add_argument('--M', type=int, default=4, help='接收天线数')
    parser.add_argument('--Lt', type=int, default=5, help='发射端路径数')
    parser.add_argument('--Lr', type=int, default=5, help='接收端路径数')
    parser.add_argument('--SNR_dB', type=float, default=25.0,
                       help='信噪比(dB) - 必须与训练时一致！')
    parser.add_argument('--A_lambda', type=float, default=3.0,
                       help='归一化区域大小 - 必须与训练时一致！')
    
    # 实验参数
    parser.add_argument('--trials', type=int, default=10,
                       help='随机试验次数（用于统计）')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='DRL每个episode的最大步数')
    
    # 设备
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='计算设备')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("快速对比：AO vs DRL")
    print("="*70)
    print(f"\n配置参数：")
    print(f"  N={args.N}, M={args.M}, Lt={args.Lt}, Lr={args.Lr}")
    print(f"  SNR={args.SNR_dB}dB, A/λ={args.A_lambda}")
    print(f"  试验次数: {args.trials}")
    print(f"  DRL模型: {args.drl_model}")
    print()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.drl_model):
        print(f"❌ 错误：DRL模型文件不存在: {args.drl_model}")
        return
    
    # 智能设备选择
    actual_device = get_device(args.device)
    if actual_device != args.device:
        print(f"设备选择: {args.device} -> {actual_device}")
    else:
        print(f"使用设备: {actual_device}")
    
    # 创建对比器
    comparator = MethodComparator(
        drl_model_path=args.drl_model,
        device=actual_device,
    )
    
    # 运行对比
    ao_capacities = []
    drl_capacities = []
    ao_times = []
    drl_times = []
    
    print("运行对比试验...")
    for trial in range(args.trials):
        # 设置随机种子（确保每次试验使用不同的信道）
        np.random.seed(trial * 1000)
        
        # 运行AO
        ao_result = comparator.run_ma_algorithm(
            N=args.N, M=args.M, Lt=args.Lt, Lr=args.Lr,
            SNR_dB=args.SNR_dB, A_lambda=args.A_lambda,
            mode='Proposed'
        )
        ao_capacities.append(ao_result['capacity'])
        ao_times.append(ao_result['time'])
        
        # 运行DRL
        drl_result = comparator.run_drl(
            N=args.N, M=args.M, Lt=args.Lt, Lr=args.Lr,
            SNR_dB=args.SNR_dB, A_lambda=args.A_lambda,
            max_steps=args.max_steps,
        )
        drl_capacities.append(drl_result['capacity'])
        drl_times.append(drl_result['time'])
        
        if (trial + 1) % 5 == 0:
            print(f"  完成 {trial+1}/{args.trials} 次试验...")
    
    # 统计结果
    ao_mean = np.mean(ao_capacities)
    ao_std = np.std(ao_capacities)
    drl_mean = np.mean(drl_capacities)
    drl_std = np.std(drl_capacities)
    
    ao_time_mean = np.mean(ao_times)
    drl_time_mean = np.mean(drl_times)
    
    gap = ao_mean - drl_mean
    gap_percent = (gap / ao_mean) * 100 if ao_mean > 0 else 0
    
    # 打印结果
    print("\n" + "="*70)
    print("对比结果")
    print("="*70)
    print(f"\n{'方法':<15} {'平均容量(bps/Hz)':<20} {'标准差':<15} {'平均时间(s)':<15}")
    print("-"*70)
    print(f"{'AO (迭代算法)':<15} {ao_mean:>8.2f} ± {ao_std:<8.2f} {ao_time_mean:>12.4f}")
    print(f"{'DRL':<15} {drl_mean:>8.2f} ± {drl_std:<8.2f} {drl_time_mean:>12.4f}")
    print("-"*70)
    print(f"\n容量差距: {gap:.2f} bps/Hz ({gap_percent:.1f}%)")
    print(f"速度提升: {ao_time_mean/drl_time_mean:.1f}x (DRL更快)")
    
    # 详细统计
    print(f"\n详细统计：")
    print(f"  AO容量范围: [{np.min(ao_capacities):.2f}, {np.max(ao_capacities):.2f}]")
    print(f"  DRL容量范围: [{np.min(drl_capacities):.2f}, {np.max(drl_capacities):.2f}]")
    
    # 建议
    print(f"\n💡 建议：")
    if gap > 5:
        print(f"  - DRL容量明显低于AO，可能需要：")
        print(f"    1. 增加训练episodes（当前可能未完全收敛）")
        print(f"    2. 增加max_steps（当前{args.max_steps}步可能不够）")
        print(f"    3. 检查奖励函数权重是否合适")
        print(f"    4. 确认训练时的SNR和A_lambda与当前测试一致")
    elif gap > 2:
        print(f"  - DRL容量略低于AO，但差距在可接受范围内")
        print(f"  - 可以考虑增加训练时间或微调超参数")
    else:
        print(f"  - DRL性能接近AO，表现良好！")
    
    # 检查配置一致性
    print(f"\n⚠️  配置检查：")
    print(f"  - 请确认训练时的SNR_dB={args.SNR_dB}和A_lambda={args.A_lambda}")
    print(f"  - 如果训练时使用不同参数，请相应调整测试参数")


if __name__ == "__main__":
    args = parse_args()
    main()

