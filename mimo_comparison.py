"""
MIMO天线位置优化 - 多模式对比脚本
包含以下模式：
1. Proposed: 收发两端联合优化 (Algorithm 2)
2. RMA: 仅接收端优化 (Rx Movable Antenna)
3. TMA: 仅发送端优化 (Tx Movable Antenna)
4. FPA: 固定位置天线 (Fixed Position Antenna)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from mimo_optimized import MIMOSystem

import argparse

def run_simulation_mode(args):
    """运行指定模式的单次试验"""
    trial, A_lambda, SNR_dB, mode, Lt, Lr = args
    
    # 初始化系统 (使用论文标准参数)
    mimo = MIMOSystem(N=4, M=4, Lt=Lt, Lr=Lr, SNR_dB=SNR_dB, lambda_val=1)
    
    try:
        # 运行优化
        capacity = mimo.run_optimization(A_lambda, mode=mode)
        return capacity
    except Exception as e:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='MIMO Antenna Position Optimization Simulation')
    parser.add_argument('--snr', type=float, default=25, help='Signal-to-Noise Ratio (dB)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Monte Carlo trials')
    parser.add_argument('--cores', type=int, default=None, help='Number of CPU cores to use (default: max available)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"MIMO多模式对比仿真 (Proposed vs RMA vs TMA vs FPA)")
    print(f"{'='*70}")
    
    # 参数设置
    num_trials = args.trials
    # 如果用户没有指定 cores，则使用默认逻辑（最多8核）
    num_cores = args.cores if args.cores is not None else min(8, cpu_count())
    SNR_dB = args.snr

    # 根据 SNR 自动调整扫描范围 (复现论文逻辑)
    # Low-SNR 通常扫描范围较小 (1-3)，High-SNR 扫描范围较大 (1-4)
    if SNR_dB < 0:
        A_lambda_values = np.arange(1, 3.1, 0.5)
        print(f"Configuration: Low-SNR Regime (SNR={SNR_dB}dB)")
    else:
        A_lambda_values = np.arange(1, 4.1, 0.5)
        print(f"Configuration: High-SNR Regime (SNR={SNR_dB}dB)")
    
    # 固定散射体参数 (论文标准)
    Lt, Lr = 5, 5
    print(f"Parameters: A/λ in {A_lambda_values}, Lt={Lt}, Lr={Lr}, Trials={num_trials}, Cores={num_cores}")

    modes = ['Proposed', 'RMA', 'TMA', 'FPA']
    results = {mode: [] for mode in modes}
    
    # 运行仿真
    for mode in modes:
        print(f"\n正在运行模式: {mode}")
        mode_capacities = []
        
        for A_lambda in A_lambda_values:
            args_list = [(t, A_lambda, SNR_dB, mode, Lt, Lr) for t in range(num_trials)]
            
            with Pool(processes=num_cores) as pool:
                capacities = list(tqdm(
                    pool.imap(run_simulation_mode, args_list),
                    total=num_trials,
                    desc=f"  A/λ={A_lambda}",
                    ncols=80
                ))
            
            # 过滤失败的试验
            valid_capacities = [c for c in capacities if c > 0]
            avg_capacity = np.mean(valid_capacities) if valid_capacities else 0
            mode_capacities.append(avg_capacity)
            print(f"  平均容量: {avg_capacity:.4f} bps/Hz")
            
        results[mode] = mode_capacities

    # 绘图 (学术风格)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    
    plt.figure(figsize=(8, 6))
    
    styles = {
        'Proposed': {'marker': 'o', 'color': '#d62728', 'label': 'Proposed (Tx+Rx)'},
        'RMA': {'marker': 's', 'color': '#1f77b4', 'label': 'RMA (Rx only)'},
        'TMA': {'marker': '^', 'color': '#2ca02c', 'label': 'TMA (Tx only)'},
        'FPA': {'marker': 'x', 'color': '#7f7f7f', 'label': 'FPA (Fixed)'}
    }
    
    for mode in modes:
        plt.plot(A_lambda_values, results[mode], 
                 marker=styles[mode]['marker'],
                 color=styles[mode]['color'],
                 label=styles[mode]['label'],
                 linestyle='-',
                 markerfacecolor='none',
                 markeredgewidth=1.5)
                 
    plt.xlabel(r'Normalized region size ($A/\lambda$)')
    plt.ylabel('Achievable rate (bps/Hz)')
    plt.title(f'Comparison of Different Antenna Configurations (SNR={SNR_dB}dB)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # 保存结果
    if not os.path.exists('results'):
        os.makedirs('results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'results/comparison_{timestamp}.png', dpi=300)
    print(f"\n结果已保存至 results/comparison_{timestamp}.png")

if __name__ == '__main__':
    import os
    main()
