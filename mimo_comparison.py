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
        results = mimo.run_optimization(A_lambda, mode=mode)
        return results
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description='MIMO Antenna Position Optimization Simulation')
    parser.add_argument('--snr', type=float, default=25, help='Signal-to-Noise Ratio (dB)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Monte Carlo trials')
    parser.add_argument('--cores', type=int, default=None, help='Number of CPU cores to use')
    parser.add_argument('--modes', nargs='+', default=['Proposed', 'RMA', 'TMA', 'FPA'], 
                      help='Modes to simulate (default: all)')
    args = parser.parse_args()
    
    # ... (参数设置保持不变) ...
    num_trials = args.trials
    # 如果用户没有指定 cores，则使用默认逻辑（最多8核）
    num_cores = args.cores if args.cores is not None else min(8, cpu_count())
    SNR_dB = args.snr

    
    # 自动调整 A_lambda_values
    if SNR_dB < 0:
        A_lambda_values = np.arange(1, 3.1, 0.2)
        print(f"Configuration: Low-SNR Regime (SNR={SNR_dB}dB)")
    else:
        A_lambda_values = np.arange(1, 4.1, 0.2)
        print(f"Configuration: High-SNR Regime (SNR={SNR_dB}dB)")

    Lt, Lr = 5, 5
    modes = args.modes # 使用用户指定的 modes
    
    print(f"Parameters: A/λ={A_lambda_values}, Lt={Lt}, Lr={Lr}, Trials={num_trials}, Cores={num_cores}")
    print(f"Modes: {modes}")

    # 数据存储结构更新
    metrics = ['capacity', 'strongest_eigen_power', 'total_power', 'cond_number']
    results = {mode: {m: [] for m in metrics} for mode in modes}
    
    # 运行仿真
    for mode in modes:
        print(f"\n正在运行模式: {mode}")
        
        for A_lambda in A_lambda_values:
            args_list = [(t, A_lambda, SNR_dB, mode, Lt, Lr) for t in range(num_trials)]
            
            with Pool(processes=num_cores) as pool:
                # 获取所有试验的结果列表
                batch_results = list(tqdm(
                    pool.imap(run_simulation_mode, args_list),
                    total=num_trials,
                    desc=f"  A/λ={A_lambda}",
                    ncols=80
                ))
            
            # 过滤失败的试验 (None)
            valid_results = [res for res in batch_results if res is not None]
            
            if valid_results:
                # 计算各项指标的平均值
                avg_capacity = np.mean([r['capacity'] for r in valid_results])
                avg_eigen = np.mean([r['strongest_eigen_power'] for r in valid_results])
                avg_total = np.mean([r['total_power'] for r in valid_results])
                avg_cond = np.mean([r['cond_number'] for r in valid_results])
                
                results[mode]['capacity'].append(avg_capacity)
                results[mode]['strongest_eigen_power'].append(avg_eigen)
                results[mode]['total_power'].append(avg_total)
                results[mode]['cond_number'].append(avg_cond)
                
                print(f"  平均容量: {avg_capacity:.4f} bps/Hz")
            else:
                # 全军覆没的情况
                for m in metrics: results[mode][m].append(0.0)
                print(f"  全部试验失败")

    # 绘图配置
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
    
    styles = {
        'Proposed': {'marker': 'o', 'color': '#d62728', 'label': 'Proposed (Tx+Rx)'},
        'RMA': {'marker': 's', 'color': '#1f77b4', 'label': 'RMA (Rx only)'},
        'TMA': {'marker': '^', 'color': '#2ca02c', 'label': 'TMA (Tx only)'},
        'FPA': {'marker': 'x', 'color': '#7f7f7f', 'label': 'FPA (Fixed)'}
    }

    # 保存时间戳
    if not os.path.exists('results'): os.makedirs('results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 绘制 3 张图 (Fig 5/6 a, b, c)
    plot_configs = [
        ('capacity', 'Achievable rate (bps/Hz)', f'Achievable Rate (SNR={SNR_dB}dB)'),
        ('strongest_eigen_power', 'Strongest eigenchannel power', f'Strongest Eigenchannel Power (SNR={SNR_dB}dB)'),
        ('total_power', 'Channel total power', f'Channel Total Power (SNR={SNR_dB}dB)')
    ]

    for metric_key, ylabel, title in plot_configs:
        plt.figure(figsize=(8, 6))
        for mode in modes:
            if mode in styles:
                plt.plot(A_lambda_values, results[mode][metric_key], 
                         marker=styles[mode]['marker'],
                         color=styles[mode]['color'],
                         label=styles[mode]['label'],
                         linestyle='-', markerfacecolor='none', markeredgewidth=1.5)
        
        plt.xlabel(r'Normalized region size ($A/\lambda$)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right' if metric_key == 'capacity' else 'best')
        plt.tight_layout()
        
        filename = f'results/{metric_key}_{timestamp}.png'
        plt.savefig(filename, dpi=300)
        print(f"保存图表: {filename}")


if __name__ == '__main__':
    import os
    main()
