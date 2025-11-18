"""
学术论文风格的结果绘图脚本（更新版）
支持三子图：(a) Achievable rate, (b) Channel total power, (c) Condition number
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob

def setup_paper_style():
    """配置IEEE/学术论文风格的matplotlib参数"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        'axes.linewidth': 1.0,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'text.usetex': False,
        'axes.formatter.use_mathtext': True,
        'mathtext.fontset': 'stix',
    })

def plot_three_metrics(data_dict, output_path='results/paper_style_three_metrics.png'):
    """
    绘制论文风格的三子图：容量、功率、条件数
    
    参数:
        data_dict: 包含三个指标数据的字典
            {
                'label': {
                    'A_lambda': [...],
                    'rate': [...],
                    'power': [...],
                    'condition': [...]
                }
            }
    """
    setup_paper_style()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12))
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    markers = ['o', 's', '^', 'v']
    linestyles = ['-', '--', '-.', ':']
    
    plot_index = 0
    for label, data in data_dict.items():
        A_lambda = data['A_lambda']
        rate = data['rate']
        power = data['power']
        condition = data['condition']
        
        # (a) Achievable rate
        ax1.plot(A_lambda, rate,
                color=colors[plot_index % len(colors)],
                marker=markers[plot_index % len(markers)],
                linestyle=linestyles[plot_index % len(linestyles)],
                linewidth=1.8, markersize=7,
                markerfacecolor='white', markeredgewidth=1.5,
                label=label, zorder=3)
        
        # (b) Channel total power
        ax2.plot(A_lambda, power,
                color=colors[plot_index % len(colors)],
                marker=markers[plot_index % len(markers)],
                linestyle=linestyles[plot_index % len(linestyles)],
                linewidth=1.8, markersize=7,
                markerfacecolor='white', markeredgewidth=1.5,
                label=label, zorder=3)
        
        # (c) Condition number (对数刻度)
        ax3.semilogy(A_lambda, condition,
                    color=colors[plot_index % len(colors)],
                    marker=markers[plot_index % len(markers)],
                    linestyle=linestyles[plot_index % len(linestyles)],
                    linewidth=1.8, markersize=7,
                    markerfacecolor='white', markeredgewidth=1.5,
                    label=label, zorder=3)
        
        plot_index += 1
    
    # 设置子图(a)
    ax1.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax1.set_ylabel('Achievable rate (bps/Hz)', fontsize=11)
    ax1.set_title('(a) Achievable rate versus normalized region size', 
                  fontsize=12, loc='left', pad=10)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    ax1.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax1.tick_params(direction='in', which='both', top=True, right=True)
    
    # 设置子图(b)
    ax2.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax2.set_ylabel('Channel total power', fontsize=11)
    ax2.set_title('(b) Channel total power versus normalized region size',
                  fontsize=12, loc='left', pad=10)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    ax2.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax2.tick_params(direction='in', which='both', top=True, right=True)
    
    # 设置子图(c)
    ax3.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax3.set_ylabel('Condition number', fontsize=11)
    ax3.set_title('(c) Channel condition number versus normalized region size',
                  fontsize=12, loc='left', pad=10)
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0, which='both')
    ax3.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax3.tick_params(direction='in', which='both', top=True, right=True)
    
    # 自动设置x轴范围
    x_min = min([min(data['A_lambda']) for data in data_dict.values()])
    x_max = max([max(data['A_lambda']) for data in data_dict.values()])
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 三子图已保存到: {output_path}")
    plt.show()

def load_latest_results():
    """加载最新的实验结果"""
    results_dir = 'results'
    
    exact_files = sorted(glob(os.path.join(results_dir, 'exact_reproduction_*.json')))
    optimized_files = sorted(glob(os.path.join(results_dir, 'optimized_*.json')))
    
    data_dict = {}
    
    # 加载exact_reproduction结果
    if exact_files:
        latest_exact = exact_files[-1]
        with open(latest_exact, 'r', encoding='utf-8') as f:
            exact_data = json.load(f)
        
        A_lambda = exact_data['parameters']['A_lambda_values']
        
        # Lr=15
        data_dict['Proposed, Lr=15'] = {
            'A_lambda': A_lambda,
            'rate': exact_data['results']['achievable_rate']['Lr_15'],
            'power': exact_data['results']['channel_total_power']['Lr_15'],
            'condition': exact_data['results']['condition_number']['Lr_15']
        }
        
        # Lr=10
        data_dict['Proposed, Lr=10'] = {
            'A_lambda': A_lambda,
            'rate': exact_data['results']['achievable_rate']['Lr_10'],
            'power': exact_data['results']['channel_total_power']['Lr_10'],
            'condition': exact_data['results']['condition_number']['Lr_10']
        }
        
        print(f"✓ 已加载: {os.path.basename(latest_exact)}")
    
    # 加载optimized结果（如果有）
    if optimized_files:
        latest_opt = optimized_files[-1]
        with open(latest_opt, 'r', encoding='utf-8') as f:
            opt_data = json.load(f)
        
        A_lambda = opt_data['parameters']['A_lambda_values']
        
        # 添加优化版的数据（如果需要对比）
        # data_dict['Optimized, Lr=15'] = {...}
        
        print(f"✓ 已加载: {os.path.basename(latest_opt)}")
    
    return data_dict

def main():
    """主函数"""
    print("\n" + "="*70)
    print("学术论文风格绘图工具（三子图版）")
    print("="*70 + "\n")
    
    print("加载实验结果...")
    data_dict = load_latest_results()
    
    if not data_dict:
        print("❌ 未找到实验结果文件！")
        print("   请先运行 mimo_exact_reproduction.py 或 mimo_optimized.py")
        return
    
    print(f"\n共加载 {len(data_dict)} 条曲线")
    
    print("\n生成论文风格三子图...")
    plot_three_metrics(data_dict, 'results/paper_style_three_metrics.png')
    
    print("\n" + "="*70)
    print("绘图完成！")
    print("="*70)
    
    print("\n生成的文件:")
    print("  - results/paper_style_three_metrics.png (三子图，300 DPI)")
    
    print("\n图表包含:")
    print("  (a) Achievable rate - 可达速率")
    print("  (b) Channel total power - 信道总功率")
    print("  (c) Channel condition number - 信道条件数（对数刻度）")
    
    print("\n可用于:")
    print("  ✓ 论文投稿 (IEEE, Elsevier, Springer等)")
    print("  ✓ 学位论文")
    print("  ✓ 学术报告")

if __name__ == "__main__":
    main()
