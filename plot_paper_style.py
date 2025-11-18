"""
学术论文风格的结果绘图脚本
仿照IEEE Transactions等顶级期刊的图表样式
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob

# 设置学术论文风格
def setup_paper_style():
    """配置IEEE/学术论文风格的matplotlib参数"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        
        # 线条设置
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        'axes.linewidth': 1.0,
        
        # 网格设置
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # 图形设置
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # LaTeX风格
        'text.usetex': False,  # 如果安装了LaTeX可以设为True
        'axes.formatter.use_mathtext': True,
        'mathtext.fontset': 'stix',
    })

def plot_capacity_comparison(data_dict, output_path='results/paper_style_plot.png'):
    """
    绘制论文风格的容量对比图
    
    参数:
        data_dict: 包含多个实验结果的字典
        output_path: 输出图片路径
    """
    setup_paper_style()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 定义论文风格的颜色和标记
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'v', 'D']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # 绘制每条曲线
    plot_index = 0
    for label, data in data_dict.items():
        A_lambda = data['A_lambda']
        capacity = data['capacity']
        
        ax.plot(A_lambda, capacity,
                color=colors[plot_index % len(colors)],
                marker=markers[plot_index % len(markers)],
                linestyle=linestyles[plot_index % len(linestyles)],
                linewidth=1.8,
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=1.5,
                label=label,
                zorder=3)
        
        plot_index += 1
    
    # 设置坐标轴
    ax.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax.set_ylabel('Average capacity (bps/Hz)', fontsize=11)
    ax.set_title('Capacity vs. Normalized Receive Region Size', fontsize=12, pad=10)
    
    # 设置网格
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    
    # 设置图例
    ax.legend(loc='lower right', 
             frameon=True,
             fancybox=False,
             edgecolor='black',
             framealpha=0.9,
             columnspacing=0.8,
             handlelength=2.5)
    
    # 设置刻度
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.set_xticks(np.arange(1, 9))
    
    # 设置坐标轴范围
    ax.set_xlim(0.8, 8.2)
    y_min = min([min(data['capacity']) for data in data_dict.values()])
    y_max = max([max(data['capacity']) for data in data_dict.values()])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 论文风格图表已保存到: {output_path}")
    
    plt.show()

def plot_dual_subplot(data_dict, success_rates=None, output_path='results/paper_style_dual.png'):
    """
    绘制双子图：(a) 容量对比 (b) 成功率对比
    
    参数:
        data_dict: 容量数据字典
        success_rates: 成功率数据字典（可选）
        output_path: 输出路径
    """
    setup_paper_style()
    
    if success_rates is None:
        # 单图模式
        plot_capacity_comparison(data_dict, output_path)
        return
    
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    markers = ['o', 's', '^', 'v']
    linestyles = ['-', '--', '-.', ':']
    
    # (a) 容量对比
    plot_index = 0
    for label, data in data_dict.items():
        ax1.plot(data['A_lambda'], data['capacity'],
                color=colors[plot_index % len(colors)],
                marker=markers[plot_index % len(markers)],
                linestyle=linestyles[plot_index % len(linestyles)],
                linewidth=1.8,
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=1.5,
                label=label,
                zorder=3)
        plot_index += 1
    
    ax1.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax1.set_ylabel('Average capacity (bps/Hz)', fontsize=11)
    ax1.set_title('(a) Capacity Performance', fontsize=12, loc='left', pad=10)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    ax1.legend(loc='lower right', frameon=True, fancybox=False, 
              edgecolor='black', framealpha=0.9)
    ax1.tick_params(direction='in', which='both', top=True, right=True)
    ax1.set_xticks(np.arange(1, 9))
    ax1.set_xlim(0.8, 8.2)
    
    # (b) 成功率对比
    plot_index = 0
    for label, data in success_rates.items():
        ax2.plot(data['A_lambda'], np.array(data['success_rate']) * 100,
                color=colors[plot_index % len(colors)],
                marker=markers[plot_index % len(markers)],
                linestyle=linestyles[plot_index % len(linestyles)],
                linewidth=1.8,
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=1.5,
                label=label,
                zorder=3)
        plot_index += 1
    
    ax2.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax2.set_ylabel('Success rate (%)', fontsize=11)
    ax2.set_title('(b) Optimization Success Rate', fontsize=12, loc='left', pad=10)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    ax2.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax2.tick_params(direction='in', which='both', top=True, right=True)
    ax2.set_xticks(np.arange(1, 9))
    ax2.set_xlim(0.8, 8.2)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 双子图已保存到: {output_path}")
    
    plt.show()

def load_latest_results():
    """加载最新的实验结果"""
    results_dir = 'results'
    
    # 查找最新的JSON文件
    exact_files = sorted(glob(os.path.join(results_dir, 'exact_reproduction_*.json')))
    optimized_files = sorted(glob(os.path.join(results_dir, 'optimized_*.json')))
    
    data_dict = {}
    success_rates = {}
    
    # 加载exact_reproduction结果
    if exact_files:
        latest_exact = exact_files[-1]
        with open(latest_exact, 'r', encoding='utf-8') as f:
            exact_data = json.load(f)
        
        A_lambda = exact_data['parameters']['A_lambda_values']
        
        # Lr=15
        data_dict['Proposed, Lr=15'] = {
            'A_lambda': A_lambda,
            'capacity': exact_data['results']['Lr_15']
        }
        
        # Lr=10
        data_dict['Proposed, Lr=10'] = {
            'A_lambda': A_lambda,
            'capacity': exact_data['results']['Lr_10']
        }
        
        print(f"✓ 已加载: {os.path.basename(latest_exact)}")
    
    # 加载optimized结果（如果有）
    if optimized_files:
        latest_opt = optimized_files[-1]
        with open(latest_opt, 'r', encoding='utf-8') as f:
            opt_data = json.load(f)
        
        A_lambda = opt_data['parameters']['A_lambda_values']
        
        # 检查是否有成功率数据
        if 'success_rates' in opt_data['results']:
            success_rates['Optimized, Lr=15'] = {
                'A_lambda': A_lambda,
                'success_rate': opt_data['results']['success_rates_Lr15']
            }
            success_rates['Optimized, Lr=10'] = {
                'A_lambda': A_lambda,
                'success_rate': opt_data['results']['success_rates_Lr10']
            }
        
        print(f"✓ 已加载: {os.path.basename(latest_opt)}")
    
    return data_dict, success_rates if success_rates else None

def main():
    """主函数"""
    print("\n" + "="*70)
    print("学术论文风格绘图工具")
    print("="*70 + "\n")
    
    # 加载数据
    print("加载实验结果...")
    data_dict, success_rates = load_latest_results()
    
    if not data_dict:
        print("❌ 未找到实验结果文件！")
        print("   请先运行 mimo_exact_reproduction.py 或 mimo_optimized.py")
        return
    
    print(f"\n共加载 {len(data_dict)} 条曲线")
    
    # 生成图表
    print("\n生成论文风格图表...")
    
    if success_rates:
        # 有成功率数据，生成双子图
        plot_dual_subplot(data_dict, success_rates, 
                         'results/paper_style_dual.png')
    else:
        # 只有容量数据，生成单图
        plot_capacity_comparison(data_dict, 
                                'results/paper_style_single.png')
    
    print("\n" + "="*70)
    print("绘图完成！")
    print("="*70)
    
    # 显示文件信息
    print("\n生成的文件:")
    if success_rates:
        print("  - results/paper_style_dual.png (双子图，300 DPI)")
    else:
        print("  - results/paper_style_single.png (单图，300 DPI)")
    
    print("\n可用于:")
    print("  ✓ 论文投稿 (IEEE, Elsevier, Springer等)")
    print("  ✓ 学位论文")
    print("  ✓ 学术报告")

if __name__ == "__main__":
    main()
