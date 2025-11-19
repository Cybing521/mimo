"""
MIMO天线位置优化仿真 - 多模式对比脚本
包含以下模式：
1. Proposed: 收发两端联合优化 (Algorithm 2)
2. RMA: 仅接收端优化 (Rx Movable Antenna)
3. TMA: 仅发送端优化 (Tx Movable Antenna)
4. FPA: 固定位置天线 (Fixed Position Antenna)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json

# 导入核心计算函数
from mimo_optimized import (
    calculate_F, calculate_G, calculate_B_m, calculate_D_n,
    calculate_S_matrix, calculate_gradients, calculate_gradients_transmit,
    compute_field_response, compute_transmit_field_response,
    initialize_antennas_smart, optimize_position_robust
)

def run_simulation_mode(args):
    """运行指定模式的单次试验"""
    trial, A_lambda, Lr, lambda_val, mode = args
    
    square_size = A_lambda * lambda_val
    
    try:
        # 系统参数（根据论文图6）
        N, M, Lt = 4, 4, 10
        power, SNR_dB = 10, 15  # 默认15dB，可根据需要修改
        SNR_linear = 10**(SNR_dB / 10)
        sigma = power / SNR_linear
        D = lambda_val / 2
        xi, xii = 1e-3, 1e-3
        
        # 初始化天线位置
        # FPA模式下，位置固定为初始随机位置（或网格中心）
        # 其他模式下，位置作为初始值
        r = initialize_antennas_smart(M, square_size, D)
        t = initialize_antennas_smart(N, square_size, D)
        
        # 信道参数初始化
        P = Lt
        theta_p = np.random.rand(P) * np.pi
        phi_p = np.random.rand(P) * np.pi
        theta_q = np.random.rand(Lr) * np.pi
        phi_q = np.random.rand(Lr) * np.pi
        
        # 路径响应矩阵Σ
        Sigma = np.zeros((Lr, Lt), dtype=complex)
        kappa = 1
        Sigma[0, 0] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(kappa / (kappa + 1) / 2)
        for i in range(1, min(Lr, Lt)):
            Sigma[i, i] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(1 / ((kappa + 1) * (Lr - 1)) / 2)
            
        channel_capacity_prev = 0
        
        # 优化循环
        max_outer_iter = 50 if mode != 'FPA' else 1  # FPA只需要一次功率分配
        
        for outer_iter in range(max_outer_iter):
            # 0. 计算当前信道矩阵
            F = calculate_F(theta_q, phi_q, lambda_val, r)
            G = calculate_G(theta_p, phi_p, lambda_val, t)
            H_r = F.T.conj() @ Sigma @ G
            
            # 1. 优化功率分配 (所有模式都需要)
            Q_var = cp.Variable((N, N), hermitian=True)
            obj = cp.log_det(np.eye(M) + (1/sigma) * H_r @ Q_var @ H_r.T.conj())
            constraints = [cp.trace(Q_var) <= power, Q_var >> 0]
            prob = cp.Problem(cp.Maximize(obj), constraints)
            
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=500, eps=1e-3)
                if prob.status in ['optimal', 'optimal_inaccurate']:
                    Q = Q_var.value
                else:
                    Q = np.eye(N) * (power / N)
            except:
                Q = np.eye(N) * (power / N)
                
            if mode == 'FPA':
                # FPA模式仅优化功率，不优化位置
                break
            
            # 2. 接收天线位置优化 (Proposed 和 RMA 模式)
            if mode in ['Proposed', 'RMA']:
                for antenna_idx in range(M):
                    r_mi = r[:, antenna_idx].copy()
                    prev_obj = 0
                    
                    for sca_iter in range(30):
                        F = calculate_F(theta_q, phi_q, lambda_val, r)
                        H_r = F.T.conj() @ Sigma @ G
                        B_m = calculate_B_m(G, Q, H_r, antenna_idx, sigma, Sigma)
                        
                        f_rm, _ = compute_field_response(r_mi, theta_q, phi_q, lambda_val)
                        grad_g, delta_m = calculate_gradients(B_m, f_rm, r_mi, lambda_val, theta_q, phi_q)
                        
                        r_new, success = optimize_position_robust(
                            r_mi, r, antenna_idx, M, D, square_size,
                            grad_g, delta_m, theta_q, phi_q, lambda_val, B_m
                        )
                        
                        r_mi = r_new
                        r[:, antenna_idx] = r_mi
                        
                        f_new, _ = compute_field_response(r_mi, theta_q, phi_q, lambda_val)
                        curr_obj = np.real(f_new.T.conj() @ B_m @ f_new)
                        
                        if abs(curr_obj - prev_obj) < xii * (abs(curr_obj) + 1e-6):
                            break
                        prev_obj = curr_obj

            # 3. 发送天线位置优化 (Proposed 和 TMA 模式)
            if mode in ['Proposed', 'TMA']:
                # 需要先计算S矩阵
                F = calculate_F(theta_q, phi_q, lambda_val, r)
                G = calculate_G(theta_p, phi_p, lambda_val, t)
                H_r = F.T.conj() @ Sigma @ G
                S_tuple = calculate_S_matrix(H_r, power, sigma)
                
                for antenna_idx in range(N):
                    t_ni = t[:, antenna_idx].copy()
                    prev_obj_t = 0
                    
                    for sca_iter in range(30):
                        G = calculate_G(theta_p, phi_p, lambda_val, t)
                        H_r = F.T.conj() @ Sigma @ G
                        D_n = calculate_D_n(F, S_tuple, G, antenna_idx, sigma, Sigma, power)
                        
                        g_tn, _ = compute_transmit_field_response(t_ni, theta_p, phi_p, lambda_val)
                        grad_g_t, delta_n = calculate_gradients_transmit(D_n, g_tn, t_ni, lambda_val, theta_p, phi_p)
                        
                        t_new, success = optimize_position_robust(
                            t_ni, t, antenna_idx, N, D, square_size,
                            grad_g_t, delta_n, theta_p, phi_p, lambda_val, D_n
                        )
                        
                        t_ni = t_new
                        t[:, antenna_idx] = t_ni
                        
                        g_new, _ = compute_transmit_field_response(t_ni, theta_p, phi_p, lambda_val)
                        curr_obj_t = np.real(g_new.T.conj() @ D_n @ g_new)
                        
                        if abs(curr_obj_t - prev_obj_t) < xii * (abs(curr_obj_t) + 1e-6):
                            break
                        prev_obj_t = curr_obj_t

            # 计算当前容量并检查收敛
            F = calculate_F(theta_q, phi_q, lambda_val, r)
            G = calculate_G(theta_p, phi_p, lambda_val, t)
            H_r = F.T.conj() @ Sigma @ G
            H_rQH = H_r @ Q @ H_r.T.conj()
            
            eigvals = np.linalg.eigvalsh(np.eye(M) + (1/sigma) * H_rQH)
            eigvals = np.maximum(eigvals, 1e-10)
            channel_capacity_current = np.sum(np.log2(eigvals))
            
            rel_change = abs(channel_capacity_current - channel_capacity_prev) / (abs(channel_capacity_current) + 1e-6)
            if rel_change < xi:
                break
            
            channel_capacity_prev = channel_capacity_current
            
        # 最终计算
        F = calculate_F(theta_q, phi_q, lambda_val, r)
        G = calculate_G(theta_p, phi_p, lambda_val, t)
        H_r = F.T.conj() @ Sigma @ G
        H_rQH = H_r @ Q @ H_r.T.conj()
        eigvals = np.linalg.eigvalsh(np.eye(M) + (1/sigma) * H_rQH)
        eigvals = np.maximum(eigvals, 1e-10)
        capacity = np.sum(np.log2(eigvals))
        
        return capacity

    except Exception as e:
        return 0.0

def main():
    print(f"\n{'='*70}")
    print(f"MIMO多模式对比仿真 (Proposed vs RMA vs TMA vs FPA)")
    print(f"{'='*70}")
    
    # 参数设置
    A_lambda_values = np.arange(1, 5, 0.5)  # 1.0 到 4.5
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    num_cores = min(8, cpu_count())
    
    modes = ['Proposed', 'RMA', 'TMA', 'FPA']
    results = {mode: [] for mode in modes}
    
    # 运行仿真
    for mode in modes:
        print(f"\n正在运行模式: {mode}")
        mode_capacities = []
        
        for A_lambda in A_lambda_values:
            args_list = [(t, A_lambda, 15, 1, mode) for t in range(num_trials)]
            
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
        'lines.markersize': 8
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
                 linestyle='-')
                 
    plt.xlabel(r'Normalized region size ($A/\lambda$)')
    plt.ylabel('Achievable rate (bps/Hz)')
    plt.title('Comparison of Different Antenna Configurations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'results/comparison_{timestamp}.png', dpi=300)
    print(f"\n结果已保存至 results/comparison_{timestamp}.png")

if __name__ == '__main__':
    main()
