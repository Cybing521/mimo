"""
MIMO天线位置优化仿真 - 完全按照MATLAB逻辑复现
此版本完全复现MATLAB代码的逻辑，包括其存在的问题（详见mimo_optimized.py）
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def compute_field_response(r, theta_q, phi_q, lambda_val):
    """
    计算单个天线位置的场响应
    
    参数:
        r: 天线位置 (2,)
        theta_q: 散射体仰角 (Lr,)
        phi_q: 散射体方向角 (Lr,)
        lambda_val: 波长
    
    返回:
        f_r: 场响应向量 (Lr,)
        phase: 相位 (Lr,)
    """
    x = r[0]
    y = r[1]
    phase = 2*np.pi/lambda_val * (x*np.sin(theta_q)*np.cos(phi_q) + y*np.cos(theta_q))
    f_r = np.exp(1j*phase)
    return f_r, phase

def calculate_F(theta_q, phi_q, lambda_val, r, f_r_m):
    """
    计算接收端阵列响应矩阵F
    
    注意: 此处复现MATLAB的逻辑，f_r_m在每次循环中被覆盖而不是累积
    """
    M = 4
    F = np.zeros((len(theta_q), M), dtype=complex)
    for m in range(M):
        x = r[0, m]
        y = r[1, m]
        phase = 2*np.pi/lambda_val * (x*np.sin(theta_q)*np.cos(phi_q) + y*np.cos(theta_q))
        f_r_m = np.exp(1j*phase)  # 注意：这里覆盖了f_r_m，而不是使用索引
        F[:, m] = f_r_m
    return F, f_r_m

def calculate_B_m(G, Q, H_r, m0, sigma, Sigma):
    """
    计算天线位置优化的目标函数矩阵B_m
    """
    N = 4
    # Q的特征分解
    eigvals, eigvecs = np.linalg.eig(Q)
    Lambda_Q_sqrt = np.sqrt(np.diag(eigvals))
    U_Q = eigvecs
    
    W_r = H_r @ U_Q @ Lambda_Q_sqrt
    W_r_H = W_r.T.conj()
    
    # 删除第m0列（MATLAB索引从1开始，Python从0开始）
    W_r_H = np.delete(W_r_H, m0, axis=1)
    
    A_m = np.linalg.inv(np.eye(N) + (1/sigma**2) * (W_r_H @ W_r_H.T.conj()))
    B_m = Sigma @ G @ U_Q @ Lambda_Q_sqrt @ A_m @ Lambda_Q_sqrt @ U_Q.T.conj() @ G.T.conj() @ Sigma.T.conj()
    
    return B_m

def calculate_gradients(B_m, f_rm, r_mi, lambda_val, theta_qi, phi_qi):
    """
    计算目标函数关于天线位置的梯度和Hessian近似
    """
    b = B_m @ f_rm
    amplitude_bq = np.abs(b)
    phase_bq = np.angle(b)
    
    x = r_mi[0]
    y = r_mi[1]
    rho_qi = x*np.sin(theta_qi)*np.cos(phi_qi) + y*np.cos(theta_qi)
    kappa = (2*np.pi/lambda_val)*rho_qi - phase_bq
    
    term_x = amplitude_bq * np.sin(theta_qi) * np.cos(phi_qi) * np.sin(kappa)
    term_y = amplitude_bq * np.cos(theta_qi) * np.sin(kappa)
    
    grad_x = -(2*np.pi/lambda_val) * np.sum(term_x)
    grad_y = -(2*np.pi/lambda_val) * np.sum(term_y)
    grad_g = np.array([grad_x, grad_y])
    
    sum_abs_b = np.sum(np.abs(b))
    delta_m = (8*np.pi**2) / (lambda_val**2) * sum_abs_b
    
    return grad_g, delta_m

def quadprog_solve(H, f, A, b, lb, ub, x0):
    """
    使用scipy模拟MATLAB的quadprog求解二次规划问题
    min 0.5 * x^T H x + f^T x
    s.t. A x <= b, lb <= x <= ub
    """
    n = len(f)
    
    def objective(x):
        return 0.5 * x.T @ H @ x + f.T @ x
    
    def jac(x):
        return H @ x + f
    
    # 约束
    constraints = []
    if A is not None and len(A) > 0:
        for i in range(len(A)):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: b[i] - A[i] @ x})
    
    bounds = [(lb[i], ub[i]) for i in range(n)]
    
    result = minimize(objective, x0, method='SLSQP', jac=jac, 
                     bounds=bounds, constraints=constraints,
                     options={'ftol': 1e-8, 'disp': False, 'maxiter': 100})
    
    if result.success:
        return result.x
    else:
        return None

def run_single_trial(args):
    """运行单次试验（用于多进程）"""
    trial, A_lambda, Lr, lambda_val = args
    
    square_size = A_lambda * lambda_val
    
    try:
        # 系统参数
        N = 4  # 发送天线
        M = 4  # 接收天线
        Lt = 5
        power = 10
        SNR_dB = 5
        SNR_linear = 10**(SNR_dB / 10)
        sigma = power / SNR_linear
        D = lambda_val / 2  # 最小天线距离
        xi = 1e-3
        xii = 1e-3
        
        # 初始化接收天线位置
        r = np.zeros((2, M))
        attempts = 0
        while True:
            candidates = np.random.rand(2, 1000) * square_size
            valid = np.ones(1000, dtype=bool)
            
            for k in range(1, 1000):
                distances = np.linalg.norm(candidates[:, :k] - candidates[:, k:k+1], axis=0)
                if np.min(distances) < D:
                    valid[k] = False
            
            valid_points = candidates[:, valid]
            if valid_points.shape[1] >= M:
                r = valid_points[:, :M]
                break
            
            attempts += 1
            if attempts > 200:
                raise RuntimeError('无法初始化合法位置')
        
        # 初始化信道参数
        P = Lt
        theta_p = np.random.rand(P) * np.pi
        phi_p = np.random.rand(P) * np.pi
        
        G = np.zeros((P, N), dtype=complex)
        for p in range(P):
            for n in range(N):
                G[p, n] = np.exp(1j * np.pi * np.sin(theta_p[p]) * np.cos(phi_p[p]) * n)
        
        theta_q = np.random.rand(Lr) * np.pi
        phi_q = np.random.rand(Lr) * np.pi
        
        Sigma = np.zeros((Lr, Lt), dtype=complex)
        kappa = 1
        Sigma[0, 0] = (np.random.randn() + 1j * np.random.randn()) * np.sqrt(kappa / (kappa + 1) / 2)
        for i in range(1, min(Lr, Lt)):
            Sigma[i, i] = (np.random.randn() + 1j * np.random.randn()) * np.sqrt(1 / ((kappa + 1) * (Lr - 1)) / 2)
        
        rho_q = np.random.rand(Lr)
        f_r_m = np.zeros(Lr, dtype=complex)
        converged = False
        channel_capacity_prev = 0
        
        F, f_r_m = calculate_F(theta_q, phi_q, lambda_val, r, f_r_m)
        H_r = F.T.conj() @ Sigma @ G
        
        # 交替优化主循环
        for iter in range(50):  # 外循环迭代次数
            # 1. 优化功率分配矩阵Q (使用CVX)
            Q_var = cp.Variable((4, 4), hermitian=True)
            obj = cp.log_det(np.eye(M) + (1/sigma) * H_r @ Q_var @ H_r.T.conj())
            constraints = [cp.trace(Q_var) <= power, Q_var >> 0]
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=500, eps=1e-3)
            
            if prob.status not in ['optimal', 'optimal_inaccurate']:
                Q = np.eye(4) * (power / 4)
            else:
                Q = Q_var.value
            
            # 清理CVX缓存
            del Q_var, prob
            gc.collect()
            
            # 2. 优化每个天线位置
            m0 = 0
            for i in range(4):
                m0 = i  # 对应MATLAB的m0 = m0 + 1后的值
                r_mi = r[:, m0].copy()
                current_objective_value = 0
                max_sca_iter = 30  # 内循环迭代次数
                
                for sca_iter in range(max_sca_iter):
                    previous_objective_value = current_objective_value
                    
                    F, f_r_m = calculate_F(theta_q, phi_q, lambda_val, r, f_r_m)
                    f_rm, _ = compute_field_response(r_mi, theta_q, phi_q, lambda_val)
                    H_r = F.T.conj() @ Sigma @ G
                    B_m = calculate_B_m(G, Q, H_r, m0, sigma, Sigma)
                    grad_g, delta_m = calculate_gradients(B_m, f_rm, r_mi, lambda_val, theta_q, phi_q)
                    
                    r_mii = grad_g / delta_m + r_mi
                    is_feasible = True
                    
                    # 检查可行性
                    if np.any(r_mii < 0) or np.any(r_mii > square_size):
                        is_feasible = False
                    else:
                        for k in range(M):
                            if k != m0:
                                distance = np.linalg.norm(r_mii - r[:, k])
                                if distance < D:
                                    is_feasible = False
                                    break
                    
                    if is_feasible:
                        r_mi = r_mii
                    else:
                        # 使用二次规划
                        H_qp = delta_m * np.eye(2)
                        f_qp = -(grad_g + delta_m * r_mi)
                        
                        A_list = []
                        b_list = []
                        for k in range(M):
                            if k != m0:
                                r_k = r[:, k]
                                norm_diff = np.linalg.norm(r_mi - r_k)
                                if norm_diff > 1e-10:
                                    a_k = (r_mi - r_k) / norm_diff
                                    b_row = -D - a_k.T @ r_k
                                    A_list.append(-a_k)
                                    b_list.append(b_row)
                        
                        lb = np.zeros(2)
                        ub = square_size * np.ones(2)
                        
                        if len(A_list) > 0:
                            A = np.array(A_list)
                            b1 = np.array(b_list)
                        else:
                            A = None
                            b1 = None
                        
                        r_mnew = quadprog_solve(H_qp, f_qp, A, b1, lb, ub, r_mi)
                        
                        if r_mnew is None:
                            r_mnew = r_mi
                        
                        r_mi = r_mnew
                    
                    r[:, m0] = r_mi
                    
                    F, f_r_m = calculate_F(theta_q, phi_q, lambda_val, r, f_r_m)
                    H_r = F.T.conj() @ Sigma @ G
                    B_m = calculate_B_m(G, Q, H_r, m0, sigma, Sigma)
                    f_r_new, _ = compute_field_response(r_mi, theta_q, phi_q, lambda_val)
                    current_objective_value = np.real(f_r_new.T.conj() @ B_m @ f_r_new)
                    
                    if abs(current_objective_value - previous_objective_value) < xii:
                        break
                
                r[:, m0] = r_mi
            
            # 计算当前信道容量
            F, f_r_m = calculate_F(theta_q, phi_q, lambda_val, r, f_r_m)
            H_r = F.T.conj() @ Sigma @ G
            H_rQH = H_r @ Q @ H_r.T.conj()
            channel_capacity_current = np.log2(np.linalg.det(np.eye(M) + (1/sigma) * H_rQH))
            
            # 检查收敛
            if abs(channel_capacity_current - channel_capacity_prev) < xi:
                break
            
            channel_capacity_prev = channel_capacity_current
        
        # 计算三个性能指标（对应论文Fig. 6）
        achievable_rate = np.real(channel_capacity_current)
        
        # 信道总功率 = trace(H @ Q @ H^H)
        channel_total_power = np.real(np.trace(H_rQH))
        
        # 信道条件数 = 最大奇异值 / 最小奇异值
        singular_values = np.linalg.svd(H_r, compute_uv=False)
        condition_number = singular_values[0] / (singular_values[-1] + 1e-10)
        
        # 清理内存
        gc.collect()
        return achievable_rate, channel_total_power, condition_number
        
    except Exception as e:
        # 如果试验失败，返回0
        gc.collect()
        return 0.0, 0.0, 0.0

def main():
    """主函数"""
    # 创建results目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 参数设置（参照论文Fig. 6）
    A_lambda_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    num_trials = 100 if len(sys.argv) <= 1 else int(sys.argv[1])
    Lr_values = [5]
    
    # 检测CPU核心数（使用一半，最多4个以避免内存溢出）
    num_cores = max(1, min(4, cpu_count() // 2))
    
    print(f"\n{'='*70}")
    print(f"MIMO天线位置优化仿真 (精确复现版 - 多进程加速)")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU核心数: {cpu_count()} (将使用 {num_cores} 个进程并行计算以避免内存溢出)")
    print(f"试验次数: {num_trials}")
    print(f"结果保存: {results_dir}/exact_reproduction_{timestamp}")
    print(f"{'='*70}\n")
    lambda_val = 1
    
    # 初始化结果存储（三个指标）
    average_capacity = np.zeros((len(A_lambda_values), len(Lr_values)))
    average_total_power = np.zeros((len(A_lambda_values), len(Lr_values)))
    average_condition_number = np.zeros((len(A_lambda_values), len(Lr_values)))
    
    # 主循环
    for j in range(len(Lr_values)):
        Lr = Lr_values[j]
        print(f"\n处理 Lr = {Lr}")
        
        for u in range(len(A_lambda_values)):
            A_lambda = A_lambda_values[u]
            
            print(f"  A/λ = {A_lambda}, 进行 {num_trials} 次试验（多进程并行）...")
            
            # 准备参数列表
            args_list = [(trial, A_lambda, Lr, lambda_val) for trial in range(num_trials)]
            
            # 使用多进程池，增量处理结果避免内存爆炸
            capacity_sum = 0.0
            power_sum = 0.0
            cond_sum = 0.0
            valid_count = 0
            
            with Pool(num_cores) as pool:
                # 使用imap_unordered并增量处理，不在内存中保存所有结果
                for result in tqdm(
                    pool.imap_unordered(run_single_trial, args_list, chunksize=20),
                    total=num_trials,
                    desc=f"    A/λ={A_lambda}",
                    ncols=80
                ):
                    capacity, power, cond = result
                    if capacity > 0:
                        capacity_sum += capacity
                        power_sum += power
                        cond_sum += cond
                        valid_count += 1
                    # 定期清理内存
                    if valid_count % 100 == 0:
                        gc.collect()
            
            # 计算平均值
            average_capacity[u, j] = capacity_sum / max(valid_count, 1)
            average_total_power[u, j] = power_sum / max(valid_count, 1)
            average_condition_number[u, j] = cond_sum / max(valid_count, 1)
            
            print(f"    平均容量 = {average_capacity[u, j]:.6f} bps/Hz (有效: {valid_count}/{num_trials})")
            
            # 强制垃圾回收
            gc.collect()
    
    # 保存数值结果（三个指标）
    results_data = {
        'timestamp': timestamp,
        'version': 'exact_reproduction',
        'parameters': {
            'N': 4,
            'M': 4,
            'Lt': 5,
            'A_lambda_values': A_lambda_values,
            'num_trials': num_trials,
            'Lr_values': Lr_values,
            'lambda_val': lambda_val
        },
        'results': {
            'achievable_rate': {
                'Lr_5': average_capacity[:, 0].tolist()
            },
            'channel_total_power': {
                'Lr_5': average_total_power[:, 0].tolist()
            },
            'condition_number': {
                'Lr_5': average_condition_number[:, 0].tolist()
            }
        }
    }
    
    # 保存JSON格式结果
    json_filename = os.path.join(results_dir, f'exact_reproduction_{timestamp}.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式结果（包含三个指标）
    csv_filename = os.path.join(results_dir, f'exact_reproduction_{timestamp}.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write('A/lambda,Rate_Lr5,Power_Lr5,Cond_Lr5\n')
        for i, a_lambda in enumerate(A_lambda_values):
            f.write(f'{a_lambda},'
                   f'{average_capacity[i, 0]:.6f},{average_total_power[i, 0]:.6f},{average_condition_number[i, 0]:.2f}\n')
    
    # 绘图 - 学术论文风格
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    # 创建三子图（对应论文Fig. 6）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12))
    
    # (a) Achievable rate
    ax1.plot(A_lambda_values, average_capacity[:, 0], 
            color='#1f77b4', linestyle='-', marker='o',
            linewidth=1.8, markersize=7,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Proposed, Lr=5', zorder=3)
    ax1.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax1.set_ylabel('Achievable rate (bps/Hz)', fontsize=11)
    ax1.set_title('(a) Achievable rate versus normalized region size', fontsize=12, loc='left', pad=10)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    ax1.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax1.tick_params(direction='in', which='both', top=True, right=True)
    ax1.set_xlim(0.9, 4.1)
    
    # (b) Channel total power
    ax2.plot(A_lambda_values, average_total_power[:, 0],
            color='#1f77b4', linestyle='-', marker='o',
            linewidth=1.8, markersize=7,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Proposed, Lr=5', zorder=3)
    ax2.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax2.set_ylabel('Channel total power', fontsize=11)
    ax2.set_title('(b) Channel total power versus normalized region size', fontsize=12, loc='left', pad=10)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    ax2.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax2.tick_params(direction='in', which='both', top=True, right=True)
    ax2.set_xlim(0.9, 4.1)
    
    # (c) Channel condition number (对数刻度)
    ax3.semilogy(A_lambda_values, average_condition_number[:, 0],
                color='#1f77b4', linestyle='-', marker='o',
                linewidth=1.8, markersize=7,
                markerfacecolor='white', markeredgewidth=1.5,
                label='Proposed, Lr=5', zorder=3)
    ax3.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax3.set_ylabel('Condition number', fontsize=11)
    ax3.set_title('(c) Channel condition number versus normalized region size', fontsize=12, loc='left', pad=10)
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0, which='both')
    ax3.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax3.tick_params(direction='in', which='both', top=True, right=True)
    ax3.set_xlim(0.9, 4.1)
    
    # 保存图像
    plt.tight_layout()
    png_filename = os.path.join(results_dir, f'exact_reproduction_{timestamp}.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印摘要
    print("\n" + "="*60)
    print("仿真完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n结果文件:")
    print(f"  - 图像: {png_filename}")
    print(f"  - JSON: {json_filename}")
    print(f"  - CSV:  {csv_filename}")
    print("\n性能指标摘要:")
    print(f"  Lr=5 (SNR=5dB): 容量={np.mean(average_capacity[:, 0]):.6f} bps/Hz, "
          f"功率={np.mean(average_total_power[:, 0]):.2f}, "
          f"条件数={np.mean(average_condition_number[:, 0]):.2f}")
    print("="*60)
    
    return average_capacity, average_total_power, average_condition_number

if __name__ == "__main__":
    results = main()
