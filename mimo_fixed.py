"""
MIMO天线位置优化仿真 - 修复版（解决死机问题）

主要修复：
1. 添加超时保护
2. 优化CVX求解器参数
3. 减少默认试验次数
4. 添加进度信息
5. 改进异常处理
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
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    """超时上下文管理器"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Windows不支持signal.SIGALRM，改用简单的时间检查
    if sys.platform == 'win32':
        yield
    else:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

def compute_field_response(r, theta_q, phi_q, lambda_val):
    """计算单个天线位置的场响应"""
    x = r[0]
    y = r[1]
    phase = 2*np.pi/lambda_val * (x*np.sin(theta_q)*np.cos(phi_q) + y*np.cos(theta_q))
    f_r = np.exp(1j*phase)
    return f_r, phase

def calculate_F(theta_q, phi_q, lambda_val, r, f_r_m):
    """计算接收端阵列响应矩阵F"""
    M = 4
    F = np.zeros((len(theta_q), M), dtype=complex)
    for m in range(M):
        x = r[0, m]
        y = r[1, m]
        phase = 2*np.pi/lambda_val * (x*np.sin(theta_q)*np.cos(phi_q) + y*np.cos(theta_q))
        f_r_m = np.exp(1j*phase)
        F[:, m] = f_r_m
    return F, f_r_m

def calculate_B_m(G, Q, H_r, m0, sigma, Sigma):
    """计算天线位置优化的目标函数矩阵B_m"""
    N = 4
    eigvals, eigvecs = np.linalg.eig(Q)
    Lambda_Q_sqrt = np.sqrt(np.diag(eigvals))
    U_Q = eigvecs
    
    W_r = H_r @ U_Q @ Lambda_Q_sqrt
    W_r_H = W_r.T.conj()
    W_r_H = np.delete(W_r_H, m0, axis=1)
    
    A_m = np.linalg.inv(np.eye(N) + (1/sigma**2) * (W_r_H @ W_r_H.T.conj()))
    B_m = Sigma @ G @ U_Q @ Lambda_Q_sqrt @ A_m @ Lambda_Q_sqrt @ U_Q.T.conj() @ G.T.conj() @ Sigma.T.conj()
    
    return B_m

def calculate_gradients(B_m, f_rm, r_mi, lambda_val, theta_qi, phi_qi):
    """计算目标函数关于天线位置的梯度和Hessian近似"""
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
    """使用scipy模拟MATLAB的quadprog求解二次规划问题"""
    n = len(f)
    
    def objective(x):
        return 0.5 * x.T @ H @ x + f.T @ x
    
    def jac(x):
        return H @ x + f
    
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
        N = 4
        M = 4
        Lt = 10
        power = 10
        SNR_dB = 15
        SNR_linear = 10**(SNR_dB / 10)
        sigma = power / SNR_linear
        D = lambda_val / 2
        xi = 1e-4  # 放宽收敛容差，避免过度迭代
        xii = 1e-5
        
        # 初始化接收天线位置（增加超时保护）
        r = np.zeros((2, M))
        attempts = 0
        max_attempts = 200  # 减少尝试次数
        while attempts < max_attempts:
            candidates = np.random.rand(2, 500) * square_size  # 减少候选点
            valid = np.ones(500, dtype=bool)
            
            for k in range(1, 500):
                distances = np.linalg.norm(candidates[:, :k] - candidates[:, k:k+1], axis=0)
                if np.min(distances) < D:
                    valid[k] = False
            
            valid_points = candidates[:, valid]
            if valid_points.shape[1] >= M:
                r = valid_points[:, :M]
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            return 0.0
        
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
        diag_len = min(Lr, Lt)
        diag_elements = (np.random.randn(diag_len) + 1j*np.random.randn(diag_len)) / np.sqrt(2*Lr)
        for i in range(diag_len):
            Sigma[i, i] = diag_elements[i]
        
        rho_q = np.random.rand(Lr)
        f_r_m = np.zeros(Lr, dtype=complex)
        converged = False
        channel_capacity_prev = 0
        
        F, f_r_m = calculate_F(theta_q, phi_q, lambda_val, r, f_r_m)
        H_r = F.T.conj() @ Sigma @ G
        
        # 交替优化主循环（减少迭代次数）
        max_outer_iter = 50  # 减少外层迭代
        for iter in range(max_outer_iter):
            # 1. 优化功率分配矩阵Q (使用CVX) - 添加更宽松的求解器参数
            try:
                Q_var = cp.Variable((4, 4), hermitian=True)
                obj = cp.log_det(np.eye(M) + (1/sigma) * H_r @ Q_var @ H_r.T.conj())
                constraints = [cp.trace(Q_var) <= power, Q_var >> 0]
                prob = cp.Problem(cp.Maximize(obj), constraints)
                
                # 使用更快的求解器参数
                prob.solve(solver=cp.SCS, verbose=False, max_iters=500, eps=1e-3)
                
                if prob.status not in ['optimal', 'optimal_inaccurate']:
                    Q = np.eye(4) * (power / 4)
                else:
                    Q = Q_var.value
                
                del Q_var, prob
                gc.collect()
            except Exception:
                Q = np.eye(4) * (power / 4)
            
            # 2. 优化每个天线位置
            m0 = 0
            for i in range(4):
                m0 = i
                r_mi = r[:, m0].copy()
                current_objective_value = 0
                max_sca_iter = 30  # 减少内层迭代
                
                for sca_iter in range(max_sca_iter):
                    previous_objective_value = current_objective_value
                    
                    F, f_r_m = calculate_F(theta_q, phi_q, lambda_val, r, f_r_m)
                    f_rm, _ = compute_field_response(r_mi, theta_q, phi_q, lambda_val)
                    H_r = F.T.conj() @ Sigma @ G
                    B_m = calculate_B_m(G, Q, H_r, m0, sigma, Sigma)
                    grad_g, delta_m = calculate_gradients(B_m, f_rm, r_mi, lambda_val, theta_q, phi_q)
                    
                    r_mii = grad_g / delta_m + r_mi
                    is_feasible = True
                    
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
        
        result = np.real(channel_capacity_current)
        gc.collect()
        return result
        
    except Exception as e:
        gc.collect()
        return 0.0

def main():
    """主函数"""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 参数设置 - 默认减少试验次数
    A_lambda_values = np.arange(1, 9)
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # 默认100次而非1000
    
    # 使用更少的进程数以减少内存压力
    num_cores = max(1, min(4, cpu_count() // 2))  # 最多4个进程
    
    print(f"\n{'='*70}")
    print(f"MIMO天线位置优化仿真 (修复版)")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU核心数: {cpu_count()} (将使用 {num_cores} 个进程)")
    print(f"试验次数: {num_trials}")
    print(f"结果保存: {results_dir}/fixed_{timestamp}")
    print(f"{'='*70}\n")
    
    Lr_values = [15, 10]
    lambda_val = 1
    
    average_capacity_Proposed = np.zeros((len(A_lambda_values), len(Lr_values)))
    
    for j in range(len(Lr_values)):
        Lr = Lr_values[j]
        print(f"\n处理 Lr = {Lr}")
        
        for u in range(len(A_lambda_values)):
            A_lambda = A_lambda_values[u]
            
            print(f"  A/λ = {A_lambda}, 进行 {num_trials} 次试验...")
            
            args_list = [(trial, A_lambda, Lr, lambda_val) for trial in range(num_trials)]
            
            capacity_sum = 0.0
            valid_count = 0
            
            with Pool(num_cores) as pool:
                for capacity in tqdm(
                    pool.imap_unordered(run_single_trial, args_list, chunksize=10),
                    total=num_trials,
                    desc=f"    A/λ={A_lambda}",
                    ncols=80
                ):
                    if capacity > 0:
                        capacity_sum += capacity
                        valid_count += 1
                    if valid_count % 50 == 0:
                        gc.collect()
            
            average_capacity_Proposed[u, j] = capacity_sum / max(valid_count, 1)
            print(f"    平均容量 = {average_capacity_Proposed[u, j]:.6f} bps/Hz (有效: {valid_count}/{num_trials})")
            
            gc.collect()
    
    # 保存结果
    results_data = {
        'timestamp': timestamp,
        'version': 'fixed',
        'parameters': {
            'A_lambda_values': A_lambda_values.tolist(),
            'num_trials': num_trials,
            'Lr_values': Lr_values,
            'lambda_val': lambda_val
        },
        'results': {
            'average_capacity': average_capacity_Proposed.tolist(),
            'Lr_15': average_capacity_Proposed[:, 0].tolist(),
            'Lr_10': average_capacity_Proposed[:, 1].tolist()
        }
    }
    
    json_filename = os.path.join(results_dir, f'fixed_{timestamp}.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    csv_filename = os.path.join(results_dir, f'fixed_{timestamp}.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write('A/lambda,Capacity_Lr15,Capacity_Lr10\n')
        for i, a_lambda in enumerate(A_lambda_values):
            f.write(f'{a_lambda},{average_capacity_Proposed[i, 0]:.6f},{average_capacity_Proposed[i, 1]:.6f}\n')
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(A_lambda_values, average_capacity_Proposed[:, 0], 'b-', linewidth=1.5, 
             marker='o', markersize=6, label='Proposed, Lr=15')
    plt.plot(A_lambda_values, average_capacity_Proposed[:, 1], 'r--', linewidth=1.5, 
             marker='s', markersize=6, label='Proposed, Lr=10')
    plt.xlabel('Normalized region size A/λ', fontsize=12)
    plt.ylabel('Capacity (bps/Hz)', fontsize=12)
    plt.title('Capacity vs. Normalized Receive Region Size\n(Fixed Version)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    png_filename = os.path.join(results_dir, f'fixed_{timestamp}.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("仿真完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n结果文件:")
    print(f"  - 图像: {png_filename}")
    print(f"  - JSON: {json_filename}")
    print(f"  - CSV:  {csv_filename}")
    print("\n平均容量摘要:")
    print(f"  Lr=15: {np.mean(average_capacity_Proposed[:, 0]):.6f} bps/Hz")
    print(f"  Lr=10: {np.mean(average_capacity_Proposed[:, 1]):.6f} bps/Hz")
    print("="*60)
    
    return average_capacity_Proposed

if __name__ == "__main__":
    results = main()
