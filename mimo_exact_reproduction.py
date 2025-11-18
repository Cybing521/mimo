"""
MIMO天线位置优化仿真 - 完全按照MATLAB逻辑复现
此版本完全复现MATLAB代码的逻辑，包括其存在的问题（详见mimo_optimized.py）
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
                     options={'ftol': 1e-8, 'disp': False})
    
    if result.success:
        return result.x
    else:
        return None

def main():
    """主函数"""
    # 参数设置
    A_lambda_values = np.arange(1, 9)  # 1:1:8
    num_trials = 1000
    Lr_values = [15, 10]
    lambda_val = 1
    
    # 初始化结果存储
    average_capacity_Proposed = np.zeros((len(A_lambda_values), len(Lr_values)))
    
    # 主循环
    for j in range(len(Lr_values)):
        Lr = Lr_values[j]
        print(f"\n处理 Lr = {Lr}")
        
        for u in range(len(A_lambda_values)):
            A_lambda = A_lambda_values[u]
            square_size = A_lambda * lambda_val
            capacity_trials = np.zeros(num_trials)
            
            print(f"  A/λ = {A_lambda}, 进行 {num_trials} 次试验...")
            
            for trial in range(num_trials):
                if (trial + 1) % 100 == 0:
                    print(f"    试验 {trial + 1}/{num_trials}")
                
                # 系统参数
                N = 4  # 发送天线
                M = 4  # 接收天线
                Lt = 10
                power = 10
                SNR_dB = 15
                SNR_linear = 10**(SNR_dB / 10)
                sigma = power / SNR_linear
                lambda_val = 1
                D = lambda_val / 2  # 最小天线距离
                xi = 1e-3  # 外循环收敛容差
                xii = 1e-4  # 内循环收敛容差
                
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
                    if attempts > 400:
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
                
                # 交替优化主循环
                for iter in range(50):
                    # 1. 优化功率分配矩阵Q (使用CVX)
                    Q_var = cp.Variable((4, 4), hermitian=True)
                    obj = cp.log_det(np.eye(M) + (1/sigma) * H_r @ Q_var @ H_r.T.conj())
                    constraints = [cp.trace(Q_var) <= power, Q_var >> 0]
                    prob = cp.Problem(cp.Maximize(obj), constraints)
                    prob.solve(solver=cp.SCS, verbose=False)
                    
                    if prob.status not in ['optimal', 'optimal_inaccurate']:
                        print(f"      警告: CVX求解失败，状态={prob.status}")
                        Q = np.eye(4) * (power / 4)
                    else:
                        Q = Q_var.value
                    
                    # 2. 优化每个天线位置
                    m0 = 0
                    for i in range(4):
                        m0 = i  # 对应MATLAB的m0 = m0 + 1后的值
                        r_mi = r[:, m0].copy()
                        current_objective_value = 0
                        max_sca_iter = 60
                        
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
                                    # print(f"        警告: SCA优化失败，保留原位置")
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
                
                capacity_trials[trial] = np.real(channel_capacity_current)
            
            # 计算平均容量
            average_capacity_Proposed[u, j] = np.mean(capacity_trials)
            print(f"    平均容量 = {average_capacity_Proposed[u, j]:.4f} bps/Hz")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(A_lambda_values, average_capacity_Proposed[:, 0], 'b-', linewidth=1.5, label='Proposed, Lr=15')
    plt.plot(A_lambda_values, average_capacity_Proposed[:, 1], 'r--', linewidth=1.5, label='Proposed, Lr=10')
    plt.xlabel('Normalized region size A/λ')
    plt.ylabel('Capacity (bps/Hz)')
    plt.title('Capacity vs. Normalized Receive Region Size for Proposed Method')
    plt.legend()
    plt.grid(True)
    plt.savefig('capacity_exact_reproduction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n完成！结果已保存到 capacity_exact_reproduction.png")
    return average_capacity_Proposed

if __name__ == "__main__":
    results = main()
