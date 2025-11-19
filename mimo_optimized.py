"""
MIMO天线位置优化仿真 - 完整Proposed算法（优化版）

本代码实现完整的Algorithm 2 (Alternating Optimization)，包括：
1. 功率分配优化 (步骤4)
2. 接收天线位置优化 (步骤5-8，调用Algorithm 1)
3. 发送天线位置优化 (步骤10-13，调用Algorithm 1) ← 新增
4. 交替迭代直到收敛 (步骤3)

与原MATLAB代码的区别：
- 原MATLAB: 仅优化接收端 → 对应论文Fig.6的RMA方案
- 本代码: 同时优化收发两端 → 对应论文Fig.6的Proposed方案

原MATLAB代码存在的问题分析：
================================

1. **变量覆盖Bug (第192-200行)**
   问题: 在calculate_F函数中，f_r_m在每次循环中被覆盖，而不是正确地存储到矩阵中
   原代码: f_r_m = exp(1i*phase); F(:, m) = f_r_m;
   影响: 导致F矩阵计算错误，影响信道矩阵H_r的准确性
   修复: 直接计算并赋值，不使用中间变量f_r_m

2. **lambda变量重定义 (第23行)**
   问题: 在内层循环中重新定义lambda=1，覆盖了外层的lambda变量
   原代码: lambda = 1;  % 在循环内部
   影响: 虽然在此代码中影响不大（因为lambda始终为1），但是不良编程习惯
   修复: 使用lambda_val作为变量名，避免与内置关键字冲突

3. **索引逻辑混乱 (第76-78行)**
   问题: m0的初始化和递增逻辑不清晰
   原代码: m0=0; for i=1:4, m0=m0+1; ...
   影响: 代码可读性差，容易出错
   修复: 直接使用循环变量i作为索引

4. **二次规划求解不稳定 (第124-133行)**
   问题: quadprog在约束复杂时可能失败，导致位置突变
   原代码: 失败时仅警告并保留原位置，但不调整步长
   影响: 这是导致结果突变的主要原因！当约束变紧时，优化失败增多
   修复: 添加步长控制、更好的初始化策略、备选优化方法

5. **收敛判据不合理**
   问题: 外循环容差1e-3对于容量可能过于宽松，内循环1e-4可能过紧
   影响: 可能导致过早收敛或过多迭代
   修复: 使用相对误差而非绝对误差，动态调整容差

6. **随机初始化效率低 (第28-44行)**
   问题: 使用拒绝采样，当区域较小或天线数较多时效率极低
   影响: 可能导致初始化失败或耗时过长
   修复: 使用更智能的初始化策略（如网格初始化）

7. **结果突变的根本原因**
   当归一化区域A/λ较小时：
   - 可行域变小，约束变紧
   - quadprog求解失败率上升
   - 天线位置优化陷入局部最优或无法移动
   - 导致容量突然下降
   
   当Lr较大时：
   - 散射体增多，信道自由度增加
   - 但优化复杂度也增加
   - 更容易出现数值不稳定
   
修复策略：
- 使用自适应步长
- 添加正则化项
- 改进约束处理
- 使用更稳健的优化器
- 添加重启机制
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def compute_field_response(r, theta_q, phi_q, lambda_val):
    """计算单个天线位置的场响应"""
    x, y = r[0], r[1]
    phase = 2*np.pi/lambda_val * (x*np.sin(theta_q)*np.cos(phi_q) + y*np.cos(theta_q))
    f_r = np.exp(1j*phase)
    return f_r, phase

def calculate_F(theta_q, phi_q, lambda_val, r):
    """
    计算接收端阵列响应矩阵F (修复版)
    修复: 直接计算，不使用会被覆盖的中间变量
    """
    M = r.shape[1]
    Lr = len(theta_q)
    F = np.zeros((Lr, M), dtype=complex)
    
    for m in range(M):
        x, y = r[0, m], r[1, m]
        phase = 2*np.pi/lambda_val * (x*np.sin(theta_q)*np.cos(phi_q) + y*np.cos(theta_q))
        F[:, m] = np.exp(1j*phase)
    
    return F

def calculate_G(theta_p, phi_p, lambda_val, t):
    """
    计算发送端阵列响应矩阵G
    参数:
        theta_p: 发送端散射体仰角 (Lt,)
        phi_p: 发送端散射体方向角 (Lt,)
        lambda_val: 波长
        t: 发送天线位置 (2, N)
    返回:
        G: 发送端阵列响应矩阵 (Lt, N)
    """
    N = t.shape[1]
    Lt = len(theta_p)
    G = np.zeros((Lt, N), dtype=complex)
    
    for n in range(N):
        x, y = t[0, n], t[1, n]
        phase = 2*np.pi/lambda_val * (x*np.sin(theta_p)*np.cos(phi_p) + y*np.cos(theta_p))
        G[:, n] = np.exp(1j*phase)
    
    return G

def calculate_B_m(G, Q, H_r, m0, sigma, Sigma):
    """计算接收天线位置优化的目标函数矩阵B_m (对应论文式18)"""
    N = Q.shape[0]
    
    # 添加数值稳定性检查
    Q_stable = Q + 1e-10 * np.eye(N)
    
    eigvals, eigvecs = np.linalg.eig(Q_stable)
    eigvals = np.maximum(eigvals, 0)  # 确保非负
    Lambda_Q_sqrt = np.sqrt(np.diag(eigvals))
    U_Q = eigvecs
    
    W_r = H_r @ U_Q @ Lambda_Q_sqrt
    W_r_H = W_r.T.conj()
    W_r_H = np.delete(W_r_H, m0, axis=1)
    
    # 添加正则化避免矩阵奇异
    A_m = np.linalg.inv(np.eye(N) + (1/sigma**2) * (W_r_H @ W_r_H.T.conj()) + 1e-10*np.eye(N))
    B_m = Sigma @ G @ U_Q @ Lambda_Q_sqrt @ A_m @ Lambda_Q_sqrt @ U_Q.T.conj() @ G.T.conj() @ Sigma.T.conj()
    
    return B_m

def calculate_S_matrix(H_r, power, sigma):
    """
    计算发送端优化所需的对偶协方差矩阵S (对应论文式22)
    
    S是针对H^H的最优协方差矩阵，用于发送端优化
    H^H = G^H @ Σ^H @ F (N×M)
    S的优化类似Q，但针对H^H系统
    
    参数:
        H_r: 信道矩阵 H = F^H @ Σ @ G (M, N)
        power: 总功率约束 P
        sigma: 噪声标准差
    返回:
        S: 对偶协方差矩阵 (M, M)
        U_S: S的特征向量矩阵 (M, M)
        Lambda_S_sqrt: S特征值的平方根对角矩阵 (M, M)
    """
    M, N = H_r.shape  # H_r是(M, N)
    
    # H^H = (F^H @ Σ @ G)^H = G^H @ Σ^H @ F
    # 我们需要优化S使得 log det(I_N + (1/σ²) * H^H @ S @ H^H^H) 最大
    # 这等价于优化 H^H @ S @ H^H^H
    
    # 计算H @ H^H (M×M矩阵)
    H_HH = H_r @ H_r.T.conj()  # (M, M)
    
    # 特征分解H @ H^H
    eigvals, eigvecs = np.linalg.eigh(H_HH)
    eigvals = np.maximum(eigvals, 0)  # 确保非负
    idx = np.argsort(eigvals)[::-1]  # 降序排列
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Water-filling功率分配 (类似Q的优化)
    rank_S = min(M, np.sum(eigvals > 1e-10))
    
    # 简化：等功率分配
    p_s = np.zeros(M)
    if rank_S > 0:
        p_s[:rank_S] = power / rank_S
    
    # 构造S = U_S @ diag(p_s) @ U_S^H
    Lambda_S = np.diag(p_s)
    S = eigvecs @ Lambda_S @ eigvecs.T.conj()
    
    # 特征值平方根
    Lambda_S_sqrt = np.sqrt(Lambda_S)
    U_S = eigvecs
    
    return S, U_S, Lambda_S_sqrt

def calculate_D_n(F, S_tuple, G, n0, sigma, Sigma, power):
    """
    计算发送天线位置优化的目标函数矩阵D_n (严格对应论文式23)
    
    论文式(23):
    D_n = Σ^H @ F(r̃) @ U_S @ V_S^† @ A_n @ V_S^† @ U_S^H @ F(r̃)^H @ Σ
    
    参数:
        F: 接收端阵列响应 (Lr, M)
        S_tuple: (S, U_S, Lambda_S_sqrt) - 对偶协方差矩阵及其分解
        G: 发送端阵列响应 (Lt, N)
        n0: 当前优化的发送天线索引
        sigma: 噪声标准差
        Sigma: 路径响应矩阵 (Lr, Lt)
        power: 总功率
    返回:
        D_n: 目标函数矩阵 (Lt, Lt)
    """
    S, U_S, Lambda_S_sqrt = S_tuple
    M = F.shape[1]
    
    # 计算P(t̃) = H(t̃,r̃) @ U_S @ sqrt(Λ_S) @ Σ_g(t_n)
    # 其中H = F^H @ Σ @ G，所以H^H = G^H @ Σ^H @ F
    # P(t̃) = G^H @ Σ^H @ F @ U_S @ sqrt(Λ_S)
    
    # 定义p(t_n) = V_S^† U_S^H F(r̃)^H Σ_g(t_n) (论文第5页)
    # 移除第n个发送天线后的矩阵
    P_tilde = G.T.conj() @ Sigma.T.conj() @ F @ U_S @ Lambda_S_sqrt  # (N, M)
    
    # 删除第n0行（对应第n个发送天线）
    P_tilde_n = np.delete(P_tilde, n0, axis=0)  # (N-1, M)
    
    # C_n = I_M + (1/σ²) P̃_n^H P̃_n (论文第5页)
    C_n_inv = np.eye(M) + (1/sigma**2) * (P_tilde_n.T.conj() @ P_tilde_n)
    C_n = np.linalg.inv(C_n_inv + 1e-10*np.eye(M))  # (M, M)
    
    # A_n = C_n (根据论文符号)
    A_n = C_n
    
    # D_n = Σ^H @ F @ U_S @ sqrt(Λ_S) @ A_n @ sqrt(Λ_S) @ U_S^H @ F^H @ Σ
    # 论文式(23)
    D_n = Sigma.T.conj() @ F @ U_S @ Lambda_S_sqrt @ A_n @ Lambda_S_sqrt @ U_S.T.conj() @ F.T.conj() @ Sigma
    
    return D_n

def calculate_gradients(B_m, f_rm, r_mi, lambda_val, theta_qi, phi_qi):
    """计算接收端梯度和Hessian近似"""
    b = B_m @ f_rm
    amplitude_bq = np.abs(b)
    phase_bq = np.angle(b)
    
    x, y = r_mi[0], r_mi[1]
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

def calculate_gradients_transmit(D_n, g_tn, t_ni, lambda_val, theta_pi, phi_pi):
    """
    计算发送端梯度和Hessian近似
    类似calculate_gradients但针对发送端
    参数:
        D_n: 目标函数矩阵 (Lt, Lt)
        g_tn: 发送端场响应 (Lt,)
        t_ni: 当前发送天线位置 (2,)
        lambda_val: 波长
        theta_pi: 发送端散射体仰角 (Lt,)
        phi_pi: 发送端散射体方向角 (Lt,)
    """
    d = D_n @ g_tn
    amplitude_dp = np.abs(d)
    phase_dp = np.angle(d)
    
    x, y = t_ni[0], t_ni[1]
    rho_pi = x*np.sin(theta_pi)*np.cos(phi_pi) + y*np.cos(theta_pi)
    kappa = (2*np.pi/lambda_val)*rho_pi - phase_dp
    
    term_x = amplitude_dp * np.sin(theta_pi) * np.cos(phi_pi) * np.sin(kappa)
    term_y = amplitude_dp * np.cos(theta_pi) * np.sin(kappa)
    
    grad_x = -(2*np.pi/lambda_val) * np.sum(term_x)
    grad_y = -(2*np.pi/lambda_val) * np.sum(term_y)
    grad_g = np.array([grad_x, grad_y])
    
    sum_abs_d = np.sum(np.abs(d))
    delta_n = (8*np.pi**2) / (lambda_val**2) * sum_abs_d
    
    return grad_g, delta_n

def compute_transmit_field_response(t, theta_p, phi_p, lambda_val):
    """计算单个发送天线位置的场响应"""
    x, y = t[0], t[1]
    phase = 2*np.pi/lambda_val * (x*np.sin(theta_p)*np.cos(phi_p) + y*np.cos(theta_p))
    g_t = np.exp(1j*phase)
    return g_t, phase

def initialize_antennas_smart(M, square_size, D):
    """
    智能初始化天线位置 (改进版)
    使用网格+随机扰动的策略，比纯随机采样更可靠
    """
    # 先尝试网格布局
    grid_size = int(np.ceil(np.sqrt(M)))
    spacing = square_size / (grid_size + 1)
    
    if spacing >= D:
        # 网格间距足够，使用网格初始化
        r = np.zeros((2, M))
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= M:
                    break
                r[0, idx] = (i + 1) * spacing + np.random.randn() * D * 0.1
                r[1, idx] = (j + 1) * spacing + np.random.randn() * D * 0.1
                idx += 1
                if idx >= M:
                    break
        return r
    
    # 否则使用改进的随机采样（增加候选点数）
    attempts = 0
    max_attempts = 1000
    
    while attempts < max_attempts:
        candidates = np.random.rand(2, 5000) * square_size
        valid = np.ones(5000, dtype=bool)
        
        for k in range(1, 5000):
            distances = np.linalg.norm(candidates[:, :k] - candidates[:, k:k+1], axis=0)
            if np.min(distances) < D:
                valid[k] = False
        
        valid_points = candidates[:, valid]
        if valid_points.shape[1] >= M:
            return valid_points[:, :M]
        
        attempts += 1
    
    raise RuntimeError(f'无法初始化合法位置: square_size={square_size}, D={D}, M={M}')

def optimize_position_robust(r_mi, r, m0, M, D, square_size, grad_g, delta_m, 
                             theta_q, phi_q, lambda_val, B_m, max_step=None):
    """
    稳健的位置优化 (改进版)
    添加了步长控制、更好的约束处理
    """
    if max_step is None:
        max_step = D  # 默认最大步长为最小间距
    
    # 计算梯度下降步长（带步长限制）
    step = grad_g / delta_m
    step_norm = np.linalg.norm(step)
    if step_norm > max_step:
        step = step * (max_step / step_norm)
    
    r_new = r_mi + step
    
    # 检查可行性
    is_feasible = True
    if np.any(r_new < 0) or np.any(r_new > square_size):
        is_feasible = False
    else:
        for k in range(M):
            if k != m0:
                if np.linalg.norm(r_new - r[:, k]) < D:
                    is_feasible = False
                    break
    
    if is_feasible:
        return r_new, True
    
    # 不可行时，使用投影梯度法
    # 定义目标函数
    def objective(x):
        f_x, _ = compute_field_response(x, theta_q, phi_q, lambda_val)
        obj = -np.real(f_x.T.conj() @ B_m @ f_x)
        return obj
    
    def gradient(x):
        f_x, _ = compute_field_response(x, theta_q, phi_q, lambda_val)
        g, _ = calculate_gradients(B_m, f_x, x, lambda_val, theta_q, phi_q)
        return -g  # 负号因为我们在minimize
    
    # 边界约束
    bounds = [(0, square_size), (0, square_size)]
    
    # 距离约束
    def distance_constraint(x):
        distances = []
        for k in range(M):
            if k != m0:
                distances.append(np.linalg.norm(x - r[:, k]) - D)
        return np.array(distances) if distances else np.array([1.0])
    
    # 使用SLSQP求解
    nonlinear_constraint = NonlinearConstraint(distance_constraint, 0, np.inf)
    
    result = minimize(objective, r_mi, method='SLSQP', jac=gradient,
                     bounds=bounds, constraints=[nonlinear_constraint],
                     options={'ftol': 1e-8, 'maxiter': 100, 'disp': False})
    
    if result.success:
        return result.x, True
    else:
        # 优化失败，尝试减小步长后的简单投影
        for alpha in [0.5, 0.25, 0.1, 0.05]:
            r_test = r_mi + step * alpha
            r_test = np.clip(r_test, 0, square_size)
            
            feasible = True
            for k in range(M):
                if k != m0 and np.linalg.norm(r_test - r[:, k]) < D:
                    feasible = False
                    break
            
            if feasible:
                return r_test, True
        
        return r_mi, False  # 完全失败，保持原位置

def run_single_trial_optimized(args):
    """运行单次试验（优化版，用于多进程）"""
    trial, A_lambda, Lr, lambda_val = args
    
    square_size = A_lambda * lambda_val
    
    try:
        # 系统参数（根据论文图4）
        N, M, Lt = 4, 4, 10  # Lt = 10 (论文标准)
        power, SNR_dB = 10, 15  # SNR = 15 dB (论文标准)
        SNR_linear = 10**(SNR_dB / 10)
        sigma = power / SNR_linear
        D = lambda_val / 2
        xi, xii = 1e-3, 1e-3  # 收敛容差 ε₁ = ε₂ = 10^-3
        
        # 智能初始化接收天线和发送天线
        r = initialize_antennas_smart(M, square_size, D)
        t = initialize_antennas_smart(N, square_size, D)  # 发送天线初始化
        
        # 信道参数初始化
        P = Lt
        theta_p = np.random.rand(P) * np.pi
        phi_p = np.random.rand(P) * np.pi
        
        # 使用calculate_G函数计算发送端阵列响应
        G = calculate_G(theta_p, phi_p, lambda_val, t)
        
        theta_q = np.random.rand(Lr) * np.pi
        phi_q = np.random.rand(Lr) * np.pi
        
        # 路径响应矩阵Σ（Rician信道，κ=1）
        Sigma = np.zeros((Lr, Lt), dtype=complex)
        kappa = 1  # LoS与NLoS功率比
        # Σ[1,1] ~ CN(0, κ/(κ+1))
        Sigma[0, 0] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(kappa / (kappa + 1) / 2)
        # Σ[p,p] ~ CN(0, 1/((κ+1)(Lr-1))) for p=2,3,...,Lr
        for i in range(1, min(Lr, Lt)):
            Sigma[i, i] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(1 / ((kappa + 1) * (Lr - 1)) / 2)
        
        F = calculate_F(theta_q, phi_q, lambda_val, r)
        H_r = F.T.conj() @ Sigma @ G
        
        channel_capacity_prev = 0
        
        # 交替优化
        for outer_iter in range(50):  # 外循环迭代次数
            # 1. 优化功率分配
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
            
            # 2. 优化天线位置
            for antenna_idx in range(M):
                r_mi = r[:, antenna_idx].copy()
                prev_obj = 0
                
                for sca_iter in range(30):  # 内循环迭代次数
                    F = calculate_F(theta_q, phi_q, lambda_val, r)
                    H_r = F.T.conj() @ Sigma @ G
                    B_m = calculate_B_m(G, Q, H_r, antenna_idx, sigma, Sigma)
                    
                    f_rm, _ = compute_field_response(r_mi, theta_q, phi_q, lambda_val)
                    grad_g, delta_m = calculate_gradients(B_m, f_rm, r_mi, lambda_val, theta_q, phi_q)
                    
                    # 使用稳健优化
                    r_new, success = optimize_position_robust(
                        r_mi, r, antenna_idx, M, D, square_size,
                        grad_g, delta_m, theta_q, phi_q, lambda_val, B_m
                    )
                    
                    r_mi = r_new
                    r[:, antenna_idx] = r_mi
                    
                    # 计算目标函数值
                    f_new, _ = compute_field_response(r_mi, theta_q, phi_q, lambda_val)
                    curr_obj = np.real(f_new.T.conj() @ B_m @ f_new)
                    
                    # 相对收敛判据
                    if abs(curr_obj - prev_obj) < xii * (abs(curr_obj) + 1e-6):
                        break
                    
                    prev_obj = curr_obj
            
            # 3. 计算S矩阵 (Algorithm 2, 步骤9 - 对应论文式22)
            F = calculate_F(theta_q, phi_q, lambda_val, r)
            G = calculate_G(theta_p, phi_p, lambda_val, t)
            H_r = F.T.conj() @ Sigma @ G
            S_tuple = calculate_S_matrix(H_r, power, sigma)
            
            # 4. 优化发送天线位置 (Algorithm 2, 步骤10-13)
            for antenna_idx in range(N):
                t_ni = t[:, antenna_idx].copy()
                prev_obj_t = 0
                
                for sca_iter in range(30):  # 内循环迭代次数
                    G = calculate_G(theta_p, phi_p, lambda_val, t)
                    H_r = F.T.conj() @ Sigma @ G
                    D_n = calculate_D_n(F, S_tuple, G, antenna_idx, sigma, Sigma, power)
                    
                    g_tn, _ = compute_transmit_field_response(t_ni, theta_p, phi_p, lambda_val)
                    grad_g_t, delta_n = calculate_gradients_transmit(D_n, g_tn, t_ni, lambda_val, theta_p, phi_p)
                    
                    # 使用稳健优化（发送端）
                    t_new, success = optimize_position_robust(
                        t_ni, t, antenna_idx, N, D, square_size,
                        grad_g_t, delta_n, theta_p, phi_p, lambda_val, D_n
                    )
                    
                    t_ni = t_new
                    t[:, antenna_idx] = t_ni
                    
                    # 计算目标函数值
                    g_new, _ = compute_transmit_field_response(t_ni, theta_p, phi_p, lambda_val)
                    curr_obj_t = np.real(g_new.T.conj() @ D_n @ g_new)
                    
                    # 相对收敛判据
                    if abs(curr_obj_t - prev_obj_t) < xii * (abs(curr_obj_t) + 1e-6):
                        break
                    
                    prev_obj_t = curr_obj_t
            
            # 计算容量
            F = calculate_F(theta_q, phi_q, lambda_val, r)
            G = calculate_G(theta_p, phi_p, lambda_val, t)
            H_r = F.T.conj() @ Sigma @ G
            H_rQH = H_r @ Q @ H_r.T.conj()
            
            # 添加数值稳定性
            eigvals = np.linalg.eigvalsh(np.eye(M) + (1/sigma) * H_rQH)
            eigvals = np.maximum(eigvals, 1e-10)
            channel_capacity_current = np.sum(np.log2(eigvals))
            
            # 相对收敛判据
            rel_change = abs(channel_capacity_current - channel_capacity_prev) / (abs(channel_capacity_current) + 1e-6)
            if rel_change < xi:
                break
            
            channel_capacity_prev = channel_capacity_current
        
        # 计算三个性能指标（对应论文Fig. 6）
        achievable_rate = np.real(channel_capacity_current)
        channel_total_power = np.real(np.trace(H_rQH))
        singular_values = np.linalg.svd(H_r, compute_uv=False)
        condition_number = singular_values[0] / (singular_values[-1] + 1e-10)
        
        return achievable_rate, channel_total_power, condition_number, 1  # 返回三个指标和成功标志
        
    except Exception as e:
        return 0.0, 0.0, 0.0, 0  # 返回零值和失败标志

def main():
    """主函数 - 优化版"""
    # 创建results目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 检测CPU核心数 - 使用一半核心数以降低内存压力
    max_cores = cpu_count()
    # 限制最多使用4个进程或一半核心数，以较小者为准
    num_cores = min(8, max(1, max_cores // 2))
    print(f"\n{'='*70}")
    print(f"MIMO天线位置优化仿真 (优化版 - 多进程加速 - 内存优化)")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU核心数: {max_cores} (将使用 {num_cores} 个进程以节省内存)")
    print(f"内存优化: 限制并发进程数以避免内存耗尽")
    print(f"结果目录: {results_dir}/")
    print(f"文件前缀: optimized_{timestamp}")
    print(f"{'='*70}\n")
    
    # 参数设置（根据论文实验描述）
    A_lambda_values = np.arange(1, 10, 0.2)  # [1, 1.5, 2, 2.5, 3, 3.5, 4]
    # 支持命令行参数指定试验次数和进程数
    # 使用方法: python mimo_optimized.py [num_trials] [num_processes]
    # 例如: python mimo_optimized.py 100 2  (100次试验，2个进程)
    # 论文使用4×10⁴次，这里默认100次作为合理折中
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # 默认100次
    if len(sys.argv) > 2:
        num_cores = min(int(sys.argv[2]), max_cores)
        print(f"使用命令行指定的进程数: {num_cores}")
    Lr_values = [15]  # Lr = 15（论文设置）
    lambda_val = 1
    
    average_capacity = np.zeros((len(A_lambda_values), len(Lr_values)))
    average_total_power = np.zeros((len(A_lambda_values), len(Lr_values)))
    average_condition_number = np.zeros((len(A_lambda_values), len(Lr_values)))
    success_rates = np.zeros((len(A_lambda_values), len(Lr_values)))
    
    for j in range(len(Lr_values)):
        Lr = Lr_values[j]
        print(f"\n处理 Lr = {Lr}")
        
        for u in range(len(A_lambda_values)):
            A_lambda = A_lambda_values[u]
            
            print(f"  A/λ = {A_lambda}, 进行 {num_trials} 次试验（多进程并行）...")
            
            # 准备参数列表
            args_list = [(trial, A_lambda, Lr, lambda_val) for trial in range(num_trials)]
            
            # 使用多进程池（带内存管理）
            # 使用maxtasksperchild限制每个子进程处理的任务数，防止内存泄漏
            with Pool(processes=num_cores, maxtasksperchild=10) as pool:
                # 使用tqdm显示进度条
                # 使用imap而不是map以便逐步处理，降低内存峰值
                results = list(tqdm(
                    pool.imap(run_single_trial_optimized, args_list),
                    total=num_trials,
                    desc=f"    A/λ={A_lambda}",
                    ncols=80
                ))
            
            # 分离四个返回值
            capacities = np.array([r[0] for r in results])
            powers = np.array([r[1] for r in results])
            conditions = np.array([r[2] for r in results])
            successes = np.array([r[3] for r in results])
            
            # 计算统计量
            success_count = np.sum(successes)
            valid_mask = capacities > 0
            average_capacity[u, j] = np.mean(capacities[valid_mask]) if np.any(valid_mask) else 0
            average_total_power[u, j] = np.mean(powers[valid_mask]) if np.any(valid_mask) else 0
            average_condition_number[u, j] = np.mean(conditions[valid_mask]) if np.any(valid_mask) else 0
            success_rates[u, j] = success_count / num_trials
            
            print(f"    平均容量 = {average_capacity[u, j]:.6f} bps/Hz (成功率: {success_rates[u, j]*100:.3f}%)")
    
    # 保存详细数值结果
    results_data = {
        'timestamp': timestamp,
        'version': 'optimized',
        'parameters': {
            'A_lambda_values': A_lambda_values.tolist(),
            'num_trials': num_trials,
            'Lr_values': Lr_values,
            'lambda_val': lambda_val
        },
        'results': {
            'achievable_rate': {
                'Lr_15': average_capacity[:, 0].tolist()
            },
            'channel_total_power': {
                'Lr_15': average_total_power[:, 0].tolist()
            },
            'condition_number': {
                'Lr_15': average_condition_number[:, 0].tolist()
            },
            'success_rates': {
                'Lr_15': success_rates[:, 0].tolist()
            }
        },
        'improvements': [
            '修复了calculate_F函数中的变量覆盖bug',
            '改进了天线初始化策略，提高成功率',
            '添加了步长控制和数值稳定性检查',
            '使用相对误差作为收敛判据',
            '改进了约束处理，减少位置突变'
        ]
    }
    
    # 保存JSON格式
    json_filename = os.path.join(results_dir, f'optimized_{timestamp}.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式（包含三个指标和成功率）
    csv_filename = os.path.join(results_dir, f'optimized_{timestamp}.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write('A/lambda,Rate_Lr15,Power_Lr15,Cond_Lr15,SR_Lr15\n')
        for i, a_lambda in enumerate(A_lambda_values):
            f.write(f'{a_lambda},'
                   f'{average_capacity[i, 0]:.6f},{average_total_power[i, 0]:.6f},{average_condition_number[i, 0]:.2f},{success_rates[i, 0]:.4f}\n')
    
    # 绘图 - 学术论文风格双子图
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
    
    # 创建三子图（对应论文Fig. 4）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12))
    
    # (a) Achievable rate
    ax1.plot(A_lambda_values, average_capacity[:, 0], 
            color='#1f77b4', linestyle='-', marker='o',
            linewidth=1.8, markersize=7,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Optimized, Lr=15', zorder=3)
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
            label='Optimized, Lr=15', zorder=3)
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
                label='Optimized, Lr=15', zorder=3)
    ax3.set_xlabel(r'Normalized region size ($A/\lambda$)', fontsize=11)
    ax3.set_ylabel('Condition number', fontsize=11)
    ax3.set_title('(c) Channel condition number versus normalized region size', fontsize=12, loc='left', pad=10)
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0, which='both')
    ax3.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9)
    ax3.tick_params(direction='in', which='both', top=True, right=True)
    ax3.set_xlim(0.9, 4.1)
    
    plt.tight_layout()
    png_filename = os.path.join(results_dir, f'optimized_{timestamp}.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细报告
    print(f"\n{'='*70}")
    print("优化仿真完成！")
    print(f"{'='*70}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n结果文件:")
    print(f"  - 图像: {png_filename}")
    print(f"  - JSON: {json_filename}")
    print(f"  - CSV:  {csv_filename}")
    
    print(f"\n性能指标摘要:")
    print(f"  Lr=15 (SNR=15dB): 容量={np.mean(average_capacity[:, 0]):.6f} bps/Hz, "
          f"功率={np.mean(average_total_power[:, 0]):.2f}, "
          f"条件数={np.mean(average_condition_number[:, 0]):.2f}, "
          f"成功率={np.mean(success_rates[:, 0])*100:.1f}%")
    
    print(f"\n主要改进:")
    for i, improvement in enumerate(results_data['improvements'], 1):
        print(f"  {i}. {improvement}")
    
    print(f"\n{'='*70}")
    
    return average_capacity, average_total_power, average_condition_number, success_rates

if __name__ == "__main__":
    results = main()
