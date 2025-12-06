"""
MIMO Core Algorithm Library - Ma et al. (2023)
================================================================
Paper: "MIMO Capacity Characterization for Movable Antenna Systems"
Author: Wenyan Ma, Lipeng Zhu, Rui Zhang
Journal: IEEE Transactions on Wireless Communications, 2023

本模块实现了论文中的核心算法：
- MIMOSystem: 可移动天线 MIMO 系统的完整建模
- Algorithm 2: 交替优化算法（Proposed 方案）
- 支持多种基准模式：RMA, TMA, FPA
================================================================
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, NonlinearConstraint
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class MIMOSystem:
    def __init__(self, N=4, M=4, Lt=10, Lr=15, SNR_dB=15, lambda_val=1):
        """
        初始化MIMO系统参数
        :param N: 发送天线数
        :param M: 接收天线数
        :param Lt: 发送端散射体数
        :param Lr: 接收端散射体数
        :param SNR_dB: 信噪比 (dB)
        :param lambda_val: 波长
        """
        self.N = N
        self.M = M
        self.Lt = Lt
        self.Lr = Lr
        self.lambda_val = lambda_val
        
        # 功率和噪声设置
        self.power = 10
        self.SNR_linear = 10**(SNR_dB / 10)
        self.sigma = self.power / self.SNR_linear
        self.D = lambda_val / 2  # 最小天线间距
        
        # 收敛容差
        self.xi = 1e-3   # 外循环容差
        self.xii = 1e-3  # 内循环容差

    def compute_field_response(self, r, theta, phi):
        """计算场响应向量"""
        x, y = r[0], r[1]
        phase = 2*np.pi/self.lambda_val * (x*np.sin(theta)*np.cos(phi) + y*np.cos(theta))
        return np.exp(1j*phase), phase

    def calculate_F(self, theta_q, phi_q, r):
        """计算接收端阵列响应矩阵 F"""
        F = np.zeros((self.Lr, self.M), dtype=complex)
        for m in range(self.M):
            f_r, _ = self.compute_field_response(r[:, m], theta_q, phi_q)
            F[:, m] = f_r
        return F

    def calculate_G(self, theta_p, phi_p, t):
        """计算发送端阵列响应矩阵 G"""
        G = np.zeros((self.Lt, self.N), dtype=complex)
        for n in range(self.N):
            g_t, _ = self.compute_field_response(t[:, n], theta_p, phi_p)
            G[:, n] = g_t
        return G

    def calculate_B_m(self, G, Q, H_r, m0, Sigma):
        """计算接收端优化目标矩阵 B_m"""
        # 数值稳定性处理
        Q_stable = Q + 1e-10 * np.eye(self.N)
        eigvals, eigvecs = np.linalg.eig(Q_stable)
        eigvals = np.maximum(eigvals, 0)
        Lambda_Q_sqrt = np.sqrt(np.diag(eigvals))
        U_Q = eigvecs
        
        W_r = H_r @ U_Q @ Lambda_Q_sqrt
        W_r_H = W_r.T.conj()
        W_r_H = np.delete(W_r_H, m0, axis=1)
        
        # 正则化求逆
        A_m = np.linalg.inv(np.eye(self.N) + (1/self.sigma**2) * (W_r_H @ W_r_H.T.conj()) + 1e-10*np.eye(self.N))
        B_m = Sigma @ G @ U_Q @ Lambda_Q_sqrt @ A_m @ Lambda_Q_sqrt @ U_Q.T.conj() @ G.T.conj() @ Sigma.T.conj()
        return B_m

    def calculate_S_matrix(self, H_r):
        """计算发送端对偶协方差矩阵 S"""
        H_HH = H_r @ H_r.T.conj()
        eigvals, eigvecs = np.linalg.eigh(H_HH)
        eigvals = np.maximum(eigvals, 0)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        rank_S = min(self.M, np.sum(eigvals > 1e-10))
        p_s = np.zeros(self.M)
        if rank_S > 0:
            p_s[:rank_S] = self.power / rank_S
            
        Lambda_S = np.diag(p_s)
        S = eigvecs @ Lambda_S @ eigvecs.T.conj()
        Lambda_S_sqrt = np.sqrt(Lambda_S)
        U_S = eigvecs
        return S, U_S, Lambda_S_sqrt

    def calculate_D_n(self, F, S_tuple, G, n0, Sigma):
        """计算发送端优化目标矩阵 D_n"""
        S, U_S, Lambda_S_sqrt = S_tuple
        
        P_tilde = G.T.conj() @ Sigma.T.conj() @ F @ U_S @ Lambda_S_sqrt
        P_tilde_n = np.delete(P_tilde, n0, axis=0)
        
        C_n_inv = np.eye(self.M) + (1/self.sigma**2) * (P_tilde_n.T.conj() @ P_tilde_n)
        C_n = np.linalg.inv(C_n_inv + 1e-10*np.eye(self.M))
        A_n = C_n
        
        D_n = Sigma.T.conj() @ F @ U_S @ Lambda_S_sqrt @ A_n @ Lambda_S_sqrt @ U_S.T.conj() @ F.T.conj() @ Sigma
        return D_n

    def calculate_gradients(self, Matrix, vec, pos, theta, phi):
        """计算梯度和Hessian近似 (通用)"""
        val = Matrix @ vec
        amplitude = np.abs(val)
        phase_val = np.angle(val)
        
        x, y = pos[0], pos[1]
        rho = x*np.sin(theta)*np.cos(phi) + y*np.cos(theta)
        kappa = (2*np.pi/self.lambda_val)*rho - phase_val
        
        term_x = amplitude * np.sin(theta) * np.cos(phi) * np.sin(kappa)
        term_y = amplitude * np.cos(theta) * np.sin(kappa)
        
        grad_x = -(2*np.pi/self.lambda_val) * np.sum(term_x)
        grad_y = -(2*np.pi/self.lambda_val) * np.sum(term_y)
        grad = np.array([grad_x, grad_y])
        
        sum_abs = np.sum(np.abs(val))
        delta = (8*np.pi**2) / (self.lambda_val**2) * sum_abs
        
        return grad, delta

    def initialize_antennas_fixed_ula(self, Count):
        """
        生成固定的均匀线阵 (ULA - Uniform Linear Array)
        FPA 模式专用：间隔固定为 lambda/2，与区域大小 A 无关
        """
        r = np.zeros((2, Count))
        # 沿 x 轴排列，间隔为 D (lambda/2)
        for i in range(Count):
            r[0, i] = i * self.D
            r[1, i] = 0  # y 坐标固定为 0
        # 将阵列中心移到原点附近
        r[0, :] -= np.mean(r[0, :])
        return r
    
    def initialize_antennas_smart(self, Count, square_size):
        """智能网格初始化 (用于 MA 模式)"""
        grid_size = int(np.ceil(np.sqrt(Count)))
        spacing = square_size / (grid_size + 1)
        
        if spacing >= self.D:
            r = np.zeros((2, Count))
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= Count: break
                    r[0, idx] = (i + 1) * spacing + np.random.randn() * self.D * 0.1
                    r[1, idx] = (j + 1) * spacing + np.random.randn() * self.D * 0.1
                    idx += 1
                    if idx >= Count: break
            return r
        
        # 随机回退
        attempts = 0
        while attempts < 1000:
            candidates = np.random.rand(2, 5000) * square_size
            valid = np.ones(5000, dtype=bool)
            for k in range(1, 5000):
                dists = np.linalg.norm(candidates[:, :k] - candidates[:, k:k+1], axis=0)
                if np.min(dists) < self.D: valid[k] = False
            
            valid_points = candidates[:, valid]
            if valid_points.shape[1] >= Count:
                return valid_points[:, :Count]
            attempts += 1
        raise RuntimeError(f'初始化失败: size={square_size}, Count={Count}')

    def optimize_position_robust(self, pos_i, all_pos, idx, Count, square_size, grad, delta, theta, phi, Matrix):
        """稳健的位置优化 (SCA/Gradient Descent)"""
        max_step = self.D
        step = grad / delta
        if np.linalg.norm(step) > max_step:
            step = step * (max_step / np.linalg.norm(step))
            
        pos_new = pos_i + step
        
        # 检查可行性
        is_feasible = True
        if np.any(pos_new < 0) or np.any(pos_new > square_size):
            is_feasible = False
        else:
            for k in range(Count):
                if k != idx and np.linalg.norm(pos_new - all_pos[:, k]) < self.D:
                    is_feasible = False; break
        
        if is_feasible: return pos_new, True
        
        # 投影梯度法 (Fallback)
        def objective(x):
            f_x, _ = self.compute_field_response(x, theta, phi)
            return -np.real(f_x.T.conj() @ Matrix @ f_x)
            
        def gradient(x):
            f_x, _ = self.compute_field_response(x, theta, phi)
            g, _ = self.calculate_gradients(Matrix, f_x, x, theta, phi)
            return -g

        def dist_constraint(x):
            dists = []
            for k in range(Count):
                if k != idx: dists.append(np.linalg.norm(x - all_pos[:, k]) - self.D)
            return np.array(dists) if dists else np.array([1.0])

        res = minimize(objective, pos_i, method='SLSQP', jac=gradient,
                      bounds=[(0, square_size)]*2,
                      constraints=[NonlinearConstraint(dist_constraint, 0, np.inf)],
                      options={'ftol': 1e-8, 'disp': False})
        
        if res.success: return res.x, True
        return pos_i, False

    def run_optimization(self, A_lambda, mode='Proposed', init_t=None, init_r=None, channel_params=None):
        """
        运行单次优化试验
        :param A_lambda: 归一化区域大小
        :param mode: 'Proposed', 'RMA', 'TMA', 'FPA'
        :param init_t: Optional initial transmit antenna positions (2, N)
        :param init_r: Optional initial receive antenna positions (2, M)
        :param channel_params: Optional dictionary containing fixed channel parameters
        :return: capacity
        """
        square_size = A_lambda * self.lambda_val
        
        # 初始化位置 - 根据模式选择初始化方式
        if mode == 'FPA':
            # FPA: 固定均匀线阵，与区域大小无关
            r = self.initialize_antennas_fixed_ula(self.M)
            t = self.initialize_antennas_fixed_ula(self.N)
        else:
            # MA 模式: 在区域内智能初始化
            # Use provided initialization if available, otherwise use smart initialization
            if init_r is not None:
                r = init_r.copy()
            else:
                r = self.initialize_antennas_smart(self.M, square_size)
                
            if init_t is not None:
                t = init_t.copy()
            else:
                t = self.initialize_antennas_smart(self.N, square_size)
        
        # 信道参数
        if channel_params:
            theta_p = channel_params['theta_p']
            phi_p = channel_params['phi_p']
            theta_q = channel_params['theta_q']
            phi_q = channel_params['phi_q']
            Sigma = channel_params['Sigma']
        else:
            theta_p = np.random.rand(self.Lt) * np.pi
            phi_p = np.random.rand(self.Lt) * np.pi
            theta_q = np.random.rand(self.Lr) * np.pi
            phi_q = np.random.rand(self.Lr) * np.pi
            
            # Rician信道
            Sigma = np.zeros((self.Lr, self.Lt), dtype=complex)
            kappa = 1
            Sigma[0, 0] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(kappa/(kappa+1)/2)
            for i in range(1, min(self.Lr, self.Lt)):
                Sigma[i, i] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(1/((kappa+1)*(self.Lr-1))/2)
            
        cap_prev = 0
        max_iter = 50 if mode != 'FPA' else 1
        
        for _ in range(max_iter):
            # 0. 更新信道
            F = self.calculate_F(theta_q, phi_q, r)
            G = self.calculate_G(theta_p, phi_p, t)
            H_r = F.T.conj() @ Sigma @ G
            
            # 1. 功率分配 (CVX - Optimized with Parameters)
            # 将问题构建移到循环外（首次运行时构建），后续只更新参数
            if not hasattr(self, '_cvx_prob_cache'):
                self._Q_var = cp.Variable((self.N, self.N), hermitian=True)
                self._Hr_param = cp.Parameter((self.M, self.N), complex=True)
                
                # 目标函数构建
                # obj = log_det(I + 1/sigma * Hr @ Q @ Hr^H)
                # 为了符合DCP规则，通常写为 log_det(I + 1/sigma * X)，其中 X = Hr @ Q @ Hr^H 
                # 但 X 与 Q 的关系是仿射的，直接写通常没问题，或者用辅助变量
                
                # 直接写法在较新版本CVXPY中通常支持，或者使用 transform
                term = (1.0/self.sigma) * (self._Hr_param @ self._Q_var @ self._Hr_param.H)
                obj = cp.log_det(np.eye(self.M) + term)
                
                self._cvx_prob_cache = cp.Problem(cp.Maximize(obj), 
                                                [cp.trace(self._Q_var) <= self.power, self._Q_var >> 0])
            
            # 更新参数并求解
            self._Hr_param.value = H_r
            try:
                # 启用 warm_start，利用上一次的结果加速
                self._cvx_prob_cache.solve(solver=cp.SCS, verbose=False, warm_start=True, eps=1e-3)
                Q = self._Q_var.value if self._cvx_prob_cache.status in ['optimal', 'optimal_inaccurate'] else np.eye(self.N)*(self.power/self.N)
            except:
                Q = np.eye(self.N)*(self.power/self.N)
                
            if mode == 'FPA': break
            
            # 2. 接收端优化
            if mode in ['Proposed', 'RMA']:
                for m in range(self.M):
                    r_mi = r[:, m].copy()
                    prev_sub_obj = 0
                    for _ in range(30):
                        F = self.calculate_F(theta_q, phi_q, r)
                        H_r = F.T.conj() @ Sigma @ G
                        B_m = self.calculate_B_m(G, Q, H_r, m, Sigma)
                        f_rm, _ = self.compute_field_response(r_mi, theta_q, phi_q)
                        grad, delta = self.calculate_gradients(B_m, f_rm, r_mi, theta_q, phi_q)
                        
                        r_new, _ = self.optimize_position_robust(r_mi, r, m, self.M, square_size, grad, delta, theta_q, phi_q, B_m)
                        r_mi = r_new
                        r[:, m] = r_mi
                        
                        f_new, _ = self.compute_field_response(r_mi, theta_q, phi_q)
                        curr_sub_obj = np.real(f_new.T.conj() @ B_m @ f_new)
                        if abs(curr_sub_obj - prev_sub_obj) < self.xii * (abs(curr_sub_obj)+1e-6): break
                        prev_sub_obj = curr_sub_obj

            # 3. 发送端优化
            if mode in ['Proposed', 'TMA']:
                F = self.calculate_F(theta_q, phi_q, r)
                G = self.calculate_G(theta_p, phi_p, t)
                H_r = F.T.conj() @ Sigma @ G
                S_tuple = self.calculate_S_matrix(H_r)
                
                for n in range(self.N):
                    t_ni = t[:, n].copy()
                    prev_sub_obj = 0
                    for _ in range(30):
                        G = self.calculate_G(theta_p, phi_p, t)
                        D_n = self.calculate_D_n(F, S_tuple, G, n, Sigma)
                        g_tn, _ = self.compute_field_response(t_ni, theta_p, phi_p)
                        grad, delta = self.calculate_gradients(D_n, g_tn, t_ni, theta_p, phi_p)
                        
                        t_new, _ = self.optimize_position_robust(t_ni, t, n, self.N, square_size, grad, delta, theta_p, phi_p, D_n)
                        t_ni = t_new
                        t[:, n] = t_ni
                        
                        g_new, _ = self.compute_field_response(t_ni, theta_p, phi_p)
                        curr_sub_obj = np.real(g_new.T.conj() @ D_n @ g_new)
                        if abs(curr_sub_obj - prev_sub_obj) < self.xii * (abs(curr_sub_obj)+1e-6): break
                        prev_sub_obj = curr_sub_obj

            # 收敛检查
            F = self.calculate_F(theta_q, phi_q, r)
            G = self.calculate_G(theta_p, phi_p, t)
            H_r = F.T.conj() @ Sigma @ G
            H_rQH = H_r @ Q @ H_r.T.conj()
            eigvals = np.linalg.eigvalsh(np.eye(self.M) + (1/self.sigma) * H_rQH)
            cap_curr = np.sum(np.log2(np.maximum(eigvals, 1e-10)))
            
            if abs(cap_curr - cap_prev) < self.xi * (abs(cap_curr)+1e-6): break
            cap_prev = cap_curr
            
        # 最终结果计算
        F = self.calculate_F(theta_q, phi_q, r)
        G = self.calculate_G(theta_p, phi_p, t)
        H_r = F.T.conj() @ Sigma @ G
        
        # 1. Achievable Rate (Capacity)
        H_rQH = H_r @ Q @ H_r.T.conj()
        eigvals_cap = np.linalg.eigvalsh(np.eye(self.M) + (1/self.sigma) * H_rQH)
        capacity = np.sum(np.log2(np.maximum(eigvals_cap, 1e-10)))
        
        # 2. Strongest Eigenchannel Power (max(lambda))
        # 对 H_r * H_r^H 进行特征值分解，找出最大特征值
        H_HH = H_r @ H_r.T.conj()
        eigvals_power = np.linalg.eigvalsh(H_HH)
        strongest_eigen_power = np.max(eigvals_power)
        
        # 3. Channel Total Power (frobenius norm squared or sum of eigenvalues)
        total_power = np.sum(eigvals_power) # trace(H H^H) = ||H||_F^2
        
        # 4. Condition Number (lambda_max / lambda_min)
        min_eig = np.min(eigvals_power)
        cond_number = strongest_eigen_power / (min_eig + 1e-10)
        
        return {
            'capacity': capacity,
            'strongest_eigen_power': strongest_eigen_power,
            'total_power': total_power,
            'cond_number': cond_number,
            't': t,
            'r': r
        }
