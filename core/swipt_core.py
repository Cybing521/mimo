import numpy as np
import cvxpy as cp

class SWIPTSystem:
    def __init__(self, Nt=4, Ne=4, Ni=4, bandwidth=1e6, noise_density_dbm=-100):
        """
        Initialize SWIPT System parameters.
        
        Args:
            Nt (int): Number of transmit antennas
            Ne (int): Number of EH receiver antennas
            Ni (int): Number of ID receiver antennas
            bandwidth (float): System bandwidth in Hz
            noise_density_dbm (float): Noise power density in dBm/Hz
        """
        self.Nt = Nt
        self.Ne = Ne
        self.Ni = Ni
        
        # Nonlinear EH parameters (default set to one configuration from paper)
        self.a = 6400
        self.b = 0.003
        self.M = 0.02 # Max harvested power (Watt), typically small, e.g., 20mW
        
        # Calculate noise power
        # -100 dBm/Hz -> 10^(-10) mW/Hz = 10^(-13) W/Hz
        noise_density_w = 10**((noise_density_dbm - 30) / 10)
        self.sigma2_I = noise_density_w * bandwidth # Noise power at ID receiver
        self.sigma2_A = 0 # Antenna noise (assumed negligible or part of sigma2 in some contexts)
        self.sigma2_P = self.sigma2_I # Circuit noise
        
    def set_nonlinear_params(self, a, b, M):
        """Set parameters for the nonlinear EH model."""
        self.a = a
        self.b = b
        self.M = M
        
    def nonlinear_eh_model(self, x):
        """
        Calculate harvested energy using the nonlinear model (Eq. 2).
        x: Input RF power (Tr(He Q He^H))
        """
        # Constants
        Omega1 = 1 / (1 + np.exp(self.a * self.b))
        
        # Logistic function
        # Note: The paper formula is Psi1(x) = (beta(x) - M*Omega1) / (1 - Omega1)
        # where beta(x) = M / (1 + exp(-a(x-b)))
        
        beta_x = self.M / (1 + np.exp(-self.a * (x - self.b)))
        Psi1_x = (beta_x - self.M * Omega1) / (1 - Omega1)
        
        # Ensure non-negative energy
        return np.maximum(Psi1_x, 0)

    def inverse_nonlinear_eh_model(self, E):
        """
        Calculate required input power for a given harvested energy E.
        Inverse of Eq. 2.
        """
        # E = (beta - M*Omega) / (1 - Omega)
        # beta = E * (1 - Omega) + M * Omega
        # M / (1 + exp(-a(x-b))) = beta
        # 1 + exp(-a(x-b)) = M / beta
        # exp(-a(x-b)) = M/beta - 1
        # -a(x-b) = ln(M/beta - 1)
        # x = b - (1/a) * ln(M/beta - 1)
        
        Omega1 = 1 / (1 + np.exp(self.a * self.b))
        beta = E * (1 - Omega1) + self.M * Omega1
        
        # Check valid range
        if np.any(beta >= self.M) or np.any(beta <= 0):
             # E approaches max possible or is invalid
             return np.inf if np.any(beta >= self.M) else 0
             
        term = self.M / beta - 1
        # Avoid log(0) or log(negative)
        term = np.maximum(term, 1e-10)
        
        x = self.b - (1/self.a) * np.log(term)
        return np.maximum(x, 0)

    def solve_separated_receivers(self, He, Hi, P_max, R_min):
        """
        Solve Problem P4/P'4 for Separated Receivers.
        Maximize Energy subject to Rate constraint.
        
        Returns:
            E_max: Maximum harvested energy
            Q_opt: Optimal covariance matrix
        """
        # We solve P'4 (Eq. 14): Max Tr(He Q He^H) s.t. Rate >= R_min, Tr(Q) <= P
        # This is a convex problem. We can use CVXPY directly which is often more robust 
        # and easier than implementing the sub-gradient method manually, 
        # especially since we have the library.
        
        Q = cp.Variable((self.Nt, self.Nt), hermitian=True)
        
        # Objective: Maximize Input RF Power (monotonic to Energy)
        input_power = cp.real(cp.trace(He @ Q @ He.conj().T))
        objective = cp.Maximize(input_power)
        
        # Constraints
        # Rate >= R_min  => log det (I + Hi Q Hi^H / sigma2) >= R_min
        # This requires carefully handling the log_det in CVXPY
        
        # Signal part matrix S = (1/sigma2) * Hi Q Hi^H
        # We need log_det(I + S) >= R_min
        
        # CVXPY supports log_det, but we need to express it correctly.
        # Since Q is PSD, Hi Q Hi^H is PSD.
        
        constraints = [
            cp.trace(Q) <= P_max,
            Q >> 0
        ]
        
        # Rate constraint
        # log_det(I + 1/sigma2 * Hi @ Q @ Hi') >= R_min
        # Note: cp.log_det(X) is concave, so cp.log_det(X) >= R is a convex constraint.
        S = (1.0 / self.sigma2_I) * (Hi @ Q @ Hi.conj().T)
        constraints.append(cp.log_det(np.eye(self.Ni) + S) >= R_min * np.log(2)) # log base e vs base 2
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, verbose=False) # SCS is a splitting conic solver
        except:
            try:
                prob.solve(solver=cp.CVXOPT, verbose=False)
            except:
                return 0.0, None

        if prob.status not in ['optimal', 'optimal_inaccurate']:
            return 0.0, None
            
        Q_opt = Q.value
        rf_power = prob.value
        energy = self.nonlinear_eh_model(rf_power)
        
        return energy, Q_opt

    def solve_co_located_ts(self, H, P_max, R_req):
        """
        Solve Co-located TS (Time Switching) problem (Problem P6).
        Using Algorithm 2: Alternating optimization of (Qe, Qi) and theta.
        """
        # Initialize theta
        theta = 0.5
        
        max_iter = 20
        epsilon = 1e-4
        E_prev = 0
        
        best_E = 0
        best_params = None
        
        for i in range(max_iter):
            # Step 1: Optimal Qe, Qi for given theta
            # Problem (25): Max Tr(H Qe H^H) s.t. constraints
            # Rate constraint: (1-theta) log... >= R_req => log... >= R_req / (1-theta)
            
            if theta >= 1.0 or theta <= 0.0:
                theta = np.clip(theta, 0.01, 0.99)
                
            R_target_inst = R_req / (1 - theta)
            
            Qe = cp.Variable((self.Nt, self.Nt), hermitian=True)
            Qi = cp.Variable((self.Nt, self.Nt), hermitian=True)
            
            obj_func = cp.real(cp.trace(H @ Qe @ H.conj().T))
            objective = cp.Maximize(obj_func)
            
            constraints = [
                theta * cp.trace(Qe) + (1 - theta) * cp.trace(Qi) <= P_max,
                Qe >> 0,
                Qi >> 0
            ]
            
            S_i = (1.0 / self.sigma2_I) * (H @ Qi @ H.conj().T)
            constraints.append(cp.log_det(np.eye(self.Ni) + S_i) >= R_target_inst * np.log(2))
            
            prob = cp.Problem(objective, constraints)
            try:
                # Enable warm_start and lower epsilon for speed
                prob.solve(solver=cp.SCS, verbose=False, warm_start=True, eps=1e-3)
            except:
                break
                
            if prob.status not in ['optimal', 'optimal_inaccurate']:
                break
                
            Qe_val = Qe.value
            Qi_val = Qi.value
            
            if Qe_val is None or Qi_val is None:
                break
                
            # Step 2: Optimal theta for given Qe, Qi (Lemma 5)
            # Calculate rate term
            try:
                term_S = (1.0 / self.sigma2_I) * (H @ Qi_val @ H.conj().T)
                eig_S = np.linalg.eigvals(np.eye(self.Ni) + term_S)
                rate_val = np.sum(np.log2(np.abs(eig_S) + 1e-10)) # log2 determinant
                
                if rate_val < 1e-9:
                    theta_new = 0.01
                else:
                    bound1 = 1 - R_req / rate_val
                    
                    tr_Qe = np.real(np.trace(Qe_val))
                    tr_Qi = np.real(np.trace(Qi_val))
                    
                    if tr_Qe > tr_Qi:
                        bound2 = (P_max - tr_Qi) / (tr_Qe - tr_Qi)
                        theta_new = min(bound1, bound2)
                    else:
                        theta_new = bound1
            except:
                theta_new = theta
                
            # Update theta
            theta = max(0.0, min(1.0, theta_new))
            
            # Calculate Energy
            # E = theta * Psi1(Tr(H Qe H^H))
            rf_power = np.real(np.trace(H @ Qe_val @ H.conj().T))
            E_curr = theta * self.nonlinear_eh_model(rf_power)
            
            if E_curr > best_E:
                best_E = E_curr
                best_params = (theta, Qe_val, Qi_val)
            
            if abs(E_curr - E_prev) < epsilon:
                break
            E_prev = E_curr
            
        return best_E, best_params

    def solve_co_located_ps(self, H, P_max, R_req):
        """
        Solve Co-located PS (Power Splitting) problem (Problem P7).
        Using Algorithm 3: Alternating optimization of Q and Omega_rho.
        """
        # Initialize Omega_rho (diagonal matrix of power splitting factors)
        # Using vector rho for diagonal elements
        rho = np.ones(self.Ni) * 0.5 
        
        max_iter = 20
        epsilon = 1e-4
        E_prev = 0
        
        best_E = 0
        best_params = None
        
        for i in range(max_iter):
            # Step 1: Optimal Q for given rho
            # Problem (31): Max Psi1(Tr(F_rho Q F_rho^H)) s.t. Rate >= R
            # F_rho = Omega_rho^(1/2) H
            # H_rho = (I-Omega_rho)^(1/2) H
            
            Omega_rho_sqrt = np.diag(np.sqrt(rho))
            Omega_bar_sqrt = np.diag(np.sqrt(1 - rho))
            
            F_rho = Omega_rho_sqrt @ H
            H_rho = Omega_bar_sqrt @ H
            
            Q = cp.Variable((self.Nt, self.Nt), hermitian=True)
            
            # Maximize Tr(F_rho Q F_rho^H)
            obj_func = cp.real(cp.trace(F_rho @ Q @ F_rho.conj().T))
            objective = cp.Maximize(obj_func)
            
            constraints = [
                cp.trace(Q) <= P_max,
                Q >> 0
            ]
            
            # Rate constraint
            S = (1.0 / self.sigma2_I) * (H_rho @ Q @ H_rho.conj().T)
            constraints.append(cp.log_det(np.eye(self.Ni) + S) >= R_req * np.log(2))
            
            prob = cp.Problem(objective, constraints)
            try:
                # Enable warm_start
                prob.solve(solver=cp.SCS, verbose=False, warm_start=True, eps=1e-3)
            except:
                break
                
            if prob.status not in ['optimal', 'optimal_inaccurate']:
                break
                
            Q_val = Q.value
            if Q_val is None: break
            
            # Step 2: Optimal rho for given Q
            # Problem (34) in paper is used to find W, then rho
            # min sigma^2 Tr(W) - Tr(H Q H^H) - sigma^2 Tr(I)
            # s.t. log|W| >= R, I <= W <= I + H Q H^H
            
            # Wait, the derivation in paper for rho update is slightly complex.
            # Let's check Eq 35.
            # wi = (1 - rho_i) * di
            # where di = (H Q H^H)_ii
            # wi comes from optimal W.
            
            # Let's solve Problem (34)
            # W is Ni x Ni positive definite
            
            HQH = H @ Q_val @ H.conj().T
            di = np.real(np.diag(HQH)) # diagonal elements
            
            W = cp.Variable((self.Ni, self.Ni), PSD=True)
            
            # Objective: min sigma^2 Tr(W)
            # (Constant terms dropped for optimization)
            obj_W = self.sigma2_I * cp.trace(W)
            objective_W = cp.Minimize(obj_W)
            
            constraints_W = [
                cp.log_det(W) >= R_req * np.log(2),
                W >> np.eye(self.Ni),
                W << np.eye(self.Ni) + HQH
            ]
            
            prob_W = cp.Problem(objective_W, constraints_W)
            try:
                prob_W.solve(solver=cp.SCS, verbose=False)
            except:
                break
                
            if prob_W.status not in ['optimal', 'optimal_inaccurate']:
                # Fallback or break
                break
                
            W_val = W.value
            
            # Update rho using Eq 35
            # rho_i = 1 - (w_ii - 1) / d_ii ?
            # Paper Eq 35: rho_i = 1 - w_i / d_i  (Wait, verify w_i definition)
            # Text says: wi is i-th diagonal entry of W - I.
            # So diag(W) = 1 + w.
            # And wi = (1-rho) * di
            # So (W_ii - 1) = (1 - rho_i) * di
            # => 1 - rho_i = (W_ii - 1) / di
            # => rho_i = 1 - (W_ii - 1) / di
            
            W_diag = np.real(np.diag(W_val))
            rho_new = []
            for k in range(self.Ni):
                if abs(di[k]) < 1e-10:
                    rho_new.append(0.0) # Avoid division by zero
                else:
                    val = 1.0 - (W_diag[k] - 1.0) / di[k]
                    rho_new.append(val)
            
            rho = np.array(rho_new)
            rho = np.clip(rho, 0.0, 1.0)
            
            # Calculate Energy
            # E = Psi1(Tr(Omega_rho H Q H^H))
            Omega_rho_new = np.diag(rho)
            rf_power = np.real(np.trace(Omega_rho_new @ HQH))
            E_curr = self.nonlinear_eh_model(rf_power)
            
            if E_curr > best_E:
                best_E = E_curr
                best_params = (rho, Q_val)
                
            if abs(E_curr - E_prev) < epsilon:
                break
            E_prev = E_curr
            
        return best_E, best_params


