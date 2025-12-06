import numpy as np
import cma
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CMAESOptimizer:
    """
    Wrapper for CMA-ES optimization of MIMO antenna positions.
    """
    def __init__(self, env, sigma0=0.5):
        """
        Args:
            env: MAMIMOEnv instance
            sigma0: Initial standard deviation for CMA-ES
        """
        self.env = env
        self.N = env.N
        self.M = env.M
        self.sigma0 = sigma0
        self.square_size = env.square_size
        self.bounds = [0, self.square_size]
        
    def optimize(self, max_fevals=1000):
        """
        Run CMA-ES optimization for the current channel realization.
        
        Args:
            max_fevals: Maximum number of function evaluations
            
        Returns:
            best_capacity: Best capacity found
            best_positions: Dictionary with 't' and 'r'
        """
        # Initial guess (random or center)
        # Flattened: [tx_x..., tx_y..., rx_x..., rx_y...]
        # But env expects specific structure. 
        # Let's use the same structure as env.t and env.r flattened.
        # env.t is (2, N), env.r is (2, M)
        
        # Initial mean: center of the region
        x0 = np.ones(2 * (self.N + self.M)) * (self.square_size / 2.0)
        
        # Define objective function
        def objective(x):
            # x is flattened vector
            # Reshape to t and r
            # Structure: [t_x1, t_x2..., t_y1..., r_x1...] ? 
            # No, let's stick to a simple convention:
            # First 2*N are Tx (x1, y1, x2, y2...)
            # Next 2*M are Rx (x1, y1, x2, y2...)
            
            # Wait, env.t is (2, N). Flattening it gives [x1, x2..., y1, y2...] (default row-major)
            # or [x1, y1, x2, y2...] (if we flatten differently).
            # Let's use simple reshaping.
            
            t_flat = x[:2*self.N]
            r_flat = x[2*self.N:]
            
            t = t_flat.reshape(2, self.N)
            r = r_flat.reshape(2, self.M)
            
            # Check constraints (soft penalty is handled by CMA-ES bounds, 
            # but minimum distance needs penalty)
            
            # 1. Boundary constraints are handled by cma.fmin bounds option, 
            # but here we might need to handle it if we use ask-and-tell.
            # Let's rely on cma bounds.
            
            # 2. Minimum distance constraint
            dist_penalty = 0
            D = self.env.mimo_system.D
            
            # Tx distance
            for i in range(self.N):
                for j in range(i+1, self.N):
                    dist = np.linalg.norm(t[:, i] - t[:, j])
                    if dist < D:
                        dist_penalty += (D - dist)**2
            
            # Rx distance
            for i in range(self.M):
                for j in range(i+1, self.M):
                    dist = np.linalg.norm(r[:, i] - r[:, j])
                    if dist < D:
                        dist_penalty += (D - dist)**2
            
            if dist_penalty > 0:
                return 100 + dist_penalty * 1000 # Large penalty
            
            # Calculate Capacity
            # We need to update the environment's temporary state to calculate capacity
            # But env.step updates state. We just want to calculate capacity for a given configuration
            # without changing the env's internal "current step" or history if possible.
            # Actually, we can just use mimo_system directly!
            
            # We need the current channel parameters (theta, phi, Sigma)
            # env has them in self._theta_p, etc.
            
            # Reconstruct channel
            F = self.env.mimo_system.calculate_F(self.env._theta_q, self.env._phi_q, r)
            G = self.env.mimo_system.calculate_G(self.env._theta_p, self.env._phi_p, t)
            H = F.T.conj() @ self.env._Sigma @ G
            
            # Water filling
            # Use env._water_filling or calculate directly
            # Let's use env._water_filling logic to be consistent
            # Or better, replicate the capacity calculation logic here to be safe/fast
            
            H_HH = H @ H.T.conj()
            eigvals = np.linalg.eigvalsh(H_HH)
            eigvals = np.maximum(eigvals, 0)
            
            # Water filling
            # We can reuse env._water_filling if we make it static or public
            # Or just copy it here for independence
            
            p_alloc = self._water_filling(eigvals, self.env.mimo_system.power, self.env.mimo_system.sigma)
            
            # Capacity
            # C = sum log2(1 + p_i * lambda_i^2 / sigma^2)
            # Note: eigvals are lambda_i^2
            snr_i = p_alloc * eigvals / self.env.mimo_system.sigma
            capacity = np.sum(np.log2(1 + snr_i))
            
            return -capacity # Minimize negative capacity

        # Run CMA-ES
        # options = {'bounds': [0, self.square_size], 'maxfevals': max_fevals, 'verbose': -1}
        # Note: cma bounds are [lower, upper] where lower/upper can be scalars or lists
        
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, {
            'bounds': [0, self.square_size], 
            'maxfevals': max_fevals,
            'verbose': -9
        })
        
        es.optimize(objective)
        
        best_x = es.result.xbest
        best_capacity = -es.result.fbest
        
        t_flat = best_x[:2*self.N]
        r_flat = best_x[2*self.N:]
        best_t = t_flat.reshape(2, self.N)
        best_r = r_flat.reshape(2, self.M)
        
        return best_capacity, {'t': best_t, 'r': best_r}

    def _water_filling(self, gains, total_power, noise_var):
        """Classic water-filling (copied from env)"""
        alloc = np.zeros_like(gains, dtype=float)
        if gains.size == 0 or total_power <= 0:
            return alloc
        
        positive = gains > 1e-10
        if not np.any(positive):
            return alloc
        
        gains_pos = gains[positive]
        inv = noise_var / (gains_pos + 1e-12)
        sort_idx = np.argsort(inv)
        inv_sorted = inv[sort_idx]
        n = len(inv_sorted)
        
        active = n
        water_level = 0.0
        for k in range(1, n + 1):
            mu = (total_power + np.sum(inv_sorted[:k])) / k
            if k == n or mu <= inv_sorted[k]:
                active = k
                water_level = mu
                break
        
        powers_sorted = np.maximum(water_level - inv_sorted[:active], 0.0)
        alloc_pos = np.zeros_like(gains_pos)
        alloc_pos[sort_idx[:active]] = powers_sorted
        
        alloc[positive] = alloc_pos
        return alloc

    def optimize_hybrid(self, max_fevals=1000, popsize=50):
        """
        Run Hybrid CMA-ES + AO optimization.
        
        Args:
            max_fevals: Maximum number of function evaluations for CMA-ES
            popsize: Population size for CMA-ES (larger = better global search)
            
        Returns:
            final_capacity: Final capacity after AO fine-tuning
            final_positions: Dictionary with 't' and 'r'
        """
        # 1. Run CMA-ES with increased population size for exploration
        # Initial mean: center of the region
        x0 = np.ones(2 * (self.N + self.M)) * (self.square_size / 2.0)
        
        # Define objective (same as optimize)
        def objective(x):
            t_flat = x[:2*self.N]
            r_flat = x[2*self.N:]
            t = t_flat.reshape(2, self.N)
            r = r_flat.reshape(2, self.M)
            
            # Constraints
            dist_penalty = 0
            D = self.env.mimo_system.D
            
            for i in range(self.N):
                for j in range(i+1, self.N):
                    dist = np.linalg.norm(t[:, i] - t[:, j])
                    if dist < D: dist_penalty += (D - dist)**2
            for i in range(self.M):
                for j in range(i+1, self.M):
                    dist = np.linalg.norm(r[:, i] - r[:, j])
                    if dist < D: dist_penalty += (D - dist)**2
            
            if dist_penalty > 0:
                return 100 + dist_penalty * 1000
            
            # Capacity
            F = self.env.mimo_system.calculate_F(self.env._theta_q, self.env._phi_q, r)
            G = self.env.mimo_system.calculate_G(self.env._theta_p, self.env._phi_p, t)
            H = F.T.conj() @ self.env._Sigma @ G
            H_HH = H @ H.T.conj()
            eigvals = np.linalg.eigvalsh(H_HH)
            eigvals = np.maximum(eigvals, 0)
            p_alloc = self._water_filling(eigvals, self.env.mimo_system.power, self.env.mimo_system.sigma)
            snr_i = p_alloc * eigvals / self.env.mimo_system.sigma
            capacity = np.sum(np.log2(1 + snr_i))
            return -capacity

        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, {
            'bounds': [0, self.square_size], 
            'maxfevals': max_fevals,
            'popsize': popsize, # Explicitly set population size
            'verbose': -9
        })
        es.optimize(objective)
        
        best_x = es.result.xbest
        cma_capacity = -es.result.fbest
        
        # 2. Extract best solution
        t_flat = best_x[:2*self.N]
        r_flat = best_x[2*self.N:]
        init_t = t_flat.reshape(2, self.N)
        init_r = r_flat.reshape(2, self.M)
        
        # 3. Run AO Fine-tuning
        # We need to pass the current channel parameters to AO
        channel_params = self.env.get_channel_params()
        
        ao_res = self.env.mimo_system.run_optimization(
            A_lambda=self.env.A_lambda,
            mode='Proposed',
            init_t=init_t,
            init_r=init_r,
            channel_params=channel_params
        )
        
        final_capacity = ao_res['capacity']
        final_positions = {'t': ao_res['t'], 'r': ao_res['r']}
        
        return final_capacity, final_positions, cma_capacity
