import torch
import torch.nn as nn
import numpy as np

class DiffMIMOSystem(nn.Module):
    """
    Differentiable MIMO System Model using PyTorch.
    Supports batch processing for Deep Unfolding / RL training.
    """
    def __init__(self, N=4, M=4, Lt=10, Lr=15, SNR_dB=15, lambda_val=1, device='cpu'):
        super().__init__()
        self.N = N
        self.M = M
        self.Lt = Lt
        self.Lr = Lr
        self.lambda_val = lambda_val
        self.device = device
        
        self.power = 10.0
        self.SNR_linear = 10**(SNR_dB / 10)
        self.sigma = self.power / self.SNR_linear
        
    def compute_field_response(self, pos, theta, phi):
        """
        Compute field response vector.
        Args:
            pos: (batch, 2, num_antennas)
            theta: (batch, num_paths)
            phi: (batch, num_paths)
        Returns:
            response: (batch, num_paths, num_antennas) complex tensor
        """
        # pos: [B, 2, N_ant]
        # theta, phi: [B, L]
        
        B, _, N_ant = pos.shape
        L = theta.shape[1]
        
        # Expand dimensions for broadcasting
        # x: [B, 1, N_ant]
        x = pos[:, 0:1, :]
        y = pos[:, 1:2, :]
        
        # theta, phi: [B, L, 1]
        theta = theta.unsqueeze(2)
        phi = phi.unsqueeze(2)
        
        # Phase calculation
        # phase = k * (x * sin(theta)cos(phi) + y * cos(theta))
        k = 2 * np.pi / self.lambda_val
        
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        cos_phi = torch.cos(phi)
        
        # [B, L, N_ant]
        phase = k * (x * sin_theta * cos_phi + y * cos_theta)
        
        return torch.exp(1j * phase)

    def forward(self, t_pos, r_pos, channel_params):
        """
        Compute Channel Matrix H.
        Args:
            t_pos: (batch, 2, N)
            r_pos: (batch, 2, M)
            channel_params: dict containing:
                - theta_t, phi_t: (batch, Lt) - Tx angles
                - theta_r, phi_r: (batch, Lr) - Rx angles
                - Sigma: (batch, Lr, Lt) - Path gain matrix
        Returns:
            H: (batch, M, N) complex channel matrix
        """
        theta_t = channel_params['theta_t']
        phi_t = channel_params['phi_t']
        theta_r = channel_params['theta_r']
        phi_r = channel_params['phi_r']
        Sigma = channel_params['Sigma']
        
        # Calculate G (Tx array response): [B, Lt, N]
        # Columns are field responses for each antenna
        # My compute_field_response returns [B, L, N_ant] which is exactly G
        G = self.compute_field_response(t_pos, theta_t, phi_t)
        
        # Calculate F (Rx array response): [B, Lr, M]
        F = self.compute_field_response(r_pos, theta_r, phi_r)
        
        # H = F^H @ Sigma @ G
        # F^H: [B, M, Lr]
        F_H = F.conj().transpose(1, 2)
        
        # Sigma: [B, Lr, Lt]
        # G: [B, Lt, N]
        
        # H: [B, M, N]
        H = F_H @ Sigma @ G
        # print(f"DEBUG: G.requires_grad={G.requires_grad}, H.requires_grad={H.requires_grad}")
        return H

def differentiable_capacity(H, power, sigma, water_filling=True):
    """
    Compute MIMO Capacity in a differentiable way.
    Args:
        H: (batch, M, N) complex channel matrix
        power: float, total power
        sigma: float, noise variance
        water_filling: bool, whether to use water filling (True) or EPA (False)
    Returns:
        capacity: (batch,) float tensor
    """
    # H @ H^H: [B, M, M]
    H_HH = H @ H.conj().transpose(1, 2)
    
    # Eigenvalues (real, non-negative for Hermitian matrix)
    # torch.linalg.eigvalsh returns eigenvalues in ascending order
    eigvals = torch.linalg.eigvalsh(H_HH)
    
    # Clamp to avoid numerical issues with log(0)
    eigvals = torch.clamp(eigvals, min=1e-12)
    
    if not water_filling:
        # Equal Power Allocation (EPA)
        # Power per mode = P / min(M, N)
        # Usually P / rank, but rank is hard to differentiate.
        # Standard EPA: P / M (or N)
        n_modes = H.shape[1] # M
        p_alloc = power / n_modes
        snr = p_alloc * eigvals / sigma
        capacity = torch.sum(torch.log2(1 + snr), dim=1)
        return capacity
    
    else:
        # Differentiable Batch Water Filling
        # We need to find mu such that sum(max(0, mu - 1/lambda)) = P
        # This is hard to solve exactly with gradients.
        # Alternative: Softmax-based allocation or iterative approach.
        # For Deep Unfolding, exact WF is often replaced by EPA or a learned allocation.
        # However, let's try to implement a "Soft Water Filling" or exact WF if possible.
        
        # Exact WF logic (batch version):
        # 1. Sort inverse gains: inv_g = sigma / eigvals
        # 2. Iterate to find active set
        
        # Since we want gradients w.r.t H (eigvals), we need the operations to be differentiable.
        # The selection of "active set" is non-differentiable (discrete).
        # But once active set is chosen, the power allocation is linear in eigvals.
        # Gradients can flow through the "selected" path.
        
        # Let's implement a simplified iterative WF that works in batches.
        # Sort eigenvalues descending (eigvalsh is ascending)
        lambdas = torch.flip(eigvals, dims=[1]) # [B, M] largest first
        inv_lambdas = sigma / lambdas
        
        B, M = lambdas.shape
        
        # We try all possible numbers of active modes k = 1...M
        # For each k, calculate potential mu
        # mu_k = (P + sum(inv_lambdas[:k])) / k
        # Check if valid: mu_k > inv_lambdas[k-1] (and mu_k <= inv_lambdas[k] if k < M?)
        # Actually, standard condition: mu > inv_lambda_i for i in active.
        
        # Let's compute cumulative sum of inv_lambdas
        cum_inv = torch.cumsum(inv_lambdas, dim=1) # [B, M]
        k_range = torch.arange(1, M + 1, device=H.device).unsqueeze(0) # [1, M]
        
        mu_candidates = (power + cum_inv) / k_range # [B, M]
        
        # Check validity: mu_k > inv_lambdas[k-1]
        # And also we want the largest k that satisfies this.
        # Or simply: power allocated to k-th mode must be non-negative.
        # p_k = mu_k - inv_lambdas[k-1] >= 0
        
        p_candidates = mu_candidates - inv_lambdas # [B, M]
        
        # We want to find the largest k where p_candidates[k] >= 0
        # Since lambdas are sorted desc, inv_lambdas are sorted asc.
        # So p_candidates should be decreasing? No.
        # Actually, water-filling usually fills the strongest channels first.
        
        # Let's use a mask.
        valid_mask = (p_candidates >= 0).float() # [B, M]
        
        # We want to select the "last" valid k.
        # But this discrete selection breaks gradients?
        # Actually, `torch.max` or indexing is fine.
        # But we need to be careful.
        
        # A robust way for gradients:
        # Just assume all are active, compute powers, clip negative to 0, re-normalize?
        # No, that's iterative.
        
        # Let's stick to the exact solution but implemented with masking.
        # Find the index of the last positive value in p_candidates?
        # Actually, if k is valid, then k-1 is also valid usually.
        # So we can sum the mask to get k_star.
        
        k_star = torch.sum(valid_mask, dim=1, keepdim=True) # [B, 1]
        # Clamp k_star to at least 1
        k_star = torch.clamp(k_star, min=1)
        
        # Now re-compute mu using k_star
        # We need to gather the cumulative sum at k_star - 1
        # cum_inv_k = cum_inv.gather(1, k_star.long() - 1)
        
        # Indexing needs long
        idx = (k_star - 1).long()
        cum_inv_selected = torch.gather(cum_inv, 1, idx) # [B, 1]
        
        mu_star = (power + cum_inv_selected) / k_star # [B, 1]
        
        # Calculate powers
        powers = torch.relu(mu_star - inv_lambdas) # [B, M]
        
        # Capacity
        snr = powers * lambdas / sigma
        capacity = torch.sum(torch.log2(1 + snr), dim=1)
        
        return capacity
