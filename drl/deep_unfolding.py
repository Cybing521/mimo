import torch
import torch.nn as nn
import torch.nn.functional as F
from drl.diff_mimo import DiffMIMOSystem, differentiable_capacity

class UnfoldingLayer(nn.Module):
    """
    Single layer of the Deep Unfolding Network.
    Mimics one iteration of an optimization algorithm.
    """
    def __init__(self, num_antennas, hidden_dim=64):
        super().__init__()
        # Input: Gradient (2*N) + Current Pos (2*N)
        input_dim = 4 * num_antennas 
        
        # LSTM Cell to maintain optimization state (momentum, etc.)
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        
        # Output head: Predicts update vector delta_x
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * num_antennas),
            nn.Tanh() # Limit step size
        )
        
        # Learnable step size scaler
        self.step_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, grad, pos, h, c):
        """
        Args:
            grad: (batch, 2*N) flattened gradient
            pos: (batch, 2*N) flattened position
            h, c: LSTM states
        Returns:
            new_pos: (batch, 2*N)
            h, c: New LSTM states
        """
        # Concatenate inputs
        inp = torch.cat([grad, pos], dim=1)
        
        # LSTM update
        h_new, c_new = self.lstm_cell(inp, (h, c))
        
        # Predict update
        delta = self.output_head(h_new)
        
        # Update position
        # x_new = x + scale * delta
        # Note: We use the predicted delta directly as the update step
        # The network learns both direction and magnitude (limited by Tanh * scale)
        new_pos = pos + self.step_scale * delta
        
        return new_pos, h_new, c_new

class DeepUnfoldingNet(nn.Module):
    """
    Deep Unfolding Network for MIMO Capacity Optimization.
    Unfolds K iterations of gradient-based optimization.
    """
    def __init__(self, N, M, num_layers=10, hidden_dim=64, square_size=10.0, SNR_dB=15.0, device='cpu'):
        super().__init__()
        self.N = N
        self.M = M
        self.num_layers = num_layers
        self.square_size = square_size
        self.device = device
        
        # Differentiable Channel Model
        self.mimo_system = DiffMIMOSystem(N=N, M=M, SNR_dB=SNR_dB, device=device)
        
        # Unfolding Layers
        # We share weights across layers (Recurrent) or use separate weights?
        # "Learning to Optimize" usually uses an RNN (shared weights).
        # Let's use shared weights for parameter efficiency and generalization.
        self.layer_t = UnfoldingLayer(N, hidden_dim)
        self.layer_r = UnfoldingLayer(M, hidden_dim)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, init_t, init_r, channel_params):
        """
        Args:
            init_t: (batch, 2, N)
            init_r: (batch, 2, M)
            channel_params: dict of channel parameters
        Returns:
            final_t: (batch, 2, N)
            final_r: (batch, 2, M)
            trajectory: list of capacities (optional)
        """
        batch_size = init_t.shape[0]
        
        # Flatten positions
        curr_t = init_t.reshape(batch_size, -1) # [B, 2N]
        curr_r = init_r.reshape(batch_size, -1) # [B, 2M]
        
        # Initialize LSTM states
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h_r = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_r = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        for k in range(self.num_layers):
            # 1. Compute Gradients w.r.t current positions
            # We need to detach current positions from computation graph to avoid 
            # backpropagating through the entire history for the gradient calculation itself?
            # No, in Deep Unfolding, we WANT to backprop through the optimization path.
            # But the gradient input to the LSTM should be treated as a feature.
            # Usually: grad = dC/dx. We compute this using torch.autograd.grad
            
            # Enable grad for inputs
            t_in = curr_t.detach().requires_grad_(True)
            r_in = curr_r.detach().requires_grad_(True)
            
            # Reshape for MIMO system
            t_reshaped = t_in.reshape(batch_size, 2, self.N)
            r_reshaped = r_in.reshape(batch_size, 2, self.M)
            
            # Forward pass through channel
            H = self.mimo_system(t_reshaped, r_reshaped, channel_params)
            
            # Calculate Capacity
            cap = differentiable_capacity(H, self.mimo_system.power, self.mimo_system.sigma)
            loss = -torch.sum(cap) # Minimize negative capacity
            
            # Compute gradients
            grads = torch.autograd.grad(loss, [t_in, r_in], create_graph=True)
            grad_t = grads[0] # [B, 2N]
            grad_r = grads[1] # [B, 2M]
            
            # 2. Update Steps using LSTM
            # We pass the *current* positions and gradients.
            # Note: We use curr_t (which tracks history), not t_in (detached).
            # But wait, if we use curr_t in LSTM, and curr_t was output of previous LSTM...
            # The gradient computation `torch.autograd.grad` creates a graph connected to `t_in`.
            # `t_in` is detached. So the gradient value itself does NOT backprop to previous layers 
            # through the physics. This is standard "truncated" or "gradient-based" unfolding.
            # The flow of gradients happens through the LSTM states and the update equation.
            
            curr_t, h_t, c_t = self.layer_t(grad_t, curr_t, h_t, c_t)
            curr_r, h_r, c_r = self.layer_r(grad_r, curr_r, h_r, c_r)
            
            # 3. Constraints
            # Boundary constraint (Hard clamp)
            curr_t = torch.clamp(curr_t, 0, self.square_size)
            curr_r = torch.clamp(curr_r, 0, self.square_size)
            
            # Minimum distance constraint?
            # We rely on training loss to discourage violations.
            # Or we can add a "Repulsion" step here.
            
        # Reshape back
        final_t = curr_t.reshape(batch_size, 2, self.N)
        final_r = curr_r.reshape(batch_size, 2, self.M)
        
        return final_t, final_r
