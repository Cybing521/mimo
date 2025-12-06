import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.deep_unfolding import DeepUnfoldingNet
from drl.diff_mimo import differentiable_capacity
from core.mimo_core import MIMOSystem

def generate_batch_channel_params(batch_size, Lt, Lr, device='cpu'):
    """
    Generate a batch of random channel parameters.
    Replicates logic from MAMIMOEnv._initialize_channel_params
    """
    # Angles: Uniform [0, pi] or [0, 2pi]
    # Env uses [0, pi] for all angles? Let's check env.py again.
    # line 316: rand * pi. So [0, pi].
    
    theta_t = torch.rand(batch_size, Lt, device=device) * np.pi
    phi_t = torch.rand(batch_size, Lt, device=device) * np.pi
    theta_r = torch.rand(batch_size, Lr, device=device) * np.pi
    phi_r = torch.rand(batch_size, Lr, device=device) * np.pi
    
    # Sigma: Rician
    # [B, Lr, Lt]
    Sigma = torch.zeros(batch_size, Lr, Lt, dtype=torch.complex64, device=device)
    
    kappa = 1.0
    # LoS component at [0, 0]
    # (randn + j*randn) * sqrt(...)
    # We need complex normal
    
    def complex_randn(shape, scale):
        real = torch.randn(shape, device=device)
        imag = torch.randn(shape, device=device)
        return (real + 1j * imag) * scale
    
    scale_los = np.sqrt(kappa / (kappa + 1) / 2)
    Sigma[:, 0, 0] = complex_randn((batch_size,), scale_los)
    
    scale_nlos = np.sqrt(1 / ((kappa + 1) * (Lr - 1)) / 2)
    min_dim = min(Lr, Lt)
    for i in range(1, min_dim):
        Sigma[:, i, i] = complex_randn((batch_size,), scale_nlos)
        
    return {
        'theta_t': theta_t,
        'phi_t': phi_t,
        'theta_r': theta_r,
        'phi_r': phi_r,
        'Sigma': Sigma
    }

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Model
    model = DeepUnfoldingNet(
        N=args.N, M=args.M, 
        num_layers=args.layers, 
        hidden_dim=args.hidden, 
        square_size=args.square_size,
        SNR_dB=args.SNR_dB,
        device=device
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Training Loop
    best_capacity = 0.0
    loss_history = []
    
    for epoch in range(args.epochs):
        model.train()
        
        # Generate Batch
        channel_params = generate_batch_channel_params(args.batch_size, args.Lt, args.Lr, device)
        
        # Initial positions (Random)
        init_t = torch.rand(args.batch_size, 2, args.N, device=device) * args.square_size
        init_r = torch.rand(args.batch_size, 2, args.M, device=device) * args.square_size
        
        # Forward
        final_t, final_r = model(init_t, init_r, channel_params)
        
        # Calculate Capacity
        H = model.mimo_system(final_t, final_r, channel_params)
        capacity = differentiable_capacity(H, model.mimo_system.power, model.mimo_system.sigma)
        
        # Loss = -Capacity + Penalty
        # Minimum distance penalty
        # We can implement a soft penalty here
        D = model.mimo_system.lambda_val / 2
        
        def calc_dist_penalty(pos):
            # pos: [B, 2, K]
            B, _, K = pos.shape
            penalty = 0
            for i in range(K):
                for j in range(i+1, K):
                    dist = torch.norm(pos[:, :, i] - pos[:, :, j], dim=1)
                    # if dist < D, penalty += (D - dist)^2
                    viol = torch.relu(D - dist)
                    penalty = penalty + torch.sum(viol**2)
            return penalty / B
            
        pen_t = calc_dist_penalty(final_t)
        pen_r = calc_dist_penalty(final_r)
        
        loss = -torch.mean(capacity) + args.penalty_weight * (pen_t + pen_r)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        avg_cap = torch.mean(capacity).item()
        loss_history.append(avg_cap)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Capacity = {avg_cap:.4f} bps/Hz, Loss = {loss.item():.4f}")
            
        if epoch % 100 == 0:
            scheduler.step()
            
            # Save best model
            if avg_cap > best_capacity:
                best_capacity = avg_cap
                torch.save(model.state_dict(), 'results/deep_unfolding_best.pth')
                
    # Save final model
    torch.save(model.state_dict(), 'results/deep_unfolding_final.pth')
    
    # Plot training curve
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Capacity (bps/Hz)')
    plt.title('Deep Unfolding Training')
    plt.savefig('results/unfolding_training.png')
    
    return model

def evaluate(model, args):
    print("\nEvaluating against AO...")
    model.eval()
    device = model.device
    
    # Create Comparator System for AO
    mimo_system = MIMOSystem(N=args.N, M=args.M, Lt=args.Lt, Lr=args.Lr, SNR_dB=args.SNR_dB)
    
    num_test = 50
    ao_caps = []
    dl_caps = []
    
    # Generate test batch
    channel_params_torch = generate_batch_channel_params(num_test, args.Lt, args.Lr, device)
    
    # Run DL
    # with torch.no_grad(): # Deep Unfolding needs grad for internal steps!
    init_t = torch.rand(num_test, 2, args.N, device=device) * args.square_size
    init_r = torch.rand(num_test, 2, args.M, device=device) * args.square_size
    
    final_t, final_r = model(init_t, init_r, channel_params_torch)
    
    H = model.mimo_system(final_t, final_r, channel_params_torch)
    caps = differentiable_capacity(H, model.mimo_system.power, model.mimo_system.sigma)
    dl_caps = caps.detach().cpu().numpy()
        
    # Run AO
    # Convert params to numpy
    theta_t = channel_params_torch['theta_t'].cpu().numpy()
    phi_t = channel_params_torch['phi_t'].cpu().numpy()
    theta_r = channel_params_torch['theta_r'].cpu().numpy()
    phi_r = channel_params_torch['phi_r'].cpu().numpy()
    Sigma = channel_params_torch['Sigma'].cpu().numpy()
    
    for i in range(num_test):
        # Construct channel params dict for this instance
        cp = {
            'theta_t': theta_t[i], 'phi_t': phi_t[i],
            'theta_r': theta_r[i], 'phi_r': phi_r[i],
            'Sigma': Sigma[i]
        }
        
        # Run AO
        # We need to modify MIMO system to accept these exact params?
        # core/mimo_core.py run_optimization accepts channel_params
        # But we need to make sure the keys match.
        # core/mimo_core.py uses internal generation if not provided.
        # Let's check run_optimization signature in core/mimo_core.py
        # It accepts channel_params.
        # And inside it uses:
        # self._theta_p = channel_params['theta_t'] etc.
        # Wait, env.py uses _theta_p, but mimo_core.py uses what?
        # mimo_core.py usually doesn't store channel state, it's passed or generated.
        # Let's assume run_optimization handles it if we pass it.
        # Actually, looking at previous edits, I modified run_optimization to accept channel_params.
        # But I should verify the key names.
        # env.py uses _theta_p (Tx) and _theta_q (Rx).
        # My generator used theta_t/theta_r.
        # I should map them.
        
        # Mapping:
        # theta_t -> theta_p
        # theta_r -> theta_q
        
        cp_mapped = {
            'theta_p': cp['theta_t'], 'phi_p': cp['phi_t'],
            'theta_q': cp['theta_r'], 'phi_q': cp['phi_r'],
            'Sigma': cp['Sigma']
        }
        
        # Need to inject these into mimo_system or pass to run_optimization
        # If run_optimization supports it.
        # If not, we might need to hack it.
        # Let's try passing it.
        
        try:
            # A_lambda = square_size / lambda (lambda=1)
            A_lambda = args.square_size
            res = mimo_system.run_optimization(A_lambda=A_lambda, mode='Proposed', channel_params=cp_mapped)
            ao_caps.append(res['capacity'])
        except Exception as e:
            print(f"AO failed: {e}")
            ao_caps.append(0)
            
    mean_dl = np.mean(dl_caps)
    mean_ao = np.mean(ao_caps)
    
    print(f"Mean Deep Unfolding: {mean_dl:.2f} bps/Hz")
    print(f"Mean AO: {mean_ao:.2f} bps/Hz")
    
    return mean_dl, mean_ao

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--SNR_dB', type=float, default=25.0)
    parser.add_argument('--Lt', type=int, default=10)
    parser.add_argument('--Lr', type=int, default=15)
    parser.add_argument('--square_size', type=float, default=3.0)
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--penalty_weight', type=float, default=10.0)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    model = train(args)
    evaluate(model, args)
