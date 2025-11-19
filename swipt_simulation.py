import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from tqdm import tqdm
from core.swipt_core import SWIPTSystem

def generate_channel(Nt, Nr, distance, path_loss_exponent=4):
    """
    Generate Rayleigh fading channel matrix with path loss.
    H = sqrt(L) * H_small
    L = d^(-alpha)
    """
    path_loss = distance ** (-path_loss_exponent)
    # Small scale fading ~ CN(0, 1)
    H_small = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
    return np.sqrt(path_loss) * H_small

def simulate_separated(args):
    """
    Simulate Separated EH and ID Receivers scenario (Fig. 5).
    """
    print("Simulating Separated Receivers Scenario...")
    
    # System Setup
    swipt = SWIPTSystem(Nt=args.Nt, Ne=args.Ne, Ni=args.Ni)
    
    # Parameter Sets
    param_sets = [
        {'label': 'a=6400, b=0.003', 'a': 6400, 'b': 0.003},
        {'label': 'a=1500, b=0.0022', 'a': 1500, 'b': 0.0022}
    ]
    
    dist_EH = 7.0
    dist_ID = 50.0
    P_max = 2.0 # Watts
    
    results = {}
    
    for params in param_sets:
        print(f"Running for {params['label']}")
        swipt.set_nonlinear_params(params['a'], params['b'], swipt.M)
        
        # Averaging over channel realizations
        num_channels = args.trials
        
        # Rate points to sweep
        # We need to determine R_max first.
        # To avg R-E region, we usually pick fixed R points and avg E, 
        # or avg the boundary.
        # Let's pick a range of R values.
        
        R_values = np.linspace(0, 6, 15) # Adjust range as needed
        E_avg = np.zeros_like(R_values)
        
        for i, r_target in enumerate(tqdm(R_values)):
            energies = []
            for _ in range(num_channels):
                He = generate_channel(args.Nt, args.Ne, dist_EH)
                Hi = generate_channel(args.Nt, args.Ni, dist_ID)
                
                # Set M dynamically if needed (Paper sets M=Emax, 0.7Emax etc)
                # Here we just use a fixed M or determine it per channel if we want to match paper exactly
                # Paper: "Emax is obtained in terms of Emax = h1_e * P" (Linear max)
                # We'll just use the fixed M from class for now, or high enough M.
                # Actually paper varies M. Let's pick one M for simplicity or pass as arg.
                
                energy, _ = swipt.solve_separated_receivers(He, Hi, P_max, r_target)
                if energy is not None:
                    energies.append(energy)
                else:
                    energies.append(0)
            
            E_avg[i] = np.mean(energies) * 1000 # Convert to mW for plotting
            
        results[params['label']] = {'R': R_values, 'E': E_avg}
        
    return results

def simulate_colocated(args):
    """
    Simulate Co-located Receivers Scenario (Fig. 10).
    Compare TS and PS.
    """
    print("Simulating Co-located Receivers Scenario...")
    
    swipt = SWIPTSystem(Nt=args.Nt, Ne=args.Ne, Ni=args.Ni)
    
    # Parameter Sets (Using just one for clarity or both)
    # Paper uses both. Let's pick one or loop.
    # Fig 10 has curves for both param sets.
    
    param_sets = [
        {'label': 'a=6400, b=0.003', 'a': 6400, 'b': 0.003},
        {'label': 'a=1500, b=0.0022', 'a': 1500, 'b': 0.0022}
    ]
    
    dist = 8.0
    P_max = 2.0
    
    results = {}
    
    for params in param_sets:
        swipt.set_nonlinear_params(params['a'], params['b'], swipt.M)
        
        # Schemes
        schemes = ['TS', 'PS']
        
        for scheme in schemes:
            key = f"{params['label']} ({scheme})"
            print(f"Running {key}")
            
            R_values = np.linspace(0, 8, 10) # Co-located might have higher rates (shorter dist)
            E_avg = np.zeros_like(R_values)
            
            for i, r_target in enumerate(tqdm(R_values)):
                energies = []
                for _ in range(args.trials):
                    H = generate_channel(args.Nt, args.Ni, dist) # H_e = H_i = H
                    
                    if scheme == 'TS':
                        energy, _ = swipt.solve_co_located_ts(H, P_max, r_target)
                    else:
                        energy, _ = swipt.solve_co_located_ps(H, P_max, r_target)
                        
                    if energy is not None:
                        energies.append(energy)
                    else:
                        energies.append(0)
                
                E_avg[i] = np.mean(energies) * 1000 # mW
                
            results[key] = {'R': R_values, 'E': E_avg}
            
    return results

def main():
    parser = argparse.ArgumentParser(description='SWIPT Simulation')
    parser.add_argument('--mode', type=str, required=True, choices=['separated', 'colocated'],
                        help='Simulation mode: separated or colocated')
    parser.add_argument('--Nt', type=int, default=2, help='Number of Tx antennas')
    parser.add_argument('--Ne', type=int, default=2, help='Number of EH Rx antennas')
    parser.add_argument('--Ni', type=int, default=2, help='Number of ID Rx antennas')
    parser.add_argument('--trials', type=int, default=20, help='Number of channel realizations')
    
    args = parser.parse_args()
    
    if args.mode == 'separated':
        results = simulate_separated(args)
        title = f'Average R-E Region (Separated, Nt={args.Nt})'
        filename = f'swipt_separated_Nt{args.Nt}'
    else:
        results = simulate_colocated(args)
        title = f'Average R-E Region (Co-located, Nt={args.Nt})'
        filename = f'swipt_colocated_Nt{args.Nt}'
        
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 12, 
        'axes.labelsize': 14, 'legend.fontsize': 10,
        'lines.linewidth': 2, 'grid.alpha': 0.5
    })
    
    markers = ['o', 's', '^', 'v', 'D', 'x']
    
    for i, (label, data) in enumerate(results.items()):
        plt.plot(data['R'], data['E'], label=label, 
                 marker=markers[i % len(markers)], markevery=1)
        
    plt.xlabel('Information Rate (bps/Hz)')
    plt.ylabel('Harvested Energy (mW)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if not os.path.exists('results/swipt2017'):
        os.makedirs('results/swipt2017')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'results/swipt2017/{filename}_{timestamp}.png', dpi=300)
    print(f"Saved plot to results/swipt2017/{filename}_{timestamp}.png")

if __name__ == "__main__":
    main()

