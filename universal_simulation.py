import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from core.mimo_core import MIMOSystem
from utils.wandb_utils import init_wandb, log_image, log_line_series, log_metrics, ensure_wandb_api_key

def run_single_simulation(args):
    """
    Execute a single simulation trial.
    args is a tuple: (trial_idx, params_dict, variable_param, variable_value, mode)
    """
    trial_idx, params, var_param, var_value, mode = args
    
    # Create a copy of params to modify the variable parameter
    current_params = params.copy()
    
    # Handle special case for antennas where N and M usually change together in these plots
    if var_param == 'antennas':
        current_params['N'] = int(var_value)
        current_params['M'] = int(var_value)
    else:
        current_params[var_param] = var_value
        
    # Initialize System
    try:
        mimo = MIMOSystem(
            N=int(current_params['N']),
            M=int(current_params['M']),
            Lt=int(current_params['Lt']),
            Lr=int(current_params['Lr']),
            SNR_dB=current_params['SNR'],
            lambda_val=1.0 
        )
        
        # Run Optimization
        results = mimo.run_optimization(
            A_lambda=current_params['A'], 
            mode=mode
        )
        return results['capacity']
    except Exception as e:
        # print(f"Error in trial {trial_idx}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Universal MIMO Simulation Script')
    
    # Simulation Control
    parser.add_argument('--trials', type=int, default=50, help='Number of Monte Carlo trials per point')
    parser.add_argument('--cores', type=int, default=4, help='Number of CPU cores to use')
    parser.add_argument('--modes', nargs='+', default=['Proposed', 'FPA'], 
                        help='Modes to simulate (e.g. Proposed RMA TMA FPA)')
    
    # Sweep Configuration
    parser.add_argument('--sweep_param', type=str, required=True, 
                        choices=['SNR', 'antennas', 'A', 'Lt', 'Lr'],
                        help='The parameter to vary (x-axis)')
    parser.add_argument('--range', nargs=3, type=float, required=True,
                        metavar=('START', 'STOP', 'STEP'),
                        help='Sweep range: start stop step (e.g., -15 16 5)')
    
    # Fixed Parameters (Default values based on paper)
    parser.add_argument('--N', type=int, default=6, help='Number of Tx antennas (fixed if not sweeping)')
    parser.add_argument('--M', type=int, default=6, help='Number of Rx antennas (fixed if not sweeping)')
    parser.add_argument('--Lt', type=int, default=5, help='Number of Tx paths (fixed if not sweeping)')
    parser.add_argument('--Lr', type=int, default=5, help='Number of Rx paths (fixed if not sweeping)')
    parser.add_argument('--A', type=float, default=4.0, help='Region size in wavelengths (fixed if not sweeping)')
    parser.add_argument('--SNR', type=float, default=5.0, help='SNR in dB (fixed if not sweeping)')
    
    # WandB logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='ma-mimo', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom WandB run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'], help='WandB run mode')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='Optional WandB tags')
    
    args = parser.parse_args()
    
    # Construct Range
    start, stop, step = args.range
    if args.sweep_param in ['antennas', 'Lt', 'Lr']:
        # Integer range for discrete parameters
        variable_values = np.arange(int(start), int(stop), int(step))
    else:
        # Float range for continuous parameters
        variable_values = np.arange(start, stop, step)
        
    # Base Parameters
    base_params = {
        'N': args.N, 'M': args.M,
        'Lt': args.Lt, 'Lr': args.Lr,
        'A': args.A, 'SNR': args.SNR
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    ensure_wandb_api_key()
    wandb_run = init_wandb(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or f"universal_{args.sweep_param}_{timestamp}",
        mode=args.wandb_mode,
        config={
            'base_params': base_params,
            'sweep_param': args.sweep_param,
            'range': variable_values.tolist(),
            'modes': args.modes,
            'trials': args.trials,
            'cores': args.cores,
        },
        tags=args.wandb_tags,
    )
    
    print(f"\n{'='*70}")
    print(f"Universal MIMO Simulation")
    print(f"{'='*70}")
    print(f"Sweeping: {args.sweep_param}")
    print(f"Range: {variable_values}")
    print(f"Fixed Parameters: {base_params}")
    print(f"Modes: {args.modes}")
    print(f"Trials: {args.trials}, Cores: {args.cores}")
    print(f"{'='*70}\n")
    
    results_data = {'variable_values': variable_values.tolist()}
    
    # Create Plot
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 12, 'axes.labelsize': 14, 'legend.fontsize': 12,
        'lines.linewidth': 2, 'lines.markersize': 8,
        'xtick.direction': 'in', 'ytick.direction': 'in'
    })
    plt.figure(figsize=(8, 6))
    
    styles = {
        'Proposed': {'marker': 'o', 'color': '#1f77b4'},
        'RMA': {'marker': 'v', 'color': '#2ca02c'},
        'TMA': {'marker': '^', 'color': '#d62728'},
        'FPA': {'marker': 's', 'color': '#ff7f0e'},
        'MA-FD': {'marker': 'o', 'color': '#1f77b4'}, # Alias for Proposed in Fig 9 context
        'FPA-FD': {'marker': 's', 'color': '#ff7f0e'} # Alias for FPA in Fig 9 context
    }

    for mode in args.modes:
        print(f"Simulating Mode: {mode}")
        y_means = []
        
        # Prepare arguments for all trials across all variable values
        # However, to show progress per value, we loop over values
        
        for val in variable_values:
            # Prepare list of arguments for multiprocessing
            # mode logic mapping for Fig 9 aliases
            actual_mode = 'Proposed' if mode == 'MA-FD' else ('FPA' if mode == 'FPA-FD' else mode)
            
            pool_args = [(t, base_params, args.sweep_param, val, actual_mode) for t in range(args.trials)]
            
            with Pool(processes=args.cores) as pool:
                # Use tqdm to show progress bar for trials
                # total=args.trials ensures the bar is scaled correctly
                capacities = list(tqdm(pool.imap(run_single_simulation, pool_args), 
                                     total=args.trials, 
                                     desc=f"  Computing {args.sweep_param}={val}",
                                     unit="trial"))
            
            valid_caps = [c for c in capacities if c is not None]
            avg_cap = np.mean(valid_caps) if valid_caps else 0
            y_means.append(avg_cap)
            
            print(f"  {args.sweep_param}={val:.2f} -> Rate: {avg_cap:.4f} bps/Hz")
            log_metrics(
                wandb_run,
                {
                    'mode': mode,
                    args.sweep_param: float(val),
                    'capacity_bps_hz': float(avg_cap),
                },
            )
            
        results_data[mode] = y_means
        
        # Plot line
        style = styles.get(mode, {'marker': 'x', 'color': 'black'})
        plt.plot(variable_values, y_means, label=mode,
                 marker=style['marker'], color=style['color'],
                 linestyle='-', markerfacecolor='none', markeredgewidth=1.5)
    
    if wandb_run is not None:
        series_dict = {mode: results_data[mode] for mode in args.modes if mode in results_data}
        log_line_series(
            wandb_run,
            variable_values.tolist(),
            series_dict,
            title=f'Achievable Rate vs {args.sweep_param}',
            x_name=args.sweep_param,
            key='plots/universal_capacity',
        )

    # Labeling
    if args.sweep_param == 'SNR':
        plt.xlabel('SNR (dB)')
    elif args.sweep_param == 'antennas':
        plt.xlabel('Number of Antennas (N=M)')
    elif args.sweep_param == 'A':
        plt.xlabel(r'Region Size $A$ ($\lambda$)')
    elif args.sweep_param == 'Lt' or args.sweep_param == 'Lr':
         plt.xlabel('Number of Paths')
    else:
        plt.xlabel(args.sweep_param)
        
    plt.ylabel('Achievable Rate (bps/Hz)')
    plt.title(f'Achievable Rate vs {args.sweep_param}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Save Results
    if not os.path.exists('results'): os.makedirs('results')
    
    # Save Image
    img_filename = f'results/universal_sweep_{args.sweep_param}_{timestamp}.png'
    plt.savefig(img_filename, dpi=300)
    
    # Save Data (JSON)
    json_filename = f'results/universal_sweep_{args.sweep_param}_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print(f"\nResults saved to:\n  {img_filename}\n  {json_filename}")
    
    if wandb_run is not None:
        log_image(
            wandb_run,
            key='plots/universal_curve',
            image_path=img_filename,
            caption=f'Achievable rate vs {args.sweep_param}',
        )
        summary_metrics = {
            f'{mode}/mean_capacity': float(np.mean(results_data[mode])) if results_data[mode] else 0.0
            for mode in args.modes if mode in results_data
        }
        log_metrics(wandb_run, summary_metrics)
        wandb_run.finish()

if __name__ == "__main__":
    main()

