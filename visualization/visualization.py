"""
Visualization and data export module
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from cvr_pricer.models.parameters import gbmParams


def create_visualizations(simulation_results, cvr_results, cvr_params, heston_params,gbm_params,simulate_gbm,
                         filename='cvr_results.png', n_sample_paths=100):
    """
    Create 4-panel visualization of CVR simulation results

    Parameters:
    -----------
    simulation_results : dict
        Output from simulate_heston_paths()
    cvr_results : dict
        Output from calculate_cvr_payoffs()
    cvr_params : CVRParams
        Contract parameters
    heston_params : HestonParams
        Model parameters
    filename : str
        Output filename for plot
    n_sample_paths : int
        Number of sample paths to display
    """

    S_paths = simulation_results['S_paths'][:n_sample_paths]
    if not(simulate_gbm):
        v_paths = simulation_results['v_paths'][:n_sample_paths]
    rolling_avgs = simulation_results['rolling_avgs'][:n_sample_paths]
    payoffs = cvr_results['payoffs']

    n_steps = S_paths.shape[1]
    time_grid = np.linspace(0, cvr_params.T, n_steps)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Sample price paths
    ax = axes[0, 0]
    for i in range(min(50, n_sample_paths)):
        ax.plot(time_grid, S_paths[i], alpha=0.3, linewidth=0.5)
    ax.axhline(y=cvr_params.barrier, color='red', linestyle='--', linewidth=2, label='Barrier')
    ax.axhline(y=cvr_params.S0, color='black', linestyle='-', linewidth=1, label='Initial Price')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Stock Price ($)', fontsize=11)
    ax.set_title('Sample Price Paths', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Rolling averages
    ax = axes[0, 1]
    for i in range(min(20, n_sample_paths)):
        mask = ~np.isnan(rolling_avgs[i])
        ax.plot(time_grid[mask], rolling_avgs[i][mask], alpha=0.4, linewidth=0.8)
    ax.axhline(y=cvr_params.barrier, color='red', linestyle='--', linewidth=2, label='Barrier')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('30-Day Rolling Average ($)', fontsize=11)
    ax.set_title('30-Day Rolling Averages', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Payoff distribution
    ax = axes[1,0]
    payoffs_nonzero = payoffs[payoffs > 0]
    if len(payoffs_nonzero) > 0:
        ax.hist(payoffs_nonzero, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=payoffs_nonzero.mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean = ${payoffs_nonzero.mean():.2f}')
    ax.set_xlabel('Payoff ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'CVR Payoff Distribution (n={len(payoffs_nonzero):,})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if not(simulate_gbm):

        # Plot 4: Variance paths
        ax = axes[1, 1]
        for i in range(min(50, n_sample_paths)):
            ax.plot(time_grid, v_paths[i], alpha=0.3, linewidth=0.5)
        ax.axhline(y=heston_params.theta, color='blue', linestyle='--', linewidth=2,
                  label=f'Long-run variance θ={heston_params.theta:.4f}')
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Variance', fontsize=11)
        ax.set_title('Stochastic Variance Paths (Heston)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if simulate_gbm:

        fig.delaxes(axes[1,1])

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {filename}")
    plt.close()
    """

    from matplotlib.gridspec import GridSpec


    # --- Prepare Grid Layout ---
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)

    # --- Top Row ---
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right

    # --- Bottom Row ---
    if simulate_gbm:
        ax3 = fig.add_subplot(gs[1, :])  # Spans both columns
    else:
        ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left
        ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right

    # ============ Plot 1: Sample price paths ============
    for i in range(min(50, n_sample_paths)):
        ax1.plot(time_grid, S_paths[i], alpha=0.3, linewidth=0.5)
    ax1.axhline(y=cvr_params.barrier, color='red', linestyle='--', linewidth=2, label='Barrier')
    ax1.axhline(y=cvr_params.S0, color='black', linestyle='-', linewidth=1, label='Initial Price')
    ax1.set_xlabel('Time (years)', fontsize=11)
    ax1.set_ylabel('Stock Price ($)', fontsize=11)
    ax1.set_title('Sample Price Paths', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: Rolling averages ============
    for i in range(min(20, n_sample_paths)):
        mask = ~np.isnan(rolling_avgs[i])
        ax2.plot(time_grid[mask], rolling_avgs[i][mask], alpha=0.4, linewidth=0.8)
    ax2.axhline(y=cvr_params.barrier, color='red', linestyle='--', linewidth=2, label='Barrier')
    ax2.set_xlabel('Time (years)', fontsize=11)
    ax2.set_ylabel('30-Day Rolling Average ($)', fontsize=11)
    ax2.set_title('30-Day Rolling Averages', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Payoff distribution ============
    payoffs_nonzero = payoffs[payoffs > 0]
    if len(payoffs_nonzero) > 0:
        ax3.hist(payoffs_nonzero, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.axvline(x=payoffs_nonzero.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean = ${payoffs_nonzero.mean():.2f}')
    ax3.set_xlabel('Payoff ($)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'CVR Payoff Distribution (n={len(payoffs_nonzero):,})', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # ============ Plot 4: Variance paths (only if not GBM) ============
    if not simulate_gbm:
        for i in range(min(50, n_sample_paths)):
            ax4.plot(time_grid, v_paths[i], alpha=0.3, linewidth=0.5)
        ax4.axhline(y=heston_params.theta, color='blue', linestyle='--', linewidth=2,
                    label=f'Long-run variance θ={heston_params.theta:.4f}')
        ax4.set_xlabel('Time (years)', fontsize=11)
        ax4.set_ylabel('Variance', fontsize=11)
        ax4.set_title('Stochastic Variance Paths (Heston)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {filename}")
    plt.close()


def export_results(simulation_results, cvr_results, cvr_params, heston_params, sim_params, gbm_params, simulate_gbm,
                  json_file='cvr_results.json', csv_file='cvr_payoffs.csv'):
    """
    Export results to JSON summary and CSV payoff data

    Parameters:
    -----------
    simulation_results : dict
        Simulation paths and data
    cvr_results : dict
        CVR valuation results
    cvr_params : CVRParams
        Contract parameters
    heston_params : HestonParams
        Model parameters
    sim_params : SimulationParams
        Simulation parameters
    json_file : str
        JSON output filename
    csv_file : str
        CSV output filename
    """

    # Create JSON summary
    if simulate_gbm:

        summary = {
            'timestamp': datetime.now().isoformat(),
            'cvr_parameters': {
                'S0': cvr_params.S0,
                'barrier': cvr_params.barrier,
                'payoff_shares': cvr_params.payoff_shares,
                'T': cvr_params.T,
                'r': cvr_params.r,
                'q': cvr_params.q,
                'rolling_window': cvr_params.rolling_window,
                'delivery_days': cvr_params.delivery_days
            },
            'gbm_params':{
                'sigma': gbm_params.sigma
            },
            'simulation_parameters': {
                'n_paths': sim_params.n_paths,
                'coc_intensity': sim_params.coc_intensity,
                'coc_premium': sim_params.coc_premium,
                'seed': sim_params.seed
            },
            'results': {
                'cvr_value': float(cvr_results['cvr_value']),
                'barrier_hit_prob': float(cvr_results['barrier_hit_prob']),
                'coc_prob': float(cvr_results['coc_prob']),
                'avg_payoff_if_triggered': float(cvr_results['avg_payoff_if_triggered'])
            },
            'statistics': {
                'mean_payoff': float(cvr_results['payoffs'].mean()),
                'std_payoff': float(cvr_results['payoffs'].std()),
                'min_payoff': float(cvr_results['payoffs'].min()),
                'max_payoff': float(cvr_results['payoffs'].max()),
                'median_payoff': float(np.median(cvr_results['payoffs']))
            }
        }

    else:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'cvr_parameters': {
                'S0': cvr_params.S0,
                'barrier': cvr_params.barrier,
                'payoff_shares': cvr_params.payoff_shares,
                'T': cvr_params.T,
                'r': cvr_params.r,
                'q': cvr_params.q,
                'rolling_window': cvr_params.rolling_window,
                'delivery_days': cvr_params.delivery_days
            },

            'heston_parameters': {
                'v0': heston_params.v0,
                'kappa': heston_params.kappa,
                'theta': heston_params.theta,
                'xi': heston_params.xi,
                'rho': heston_params.rho
            },
            'simulation_parameters': {
                'n_paths': sim_params.n_paths,
                'coc_intensity': sim_params.coc_intensity,
                'coc_premium': sim_params.coc_premium,
                'seed': sim_params.seed
            },
            'results': {
                'cvr_value': float(cvr_results['cvr_value']),
                'barrier_hit_prob': float(cvr_results['barrier_hit_prob']),
                'coc_prob': float(cvr_results['coc_prob']),
                'avg_payoff_if_triggered': float(cvr_results['avg_payoff_if_triggered'])
            },
            'statistics': {
                'mean_payoff': float(cvr_results['payoffs'].mean()),
                'std_payoff': float(cvr_results['payoffs'].std()),
                'min_payoff': float(cvr_results['payoffs'].min()),
                'max_payoff': float(cvr_results['payoffs'].max()),
                'median_payoff': float(np.median(cvr_results['payoffs']))
            }
        }


    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved: {json_file}")

    # Export payoffs to CSV
    payoffs = cvr_results['payoffs']
    np.savetxt(csv_file, payoffs, delimiter=',', header='payoff', comments='')
    print(f"CSV payoffs saved: {csv_file}")

    # Export sample paths to CSV (first 100 paths)
    sample_paths_file = 'cvr_sample_paths.csv'
    S_sample = simulation_results['S_paths'][:100]
    np.savetxt(sample_paths_file, S_sample, delimiter=',')
    print(f"Sample paths saved: {sample_paths_file}")


def print_detailed_summary(cvr_results, cvr_params):
    """
    Print detailed statistical summary
    """
    payoffs = cvr_results['payoffs']

    print("\n" + "="*70)
    print("DETAILED STATISTICAL SUMMARY")
    print("="*70)

    print(f"\nTrigger Analysis:")
    print(f"   Paths with payout: {(payoffs > 0).sum():,} ({(payoffs > 0).mean():.1%})")
    print(f"   Paths with no payout: {(payoffs == 0).sum():,} ({(payoffs == 0).mean():.1%})")

    print(f"\nPayoff Statistics:")
    print(f"   Mean: ${payoffs.mean():.2f}")
    print(f"   Median: ${np.median(payoffs):.2f}")
    print(f"   Std Dev: ${payoffs.std():.2f}")
    print(f"   Min: ${payoffs.min():.2f}")
    print(f"   Max: ${payoffs.max():.2f}")


    if (payoffs > 0).any():
        print(f"\n Conditional on Positive Payout:")
        positive_payoffs = payoffs[payoffs > 0]
        print(f"   Mean: ${positive_payoffs.mean():.2f}")
        print(f"   Median: ${np.median(positive_payoffs):.2f}")
        print(f"   Std Dev: ${positive_payoffs.std():.2f}")

    print("="*70)