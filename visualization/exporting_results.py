"""
Export results module - CSV data export
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os


def export_all_results(simulation_results, cvr_results, cvr_params, heston_params,
                       sim_params, gbm_params, simulate_gbm, output_dir='cvr_output'):
    """
    Export all simulation data to CSV files

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
    output_dir : str
        Output directory for CSV files
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f"{output_dir}/Outputs_{timestamp}")
    print(f"\n{'=' * 70}")
    print(f"EXPORTING RESULTS TO: {output_dir}/Outputs_{timestamp}/")
    print(f"{'=' * 70}")

    # 1. Export summary statistics
    export_summary(cvr_results, cvr_params, heston_params, sim_params, gbm_params, simulate_gbm,
                   f"{output_dir}/Outputs_{timestamp}/summary_{timestamp}.csv")

    # 2. Export all payoffs
    export_payoffs(cvr_results, f"{output_dir}/Outputs_{timestamp}/payoffs_{timestamp}.csv")

    # 3. Export sample price paths
    export_price_paths(simulation_results, f"{output_dir}/Outputs_{timestamp}/price_paths_{timestamp}.csv")

    # 4. Export sample variance paths
    if not(simulate_gbm):
        export_variance_paths(simulation_results, f"{output_dir}/Outputs_{timestamp}/variance_paths_{timestamp}.csv")

    # 5. Export rolling averages
    export_rolling_averages(simulation_results, f"{output_dir}/Outputs_{timestamp}/rolling_avgs_{timestamp}.csv")

    # 6. Export trigger times and types
    export_trigger_data(simulation_results, cvr_results, f"{output_dir}/Outputs_{timestamp}/triggers_{timestamp}.csv")

    # 7. Export percentile analysis
    export_percentiles(cvr_results, f"{output_dir}/Outputs_{timestamp}/percentiles_{timestamp}.csv")

    print(f"\nAll data exported to {output_dir}/")
    print(f"{'=' * 70}\n")


def export_summary(cvr_results, cvr_params, heston_params, sim_params, gbm_params, simulate_gbm, filename):
    """Export summary statistics"""
    if simulate_gbm:

        summary_data = {
            'Metric': [
                'CVR Fair Value',
                'Barrier Hit Probability',
                'CoC Probability',
                'Total Payout Probability',
                'Avg Payoff (if triggered)',
                'Mean Payoff (all paths)',
                'Median Payoff',
                'Std Dev Payoff',
                'Min Payoff',
                'Max Payoff',
                '',
                'Current Stock Price',
                'Barrier Level',
                'Payoff Shares',
                'Time to Expiration',
                'Risk-Free Rate',
                'Dividend Yield',
                '',
                'Constant Volatility',
                '',
                'Number of Paths',
                'CoC Intensity',
                'CoC Premium'
            ],
            'Value': [
                cvr_results['cvr_value'],
                cvr_results['barrier_hit_prob'],
                cvr_results['coc_prob'],
                cvr_results['barrier_hit_prob'] + cvr_results['coc_prob'],
                cvr_results['avg_payoff_if_triggered'],
                cvr_results['payoffs'].mean(),
                np.median(cvr_results['payoffs']),
                cvr_results['payoffs'].std(),
                cvr_results['payoffs'].min(),
                cvr_results['payoffs'].max(),
                '',
                cvr_params.S0,
                cvr_params.barrier,
                cvr_params.payoff_shares,
                cvr_params.T,
                cvr_params.r,
                cvr_params.q,
                '',
                gbm_params.sigma,
                '',
                sim_params.n_paths,
                sim_params.coc_intensity,
                sim_params.coc_premium
            ]
        }
    else:
        summary_data = {
            'Metric': [
                'CVR Fair Value',
                'Barrier Hit Probability',
                'CoC Probability',
                'Total Payout Probability',
                'Avg Payoff (if triggered)',
                'Mean Payoff (all paths)',
                'Median Payoff',
                'Std Dev Payoff',
                'Min Payoff',
                'Max Payoff',
                '',
                'Current Stock Price',
                'Barrier Level',
                'Payoff Shares',
                'Time to Expiration',
                'Risk-Free Rate',
                'Dividend Yield',
                '',
                'Initial Variance',
                'Mean Reversion Speed',
                'Long-run Variance',
                'Vol of Vol',
                'Correlation',
                '',
                'Number of Paths',
                'CoC Intensity',
                'CoC Premium'
            ],
            'Value': [
                cvr_results['cvr_value'],
                cvr_results['barrier_hit_prob'],
                cvr_results['coc_prob'],
                cvr_results['barrier_hit_prob'] + cvr_results['coc_prob'],
                cvr_results['avg_payoff_if_triggered'],
                cvr_results['payoffs'].mean(),
                np.median(cvr_results['payoffs']),
                cvr_results['payoffs'].std(),
                cvr_results['payoffs'].min(),
                cvr_results['payoffs'].max(),
                '',
                cvr_params.S0,
                cvr_params.barrier,
                cvr_params.payoff_shares,
                cvr_params.T,
                cvr_params.r,
                cvr_params.q,
                '',
                heston_params.v0,
                heston_params.kappa,
                heston_params.theta,
                heston_params.xi,
                heston_params.rho,
                '',
                sim_params.n_paths,
                sim_params.coc_intensity,
                sim_params.coc_premium
            ]
        }
    df = pd.DataFrame(summary_data)
    df.to_csv(filename, index=False)
    print(f" Summary: {filename}")


def export_payoffs(cvr_results, filename):
    """Export all payoff data"""

    payoffs = cvr_results['payoffs']

    df = pd.DataFrame({
        'path_id': range(len(payoffs)),
        'payoff': payoffs,
        'triggered': (payoffs > 0).astype(int)
    })

    df.to_csv(filename, index=False)
    print(f" Payoffs: {filename} ({len(payoffs):,} paths)")


def export_price_paths(simulation_results, filename, n_paths=1000):
    """Export sample price paths"""

    S_paths = simulation_results['S_paths'][:n_paths]
    n_steps = S_paths.shape[1]

    # Create column names
    columns = ['path_id'] + [f't_{i}' for i in range(n_steps)]

    # Create dataframe
    data = np.column_stack([np.arange(n_paths), S_paths])
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(filename, index=False)
    print(f" Price paths: {filename} ({n_paths} paths × {n_steps} steps)")


def export_variance_paths(simulation_results, filename, n_paths=1000):
    """Export sample variance paths"""

    v_paths = simulation_results['v_paths'][:n_paths]
    n_steps = v_paths.shape[1]

    # Create column names
    columns = ['path_id'] + [f't_{i}' for i in range(n_steps)]

    # Create dataframe
    data = np.column_stack([np.arange(n_paths), v_paths])
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(filename, index=False)
    print(f" Variance paths: {filename} ({n_paths} paths × {n_steps} steps)")


def export_rolling_averages(simulation_results, filename, n_paths=1000):
    """Export rolling average data"""

    rolling_avgs = simulation_results['rolling_avgs'][:n_paths]
    n_steps = rolling_avgs.shape[1]

    # Create column names
    columns = ['path_id'] + [f't_{i}' for i in range(n_steps)]

    # Create dataframe
    data = np.column_stack([np.arange(n_paths), rolling_avgs])
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(filename, index=False)
    print(f" Rolling averages: {filename} ({n_paths} paths × {n_steps} steps)")


def export_trigger_data(simulation_results, cvr_results, filename):
    """Export trigger event data"""

    n_paths = len(cvr_results['payoffs'])
    coc_times = simulation_results['coc_times']
    payoffs = cvr_results['payoffs']

    # Determine trigger type for each path
    trigger_types = []
    for i in range(n_paths):
        if payoffs[i] > 0:
            # Check if CoC happened
            if coc_times[i] < simulation_results['n_steps']:
                trigger_types.append('CoC')
            else:
                trigger_types.append('Barrier')
        else:
            trigger_types.append('None')

    df = pd.DataFrame({
        'path_id': range(n_paths),
        'trigger_type': trigger_types,
        'coc_time_step': coc_times,
        'payoff': payoffs,
        'triggered': (payoffs > 0).astype(int)
    })

    df.to_csv(filename, index=False)
    print(f" Trigger data: {filename}")


def export_percentiles(cvr_results, filename):
    """Export percentile analysis"""

    payoffs = cvr_results['payoffs']
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    data = {
        'percentile': percentiles,
        'payoff_value': [np.percentile(payoffs, p) for p in percentiles]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f" Percentiles: {filename}")


def export_compact_summary(cvr_results, filename='cvr_quick_summary.csv'):
    """
    Export a compact single-row summary for easy comparison across runs
    """

    payoffs = cvr_results['payoffs']

    summary = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'cvr_value': [cvr_results['cvr_value']],
        'barrier_prob': [cvr_results['barrier_hit_prob']],
        'coc_prob': [cvr_results['coc_prob']],
        'total_payout_prob': [cvr_results['barrier_hit_prob'] + cvr_results['coc_prob']],
        'mean_payoff': [payoffs.mean()],
        'median_payoff': [np.median(payoffs)],
        'std_payoff': [payoffs.std()],
        'p10': [np.percentile(payoffs, 10)],
        'p90': [np.percentile(payoffs, 90)]
    }

    df = pd.DataFrame(summary)

    # Append to file if it exists
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

    print(f"\n Compact summary appended to: {filename}")