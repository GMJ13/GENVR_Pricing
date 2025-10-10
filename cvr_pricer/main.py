"""
Main script - Gen Digital CVR Valuation
"""

from multiprocessing import cpu_count
from models.parameters import HestonParams, CVRParams, SimulationParams
from simulation.heston_simulator import simulate_heston_paths
from pricing.payoff_calculator import calculate_cvr_payoffs
from visualization.visualization import create_visualizations, print_detailed_summary
from visualization.exporting_results import export_all_results, export_compact_summary


def main():
    """Run CVR valuation"""

    print("=" * 70)
    print("GEN DIGITAL CVR VALUATION")
    print("=" * 70)

    # Step 1: Define parameters
    print("\nStep 1: Setting up parameters...")

    cvr_params = CVRParams(
        S0=26.54,
        barrier=37.50,
        payoff_shares=0.7546,
        T=1.52,
        r=0.045,
        q=0.0185,
        rolling_window=30,
        delivery_days=12
    )

    heston_params = HestonParams(
        v0=0.0444,
        kappa=7.53,
        theta=0.0956,
        xi=0.72,
        rho=-0.13
    )

    sim_params = SimulationParams(
        n_paths=750000,
        coc_intensity=0,
        coc_premium=1.25,
        seed=22,
        n_jobs= 8
    )

    print(f"  Stock Price: ${cvr_params.S0:.2f}")
    print(f"  Barrier: ${cvr_params.barrier:.2f}")
    print(f"  Paths: {sim_params.n_paths:,}")
    print(f"  Available CPU Cores: {cpu_count()}",f"Using: {sim_params.n_jobs} cores currently")

    # Step 2: Simulate paths
    print("\nStep 2: Simulating Heston paths in parallel...")
    simulation_results = simulate_heston_paths(cvr_params, heston_params, sim_params)
    print(f"  Generated {sim_params.n_paths:,} paths across {cpu_count()} cores")

    # Step 3: Calculate CVR value
    print("\nStep 3: Calculating CVR payoffs...")
    results = calculate_cvr_payoffs(simulation_results, cvr_params, sim_params)

    # Step 4: Generate visualizations
    print("\nStep 4: Creating visualizations...")
    create_visualizations(simulation_results, results, cvr_params, heston_params,
                          filename='cvr_simulation_results.png', n_sample_paths=100)

    # Step 5: Export all data to CSV files
    print("\nStep 5: Exporting data to CSV files...")
    export_all_results(simulation_results, results, cvr_params, heston_params, sim_params,
                       output_dir='cvr_output')

    # Step 6: Export compact summary
    export_compact_summary(results, filename='cvr_quick_summary.csv')

    # Step 7: Print detailed summary
    print_detailed_summary(results, cvr_params)

    # Print results
    print("RESULTS")
    print(f"\nCVR FAIR VALUE: ${results['cvr_value']:.2f}")
    print(f"\nProbabilities:")
    print(f"   Barrier Hit: {results['barrier_hit_prob']:.2%}")
    print(f"   CoC Triggered: {results['coc_prob']:.2%}")
    print(f"   Total Payout: {(results['barrier_hit_prob'] + results['coc_prob']):.2%}")
    print(f"\nAvg Payoff (if triggered): ${results['avg_payoff_if_triggered']:.2f}")


    return results


if __name__ == "__main__":
    import gc
    from multiprocessing import active_children

    results = main()

    # Cleanup multiprocessing
    for child in active_children():
        child.terminate()
        child.join(timeout=1)

    gc.collect()

    print("\nAll done!")