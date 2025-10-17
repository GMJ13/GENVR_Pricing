"""
Main script - Gen Digital CVR Valuation
"""

from multiprocessing import cpu_count
from models.parameters import HestonParams, CVRParams, SimulationParams, gbmParams
from simulation.heston_simulator import simulate_heston_paths
from simulation.gbm_simulator import simulate_gbm_paths
from pricing.payoff_calculator import calculate_cvr_payoffs
from visualization.visualization import create_visualizations, print_detailed_summary
from visualization.exporting_results import export_all_results, export_compact_summary
import time

def main():
    """Run CVR valuation"""

    print("=" * 70)
    print("GEN DIGITAL CVR VALUATION")
    print("=" * 70)

    # Step 1: Define parameters
    print("\nStep 1: Setting up parameters...")

    simulate_gbm = True #Model Choice between GBM and Heston

    cvr_params = CVRParams(
        S0=26.47,
        barrier=37.50,
        payoff_shares=0.7546,
        T=1.52,
        r=0.0405,
        q=0.0185,
        rolling_window=30,
        delivery_days=12
    )

    heston_params = HestonParams(
        v0=0.0604707294581154,
        kappa=7.53,
        theta=0.0956,
        xi=0.72,
        rho=-0.13
    )

    gbm_params = gbmParams(
        sigma=0.262871508582369
    )

    sim_params = SimulationParams(
        n_paths=50000,
        coc_intensity=0,
        coc_premium=1,
        seed=22,             #int(time.time()),
        n_jobs= -1
    )

    print(f"  Stock Price: ${cvr_params.S0:.2f}")
    print(f"  Barrier: ${cvr_params.barrier:.2f}")
    print(f"  Paths: {sim_params.n_paths:,}")
    if sim_params.n_jobs ==-1:
        print(f"  Available CPU Cores: {cpu_count()}",f"Using: all {cpu_count()} cores currently")
    else:
        print(f"  Available CPU Cores: {cpu_count()}", f"Using: all {sim_params.n_jobs} cores currently")

    start_simulate = time.perf_counter()
    # Step 2: Simulate paths
    if simulate_gbm:
        print("\nStep 2: Simulating GBM paths in parallel...")
        simulation_results = simulate_gbm_paths(cvr_params, gbm_params, sim_params, use_antithetic=True )
        print(f"  Generated {sim_params.n_paths:,} paths across {cpu_count()} cores")
    else:

        print("\nStep 2: Simulating Heston paths in parallel...")
        simulation_results = simulate_heston_paths(cvr_params, heston_params, sim_params)
        print(f"  Generated {sim_params.n_paths:,} paths across {cpu_count()} cores")

    # Step 3: Calculate CVR value
    print("\nStep 3: Calculating CVR payoffs...")
    results = calculate_cvr_payoffs(simulation_results, cvr_params, sim_params)

    end_simulate = time.perf_counter()
    print("\n\n")
    print("="*20)
    print(f"Time taken to simulate and price CVR is: {(end_simulate-start_simulate):.2f} seconds")
    print("="*20)

    # Step 4: Generate visualizations
    print("\n\nStep 4: Creating visualizations...")
    create_visualizations(simulation_results, results, cvr_params, heston_params, gbm_params, simulate_gbm,
                          filename='cvr_simulation_results.png', n_sample_paths=100)

    # Step 5: Export all data to CSV files
    print("\nStep 5: Exporting data to CSV files...")
    export_all_results(simulation_results, results, cvr_params, heston_params, sim_params, gbm_params, simulate_gbm,
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