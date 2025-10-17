"""
CVR Greeks Analysis Runner
Main script to calculate and analyze CVR Greeks

Usage:
    python greeks_runner.py
"""

import os
from datetime import datetime

# Import from correct modules
from models.parameters import CVRParams, gbmParams, SimulationParams
from greeks.greeks import get_cvr_greeks, calculate_scenario_analysis
from greeks.visualization import create_greeks_report
from greeks.hedging import CVRHedger


def main():
    """
    Main execution function for Greeks analysis
    """
    print("\n" + "="*60)
    print("CVR GREEKS ANALYSIS")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ================================================================
    # STEP 1: Setup Parameters
    # ================================================================
    print("\n[STEP 1] Setting up parameters...")

    # CVR contract parameters
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

    # GBM model parameters
    gbm_params = gbmParams(
        sigma=0.262871508582369             # Volatility - UPDATE from calibration or historical
    )

    # Simulation parameters
    sim_params = SimulationParams(
        n_paths=50000,         # Number of Monte Carlo paths
        coc_intensity=0,    # 15% annual CoC probability
        coc_premium=1,      # 25% takeover premium
        seed=22,               # Random seed for reproducibility
        n_jobs=-1              # Use all CPU cores
    )

    # Display parameters
    print(f"\n  CVR Parameters:")
    print(f"    Stock Price:    ${cvr_params.S0:.2f}")
    print(f"    Barrier:        ${cvr_params.barrier:.2f}")
    print(f"    Distance:       {((cvr_params.barrier - cvr_params.S0)/cvr_params.S0)*100:+.1f}%")
    print(f"    Days to Expiry: {int(cvr_params.T * 365)}")

    print(f"\n  Model Parameters:")
    print(f"    Volatility:     {gbm_params.sigma*100:.1f}%")
    print(f"    CoC Intensity:  {sim_params.coc_intensity*100:.0f}%/year")

    print(f"\n  Simulation:")
    print(f"    Paths:          {sim_params.n_paths:,}")

    # ================================================================
    # STEP 2: Calculate Greeks
    # ================================================================
    print("\n[STEP 2] Calculating Greeks...")
    print("  (This may take 2-3 minutes with 50,000 paths)")

    greeks = get_cvr_greeks(cvr_params, gbm_params, sim_params)

    # ================================================================
    # STEP 3: Create Visualizations
    # ================================================================
    print("\n[STEP 3] Generating visualizations...")

    # Create output directory
    output_dir = f'./greeks_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Generate complete report
    create_greeks_report(cvr_params, gbm_params, sim_params, output_dir=output_dir)

    # ================================================================
    # STEP 4: Hedging Analysis (Optional)
    # ================================================================
    print("\n[STEP 4] Hedging Analysis...")

    response = input("\nRun hedging analysis for a position? (y/n): ").lower()

    if response == 'y':
        position_size = int(input("Enter position size (number of CVRs): "))

        hedger = CVRHedger(
            cvr_params,
            gbm_params,
            sim_params,
            cvr_position_size=position_size
        )

        # Print recommendation
        hedger.print_hedge_recommendation()

        # Compare strategies
        strategies = hedger.compare_hedge_strategies()

        # Save strategies to CSV
        strategies.to_csv(f'{output_dir}/hedge_strategies.csv')
        print(f"  [SAVED] Hedge strategies saved to {output_dir}/hedge_strategies.csv")
    else:
        print("  Skipping hedging analysis")

    # ================================================================
    # STEP 5: Scenario Analysis
    # ================================================================
    print("\n[STEP 5] Scenario Analysis...")

    scenarios = calculate_scenario_analysis(cvr_params, gbm_params, sim_params)

    # Save scenarios
    import pandas as pd
    scenarios_df = pd.DataFrame(scenarios).T
    scenarios_df.to_csv(f'{output_dir}/scenario_analysis.csv')
    print(f"  [SAVED] Scenarios saved to {output_dir}/scenario_analysis.csv")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n[SUCCESS] All outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  * greeks_dashboard.png     - Visual Greeks summary")
    print(f"  * greeks_heatmap.png       - Sensitivity surface")
    print(f"  * greeks_data.csv          - Exportable Greeks data")
    print(f"  * scenario_analysis.csv    - Scenario results")

    if response == 'y':
        print(f"  * hedge_strategies.csv     - Hedge comparison")

    print("\nKey Results:")
    print(f"  CVR Fair Value:      ${greeks.base_value:.2f}")
    print(f"  Delta:               {greeks.delta:.4f}")
    print(f"  Vega:                {greeks.vega:.2f}")
    print(f"  Theta (per day):     {greeks.theta:.4f}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()