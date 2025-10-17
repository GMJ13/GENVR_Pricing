"""
CVR Implied Volatility Calibration from Market Price
Calibrates GBM volatility to match observed GENVR trading price
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import your existing modules
from models.parameters import CVRParams, SimulationParams, gbmParams
from simulation.gbm_simulator import simulate_gbm_paths
from pricing.payoff_calculator import calculate_cvr_payoffs


class CVRImpliedVolCalibrator:
    """
    Calibrate GBM volatility to match market price of GENVR
    """

    def __init__(self, cvr_params: CVRParams, sim_params: SimulationParams):
        self.cvr_params = cvr_params
        self.sim_params = sim_params

    def price_cvr_given_vol(self, sigma: float) -> float:
        """
        Price CVR for a given volatility

        Parameters:
        -----------
        sigma : float
            Volatility to test

        Returns:
        --------
        float : CVR fair value under this volatility
        """
        # Create GBM params with test volatility
        gbm_params_test = gbmParams(sigma=sigma)

        # Simulate paths
        sim_results = simulate_gbm_paths(
            self.cvr_params,
            gbm_params_test,
            self.sim_params
        )

        # Calculate payoff
        payoff_results = calculate_cvr_payoffs(
            sim_results,
            self.cvr_params,
            self.sim_params
        )

        return payoff_results['cvr_value']

    def objective_function(self, sigma: float, market_price: float) -> float:
        """
        Objective: (Model_Price - Market_Price)^2

        We want to minimize this
        """
        model_price = self.price_cvr_given_vol(sigma)
        error = model_price - market_price
        return error ** 2

    def calibrate_to_market(self,
                            market_price: float,
                            vol_bounds: tuple = (0.10, 0.80),
                            method: str = 'brentq',
                            tolerance: float = 0.01) -> dict:
        """
        Calibrate volatility to match market price

        Parameters:
        -----------
        market_price : float
            Observed GENVR trading price
        vol_bounds : tuple
            (min_vol, max_vol) search range
        method : str
            'brentq' (fast, requires bracketing) or 'minimize' (robust)
        tolerance : float
            Acceptable pricing error ($)

        Returns:
        --------
        dict with implied_vol, model_price, error, iterations
        """
        print(f"\n{'=' * 60}")
        print("CVR IMPLIED VOLATILITY CALIBRATION")
        print(f"{'=' * 60}")
        print(f"Market Price (GENVR): ${market_price:.2f}")
        print(f"Search Range: {vol_bounds[0] * 100:.0f}% - {vol_bounds[1] * 100:.0f}%")
        print(f"Tolerance: ±${tolerance:.2f}")
        print(f"\nCalibrating...")

        if method == 'brentq':
            # Brent's method (fast, requires sign change)
            def f(sigma):
                return self.price_cvr_given_vol(sigma) - market_price

            # Check if solution is bracketed
            f_low = f(vol_bounds[0])
            f_high = f(vol_bounds[1])

            print(f"  At σ={vol_bounds[0] * 100:.0f}%: Model=${f_low + market_price:.2f}")
            print(f"  At σ={vol_bounds[1] * 100:.0f}%: Model=${f_high + market_price:.2f}")

            if f_low * f_high > 0:
                print(f"\n WARNING: Solution not bracketed!")
                print(f"  Switching to minimize method...")
                method = 'minimize'
            else:
                implied_vol = brentq(f, vol_bounds[0], vol_bounds[1],
                                     xtol=0.001, maxiter=50)
                model_price = self.price_cvr_given_vol(implied_vol)
                error = abs(model_price - market_price)

                print(f"\n Calibration Complete")
                print(f"  Implied Volatility: {implied_vol * 100:.2f}%")
                print(f"  Model Price: ${model_price:.2f}")
                print(f"  Pricing Error: ${error:.2f}")

                return {
                    'implied_vol': implied_vol,
                    'model_price': model_price,
                    'market_price': market_price,
                    'error': error,
                    'method': 'brentq',
                    'converged': error < tolerance
                }

        if method == 'minimize':
            # Scipy minimize (more robust but slower)
            result = minimize_scalar(
                lambda sigma: self.objective_function(sigma, market_price),
                bounds=vol_bounds,
                method='bounded',
                options={'xatol': 0.001, 'maxiter': 50}
            )

            implied_vol = result.x
            model_price = self.price_cvr_given_vol(implied_vol)
            error = abs(model_price - market_price)

            print(f"\n Calibration Complete")
            print(f"  Implied Volatility: {implied_vol * 100:.2f}%")
            print(f"  Model Price: ${model_price:.2f}")
            print(f"  Pricing Error: ${error:.2f}")
            print(f"  Iterations: {result.nfev}")

            return {
                'implied_vol': implied_vol,
                'model_price': model_price,
                'market_price': market_price,
                'error': error,
                'method': 'minimize',
                'iterations': result.nfev,
                'converged': error < tolerance
            }

    def calibration_surface(self,
                            market_price: float,
                            vol_range: np.ndarray = None) -> dict:
        """
        Generate calibration surface showing model price vs volatility

        Useful for diagnostics and understanding sensitivity
        """
        if vol_range is None:
            vol_range = np.linspace(0.15, 0.60, 10)

        print(f"\nGenerating calibration surface...")
        model_prices = []

        for sigma in vol_range:
            price = self.price_cvr_given_vol(sigma)
            model_prices.append(price)
            print(f"  σ = {sigma * 100:5.1f}%  →  CVR = ${price:6.2f}")

        return {
            'volatilities': vol_range,
            'model_prices': np.array(model_prices),
            'market_price': market_price
        }


def fetch_genvr_price() -> dict:
    """
    Fetch current GENVR market data from Yahoo Finance

    Returns:
    --------
    dict with price, volume, bid, ask, last_trade_time
    """
    try:
        print("\nFetching GENVR market data...")
        ticker = yf.Ticker("GENVR")
        info = ticker.info

        # Try to get current price
        if 'regularMarketPrice' in info:
            price = info['regularMarketPrice']
        elif 'previousClose' in info:
            price = info['previousClose']
        else:
            price = None

        # Get bid-ask if available
        bid = info.get('bid', None)
        ask = info.get('ask', None)
        volume = info.get('volume', None)

        # Get recent trade
        hist = ticker.history(period='1d')
        if not hist.empty:
            last_price = hist['Close'].iloc[-1]
            last_time = hist.index[-1]
        else:
            last_price = price
            last_time = None

        print(f"  GENVR Data Retrieved:")
        print(f"  Last Price: ${last_price:.2f}" if last_price else "  Last Price: N/A")
        print(f"  Bid: ${bid:.2f}" if bid else "  Bid: N/A")
        print(f"  Ask: ${ask:.2f}" if ask else "  Ask: N/A")
        print(f"  Volume: {volume:,.0f}" if volume else "  Volume: N/A")
        print(f"  Last Trade: {last_time}" if last_time else "")

        return {
            'price': last_price,
            'bid': bid,
            'ask': ask,
            'volume': volume,
            'last_trade_time': last_time,
            'midpoint': (bid + ask) / 2 if (bid and ask) else last_price
        }

    except Exception as e:
        print(f"Error fetching GENVR: {str(e)}")
        print("\nManual entry required.")
        return None


def main():
    """
    Example: Calibrate volatility to GENVR market price
    """

    # Step 1: Fetch current GENVR price
    market_data = fetch_genvr_price()

    if market_data and market_data['price']:
        market_price = market_data['midpoint'] or market_data['price']
        print(f"\nUsing market price: ${market_price:.2f}")
    else:
        # Manual entry if fetch fails
        print("\n" + "=" * 60)
        market_price = float(input("Enter GENVR market price: $"))

    # Step 2: Setup parameters
    cvr_params = CVRParams(
        S0=26.47,  # Current Gen Digital stock price
        barrier=37.50,
        payoff_shares=0.7546,
        T=1.52,
        r=0.0405,
        q=0.0185
    )

    # Use fewer paths for calibration (speed vs accuracy tradeoff)
    sim_params = SimulationParams(
        n_paths=50000,  # Reduce for faster calibration
        coc_intensity=0,
        coc_premium=1,
        seed=22,
        n_jobs=-1
    )

    # Step 3: Create calibrator
    calibrator = CVRImpliedVolCalibrator(cvr_params, sim_params)

    # Step 4: Optional - Show calibration surface
    print("\n" + "=" * 60)
    response = input("Generate calibration surface? (y/n): ").lower()
    if response == 'y':
        surface = calibrator.calibration_surface(
            market_price=market_price,
            vol_range=np.linspace(0.20, 0.50, 7)
        )

        # Plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(surface['volatilities'] * 100, surface['model_prices'],
                     'b-o', linewidth=2, markersize=8, label='Model Price')
            plt.axhline(market_price, color='r', linestyle='--',
                        linewidth=2, label=f'Market Price (${market_price:.2f})')
            plt.xlabel('Volatility (%)', fontsize=12)
            plt.ylabel('CVR Price ($)', fontsize=12)
            plt.title('CVR Calibration Surface', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('cvr_calibration_surface.png', dpi=300, bbox_inches='tight')
            print("\nSurface plot saved as 'cvr_calibration_surface.png'")
            plt.show()
        except:
            pass

    # Step 5: Calibrate implied volatility
    print("\n" + "=" * 60)
    start_impVol = time.perf_counter()
    print("Starting implied volatility calibration...")
    print("This may take 1-2 minutes depending on simulation settings.")

    result = calibrator.calibrate_to_market(
        market_price=market_price,
        vol_bounds=(0.15, 0.65),
        method='brentq',  # Try brentq first
        tolerance=0.05  # Accept $0.05 error
    )

    end_impVol = time.perf_counter()
    print("\n" + "=" * 20)
    print(f"\n\nTime taken to calibrate Implied Volatility to current GENVR market price {(end_impVol-start_impVol)/60} minutes.\n\n")
    print("\n" + "=" * 20)
    # Step 6: Display results
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Market Price (GENVR): ${result['market_price']:.2f}")
    print(f"Implied Volatility:   {result['implied_vol'] * 100:.2f}%")
    print(f"Model Price:          ${result['model_price']:.2f}")
    print(f"Pricing Error:        ${result['error']:.2f}")
    print(f"Converged:            {'✓' if result['converged'] else '✗'}")
    print("=" * 60)

    # Step 7: Compare to historical volatility
    print("\nCOMPARISON TO HISTORICAL VOLATILITY")
    print("-" * 60)

    # Fetch Gen Digital historical data
    try:
        gen = yf.Ticker("GEN")
        hist = gen.history(period="1y")

        if not hist.empty:
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()

            # Calculate different horizons
            vol_30d = returns.tail(30).std() * np.sqrt(252)
            vol_90d = returns.tail(90).std() * np.sqrt(252)
            vol_1y = returns.std() * np.sqrt(252)

            print(f"30-day realized vol:  {vol_30d * 100:.2f}%")
            print(f"90-day realized vol:  {vol_90d * 100:.2f}%")
            print(f"1-year realized vol:  {vol_1y * 100:.2f}%")
            print(f"IMPLIED vol (GENVR):  {result['implied_vol'] * 100:.2f}%")
            print()

            # Analysis
            if result['implied_vol'] > vol_1y * 1.15:
                print("️  Implied vol is HIGHER than historical vol")
                print("    Market expects increased volatility")
                print("    CVR may be pricing in event risk")
            elif result['implied_vol'] < vol_1y * 0.85:
                print("   Implied vol is LOWER than historical vol")
                print("    Market expects decreased volatility")
                print("    CVR may be undervalued")
            else:
                print(" Implied vol consistent with historical vol")

    except Exception as e:
        print(f"Could not fetch GEN historical data: {str(e)}")

    print("=" * 60)

    # Step 8: Save results
    results_df = pd.DataFrame([{
        'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'GENVR_Price': result['market_price'],
        'Implied_Vol': result['implied_vol'],
        'Model_Price': result['model_price'],
        'Error': result['error'],
        'GEN_Price': cvr_params.S0,
        'Days_to_Expiry': cvr_params.T * 365
    }])

    results_df.to_csv('cvr_implied_vol_history.csv',
                      mode='a',
                      header=not pd.io.common.file_exists('cvr_implied_vol_history.csv'),
                      index=False)

    print("\n Results saved to 'cvr_implied_vol_history.csv'")

    return result

if __name__ == "__main__":
    start_time = time.perf_counter()
    result = main()
    end_time = time.perf_counter()
    print(f"Finished in {(end_time - start_time)/60} minutes.")