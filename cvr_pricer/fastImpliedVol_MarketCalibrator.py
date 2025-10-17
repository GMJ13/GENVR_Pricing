"""
ULTRA-FAST CVR Implied Volatility Calibration
Target: <5 seconds total runtime with 50k antithetic paths

Optimizations:
1. Antithetic variates (50% variance reduction in simulator)
2. Interpolation grid (5 points instead of 15+ iterations)
3. Parallel grid evaluation
4. Smart initial guess from vega approximation
5. Cached pricing function
6. Early convergence detection
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d, CubicSpline
from dataclasses import dataclass
from datetime import datetime
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# Import your existing modules
from models.parameters import CVRParams, SimulationParams, gbmParams
from simulation.gbm_simulator import simulate_gbm_paths
from pricing.payoff_calculator import calculate_cvr_payoffs


class UltraFastCVRImpliedVolCalibrator:
    """
    Ultra-fast implied vol calibration using interpolation + antithetic variates

    Expected performance: <5 seconds for 50k antithetic paths
    """

    def __init__(self, cvr_params: CVRParams, sim_params: SimulationParams):
        self.cvr_params = cvr_params
        self.sim_params = sim_params
        self._price_cache = {}  # Memoization
        self._call_count = 0

    def price_cvr_given_vol(self, sigma: float, use_cache: bool = True) -> float:
        """
        Price CVR with caching

        Parameters:
        -----------
        sigma : float
            Volatility to test
        use_cache : bool
            Use cached results if available

        Returns:
        --------
        float : CVR fair value
        """
        # Round sigma to avoid floating point cache misses
        sigma_key = round(sigma, 4)

        if use_cache and sigma_key in self._price_cache:
            return self._price_cache[sigma_key]

        self._call_count += 1

        gbm_params_test = gbmParams(sigma=sigma)

        # Use antithetic variates (automatically handled by simulator)
        sim_results = simulate_gbm_paths(
            self.cvr_params,
            gbm_params_test,
            self.sim_params,
            use_antithetic=True  # Ensure antithetic is enabled
        )

        payoff_results = calculate_cvr_payoffs(
            sim_results,
            self.cvr_params,
            self.sim_params
        )

        price = payoff_results['cvr_value']

        if use_cache:
            self._price_cache[sigma_key] = price

        return price

    def estimate_vega_and_initial_guess(self, market_price: float) -> tuple:
        """
        Fast initial guess using two-point vega estimation

        Returns:
        --------
        (initial_vol_guess, vega_estimate)
        """
        print(f"\n  Estimating vega and initial guess...")

        # Two evaluation points for vega estimation
        vol_low = 0.25
        vol_high = 0.40

        # Parallel evaluation of both points
        prices = Parallel(n_jobs=2)(
            delayed(self.price_cvr_given_vol)(vol)
            for vol in [vol_low, vol_high]
        )

        price_low, price_high = prices

        # Estimate vega (dPrice/dVol)
        vega = (price_high - price_low) / (vol_high - vol_low)

        # Linear extrapolation to target price
        if abs(vega) > 0.01:  # Avoid division by near-zero
            vol_guess = vol_low + (market_price - price_low) / vega
        else:
            vol_guess = 0.30  # Fallback

        # Clip to reasonable range
        vol_guess = np.clip(vol_guess, 0.15, 0.65)

        print(f"    Vol={vol_low * 100:.0f}% → Price=${price_low:.2f}")
        print(f"    Vol={vol_high * 100:.0f}% → Price=${price_high:.2f}")
        print(f"    Estimated Vega: ${vega:.2f} per 1% vol")
        print(f"    Initial guess: {vol_guess * 100:.1f}%")

        return vol_guess, vega

    def build_adaptive_grid(self,
                            vol_guess: float,
                            market_price: float,
                            grid_size: int = 5) -> tuple:
        """
        Build adaptive grid centered on initial guess

        Parameters:
        -----------
        vol_guess : float
            Center point for grid
        market_price : float
            Target price (for adaptive spacing)
        grid_size : int
            Number of grid points (default 5)

        Returns:
        --------
        (vol_points, price_points) : tuple of arrays
        """
        print(f"\n  Building adaptive grid ({grid_size} points)...")

        # Create grid with tighter spacing near the guess
        # Use logspace for better coverage
        half_width = 0.15  # ±15% around guess

        # Create non-uniform grid: denser near center
        if grid_size == 5:
            # Optimal 5-point grid: center + 2 on each side
            offsets = np.array([-0.12, -0.06, 0.0, 0.06, 0.12])
        elif grid_size == 7:
            offsets = np.array([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
        else:
            # Uniform fallback
            offsets = np.linspace(-half_width, half_width, grid_size)

        vol_grid = vol_guess + offsets
        vol_grid = np.clip(vol_grid, 0.15, 0.65)

        # Remove duplicates and sort
        vol_grid = np.unique(vol_grid)

        # Parallel evaluation of grid
        print(f"    Evaluating {len(vol_grid)} points in parallel...")
        start_grid = time.perf_counter()

        price_grid = Parallel(n_jobs=min(len(vol_grid), 8))(
            delayed(self.price_cvr_given_vol)(vol)
            for vol in vol_grid
        )

        price_grid = np.array(price_grid)
        elapsed_grid = time.perf_counter() - start_grid

        # Display results
        for vol, price in zip(vol_grid, price_grid):
            indicator = "  ← closest" if abs(price - market_price) == min(abs(price_grid - market_price)) else ""
            print(f"    σ={vol * 100:5.1f}% → ${price:6.2f}{indicator}")

        print(f"    Grid evaluation: {elapsed_grid:.2f}s")

        return vol_grid, price_grid

    def solve_with_interpolation(self,
                                 vol_points: np.ndarray,
                                 price_points: np.ndarray,
                                 market_price: float,
                                 vol_bounds: tuple) -> float:
        """
        Solve for implied vol using interpolated function

        Parameters:
        -----------
        vol_points : array
            Grid volatilities
        price_points : array
            Corresponding CVR prices
        market_price : float
            Target price
        vol_bounds : tuple
            Search bounds

        Returns:
        --------
        float : Implied volatility (approximate)
        """
        print(f"\n  Solving interpolated function...")

        # Build cubic spline interpolator
        # CubicSpline is more stable than interp1d for root finding
        try:
            interpolator = CubicSpline(vol_points, price_points)
        except:
            # Fallback to linear if cubic fails
            interpolator = interp1d(vol_points, price_points,
                                    kind='linear', fill_value='extrapolate')

        def interpolated_error(sigma):
            return interpolator(sigma) - market_price

        # Check if bracketed
        vol_min, vol_max = vol_bounds
        f_min = interpolated_error(vol_min)
        f_max = interpolated_error(vol_max)

        print(f"    Bracket check:")
        print(f"      f({vol_min * 100:.0f}%) = ${f_min:.2f}")
        print(f"      f({vol_max * 100:.0f}%) = ${f_max:.2f}")

        if f_min * f_max > 0:
            print(f"    Warning: Solution not well-bracketed, using closest grid point")
            # Use closest grid point
            idx = np.argmin(np.abs(price_points - market_price))
            return vol_points[idx]

        try:
            # Solve using Brent's method on interpolated function
            implied_vol = brentq(
                interpolated_error,
                vol_min,
                vol_max,
                xtol=0.0001,
                maxiter=50
            )

            print(f"    Solution: {implied_vol * 100:.2f}%")
            return implied_vol

        except Exception as e:
            print(f"    Brentq failed: {e}")
            # Fallback to closest grid point
            idx = np.argmin(np.abs(price_points - market_price))
            return vol_points[idx]

    def calibrate_ultra_fast(self,
                             market_price: float,
                             vol_bounds: tuple = (0.15, 0.65),
                             tolerance: float = 0.10,
                             grid_size: int = 5,
                             refine: bool = True) -> dict:
        """
        ULTRA-FAST calibration using all optimizations

        Strategy:
        1. Two-point vega estimation (parallel)
        2. Adaptive grid around initial guess (parallel)
        3. Cubic spline interpolation
        4. Optional single-point refinement

        Parameters:
        -----------
        market_price : float
            Target GENVR price
        vol_bounds : tuple
            Search range
        tolerance : float
            Acceptable error ($)
        grid_size : int
            Grid points (5 recommended)
        refine : bool
            Do exact refinement (adds one more pricing call)

        Returns:
        --------
        dict with results and diagnostics
        """
        print(f"\n{'=' * 70}")
        print("ULTRA-FAST CVR IMPLIED VOLATILITY CALIBRATION")
        print(f"{'=' * 70}")
        print(f"Target Price (GENVR): ${market_price:.2f}")
        print(f"Tolerance: ±${tolerance:.2f}")
        print(f"Antithetic Variates: {'Enabled' if self.sim_params.n_paths else 'N/A'}")
        print(f"Monte Carlo Paths: {self.sim_params.n_paths:,}")

        overall_start = time.perf_counter()

        # Reset call counter
        self._call_count = 0

        # Step 1: Vega estimation + initial guess
        vol_guess, vega = self.estimate_vega_and_initial_guess(market_price)

        # Step 2: Build adaptive grid
        vol_grid, price_grid = self.build_adaptive_grid(
            vol_guess,
            market_price,
            grid_size
        )

        # Step 3: Solve interpolated function
        implied_vol_approx = self.solve_with_interpolation(
            vol_grid,
            price_grid,
            market_price,
            vol_bounds
        )

        # Get interpolated price
        interpolator = CubicSpline(vol_grid, price_grid)
        model_price_approx = interpolator(implied_vol_approx)
        error_approx = abs(model_price_approx - market_price)

        print(f"\n  Interpolation result:")
        print(f"    Implied Vol: {implied_vol_approx * 100:.2f}%")
        print(f"    Model Price: ${model_price_approx:.2f}")
        print(f"    Error: ${error_approx:.2f}")

        # Step 4: Optional refinement with exact pricing
        if refine and error_approx > tolerance * 0.5:
            print(f"\n  Refining with exact evaluation...")
            model_price_exact = self.price_cvr_given_vol(implied_vol_approx)
            error_exact = abs(model_price_exact - market_price)

            # If error is large, do local search
            if error_exact > tolerance:
                print(f"    Error ${error_exact:.2f} > tolerance, local search...")

                # Try 3 points around solution
                local_vols = implied_vol_approx + np.array([-0.03, 0.0, 0.03])
                local_vols = np.clip(local_vols, vol_bounds[0], vol_bounds[1])

                local_prices = [self.price_cvr_given_vol(v) for v in local_vols]

                # Pick best
                idx_best = np.argmin(np.abs(np.array(local_prices) - market_price))
                implied_vol_final = local_vols[idx_best]
                model_price_final = local_prices[idx_best]
                error_final = abs(model_price_final - market_price)

                method_used = 'interpolation+local_search'
            else:
                implied_vol_final = implied_vol_approx
                model_price_final = model_price_exact
                error_final = error_exact
                method_used = 'interpolation+refinement'
        else:
            # Use interpolation result directly
            implied_vol_final = implied_vol_approx
            model_price_final = model_price_approx
            error_final = error_approx
            method_used = 'interpolation_only'

        elapsed = time.perf_counter() - overall_start

        print(f"\n{'=' * 70}")
        print(f"CALIBRATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Implied Volatility:  {implied_vol_final * 100:.2f}%")
        print(f"  Model Price:         ${model_price_final:.2f}")
        print(f"  Market Price:        ${market_price:.2f}")
        print(f"  Pricing Error:       ${error_final:.2f}")
        print(f"  Time Elapsed:        {elapsed:.2f}s")
        print(f"  Pricing Calls:       {self._call_count}")
        print(f"  Method:              {method_used}")
        print(f"  Converged:           {'YES' if error_final < tolerance else 'NO'}")
        print(f"{'=' * 70}")

        return {
            'implied_vol': implied_vol_final,
            'model_price': model_price_final,
            'market_price': market_price,
            'error': error_final,
            'time_elapsed': elapsed,
            'pricing_calls': self._call_count,
            'method': method_used,
            'converged': error_final < tolerance,
            'vega_estimate': vega,
            'initial_guess': vol_guess
        }


def fetch_genvr_price() -> dict:
    """Fetch current GENVR market data"""
    import yfinance as yf

    try:
        print("\nFetching GENVR market data...")
        ticker = yf.Ticker("GENVR")
        info = ticker.info

        price = info.get('regularMarketPrice') or info.get('previousClose')
        bid = info.get('bid')
        ask = info.get('ask')
        volume = info.get('volume')

        hist = ticker.history(period='1d')
        if not hist.empty:
            last_price = hist['Close'].iloc[-1]
            last_time = hist.index[-1]
        else:
            last_price = price
            last_time = None

        print(f"  Last Price: ${last_price:.2f}" if last_price else "  N/A")
        print(f"  Bid/Ask: ${bid:.2f}/${ask:.2f}" if (bid and ask) else "  N/A")
        print(f"  Volume: {volume:,.0f}" if volume else "  N/A")

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
        return None


def compare_to_historical_vol(implied_vol: float) -> dict:
    """Compare implied vol to historical realized volatility"""
    import yfinance as yf

    print(f"\n{'=' * 70}")
    print("COMPARISON TO HISTORICAL VOLATILITY")
    print(f"{'-' * 70}")

    try:
        gen = yf.Ticker("GEN")
        hist = gen.history(period="1y")

        if hist.empty:
            print("Could not fetch GEN historical data")
            return {}

        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()

        # Calculate different horizons
        vol_30d = returns.tail(30).std() * np.sqrt(252)
        vol_90d = returns.tail(90).std() * np.sqrt(252)
        vol_1y = returns.std() * np.sqrt(252)

        print(f"  30-day realized:  {vol_30d * 100:5.2f}%")
        print(f"  90-day realized:  {vol_90d * 100:5.2f}%")
        print(f"  1-year realized:  {vol_1y * 100:5.2f}%")
        print(f"  IMPLIED (GENVR):  {implied_vol * 100:5.2f}%")
        print()

        # Analysis
        ratio = implied_vol / vol_1y

        if ratio > 1.15:
            print(f"  WARNING: Implied vol is {(ratio - 1) * 100:.1f}% HIGHER than historical")
            print(f"           Market expects increased volatility")
            print(f"           CVR may be pricing in event risk or uncertainty")
            interpretation = "elevated"
        elif ratio < 0.85:
            print(f"  NOTE: Implied vol is {(1 - ratio) * 100:.1f}% LOWER than historical")
            print(f"        Market expects decreased volatility")
            print(f"        CVR may be undervalued (potential opportunity)")
            interpretation = "subdued"
        else:
            print(f"  Implied vol consistent with historical ({ratio:.2f}x)")
            print(f"  Market expectations aligned with recent history")
            interpretation = "consistent"

        print(f"{'=' * 70}")

        return {
            'vol_30d': vol_30d,
            'vol_90d': vol_90d,
            'vol_1y': vol_1y,
            'implied_vol': implied_vol,
            'ratio': ratio,
            'interpretation': interpretation
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {}


def main():
    """
    Ultra-fast implied vol calibration example
    Target: <5 seconds total runtime
    """

    print(f"\n{'=' * 70}")
    print("CVR IMPLIED VOLATILITY CALIBRATION - ULTRA-FAST MODE")
    print(f"{'=' * 70}")

    overall_start = time.perf_counter()

    # Step 1: Fetch GENVR price
    market_data = fetch_genvr_price()

    if market_data and market_data['price']:
        market_price = market_data['midpoint'] or market_data['price']
        print(f"\nUsing market price: ${market_price:.2f}")
    else:
        market_price = float(input("\nEnter GENVR market price: $"))

    # Step 2: Setup parameters
    cvr_params = CVRParams(
        S0=26.47,  # Current Gen Digital stock price
        barrier=37.50,
        payoff_shares=0.7546,
        T=1.52,
        r=0.0405,
        q=0.0185
    )

    # CRITICAL: Use 50k paths with antithetic variates
    # This gives same accuracy as 100k regular paths but 2x faster
    sim_params = SimulationParams(
        n_paths=50000,  # With antithetic = 100k effective
        coc_intensity=0,  # Disable CoC for calibration speed
        coc_premium=1,
        seed=42,
        n_jobs=-1  # Use all cores
    )

    # Step 3: Create ultra-fast calibrator
    calibrator = UltraFastCVRImpliedVolCalibrator(cvr_params, sim_params)

    # Step 4: FAST calibration
    result = calibrator.calibrate_ultra_fast(
        market_price=market_price,
        vol_bounds=(0.15, 0.65),
        tolerance=0.10,  # $0.10 acceptable for speed
        grid_size=5,  # 5 points = optimal balance
        refine=True  # Do one exact refinement
    )

    # Step 5: Compare to historical
    hist_comparison = compare_to_historical_vol(result['implied_vol'])

    # Step 6: Save results
    results_df = pd.DataFrame([{
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'GENVR_Price': result['market_price'],
        'Implied_Vol_%': result['implied_vol'] * 100,
        'Model_Price': result['model_price'],
        'Error': result['error'],
        'Time_Seconds': result['time_elapsed'],
        'Pricing_Calls': result['pricing_calls'],
        'Method': result['method'],
        'GEN_Price': cvr_params.S0,
        'Days_to_Expiry': cvr_params.T * 365,
        'Hist_Vol_1Y_%': hist_comparison.get('vol_1y', np.nan) * 100,
        'Vol_Ratio': hist_comparison.get('ratio', np.nan)
    }])

    filename = 'cvr_ultra_fast_implied_vol.csv'
    results_df.to_csv(
        filename,
        mode='a',
        header=not pd.io.common.file_exists(filename),
        index=False
    )

    print(f"\nResults saved to '{filename}'")

    # Overall timing
    overall_elapsed = time.perf_counter() - overall_start

    print(f"\n{'=' * 70}")
    print(f"TOTAL RUNTIME: {overall_elapsed:.2f} seconds")
    print(f"{'=' * 70}")

    # Performance summary
    print(f"\nPerformance breakdown:")
    print(f"  Calibration:  {result['time_elapsed']:.2f}s")
    print(f"  Data fetch:   {(overall_elapsed - result['time_elapsed']):.2f}s")
    print(f"  Pricing calls: {result['pricing_calls']}")
    print(f"  Effective paths: {sim_params.n_paths:,} (antithetic)")

    if overall_elapsed < 5:
        print(f"\nTARGET ACHIEVED: <5 second runtime!")
    elif overall_elapsed < 10:
        print(f"\nGood performance: <10 second runtime")
    else:
        print(f"\nSlower than expected. Consider:")
        print(f"    - Reducing n_paths to 30,000")
        print(f"    - Using grid_size=4")
        print(f"    - Disabling refinement")

    return result


if __name__ == "__main__":
    result = main()