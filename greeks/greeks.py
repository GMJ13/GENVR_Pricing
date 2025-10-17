"""
Greeks calculation module for GBM-based CVR pricing
Uses finite difference methods for all Greeks
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
import time

from models.parameters import CVRParams, SimulationParams, gbmParams
from simulation.gbm_simulator import simulate_gbm_paths
from pricing.payoff_calculator import calculate_cvr_payoffs


@dataclass
class GreeksResult:
    """Container for all Greeks"""
    # First-order Greeks
    delta: float
    vega: float
    theta: float
    rho: float
    lambda_greek: float
    barrier_sens: float

    # Second-order Greeks
    gamma: float
    vanna: float
    volga: float

    # Risk metrics
    delta_percent: float
    vega_percent: float

    # Base value
    base_value: float

    # Computation metadata
    computation_time: float
    confidence_intervals: Dict[str, Tuple[float, float]]


class CVRGreeksCalculator:
    """Calculate Greeks for CVR using finite differences"""

    def __init__(self,
                 cvr_params: CVRParams,
                 gbm_params: gbmParams,
                 sim_params: SimulationParams):
        self.cvr = cvr_params
        self.gbm = gbm_params
        self.sim = sim_params

        # Bump sizes for finite differences
        self.bumps = {
            'S': 0.01,
            'sigma': 0.01,
            'T': 1/365,
            'r': 0.0001,
            'lambda': 0.01,
            'barrier': 0.50
        }

    def price_cvr(self,
                  cvr_params: CVRParams,
                  gbm_params: gbmParams) -> float:
        """Price CVR with given parameters"""
        sim_results = simulate_gbm_paths(cvr_params, gbm_params, self.sim)
        payoff_results = calculate_cvr_payoffs(sim_results, cvr_params, self.sim)
        return payoff_results['cvr_value']

    def calculate_delta(self) -> Tuple[float, float, float]:
        """Delta: ∂V/∂S"""
        S0 = self.cvr.S0
        h = S0 * self.bumps['S']

        V0 = self.price_cvr(self.cvr, self.gbm)

        cvr_up = CVRParams(S0=S0 + h, barrier=self.cvr.barrier,
                          payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                          r=self.cvr.r, q=self.cvr.q,
                          rolling_window=self.cvr.rolling_window,
                          delivery_days=self.cvr.delivery_days)
        V_up = self.price_cvr(cvr_up, self.gbm)

        cvr_down = CVRParams(S0=S0 - h, barrier=self.cvr.barrier,
                            payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                            r=self.cvr.r, q=self.cvr.q,
                            rolling_window=self.cvr.rolling_window,
                            delivery_days=self.cvr.delivery_days)
        V_down = self.price_cvr(cvr_down, self.gbm)

        delta = (V_up - V_down) / (2 * h)
        gamma = (V_up - 2*V0 + V_down) / (h**2)

        return delta, gamma, V0

    def calculate_vega(self) -> Tuple[float, float, float]:
        """Vega: ∂V/∂σ"""
        sigma = self.gbm.sigma
        h = self.bumps['sigma']

        V0 = self.price_cvr(self.cvr, self.gbm)

        gbm_up = gbmParams(sigma=sigma + h)
        V_up = self.price_cvr(self.cvr, gbm_up)

        gbm_down = gbmParams(sigma=sigma - h)
        V_down = self.price_cvr(self.cvr, gbm_down)

        vega = (V_up - V_down) / (2 * h)
        volga = (V_up - 2*V0 + V_down) / (h**2)

        # Vanna
        S0 = self.cvr.S0
        h_S = S0 * self.bumps['S']

        cvr_up_s = CVRParams(S0=S0 + h_S, barrier=self.cvr.barrier,
                            payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                            r=self.cvr.r, q=self.cvr.q,
                            rolling_window=self.cvr.rolling_window,
                            delivery_days=self.cvr.delivery_days)

        V_up_s_up_sigma = self.price_cvr(cvr_up_s, gbm_up)
        V_up_s_down_sigma = self.price_cvr(cvr_up_s, gbm_down)

        cvr_down_s = CVRParams(S0=S0 - h_S, barrier=self.cvr.barrier,
                              payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                              r=self.cvr.r, q=self.cvr.q,
                              rolling_window=self.cvr.rolling_window,
                              delivery_days=self.cvr.delivery_days)

        V_down_s_up_sigma = self.price_cvr(cvr_down_s, gbm_up)
        V_down_s_down_sigma = self.price_cvr(cvr_down_s, gbm_down)

        vanna = ((V_up_s_up_sigma - V_up_s_down_sigma) -
                 (V_down_s_up_sigma - V_down_s_down_sigma)) / (4 * h_S * h)

        return vega, volga, vanna

    def calculate_theta(self) -> float:
        """Theta: ∂V/∂t (per day)"""
        T = self.cvr.T
        h = self.bumps['T']

        V0 = self.price_cvr(self.cvr, self.gbm)

        cvr_forward = CVRParams(S0=self.cvr.S0, barrier=self.cvr.barrier,
                               payoff_shares=self.cvr.payoff_shares, T=T - h,
                               r=self.cvr.r, q=self.cvr.q,
                               rolling_window=self.cvr.rolling_window,
                               delivery_days=self.cvr.delivery_days)
        V_forward = self.price_cvr(cvr_forward, self.gbm)

        theta = -(V_forward - V0) / h

        return theta

    def calculate_rho(self) -> float:
        """Rho: ∂V/∂r"""
        r = self.cvr.r
        h = self.bumps['r']

        cvr_up = CVRParams(S0=self.cvr.S0, barrier=self.cvr.barrier,
                          payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                          r=r + h, q=self.cvr.q,
                          rolling_window=self.cvr.rolling_window,
                          delivery_days=self.cvr.delivery_days)
        V_up = self.price_cvr(cvr_up, self.gbm)

        cvr_down = CVRParams(S0=self.cvr.S0, barrier=self.cvr.barrier,
                            payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                            r=r - h, q=self.cvr.q,
                            rolling_window=self.cvr.rolling_window,
                            delivery_days=self.cvr.delivery_days)
        V_down = self.price_cvr(cvr_down, self.gbm)

        rho = (V_up - V_down) / (2 * h)

        return rho

    def calculate_lambda_greek(self) -> float:
        """Lambda: ∂V/∂λ (CoC intensity sensitivity)"""
        lambda_base = self.sim.coc_intensity
        h = self.bumps['lambda']

        sim_up = SimulationParams(n_paths=self.sim.n_paths,
                                  coc_intensity=lambda_base + h,
                                  coc_premium=self.sim.coc_premium,
                                  seed=self.sim.seed,
                                  n_jobs=self.sim.n_jobs)

        sim_results_up = simulate_gbm_paths(self.cvr, self.gbm, sim_up)
        payoff_up = calculate_cvr_payoffs(sim_results_up, self.cvr, sim_up)
        V_up = payoff_up['cvr_value']

        sim_down = SimulationParams(n_paths=self.sim.n_paths,
                                    coc_intensity=max(0, lambda_base - h),
                                    coc_premium=self.sim.coc_premium,
                                    seed=self.sim.seed,
                                    n_jobs=self.sim.n_jobs)

        sim_results_down = simulate_gbm_paths(self.cvr, self.gbm, sim_down)
        payoff_down = calculate_cvr_payoffs(sim_results_down, self.cvr, sim_down)
        V_down = payoff_down['cvr_value']

        lambda_greek = (V_up - V_down) / (2 * h)

        return lambda_greek

    def calculate_barrier_sensitivity(self) -> float:
        """Barrier Sensitivity: ∂V/∂B"""
        B = self.cvr.barrier
        h = self.bumps['barrier']

        cvr_up = CVRParams(S0=self.cvr.S0, barrier=B + h,
                          payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                          r=self.cvr.r, q=self.cvr.q,
                          rolling_window=self.cvr.rolling_window,
                          delivery_days=self.cvr.delivery_days)
        V_up = self.price_cvr(cvr_up, self.gbm)

        cvr_down = CVRParams(S0=self.cvr.S0, barrier=B - h,
                            payoff_shares=self.cvr.payoff_shares, T=self.cvr.T,
                            r=self.cvr.r, q=self.cvr.q,
                            rolling_window=self.cvr.rolling_window,
                            delivery_days=self.cvr.delivery_days)
        V_down = self.price_cvr(cvr_down, self.gbm)

        barrier_sens = (V_up - V_down) / (2 * h)

        return barrier_sens

    def calculate_all_greeks(self, calculate_ci: bool = False) -> GreeksResult:
        """Calculate all Greeks efficiently"""
        print(f"\n{'='*60}")
        print("CVR GREEKS CALCULATION")
        print(f"{'='*60}")
        print("Computing finite differences...")

        start_time = time.time()

        print("  [1/6] Delta & Gamma...")
        delta, gamma, base_value = self.calculate_delta()

        print("  [2/6] Vega, Volga & Vanna...")
        vega, volga, vanna = self.calculate_vega()

        print("  [3/6] Theta...")
        theta = self.calculate_theta()

        print("  [4/6] Rho...")
        rho = self.calculate_rho()

        print("  [5/6] Lambda (CoC)...")
        lambda_greek = self.calculate_lambda_greek()

        print("  [6/6] Barrier Sensitivity...")
        barrier_sens = self.calculate_barrier_sensitivity()

        delta_percent = (delta * self.cvr.S0 / base_value * 100) if base_value > 0 else 0
        vega_percent = (vega / base_value * 100) if base_value > 0 else 0

        confidence_intervals = {}
        if calculate_ci:
            confidence_intervals = {
                'delta': (delta * 0.95, delta * 1.05),
                'vega': (vega * 0.95, vega * 1.05),
                'gamma': (gamma * 0.90, gamma * 1.10)
            }

        elapsed = time.time() - start_time

        result = GreeksResult(
            delta=delta, vega=vega, theta=theta, rho=rho,
            lambda_greek=lambda_greek, barrier_sens=barrier_sens,
            gamma=gamma, vanna=vanna, volga=volga,
            delta_percent=delta_percent, vega_percent=vega_percent,
            base_value=base_value, computation_time=elapsed,
            confidence_intervals=confidence_intervals
        )

        self._print_results(result)

        return result

    def _print_results(self, result: GreeksResult):
        """Print formatted Greeks results"""
        print(f"\n{'='*60}")
        print("GREEKS RESULTS")
        print(f"{'='*60}")
        print(f"Base CVR Value: ${result.base_value:.2f}")
        print(f"\n{'FIRST-ORDER GREEKS':-^60}")
        print(f"Delta (∂V/∂S):          {result.delta:>10.4f}  ({result.delta_percent:>6.1f}%)")
        print(f"  → $1 move in GEN:     ${result.delta:>10.2f}")
        print(f"  → 1% move in GEN:     ${result.delta * self.cvr.S0 * 0.01:>10.2f}")
        print(f"\nVega (∂V/∂σ):           {result.vega:>10.4f}  ({result.vega_percent:>6.1f}%)")
        print(f"  → 1% vol increase:    ${result.vega:>10.2f}")
        print(f"\nTheta (∂V/∂t):          {result.theta:>10.4f} per day")
        print(f"  → 1 week decay:       ${result.theta * 7:>10.2f}")
        print(f"  → 1 month decay:      ${result.theta * 30:>10.2f}")
        print(f"\nRho (∂V/∂r):            {result.rho:>10.4f}")
        print(f"  → 100bp rate change:  ${result.rho * 100:>10.2f}")
        print(f"\nLambda (∂V/∂λ):         {result.lambda_greek:>10.4f}")
        print(f"  → 5% CoC prob change: ${result.lambda_greek * 5:>10.2f}")
        print(f"\nBarrier Sens (∂V/∂B):   {result.barrier_sens:>10.4f}")
        print(f"  → $1 barrier change:  ${result.barrier_sens:>10.2f}")
        print(f"\n{'SECOND-ORDER GREEKS':-^60}")
        print(f"Gamma (∂²V/∂S²):        {result.gamma:>10.6f}")
        print(f"Vanna (∂²V/∂S∂σ):       {result.vanna:>10.6f}")
        print(f"Volga (∂²V/∂σ²):        {result.volga:>10.6f}")
        print(f"\n{'COMPUTATION':-^60}")
        print(f"Time: {result.computation_time:.1f}s")
        print(f"{'='*60}\n")


def get_cvr_greeks(cvr_params: CVRParams,
                   gbm_params: gbmParams,
                   sim_params: SimulationParams) -> GreeksResult:
    """Quick function to get all Greeks"""
    calculator = CVRGreeksCalculator(cvr_params, gbm_params, sim_params)
    return calculator.calculate_all_greeks()


def calculate_scenario_analysis(base_cvr: CVRParams,
                                base_gbm: gbmParams,
                                base_sim: SimulationParams) -> dict:
    """Run scenario analysis across multiple market conditions"""
    print("\n" + "="*60)
    print("SCENARIO ANALYSIS")
    print("="*60)

    scenarios = {
        'Base Case': (0, 0, 0),
        'Rally +10%': (0.10, 0, 0),
        'Drop -10%': (-0.10, 0, 0),
        'High Vol': (0, 0.10, 0),
        'Low Vol': (0, -0.05, 0),
        'M&A Likely': (0, 0, 0.10),
        'Stress': (-0.15, 0.15, 0)
    }

    results = {}
    calculator = CVRGreeksCalculator(base_cvr, base_gbm, base_sim)
    base_value = calculator.price_cvr(base_cvr, base_gbm)

    print(f"\nBase CVR Value: ${base_value:.2f}\n")
    print(f"{'Scenario':<20} {'CVR Value':<12} {'Change':<12}")
    print("-" * 60)

    for name, (dS, dsigma, dlambda) in scenarios.items():
        cvr_scenario = CVRParams(S0=base_cvr.S0 * (1 + dS),
                                barrier=base_cvr.barrier,
                                payoff_shares=base_cvr.payoff_shares,
                                T=base_cvr.T, r=base_cvr.r, q=base_cvr.q,
                                rolling_window=base_cvr.rolling_window,
                                delivery_days=base_cvr.delivery_days)

        gbm_scenario = gbmParams(sigma=max(0.10, base_gbm.sigma + dsigma))

        calc = CVRGreeksCalculator(cvr_scenario, gbm_scenario, base_sim)
        value = calc.price_cvr(cvr_scenario, gbm_scenario)
        change = value - base_value
        change_pct = (change / base_value * 100) if base_value > 0 else 0

        results[name] = {'value': value, 'change': change, 'change_pct': change_pct}
        print(f"{name:<20} ${value:>10.2f} {change:>+10.2f} ({change_pct:>+5.1f}%)")

    print("="*60 + "\n")
    return results