"""
Hedging analysis module for CVR positions
Provides delta hedging, gamma hedging, and portfolio risk metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

from models.parameters import CVRParams, gbmParams, SimulationParams
from greeks.greeks import CVRGreeksCalculator, GreeksResult


@dataclass
class HedgePosition:
    """Represents a hedging position"""
    instrument: str
    quantity: float
    price: float
    delta: float
    gamma: float
    vega: float
    cost: float


@dataclass
class PortfolioRisk:
    """Portfolio risk metrics after hedging"""
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    hedge_cost: float
    hedge_ratio: float
    residual_risk: float


class CVRHedger:
    """Analyze hedging strategies for CVR positions"""

    def __init__(self,
                 cvr_params: CVRParams,
                 gbm_params: gbmParams,
                 sim_params: SimulationParams,
                 cvr_position_size: int = 10000):
        self.cvr = cvr_params
        self.gbm = gbm_params
        self.sim = sim_params
        self.position_size = cvr_position_size

        self.calculator = CVRGreeksCalculator(cvr_params, gbm_params, sim_params)
        self.cvr_greeks = self.calculator.calculate_all_greeks()

        self.portfolio_delta = self.cvr_greeks.delta * cvr_position_size
        self.portfolio_gamma = self.cvr_greeks.gamma * cvr_position_size
        self.portfolio_vega = self.cvr_greeks.vega * cvr_position_size
        self.portfolio_theta = self.cvr_greeks.theta * cvr_position_size

    def calculate_delta_hedge(self) -> HedgePosition:
        """Calculate simple delta hedge using Gen Digital stock"""
        stock_delta = 1.0
        hedge_quantity = -self.portfolio_delta / stock_delta
        hedge_cost = hedge_quantity * self.cvr.S0

        hedge = HedgePosition(
            instrument="GEN Stock",
            quantity=hedge_quantity,
            price=self.cvr.S0,
            delta=stock_delta * hedge_quantity,
            gamma=0.0,
            vega=0.0,
            cost=hedge_cost
        )

        return hedge

    def calculate_option_hedge(self,
                              strike: float,
                              maturity: float,
                              option_type: str = 'call') -> HedgePosition:
        """Calculate hedge using Gen Digital options"""
        S = self.cvr.S0
        K = strike
        T = maturity
        r = self.cvr.r
        q = self.cvr.q
        sigma = self.gbm.sigma

        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        from scipy.stats import norm

        if option_type == 'call':
            option_delta = np.exp(-q*T) * norm.cdf(d1)
            option_price = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            option_delta = -np.exp(-q*T) * norm.cdf(-d1)
            option_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)

        option_gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        option_vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)

        hedge_quantity = -self.portfolio_delta / option_delta

        hedge = HedgePosition(
            instrument=f"GEN {option_type.upper()} {strike}",
            quantity=hedge_quantity,
            price=option_price,
            delta=option_delta * hedge_quantity,
            gamma=option_gamma * hedge_quantity,
            vega=option_vega * hedge_quantity,
            cost=hedge_quantity * option_price
        )

        return hedge

    def calculate_gamma_hedge(self,
                             option_strike: float,
                             option_maturity: float) -> Tuple[HedgePosition, HedgePosition]:
        """Delta-Gamma neutral hedge using options + stock"""
        option_greeks = self.calculate_option_hedge(option_strike, option_maturity, 'call')

        option_quantity = -self.portfolio_gamma / (option_greeks.gamma / option_greeks.quantity)

        option_hedge = HedgePosition(
            instrument=option_greeks.instrument,
            quantity=option_quantity,
            price=option_greeks.price,
            delta=option_greeks.delta / option_greeks.quantity * option_quantity,
            gamma=option_greeks.gamma / option_greeks.quantity * option_quantity,
            vega=option_greeks.vega / option_greeks.quantity * option_quantity,
            cost=option_quantity * option_greeks.price
        )

        remaining_delta = self.portfolio_delta + option_hedge.delta

        stock_quantity = -remaining_delta
        stock_hedge = HedgePosition(
            instrument="GEN Stock",
            quantity=stock_quantity,
            price=self.cvr.S0,
            delta=stock_quantity,
            gamma=0.0,
            vega=0.0,
            cost=stock_quantity * self.cvr.S0
        )

        return option_hedge, stock_hedge

    def analyze_portfolio_risk(self, hedges: List[HedgePosition]) -> PortfolioRisk:
        """Calculate net portfolio risk after hedging"""
        net_delta = self.portfolio_delta + sum(h.delta for h in hedges)
        net_gamma = self.portfolio_gamma + sum(h.gamma for h in hedges)
        net_vega = self.portfolio_vega + sum(h.vega for h in hedges)
        net_theta = self.portfolio_theta

        hedge_cost = sum(h.cost for h in hedges)

        cvr_value = self.cvr_greeks.base_value * self.position_size
        hedge_ratio = abs(hedge_cost) / cvr_value if cvr_value > 0 else 0

        residual_risk = abs(net_delta) + abs(net_gamma * 10) + abs(net_vega)

        return PortfolioRisk(
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_vega=net_vega,
            net_theta=net_theta,
            hedge_cost=hedge_cost,
            hedge_ratio=hedge_ratio,
            residual_risk=residual_risk
        )

    def compare_hedge_strategies(self) -> pd.DataFrame:
        """Compare different hedging strategies"""
        print("\n" + "="*60)
        print("HEDGE STRATEGY COMPARISON")
        print("="*60)

        strategies = {}

        # No hedge
        no_hedge_risk = self.analyze_portfolio_risk([])
        strategies['No Hedge'] = {
            'Net Delta': no_hedge_risk.net_delta,
            'Net Gamma': no_hedge_risk.net_gamma,
            'Net Vega': no_hedge_risk.net_vega,
            'Cost': 0,
            'Hedge Ratio': 0,
            'Residual Risk': no_hedge_risk.residual_risk
        }

        # Delta hedge with stock
        print("\n1. Calculating delta hedge with stock...")
        stock_hedge = self.calculate_delta_hedge()
        delta_risk = self.analyze_portfolio_risk([stock_hedge])
        strategies['Delta Hedge (Stock)'] = {
            'Net Delta': delta_risk.net_delta,
            'Net Gamma': delta_risk.net_gamma,
            'Net Vega': delta_risk.net_vega,
            'Cost': delta_risk.hedge_cost,
            'Hedge Ratio': delta_risk.hedge_ratio,
            'Residual Risk': delta_risk.residual_risk
        }

        # Delta hedge with ATM call
        print("2. Calculating delta hedge with ATM call...")
        atm_call = self.calculate_option_hedge(
            strike=self.cvr.S0,
            maturity=min(1.0, self.cvr.T),
            option_type='call'
        )
        call_risk = self.analyze_portfolio_risk([atm_call])
        strategies['Delta Hedge (ATM Call)'] = {
            'Net Delta': call_risk.net_delta,
            'Net Gamma': call_risk.net_gamma,
            'Net Vega': call_risk.net_vega,
            'Cost': call_risk.hedge_cost,
            'Hedge Ratio': call_risk.hedge_ratio,
            'Residual Risk': call_risk.residual_risk
        }

        # Gamma hedge
        print("3. Calculating delta-gamma hedge...")
        option_hedge, stock_hedge2 = self.calculate_gamma_hedge(
            option_strike=self.cvr.S0,
            option_maturity=min(1.0, self.cvr.T)
        )
        gamma_risk = self.analyze_portfolio_risk([option_hedge, stock_hedge2])
        strategies['Delta-Gamma Hedge'] = {
            'Net Delta': gamma_risk.net_delta,
            'Net Gamma': gamma_risk.net_gamma,
            'Net Vega': gamma_risk.net_vega,
            'Cost': gamma_risk.hedge_cost,
            'Hedge Ratio': gamma_risk.hedge_ratio,
            'Residual Risk': gamma_risk.residual_risk
        }

        df = pd.DataFrame(strategies).T

        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(df.to_string())
        print("="*60 + "\n")

        return df

    def print_hedge_recommendation(self):
        """Print hedging recommendation based on analysis"""
        print("\n" + "="*60)
        print("HEDGE RECOMMENDATION")
        print("="*60)

        stock_hedge = self.calculate_delta_hedge()

        print(f"\nCVR Position: {self.position_size:,} CVRs")
        print(f"CVR Value: ${self.cvr_greeks.base_value:.2f} per CVR")
        print(f"Total Exposure: ${self.cvr_greeks.base_value * self.position_size:,.2f}")

        print(f"\n{'PORTFOLIO GREEKS (UNHEDGED)':-^60}")
        print(f"Delta:  {self.portfolio_delta:>12,.2f}")
        print(f"Gamma:  {self.portfolio_gamma:>12,.6f}")
        print(f"Vega:   {self.portfolio_vega:>12,.2f}")
        print(f"Theta:  {self.portfolio_theta:>12,.2f} per day")

        print(f"\n{'RECOMMENDED HEDGE':-^60}")
        print(f"Instrument:  {stock_hedge.instrument}")
        print(f"Action:      {'SELL' if stock_hedge.quantity > 0 else 'BUY'}")
        print(f"Quantity:    {abs(stock_hedge.quantity):,.0f} shares")
        print(f"Price:       ${stock_hedge.price:.2f}")
        print(f"Cost:        ${abs(stock_hedge.cost):,.2f}")

        hedged_risk = self.analyze_portfolio_risk([stock_hedge])

        print(f"\n{'AFTER HEDGE':-^60}")
        print(f"Net Delta:   {hedged_risk.net_delta:>12,.2f}")
        print(f"Net Gamma:   {hedged_risk.net_gamma:>12,.6f}")
        print(f"Net Vega:    {hedged_risk.net_vega:>12,.2f}")
        print(f"Hedge Ratio: {hedged_risk.hedge_ratio*100:>11,.1f}%")

        print(f"\n{'REHEDGING GUIDANCE':-^60}")
        print("Rehedge when:")
        print(f"  • Gen stock moves ±5% (±${self.cvr.S0 * 0.05:.2f})")
        print(f"  • Delta drifts >10% from initial hedge")
        print(f"  • Every 2 weeks (time decay effect)")

        print("="*60 + "\n")


def hedge_example():
    """Complete example of hedging analysis"""
    from models.parameters import CVRParams, gbmParams, SimulationParams

    cvr_params = CVRParams(S0=27.30, barrier=37.50, payoff_shares=0.7546,
                          T=1.52, r=0.045, q=0.0183)
    gbm_params = gbmParams(sigma=0.35)
    sim_params = SimulationParams(n_paths=30000, coc_intensity=0.15, coc_premium=1.25)

    hedger = CVRHedger(cvr_params, gbm_params, sim_params,
                      cvr_position_size=10000)

    hedger.print_hedge_recommendation()
    hedger.compare_hedge_strategies()


if __name__ == "__main__":
    hedge_example()