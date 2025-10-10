"""
Parameters module for CVR valuation
"""

from dataclasses import dataclass


@dataclass
class HestonParams:
    """Heston model parameters"""
    v0: float      # Initial variance
    kappa: float   # Mean reversion speed
    theta: float   # Long-run variance
    xi: float      # Vol of vol
    rho: float     # Correlation


@dataclass
class CVRParams:
    """CVR contract parameters"""
    S0: float              # Current stock price
    barrier: float         # Barrier level
    payoff_shares: float   # Shares per CVR
    T: float               # Time to expiration (years)
    r: float               # Risk-free rate
    q: float               # Dividend yield
    rolling_window: int = 30    # Days in rolling average
    delivery_days: int = 12     # Business days to delivery


@dataclass
class SimulationParams:
    """Monte Carlo simulation parameters"""
    n_paths: int           # Number of paths
    coc_intensity: float   # Change of control probability
    coc_premium: float     # Takeover premium
    seed: int = 22         # Random seed
    n_jobs: int = -1       # -1 means use all cores