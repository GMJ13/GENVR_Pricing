"""
CVR payoff calculator module
"""

import numpy as np
from models.parameters import CVRParams, SimulationParams


def calculate_cvr_payoffs(simulation_results: dict,
                          cvr_params: CVRParams,
                          sim_params: SimulationParams):
    """
    Calculate CVR payoffs from simulated paths

    Parameters:
    -----------
    simulation_results : dict
        Output from simulate_heston_paths()
    cvr_params : CVRParams
        Contract parameters
    sim_params : SimulationParams
        Simulation parameters

    Returns:
    --------
    dict with:
        - cvr_value: Fair value
        - barrier_hit_prob: Probability of barrier hit
        - coc_prob: Probability of CoC
        - avg_payoff_if_triggered: Average payoff when triggered
        - payoffs: Array of all payoffs
    """

    S_paths = simulation_results['S_paths']
    rolling_avgs = simulation_results['rolling_avgs']
    coc_times = simulation_results['coc_times']
    dt = simulation_results['dt']
    n_steps = simulation_results['n_steps']

    n_paths = S_paths.shape[0]
    delivery_time = cvr_params.delivery_days / 252

    # Calculate payoffs
    payoffs = np.zeros(n_paths)
    barrier_hits = np.zeros(n_paths, dtype=bool)
    coc_triggers = np.zeros(n_paths, dtype=bool)

    for path_idx in range(n_paths):
        # Check barrier hit
        barrier_mask = rolling_avgs[path_idx, :] >= cvr_params.barrier
        barrier_mask = barrier_mask & ~np.isnan(rolling_avgs[path_idx, :])

        if barrier_mask.any():
            barrier_time_idx = np.where(barrier_mask)[0][0]
        else:
            barrier_time_idx = n_steps + 1

        coc_time_idx = coc_times[path_idx]

        # First trigger wins
        if barrier_time_idx < coc_time_idx:
            # Barrier hit first
            trigger_time = barrier_time_idx * dt
            delivery_time_actual = min(trigger_time + delivery_time, cvr_params.T)
            delivery_idx = int(delivery_time_actual / dt)
            delivery_idx = min(delivery_idx, n_steps)

            delivery_price = S_paths[path_idx, delivery_idx]
            payoff = cvr_params.payoff_shares * delivery_price
            discount = np.exp(-cvr_params.r * delivery_time_actual)

            payoffs[path_idx] = payoff * discount
            barrier_hits[path_idx] = True

        elif coc_time_idx < n_steps + 1:
            # CoC occurred first
            trigger_time = coc_time_idx * dt
            delivery_time_actual = min(trigger_time + delivery_time, cvr_params.T)
            delivery_idx = int(delivery_time_actual / dt)
            delivery_idx = min(delivery_idx, n_steps)

            coc_price = S_paths[path_idx, int(coc_time_idx)] * sim_params.coc_premium
            delivery_price = coc_price * np.exp((cvr_params.r - cvr_params.q) * (delivery_time_actual - trigger_time))

            payoff = cvr_params.payoff_shares * delivery_price
            discount = np.exp(-cvr_params.r * delivery_time_actual)

            payoffs[path_idx] = payoff * discount
            coc_triggers[path_idx] = True

    # Calculate statistics
    cvr_value = payoffs.mean()
    barrier_hit_prob = barrier_hits.mean()
    coc_prob = coc_triggers.mean()
    avg_payoff_if_triggered = payoffs[payoffs > 0].mean() if (payoffs > 0).any() else 0

    return {
        'cvr_value': cvr_value,
        'barrier_hit_prob': barrier_hit_prob,
        'coc_prob': coc_prob,
        'avg_payoff_if_triggered': avg_payoff_if_triggered,
        'payoffs': payoffs
    }