"""
Heston path simulator module with parallel processing
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from models.parameters import HestonParams, CVRParams, SimulationParams


def _simulate_chunk(args):
    """
    Worker function to simulate a chunk of paths in parallel

    Parameters:
    -----------
    args : tuple
        (n_paths_chunk, cvr_params, heston_params, sim_params, chunk_id)
    """
    n_paths_chunk, cvr_params, heston_params, sim_params, chunk_id = args

    # Unique seed for this worker
    np.random.seed(sim_params.seed + chunk_id * 10000)

    # Setup
    n_steps = int(cvr_params.T * 252)
    dt = cvr_params.T / n_steps

    # Initialize arrays
    S = np.zeros((n_paths_chunk, n_steps + 1))
    v = np.zeros((n_paths_chunk, n_steps + 1))

    S[:, 0] = cvr_params.S0
    v[:, 0] = heston_params.v0

    # Heston parameters
    kappa = heston_params.kappa
    theta = heston_params.theta
    xi = heston_params.xi
    rho = heston_params.rho

    # QE scheme threshold
    psi_c = 1.5

    # Generate random numbers
    Z1 = np.random.randn(n_paths_chunk, n_steps)
    Z2 = np.random.randn(n_paths_chunk, n_steps)
    W_v = Z1
    W_S = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    # Precompute constants
    K0 = -rho * kappa * theta / xi * dt
    K1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = 0.5 * dt * (1 - rho**2)

    # Simulate paths
    for t in range(n_steps):
        v_t = v[:, t]
        S_t = S[:, t]

        # Variance evolution (QE scheme)
        m = theta + (v_t - theta) * np.exp(-kappa * dt)
        s2 = (v_t * xi**2 * np.exp(-kappa * dt) / kappa * (1 - np.exp(-kappa * dt)) +
              theta * xi**2 / (2 * kappa) * (1 - np.exp(-kappa * dt))**2)

        psi = s2 / (m**2 + 1e-10)
        v_next = np.zeros_like(v_t)

        # QE branching
        mask_quad = psi <= psi_c
        if mask_quad.any():
            b2 = 2 / (psi[mask_quad] + 1e-10) - 1 + np.sqrt(2 / (psi[mask_quad] + 1e-10)) * np.sqrt(2 / (psi[mask_quad] + 1e-10) - 1)
            a = m[mask_quad] / (1 + b2)
            Z_v = W_v[mask_quad, t]
            v_next[mask_quad] = a * (np.sqrt(b2) + Z_v)**2

        mask_exp = ~mask_quad
        if mask_exp.any():
            p = (psi[mask_exp] - 1) / (psi[mask_exp] + 1)
            beta = (1 - p) / (m[mask_exp] + 1e-10)
            U = np.random.rand(mask_exp.sum())
            v_next[mask_exp] = np.where(U <= p, 0.0, np.log((1 - p) / (1 - U + 1e-10)) / (beta + 1e-10))

        v[:, t+1] = np.maximum(v_next, 0)

        # Price evolution
        S[:, t+1] = S_t * np.exp(
            (cvr_params.r - cvr_params.q) * dt +
            K0 + K1 * v_t + K2 * v_next +
            np.sqrt(np.maximum(K3 * (v_t + v_next), 0)) * W_S[:, t]
        )

    # Calculate 30-day rolling averages
    rolling_window = cvr_params.rolling_window
    rolling_avgs = np.full_like(S, np.nan)

    for i in range(rolling_window - 1, n_steps + 1):
        rolling_avgs[:, i] = np.mean(S[:, max(0, i-rolling_window+1):i+1], axis=1)

    # Simulate CoC events
    coc_rate = sim_params.coc_intensity * dt
    coc_occurred = np.random.rand(n_paths_chunk, n_steps) < coc_rate
    coc_times = np.full(n_paths_chunk, n_steps + 1)

    for path in range(n_paths_chunk):
        occurrences = np.where(coc_occurred[path])[0]
        if len(occurrences) > 0:
            coc_times[path] = occurrences[0]

    return {
        'S_paths': S,
        'v_paths': v,
        'rolling_avgs': rolling_avgs,
        'coc_times': coc_times
    }


def simulate_heston_paths(cvr_params: CVRParams,
                         heston_params: HestonParams,
                         sim_params: SimulationParams):
    """
    Simulate price paths using parallel Heston Monte Carlo

    Returns:
    --------
    dict with:
        - S_paths: Stock price paths (n_paths, n_steps+1)
        - v_paths: Variance paths (n_paths, n_steps+1)
        - rolling_avgs: 30-day rolling averages (n_paths, n_steps+1)
        - coc_times: Change of control event times (n_paths,)
        - dt: Time step
        - n_steps: Number of time steps
    """

    # Determine number of workers
    n_cores = cpu_count()
    if sim_params.n_jobs == -1:
        n_workers = n_cores  # Use all cores
    else:
        n_workers = min(sim_params.n_jobs, n_cores)

    # Divide paths among workers
    n_paths = sim_params.n_paths
    paths_per_worker = n_paths // n_workers
    remainder = n_paths % n_workers

    # Create arguments for each worker
    worker_args = []
    for i in range(n_workers):
        n_paths_chunk = paths_per_worker + (1 if i < remainder else 0)
        worker_args.append((n_paths_chunk, cvr_params, heston_params, sim_params, i))

    # Run parallel simulation
    with Pool(processes=n_workers) as pool:
        chunk_results = pool.map(_simulate_chunk, worker_args)

    # Aggregate results from all workers
    S_paths = np.vstack([r['S_paths'] for r in chunk_results])
    v_paths = np.vstack([r['v_paths'] for r in chunk_results])
    rolling_avgs = np.vstack([r['rolling_avgs'] for r in chunk_results])
    coc_times = np.concatenate([r['coc_times'] for r in chunk_results])

    n_steps = int(cvr_params.T * 252)
    dt = cvr_params.T / n_steps

    return {
        'S_paths': S_paths,
        'v_paths': v_paths,
        'rolling_avgs': rolling_avgs,
        'coc_times': coc_times,
        'dt': dt,
        'n_steps': n_steps
    }