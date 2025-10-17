"""
Heston path simulator module with parallel processing
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from models.parameters import CVRParams, SimulationParams, gbmParams


def _simulate_chunk_antithetic(args):
    """
    Worker function to simulate a chunk of paths in parallel

    Parameters:
    -----------
    args : tuple
        (n_paths_chunk, cvr_params, gbm_params, sim_params, chunk_id)
    """
    n_paths_chunk, cvr_params, gbm_params, sim_params, chunk_id = args

    # Unique seed for this worker
    np.random.seed(sim_params.seed + chunk_id * 10000)

    # Setup
    n_steps = int(cvr_params.T * 252)
    dt = cvr_params.T / n_steps

    # CRITICAL: We'll generate n_paths_chunk/2 random numbers
    # Then create antithetic pairs to get full n_paths_chunk
    n_paths_half = n_paths_chunk // 2
    n_paths_actual = n_paths_half * 2  # Ensure even number

    # Initialize arrays for FULL paths (including antithetic pairs)
    S = np.zeros((n_paths_actual, n_steps + 1))
    S[:, 0] = cvr_params.S0

    # Constant volatility for GBM
    sigma = gbm_params.sigma

    # Generate random numbers for HALF the paths
    Z_half = np.random.standard_normal((n_paths_half, n_steps))

    # Create antithetic pairs: Z and -Z
    Z_positive = Z_half
    Z_negative = -Z_half

    # Stack them: [path1+, path1-, path2+, path2-, ...]
    # This ensures antithetic pairs are adjacent for analysis
    Z = np.empty((n_paths_actual, n_steps))
    Z[0::2, :] = Z_positive  # Even indices: original paths
    Z[1::2, :] = Z_negative  # Odd indices: antithetic paths

    # Precompute drift term (same for all paths)
    drift = (cvr_params.r - cvr_params.q - 0.5 * sigma ** 2) * dt
    vol_sqrt_dt = sigma * np.sqrt(dt)

    # Simulate paths
    for t in range(n_steps):
        S_t = S[:, t]

        # GBM evolution: S(t+1) = S(t) * exp(drift*dt + sigma*sqrt(dt)*Z)
        S[:, t + 1] = S_t * np.exp(drift + vol_sqrt_dt * Z[:, t])

    # Calculate 30-day rolling averages
    rolling_window = cvr_params.rolling_window
    rolling_avgs = np.full_like(S, np.nan)

    for i in range(rolling_window - 1, n_steps + 1):
        rolling_avgs[:, i] = np.mean(S[:, max(0, i-rolling_window+1):i+1], axis=1)

    # Simulate CoC events
    coc_rate = sim_params.coc_intensity * dt

    # Generate uniform random numbers for half the paths
    U_half = np.random.rand(n_paths_half, n_steps)

    # Create antithetic pairs: U and (1-U)
    U_positive = U_half
    U_negative = 1.0 - U_half

    # Stack antithetic pairs
    U = np.empty((n_paths_actual, n_steps))
    U[0::2, :] = U_positive
    U[1::2, :] = U_negative

    coc_occurred = U < coc_rate
    coc_times = np.full(n_paths_actual, n_steps + 1)

    for path in range(n_paths_actual):
        occurrences = np.where(coc_occurred[path])[0]
        if len(occurrences) > 0:
            coc_times[path] = occurrences[0]

    return {
        'S_paths': S,
        'rolling_avgs': rolling_avgs,
        'coc_times': coc_times,
        'n_paths_actual': n_paths_actual
    }


def simulate_gbm_paths(cvr_params: CVRParams,
                         gbm_params: gbmParams,
                         sim_params: SimulationParams,
                         use_antithetic: bool=True):
    """
    Simulate price paths using parallel GBM Monte Carlo

    Parameters:
    -----------
    cvr_params : CVRParams
        CVR parameters (S0, barrier, etc.)
    gbm_params : gbmParams
        GBM volatility parameter
    sim_params : SimulationParams
        Simulation settings (n_paths, seed, n_jobs)
    use_antithetic : bool
        If True, use antithetic variates (default: True for variance reduction)

    Returns:
    --------
    dict with:
        - S_paths: Stock price paths (n_paths, n_steps+1)
        - rolling_avgs: 30-day rolling averages (n_paths, n_steps+1)
        - coc_times: Change of control event times (n_paths,)
        - dt: Time step
        - n_steps: Number of time steps
        - use_antithetic: Whether antithetic variates were used
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

    if use_antithetic:
        # Use antithetic variates
        worker_func = _simulate_chunk_antithetic

        # Create arguments for each worker
        worker_args = []
        for i in range(n_workers):
            n_paths_chunk = paths_per_worker + (1 if i < remainder else 0)
            worker_args.append((n_paths_chunk, cvr_params, gbm_params, sim_params, i))

        # Run parallel simulation
        with Pool(processes=n_workers) as pool:
            chunk_results = pool.map(worker_func, worker_args)

        # Aggregate results
        S_paths = np.vstack([r['S_paths'] for r in chunk_results])
        rolling_avgs = np.vstack([r['rolling_avgs'] for r in chunk_results])
        coc_times = np.concatenate([r['coc_times'] for r in chunk_results])

    n_steps = int(cvr_params.T * 252)
    dt = cvr_params.T / n_steps

    return {
        'S_paths': S_paths,
        'rolling_avgs': rolling_avgs,
        'coc_times': coc_times,
        'dt': dt,
        'n_steps': n_steps
    }