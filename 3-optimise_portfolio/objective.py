import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.integrate import quad
import time

from config import Config
from plot_figures import plot_figures, plot_final_distributions
from get_utilty_function import get_utility_func



def load_gp_predictions(cfg: Config):
    X_pred = np.load(cfg.x_pred_pkl)
    y_pred = np.load(cfg.y_pred_pkl)
    y_std = np.load(cfg.y_std_pkl)

    df = pd.read_csv(cfg.log_csv, sep=';').sort_values(by='timestamp')
    current_log_price = np.log(df['close'].astype(float).values[-1])

    current_index = len(df)
    target_index = np.searchsorted(X_pred.ravel(), current_index)

    mu_seq = y_pred[target_index : target_index + cfg.horizon_weeks]
    sigma_seq = y_std[target_index : target_index + cfg.horizon_weeks]

    return mu_seq, sigma_seq, current_log_price


def objective_numerical_integral(p, mu_seq, sigma_seq, current_log_price, cfg):
    utility = get_utility_func(cfg)

    T = cfg.horizon_weeks // cfg.rebalance_every
    assert len(p) == T, f"Expected p of length {T}"

    # Calculate maximum grid points per dimension to stay under 1M paths
    max_total_paths = 1000000
    grid_points_per_dim = int(max_total_paths**(1/T))
    
    # Ensure we don't exceed the limit
    actual_paths = grid_points_per_dim ** T
    if actual_paths > max_total_paths:
        grid_points_per_dim -= 1
        actual_paths = grid_points_per_dim ** T
    
    print(f"Using {grid_points_per_dim} grid points per dimension for {T} dimensions")
    print(f"Total paths: {actual_paths:,}")

    # Build 1D grids for each future rebalance log-price x_t
    grid_limits = [
        np.linspace(mu_seq[t] - 4*sigma_seq[t], mu_seq[t] + 4*sigma_seq[t], grid_points_per_dim)
        for t in range(T)
    ]
    dx = np.array([g[1] - g[0] for g in grid_limits])

    # Create meshgrid for all path combinations
    grids = np.meshgrid(*grid_limits, indexing='ij')
    
    # Stack to get all paths: shape (n_total_paths, T)
    all_paths = np.stack([g.ravel() for g in grids], axis=1)
    n_paths = all_paths.shape[0]

    # Vectorized wealth calculation
    wealth = np.full(n_paths, cfg.initial_wealth)
    x_prev = np.full(n_paths, current_log_price)

    # Pre-compute p array for faster indexing
    p_array = np.array(p)
    
    for t in range(T):
        x_now = all_paths[:, t]
        # Vectorized exponential operations
        price_prev = np.exp(x_prev)
        price_now = np.exp(x_now)

        # Vectorized portfolio calculations
        p_t = p_array[t]
        cash = wealth * (1 - p_t)
        btc = (wealth * p_t) / price_prev
        wealth = cash + btc * price_now

        x_prev = x_now

    # Vectorize utility calculation where possible
    if cfg.utility_function in ['log', 'sqrt', 'identity', 'linear', 'crra']:
        # These can be fully vectorized
        if cfg.utility_function == 'log':
            utilities = np.log(np.maximum(wealth, 1e-10))  # Avoid log(0)
        elif cfg.utility_function == 'sqrt':
            utilities = np.sqrt(np.maximum(wealth, 0))
        elif cfg.utility_function in ['identity', 'linear']:
            utilities = wealth
        elif cfg.utility_function == 'crra':
            gamma = cfg.gamma
            if gamma == 1.0:
                utilities = np.log(np.maximum(wealth, 1e-10))
            else:
                utilities = (np.maximum(wealth, 1e-10)**(1-gamma) - 1) / (1-gamma)
    else:
        # For complex utility functions, use list comprehension (still faster than loop)
        utilities = np.array([utility(w) for w in wealth])

    # More efficient probability calculation
    # Pre-compute mu and sigma arrays
    mu_array = np.array([mu_seq[t] for t in range(T)])
    sigma_array = np.array([sigma_seq[t] for t in range(T)])
    
    # Vectorized log probability calculation
    log_probs = np.sum(
        norm.logpdf(all_paths, loc=mu_array, scale=sigma_array), 
        axis=1
    )
    prob_densities = np.exp(log_probs)

    # Pre-compute volume element
    volume_element = np.prod(dx)

    # Final integration
    total = np.sum(utilities * prob_densities) * volume_element

    return total

def run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, months=12):
    # Search space: p_t in [0, 1] for each month
    search_space = [Real(0.0, 1.0, name=f"p{i}") for i in range(months)]

    @use_named_args(search_space)
    def objective_wrapped(**kwargs):
        p = np.array([kwargs[f"p{i}"] for i in range(months)])
        util = objective_numerical_integral(p, mu_seq, sigma_seq, current_log_price, cfg)
        return -util  # Negative for minimisation

    result = gp_minimize(
        func=objective_wrapped,
        dimensions=search_space,
        n_calls=cfg.n_calls_optimiser,
        n_initial_points=10,
        acq_func="EI",  # Expected improvement
        random_state=42,
        verbose=True
    )

    optimal_p = np.array(result.x)
    max_utility = -result.fun

    return optimal_p, max_utility, result

def main():
    start_time = time.time()

    cfg = Config()
    mu_seq, sigma_seq, current_log_price = load_gp_predictions(cfg)
    
    # Debug: Check expected returns by period
    print(f"Current log price: {current_log_price:.4f}")
    print(f"GP Predictions by period:")
    for i in range(len(mu_seq)):
        expected_return = mu_seq[i] - current_log_price
        print(f"  Week {i}: μ={mu_seq[i]:.4f}, expected return={expected_return:.4f}")
    
    # Number of rebalancing points = horizon_weeks / rebalance_every
    T = cfg.horizon_weeks // cfg.rebalance_every

    # Evaluate objective at a naive initial guess (e.g. 50/50 BTC)
    p_init = np.full(T, 0.5)
    expected_util = objective_numerical_integral(p_init, mu_seq, sigma_seq, current_log_price, cfg)
    print(f"Initial Expected Utility: {expected_util:.4f}")
    print(f"Initial Allocation: {np.round(p_init, 3)}")

    # Run Bayesian optimisation
    print("\nRunning Bayesian Optimisation...")
    optimal_p, max_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T)

    print(f"\nOptimal allocation vector:")
    print(np.round(optimal_p, 3))
    print(f"Maximum expected utility: {max_util:.4f}")

    # Print timing information
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    # Generate all plots using first period data for visualization
    mu_first_period = mu_seq[0]  # First period prediction
    sigma_first_period = sigma_seq[0]  # First period uncertainty
    optimal_weight_first_period = optimal_p[0]  # First period allocation
    
    plot_figures(mu_first_period, sigma_first_period, current_log_price, optimal_weight_first_period, cfg)
    plot_final_distributions(mu_seq, sigma_seq, current_log_price, optimal_p, cfg)


if __name__ == '__main__':
    main()
