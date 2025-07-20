import numpy as np
import pandas as pd
from scipy.stats import norm
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time

from config import Config
from plot_figures import plot_allocation_vs_utility, plot_preference_curve, plot_final_wealth_distribution
from get_utility_function import get_utility_func


def calculate_grid_parameters(T, max_total_paths=1000000):
    grid_points_per_dim = min(100, int(max_total_paths**(1/T)))
    
    # Ensure we don't exceed the limit
    actual_paths = grid_points_per_dim ** T
    if actual_paths > max_total_paths:
        grid_points_per_dim -= 1
        actual_paths = grid_points_per_dim ** T
    
    print(f"Using {grid_points_per_dim} grid points per dimension for {T} dimensions")
    print(f"Total paths: {actual_paths:,}")
    
    return grid_points_per_dim, actual_paths


def load_gp_predictions(cfg: Config):
    X_pred = np.load(cfg.x_pred_pkl)
    y_pred = np.load(cfg.y_pred_pkl)
    y_std = np.load(cfg.y_std_pkl)

    df = pd.read_csv(cfg.price_csv, sep=',').sort_values(by='timestamp')
    current_log_price = np.log(df['price'].astype(float).values[-1])

    current_index = len(df)
    target_index = np.searchsorted(X_pred.ravel(), current_index)

    # Calculate number of rebalancing periods
    T = cfg.horizon_weeks // cfg.rebalance_every
    
    # Load predictions at rebalancing intervals
    rebalance_indices = [target_index + (t + 1) * cfg.rebalance_every for t in range(T)]
    mu_seq = y_pred[rebalance_indices]
    sigma_seq = y_std[rebalance_indices]

    return mu_seq, sigma_seq, current_log_price


def objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim):
    utility = get_utility_func(cfg)

    T = cfg.horizon_weeks // cfg.rebalance_every
    assert len(p) == T, f"Expected p of length {T}"

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

    # Start with log wealth
    log_wealth = np.full(n_paths, np.log(cfg.initial_wealth))
    x_prev = np.full(n_paths, current_log_price)

    # Pre-compute p array for faster indexing
    p_array = np.array(p)
    
    for t in range(T):
        x_now = all_paths[:, t]
        
        # Calculate log returns
        log_return = x_now - x_prev
        
        # Update log wealth using the formula from the image:
        # log(W_{t+1}) = log(W_t) + log[(1-p_t) + p_t * exp(log_return)]
        portfolio_return = (1 - p_array[t]) + p_array[t] * np.exp(log_return)
        
        # Add numerical stability check
        portfolio_return = np.maximum(portfolio_return, 1e-10)
        
        log_wealth += np.log(portfolio_return)
        x_prev = x_now

    # Pass entire log_wealth array to utility function (vectorized)
    utilities = utility(log_wealth)

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

def run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, months, grid_points_per_dim):
    # Search space: p_t in [0, 1] for each month
    search_space = [Real(0.0, 1.0, name=f"p{i}") for i in range(months)]

    @use_named_args(search_space)
    def objective_wrapped(**kwargs):
        p = np.array([kwargs[f"p{i}"] for i in range(months)])
        util = objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        return -util  # Negative for minimisation

    result = gp_minimize(
        func=objective_wrapped,
        dimensions=search_space,
        n_calls=cfg.n_calls_optimiser,
        n_initial_points=10,
        acq_func="EI", # EI, PI, or LCB
        random_state=42,
        verbose=True
    )

    optimal_p = np.array(result.x)
    max_utility = -result.fun

    return optimal_p, max_utility, result

def coordinate_descent_refinement(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim, max_recursive_calls=3):
    """More efficient refinement using coordinate descent for high-dimensional problems"""
    print("Using coordinate descent refinement...")
    
    refinement_deltas = np.array([-0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1])
    
    T = len(initial_p)
    current_p = np.array(initial_p)
    current_utility = objective_func(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    
    print(f"Starting utility: {current_utility:.6f}")
    
    overall_improvement = False
    found_edge_improvement = False
    
    # Iterate through each dimension
    for dim in range(T):
        best_delta = 0
        best_utility_for_dim = current_utility
        
        print(f"\nOptimizing dimension {dim} (current value: {current_p[dim]:.3f})")
        
        # Try each delta for this dimension
        for delta in refinement_deltas:
            test_p = current_p.copy()
            test_p[dim] += delta
            
            # Check bounds
            if test_p[dim] < 0 or test_p[dim] > 1:
                continue
            
            test_utility = objective_func(test_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
            
            if test_utility > best_utility_for_dim:
                best_utility_for_dim = test_utility
                best_delta = delta
        
        # Apply the best delta for this dimension
        if best_delta != 0:
            current_p[dim] += best_delta
            current_utility = best_utility_for_dim
            overall_improvement = True
            
            # Check if we hit the edge (±0.1) - suggests more improvement possible
            if abs(best_delta) == 0.1:
                found_edge_improvement = True
            
            print(f"  Improved dimension {dim}: {current_p[dim]:.3f} -> {current_utility:.6f} (delta: {best_delta:+.3f})")
        else:
            print(f"  No improvement for dimension {dim}")
    
    if overall_improvement:
        improvement = current_utility - objective_func(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        print(f"\nOverall improvement: {improvement:.6f}")
        print(f"Final allocation: {np.round(current_p, 3)}")
    else:
        print("\nNo improvement found - Bayesian optimization result was already locally optimal")
    
    # If we found improvement at the edge (±0.1), recursively call for further refinement
    if found_edge_improvement and max_recursive_calls > 0:
        print(f"\nFound edge improvement (±0.1 delta), recursively searching further...")
        print(f"Recursive calls remaining: {max_recursive_calls}")
        
        # Recursively call with current_p as the new starting point
        final_p, final_utility = coordinate_descent_refinement(
            current_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim, 
            max_recursive_calls - 1
        )
        return final_p, final_utility
    
    return current_p, current_utility

def main():
    start_time = time.time()

    cfg = Config()
    mu_seq, sigma_seq, current_log_price = load_gp_predictions(cfg)
    
    # Number of rebalancing points = horizon_weeks / rebalance_every
    T = cfg.horizon_weeks // cfg.rebalance_every
    
    # Calculate grid parameters once
    grid_points_per_dim, actual_paths = calculate_grid_parameters(T)
    
    # Evaluate objective at a naive initial guess (e.g. 50/50 BTC)
    p_init = np.full(T, 0.5)
    expected_util = objective_func(p_init, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    print(f"Initial Expected Utility: {expected_util:.4f}")
    print(f"Initial Allocation: {np.round(p_init, 3)}")
    
    if cfg.optimisation_method == "bayesian":
        print("\nRunning Bayesian Optimisation...")
        optimal_p, max_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)

        print(f"\nBayesian optimization result:")
        print(f"Allocation: {np.round(optimal_p, 3)}")
        print(f"Utility: {max_util:.6f}")
    elif cfg.optimisation_method == "bayesian_with_refinement":
        print("\nRunning Bayesian Optimisation with refinement...")
        bayesian_p, bayesian_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
        
        print(f"\nBayesian optimization result:")
        print(f"Allocation: {np.round(bayesian_p, 3)}")
        print(f"Utility: {bayesian_util:.6f}")

        optimal_p, max_util = coordinate_descent_refinement(bayesian_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)

    print(f"\nFinal optimal allocation:")
    print(f"Allocation: {np.round(optimal_p, 3)}")
    print(f"Maximum expected utility: {max_util:.6f}")

    mid_time = time.time()
    elapsed_time = mid_time - start_time
    print(f"\nElapsed time for optimization: {elapsed_time:.2f} seconds")
    
    plot_allocation_vs_utility(mu_seq, sigma_seq, current_log_price, optimal_p, cfg, grid_points_per_dim, objective_func)
    plot_preference_curve(cfg)
    plot_final_wealth_distribution(mu_seq, sigma_seq, current_log_price, optimal_p, cfg)

    # Print timing information
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
