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
from plot_figures import plot_figures, plot_final_distributions, plot_allocation_vs_utility
from get_utilty_function import get_utility_func


def calculate_grid_parameters(T, max_total_paths=1000000):
    grid_points_per_dim = int(max_total_paths**(1/T))
    
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

    mu_seq = y_pred[target_index : target_index + cfg.horizon_weeks]
    sigma_seq = y_std[target_index : target_index + cfg.horizon_weeks]

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

def verify_optimal_allocation(optimal_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim):
    """Verify that the optimal allocation is actually optimal by testing nearby points"""
    print("\nVerifying optimal allocation...")
    
    # Test the optimal allocation
    optimal_utility = objective_func(optimal_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    print(f"Optimal allocation {np.round(optimal_p, 3)}: {optimal_utility:.6f}")
    
    # Test small perturbations around the optimal first allocation
    for delta in [-0.05, -0.01, 0.01, 0.05]:
        if 0 <= optimal_p[0] + delta <= 1:  # Check bounds
            test_p = np.array(optimal_p)
            test_p[0] += delta
            test_utility = objective_func(test_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
            print(f"First allocation {test_p[0]:.3f}: {test_utility:.6f} (diff: {test_utility - optimal_utility:.6f})")

def coordinate_descent_refinement(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim):
    """More efficient refinement using coordinate descent for high-dimensional problems"""
    print("Using coordinate descent refinement...")
    
    refinement_deltas = np.array([-0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1])
    
    T = len(initial_p)
    current_p = np.array(initial_p)
    current_utility = objective_func(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    
    print(f"Starting utility: {current_utility:.6f}")
    
    overall_improvement = False
    
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
            print(f"  Improved dimension {dim}: {current_p[dim]:.3f} -> {current_utility:.6f} (delta: {best_delta:+.3f})")
        else:
            print(f"  No improvement for dimension {dim}")
    
    if overall_improvement:
        improvement = current_utility - objective_func(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        print(f"\nOverall improvement: {improvement:.6f}")
        print(f"Final allocation: {np.round(current_p, 3)}")
    else:
        print("\nNo improvement found - Bayesian optimization result was already locally optimal")
    
    return current_p, current_utility

def coordinate_descent_optimization(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim, max_iterations=5):
    """Full coordinate descent optimization from random starting points"""
    print("\nRunning Coordinate Descent Optimization...")
    
    # Define search grids (coarse to fine)
    search_grids = [
        np.arange(0.0, 1.01, 0.1),    # Coarse: 0.0, 0.1, 0.2, ..., 1.0
        np.arange(0.0, 1.01, 0.05),   # Medium: 0.0, 0.05, 0.1, ..., 1.0
        np.arange(0.0, 1.01, 0.025),  # Fine: 0.0, 0.025, 0.05, ..., 1.0
    ]
    
    best_overall_p = None
    best_overall_utility = -np.inf
    
    # Try multiple random starting points
    starting_points = [
        np.full(T, 0.5),  # 50-50 allocation
        np.full(T, 0.3),  # Conservative
        np.full(T, 0.7),  # Aggressive
        np.random.uniform(0.2, 0.8, T),  # Random 1
        np.random.uniform(0.2, 0.8, T),  # Random 2
    ]
    
    for start_idx, start_p in enumerate(starting_points):
        print(f"\nStarting point {start_idx + 1}: {np.round(start_p, 3)}")
        
        current_p = start_p.copy()
        current_utility = objective_func(current_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        
        # Multi-resolution coordinate descent
        for resolution_idx, search_grid in enumerate(search_grids):
            print(f"  Resolution {resolution_idx + 1} (step size: {search_grid[1] - search_grid[0]:.3f})")
            
            for iteration in range(max_iterations):
                improved = False
                
                # Optimize each dimension
                for dim in range(T):
                    best_val = current_p[dim]
                    best_utility_dim = current_utility
                    
                    # Try all values in the search grid for this dimension
                    for val in search_grid:
                        test_p = current_p.copy()
                        test_p[dim] = val
                        
                        test_utility = objective_func(test_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
                        
                        if test_utility > best_utility_dim:
                            best_utility_dim = test_utility
                            best_val = val
                    
                    # Update if improvement found
                    if best_val != current_p[dim]:
                        current_p[dim] = best_val
                        current_utility = best_utility_dim
                        improved = True
                        print(f"    Dim {dim}: {best_val:.3f} -> {current_utility:.6f}")
                
                if not improved:
                    print(f"    Converged at iteration {iteration + 1}")
                    break
        
        print(f"  Final: {np.round(current_p, 3)} -> {current_utility:.6f}")
        
        # Update global best
        if current_utility > best_overall_utility:
            best_overall_utility = current_utility
            best_overall_p = current_p.copy()
            print(f"  New global best!")
    
    print(f"\nBest result: {np.round(best_overall_p, 3)} -> {best_overall_utility:.6f}")
    return best_overall_p, best_overall_utility

def compute_gradient(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim, epsilon=1e-6):
    """Compute gradient of objective function using finite differences"""
    T = len(p)
    gradient = np.zeros(T)
    
    # Current function value
    f_current = objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    
    # Compute partial derivatives using finite differences
    for i in range(T):
        # Forward difference
        p_forward = p.copy()
        p_forward[i] = min(1.0, p_forward[i] + epsilon)  # Respect bounds
        f_forward = objective_func(p_forward, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        
        # Backward difference
        p_backward = p.copy()
        p_backward[i] = max(0.0, p_backward[i] - epsilon)  # Respect bounds
        f_backward = objective_func(p_backward, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        
        # Central difference for better accuracy
        gradient[i] = (f_forward - f_backward) / (2 * epsilon)
    
    return gradient

def gradient_descent_optimization(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim, 
                                 max_iterations=100, learning_rate=0.1, tolerance=1e-6):
    """Gradient descent optimization with adaptive learning rate"""
    print("\nRunning Gradient Descent Optimization...")
    
    best_overall_p = None
    best_overall_utility = -np.inf
    
    # Try multiple starting points
    starting_points = [
        np.full(T, 0.5),  # 50-50 allocation
        np.full(T, 0.3),  # Conservative
        np.full(T, 0.7),  # Aggressive
        np.random.uniform(0.2, 0.8, T),  # Random 1
        np.random.uniform(0.2, 0.8, T),  # Random 2
    ]
    
    for start_idx, start_p in enumerate(starting_points):
        print(f"\nStarting point {start_idx + 1}: {np.round(start_p, 3)}")
        
        current_p = start_p.copy()
        current_utility = objective_func(current_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        current_lr = learning_rate
        
        print(f"  Initial utility: {current_utility:.6f}")
        
        for iteration in range(max_iterations):
            # Compute gradient
            gradient = compute_gradient(current_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
            
            # Gradient magnitude for convergence check
            grad_norm = np.linalg.norm(gradient)
            
            if grad_norm < tolerance:
                print(f"  Converged at iteration {iteration + 1} (gradient norm: {grad_norm:.8f})")
                break
            
            # Adaptive learning rate: try the step, if it doesn't improve, reduce learning rate
            step_found = False
            attempts = 0
            
            while not step_found and attempts < 10:
                # Proposed step
                proposed_p = current_p + current_lr * gradient
                
                # Project onto bounds [0, 1]
                proposed_p = np.clip(proposed_p, 0.0, 1.0)
                
                # Evaluate new point
                proposed_utility = objective_func(proposed_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
                
                # Check if improvement
                if proposed_utility > current_utility:
                    current_p = proposed_p
                    current_utility = proposed_utility
                    step_found = True
                    
                    # Increase learning rate slightly for next iteration
                    current_lr = min(current_lr * 1.1, 0.5)
                    
                    if iteration % 10 == 0:
                        print(f"    Iter {iteration + 1}: {np.round(current_p, 3)} -> {current_utility:.6f} (lr: {current_lr:.4f})")
                else:
                    # Reduce learning rate and try again
                    current_lr *= 0.5
                    attempts += 1
            
            if not step_found:
                print(f"  Learning rate too small, stopping at iteration {iteration + 1}")
                break
        
        print(f"  Final: {np.round(current_p, 3)} -> {current_utility:.6f}")
        
        # Update global best
        if current_utility > best_overall_utility:
            best_overall_utility = current_utility
            best_overall_p = current_p.copy()
            print(f"  New global best!")
    
    print(f"\nBest result: {np.round(best_overall_p, 3)} -> {best_overall_utility:.6f}")
    return best_overall_p, best_overall_utility

def analytical_gradient_descent(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim):
    """More efficient gradient descent using analytical gradients where possible"""
    print("\nRunning Analytical Gradient Descent...")
    
    # For log utility, we can compute some gradients analytically
    # This is more complex but would be faster than finite differences
    
    # For now, let's use a more efficient finite difference implementation
    def efficient_gradient(p, delta=1e-5):
        """Compute gradient more efficiently by reusing computations"""
        gradient = np.zeros(len(p))
        
        # Base function value
        f_base = objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        
        # Compute all forward differences in one go
        for i in range(len(p)):
            p_perturbed = p.copy()
            p_perturbed[i] = min(1.0, p_perturbed[i] + delta)
            f_perturbed = objective_func(p_perturbed, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
            gradient[i] = (f_perturbed - f_base) / delta
        
        return gradient
    
    # Use L-BFGS-B for bounded optimization
    from scipy.optimize import minimize
    
    def objective_for_scipy(p):
        return -objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    
    def gradient_for_scipy(p):
        return -efficient_gradient(p)
    
    # Try multiple starting points
    starting_points = [
        np.full(T, 0.5),
        np.full(T, 0.3),
        np.full(T, 0.7),
        np.random.uniform(0.2, 0.8, T),
        np.random.uniform(0.2, 0.8, T),
    ]
    
    best_result = None
    best_utility = -np.inf
    
    for i, start_p in enumerate(starting_points):
        print(f"\nStarting point {i + 1}: {np.round(start_p, 3)}")
        
        # Bounds for each variable
        bounds = [(0.0, 1.0) for _ in range(T)]
        
        # Run optimization
        result = minimize(
            fun=objective_for_scipy,
            x0=start_p,
            method='L-BFGS-B',
            jac=gradient_for_scipy,
            bounds=bounds,
            options={'maxiter': 100, 'disp': True}
        )
        
        utility = -result.fun
        print(f"  Result: {np.round(result.x, 3)} -> {utility:.6f}")
        
        if utility > best_utility:
            best_utility = utility
            best_result = result.x
            print(f"  New best!")
    
    print(f"\nBest result: {np.round(best_result, 3)} -> {best_utility:.6f}")
    return best_result, best_utility

def analytical_gradient_descent_with_warmstart(cfg, warm_start_p, warm_start_utility, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim):
    """Analytical gradient descent with coordinate descent warm start"""
    print("\nRunning Analytical Gradient Descent with Warm Start...")
    
    # Step 1: Quick coordinate descent to get near the optimum
    print("Step 1: Coordinate descent warm start...")
    
    print(f"Warm start result: {np.round(warm_start_p, 3)} -> {warm_start_utility:.6f}")
    
    # Step 2: Use L-BFGS-B from the warm start
    print("\nStep 2: L-BFGS-B refinement from warm start...")
    
    def efficient_gradient(p, delta=1e-5):
        """Compute gradient more efficiently by reusing computations"""
        gradient = np.zeros(len(p))
        
        # Base function value
        f_base = objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        
        # Compute all forward differences in one go
        for i in range(len(p)):
            p_perturbed = p.copy()
            p_perturbed[i] = min(1.0, p_perturbed[i] + delta)
            f_perturbed = objective_func(p_perturbed, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
            gradient[i] = (f_perturbed - f_base) / delta
        
        return gradient
    
    # Use L-BFGS-B for bounded optimization
    from scipy.optimize import minimize
    
    def objective_for_scipy(p):
        return -objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    
    def gradient_for_scipy(p):
        return -efficient_gradient(p)
    
    # Bounds for each variable
    bounds = [(0.0, 1.0) for _ in range(T)]
    
    # Run optimization starting from warm start
    result = minimize(
        fun=objective_for_scipy,
        x0=warm_start_p,
        method='L-BFGS-B',
        jac=gradient_for_scipy,
        bounds=bounds,
        options={'maxiter': 10, 'disp': True}  # Fewer iterations needed with good start
    )
    
    final_utility = -result.fun
    print(f"Final result: {np.round(result.x, 3)} -> {final_utility:.6f}")
    print(f"Improvement from warm start: {final_utility - warm_start_utility:.6f}")
    
    return result.x, final_utility

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
        print(f"Allocation: {np.round(bayesian_p, 3)}")
        print(f"Utility: {bayesian_util:.6f}")
    elif cfg.optimisation_method == "bayesian_with_refinement":
        print("\nRunning Bayesian Optimisation with refinement...")
        bayesian_p, bayesian_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
        
        print(f"\nBayesian optimization result:")
        print(f"Allocation: {np.round(bayesian_p, 3)}")
        print(f"Utility: {bayesian_util:.6f}")

        optimal_p, max_util = coordinate_descent_refinement(bayesian_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    elif cfg.optimisation_method == "bayesian_with_refinement2":
        print("\nRunning Bayesian Optimisation with refinement...")
        bayesian_p, bayesian_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
        
        print(f"\nBayesian optimization result:")
        print(f"Allocation: {np.round(bayesian_p, 3)}")
        print(f"Utility: {bayesian_util:.6f}")

        optimal_p, max_util = analytical_gradient_descent_with_warmstart(cfg, bayesian_p, bayesian_util, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
    elif cfg.optimisation_method == "coordinate":
        optimal_p, max_util = coordinate_descent_optimization(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
    elif cfg.optimisation_method == "gradient":
        optimal_p, max_util = gradient_descent_optimization(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
    elif cfg.optimisation_method == "analytical_gradient":
        optimal_p, max_util = analytical_gradient_descent(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)

    print(f"\nFinal optimal allocation:")
    print(f"Allocation: {np.round(optimal_p, 3)}")
    print(f"Maximum expected utility: {max_util:.6f}")

    mid_time = time.time()
    elapsed_time = mid_time - start_time
    print(f"\nElapsed time for optimization: {elapsed_time:.2f} seconds")

    # Generate single-step plots using first period data for visualization
    mu_first_period = mu_seq[0]  # First period prediction
    sigma_first_period = sigma_seq[0]  # First period uncertainty
    optimal_weight_first_period = optimal_p[0]  # First period allocation
    
    plot_figures(mu_first_period, sigma_first_period, current_log_price, optimal_weight_first_period, cfg)
    
    # Generate the multi-step allocation plot - pass the objective function
    plot_allocation_vs_utility(mu_seq, sigma_seq, current_log_price, optimal_p, cfg, grid_points_per_dim, objective_func)
    print("Saved allocation_vs_utility.png")
    
    # Generate final distribution plots
    plot_final_distributions(mu_seq, sigma_seq, current_log_price, optimal_p, cfg)

    # Print timing information
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    # Add verification
    verify_optimal_allocation(optimal_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)

if __name__ == '__main__':
    main()
