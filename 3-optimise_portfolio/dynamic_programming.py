import numpy as np
from scipy.optimize import minimize_scalar

def expected_future_value_vectorized_1d(p, current_log_price, next_grid, value_fn, t, next_pdf):
    """
    Compute expected value for 1D scale-invariant model.
    The portfolio return is: (1 - p) + p * exp(x_next - x_current)
    
    For log utility: V(w) = log(w) + V_relative
    Since log(w * return) = log(w) + log(return), we add log(return) to the relative value function
    """
    # Portfolio returns for each next log price
    portfolio_returns = (1 - p) + p * np.exp(next_grid - current_log_price)
    
    # Avoid log of negative numbers
    portfolio_returns = np.maximum(portfolio_returns, 1e-8)
    
    # For scale-invariant utility: add log(portfolio_return) to next period value
    log_returns = np.log(portfolio_returns)
    
    # Expected value: E[V_{t+1}(x_{t+1}) + log(portfolio_return)]
    exp_val = np.sum((value_fn[t+1, :] + log_returns) * next_pdf)
    return exp_val

def dynamic_programming_policy(
    mu_seq, sigma_seq, utility_func, initial_wealth, current_log_price,
    n_steps=12, price_grid_size=101, wealth_grid_size=101, verbose=True,
    compute_utility_curve=False, n_alloc_points=100, moving_avg=1
):
    """
    1D Dynamic programming for optimal BTC allocation using scale-invariant utility.
    
    Uses only log-price as state variable. Assumes log or CRRA utility.
    Portfolio return = (1 - p) + p * exp(x_next - x_current)
    
    Bellman equation:
    V_t(x_t) = max_p E[ V_{t+1}(x_{t+1}) + log((1 - p) + p * exp(x_{t+1} - x_t)) ]
    """
    
    # Check utility function compatibility
    print(f"[DP] Using 1D scale-invariant model")
    
    # Create price grids for each time step
    price_grids = []
    for k in range(n_steps):
        grid = np.linspace(mu_seq[k] - 4*sigma_seq[k], mu_seq[k] + 4*sigma_seq[k], price_grid_size)
        price_grids.append(grid)
    
    # 2D arrays: value_fn[t, p_idx] and policy[t, p_idx]
    value_fn = np.zeros((n_steps, price_grid_size))
    policy = np.zeros((n_steps, price_grid_size))
    
    # Terminal step: V_T(x_T) = 0 for log utility (since log(initial_wealth) is constant)
    # The relative value is what matters for portfolio choice
    value_fn[n_steps-1, :] = 0.0
    
    # Backward induction
    for t in reversed(range(n_steps-1)):
        if verbose:
            print(f"[DP] Time step {t}")
            
        current_grid = price_grids[t]
        next_grid = price_grids[t+1]
        next_mu = mu_seq[t+1]
        next_sigma = sigma_seq[t+1]
        
        # Compute PDF for next period
        next_pdf = (
            1/(next_sigma * np.sqrt(2*np.pi)) *
            np.exp(-0.5 * ((next_grid - next_mu)/next_sigma)**2)
        )
        next_pdf /= next_pdf.sum()
        
        for p_idx, log_price in enumerate(current_grid):
            def neg_exp_val(p):
                exp_val = expected_future_value_vectorized_1d(
                    p, log_price, next_grid, value_fn, t, next_pdf
                )
                return -exp_val  # Negative for minimization
            
            # Optimize allocation
            res = minimize_scalar(neg_exp_val, bounds=(0, 1), method='bounded')
            best_p = res.x
            best_val = -res.fun
            
            value_fn[t, p_idx] = best_val
            policy[t, p_idx] = best_p
    
    # Find initial allocation based on current log price
    p0_idx = np.searchsorted(price_grids[0], current_log_price, side='left')
    p0_idx = np.clip(p0_idx, 0, len(price_grids[0])-1)
    
    # Use interpolation for smoother result
    if p0_idx > 0 and p0_idx < len(price_grids[0])-1:
        # Linear interpolation between adjacent grid points
        p_low, p_high = price_grids[0][p0_idx-1], price_grids[0][p0_idx]
        weight = (current_log_price - p_low) / (p_high - p_low)
        optimal_initial_allocation = (
            policy[0, p0_idx-1] * (1 - weight) + 
            policy[0, p0_idx] * weight
        )
    else:
        optimal_initial_allocation = policy[0, p0_idx]
    
    # Compute utility curve if requested
    utility_curve_data = None
    if compute_utility_curve:
        allocs = np.linspace(0, 1, n_alloc_points)
        utilities = []
        
        # Use the exact grid point for consistency
        grid_log_price = price_grids[0][p0_idx]
        next_mu = mu_seq[1]
        next_sigma = sigma_seq[1]
        next_grid = price_grids[1]
        next_pdf = (
            1/(next_sigma * np.sqrt(2*np.pi)) *
            np.exp(-0.5 * ((next_grid - next_mu)/next_sigma)**2)
        )
        next_pdf /= next_pdf.sum()
        
        for alloc in allocs:
            exp_val = expected_future_value_vectorized_1d(
                alloc, grid_log_price, next_grid, value_fn, 0, next_pdf
            )
            utilities.append(exp_val)
        
        utilities = np.array(utilities)
        
        # Apply moving average smoothing if requested
        if moving_avg > 1:
            smoothed_utilities = np.full_like(utilities, np.nan)
            half_window = moving_avg // 2
            
            for i in range(len(utilities)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(utilities), i + half_window + 1)
                window_values = utilities[start_idx:end_idx]
                smoothed_utilities[i] = np.mean(window_values)
            
            if verbose:
                print(f"Applied moving average smoothing with window size {moving_avg}")
        else:
            smoothed_utilities = utilities
        
        utility_curve_data = (allocs, smoothed_utilities)
        
        # Use the allocation that gives maximum utility from the smoothed curve
        valid_mask = ~np.isnan(smoothed_utilities)
        if np.any(valid_mask):
            valid_utilities = smoothed_utilities[valid_mask]
            valid_allocs = allocs[valid_mask]
            max_idx_valid = np.argmax(valid_utilities)
            max_alloc = valid_allocs[max_idx_valid]
        else:
            max_idx = np.argmax(utilities)
            max_alloc = allocs[max_idx]
        
        # Override with curve maximum
        optimal_initial_allocation = max_alloc
        
        if verbose:
            print(f"Debug: Max utility allocation from smoothed curve: {max_alloc:.4f}")
            print(f"Debug: Grid-based optimal allocation: {policy[0, p0_idx]:.4f}")
            print(f"Debug: Using smoothed curve maximum as optimal allocation")
    
    if verbose:
        print(f"[DP] Initial log_price: {current_log_price:.4f}, closest grid idx: {p0_idx}")
        print(f"[DP] Optimal initial allocation: {optimal_initial_allocation:.4f}")
        print("Initial log price:", current_log_price)
        print("Price grid sample:", price_grids[0][:5], "...")
        print("Initial log price idx:", p0_idx)
        print("Optimal allocation (grid):", policy[0, p0_idx])
    
    # Create dummy wealth_grid for compatibility with existing plotting code
    wealth_grid = np.linspace(initial_wealth * 0.2, initial_wealth * 5.0, wealth_grid_size)
    
    if compute_utility_curve:
        return policy, value_fn, optimal_initial_allocation, utility_curve_data, wealth_grid, price_grids
    else:
        return policy, value_fn, optimal_initial_allocation, wealth_grid, price_grids