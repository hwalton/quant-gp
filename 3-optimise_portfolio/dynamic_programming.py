import numpy as np
from scipy.optimize import minimize_scalar

def expected_future_value_vectorized(cash, btc_units, next_grid, wealth_grid, value_fn, t, next_pdf):
    next_wealth = cash + btc_units * np.exp(next_grid)
    # Use searchsorted for better index finding
    next_w_idx = np.searchsorted(wealth_grid, next_wealth, side='left')
    next_w_idx = np.clip(next_w_idx, 0, len(wealth_grid)-1)
    
    # Use proper array indexing
    v = value_fn[t+1, next_w_idx, np.arange(len(next_grid))]
    exp_val = np.sum(v * next_pdf)
    return exp_val

def dynamic_programming_policy(
    mu_seq, sigma_seq, utility_func, initial_wealth, current_log_price,
    n_steps=12, price_grid_size=101, wealth_grid_size=101, verbose=True,
    compute_utility_curve=False, n_alloc_points=100
):
    """
    Dynamic programming for optimal BTC allocation over a 1-year horizon (monthly rebalancing),
    using minimize_scalar for allocation optimization.
    
    If compute_utility_curve=True, also returns expected utilities for different allocations
    at the initial state for plotting purposes.
    """
    price_grids = []
    for k in range(n_steps):
        grid = np.linspace(mu_seq[k] - 4*sigma_seq[k], mu_seq[k] + 4*sigma_seq[k], price_grid_size)
        price_grids.append(grid)
    min_wealth = initial_wealth * 0.2
    max_wealth = initial_wealth * 5.0
    wealth_grid = np.linspace(min_wealth, max_wealth, wealth_grid_size)
    value_fn = np.zeros((n_steps, wealth_grid_size, price_grid_size))
    policy = np.zeros((n_steps, wealth_grid_size, price_grid_size))
    
    # Terminal step: utility of final wealth for all (wealth, log_price)
    for w_idx, wealth in enumerate(wealth_grid):
        for p_idx, log_price in enumerate(price_grids[-1]):
            value_fn[n_steps-1, w_idx, p_idx] = utility_func(wealth)
    
    # Backward induction
    for t in reversed(range(n_steps-1)):
        if verbose:
            print(f"[DP] Time step {t}")
        for w_idx, wealth in enumerate(wealth_grid):
            for p_idx, log_price in enumerate(price_grids[t]):
                def neg_exp_val(p):
                    cash = wealth * (1 - p)
                    btc_units = (wealth * p) / np.exp(log_price)
                    next_mu = mu_seq[t+1]
                    next_sigma = sigma_seq[t+1]
                    next_grid = price_grids[t+1]
                    next_pdf = (
                        1/(next_sigma * np.sqrt(2*np.pi)) *
                        np.exp(-0.5 * ((next_grid - next_mu)/next_sigma)**2)
                    )
                    next_pdf /= next_pdf.sum()
                    exp_val = expected_future_value_vectorized(
                        cash, btc_units, next_grid, wealth_grid, value_fn, t, next_pdf
                    )
                    return -exp_val  # Negative for minimization

                # Use minimize_scalar instead of grid search
                res = minimize_scalar(neg_exp_val, bounds=(0, 1), method='bounded')
                best_p = res.x
                best_val = -res.fun

                value_fn[t, w_idx, p_idx] = best_val
                policy[t, w_idx, p_idx] = best_p
    
    # Find closest grid points to current state
    w0_idx = np.searchsorted(wealth_grid, initial_wealth, side='left')
    w0_idx = np.clip(w0_idx, 0, len(wealth_grid)-1)
    p0_idx = np.searchsorted(price_grids[0], current_log_price, side='left')
    p0_idx = np.clip(p0_idx, 0, len(price_grids[0])-1)
    
    # Use interpolation for initial allocation
    optimal_initial_allocation = policy[0, w0_idx, p0_idx]
    
    # Compute utility curve if requested
    utility_curve_data = None
    if compute_utility_curve:
        allocs = np.linspace(0, 1, n_alloc_points)
        utilities = []
        
        # Find the exact grid indices used for the optimal allocation
        w0_idx = np.searchsorted(wealth_grid, initial_wealth, side='left')
        w0_idx = np.clip(w0_idx, 0, len(wealth_grid)-1)
        p0_idx = np.searchsorted(price_grids[0], current_log_price, side='left') 
        p0_idx = np.clip(p0_idx, 0, len(price_grids[0])-1)
        
        # Use the exact same grid point values as the optimization
        grid_wealth = wealth_grid[w0_idx]
        grid_log_price = price_grids[0][p0_idx]
        
        for alloc in allocs:
            # Use the grid values instead of the exact initial values
            cash = grid_wealth * (1 - alloc)
            btc_units = (grid_wealth * alloc) / np.exp(grid_log_price)
            next_mu = mu_seq[1]
            next_sigma = sigma_seq[1]
            next_grid = price_grids[1]
            next_pdf = (
                1/(next_sigma * np.sqrt(2*np.pi)) *
                np.exp(-0.5 * ((next_grid - next_mu)/next_sigma)**2)
            )
            next_pdf /= next_pdf.sum()
            
            exp_val = expected_future_value_vectorized(
                cash, btc_units, next_grid, wealth_grid, value_fn, 0, next_pdf
            )
            utilities.append(exp_val)
        
        utility_curve_data = (allocs, np.array(utilities))
        
        # Verify: find the allocation that gives maximum utility
        max_idx = np.argmax(utilities)
        max_alloc = allocs[max_idx]
        
        if verbose:
            print(f"Debug: Max utility allocation from curve: {max_alloc:.4f}")
            print(f"Debug: optimal allocation: {optimal_initial_allocation:.4f}")
            print(f"Debug: Using grid wealth: {grid_wealth:.2f} vs actual: {initial_wealth:.2f}")
            print(f"Debug: Using grid log_price: {grid_log_price:.6f} vs actual: {current_log_price:.6f}")
    
    if verbose:
        print(f"Initial wealth: {initial_wealth:.2f}, closest grid idx: {w0_idx}")
        print(f"Initial log_price: {current_log_price:.4f}, closest grid idx: {p0_idx}")
        # print(f"[DP] Interpolated optimal initial allocation: {optimal_initial_allocation:.4f}")
        print("Initial wealth:", initial_wealth)
        # print("Initial wealth idx:", w0_idx)
        print("Initial log price:", current_log_price)
        # print("Price grid:", price_grids[0])
        print("Initial log price idx:", p0_idx)
        print(f"Optimal allocation: {optimal_initial_allocation:.4f}")
    
    if compute_utility_curve:
        return policy, value_fn, optimal_initial_allocation, utility_curve_data
    else:
        return policy, value_fn, optimal_initial_allocation

def interpolate_policy(policy, wealth_grid, price_grid, initial_wealth, initial_log_price):
    # Find indices below and above initial wealth
    w_idx = np.searchsorted(wealth_grid, initial_wealth, side='left')
    w_idx_low = max(w_idx - 1, 0)
    w_idx_high = min(w_idx, len(wealth_grid) - 1)
    # Find indices below and above initial log price
    p_idx = np.searchsorted(price_grid, initial_log_price, side='left')
    p_idx_low = max(p_idx - 1, 0)
    p_idx_high = min(p_idx, len(price_grid) - 1)
    # Bilinear interpolation
    w0, w1 = wealth_grid[w_idx_low], wealth_grid[w_idx_high]
    p0, p1 = price_grid[p_idx_low], price_grid[p_idx_high]
    f00 = policy[0, w_idx_low, p_idx_low]
    f01 = policy[0, w_idx_low, p_idx_high]
    f10 = policy[0, w_idx_high, p_idx_low]
    f11 = policy[0, w_idx_high, p_idx_high]
    # Weights
    dw = (initial_wealth - w0) / (w1 - w0) if w1 != w0 else 0
    dp = (initial_log_price - p0) / (p1 - p0) if p1 != p0 else 0
    alloc = (
        f00 * (1-dw) * (1-dp) +
        f01 * (1-dw) * dp +
        f10 * dw * (1-dp) +
        f11 * dw * dp
    )
    return alloc