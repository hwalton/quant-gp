import numpy as np

def dynamic_programming_policy(
    mu_seq, sigma_seq, utility_func, initial_wealth, current_log_price,
    n_steps=12, price_grid_size=101, alloc_grid_size=21, verbose=True
):
    """
    Dynamic programming for optimal BTC allocation over a 1-year horizon (monthly rebalancing).
    """
    alloc_grid = np.linspace(0, 1, alloc_grid_size)
    price_grids = []
    for k in range(n_steps):
        grid = np.linspace(mu_seq[k] - 4*sigma_seq[k], mu_seq[k] + 4*sigma_seq[k], price_grid_size)
        price_grids.append(grid)
    value_fn = {}
    policy = {}
    # Terminal step
    for idx, log_price in enumerate(price_grids[-1]):
        btc_units = initial_wealth / np.exp(current_log_price)
        final_wealth = btc_units * np.exp(log_price)
        value_fn[(n_steps-1, idx)] = utility_func(final_wealth)
    # Backward induction
    for t in reversed(range(n_steps-1)):
        for i, log_price in enumerate(price_grids[t]):
            best_val = -np.inf
            best_p = 0.0
            for p in alloc_grid:
                # For each possible next log-price, compute next wealth
                cash = initial_wealth * (1 - p)
                btc_units = (initial_wealth * p) / np.exp(log_price)
                next_mu = mu_seq[t+1]
                next_sigma = sigma_seq[t+1]
                next_grid = price_grids[t+1]
                next_pdf = (
                    1/(next_sigma * np.sqrt(2*np.pi)) *
                    np.exp(-0.5 * ((next_grid - next_mu)/next_sigma)**2)
                )
                next_pdf /= next_pdf.sum()
                # Wealth at next step for each possible next log-price
                next_wealth = cash + btc_units * np.exp(next_grid)
                # Interpolate value function at next step using next_wealth and next_log_price
                # (This requires a 2D value function: value_fn[(t+1, j, w_idx)])
                # For now, you can try using utility_func(next_wealth) directly:
                next_v = utility_func(next_wealth)
                exp_val = np.sum(next_v * next_pdf)
                if exp_val > best_val:
                    best_val = exp_val
                    best_p = p
            value_fn[(t, i)] = best_val
            policy[(t, i)] = best_p
            if verbose:
                print(f"  [DP] t={t}, log_price_idx={i}, best_p={best_p:.2f}, best_val={best_val:.4f}")
    idx0 = np.argmin(np.abs(price_grids[0] - current_log_price))
    optimal_initial_allocation = policy[(0, idx0)]
    if verbose:
        print(f"[DP] Initial log_price: {current_log_price:.4f}, closest grid idx: {idx0}")
        print(f"[DP] Optimal initial allocation: {optimal_initial_allocation:.2f}")
    return policy, value_fn, optimal_initial_allocation