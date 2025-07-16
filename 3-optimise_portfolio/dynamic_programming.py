import numpy as np
from scipy.optimize import minimize_scalar

def expected_future_value_vectorized(cash, btc_units, next_grid, wealth_grid, value_fn, t, next_pdf):
    # Vectorized computation of next wealth and lookup of value function
    next_wealth = cash + btc_units * np.exp(next_grid)
    # Find closest wealth grid indices for all next_wealth
    next_w_idx = np.abs(wealth_grid[:, None] - next_wealth).argmin(axis=0)
    # Gather value function for all (next_w_idx, next_p_idx)
    v = np.array([value_fn[(t+1, w_idx, p_idx)] for p_idx, w_idx in enumerate(next_w_idx)])
    # Weighted sum for expectation
    exp_val = np.sum(v * next_pdf)
    return exp_val

def dynamic_programming_policy(
    mu_seq, sigma_seq, utility_func, initial_wealth, current_log_price,
    n_steps=12, price_grid_size=101, wealth_grid_size=101, verbose=True
):
    """
    Dynamic programming for optimal BTC allocation over a 1-year horizon (monthly rebalancing),
    using minimize_scalar for allocation optimization.
    """
    price_grids = []
    for k in range(n_steps):
        grid = np.linspace(mu_seq[k] - 4*sigma_seq[k], mu_seq[k] + 4*sigma_seq[k], price_grid_size)
        price_grids.append(grid)
    min_wealth = initial_wealth * 0.2
    max_wealth = initial_wealth * 5.0
    wealth_grid = np.linspace(min_wealth, max_wealth, wealth_grid_size)
    value_fn = {}
    policy = {}
    # Terminal step: utility of final wealth for all (wealth, log_price)
    for w_idx, wealth in enumerate(wealth_grid):
        for p_idx, log_price in enumerate(price_grids[-1]):
            value_fn[(n_steps-1, w_idx, p_idx)] = utility_func(wealth)
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

                value_fn[(t, w_idx, p_idx)] = best_val
                policy[(t, w_idx, p_idx)] = best_p
                if verbose and w_idx % (wealth_grid_size // 5) == 0 and p_idx % (price_grid_size // 5) == 0:
                    print(f"  [DP] t={t}, wealth_idx={w_idx}, log_price_idx={p_idx}, best_p={best_p:.2f}, best_val={best_val:.4f}")
    # Find closest grid points to current state
    w0_idx = np.argmin(np.abs(wealth_grid - initial_wealth))
    p0_idx = np.argmin(np.abs(price_grids[0] - current_log_price))
    optimal_initial_allocation = policy[(0, w0_idx, p0_idx)]
    if verbose:
        print(f"[DP] Initial wealth: {initial_wealth:.2f}, closest grid idx: {w0_idx}")
        print(f"[DP] Initial log_price: {current_log_price:.4f}, closest grid idx: {p0_idx}")
        print(f"[DP] Optimal initial allocation: {optimal_initial_allocation:.2f}")
    return policy, value_fn, optimal_initial_allocation