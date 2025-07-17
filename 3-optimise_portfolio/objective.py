import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
from itertools import product
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

@dataclass(frozen=True)
class Config:
    x_pred_pkl: str = '../2-gp_fit/X_pred.npy'
    y_pred_pkl: str = '../2-gp_fit/y_pred.npy'
    y_std_pkl: str = '../2-gp_fit/y_std.npy'
    log_csv: str = '../0-data/btc_weekly_prices.csv'
    initial_wealth: float = 1000.0
    utility_function: str = ['step', 'smooth_step', 'sigmoid', 'tanh', 'tanh_custom', 'identity', 'linear', 'log', 'sqrt', 'crra'][1]
    gamma: float = 1.5  # Only used if utility_function is 'crra'
    sigmoid_k: float = 25.0
    w0: float = 0.98
    step_threshold: float = 1200
    step_steepness: float = 100.0

    horizon_weeks: int = 8
    rebalance_every: int = 4  # weeks

def get_utility_func(cfg: Config):
    if cfg.utility_function == 'identity' or cfg.utility_function == 'linear':
        return lambda w: w
    elif cfg.utility_function == 'log':
        return lambda w: np.log(w) if w > 0 else -np.inf
    elif cfg.utility_function == 'sqrt':
        return lambda w: np.sqrt(w) if w >= 0 else 0
    elif cfg.utility_function == 'step':
        return lambda w: 1.0 if w > cfg.step_threshold else 0.0
    elif cfg.utility_function == 'smooth_step':
        # Add numerical stability to prevent overflow
        def smooth_step(w):
            x = -cfg.step_steepness * (w - cfg.step_threshold)
            if x > 500:  # Prevent overflow
                return 0.0
            elif x < -500:
                return 1.0
            else:
                return 1 / (1 + np.exp(x))
        return smooth_step
    elif cfg.utility_function == 'sigmoid':
        return lambda w: 1 / (1 + np.exp(-cfg.sigmoid_k * (w - cfg.w0)))
    elif cfg.utility_function == 'tanh':
        return lambda w: np.tanh((w - 800) / 20)
    elif cfg.utility_function == 'tanh_custom':
        return lambda w: np.tanh((w - 700) / 150) + 1 + w / 5000
    elif cfg.utility_function == 'crra':
        gamma = cfg.gamma
        return lambda w: (w**(1-gamma) - 1) / (1-gamma) if gamma != 1 else np.log(w)
    else:
        raise ValueError(f"Unsupported utility function: {cfg.utility_function}")

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

    # Build 1D grids for each future rebalance log-price x_t
    grid_limits = [
        np.linspace(mu_seq[t] - 4*sigma_seq[t], mu_seq[t] + 4*sigma_seq[t], 30)
        for t in range(T)
    ]
    dx = [g[1] - g[0] for g in grid_limits]

    # Cartesian product of all grid points: shape (n_points, T)
    all_paths = list(product(*grid_limits))

    total = 0.0
    for x_path in all_paths:
        wealth = cfg.initial_wealth
        x_prev = current_log_price

        for t in range(T):
            x_now = x_path[t]
            price_prev = np.exp(x_prev)
            price_now = np.exp(x_now)

            cash = wealth * (1 - p[t])
            btc = (wealth * p[t]) / price_prev
            wealth = cash + btc * price_now

            x_prev = x_now  # advance to next step

        u = utility(wealth)

        # Compute joint PDF (assume independence for now)
        prob_density = np.prod([
            norm.pdf(x_path[t], loc=mu_seq[t], scale=sigma_seq[t])
            for t in range(T)
        ])

        volume_element = np.prod(dx)
        total += u * prob_density * volume_element

    return total

def run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, months=12):
    # Search space: p_t in [0, 1] for each month
    search_space = [Real(0.0, 1.0, name=f"p{i}") for i in range(months)]
    
    utility = get_utility_func(cfg)

    @use_named_args(search_space)
    def objective_wrapped(**kwargs):
        p = np.array([kwargs[f"p{i}"] for i in range(months)])
        util = objective_numerical_integral(p, mu_seq, sigma_seq, current_log_price, cfg)
        return -util  # Negative for minimisation

    result = gp_minimize(
        func=objective_wrapped,
        dimensions=search_space,
        n_calls=25,
        n_initial_points=10,
        acq_func="EI",  # Expected improvement
        random_state=42,
        verbose=True
    )

    optimal_p = np.array(result.x)
    max_utility = -result.fun

    return optimal_p, max_utility, result

def main():
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


if __name__ == '__main__':
    main()
