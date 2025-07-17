import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
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
    utility_function: str = 'crra'  # Options: 'log', 'identity', 'sqrt', 'crra', etc.
    gamma: float = 0.8  # Only used if utility_function is 'crra'
    sigmoid_k: float = 25.0
    w0: float = 0.98
    step_threshold: float = 1001
    step_steepness: float = 100.0

def get_utility_func(cfg: Config):
    if cfg.utility_function == 'identity':
        return lambda w: w
    elif cfg.utility_function == 'log':
        return lambda w: np.log(w) if w > 0 else -np.inf
    elif cfg.utility_function == 'sqrt':
        return lambda w: np.sqrt(w) if w >= 0 else 0.0
    elif cfg.utility_function == 'step':
        return lambda w: 1.0 if w > cfg.step_threshold else 0.0
    elif cfg.utility_function == 'smooth_step':
        def smooth_step(w):
            x = -cfg.step_steepness * (w - cfg.step_threshold)
            if x > 500:
                return 0.0
            elif x < -500:
                return 1.0
            else:
                return 1 / (1 + np.exp(x))
        return smooth_step
    elif cfg.utility_function == 'sigmoid':
        return lambda w: 1 / (1 + np.exp(-cfg.sigmoid_k * (w - cfg.w0)))
    elif cfg.utility_function == 'crra':
        gamma = cfg.gamma
        return lambda w: (w**(1 - gamma) - 1) / (1 - gamma) if w > 0 else -np.inf
    else:
        raise ValueError(f"Unsupported utility function: {cfg.utility_function}")

def load_gp_predictions(cfg: Config, months: int = 12):
    X_pred = np.load(cfg.x_pred_pkl)
    y_pred = np.load(cfg.y_pred_pkl)
    y_std = np.load(cfg.y_std_pkl)

    df = pd.read_csv(cfg.log_csv, sep=';').sort_values(by='timestamp')
    current_log_price = np.log(df['close'].astype(float).values[-1])

    current_index = len(df)
    target_index = np.searchsorted(X_pred.ravel(), current_index)
    mu_seq = y_pred[target_index : target_index + months]
    sigma_seq = y_std[target_index : target_index + months]

    return mu_seq, sigma_seq, current_log_price

def simulate_terminal_wealth(p, mu_seq, sigma_seq, current_log_price, cfg, n_samples=1000):
    T = len(p)
    log_returns = np.random.normal(loc=mu_seq, scale=sigma_seq, size=(n_samples, T))
    
    # Build cumulative log-price paths
    log_prices = np.cumsum(np.hstack([current_log_price * np.ones((n_samples, 1)), log_returns]), axis=1)

    wealth = np.ones(n_samples) * cfg.initial_wealth
    for t in range(T):
        cash = wealth * (1 - p[t])
        btc_units = (wealth * p[t]) / np.exp(log_prices[:, t])
        wealth = cash + btc_units * np.exp(log_prices[:, t+1])
    
    return wealth  # shape (n_samples,)

def objective(p, mu_seq, sigma_seq, current_log_price, cfg, utility=None, n_samples=1000):
    if utility is None:
        utility = get_utility_func(cfg)
    
    final_wealth = simulate_terminal_wealth(p, mu_seq, sigma_seq, current_log_price, cfg, n_samples=n_samples)
    utilities = np.array([utility(w) for w in final_wealth])
    return np.mean(utilities)

def run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, months=12):
    # Search space: p_t in [0, 1] for each month
    search_space = [Real(0.0, 1.0, name=f"p{i}") for i in range(months)]
    
    utility = get_utility_func(cfg)

    @use_named_args(search_space)
    def objective_wrapped(**kwargs):
        p = np.array([kwargs[f"p{i}"] for i in range(months)])
        util = objective(p, mu_seq, sigma_seq, current_log_price, cfg, utility=utility)
        return -util  # Negative for minimisation

    result = gp_minimize(
        func=objective_wrapped,
        dimensions=search_space,
        n_calls=50,
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
    months = 12

    mu_seq, sigma_seq, current_log_price = load_gp_predictions(cfg, months=months)

    # # Initial allocation guess (e.g. 50% BTC each month)
    # p_init = np.full(months, 0.5)

    # # Evaluate objective at initial guess
    # expected_util = objective(p_init, mu_seq, sigma_seq, current_log_price, cfg)
    # print(f"Expected Utility: {expected_util:.4f}")
    # print(f"Initial Allocation: {p_init}")

    # Run Bayesian optimisation
    print("\nRunning Bayesian Optimisation...")
    optimal_p, max_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, months)

    print(f"\nOptimal allocation vector:")
    print(np.round(optimal_p, 3))
    print(f"Maximum expected utility: {max_util:.4f}")

if __name__ == '__main__':
    main()
