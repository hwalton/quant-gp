from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    x_pred_pkl: str = '../2-gp_fit/X_pred.npy'
    y_pred_pkl: str = '../2-gp_fit/y_pred.npy'
    y_std_pkl: str = '../2-gp_fit/y_std.npy'
    log_csv: str = '../0-data/btc_weekly_prices.csv'
    initial_wealth: float = 1000.0
    utility_function: str = ['step', 'smooth_step', 'sigmoid', 'tanh', 'tanh_custom', 'identity', 'linear', 'log', 'sqrt', 'crra'][4]
    gamma: float = 1.5  # Only used if utility_function is 'crra'
    sigmoid_k: float = 25.0
    w0: float = 0.98
    step_threshold: float = 1100
    step_steepness: float = 100.0

    horizon_weeks: int = 4*4
    rebalance_every: int = 4  # weeks

    n_calls_optimiser: int = 25  # Number of calls to the optimiser