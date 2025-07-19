from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    x_pred_pkl: str = '../2-gp_fit/X_pred.npy'
    y_pred_pkl: str = '../2-gp_fit/y_pred.npy'
    y_std_pkl: str = '../2-gp_fit/y_std.npy'
    price_csv: str = '../0-data/bitcoin_combined_weekly_data.csv'
    initial_wealth: float = 600
    preference_curve: str = ['step', 'coordinate_points', 'log_risk_averse', 'general_risk_level'][1]
    gamma: float = 5  # Only used if utility_function is 'crra'
    step_threshold: float = 1100
    step_steepness: float = 100.0

    horizon_weeks: int = 4*2
    rebalance_every: int = 4  # weeks

    optimisation_method = ['bayesian', 'bayesian_with_refinement'][1]
    
    n_calls_optimiser: int = 15  # Number of calls to the optimiser