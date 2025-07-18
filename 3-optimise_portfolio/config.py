from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    x_pred_pkl: str = '../2-gp_fit/X_pred.npy'
    y_pred_pkl: str = '../2-gp_fit/y_pred.npy'
    y_std_pkl: str = '../2-gp_fit/y_std.npy'
    price_csv: str = '../0-data/bitcoin_combined_weekly_data.csv'
    initial_wealth: float = 1400
    preference_curve: str = ['step_below_1000', 'step_above_1000', 'not_below_920', 'get_to_4500', 'v_shape', 'risk_averse', 'linear', 'coordinate_points', 'log_risk_averse', 'power_risk_averse'][7]
    gamma: float = 1  # Only used if utility_function is 'crra'
    step_threshold: float = 1100
    step_steepness: float = 100.0

    horizon_weeks: int = 4
    rebalance_every: int = 4  # weeks

    optimisation_method = ['bayesian', 'bayesian_with_refinement'][1]
    
    n_calls_optimiser: int = 15  # Number of calls to the optimiser