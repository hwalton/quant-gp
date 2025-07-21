from dataclasses import dataclass

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')
# LOG_PKL_PATH = os.path.join(PROJECT_ROOT, 'b_log_fit', 'log_trend_params.pkl')
# GP_PKL_PATH = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'variational_gp_model.pth')
X_PRED_PKL = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'X_pred.npy')
Y_PRED_PKL = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'y_pred.npy')
Y_STD_PKL = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'y_std.npy')

@dataclass(frozen=True)
class Config:
    x_pred_pkl: str = X_PRED_PKL
    y_pred_pkl: str = Y_PRED_PKL
    y_std_pkl: str = Y_STD_PKL
    price_csv: str = DATA_PATH
    initial_wealth: float = 1000
    preference_curve: str = ['step', 'coordinate_points', 'log_risk_averse', 'general_risk_level', 'max_expected_wealth'][1]
    gamma: float = 5  # Only used if utility_function is 'crra'
    step_threshold: float = 1100
    step_steepness: float = 100.0

    horizon_weeks: int = 4
    rebalance_every: int = 4  # weeks

    optimisation_method: str = ['bayesian', 'bayesian_with_refinement', 'forest_minimize'][2]  # Add type annotation
    
    n_calls_optimiser: int = 15  # Number of calls to the optimiser