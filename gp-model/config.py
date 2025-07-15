"""
Shared configuration for the QuantGP analysis pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class GPModelConfig:
    """Configuration for the entire GP model pipeline."""
    
    # Paths
    base_dir: Path = Path(__file__).parent
    data_dir: Path = base_dir / "step1_data"
    log_fit_dir: Path = base_dir / "step2_log_fit" 
    gp_fit_dir: Path = base_dir / "step3_gp_fit"
    portfolio_dir: Path = base_dir / "step4_portfolio"
    outputs_dir: Path = base_dir / "outputs"
    
    # Data files
    btc_data_file: str = "btc_weekly_prices.csv"
    
    # Model files
    log_params_file: str = "log_trend_params.pkl"
    gp_model_file: str = "gp_model.pkl"
    X_pred_file: str = "X_pred.npy"
    y_pred_file: str = "y_pred.npy" 
    y_std_file: str = "y_std.npy"
    
    # Output files
    gp_plot_file: str = "gp_output.png"
    utility_func_plot: str = "utility_func.png"
    utility_curve_plot: str = "utility_curve.png"
    wealth_dist_plot: str = "wealth_distribution.png"
    utility_dist_plot: str = "utility_distribution.png"
    
    # Model parameters
    cycle_length: int = 208
    points_into_future: int = 48 * 7
    y_limit: Tuple[float, float] = (4, 18)
    
    # Portfolio parameters
    initial_wealth: float = 1000
    predict_index_offset: int = 4  # 4 weeks into the future
    utility_function: str = 'tanh_custom'  # Options: 'identity', 'log', 'sqrt', 'step', 'smooth_step', 'sigmoid', 'tanh', 'tanh_custom', 'crra'
    
    # Utility function specific parameters
    sigmoid_k: float = 25.0
    w0: float = 0.98
    step_threshold: float = 999
    step_steepness: float = 100.0
    crra_gamma: float = 2.0
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_dir, self.log_fit_dir, self.gp_fit_dir, 
                        self.portfolio_dir, self.outputs_dir]:
            dir_path.mkdir(exist_ok=True)
