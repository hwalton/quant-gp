"""
Logarithmic trend fitting for Bitcoin price data.
"""
import numpy as np
import joblib
from scipy.optimize import curve_fit
from config import GPModelConfig
from data.loader import load_btc_data

def log_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Logarithmic trend function: a * log(b*x + 1) + c
    
    Args:
        x: Time indices
        a, b, c: Parameters to fit
        
    Returns:
        Log price predictions
    """
    z = b * x + 1
    z = np.clip(z, 1e-8, np.inf)  # Avoid log(0) or negative
    return a * np.log(z) + c

def fit_log_trend(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit logarithmic trend to price data.
    
    Args:
        X: Time indices
        y: Log prices
        
    Returns:
        Fitted parameters [a, b, c]
    """
    params, _ = curve_fit(
        log_func,
        X,
        y,
        p0=[1, 0.01, 1],
        bounds=([0, 1e-6, -np.inf], [np.inf, 1.0, np.inf]),
        maxfev=5000
    )
    return params

def save_log_params(params: np.ndarray, config: GPModelConfig) -> None:
    """Save log trend parameters to file."""
    output_path = config.log_fit_dir / config.log_params_file
    joblib.dump(params, output_path)
    print(f"Saved log trend params to {output_path}")

def load_log_params(config: GPModelConfig) -> np.ndarray:
    """Load log trend parameters from file."""
    params_path = config.log_fit_dir / config.log_params_file
    if not params_path.exists():
        # Try old location
        old_params_path = config.base_dir.parent / "1-log-fit" / config.log_params_file
        if old_params_path.exists():
            params = joblib.load(old_params_path)
            # Save to new location
            save_log_params(params, config)
            return params
        else:
            raise FileNotFoundError(f"Log trend parameters not found at {params_path}")
    
    return joblib.load(params_path)

def create_log_trend_function(params: np.ndarray):
    """Create a log trend function from fitted parameters."""
    def trend(x):
        return log_func(x, *params)
    return trend

def run_log_fit(config: GPModelConfig) -> np.ndarray:
    """
    Run the complete log fitting pipeline.
    
    Returns:
        Fitted parameters
    """
    print("Loading Bitcoin data...")
    X, y = load_btc_data(config)
    
    print("Fitting logarithmic trend...")
    params = fit_log_trend(X, y)
    
    print(f"Fitted log curve params: a={params[0]:.4f}, b={params[1]:.6f}, c={params[2]:.4f}")
    
    save_log_params(params, config)
    
    return params

if __name__ == "__main__":
    config = GPModelConfig()
    params = run_log_fit(config)
