"""
Gaussian Process fitting for Bitcoin price prediction.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared

from config import GPModelConfig
from step1_data.loader import load_btc_data
from step2_log_fit.fit import load_log_params, create_log_trend_function

def build_kernel():
    """Build the GP kernel for Bitcoin price modeling."""
    return (
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=200.0, length_scale_bounds=(10.0, 2000.0)) +
        C(1.0, (1e-3, 1.0)) * ExpSineSquared(length_scale=10.0, periodicity=208.0,
                                             length_scale_bounds=(1.0, 100.0),
                                             periodicity_bounds=(150, 300)) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(0.05, 200.0))
    )

def fit_gp(X: np.ndarray, residuals: np.ndarray) -> GaussianProcessRegressor:
    """
    Fit Gaussian Process to residuals after log trend removal.
    
    Args:
        X: Time indices (reshaped for sklearn)
        residuals: Log prices minus log trend
        
    Returns:
        Fitted GP model
    """
    kernel = build_kernel()
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=3,
        normalize_y=True
    )
    
    gp.fit(X, residuals)
    return gp

def predict_gp(gp: GaussianProcessRegressor, X: np.ndarray, 
               trend_func, config: GPModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions from fitted GP model.
    
    Args:
        gp: Fitted GP model
        X: Original time indices
        trend_func: Log trend function
        config: Configuration
        
    Returns:
        tuple: (X_pred, y_pred, y_std)
    """
    # Create prediction points
    X_pred = np.linspace(0, len(X) + config.points_into_future, 700).reshape(-1, 1)
    
    # Predict residuals
    y_resid_pred, y_std = gp.predict(X_pred, return_std=True)
    
    # Add back the trend
    y_pred = y_resid_pred + trend_func(X_pred.ravel())
    
    return X_pred, y_pred, y_std

def save_gp_outputs(gp: GaussianProcessRegressor, X_pred: np.ndarray, 
                   y_pred: np.ndarray, y_std: np.ndarray, config: GPModelConfig) -> None:
    """Save GP model and predictions."""
    # Save model
    model_path = config.gp_fit_dir / config.gp_model_file
    joblib.dump(gp, model_path, compress=0)
    
    # Save predictions
    np.save(config.gp_fit_dir / config.X_pred_file, X_pred)
    np.save(config.gp_fit_dir / config.y_pred_file, y_pred)
    np.save(config.gp_fit_dir / config.y_std_file, y_std)
    
    print(f"Saved GP model and predictions to {config.gp_fit_dir}")

def plot_gp_results(X: np.ndarray, y: np.ndarray, X_pred: np.ndarray, 
                   y_pred: np.ndarray, y_std: np.ndarray, 
                   log_trend_func, config: GPModelConfig) -> None:
    """Create and save GP visualization."""
    fig, ax = plt.subplots(figsize=(30, 18))
    
    ax.plot(X, y, 'kx', label='Observed BTC prices')
    ax.plot(X_pred, y_pred, 'b-', label='GP mean prediction')
    ax.plot(X_pred, log_trend_func(X_pred.ravel()), 'g--', label='Log trend fit')
    ax.fill_between(X_pred.ravel(), y_pred - y_std, y_pred + y_std, 
                   alpha=0.2, label='1σ confidence')
    
    ax.set_xlabel('Weeks since start')
    ax.set_ylabel('Log(BTC Price (USD))')
    ax.set_title('Gaussian Process Regression on BTC Weekly Prices')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(*config.y_limit)
    
    fig.tight_layout()
    
    # Save to outputs directory
    plot_path = config.outputs_dir / config.gp_plot_file
    fig.savefig(plot_path)
    plt.close('all')
    
    print(f"Saved GP plot to {plot_path}")

def load_gp_outputs(config: GPModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GP predictions from files."""
    X_pred = np.load(config.gp_fit_dir / config.X_pred_file)
    y_pred = np.load(config.gp_fit_dir / config.y_pred_file)
    y_std = np.load(config.gp_fit_dir / config.y_std_file)
    return X_pred, y_pred, y_std

def run_gp_fit(config: GPModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the complete GP fitting pipeline.
    
    Returns:
        tuple: (X_pred, y_pred, y_std)
    """
    print("Loading data and log trend...")
    X, y = load_btc_data(config)
    log_params = load_log_params(config)
    log_trend_func = create_log_trend_function(log_params)
    
    # Calculate residuals
    residuals = y - log_trend_func(X)
    
    print("Fitting Gaussian Process...")
    X_reshaped = X.reshape(-1, 1)
    gp = fit_gp(X_reshaped, residuals)
    
    print("Generating predictions...")
    X_pred, y_pred, y_std = predict_gp(gp, X, log_trend_func, config)
    
    print("Saving outputs...")
    save_gp_outputs(gp, X_pred, y_pred, y_std, config)
    
    print("Creating visualization...")
    plot_gp_results(X, y, X_pred, y_pred, y_std, log_trend_func, config)
    
    return X_pred, y_pred, y_std

if __name__ == "__main__":
    config = GPModelConfig()
    X_pred, y_pred, y_std = run_gp_fit(config)
