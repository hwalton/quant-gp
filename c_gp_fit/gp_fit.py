import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared

@dataclass(frozen=True)
class Config:
    data_path: str ='../0-data/bitcoin_combined_weekly_data.csv'
    log_pkl_path: str ='../1-log-fit/log_trend_params.pkl'
    gp_pkl_path: str ='gp_model.pkl'
    x_pred_pkl: str ='X_pred.npy'
    y_pred_pkl: str ='y_pred.npy'
    y_std_pkl: str ='y_std.npy'
    plot_path: str ='gp_output.png'
    points_into_future: int = 48*3
    y_limit: tuple =(4, 18)

def load_data(cfg: Config):
    df = pd.read_csv(cfg.data_path, sep=',')
    print("Columns:", df.columns.tolist())
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    y_all = np.log(df['price'].astype(float).values)
    
    # Use all data instead of trimming to cycles
    y = y_all
    X = np.arange(len(y))
    
    return X, y

def load_log_trend(cfg: Config):
    params = joblib.load(cfg.log_pkl_path)
    def trend(x):
        return params[0] * np.log(params[1] * x + 1) + params[2]
    return trend

def load_saved_predictions(cfg: Config):
    """Load saved GP predictions from files"""
    X_pred = np.load(cfg.x_pred_pkl)
    y_pred = np.load(cfg.y_pred_pkl) 
    y_std = np.load(cfg.y_std_pkl)
    return X_pred, y_pred, y_std

def build_kernel():
    return (
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1.0, 100.0)) +
        C(1.0, (1e-3, 1.0)) * ExpSineSquared(length_scale=10.0, periodicity=208.0,
                                             length_scale_bounds=(0.1, 100.0),
                                             periodicity_bounds=(150, 300)) +
        WhiteKernel(noise_level=1, noise_level_bounds=(0.005, 200.0))
    )

def fit_gp(X, residuals, kernel):
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=3,
        normalize_y=True
    )
    # Reshape X to 2D array
    X_reshaped = X.reshape(-1, 1)
    gp.fit(X_reshaped, residuals)
    
    # Print optimized kernel parameters
    print(f"\nOptimized kernel: {gp.kernel_}")
    print("-" * 40)
    
    return gp

def predict_gp(gp, X, trend_func, cfg: Config):
    X_pred = np.linspace(0, len(X) + cfg.points_into_future, 700).reshape(-1, 1)
    y_resid_pred, y_std = gp.predict(X_pred, return_std=True)
    y_pred = y_resid_pred + trend_func(X_pred.ravel())
    return X_pred, y_pred, y_std

def save_outputs(gp, X_pred, y_pred, y_std, cfg: Config):
    joblib.dump(gp, cfg.gp_pkl_path, compress=0)
    np.save(cfg.x_pred_pkl, X_pred)
    np.save(cfg.y_pred_pkl, y_pred)
    np.save(cfg.y_std_pkl, y_std)

def plot_results(cfg: Config):
    """Plot results using saved data - no need to refit GP"""
    # Load original data
    X, y = load_data(cfg)
    
    # Load timestamps for axis labeling
    df = pd.read_csv(cfg.data_path, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    timestamps = df['timestamp'].values
    
    # Load log trend function
    log_trend = load_log_trend(cfg)
    
    # Load saved predictions
    X_pred, y_pred, y_std = load_saved_predictions(cfg)
    
    # Only show the final half of the data for axis limits
    start_idx = len(X) - 364
    X_display = X[start_idx:]
    y_display = y[start_idx:]
    
    # But keep the full X_pred range for predictions
    X_pred_display = X_pred[X_pred.ravel() >= X_display[0]]
    y_pred_display = y_pred[X_pred.ravel() >= X_display[0]]
    y_std_display = y_std[X_pred.ravel() >= X_display[0]]

    # Convert log prices back to actual prices for plotting
    y_actual = np.exp(y)
    y_display_actual = np.exp(y_display)
    y_pred_display_actual = np.exp(y_pred_display)
    y_std_upper_actual = np.exp(y_pred_display + y_std_display)
    y_std_lower_actual = np.exp(y_pred_display - y_std_display)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all observed data points with smaller x's and lower zorder (behind axis)
    ax.plot(X, y_actual, 'kx', label='Historical BTC Prices', markersize=5, zorder=1)
    
    # Plot GP mean prediction and confidence band over full range
    ax.plot(X_pred_display, y_pred_display_actual, 'b-', label='Regression Mean', linewidth=2, zorder=2)
    ax.fill_between(X_pred_display.ravel(), y_std_lower_actual, y_std_upper_actual, 
                    alpha=0.2, label='Regression 1σ Confidence Interval (68%)', color='skyblue', zorder=2)
    
    # Set log10 scale for y-axis
    ax.set_yscale('log')
    
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('BTC Price (USD)', fontsize=18)
    ax.set_title('Probabilistic Regression on BTC Weekly Prices', fontsize=21)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines
    
    # Set axis limits to only show the final half
    ax.set_xlim(X_display[0], X_pred.ravel()[-1])
    
    # Set yearly ticks - find positions where year changes
    year_positions = []
    year_labels = []
    
    # Start from display range, but skip the first partial year
    for i in range(int(X_display[0]), int(X_pred.ravel()[-1]) + 1):
        if i < len(timestamps):
            current_year = pd.Timestamp(timestamps[i]).year
            # Only add year ticks when we hit January 1st (or close to it)
            current_date = pd.Timestamp(timestamps[i])
            if current_date.month == 1 and current_date.day <= 7:  # First week of January
                year_positions.append(i)
                year_labels.append(str(current_year))
        else:
            # For future predictions, add yearly ticks
            weeks_beyond = i - len(timestamps) + 1
            future_date = pd.Timestamp(timestamps[-1]) + pd.Timedelta(weeks=weeks_beyond)
            if future_date.month == 1 and future_date.day <= 7:  # First week of January
                year_positions.append(i)
                year_labels.append(str(future_date.year))
    
    # Set custom ticks and labels
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels)
    
    # Custom formatter to replace ,000 with k
    def currency_formatter(y, p):
        if y >= 1000:
            return f'{int(y/1000)}k'
        else:
            return f'{int(y)}'
    
    # Format y-axis with custom currency formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Calculate dynamic y-limits based on displayed data (using actual prices)
    y_min = min(y_display_actual.min(), y_std_lower_actual.min())
    y_max = max(y_display_actual.max(), y_std_upper_actual.max())
    ax.set_ylim(y_min * 0.9, y_max * 1.1)  # Add 10% padding on log scale
    
    fig.tight_layout()
    fig.savefig(cfg.plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    print(f"Plot saved to {cfg.plot_path}")

def main():
    cfg = Config()

    X, y = load_data(cfg)
    log_trend = load_log_trend(cfg)
    residuals = y - log_trend(X.ravel())

    kernel = build_kernel()
    gp = fit_gp(X, residuals, kernel)
    X_pred, y_pred, y_std = predict_gp(gp, X, log_trend, cfg)

    save_outputs(gp, X_pred, y_pred, y_std, cfg)
    plot_results(cfg)

if __name__ == '__main__':    
    main()
