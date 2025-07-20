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
    data_path: str ='../a_data/bitcoin_combined_weekly_data.csv'
    log_pkl_path: str ='../b_log_fit/log_trend_params.pkl'
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

def plot_full_dataset_results(cfg: Config):
    """Plot GP results showing the FULL dataset for debugging"""
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
    
    # FIX: Flatten X_pred if it's 2D (sklearn GP returns 2D arrays)
    if X_pred.ndim > 1:
        X_pred = X_pred.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_std.ndim > 1:
        y_std = y_std.flatten()

    # Convert log prices back to actual prices for plotting
    y_actual = np.exp(y)
    y_pred_actual = np.exp(y_pred)
    y_std_upper_actual = np.exp(y_pred + y_std)
    y_std_lower_actual = np.exp(y_pred - y_std)

    # Create larger figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # TOP PLOT: Full dataset in log space (easier to see residuals)
    ax1.plot(X, y, 'kx', label='Historical BTC Log Prices', markersize=3, alpha=0.7, zorder=1)
    ax1.plot(X, log_trend(X), 'r-', label='Log Trend', linewidth=2, zorder=2)
    ax1.plot(X_pred, y_pred, 'b-', label='GP (Log Space)', linewidth=2, zorder=3)
    ax1.fill_between(X_pred, y_pred - y_std, y_pred + y_std, 
                    alpha=0.2, label='GP 1σ (Log Space)', color='skyblue', zorder=2)
    
    ax1.set_xlabel('Time (weeks)', fontsize=14)
    ax1.set_ylabel('Log BTC Price', fontsize=14)
    ax1.set_title('GP Fit: Full Dataset (Log Space) - DEBUGGING VIEW', fontsize=16)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis to show years
    year_positions = []
    year_labels = []
    for i in range(0, len(timestamps), 52):  # Every ~52 weeks (1 year)
        if i < len(timestamps):
            year = pd.Timestamp(timestamps[i]).year
            year_positions.append(i)
            year_labels.append(str(year))
    
    ax1.set_xticks(year_positions)
    ax1.set_xticklabels(year_labels)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # BOTTOM PLOT: Full dataset in actual price space
    ax2.plot(X, y_actual, 'kx', label='Historical BTC Prices', markersize=3, alpha=0.7, zorder=1)
    ax2.plot(X_pred, y_pred_actual, 'b-', label='GP Mean', linewidth=2, zorder=2)
    ax2.fill_between(X_pred, y_std_lower_actual, y_std_upper_actual, 
                    alpha=0.2, label='GP 1σ Confidence', color='skyblue', zorder=2)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (weeks)', fontsize=14)
    ax2.set_ylabel('BTC Price (USD)', fontsize=14)
    ax2.set_title('GP Fit: Full Dataset (Actual Prices)', fontsize=16)
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Custom formatter for price axis
    def currency_formatter(y, p):
        if y >= 1000:
            return f'{int(y/1000)}k'
        else:
            return f'{int(y)}'
    
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
    ax2.set_xticks(year_positions)
    ax2.set_xticklabels(year_labels)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save with different name
    full_plot_path = cfg.plot_path.replace('.png', '_full_dataset.png')
    fig.savefig(full_plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    print(f"Full dataset plot saved to {full_plot_path}")
    
    # Print some diagnostics
    print(f"\nDiagnostics:")
    print(f"  Data range: {X.min():.0f} to {X.max():.0f} weeks")
    print(f"  Log price range: {y.min():.2f} to {y.max():.2f}")
    print(f"  Residuals std: {(y - log_trend(X)).std():.4f}")
    print(f"  Prediction range: {X_pred.min():.0f} to {X_pred.max():.0f} weeks")

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
    plot_full_dataset_results(cfg)

if __name__ == '__main__':    
    main()
