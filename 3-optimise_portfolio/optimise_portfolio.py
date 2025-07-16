import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dynamic_programming import dynamic_programming_policy

@dataclass(frozen=True)
class Config:
    # --- Data and File Paths ---
    x_pred_pkl: str = '../2-gp_fit/X_pred.npy'
    y_pred_pkl: str = '../2-gp_fit/y_pred.npy'
    ystd_pkl: str = '../2-gp_fit/y_std.npy'
    log_csv: str = '../0-data/btc_weekly_prices.csv'
    
    # --- Portfolio and Prediction Settings ---
    initial_wealth: float = 1000
    predict_index_offset: int = 52
    # y_limit: tuple = (4, 18) #TODO: Delete
    
    # --- Utility Function Configuration ---
    utility_function: str = ['step', 'smooth_step', 'sigmoid', 'tanh', 'tanh_custom', 'identity', 'linear', 'log', 'sqrt', 'crra'][0]  # step
    # Step function parameters
    step_threshold: float = 900
    step_steepness: float = 100.0  # Controls how sharp the transition is (for smooth_step)
    # Sigmoid parameters
    sigmoid_k: float = 25.0
    w0: float = 0.98
    # CRRA parameter (gamma) is hardcoded in get_utility_func
    
    # --- Dynamic Programming Configuration ---
    dp_n_steps: int = 2  # Number of time steps for DP
    dp_price_grid_size: int = 73  # Grid size for price discretization
    dp_wealth_grid_size: int = 73  # Grid size for wealth discretization
    dp_weeks_per_step: int = 4  # Weeks between time steps (4 = monthly)
    dp_verbose: bool = False  # Print debug information
    dp_compute_utility_curve: bool = True  # Compute utility curve for plotting
    dp_n_alloc_points: int = 100  # Number of allocation points for utility curve
    dp_moving_avg_window: int = 5  # Moving average window for utility curve smoothing

def load_data(cfg: Config):
    X_pred = np.load(cfg.x_pred_pkl)
    y_pred = np.load(cfg.y_pred_pkl)
    y_std = np.load(cfg.ystd_pkl)
    df = pd.read_csv(cfg.log_csv, sep=';').sort_values(by='timestamp')
    y_actual = np.log(df['close'].astype(float).values)
    return X_pred, y_pred, y_std, y_actual

def get_utility_func(cfg: Config):
    if cfg.utility_function == 'identity' or cfg.utility_function == 'linear':
        return lambda w: w
    elif cfg.utility_function == 'log':
        return lambda w: np.log(w) if w > 0 else -np.inf
    elif cfg.utility_function == 'sqrt':
        return lambda w: np.sqrt(w) if w >= 0 else 0
    elif cfg.utility_function == 'step':
        return lambda w: 1.0 if w > cfg.step_threshold else 0.0
    elif cfg.utility_function == 'smooth_step':
        # Add numerical stability to prevent overflow
        def smooth_step(w):
            x = -cfg.step_steepness * (w - cfg.step_threshold)
            if x > 500:  # Prevent overflow
                return 0.0
            elif x < -500:
                return 1.0
            else:
                return 1 / (1 + np.exp(x))
        return smooth_step
    elif cfg.utility_function == 'sigmoid':
        return lambda w: 1 / (1 + np.exp(-cfg.sigmoid_k * (w - cfg.w0)))
    elif cfg.utility_function == 'tanh':
        return lambda w: np.tanh((w - 800) / 20)
    elif cfg.utility_function == 'tanh_custom':
        return lambda w: np.tanh((w - 700) / 150) + 1 + w / 5000
    elif cfg.utility_function == 'crra':
        gamma = 0.8
        return lambda w: (w**(1-gamma) - 1) / (1-gamma) if gamma != 1 else np.log(w)
    else:
        raise ValueError(f"Unsupported utility function: {cfg.utility_function}")

def expected_utility(pp, mu_log_price, sigma_log_price, current_log_price, cfg: Config):
    utility = get_utility_func(cfg)
    
    def integrand(pred_log_price):
        # Portfolio calculation
        cash_portion = cfg.initial_wealth * (1 - pp)
        btc_units = (cfg.initial_wealth * pp) / np.exp(current_log_price)
        btc_value = btc_units * np.exp(pred_log_price)
        portfolio_value = cash_portion + btc_value
        
        return utility(portfolio_value) * norm.pdf(pred_log_price, loc=mu_log_price, scale=sigma_log_price)
    
    result, _ = quad(integrand, mu_log_price - 6*sigma_log_price, mu_log_price + 6*sigma_log_price)
    return -result  # Negative for minimization

def compute_gp_stats(X_pred, y_pred, y_std, y_actual, cfg: Config):
    current_log_price = y_actual[-1]
    target_index = np.searchsorted(X_pred.ravel(), len(y_actual) + cfg.predict_index_offset)
    mu_log_price = y_pred[target_index]
    sigma_log_price = y_std[target_index]
    return mu_log_price, sigma_log_price, current_log_price

def optimise_allocation(mu, sigma, current_log_price, cfg: Config):
    return minimize_scalar(expected_utility, bounds=(0, 1), args=(mu, sigma, current_log_price, cfg), method='bounded').x

def plot_expected_utility_curve(mu, sigma, current_log_price, optimal_weight, cfg: Config):
    weights = np.linspace(0, 1, 100)
    utilities = [-expected_utility(w, mu, sigma, current_log_price, cfg) for w in weights]
    plt.figure(figsize=(8, 4))
    plt.plot(weights, utilities, label='Expected Utility')
    plt.axvline(optimal_weight, color='r', linestyle='--', label='Optimal weight')
    plt.xlabel('BTC Allocation')
    plt.ylabel('Expected Utility')
    plt.title('Expected Utility vs BTC Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("utility_curve.png")

def plot_wealth_distribution(mu, sigma, current_log_price, optimal_weight, cfg: Config):
    pred_log_price_vals = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
    pdf_vals = norm.pdf(pred_log_price_vals, loc=mu, scale=sigma)
    
    # Calculate wealth using the same formula as in expected_utility
    wealth_vals = []
    for pred_log_price in pred_log_price_vals:
        cash_portion = cfg.initial_wealth * (1 - optimal_weight)
        btc_units = (cfg.initial_wealth * optimal_weight) / np.exp(current_log_price)
        btc_value = btc_units * np.exp(pred_log_price)
        portfolio_value = cash_portion + btc_value
        wealth_vals.append(portfolio_value)
    
    wealth_vals = np.array(wealth_vals)
    
    # CORRECT: Calculate expected wealth by integrating over log-price space
    # E[W] = ∫ W(log_price) * p(log_price) d(log_price)
    d_log_price = pred_log_price_vals[1] - pred_log_price_vals[0]  # Grid spacing
    expected_wealth = np.sum(wealth_vals * pdf_vals * d_log_price)
    
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_vals, pdf_vals, label='Wealth PDF', linewidth=2)
    plt.axvline(cfg.initial_wealth, color='r', linestyle='--', label='Initial Wealth', linewidth=2)
    plt.axvline(expected_wealth, color='orange', linestyle=':', label=f'Expected Wealth: ${expected_wealth:.0f}', linewidth=2)
    
    plt.xlabel('Simulated Future Wealth ($)')
    plt.ylabel('Probability Density')
    plt.title('Wealth Distribution of Optimal Portfolio')
    
    # Fixed scale: always 0 to 2x initial wealth
    plt.xlim(0, 2 * cfg.initial_wealth)
    
    # Fix the x-axis formatting
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='x')
    
    # Set clean tick spacing every $200
    tick_spacing = 200
    ax.set_xticks(np.arange(0, 2 * cfg.initial_wealth + tick_spacing, tick_spacing))
    
    # Format x-axis labels as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x)}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Expected Wealth: ${expected_wealth:.2f}\nOptimal BTC: {optimal_weight:.1%}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig("wealth_distribution.png", dpi=150)

def plot_utility_function(cfg: Config):
    utility = get_utility_func(cfg)
    if cfg.utility_function == 'log':
        x_vals = np.linspace(0.01, 3000, 500)
    elif cfg.utility_function == 'sqrt':
        x_vals = np.linspace(0, 3000, 500)
    elif cfg.utility_function in ['sigmoid']:
        x_vals = np.linspace(cfg.w0 - 1.0, cfg.w0 + 1.0, 500)
    elif cfg.utility_function in ['identity', 'linear']:
        x_vals = np.linspace(0, 3000, 500)
    elif cfg.utility_function in ['step', 'smooth_step']:
        x_vals = np.linspace(0, 3000, 500)
    elif cfg.utility_function in ['tanh','tanh_custom']:
        x_vals = np.linspace(0, 5000, 500)  # Wider range to see the tanh curve
    elif cfg.utility_function == 'crra':
        x_vals = np.linspace(0.01, 3000, 500)  # Start from 0.01 to avoid issues with power function
    else:
        raise ValueError(f"Unsupported utility function: {cfg.utility_function}")

    y_vals = [utility(w) for w in x_vals]
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'{cfg.utility_function.capitalize()} utility', color='blue')

    if cfg.utility_function in ['sigmoid', 'tanh']:
        plt.axvline(cfg.initial_wealth, color='grey', linestyle='--', label=f'Initial wealth ({cfg.initial_wealth})')
    elif cfg.utility_function in ['step', 'smooth_step']:
        plt.axvline(cfg.step_threshold, color='grey', linestyle='--', label=f'Threshold ({cfg.step_threshold})')
    elif cfg.utility_function == 'tanh_custom':
        plt.axvline(cfg.initial_wealth, color='grey', linestyle='--', label=f'Initial wealth ({cfg.initial_wealth})')
    elif cfg.utility_function == 'crra':
        plt.axvline(cfg.initial_wealth, color='grey', linestyle='--', label=f'Initial wealth ({cfg.initial_wealth})')

    plt.xlabel('Wealth')
    plt.ylabel('Utility')
    plt.title('Utility Function')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("utility_func.png")

def plot_utility_distribution(mu, sigma, current_log_price, optimal_weight, cfg: Config):
    pred_log_price_vals = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
    pdf_vals = norm.pdf(pred_log_price_vals, loc=mu, scale=sigma)
    
    utility = get_utility_func(cfg)
    
    # Calculate wealth and utility using the same formula as in expected_utility
    wealth_vals = []
    utility_vals = []
    for pred_log_price in pred_log_price_vals:
        cash_portion = cfg.initial_wealth * (1 - optimal_weight)
        btc_units = (cfg.initial_wealth * optimal_weight) / np.exp(current_log_price)
        btc_value = btc_units * np.exp(pred_log_price)
        portfolio_value = cash_portion + btc_value
        
        wealth_vals.append(portfolio_value)
        utility_vals.append(utility(portfolio_value))
    
    wealth_vals = np.array(wealth_vals)
    utility_vals = np.array(utility_vals)
    
    # Calculate expected utility
    d_log_price = pred_log_price_vals[1] - pred_log_price_vals[0]
    expected_utility_val = np.sum(utility_vals * pdf_vals * d_log_price)
    
    # Initial wealth utility for reference
    initial_utility = utility(cfg.initial_wealth)
    
    plt.figure(figsize=(10, 6))
    plt.plot(utility_vals, pdf_vals, label='Utility PDF', linewidth=2, color='purple')
    plt.axvline(initial_utility, color='r', linestyle='--', 
                label=f'Initial Utility: {initial_utility:.3f}', linewidth=2)
    plt.axvline(expected_utility_val, color='orange', linestyle=':', 
                label=f'Expected Utility: {expected_utility_val:.3f}', linewidth=2)
    
    plt.xlabel('Utility Value')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Portfolio Utility')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Expected Utility: {expected_utility_val:.4f}\nOptimal BTC: {optimal_weight:.1%}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.savefig("utility_distribution.png", dpi=150)

def plot_dp_expected_utility_curve(utility_curve_data, alloc0):
    allocs, utilities = utility_curve_data
    
    plt.figure(figsize=(8, 4))
    plt.plot(allocs, utilities, label='DP Expected Utility')
    plt.axvline(alloc0, color='r', linestyle='--', label=f'Optimal allocation: {alloc0:.2f}')
    plt.xlabel('BTC Allocation')
    plt.ylabel('Expected Utility')
    plt.title('DP Expected Utility vs BTC Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dp_utility_curve.png")
    print("Saved dp_utility_curve.png")

def plot_dp_utility_distribution(mu_seq, sigma_seq, current_log_price, optimal_weight, cfg: Config):
    """Plot the distribution of portfolio utility using DP optimal allocation"""
    
    # Use 1-step ahead predictions
    next_mu = mu_seq[1]
    next_sigma = sigma_seq[1]
    
    # Create log price scenarios
    pred_log_price_vals = np.linspace(next_mu - 5 * next_sigma, next_mu + 5 * next_sigma, 1000)
    pdf_vals = norm.pdf(pred_log_price_vals, loc=next_mu, scale=next_sigma)
    
    utility = get_utility_func(cfg)
    
    # Calculate wealth and utility using the DP optimal allocation
    wealth_vals = []
    utility_vals = []
    for pred_log_price in pred_log_price_vals:
        cash_portion = cfg.initial_wealth * (1 - optimal_weight)
        btc_units = (cfg.initial_wealth * optimal_weight) / np.exp(current_log_price)
        btc_value = btc_units * np.exp(pred_log_price)
        portfolio_value = cash_portion + btc_value
        
        wealth_vals.append(portfolio_value)
        utility_vals.append(utility(portfolio_value))
    
    wealth_vals = np.array(wealth_vals)
    utility_vals = np.array(utility_vals)
    
    # Calculate expected utility
    d_log_price = pred_log_price_vals[1] - pred_log_price_vals[0]
    expected_utility_val = np.sum(utility_vals * pdf_vals * d_log_price)
    
    # Initial wealth utility for reference
    initial_utility = utility(cfg.initial_wealth)
    
    plt.figure(figsize=(10, 6))
    plt.plot(utility_vals, pdf_vals, label='DP Utility PDF', linewidth=2, color='green')
    plt.axvline(initial_utility, color='r', linestyle='--', 
                label=f'Initial Utility: {initial_utility:.3f}', linewidth=2)
    plt.axvline(expected_utility_val, color='orange', linestyle=':', 
                label=f'Expected Utility: {expected_utility_val:.3f}', linewidth=2)
    
    plt.xlabel('Utility Value')
    plt.ylabel('Probability Density')
    plt.title('Distribution of DP Portfolio Utility (1 Step Ahead)')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Expected Utility: {expected_utility_val:.4f}\nDP Optimal BTC: {optimal_weight:.1%}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.savefig("dp_utility_distribution.png", dpi=150)
    print("Saved dp_utility_distribution.png")

def plot_dp_wealth_distribution(mu_seq, sigma_seq, current_log_price, optimal_weight, cfg: Config):
    """Plot the distribution of portfolio wealth using DP optimal allocation"""
    
    # Use 1-step ahead predictions
    next_mu = mu_seq[1]
    next_sigma = sigma_seq[1]
    
    # Create log price scenarios
    pred_log_price_vals = np.linspace(next_mu - 5 * next_sigma, next_mu + 5 * next_sigma, 1000)
    pdf_vals = norm.pdf(pred_log_price_vals, loc=next_mu, scale=next_sigma)
    
    # Calculate wealth using the DP optimal allocation
    wealth_vals = []
    for pred_log_price in pred_log_price_vals:
        cash_portion = cfg.initial_wealth * (1 - optimal_weight)
        btc_units = (cfg.initial_wealth * optimal_weight) / np.exp(current_log_price)
        btc_value = btc_units * np.exp(pred_log_price)
        portfolio_value = cash_portion + btc_value
        wealth_vals.append(portfolio_value)
    
    wealth_vals = np.array(wealth_vals)
    
    # Calculate expected wealth by integrating over log-price space
    d_log_price = pred_log_price_vals[1] - pred_log_price_vals[0]
    expected_wealth = np.sum(wealth_vals * pdf_vals * d_log_price)
    
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_vals, pdf_vals, label='DP Wealth PDF', linewidth=2, color='green')
    plt.axvline(cfg.initial_wealth, color='r', linestyle='--', label='Initial Wealth', linewidth=2)
    plt.axvline(expected_wealth, color='orange', linestyle=':', 
                label=f'Expected Wealth: ${expected_wealth:.0f}', linewidth=2)
    
    # Add threshold line if using step functions
    if cfg.utility_function in ['step', 'smooth_step']:
        plt.axvline(cfg.step_threshold, color='purple', linestyle='-.', 
                    label=f'Utility Threshold: ${cfg.step_threshold}', linewidth=2)
    
    plt.xlabel('Simulated Future Wealth ($)')
    plt.ylabel('Probability Density')
    plt.title('DP Wealth Distribution (1 Step Ahead)')
    
    # Fixed scale: always 0 to 2x initial wealth
    plt.xlim(0, 2 * cfg.initial_wealth)
    
    # Fix the x-axis formatting
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='x')
    
    # Set clean tick spacing every $200
    tick_spacing = 200
    ax.set_xticks(np.arange(0, 2 * cfg.initial_wealth + tick_spacing, tick_spacing))
    
    # Format x-axis labels as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x)}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.text(0.02, 0.98, f'Expected Wealth: ${expected_wealth:.2f}\nDP Optimal BTC: {optimal_weight:.1%}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.savefig("dp_wealth_distribution.png", dpi=150)
    print("Saved dp_wealth_distribution.png")

def main():
    cfg = Config()
    X_pred, y_pred, y_std, y_actual = load_data(cfg)

    mu, sigma, current_log_price = compute_gp_stats(X_pred, y_pred, y_std, y_actual, cfg)

    plot_utility_function(cfg)
    print("Saved utility_func.png")

    # # Calculate the actual returns
    # expected_log_return = mu - current_log_price
    # expected_percent_return = (np.exp(mu) / np.exp(current_log_price) - 1) * 100

    # print(f"\nOriginal GP predictions:")
    # print(f"Current log price: {current_log_price:.6f}")
    # print(f"Predicted future log price: {mu:.6f}")
    # print(f"Expected log return: {expected_log_return:.6f}")
    # print(f"Expected % return: {expected_percent_return:.1f}%")
    # print(f"Predicted standard deviation: {sigma:.6f}")
    # print(f"Sharpe-like ratio: {expected_log_return/sigma:.3f}")

    # optimal_weight = optimise_allocation(mu, sigma, current_log_price, cfg)
    # print(f"Optimal BTC allocation: {optimal_weight:.3f}")
    # print(f"Optimal cash allocation: {1 - optimal_weight:.3f}")

    # plot_expected_utility_curve(mu, sigma, current_log_price, optimal_weight, cfg)
    # print("Saved utility_curve.png")

    # plot_wealth_distribution(mu, sigma, current_log_price, optimal_weight, cfg)
    # print("Saved wealth_distribution.png")

    # plot_utility_distribution(mu, sigma, current_log_price, optimal_weight, cfg)
    # print("Saved utility_distribution.png")

    # --- Dynamic Programming Section ---
    print("\n--- Dynamic Programming ---")
    print(f"Configuration:")
    print(f"  Utility function: {cfg.utility_function}")
    print(f"  Initial wealth: ${cfg.initial_wealth}")
    print(f"  Time horizon: {cfg.dp_n_steps * cfg.dp_weeks_per_step // 4} months")
    print(f"  Grid size: {cfg.dp_price_grid_size}x{cfg.dp_wealth_grid_size}")
    
    # Build mu and sigma sequences based on configuration
    current_idx = len(y_actual)
    mu_seq = []
    sigma_seq = []
    for k in range(cfg.dp_n_steps + 1):
        idx = np.searchsorted(X_pred.ravel(), current_idx + cfg.dp_weeks_per_step * k)
        mu_seq.append(y_pred[idx])
        sigma_seq.append(y_std[idx])
    
    utility_func = get_utility_func(cfg)
    initial_wealth = cfg.initial_wealth
    current_log_price = y_actual[-1]

    # Dynamic Programming with configuration from Config
    policy, value_fn, alloc0, utility_curve_data = dynamic_programming_policy(
        mu_seq, sigma_seq, utility_func, initial_wealth, current_log_price,
        cfg.dp_n_steps, cfg.dp_price_grid_size, cfg.dp_wealth_grid_size, 
        verbose=cfg.dp_verbose,
        compute_utility_curve=cfg.dp_compute_utility_curve, 
        n_alloc_points=cfg.dp_n_alloc_points, 
        moving_avg=cfg.dp_moving_avg_window
    )
    print(f"Optimal initial allocation: {alloc0:.3f}")

    # Use the utility curve data from DP
    plot_dp_expected_utility_curve(utility_curve_data, alloc0)
    
    # Plot DP utility distribution
    plot_dp_utility_distribution(mu_seq, sigma_seq, current_log_price, alloc0, cfg)
    
    # Plot DP wealth distribution  
    plot_dp_wealth_distribution(mu_seq, sigma_seq, current_log_price, alloc0, cfg)

if __name__ == '__main__':
    main()
