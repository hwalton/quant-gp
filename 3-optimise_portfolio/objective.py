import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
from itertools import product
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad


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
        gamma = cfg.gamma
        return lambda w: (w**(1-gamma) - 1) / (1-gamma) if gamma != 1 else np.log(w)
    else:
        raise ValueError(f"Unsupported utility function: {cfg.utility_function}")

def load_gp_predictions(cfg: Config):
    X_pred = np.load(cfg.x_pred_pkl)
    y_pred = np.load(cfg.y_pred_pkl)
    y_std = np.load(cfg.y_std_pkl)

    df = pd.read_csv(cfg.log_csv, sep=';').sort_values(by='timestamp')
    current_log_price = np.log(df['close'].astype(float).values[-1])

    current_index = len(df)
    target_index = np.searchsorted(X_pred.ravel(), current_index)

    mu_seq = y_pred[target_index : target_index + cfg.horizon_weeks]
    sigma_seq = y_std[target_index : target_index + cfg.horizon_weeks]

    return mu_seq, sigma_seq, current_log_price


def objective_numerical_integral(p, mu_seq, sigma_seq, current_log_price, cfg):
    utility = get_utility_func(cfg)

    T = cfg.horizon_weeks // cfg.rebalance_every
    assert len(p) == T, f"Expected p of length {T}"

    # Calculate maximum grid points per dimension to stay under 1M paths
    max_total_paths = 1000000
    grid_points_per_dim = int(max_total_paths**(1/T))
    
    # Ensure we don't exceed the limit
    actual_paths = grid_points_per_dim ** T
    if actual_paths > max_total_paths:
        grid_points_per_dim -= 1
        actual_paths = grid_points_per_dim ** T
    
    print(f"Using {grid_points_per_dim} grid points per dimension for {T} dimensions")
    print(f"Total paths: {actual_paths:,}")

    # Build 1D grids for each future rebalance log-price x_t
    grid_limits = [
        np.linspace(mu_seq[t] - 4*sigma_seq[t], mu_seq[t] + 4*sigma_seq[t], grid_points_per_dim)
        for t in range(T)
    ]
    dx = np.array([g[1] - g[0] for g in grid_limits])

    # Create meshgrid for all path combinations
    grids = np.meshgrid(*grid_limits, indexing='ij')
    
    # Stack to get all paths: shape (n_total_paths, T)
    all_paths = np.stack([g.ravel() for g in grids], axis=1)
    n_paths = all_paths.shape[0]

    # Vectorized wealth calculation
    wealth = np.full(n_paths, cfg.initial_wealth)
    x_prev = np.full(n_paths, current_log_price)

    # Pre-compute p array for faster indexing
    p_array = np.array(p)
    
    for t in range(T):
        x_now = all_paths[:, t]
        # Vectorized exponential operations
        price_prev = np.exp(x_prev)
        price_now = np.exp(x_now)

        # Vectorized portfolio calculations
        p_t = p_array[t]
        cash = wealth * (1 - p_t)
        btc = (wealth * p_t) / price_prev
        wealth = cash + btc * price_now

        x_prev = x_now

    # Vectorize utility calculation where possible
    if cfg.utility_function in ['log', 'sqrt', 'identity', 'linear', 'crra']:
        # These can be fully vectorized
        if cfg.utility_function == 'log':
            utilities = np.log(np.maximum(wealth, 1e-10))  # Avoid log(0)
        elif cfg.utility_function == 'sqrt':
            utilities = np.sqrt(np.maximum(wealth, 0))
        elif cfg.utility_function in ['identity', 'linear']:
            utilities = wealth
        elif cfg.utility_function == 'crra':
            gamma = cfg.gamma
            if gamma == 1.0:
                utilities = np.log(np.maximum(wealth, 1e-10))
            else:
                utilities = (np.maximum(wealth, 1e-10)**(1-gamma) - 1) / (1-gamma)
    else:
        # For complex utility functions, use list comprehension (still faster than loop)
        utilities = np.array([utility(w) for w in wealth])

    # More efficient probability calculation
    # Pre-compute mu and sigma arrays
    mu_array = np.array([mu_seq[t] for t in range(T)])
    sigma_array = np.array([sigma_seq[t] for t in range(T)])
    
    # Vectorized log probability calculation
    log_probs = np.sum(
        norm.logpdf(all_paths, loc=mu_array, scale=sigma_array), 
        axis=1
    )
    prob_densities = np.exp(log_probs)

    # Pre-compute volume element
    volume_element = np.prod(dx)

    # Final integration
    total = np.sum(utilities * prob_densities) * volume_element

    return total

def run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, months=12):
    # Search space: p_t in [0, 1] for each month
    search_space = [Real(0.0, 1.0, name=f"p{i}") for i in range(months)]

    @use_named_args(search_space)
    def objective_wrapped(**kwargs):
        p = np.array([kwargs[f"p{i}"] for i in range(months)])
        util = objective_numerical_integral(p, mu_seq, sigma_seq, current_log_price, cfg)
        return -util  # Negative for minimisation

    result = gp_minimize(
        func=objective_wrapped,
        dimensions=search_space,
        n_calls=25,
        n_initial_points=10,
        acq_func="EI",  # Expected improvement
        random_state=42,
        verbose=True
    )

    optimal_p = np.array(result.x)
    max_utility = -result.fun

    return optimal_p, max_utility, result

def plot_figures(mu, sigma, current_log_price, optimal_weight, cfg: Config):
    """Generate all visualization plots and save as PNG files"""
    
    def plot_wealth_distribution():
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
        
        # Calculate expected wealth
        d_log_price = pred_log_price_vals[1] - pred_log_price_vals[0]
        expected_wealth = np.sum(wealth_vals * pdf_vals * d_log_price)
        
        plt.figure(figsize=(10, 6))
        plt.plot(wealth_vals, pdf_vals, label='Wealth PDF', linewidth=2)
        plt.axvline(cfg.initial_wealth, color='r', linestyle='--', label='Initial Wealth', linewidth=2)
        plt.axvline(expected_wealth, color='orange', linestyle=':', label=f'Expected Wealth: ${expected_wealth:.0f}', linewidth=2)
        
        plt.xlabel('Simulated Future Wealth ($)')
        plt.ylabel('Probability Density')
        plt.title('Wealth Distribution of Optimal Portfolio After 1 Step')
        
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
        plt.close()
    
    def plot_utility_distribution():
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
        plt.title('Distribution of Portfolio Utility After 1 Step')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Add statistics text
        plt.text(0.02, 0.98, f'Expected Utility: {expected_utility_val:.4f}\nOptimal BTC: {optimal_weight:.1%}', 
                 transform=plt.gca().transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.savefig("utility_distribution.png", dpi=150)
        plt.close()
    
    def plot_utility_function():
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
            x_vals = np.linspace(0, 5000, 500)
        elif cfg.utility_function == 'crra':
            x_vals = np.linspace(0.01, 3000, 500)
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
        plt.close()

    
    plot_wealth_distribution()
    print("Saved wealth_distribution.png")
    
    plot_utility_distribution()
    print("Saved utility_distribution.png")
    
    plot_utility_function()
    print("Saved utility_func.png")

def main():
    start_time = time.time()

    cfg = Config()
    mu_seq, sigma_seq, current_log_price = load_gp_predictions(cfg)
    
    # Debug: Check expected returns by period
    print(f"Current log price: {current_log_price:.4f}")
    print(f"GP Predictions by period:")
    for i in range(len(mu_seq)):
        expected_return = mu_seq[i] - current_log_price
        print(f"  Week {i}: μ={mu_seq[i]:.4f}, expected return={expected_return:.4f}")
    
    # Number of rebalancing points = horizon_weeks / rebalance_every
    T = cfg.horizon_weeks // cfg.rebalance_every

    # Evaluate objective at a naive initial guess (e.g. 50/50 BTC)
    p_init = np.full(T, 0.5)
    expected_util = objective_numerical_integral(p_init, mu_seq, sigma_seq, current_log_price, cfg)
    print(f"Initial Expected Utility: {expected_util:.4f}")
    print(f"Initial Allocation: {np.round(p_init, 3)}")

    # Run Bayesian optimisation
    print("\nRunning Bayesian Optimisation...")
    optimal_p, max_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T)

    print(f"\nOptimal allocation vector:")
    print(np.round(optimal_p, 3))
    print(f"Maximum expected utility: {max_util:.4f}")

    # Print timing information
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    # Generate all plots using first period data for visualization
    mu_first_period = mu_seq[0]  # First period prediction
    sigma_first_period = sigma_seq[0]  # First period uncertainty
    optimal_weight_first_period = optimal_p[0]  # First period allocation
    
    plot_figures(mu_first_period, sigma_first_period, current_log_price, optimal_weight_first_period, cfg)


if __name__ == '__main__':
    main()
