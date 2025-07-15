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

@dataclass(frozen=True)
class Config:
    x_pred_pkl: str = '../2-gp_fit/X_pred.npy'
    y_pred_pkl: str = '../2-gp_fit/y_pred.npy'
    ystd_pkl: str = '../2-gp_fit/y_std.npy'
    log_csv: str = '../0-data/btc_weekly_prices.csv'
    initial_wealth: float = 1000
    utility_function: str = ['step', 'sigmoid', 'tanh', 'tanh_custom', 'identity', 'linear', 'log', 'sqrt', 'crra'][3]
    sigmoid_k: float = 25.0
    w0: float = 0.98
    predict_index_offset: int = 10
    y_limit: tuple = (4, 18)
    step_threshold: float = 999

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
    elif cfg.utility_function == 'sigmoid':
        return lambda w: 1 / (1 + np.exp(-cfg.sigmoid_k * (w - cfg.w0)))
    elif cfg.utility_function == 'tanh':
        return lambda w: np.tanh(cfg.sigmoid_k * (w - cfg.w0))
    elif cfg.utility_function == 'tanh_custom':
        return lambda w: np.tanh((w - 700) / 200) + 1
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
    
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_vals, pdf_vals, label='Wealth PDF', linewidth=2)
    plt.axvline(cfg.initial_wealth, color='r', linestyle='--', label='Initial Wealth', linewidth=2)
    plt.xlabel('Simulated Future Wealth ($)')
    plt.ylabel('Probability Density')
    plt.title('Wealth Distribution')
    
    # Fixed scale: always 0 to 2x initial wealth
    plt.xlim(0, 2 * cfg.initial_wealth)
    
    # Fix the x-axis formatting
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='x')  # No scientific notation
    
    # Set clean tick spacing every $200
    tick_spacing = 200
    ax.set_xticks(np.arange(0, 3 * cfg.initial_wealth + tick_spacing, tick_spacing))
    
    # Format x-axis labels as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x)}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add some statistics as text
    mean_wealth = np.average(wealth_vals, weights=pdf_vals)
    plt.text(0.02, 0.98, f'Mean wealth: ${mean_wealth:.2f}\nOptimal BTC: {optimal_weight:.1%}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig("wealth_distribution.png", dpi=150)

def plot_utility_function(cfg: Config):
    utility = get_utility_func(cfg)
    if cfg.utility_function == 'log':
        x_vals = np.linspace(0.01, 3000, 500)
    elif cfg.utility_function == 'sqrt':
        x_vals = np.linspace(0, 3000, 500)
    elif cfg.utility_function in ['sigmoid', 'tanh']:
        x_vals = np.linspace(cfg.w0 - 1.0, cfg.w0 + 1.0, 500)
    elif cfg.utility_function in ['identity', 'linear']:
        x_vals = np.linspace(0, 3000, 500)
    elif cfg.utility_function == 'step':
        x_vals = np.linspace(0, 3000, 500)
    elif cfg.utility_function == 'tanh_custom':
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
    elif cfg.utility_function == 'step':
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

def main():
    cfg = Config()
    X_pred, y_pred, y_std, y_actual = load_data(cfg)

    mu, sigma, current_log_price = compute_gp_stats(X_pred, y_pred, y_std, y_actual, cfg)

    print(f"\nOriginal GP predictions:")
    print(f"Expected BTC return over {cfg.predict_index_offset} closes: {mu:.6f}")
    print(f"Predicted standard deviation: {sigma:.6f}")
    print(f"Sharpe-like ratio: {mu/sigma:.3f}")

    optimal_weight = optimise_allocation(mu, sigma, current_log_price, cfg)
    print(f"Optimal BTC allocation: {optimal_weight:.3f}")
    print(f"Optimal cash allocation: {1 - optimal_weight:.3f}")

    plot_expected_utility_curve(mu, sigma, current_log_price, optimal_weight, cfg)
    print("Saved utility_curve.png")

    plot_wealth_distribution(mu, sigma, current_log_price, optimal_weight, cfg)
    print("Saved wealth_distribution.png")

    plot_utility_function(cfg)
    print("Saved utility_func.png")

if __name__ == '__main__':
    main()
