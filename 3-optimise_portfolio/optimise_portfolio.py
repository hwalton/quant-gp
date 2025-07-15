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
    utility_function: str = ['step', 'sigmoid', 'tanh', 'identity', 'linear', 'log', 'sqrt'][3]
    sigmoid_k: float = 25.0
    w0: float = 0.98
    predict_index_offset: int = 10
    y_limit: tuple = (4, 18)
    step_threshold: float = 999  # Add threshold parameter

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
    
    plt.figure(figsize=(8, 4))
    plt.plot(wealth_vals, pdf_vals, label='Wealth PDF')
    plt.axvline(cfg.initial_wealth, color='r', linestyle='--', label='Initial Wealth')
    plt.xlabel('Simulated Future Wealth')
    plt.ylabel('Probability Density')
    plt.title('Wealth Distribution')
    plt.xscale('log')  # Add log scale for x-axis
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Include a reasonable range for log scale
    min_x = max(1, min(wealth_vals))  # Avoid log(0)
    max_x = max(max(wealth_vals), cfg.initial_wealth)
    plt.xlim(min_x, max_x)
    plt.savefig("wealth_distribution.png")

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
    else:
        raise ValueError(f"Unsupported utility function: {cfg.utility_function}")

    y_vals = [utility(w) for w in x_vals]
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'{cfg.utility_function.capitalize()} utility', color='blue')

    if cfg.utility_function in ['sigmoid', 'tanh']:
        plt.axvline(cfg.w0, color='grey', linestyle='--', label='Inflection point (w0)')
    elif cfg.utility_function == 'step':
        plt.axvline(cfg.step_threshold, color='grey', linestyle='--', label=f'Threshold ({cfg.step_threshold})')

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

    print(f"\nExpected BTC return over {cfg.predict_index_offset} closes: {mu:.6f}")
    print(f"Predicted standard deviation: {sigma:.6f}")

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
