"""
Portfolio optimization using utility theory and GP predictions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from config import GPModelConfig
from data.loader import load_btc_data
from gp_fit.fit import load_gp_outputs

def get_utility_func(config: GPModelConfig):
    """Get utility function based on configuration."""
    if config.utility_function in ['identity', 'linear']:
        return lambda w: w
    elif config.utility_function == 'log':
        return lambda w: np.log(w) if w > 0 else -np.inf
    elif config.utility_function == 'sqrt':
        return lambda w: np.sqrt(w) if w >= 0 else 0
    elif config.utility_function == 'step':
        return lambda w: 1.0 if w > config.step_threshold else 0.0
    elif config.utility_function == 'smooth_step':
        def smooth_step(w):
            x = -config.step_steepness * (w - config.step_threshold)
            if x > 500:  # Prevent overflow
                return 0.0
            elif x < -500:
                return 1.0
            else:
                return 1 / (1 + np.exp(x))
        return smooth_step
    elif config.utility_function == 'sigmoid':
        return lambda w: 1 / (1 + np.exp(-config.sigmoid_k * (w - config.w0)))
    elif config.utility_function == 'tanh':
        return lambda w: np.tanh((w - 200) / 200)
    elif config.utility_function == 'tanh_custom':
        return lambda w: np.tanh((w - 2000) / 1000) + 1
    elif config.utility_function == 'crra':
        gamma = config.crra_gamma
        return lambda w: (w**(1-gamma) - 1) / (1-gamma) if gamma != 1 else np.log(w)
    else:
        raise ValueError(f"Unsupported utility function: {config.utility_function}")

def compute_gp_stats(X_pred: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, 
                    y_actual: np.ndarray, config: GPModelConfig) -> tuple[float, float, float]:
    """Extract GP prediction statistics for portfolio optimization."""
    current_log_price = y_actual[-1]
    target_index = np.searchsorted(X_pred.ravel(), len(y_actual) + config.predict_index_offset)
    mu_log_price = y_pred[target_index]
    sigma_log_price = y_std[target_index]
    return mu_log_price, sigma_log_price, current_log_price

def expected_utility(pp: float, mu_log_price: float, sigma_log_price: float, 
                    current_log_price: float, config: GPModelConfig) -> float:
    """Calculate expected utility for a given portfolio allocation."""
    utility = get_utility_func(config)
    
    def integrand(pred_log_price):
        # Portfolio calculation
        cash_portion = config.initial_wealth * (1 - pp)
        btc_units = (config.initial_wealth * pp) / np.exp(current_log_price)
        btc_value = btc_units * np.exp(pred_log_price)
        portfolio_value = cash_portion + btc_value
        
        return utility(portfolio_value) * norm.pdf(pred_log_price, loc=mu_log_price, scale=sigma_log_price)
    
    result, _ = quad(integrand, mu_log_price - 6*sigma_log_price, mu_log_price + 6*sigma_log_price)
    return -result  # Negative for minimization

def optimize_allocation(mu: float, sigma: float, current_log_price: float, 
                       config: GPModelConfig) -> float:
    """Find optimal portfolio allocation."""
    result = minimize_scalar(expected_utility, bounds=(0, 1), 
                           args=(mu, sigma, current_log_price, config), method='bounded')
    return result.x

def plot_expected_utility_curve(mu: float, sigma: float, current_log_price: float, 
                               optimal_weight: float, config: GPModelConfig) -> None:
    """Plot expected utility as a function of BTC allocation."""
    weights = np.linspace(0, 1, 100)
    utilities = [-expected_utility(w, mu, sigma, current_log_price, config) for w in weights]
    
    plt.figure(figsize=(8, 4))
    plt.plot(weights, utilities, label='Expected Utility')
    plt.axvline(optimal_weight, color='r', linestyle='--', label='Optimal weight')
    plt.xlabel('BTC Allocation')
    plt.ylabel('Expected Utility')
    plt.title('Expected Utility vs BTC Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_path = config.outputs_dir / config.utility_curve_plot
    plt.savefig(output_path)
    plt.close()
    print(f"Saved utility curve to {output_path}")

def plot_wealth_distribution(mu: float, sigma: float, current_log_price: float, 
                           optimal_weight: float, config: GPModelConfig) -> None:
    """Plot distribution of future wealth under optimal allocation."""
    pred_log_price_vals = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
    pdf_vals = norm.pdf(pred_log_price_vals, loc=mu, scale=sigma)
    
    # Calculate wealth using the same formula as in expected_utility
    wealth_vals = []
    for pred_log_price in pred_log_price_vals:
        cash_portion = config.initial_wealth * (1 - optimal_weight)
        btc_units = (config.initial_wealth * optimal_weight) / np.exp(current_log_price)
        btc_value = btc_units * np.exp(pred_log_price)
        portfolio_value = cash_portion + btc_value
        wealth_vals.append(portfolio_value)
    
    wealth_vals = np.array(wealth_vals)
    
    # Calculate expected wealth
    d_log_price = pred_log_price_vals[1] - pred_log_price_vals[0]
    expected_wealth = np.sum(wealth_vals * pdf_vals * d_log_price)
    
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_vals, pdf_vals, label='Wealth PDF', linewidth=2)
    plt.axvline(config.initial_wealth, color='r', linestyle='--', label='Initial Wealth', linewidth=2)
    plt.axvline(expected_wealth, color='orange', linestyle=':', 
               label=f'Expected Wealth: ${expected_wealth:.0f}', linewidth=2)
    
    plt.xlabel('Simulated Future Wealth ($)')
    plt.ylabel('Probability Density')
    plt.title('Wealth Distribution of Optimal Portfolio')
    
    # Fixed scale: always 0 to 2x initial wealth
    plt.xlim(0, 2 * config.initial_wealth)
    
    # Fix the x-axis formatting
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='x')
    
    # Set clean tick spacing every $200
    tick_spacing = 200
    ax.set_xticks(np.arange(0, 2 * config.initial_wealth + tick_spacing, tick_spacing))
    
    # Format x-axis labels as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x)}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Expected Wealth: ${expected_wealth:.2f}\nOptimal BTC: {optimal_weight:.1%}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    output_path = config.outputs_dir / config.wealth_dist_plot
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved wealth distribution to {output_path}")

def plot_utility_function(config: GPModelConfig) -> None:
    """Plot the utility function."""
    utility = get_utility_func(config)
    
    if config.utility_function == 'log':
        x_vals = np.linspace(0.01, 3000, 500)
    elif config.utility_function == 'sqrt':
        x_vals = np.linspace(0, 3000, 500)
    elif config.utility_function in ['sigmoid', 'tanh']:
        x_vals = np.linspace(0, 3000, 500)
    elif config.utility_function in ['identity', 'linear']:
        x_vals = np.linspace(0, 3000, 500)
    elif config.utility_function in ['step', 'smooth_step']:
        x_vals = np.linspace(0, 3000, 500)
    elif config.utility_function == 'tanh_custom':
        x_vals = np.linspace(0, 5000, 500)
    elif config.utility_function == 'crra':
        x_vals = np.linspace(0.01, 3000, 500)
    else:
        x_vals = np.linspace(0.01, 3000, 500)

    y_vals = [utility(w) for w in x_vals]
    
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'{config.utility_function.capitalize()} utility', color='blue')

    if config.utility_function in ['sigmoid', 'tanh']:
        plt.axvline(config.initial_wealth, color='grey', linestyle='--', 
                   label=f'Initial wealth ({config.initial_wealth})')
    elif config.utility_function in ['step', 'smooth_step']:
        plt.axvline(config.step_threshold, color='grey', linestyle='--', 
                   label=f'Threshold ({config.step_threshold})')
    elif config.utility_function == 'tanh_custom':
        plt.axvline(config.initial_wealth, color='grey', linestyle='--', 
                   label=f'Initial wealth ({config.initial_wealth})')
    elif config.utility_function == 'crra':
        plt.axvline(config.initial_wealth, color='grey', linestyle='--', 
                   label=f'Initial wealth ({config.initial_wealth})')

    plt.xlabel('Wealth')
    plt.ylabel('Utility')
    plt.title('Utility Function')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    output_path = config.outputs_dir / config.utility_func_plot
    plt.savefig(output_path)
    plt.close()
    print(f"Saved utility function to {output_path}")

def run_portfolio_optimization(config: GPModelConfig) -> dict:
    """
    Run the complete portfolio optimization pipeline.
    
    Returns:
        Dictionary with optimization results
    """
    print("Loading GP predictions and data...")
    X_pred, y_pred, y_std = load_gp_outputs(config)
    _, y_actual = load_btc_data(config)
    
    print("Computing GP statistics...")
    mu, sigma, current_log_price = compute_gp_stats(X_pred, y_pred, y_std, y_actual, config)
    
    # Calculate returns for display
    expected_log_return = mu - current_log_price
    expected_percent_return = (np.exp(mu) / np.exp(current_log_price) - 1) * 100
    
    print(f"\nGP Predictions:")
    print(f"Current log price: {current_log_price:.6f}")
    print(f"Predicted future log price: {mu:.6f}")
    print(f"Expected log return: {expected_log_return:.6f}")
    print(f"Expected % return: {expected_percent_return:.1f}%")
    print(f"Predicted standard deviation: {sigma:.6f}")
    print(f"Sharpe-like ratio: {expected_log_return/sigma:.3f}")
    
    print("\nOptimizing portfolio allocation...")
    optimal_weight = optimize_allocation(mu, sigma, current_log_price, config)
    
    print(f"\nOptimal Allocation:")
    print(f"BTC allocation: {optimal_weight:.1%}")
    print(f"Cash allocation: {1 - optimal_weight:.1%}")
    
    print("\nGenerating visualizations...")
    plot_expected_utility_curve(mu, sigma, current_log_price, optimal_weight, config)
    plot_wealth_distribution(mu, sigma, current_log_price, optimal_weight, config)
    plot_utility_function(config)
    
    return {
        'optimal_btc_weight': optimal_weight,
        'optimal_cash_weight': 1 - optimal_weight,
        'expected_log_return': expected_log_return,
        'expected_percent_return': expected_percent_return,
        'volatility': sigma,
        'sharpe_ratio': expected_log_return / sigma,
        'utility_function': config.utility_function
    }

if __name__ == "__main__":
    config = GPModelConfig()
    results = run_portfolio_optimization(config)
