import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from config import Config
from get_utilty_function import get_utility_func

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