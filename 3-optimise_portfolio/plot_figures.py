import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy import interpolate

from config import Config
from get_utilty_function import get_utility_func

def plot_wealth_distribution(mu, sigma, current_log_price, optimal_weight, cfg: Config):
    """Plot wealth distribution after 1 step"""
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

def plot_utility_distribution(mu, sigma, current_log_price, optimal_weight, cfg: Config):
    """Plot utility distribution after 1 step"""
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

# def plot_utility_function(cfg: Config):
#     """Plot the utility function"""
#     utility = get_utility_func(cfg)
#     if cfg.utility_function == 'log':
#         x_vals = np.linspace(0.01, 3000, 500)
#     elif cfg.utility_function == 'sqrt':
#         x_vals = np.linspace(0, 3000, 500)
#     elif cfg.utility_function in ['sigmoid']:
#         x_vals = np.linspace(cfg.w0 - 1.0, cfg.w0 + 1.0, 500)
#     elif cfg.utility_function in ['identity', 'linear']:
#         x_vals = np.linspace(0, 3000, 500)
#     elif cfg.utility_function in ['step', 'smooth_step']:
#         x_vals = np.linspace(0, 3000, 500)
#     elif cfg.utility_function in ['tanh','tanh_custom']:
#         x_vals = np.linspace(0, 5000, 500)
#     elif cfg.utility_function == 'crra':
#         x_vals = np.linspace(0.01, 3000, 500)
#     else:
#         raise ValueError(f"Unsupported utility function: {cfg.utility_function}")

#     y_vals = [utility(w) for w in x_vals]
#     plt.figure(figsize=(8, 4))
#     plt.plot(x_vals, y_vals, label=f'{cfg.utility_function.capitalize()} utility', color='blue')

#     if cfg.utility_function in ['sigmoid', 'tanh']:
#         plt.axvline(cfg.initial_wealth, color='grey', linestyle='--', label=f'Initial wealth ({cfg.initial_wealth})')
#     elif cfg.utility_function in ['step', 'smooth_step']:
#         plt.axvline(cfg.step_threshold, color='grey', linestyle='--', label=f'Threshold ({cfg.step_threshold})')
#     elif cfg.utility_function == 'tanh_custom':
#         plt.axvline(cfg.initial_wealth, color='grey', linestyle='--', label=f'Initial wealth ({cfg.initial_wealth})')
#     elif cfg.utility_function == 'crra':
#         plt.axvline(cfg.initial_wealth, color='grey', linestyle='--', label=f'Initial wealth ({cfg.initial_wealth})')

#     plt.xlabel('Wealth')
#     plt.ylabel('Utility')
#     plt.title('Utility Function')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("utility_func.png")
#     plt.close()

def plot_preference_curve(cfg: Config):
    """Plot the preference curve function"""
    from get_utilty_function import get_preference_curve
    
    preference_curve = get_preference_curve(cfg)
    
    # Always use wealth range from 0 to 5000 for preference curves
    x_vals = np.linspace(100, 5100, 1000)
    y_vals = [preference_curve(w) for w in x_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f'{cfg.preference_curve.replace("_", " ").title()} Preference', 
             color='blue', linewidth=2)
    
    # Add reference lines
    plt.axvline(cfg.initial_wealth, color='red', linestyle='--', 
                label=f'Initial Wealth: ${cfg.initial_wealth:.0f}', linewidth=2)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Neutral (0.5)')
    
    # Highlight interesting regions
    if 'step' in cfg.preference_curve:
        plt.axvline(1000, color='orange', linestyle=':', 
                    label='Step Threshold: $1000', alpha=0.7)
    elif cfg.preference_curve == 'target_seeking':
        plt.axvline(2500, color='green', linestyle=':', 
                    label='Target: $2500', alpha=0.7)
    elif cfg.preference_curve == 'fast_climb_drop':
        plt.axvline(2000, color='purple', linestyle=':', 
                    label='Peak: $2000', alpha=0.7)
    
    plt.xlabel('Wealth ($)')
    plt.ylabel('Preference Value')
    plt.title(f'Preference Curve: {cfg.preference_curve.replace("_", " ").title()}')
    plt.ylim(-1, 1)
    plt.xlim(0, 5100)
    
    # Format x-axis as currency
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x)}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig("preference_curve.png", dpi=150)
    plt.close()

def plot_final_wealth_distribution(mu_seq, sigma_seq, current_log_price, optimal_p, cfg: Config):
    """Plot wealth distribution after all optimization steps"""
    T = len(optimal_p)
    
    # Use same grid calculation as in objective_numerical_integral
    max_total_paths = 10000  # Smaller for plotting
    grid_points_per_dim = max(3, int(max_total_paths**(1/T)))
    
    # Ensure we don't exceed the limit
    actual_paths = grid_points_per_dim ** T
    if actual_paths > max_total_paths:
        grid_points_per_dim -= 1
        actual_paths = grid_points_per_dim ** T
    
    # Build 1D grids for each future rebalance log-price x_t
    grid_limits = [
        np.linspace(mu_seq[t] - 4*sigma_seq[t], mu_seq[t] + 4*sigma_seq[t], grid_points_per_dim)
        for t in range(T)
    ]
    
    # Create meshgrid for all path combinations
    grids = np.meshgrid(*grid_limits, indexing='ij')
    all_paths = np.stack([g.ravel() for g in grids], axis=1)
    n_paths = all_paths.shape[0]
    
    # Vectorized wealth calculation (same as objective_numerical_integral)
    wealth = np.full(n_paths, cfg.initial_wealth)
    x_prev = np.full(n_paths, current_log_price)
    p_array = np.array(optimal_p)
    
    for t in range(T):
        x_now = all_paths[:, t]
        price_prev = np.exp(x_prev)
        price_now = np.exp(x_now)
        
        p_t = p_array[t]
        cash = wealth * (1 - p_t)
        btc = (wealth * p_t) / price_prev
        wealth = cash + btc * price_now
        
        x_prev = x_now
    
    # Calculate probabilities for each path
    mu_array = np.array([mu_seq[t] for t in range(T)])
    sigma_array = np.array([sigma_seq[t] for t in range(T)])
    log_probs = np.sum(norm.logpdf(all_paths, loc=mu_array, scale=sigma_array), axis=1)
    probabilities = np.exp(log_probs)
    
    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)
    
    # Calculate expected wealth
    expected_wealth = np.sum(wealth * probabilities)
    
    # Create weighted samples for KDE
    # Repeat each wealth value according to its probability (scaled up)
    sample_weights = (probabilities * 10000).astype(int)  # Scale probabilities to integer weights
    wealth_samples = np.repeat(wealth, sample_weights)
    
    # Use KDE to create smooth PDF
    if len(wealth_samples) > 0:
        kde = gaussian_kde(wealth_samples)
        wealth_range = np.linspace(wealth.min(), wealth.max(), 1000)
        pdf_values = kde(wealth_range)
    else:
        # Fallback if no samples
        wealth_range = np.linspace(wealth.min(), wealth.max(), 1000)
        pdf_values = np.zeros_like(wealth_range)
    
    # Create smooth PDF plot
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_range, pdf_values, label='Final Wealth PDF', linewidth=2, color='skyblue')
    plt.axvline(cfg.initial_wealth, color='r', linestyle='--', 
                label=f'Initial Wealth: ${cfg.initial_wealth:.0f}', linewidth=2)
    plt.axvline(expected_wealth, color='orange', linestyle=':', 
                label=f'Expected Wealth: ${expected_wealth:.0f}', linewidth=2)
    
    plt.xlabel('Final Wealth ($)')
    plt.ylabel('Probability Density')
    plt.title('Final Wealth Distribution After All Steps')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Expected Final Wealth: ${expected_wealth:.2f}\nOptimal Strategy: {np.round(optimal_p, 2)}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig("final_wealth_distribution.png", dpi=150)
    plt.close()

def plot_final_utility_distribution(mu_seq, sigma_seq, current_log_price, optimal_p, cfg: Config):
    """Plot utility distribution after all optimization steps"""
    utility = get_utility_func(cfg)
    T = len(optimal_p)
    
    # Use same grid calculation as in objective_numerical_integral
    max_total_paths = 10000  # Smaller for plotting
    grid_points_per_dim = max(3, int(max_total_paths**(1/T)))
    
    # Ensure we don't exceed the limit
    actual_paths = grid_points_per_dim ** T
    if actual_paths > max_total_paths:
        grid_points_per_dim -= 1
        actual_paths = grid_points_per_dim ** T
    
    # Build 1D grids for each future rebalance log-price x_t
    grid_limits = [
        np.linspace(mu_seq[t] - 4*sigma_seq[t], mu_seq[t] + 4*sigma_seq[t], grid_points_per_dim)
        for t in range(T)
    ]
    
    # Create meshgrid for all path combinations
    grids = np.meshgrid(*grid_limits, indexing='ij')
    all_paths = np.stack([g.ravel() for g in grids], axis=1)
    n_paths = all_paths.shape[0]
    
    # Vectorized wealth calculation (same as objective_numerical_integral)
    wealth = np.full(n_paths, cfg.initial_wealth)
    x_prev = np.full(n_paths, current_log_price)
    p_array = np.array(optimal_p)
    
    for t in range(T):
        x_now = all_paths[:, t]
        price_prev = np.exp(x_prev)
        price_now = np.exp(x_now)
        
        p_t = p_array[t]
        cash = wealth * (1 - p_t)
        btc = (wealth * p_t) / price_prev
        wealth = cash + btc * price_now
        
        x_prev = x_now
    
    # Calculate utilities using the preference curve approach
    utilities = np.array([utility(w) for w in wealth])
    
    # Calculate probabilities for each path
    mu_array = np.array([mu_seq[t] for t in range(T)])
    sigma_array = np.array([sigma_seq[t] for t in range(T)])
    log_probs = np.sum(norm.logpdf(all_paths, loc=mu_array, scale=sigma_array), axis=1)
    probabilities = np.exp(log_probs)
    
    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)
    
    # Calculate expected utility
    expected_utility_val = np.sum(utilities * probabilities)
    initial_utility = utility(cfg.initial_wealth)
    
    # Create weighted samples for KDE
    # Repeat each utility value according to its probability (scaled up)
    sample_weights = (probabilities * 10000).astype(int)  # Scale probabilities to integer weights
    utility_samples = np.repeat(utilities, sample_weights)
    
    # Use KDE to create smooth PDF
    if len(utility_samples) > 0:
        kde = gaussian_kde(utility_samples)
        utility_range = np.linspace(utilities.min(), utilities.max(), 1000)
        pdf_values = kde(utility_range)
    else:
        # Fallback if no samples
        utility_range = np.linspace(utilities.min(), utilities.max(), 1000)
        pdf_values = np.zeros_like(utility_range)
    
    # Create smooth PDF plot
    plt.figure(figsize=(10, 6))
    plt.plot(utility_range, pdf_values, label='Final Utility PDF', linewidth=2, color='purple')
    plt.axvline(initial_utility, color='r', linestyle='--', 
                label=f'Initial Utility: {initial_utility:.3f}', linewidth=2)
    plt.axvline(expected_utility_val, color='orange', linestyle=':', 
                label=f'Expected Utility: {expected_utility_val:.3f}', linewidth=2)
    
    plt.xlabel('Final Utility Value')
    plt.ylabel('Probability Density')
    plt.title(f'Final Utility Distribution After All Steps ({cfg.preference_curve.replace("_", " ").title()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Expected Final Utility: {expected_utility_val:.4f}\nOptimal Strategy: {np.round(optimal_p, 2)}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.savefig("final_utility_distribution.png", dpi=150)
    plt.close()

def plot_allocation_vs_utility(mu_seq, sigma_seq, current_log_price, optimal_p, cfg: Config, grid_points_per_dim, objective_func):
    """Plot first-step BTC allocation vs expected utility, keeping future allocations fixed"""
    T = len(optimal_p)
    
    # Range of BTC allocations from 0% to 100% for FIRST step only
    allocation_range = np.linspace(0, 1, 21)  # 101 points from 0% to 100%
    expected_utilities = []
    
    print(f"Calculating allocation vs utility plot with {len(allocation_range)} points...")
    
    for i, first_allocation in enumerate(allocation_range):
        if i % 16 == 0:
            print(f"  Processing allocation {i+1}/{len(allocation_range)}: {first_allocation:.1%}")
        
        # Create allocation vector: vary first, keep rest optimal
        p_test = np.array(optimal_p)
        p_test[0] = first_allocation
        
        # Just call the objective function with the new allocation
        expected_utility_val = objective_func(p_test, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        expected_utilities.append(expected_utility_val)
    
    expected_utilities = np.array(expected_utilities)
    
    plt.figure(figsize=(10, 6))
    plt.plot(allocation_range * 100, expected_utilities, linewidth=2, color='green')
    plt.axvline(optimal_p[0] * 100, color='red', linestyle='--', 
                label=f'Optimal First Allocation: {optimal_p[0]:.1%}', linewidth=2)
    
    # Mark the optimal point
    optimal_utility = expected_utilities[np.argmin(np.abs(allocation_range - optimal_p[0]))]
    plt.plot(optimal_p[0] * 100, optimal_utility, 'ro', markersize=8, 
             label=f'Optimal Utility: {optimal_utility:.4f}')
    
    plt.xlabel('First Step BTC Allocation (%)')
    plt.ylabel('Expected Utility')
    plt.title('Final Expected Utility vs First Step BTC Allocation (Multi-Step)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Maximum Final Expected Utility: {np.max(expected_utilities):.4f}\nOptimal First Allocation: {optimal_p[0]:.1%}\nOptimal Strategy: {np.round(optimal_p, 2)}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.savefig("allocation_vs_utility.png", dpi=150)
    plt.close()
    
    print("Completed allocation vs utility plot")

def plot_figures(mu, sigma, current_log_price, optimal_weight, cfg: Config):
    """Generate all single-step visualization plots and save as PNG files"""
    
    plot_wealth_distribution(mu, sigma, current_log_price, optimal_weight, cfg)
    print("Saved wealth_distribution.png")
    
    plot_utility_distribution(mu, sigma, current_log_price, optimal_weight, cfg)
    print("Saved utility_distribution.png")
    
    plot_preference_curve(cfg)
    print("Saved preference_curve.png")
    
    # Remove this line - we call it separately in main() with multi-step parameters
    # plot_allocation_vs_utility(mu, sigma, current_log_price, optimal_weight, cfg)
    # print("Saved allocation_vs_utility.png")

def plot_final_distributions(mu_seq, sigma_seq, current_log_price, optimal_p, cfg: Config):
    """Generate final distribution plots after all steps"""
    
    plot_final_wealth_distribution(mu_seq, sigma_seq, current_log_price, optimal_p, cfg)
    print("Saved final_wealth_distribution.png")
    
    plot_final_utility_distribution(mu_seq, sigma_seq, current_log_price, optimal_p, cfg)
    print("Saved final_utility_distribution.png")