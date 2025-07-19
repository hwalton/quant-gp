import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import interpolate

from config import Config

def plot_preference_curve(cfg: Config):
    """Plot the preference curve function"""
    from get_utility_function import get_preference_curve
    
    preference_curve = get_preference_curve(cfg)
    
    # Always use wealth range from 0 to 5000 for preference curves
    x_vals = np.linspace(np.log(100), np.log(5000), 1000)
    
    # Vectorized preference curve calculation
    y_vals = preference_curve(x_vals)
    
    # Convert ln(wealth) back to actual wealth for plotting
    wealth_vals = np.exp(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_vals, y_vals, label=f'{cfg.preference_curve.replace("_", " ").title()} Preference', 
                 color='blue', linewidth=2)
    
    # Add reference lines (convert back to actual wealth)
    plt.axvline(cfg.initial_wealth, color='red', linestyle='--', 
                label=f'Initial Wealth: ${cfg.initial_wealth:.0f}', linewidth=2)
    
    # Set log10 scale for x-axis
    plt.xscale('log')
    
    plt.xlabel('Wealth (USD)', fontsize=18)
    plt.ylabel('Preference Value', fontsize=18)
    plt.title(f'Preference Curve: {cfg.preference_curve.replace("_", " ").title()}', fontsize=21)
    
    # Format x-axis with powers of 10
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines
    
    # Auto-position legend to avoid lines
    plt.legend(loc='best', fontsize=15)
    plt.tight_layout()

    plt.savefig("preference_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved preference_curve.png")

def plot_final_wealth_distribution(mu_seq, sigma_seq, current_log_price, optimal_p, cfg: Config):
    """Plot wealth distribution after all optimization steps"""
    
    # Handle both single value and array cases
    if np.isscalar(optimal_p):
        optimal_p = [optimal_p]
        mu_seq = [mu_seq] if np.isscalar(mu_seq) else mu_seq[:1]
        sigma_seq = [sigma_seq] if np.isscalar(sigma_seq) else sigma_seq[:1]
    
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
    
    # Vectorized wealth calculation using log wealth approach
    log_wealth = np.full(n_paths, np.log(cfg.initial_wealth))
    x_prev = np.full(n_paths, current_log_price)
    p_array = np.array(optimal_p)
    
    for t in range(T):
        x_now = all_paths[:, t]
        
        # Calculate log returns
        log_return = x_now - x_prev
        
        # Update log wealth using the formula
        portfolio_return = (1 - p_array[t]) + p_array[t] * np.exp(log_return)
        portfolio_return = np.maximum(portfolio_return, 1e-10)
        
        log_wealth += np.log(portfolio_return)
        x_prev = x_now
    
    # Convert back to actual wealth
    wealth = np.exp(log_wealth)
    
    # Calculate probabilities for each path
    mu_array = np.array([mu_seq[t] for t in range(T)])
    sigma_array = np.array([sigma_seq[t] for t in range(T)])
    log_probs = np.sum(norm.logpdf(all_paths, loc=mu_array, scale=sigma_array), axis=1)
    probabilities = np.exp(log_probs)
    
    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)
    
    # Calculate expected wealth
    expected_wealth = np.sum(wealth * probabilities)
    
    # Create histogram-based PDF using smooth curve through midpoints
    num_bins = min(50, len(np.unique(wealth)))  # Adaptive number of bins
    
    # Calculate weighted histogram
    hist_counts, bin_edges = np.histogram(wealth, bins=num_bins, weights=probabilities, density=True)
    
    # Calculate bin midpoints
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter out zero-count bins for smoother interpolation
    non_zero_mask = hist_counts > 0
    filtered_midpoints = bin_midpoints[non_zero_mask]
    filtered_counts = hist_counts[non_zero_mask]
    
    # Create smooth curve through the midpoints using interpolation
    if len(filtered_midpoints) > 3:  # Need at least 4 points for cubic interpolation
        # Use cubic spline interpolation for smooth curve
        wealth_range = np.linspace(wealth.min(), wealth.max(), 1000)
        
        # Extend the range slightly for better interpolation at edges
        interp_func = interpolate.interp1d(
            filtered_midpoints, filtered_counts, 
            kind='cubic', bounds_error=False, fill_value=0
        )
        pdf_values = interp_func(wealth_range)
        
        # Ensure no negative values (can happen with cubic interpolation)
        pdf_values = np.maximum(pdf_values, 0)
        
        # Normalize to ensure it's a proper PDF
        if np.sum(pdf_values) > 0:
            pdf_values = pdf_values / np.trapz(pdf_values, wealth_range)
        
    else:
        # Fallback to linear interpolation for few points
        wealth_range = np.linspace(wealth.min(), wealth.max(), 1000)
        if len(filtered_midpoints) > 1:
            interp_func = interpolate.interp1d(
                filtered_midpoints, filtered_counts, 
                kind='linear', bounds_error=False, fill_value=0
            )
            pdf_values = interp_func(wealth_range)
        else:
            # Single point - create a narrow spike
            pdf_values = np.zeros_like(wealth_range)
            closest_idx = np.argmin(np.abs(wealth_range - filtered_midpoints[0]))
            pdf_values[closest_idx] = filtered_counts[0]
    
    # Find the peak of the distribution (mode)
    peak_idx = np.argmax(pdf_values)
    peak_wealth = wealth_range[peak_idx]
    
    # Create smooth PDF plot
    plt.figure(figsize=(10, 6))
    
    # Scale PDF values by 10^3 for readability
    pdf_values_scaled = pdf_values * 1e3
    
    # Plot only the smooth curve
    plt.plot(wealth_range, pdf_values_scaled, label='Wealth Distribution', 
             linewidth=2, color='skyblue')
    plt.axvline(cfg.initial_wealth, color='r', linestyle='--', 
                label=f'Initial Wealth: ${cfg.initial_wealth:.0f}', linewidth=2)
    plt.axvline(expected_wealth, color='orange', linestyle='--', 
                label=f'Expected (Mean) Wealth: ${expected_wealth:.0f}', linewidth=2)
    plt.axvline(peak_wealth, color='green', linestyle='--', 
                label=f'Most Probable Wealth: ${peak_wealth:.0f}', linewidth=2)
    
    plt.xlabel('Wealth (USD)', fontsize=18)
    plt.ylabel('Probability Density (×1E-3)', fontsize=18)
    plt.title(f'Wealth Distribution At Investment Horizon: {cfg.horizon_weeks} Weeks', fontsize=21)

    # Format x-axis as currency
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.grid(True, alpha=0.3)
    
    # Auto-position legend to avoid lines
    plt.legend(loc='best', fontsize=15)
    
    # Find best position for text box (avoid high density areas)
    # Calculate which corner has lowest PDF values
    corners = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98), 
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02)
    }
    
    corner_scores = {}
    for corner_name, (x_frac, y_frac) in corners.items():
        # Convert fractional coordinates to data coordinates
        x_data = wealth_range.min() + x_frac * (wealth_range.max() - wealth_range.min())
        # Find closest point in wealth_range
        closest_idx = np.argmin(np.abs(wealth_range - x_data))
        corner_scores[corner_name] = pdf_values_scaled[closest_idx]
    
    # Choose corner with lowest PDF value
    best_corner = min(corner_scores, key=corner_scores.get)
    x_pos, y_pos = corners[best_corner]
    
    # Adjust text alignment based on position
    if x_pos > 0.5:
        ha = 'right'
    else:
        ha = 'left'
        
    if y_pos > 0.5:
        va = 'top'
    else:
        va = 'bottom'
    
    plt.text(
        x_pos, y_pos,
        f'Optimal Current Proportion BTC: {round(optimal_p[0], 2)}',
        transform=ax.transAxes, verticalalignment=va, horizontalalignment=ha,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=13.5
    )
    
    plt.tight_layout()
    plt.savefig("final_wealth_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved final_wealth_distribution.png")

def plot_allocation_vs_utility(mu_seq, sigma_seq, current_log_price, optimal_p, cfg: Config, grid_points_per_dim, objective_func):
    """Plot first-step BTC allocation vs expected utility, keeping future allocations fixed"""
    T = len(optimal_p)
    
    # Range of BTC allocations from 0% to 100% for FIRST step only
    allocation_range = np.linspace(0, 1, 21)
    expected_utilities = []
    
    print(f"Calculating allocation vs utility plot with {len(allocation_range)} points...")
    
    for i, first_allocation in enumerate(allocation_range):
        if i % 20 == 0:
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
    
    # Mark the optimal point (without showing utility value)
    optimal_utility = expected_utilities[np.argmin(np.abs(allocation_range - optimal_p[0]))]
    plt.plot(optimal_p[0] * 100, optimal_utility, 'ro', markersize=8, 
             label=f'Optimal Point')
    
    plt.xlabel('First Step BTC Allocation (%)', fontsize=18)
    plt.ylabel('Expected Utility', fontsize=18)
    plt.title('Expected Utility vs First Step BTC Allocation', fontsize=21)
    
    # Remove numbers from y-axis (utility axis)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='major', labelsize=14)
    
    plt.grid(True, alpha=0.3)
    
    # # Auto-position legend to avoid lines
    # plt.legend(loc='best', fontsize=15)
    
    # Find best position for text box (avoid the curve)
    # Check which side of the optimal point has more space
    optimal_idx = np.argmin(np.abs(allocation_range - optimal_p[0]))
    
    # Calculate average utility on left and right sides of optimal
    left_util = np.mean(expected_utilities[:optimal_idx]) if optimal_idx > 0 else expected_utilities[0]
    right_util = np.mean(expected_utilities[optimal_idx:]) if optimal_idx < len(expected_utilities)-1 else expected_utilities[-1]
    
    # Place text box on the side with lower average utility (more space)
    if left_util < right_util:
        x_pos, ha = 0.02, 'left'
    else:
        x_pos, ha = 0.98, 'right'
    
    plt.text(
        x_pos, 0.98,
        f'Optimal Current Proportion BTC: {round(optimal_p[0], 2)}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment=ha,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=13.5
    )

    plt.tight_layout()
    plt.savefig("allocation_vs_utility.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved allocation_vs_utility.png")