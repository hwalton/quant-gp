import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import joblib
import torch

import os
import sys

# Add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.utils import load_data
from b_log_fit.log_fit import fit_log_trend, log_func, Config as LogFitConfig
from c_gp_fit.online_gp_fit import OnlineVariationalGP, Config as OnlineGPConfig, plot_full_dataset_results
from d_optimise_portfolio.optimise_portfolio import load_gp_predictions, objective_func, run_bayesian_optimisation, coordinate_descent_refinement, calculate_grid_parameters
from d_optimise_portfolio.config import Config as OptimiseConfig

@dataclass(frozen=True)
class Config:
    data_path: str = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')
    starting_wealth: float = 1000
    start_datetime: str = "2023-06-01"  # Start date for backtesting
    end_datetime: str = "2024-06-01"    # End date for backtesting
    
    # Portfolio optimization config
    preference_curve: str = 'identity'
    horizon_weeks: int = 1
    rebalance_every: int = 1
    optimisation_method: str = 'bayesian_with_refinement'
    n_calls_optimiser: int = 15
    gamma: float = 5
    step_threshold: float = 1100
    step_steepness: float = 100.0

    # GP update mode
    gp_update_mode: str = ['online', 'full'][1]

def create_log_trend_function(params):
    """Create log trend function from fitted parameters"""
    def trend_func(x):
        return log_func(x, params[0], params[1], params[2])
    return trend_func

def optimize_portfolio_for_current_state(current_log_price, mu_seq, sigma_seq, cfg):
    """Optimize portfolio allocation for current state"""
    
    # Create a new config for portfolio optimization with the right values
    opt_config = OptimiseConfig(
        initial_wealth=cfg.starting_wealth,
        preference_curve=cfg.preference_curve,
        horizon_weeks=cfg.horizon_weeks,
        rebalance_every=cfg.rebalance_every,
        optimisation_method=cfg.optimisation_method,
        n_calls_optimiser=cfg.n_calls_optimiser,
        gamma=cfg.gamma,
        step_threshold=cfg.step_threshold,
        step_steepness=cfg.step_steepness
    )
    
    # Calculate grid parameters
    T = cfg.horizon_weeks // cfg.rebalance_every
    grid_points_per_dim, _ = calculate_grid_parameters(T)
    
    if cfg.optimisation_method == "bayesian":
        optimal_p, max_util, result = run_bayesian_optimisation(
            opt_config, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
    elif cfg.optimisation_method == "bayesian_with_refinement":
        bayesian_p, bayesian_util, result = run_bayesian_optimisation(
            opt_config, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
        optimal_p, max_util = coordinate_descent_refinement(
            bayesian_p, mu_seq, sigma_seq, current_log_price, opt_config, grid_points_per_dim)
    
    return optimal_p[0]  # Return only the first step allocation

def plot_backtesting_gp_state(gp_model, X_current, y_current, current_date, cfg):
    """Plot current GP state during backtesting"""
    import matplotlib.pyplot as plt
    
    # Make predictions for plotting
    X_plot = np.linspace(X_current.min(), X_current.max() + 50, 500)
    y_pred_plot, y_std_plot = gp_model.predict(X_plot)
    
    # Load log trend function  
    log_trend = create_log_trend_function(fit_log_trend(X_current, y_current))
    
    # Add back log trend
    y_pred_with_trend = y_pred_plot + log_trend(X_plot)
    y_trend_current = log_trend(X_current)
    y_trend_plot = log_trend(X_plot)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot current data points
    plt.plot(X_current, y_current, 'kx', label='Historical BTC Log Prices', markersize=4, alpha=0.8)
    
    # Plot log trend
    plt.plot(X_plot, y_trend_plot, 'r-', label='Log Trend', linewidth=2)
    
    # Plot GP prediction  
    plt.plot(X_plot, y_pred_with_trend, 'b-', label='GP Prediction', linewidth=2)
    plt.fill_between(X_plot, 
                     y_pred_with_trend - y_std_plot, 
                     y_pred_with_trend + y_std_plot,
                     alpha=0.2, label='GP 1σ', color='skyblue')
    
    plt.xlabel('Time Index')
    plt.ylabel('Log BTC Price') 
    plt.title(f'GP State at {current_date.strftime("%Y-%m-%d")} | {len(X_current)} datapoints')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save with unique filename
    plot_filename = f'gp_backtest_state.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved GP state plot: {plot_filename}")

def main(cfg: Config = Config()):
    print("="*60)
    print("BITCOIN PORTFOLIO BACKTESTING")
    print("="*60)
    
    # Load data for the entire backtesting period
    start_dt = pd.to_datetime(cfg.start_datetime)
    end_dt = pd.to_datetime(cfg.end_datetime)
    
    print(f"Backtesting period: {cfg.start_datetime} to {cfg.end_datetime}")
    print(f"Starting wealth: ${cfg.starting_wealth}")
    
    # Load full dataset to get indices
    df = pd.read_csv(cfg.data_path, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    
    # Find start and end indices
    start_idx = df[df['timestamp'] >= start_dt].index[0]
    end_idx = df[df['timestamp'] <= end_dt].index[-1]
    
    print(f"Data points in backtesting period: {end_idx - start_idx + 1}")
    
    # Initialize portfolios
    strategic_wealth = cfg.starting_wealth
    cash_wealth = cfg.starting_wealth
    btc_wealth = cfg.starting_wealth
    
    # Portfolio tracking
    strategic_allocations = []
    strategic_wealths = []
    cash_wealths = []
    btc_wealths = []
    dates = []
    btc_prices = []
    
    # Get initial BTC price for buy-and-hold
    initial_btc_price = df.iloc[start_idx]['price']
    btc_holdings = btc_wealth / initial_btc_price  # Buy BTC at start, hold forever
    
    print(f"Initial BTC price: ${initial_btc_price:.2f}")
    print(f"Buy-and-hold BTC shares: {btc_holdings:.6f}")
    
    # Initialize GP model
    gp_config = OnlineGPConfig()
    gp_model = OnlineVariationalGP(gp_config)
    
    print("\nStarting backtesting simulation...")
    print("-" * 60)
    
    # Loop through each time step
    for current_idx in range(start_idx, end_idx + 1):
        current_date = df.iloc[current_idx]['timestamp']
        current_price = df.iloc[current_idx]['price']
        current_log_price = np.log(current_price)
        
        # Get data up to current point
        data_up_to_now = df.iloc[0:current_idx + 1].copy()
        # Do NOT reset index!
        # data_up_to_now.reset_index(drop=True, inplace=True)
        
        # Use actual indices for X_current
        X_current = data_up_to_now.index.values
        y_current = np.log(data_up_to_now['price'].astype(float).values)
        
        # Fit log trend up to current point
        try:
            log_params = fit_log_trend(X_current, y_current)
            log_trend = create_log_trend_function(log_params)
            residuals = y_current - log_trend(X_current)
        except Exception as e:
            print(f"Warning: Log trend fitting failed at {current_date}: {e}")
            # Use previous parameters or simple trend
            log_trend = lambda x: np.mean(y_current)
            residuals = y_current - log_trend(X_current)
        
        # Train/update GP model
        if current_idx == start_idx:
            # Initial fit
            print(f"Initial GP training with {len(residuals)} data points...")
            try:
                gp_model.fit_initial(X_current, residuals)
                print("✓ Initial GP training completed")
            except Exception as e:
                print(f"Warning: Initial GP training failed: {e}")
                optimal_btc_allocation = 0.5  # Default allocation
        else:
            try:
                if cfg.gp_update_mode == "full":
                    # Refit GP from scratch with all data up to now
                    print(f"Full GP retrain with {len(residuals)} data points...")
                    gp_model.fit_initial(X_current, residuals)
                else:
                    # Online update (default)
                    new_x = current_idx
                    new_y = residuals[-1]
                    gp_model.add_datapoint(new_x, new_y)
                    if len(gp_model.train_x) % 50 == 0:
                        print(f"Retrain full model {len(gp_model.train_x)} total points...")
                        gp_model.fit_initial(X_current, residuals)

                plot_backtesting_gp_state(gp_model, X_current, y_current, current_date, cfg)
            except Exception as e:
                print(f"Warning: GP update failed at {current_date}: {e}")
                optimal_btc_allocation = 0.5  # Default allocation

        # Make predictions for portfolio optimization
        try:
            # Predict future points for horizon - USE ACTUAL FUTURE INDICES
            future_X = np.arange(current_idx + 1, current_idx + 1 + cfg.horizon_weeks)
            residual_pred, residual_std = gp_model.predict(future_X)
            
            # Add back log trend to get log price predictions
            log_price_pred = residual_pred + log_trend(future_X)
            log_price_std = residual_std  # Uncertainty carries through
            
            # Extract mu and sigma sequences for optimization
            mu_seq = log_price_pred
            sigma_seq = log_price_std
            
            # Optimize portfolio allocation
            optimal_btc_allocation = optimize_portfolio_for_current_state(
                current_log_price, mu_seq, sigma_seq, cfg)
            
            # Print the optimised portfolio for this step
            print(f"=======================================\n"
                  f"Allocation (BTC): {optimal_btc_allocation:.4f}\n"
                  f"=======================================")

        except Exception as e:
            print(f"Warning: Portfolio optimization failed at {current_date}: {e}")
            optimal_btc_allocation = 0.5  # Default allocation
        
        # Calculate returns if not the first period
        if current_idx > start_idx:
            prev_price = df.iloc[current_idx - 1]['price']
            current_price = df.iloc[current_idx]['price']
            
            # Update strategic portfolio - FIX THE BUG
            strategic_wealth = prev_strategic_cash + prev_strategic_btc * current_price
            
            # Update BTC portfolio (buy-and-hold)
            btc_wealth = btc_holdings * current_price
            
            # Cash portfolio stays the same (no change needed)
        
        # Record current allocation and rebalance strategic portfolio
        strategic_cash = strategic_wealth * (1 - optimal_btc_allocation)
        strategic_btc = strategic_wealth * optimal_btc_allocation / current_price
        
        # Store for next iteration
        prev_strategic_cash = strategic_cash
        prev_strategic_btc = strategic_btc
        
        # Record data
        strategic_allocations.append(optimal_btc_allocation)
        strategic_wealths.append(strategic_wealth)
        cash_wealths.append(cash_wealth)
        btc_wealths.append(btc_wealth)
        dates.append(current_date)
        btc_prices.append(current_price)
        
        # Print progress
        if (current_idx - start_idx) % 4 == 0:  # Every ~quarter
            print(f"Date: {current_date.strftime('%Y-%m-%d')} | "
                  f"BTC: ${current_price:8.2f} | "
                  f"Allocation: {optimal_btc_allocation:5.1%} | "
                  f"Strategic: ${strategic_wealth:8.0f} | "
                  f"Cash: ${cash_wealth:8.0f} | "
                  f"BTC: ${btc_wealth:8.0f}")
    
    # Final results
    final_strategic = strategic_wealths[-1]
    final_cash = cash_wealths[-1]
    final_btc = btc_wealths[-1]
    
    print("\n" + "="*60)
    print("BACKTESTING RESULTS")
    print("="*60)
    print(f"Period: {cfg.start_datetime} to {cfg.end_datetime}")
    print(f"Initial wealth: ${cfg.starting_wealth:,.2f}")
    print()
    print("FINAL WEALTH:")
    print(f"  Strategic Portfolio: ${final_strategic:,.2f}")
    print(f"  Cash Only:          ${final_cash:,.2f}")
    print(f"  BTC Only:           ${final_btc:,.2f}")
    print()
    print("TOTAL RETURNS:")
    strategic_return = (final_strategic / cfg.starting_wealth - 1) * 100
    cash_return = (final_cash / cfg.starting_wealth - 1) * 100
    btc_return = (final_btc / cfg.starting_wealth - 1) * 100
    
    print(f"  Strategic Portfolio: {strategic_return:+7.2f}%")
    print(f"  Cash Only:          {cash_return:+7.2f}%")
    print(f"  BTC Only:           {btc_return:+7.2f}%")
    print()
    print("OUTPERFORMANCE:")
    strategic_vs_cash = strategic_return - cash_return
    strategic_vs_btc = strategic_return - btc_return
    
    print(f"  Strategic vs Cash:   {strategic_vs_cash:+7.2f}%")
    print(f"  Strategic vs BTC:    {strategic_vs_btc:+7.2f}%")
    print()
    
    # Additional statistics
    strategic_wealths = np.array(strategic_wealths)
    btc_wealths = np.array(btc_wealths)
    strategic_allocations = np.array(strategic_allocations)
    
    print("PORTFOLIO STATISTICS:")
    print(f"  Average BTC allocation: {np.mean(strategic_allocations):7.2%}")
    print(f"  Min BTC allocation:     {np.min(strategic_allocations):7.2%}")
    print(f"  Max BTC allocation:     {np.max(strategic_allocations):7.2%}")
    print(f"  Allocation volatility:  {np.std(strategic_allocations):7.2%}")
    
    # Wealth volatility (week-over-week changes)
    strategic_returns = np.diff(strategic_wealths) / strategic_wealths[:-1]
    btc_returns = np.diff(btc_wealths) / btc_wealths[:-1]
    
    print(f"  Strategic volatility:   {np.std(strategic_returns)*100:7.2f}%")
    print(f"  BTC volatility:         {np.std(btc_returns)*100:7.2f}%")
    
    print()
    print("✓ Backtesting completed successfully!")
    
    return {
        'strategic_wealth': strategic_wealths,
        'cash_wealth': cash_wealths,
        'btc_wealth': btc_wealths,
        'strategic_allocations': strategic_allocations,
        'dates': dates,
        'btc_prices': btc_prices,
        'final_results': {
            'strategic': final_strategic,
            'cash': final_cash,
            'btc': final_btc,
            'strategic_return': strategic_return,
            'cash_return': cash_return,
            'btc_return': btc_return
        }
    }

if __name__ == "__main__":
    main()