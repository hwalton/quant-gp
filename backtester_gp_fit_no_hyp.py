import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import joblib
import os
import sys

# Add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.utils import load_data
from b_log_fit.log_fit import fit_log_trend, log_func, Config as LogFitConfig
from c_gp_fit.gp_fit import fit_gp, build_fixed_kernel, predict_gp, Config as GPFitConfig
from d_optimise_portfolio.optimise_portfolio import load_gp_predictions, objective_func, run_bayesian_optimisation, coordinate_descent_refinement, calculate_grid_parameters
from d_optimise_portfolio.config import Config as OptimiseConfig

@dataclass(frozen=True)
class Config:
    data_path: str = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')
    starting_wealth: float = 1000
    start_datetime: str = "2023-06-01"
    end_datetime: str = "2024-06-01"
    preference_curve: str = 'identity'
    horizon_weeks: int = 1
    rebalance_every: int = 1
    optimisation_method: str = 'bayesian_with_refinement'
    n_calls_optimiser: int = 15
    gamma: float = 5
    step_threshold: float = 1100
    step_steepness: float = 100.0

def create_log_trend_function(params):
    def trend_func(x):
        return log_func(x, params[0], params[1], params[2])
    return trend_func

def optimize_portfolio_for_current_state(current_log_price, mu_seq, sigma_seq, cfg):
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
    return optimal_p[0]

def plot_backtesting_gp_state(gp_model, X_current, y_current, current_date, cfg):
    """Plot current GP state during backtesting"""
    import matplotlib.pyplot as plt

    # Make predictions for plotting
    X_plot = np.linspace(X_current.min(), X_current.max() + 50, 500)
    y_pred_plot, y_std_plot = gp_model.predict(X_plot.reshape(-1, 1), return_std=True)

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
    print("BITCOIN PORTFOLIO BACKTESTING (scikit-learn GP)")
    print("="*60)
    start_dt = pd.to_datetime(cfg.start_datetime)
    end_dt = pd.to_datetime(cfg.end_datetime)
    print(f"Backtesting period: {cfg.start_datetime} to {cfg.end_datetime}")
    print(f"Starting wealth: ${cfg.starting_wealth}")

    df = pd.read_csv(cfg.data_path, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    start_idx = df[df['timestamp'] >= start_dt].index[0]
    end_idx = df[df['timestamp'] <= end_dt].index[-1]
    print(f"Data points in backtesting period: {end_idx - start_idx + 1}")

    strategic_wealth = cfg.starting_wealth
    cash_wealth = cfg.starting_wealth
    btc_wealth = cfg.starting_wealth

    strategic_allocations = []
    strategic_wealths = []
    cash_wealths = []
    btc_wealths = []
    dates = []
    btc_prices = []

    initial_btc_price = df.iloc[start_idx]['price']
    btc_holdings = btc_wealth / initial_btc_price
    print(f"Initial BTC price: ${initial_btc_price:.2f}")
    print(f"Buy-and-hold BTC shares: {btc_holdings:.6f}")

    print("\nStarting backtesting simulation...")
    print("-" * 60)

    # Build kernel ONCE with fixed hyperparameters
    kernel = build_fixed_kernel(
        rbf_lengthscale=10.0,
        rbf_constant=1.0,
        periodic_lengthscale=10.0,
        periodic_period=298.0,
        periodic_constant=1.0,
        noise_level=1.0
    )

    for current_idx in range(start_idx, end_idx + 1):
        current_date = df.iloc[current_idx]['timestamp']
        current_price = df.iloc[current_idx]['price']
        current_log_price = np.log(current_price)
        data_up_to_now = df.iloc[0:current_idx + 1].copy()
        X_current = data_up_to_now.index.values
        y_current = np.log(data_up_to_now['price'].astype(float).values)

        try:
            log_params = fit_log_trend(X_current, y_current)
            log_trend = create_log_trend_function(log_params)
            residuals = y_current - log_trend(X_current)
        except Exception as e:
            print(f"Warning: Log trend fitting failed at {current_date}: {e}")
            log_trend = lambda x: np.mean(y_current)
            residuals = y_current - log_trend(X_current)

        # Fit GP with fixed kernel (no hyperparameter optimization)
        try:
            gp_model = fit_gp(X_current, residuals, kernel, opt=False)
        except Exception as e:
            print(f"Warning: GP fitting failed at {current_date}: {e}")
            optimal_btc_allocation = 0.5

        plot_backtesting_gp_state(gp_model, X_current, y_current, current_date, cfg)

        # Make predictions for portfolio optimization
        try:
            future_X = np.arange(current_idx + 1, current_idx + 1 + cfg.horizon_weeks)
            X_pred = future_X.reshape(-1, 1)
            y_resid_pred, y_std = gp_model.predict(X_pred, return_std=True)
            log_price_pred = y_resid_pred + log_trend(future_X)
            log_price_std = y_std
            mu_seq = log_price_pred
            sigma_seq = log_price_std
            optimal_btc_allocation = optimize_portfolio_for_current_state(
                current_log_price, mu_seq, sigma_seq, cfg)
            print(f"=======================================\n"
                  f"Allocation (BTC): {optimal_btc_allocation:.4f}\n"
                  f"=======================================")
        except Exception as e:
            print(f"Warning: Portfolio optimization failed at {current_date}: {e}")
            optimal_btc_allocation = 0.5

        if current_idx > start_idx:
            prev_price = df.iloc[current_idx - 1]['price']
            current_price = df.iloc[current_idx]['price']
            strategic_wealth = prev_strategic_cash + prev_strategic_btc * current_price
            btc_wealth = btc_holdings * current_price

        strategic_cash = strategic_wealth * (1 - optimal_btc_allocation)
        strategic_btc = strategic_wealth * optimal_btc_allocation / current_price
        prev_strategic_cash = strategic_cash
        prev_strategic_btc = strategic_btc

        strategic_allocations.append(optimal_btc_allocation)
        strategic_wealths.append(strategic_wealth)
        cash_wealths.append(cash_wealth)
        btc_wealths.append(btc_wealth)
        dates.append(current_date)
        btc_prices.append(current_price)

        if (current_idx - start_idx) % 4 == 0:
            print(f"Date: {current_date.strftime('%Y-%m-%d')} | "
                  f"BTC: ${current_price:8.2f} | "
                  f"Allocation: {optimal_btc_allocation:5.1%} | "
                  f"Strategic: ${strategic_wealth:8.0f} | "
                  f"Cash: ${cash_wealth:8.0f} | "
                  f"BTC: ${btc_wealth:8.0f}")

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

    strategic_wealths = np.array(strategic_wealths)
    btc_wealths = np.array(btc_wealths)
    strategic_allocations = np.array(strategic_allocations)

    print("PORTFOLIO STATISTICS:")
    print(f"  Average BTC allocation: {np.mean(strategic_allocations):7.2%}")
    print(f"  Min BTC allocation:     {np.min(strategic_allocations):7.2%}")
    print(f"  Max BTC allocation:     {np.max(strategic_allocations):7.2%}")
    print(f"  Allocation volatility:  {np.std(strategic_allocations):7.2%}")

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