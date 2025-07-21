import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import os
import sys
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from b_log_fit.log_fit import fit_log_trend, log_func
from c_gp_fit.gp_fit import fit_gp, build_fixed_kernel

from d_optimise_portfolio.config import Config as OptimiseConfig
from d_optimise_portfolio.get_utility_function import get_utility_func
from d_optimise_portfolio.optimise_portfolio import load_gp_predictions, objective_func, run_bayesian_optimisation, coordinate_descent_refinement, calculate_grid_parameters
from d_optimise_portfolio.config import Config as OptimiseConfig

@dataclass(frozen=True)
class Config:
    data_path: str = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')
    starting_wealth: float = 1000
    start_datetime: str = "2019-01-01"
    end_datetime: str = "2020-01-01"
    preference_curve: str = 'identity'
    horizon_weeks: int = 1
    rebalance_every: int = 1
    optimisation_method: str = 'bayesian_with_refinement'
    n_calls_optimiser: int = 15
    gamma: float = 5
    step_threshold: float = 1100
    step_steepness: float = 100.0

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

def create_log_trend_function(params):
    def trend_func(x):
        return log_func(x, params[0], params[1], params[2])
    return trend_func

def backtest_objective(kernel_params, static, iteration=None):
    # Unpack kernel parameters
    rbf_lengthscale, rbf_constant, periodic_lengthscale, periodic_period, periodic_constant, noise_level = kernel_params

    # Unpack static variables
    df, cfg, utility_func = static['df'], static['cfg'], static['utility_func']
    start_idx, end_idx = static['start_idx'], static['end_idx']

    strategic_wealth = cfg.starting_wealth
    cash_wealth = cfg.starting_wealth
    btc_wealth = cfg.starting_wealth

    initial_btc_price = df.iloc[start_idx]['price']
    btc_holdings = btc_wealth / initial_btc_price

    kernel = build_fixed_kernel(
        rbf_lengthscale=rbf_lengthscale,
        rbf_constant=rbf_constant,
        periodic_lengthscale=periodic_lengthscale,
        periodic_period=periodic_period,
        periodic_constant=periodic_constant,
        noise_level=noise_level
    )

    prev_strategic_cash = strategic_wealth
    prev_strategic_btc = 0.0

    last_gp_model = None
    last_X_current = None
    last_y_current = None
    last_date = None

    for current_idx in range(start_idx, end_idx + 1):
        current_price = df.iloc[current_idx]['price']
        data_up_to_now = df.iloc[0:current_idx + 1].copy()
        X_current = data_up_to_now.index.values
        y_current = np.log(data_up_to_now['price'].astype(float).values)

        try:
            log_params = fit_log_trend(X_current, y_current)
            log_trend = create_log_trend_function(log_params)
            residuals = y_current - log_trend(X_current)
        except Exception:
            log_trend = lambda x: np.mean(y_current)
            residuals = y_current - log_trend(X_current)

        try:
            gp_model = fit_gp(X_current, residuals, kernel, opt=False)
            last_gp_model = gp_model
            last_X_current = X_current
            last_y_current = y_current
            last_date = df.iloc[current_idx]['timestamp']
        except Exception:
            return 1e6  # Penalize failed fits

        # Make predictions for portfolio optimization
        try:
            future_X = np.arange(current_idx + 1, current_idx + 1 + cfg.horizon_weeks)
            X_pred = future_X.reshape(-1, 1)
            y_resid_pred, y_std = gp_model.predict(X_pred, return_std=True)
            log_price_pred = y_resid_pred + log_trend(future_X)
            log_price_std = y_std
            mu_seq = log_price_pred
            sigma_seq = log_price_std

            # Portfolio optimization (always use identity utility)
            optimal_btc_allocation = 0.5
            try:
                # Use a simple one-step allocation for speed
                optimal_btc_allocation = mu_seq[0] > np.log(current_price)
                optimal_btc_allocation = float(optimal_btc_allocation)
            except Exception:
                optimal_btc_allocation = 0.5
        except Exception:
            optimal_btc_allocation = 0.5

        if current_idx > start_idx:
            strategic_wealth = prev_strategic_cash + prev_strategic_btc * current_price
            btc_wealth = btc_holdings * current_price

        strategic_cash = strategic_wealth * (1 - optimal_btc_allocation)
        strategic_btc = strategic_wealth * optimal_btc_allocation / current_price
        prev_strategic_cash = strategic_cash
        prev_strategic_btc = strategic_btc

    # Compute utility of final strategic wealth
    final_utility = utility_func(np.log(strategic_wealth))
    minval = -float(final_utility)
    # Plot after full backtest if iteration is provided
    if iteration is not None and last_gp_model is not None:
        plot_backtesting_gp_state_final(
            last_gp_model, last_X_current, last_y_current, last_date, cfg, iteration, minval
        )
    return minval

def plot_backtesting_gp_state_final(gp_model, X_current, y_current, current_date, cfg, iteration, minval):
    """Plot GP state after full backtest for a given kernel hyperparameter set."""
    import matplotlib.pyplot as plt

    X_plot = np.linspace(X_current.min(), X_current.max() + 208, 500)
    y_pred_plot, y_std_plot = gp_model.predict(X_plot.reshape(-1, 1), return_std=True)
    log_trend = create_log_trend_function(fit_log_trend(X_current, y_current))
    y_pred_with_trend = y_pred_plot + log_trend(X_plot)
    y_trend_plot = log_trend(X_plot)

    plt.figure(figsize=(12, 8))
    plt.plot(X_current, y_current, 'kx', label='Historical BTC Log Prices', markersize=4, alpha=0.8)
    plt.plot(X_plot, y_trend_plot, 'r-', label='Log Trend', linewidth=2)
    plt.plot(X_plot, y_pred_with_trend, 'b-', label='GP Prediction', linewidth=2)
    plt.fill_between(X_plot, 
                     y_pred_with_trend - y_std_plot, 
                     y_pred_with_trend + y_std_plot,
                     alpha=0.2, label='GP 1σ', color='skyblue')
    plt.xlabel('Time Index')
    plt.ylabel('Log BTC Price') 
    plt.title(f'GP State (iter {iteration}) | min={minval:.4f} | {current_date.strftime("%Y-%m-%d")} | {len(X_current)} pts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_filename = f'gp_backtest_state_iter_{iteration}_{minval:.4f}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved GP state plot: {plot_filename}")

def main():
    cfg = Config()
    # Load data and static variables ONCE
    df = pd.read_csv(cfg.data_path, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    start_idx = df[df['timestamp'] >= pd.to_datetime(cfg.start_datetime)].index[0]
    end_idx = df[df['timestamp'] <= pd.to_datetime(cfg.end_datetime)].index[-1]
    utility_func = get_utility_func(cfg)

    static = {
        'df': df,
        'cfg': cfg,
        'utility_func': utility_func,
        'start_idx': start_idx,
        'end_idx': end_idx
    }

    # Define search space for kernel hyperparameters
    space = [
        Real(1.0, 20.0, name='rbf_lengthscale'),
        Real(0.1, 2.0, name='rbf_constant'),
        Real(0.1, 10.0, name='periodic_lengthscale'),
        Real(100.0, 300.0, name='periodic_period'),
        Real(0.1, 2.0, name='periodic_constant'),
        Real(0.001, 0.1, name='noise_level')
    ]

    iteration_counter = {'i': 1}
    @use_named_args(space)
    def objective_wrapped(**params):
        param_list = [params[name] for name in [d.name for d in space]]
        iteration = iteration_counter['i']
        minval = backtest_objective(param_list, static, iteration=iteration)
        iteration_counter['i'] += 1
        return minval

    print("Starting hyperparameter optimization (this may take a while)...")
    result = gp_minimize(
        objective_wrapped,
        space,
        n_calls=30,
        n_initial_points=12,
        acq_func="EI",
        random_state=42,
        verbose=True
    )

    print("\nBest kernel parameters found:")
    for name, val in zip([d.name for d in space], result.x):
        print(f"  {name}: {val:.4f}")
    print(f"Best (max) utility: {-result.fun:.4f}")

if __name__ == "__main__":
    main()