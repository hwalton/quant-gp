import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import os
import sys
from skopt import gp_minimize, forest_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from b_log_fit.log_fit import fit_log_trend, log_func
from c_gp_fit.gp_fit import fit_gp, build_fixed_kernel

from d_optimise_portfolio.config import Config as OptimiseConfig
from d_optimise_portfolio.get_utility_function import get_utility_func
from d_optimise_portfolio.optimise_portfolio import load_gp_predictions, objective_func, run_bayesian_optimisation, coordinate_descent_refinement, calculate_grid_parameters, run_forest_minimize_optimisation
from d_optimise_portfolio.config import Config as OptimiseConfig

@dataclass(frozen=True)
class Config:
    data_path: str = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')
    starting_wealth: float = 1000
    start_datetime: str = "2019-01-01"
    end_datetime: str = "2023-01-01"
    preference_curve: str = 'identity'
    horizon_weeks: int = 1
    rebalance_every: int = 1
    optimisation_method: str = 'forest_minimize'  # 'bayesian', 'bayesian_with_refinement', or 'forest_minimize'
    n_calls_optimiser: int = 20
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
    elif cfg.optimisation_method == "forest_minimize":
        optimal_p, max_util, result = run_forest_minimize_optimisation(
            opt_config, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
        # print(f"Allocation: {np.round(optimal_p, 3)}")
        # optimal_p, max_util = coordinate_descent_refinement(
        #     optimal_p, mu_seq, sigma_seq, current_log_price, opt_config, grid_points_per_dim)
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

    # Add these tracking lists at the top of the function:
    strategic_allocations = []
    strategic_wealths = []
    cash_wealths = []
    btc_wealths = []
    dates = []
    btc_prices = []

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

            try:
                future_X = np.arange(current_idx + 1, current_idx + 1 + cfg.horizon_weeks)
                X_pred = future_X.reshape(-1, 1)
                y_resid_pred, y_std = gp_model.predict(X_pred, return_std=True)
                log_price_pred = y_resid_pred + log_trend(future_X)
                log_price_std = y_std
                mu_seq = log_price_pred
                sigma_seq = log_price_std

                # Use the full optimizer for allocation (uses mean and variance)
                current_log_price = np.log(current_price)
                optimal_btc_allocation = optimize_portfolio_for_current_state(
                    current_log_price, mu_seq, sigma_seq, cfg
                )
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

        # After updating prev_strategic_cash and prev_strategic_btc, add:
        strategic_allocations.append(optimal_btc_allocation)
        strategic_wealths.append(strategic_wealth)
        cash_wealths.append(cash_wealth)
        btc_wealths.append(btc_wealth)
        dates.append(df.iloc[current_idx]['timestamp'])
        btc_prices.append(current_price)

    # Compute utility of final strategic wealth
    final_utility = utility_func(np.log(strategic_wealth))
    minval = -float(final_utility)
    # Plot after full backtest if iteration is provided
    if iteration is not None and last_gp_model is not None:
        plot_backtesting_gp_state_final(
            last_gp_model, last_X_current, last_y_current, last_date, cfg, iteration, minval
        )

    # At the end of the function, after computing minval:
    final_strategic = strategic_wealths[-1]
    final_cash = cash_wealths[-1]
    final_btc = btc_wealths[-1]

    strategic_return = (final_strategic / cfg.starting_wealth - 1) * 100
    cash_return = (final_cash / cfg.starting_wealth - 1) * 100
    btc_return = (final_btc / cfg.starting_wealth - 1) * 100

    strategic_vs_cash = strategic_return - cash_return
    strategic_vs_btc = strategic_return - btc_return

    strategic_wealths = np.array(strategic_wealths)
    btc_wealths = np.array(btc_wealths)
    strategic_allocations = np.array(strategic_allocations)

    strategic_returns = np.diff(strategic_wealths) / strategic_wealths[:-1]
    btc_returns = np.diff(btc_wealths) / btc_wealths[:-1]

    # At the end of backtest_objective, after all calculations:
    stats = {
        "final_strategic": final_strategic,
        "final_cash": final_cash,
        "final_btc": final_btc,
        "strategic_return": strategic_return,
        "cash_return": cash_return,
        "btc_return": btc_return,
        "strategic_vs_cash": strategic_vs_cash,
        "strategic_vs_btc": strategic_vs_btc,
        "avg_alloc": np.mean(strategic_allocations),
        "min_alloc": np.min(strategic_allocations),
        "max_alloc": np.max(strategic_allocations),
        "alloc_vol": np.std(strategic_allocations),
        "strategic_vol": np.std(strategic_returns)*100,
        "btc_vol": np.std(btc_returns)*100,
        "period": (cfg.start_datetime, cfg.end_datetime),
        "initial_wealth": cfg.starting_wealth,
    }
    return minval, stats

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
    best_stats = None
    best_minval = None

    @use_named_args(space)
    def objective_wrapped(**params):
        param_list = [params[name] for name in [d.name for d in space]]
        iteration = iteration_counter['i']
        minval, stats = backtest_objective(param_list, static, iteration=iteration)
        iteration_counter['i'] += 1
        nonlocal best_stats, best_minval
        if best_minval is None or minval < best_minval:
            best_minval = minval
            best_stats = stats
        print(f"Iteration {iteration}: minval={minval:.4f}, params={param_list}")
        return minval

    print("Starting hyperparameter optimization (this may take a while)...")
    if cfg.optimisation_method == "bayesian" or cfg.optimisation_method == "bayesian_with_refinement" or cfg.optimisation_method == "forest_minimizes":
        result = gp_minimize(
            objective_wrapped,
            space,
            n_calls=cfg.n_calls_optimiser,
            n_initial_points=8,
            acq_func="EI",
            random_state=42,
            verbose=True
        )
    elif cfg.optimisation_method == "forest_minimize":
        result = forest_minimize(
            func=objective_wrapped,
            dimensions=space,
            n_calls=cfg.n_calls_optimiser,
            n_initial_points=8,
            acq_func="EI",
            random_state=42,
            verbose=False,
            n_jobs=-1  # parallelism supported
        )

    print("\nBest kernel parameters found:")
    for name, val in zip([d.name for d in space], result.x):
        print(f"  {name}={val:.12f},")
    print(f"Best (max) utility: {-result.fun:.4f}")

    if best_stats is not None:
        print("="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        print(f"Period: {best_stats['period'][0]} to {best_stats['period'][1]}")
        print(f"Initial wealth: ${best_stats['initial_wealth']:,.2f}\n")
        print("FINAL WEALTH:")
        print(f"  Strategic Portfolio: ${best_stats['final_strategic']:,.2f}")
        print(f"  Cash Only:          ${best_stats['final_cash']:,.2f}")
        print(f"  BTC Only:           ${best_stats['final_btc']:,.2f}\n")
        print("TOTAL RETURNS:")
        print(f"  Strategic Portfolio: {best_stats['strategic_return']:+7.2f}%")
        print(f"  Cash Only:          {best_stats['cash_return']:+7.2f}%")
        print(f"  BTC Only:           {best_stats['btc_return']:+7.2f}%\n")
        print("OUTPERFORMANCE:")
        print(f"  Strategic vs Cash:   {best_stats['strategic_vs_cash']:+7.2f}%")
        print(f"  Strategic vs BTC:    {best_stats['strategic_vs_btc']:+7.2f}%\n")
        print("PORTFOLIO STATISTICS:")
        print(f"  Average BTC allocation: {best_stats['avg_alloc']:7.2%}")
        print(f"  Min BTC allocation:     {best_stats['min_alloc']:7.2%}")
        print(f"  Max BTC allocation:     {best_stats['max_alloc']:7.2%}")
        print(f"  Allocation volatility:  {best_stats['alloc_vol']:7.2%}")
        print(f"  Strategic volatility:   {best_stats['strategic_vol']:7.2f}%")
        print(f"  BTC volatility:         {best_stats['btc_vol']:7.2f}%")
        print()

if __name__ == "__main__":
    main()