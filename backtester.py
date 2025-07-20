import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass
from datetime import datetime
import joblib
import tempfile
import shutil

# Add subdirectories to path for imports
sys.path.append('1-log-fit')
sys.path.append('2-gp_fit')
sys.path.append('3-optimise_portfolio')

from scipy.optimize import curve_fit

# Import from subdirectories
from online_gp_fit import OnlineVariationalGP, load_log_trend
from get_utility_function import get_utility_func

@dataclass(frozen=True)
class BacktesterConfig:
    # Data paths
    price_csv: str = '0-data/bitcoin_combined_weekly_data.csv'
    log_trend_params_path: str = '1-log-fit/log_trend_params.pkl'
    
    # Backtest settings
    starting_wealth: float = 1000
    start_datetime: str = '2020-01-01'  # YYYY-MM-DD format
    end_datetime: str = '2020-03-31'    # YYYY-MM-DD format
    
    # Portfolio optimization settings
    preference_curve: str = 'max_expected_wealth'
    gamma: float = 5  # Only used if utility_function is 'crra'
    step_threshold: float = 1100
    step_steepness: float = 100.0
    
    horizon_weeks: int = 1
    rebalance_every: int = 1  # weeks
    
    optimisation_method: str = 'bayesian_with_refinement'
    n_calls_optimiser: int = 15
    
    # GP settings
    gp_inducing_points: int = 100
    gp_training_iter: int = 300
    gp_learning_rate: float = 0.001

@dataclass(frozen=True) 
class PortfolioOptimizerConfig:
    """Config format expected by the portfolio optimizer"""
    x_pred_pkl: str = 'temp_X_pred.npy'
    y_pred_pkl: str = 'temp_y_pred.npy'
    y_std_pkl: str = 'temp_y_std.npy'
    price_csv: str = 'temp_price.csv'
    initial_wealth: float = 1000
    preference_curve: str = 'max_expected_wealth'
    gamma: float = 5
    step_threshold: float = 1100
    step_steepness: float = 100.0
    horizon_weeks: int = 1
    rebalance_every: int = 1
    optimisation_method: str = 'bayesian_with_refinement'
    n_calls_optimiser: int = 15

def log_func(x, a, b, c):
    """Log trend function"""
    z = b * x + 1
    z = np.clip(z, 1e-8, np.inf)
    return a * np.log(z) + c

def fit_log_trend_to_data(X, y):
    """Fit log trend to data subset"""
    params, _ = curve_fit(
        log_func,
        X,
        y,
        p0=[1, 0.01, 1],
        bounds=([0, 1e-6, -np.inf], [np.inf, 1.0, np.inf]),
        maxfev=5000
    )
    return params

def calculate_grid_parameters(T, max_total_paths=1000000):
    """Calculate grid parameters for optimization"""
    grid_points_per_dim = min(100, int(max_total_paths**(1/T)))
    
    actual_paths = grid_points_per_dim ** T
    if actual_paths > max_total_paths:
        grid_points_per_dim -= 1
        actual_paths = grid_points_per_dim ** T
    
    return grid_points_per_dim, actual_paths

def objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim):
    """Portfolio optimization objective function"""
    from scipy.stats import norm
    
    utility = get_utility_func(cfg)

    T = cfg.horizon_weeks // cfg.rebalance_every
    assert len(p) == T, f"Expected p of length {T}"

    # Build 1D grids for each future rebalance log-price x_t
    grid_limits = [
        np.linspace(mu_seq[t] - 4*sigma_seq[t], mu_seq[t] + 4*sigma_seq[t], grid_points_per_dim)
        for t in range(T)
    ]
    dx = np.array([g[1] - g[0] for g in grid_limits])

    # Create meshgrid for all path combinations
    grids = np.meshgrid(*grid_limits, indexing='ij')
    
    # Stack to get all paths: shape (n_total_paths, T)
    all_paths = np.stack([g.ravel() for g in grids], axis=1)
    n_paths = all_paths.shape[0]

    # Start with log wealth
    log_wealth = np.full(n_paths, np.log(cfg.initial_wealth))
    x_prev = np.full(n_paths, current_log_price)

    # Pre-compute p array for faster indexing
    p_array = np.array(p)
    
    for t in range(T):
        x_now = all_paths[:, t]
        
        # Calculate log returns
        log_return = x_now - x_prev
        
        # Update log wealth
        portfolio_return = (1 - p_array[t]) + p_array[t] * np.exp(log_return)
        portfolio_return = np.maximum(portfolio_return, 1e-10)
        
        log_wealth += np.log(portfolio_return)
        x_prev = x_now

    # Calculate utilities
    utilities = utility(log_wealth)

    # Calculate probabilities
    mu_array = np.array([mu_seq[t] for t in range(T)])
    sigma_array = np.array([sigma_seq[t] for t in range(T)])
    
    log_probs = np.sum(
        norm.logpdf(all_paths, loc=mu_array, scale=sigma_array), 
        axis=1
    )
    prob_densities = np.exp(log_probs)

    # Integration
    volume_element = np.prod(dx)
    total = np.sum(utilities * prob_densities) * volume_element

    return total

def run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, months, grid_points_per_dim):
    """Run Bayesian optimization"""
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    
    # Search space: p_t in [0, 1] for each month
    search_space = [Real(0.0, 1.0, name=f"p{i}") for i in range(months)]

    @use_named_args(search_space)
    def objective_wrapped(**kwargs):
        p = np.array([kwargs[f"p{i}"] for i in range(months)])
        util = objective_func(p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
        return -util  # Negative for minimisation

    result = gp_minimize(
        func=objective_wrapped,
        dimensions=search_space,
        n_calls=cfg.n_calls_optimiser,
        n_initial_points=10,
        acq_func="EI",
        random_state=42,
        verbose=False  # Keep quiet during backtesting
    )

    optimal_p = np.array(result.x)
    max_utility = -result.fun

    return optimal_p, max_utility, result

def coordinate_descent_refinement(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim, max_recursive_calls=3):
    """Coordinate descent refinement"""
    refinement_deltas = np.array([-0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1])
    
    T = len(initial_p)
    current_p = np.array(initial_p)
    current_utility = objective_func(initial_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    
    overall_improvement = False
    found_edge_improvement = False
    
    # Iterate through each dimension
    for dim in range(T):
        best_delta = 0
        best_utility_for_dim = current_utility
        
        # Try each delta for this dimension
        for delta in refinement_deltas:
            test_p = current_p.copy()
            test_p[dim] += delta
            
            # Check bounds
            if test_p[dim] < 0 or test_p[dim] > 1:
                continue
            
            test_utility = objective_func(test_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
            
            if test_utility > best_utility_for_dim:
                best_utility_for_dim = test_utility
                best_delta = delta
        
        # Apply the best delta for this dimension
        if best_delta != 0:
            current_p[dim] += best_delta
            current_utility = best_utility_for_dim
            overall_improvement = True
            
            # Check if we hit the edge (±0.1) - suggests more improvement possible
            if abs(best_delta) == 0.1:
                found_edge_improvement = True
    
    # If we found improvement at the edge, recursively call for further refinement
    if found_edge_improvement and max_recursive_calls > 0:
        final_p, final_utility = coordinate_descent_refinement(
            current_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim, 
            max_recursive_calls - 1
        )
        return final_p, final_utility
    
    return current_p, current_utility

def optimize_portfolio(mu_seq, sigma_seq, current_log_price, cfg):
    """Optimize portfolio allocation"""
    T = cfg.horizon_weeks // cfg.rebalance_every
    grid_points_per_dim, actual_paths = calculate_grid_parameters(T)
    
    if cfg.optimisation_method == "bayesian":
        optimal_p, max_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
    elif cfg.optimisation_method == "bayesian_with_refinement":
        bayesian_p, bayesian_util, result = run_bayesian_optimisation(cfg, mu_seq, sigma_seq, current_log_price, T, grid_points_per_dim)
        optimal_p, max_util = coordinate_descent_refinement(bayesian_p, mu_seq, sigma_seq, current_log_price, cfg, grid_points_per_dim)
    
    return optimal_p

class Backtester:
    def __init__(self, cfg: BacktesterConfig):
        self.cfg = cfg
        self.data = None
        self.gp_model = None
        self.log_trend_params = None
        
        # Portfolio tracking
        self.portfolio_values = []
        self.btc_values = []
        self.cash_values = []
        self.allocations = []
        self.dates = []
        
    def load_data(self):
        """Load and filter price data"""
        df = pd.read_csv(self.cfg.price_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values(by='timestamp')
        
        # Filter by date range
        start_date = pd.to_datetime(self.cfg.start_datetime)
        end_date = pd.to_datetime(self.cfg.end_datetime)
        
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        self.data = df[mask].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} data points from {self.cfg.start_datetime} to {self.cfg.end_datetime}")
        print(f"Price range: ${self.data['price'].min():.2f} to ${self.data['price'].max():.2f}")
        
    def initialize_gp_model(self):
        """Initialize the GP model"""
        # Create a temporary config for the GP
        from online_gp_fit import Config as GPConfig
        
        gp_cfg = GPConfig(
            inducing_points=self.cfg.gp_inducing_points,
            training_iter=self.cfg.gp_training_iter,
            learning_rate=self.cfg.gp_learning_rate
        )
        
        self.gp_model = OnlineVariationalGP(gp_cfg)
        
    def run_backtest(self):
        """Run the main backtest loop"""
        self.load_data()
        self.initialize_gp_model()
        
        # Initialize portfolios
        initial_price = self.data['price'].iloc[0]
        initial_log_price = np.log(initial_price)
        
        # Portfolio starts with all cash
        portfolio_wealth = self.cfg.starting_wealth
        portfolio_btc_amount = 0.0
        portfolio_cash_amount = self.cfg.starting_wealth
        
        # Benchmark portfolios
        btc_wealth = self.cfg.starting_wealth  # All BTC
        btc_amount = self.cfg.starting_wealth / initial_price
        
        cash_wealth = self.cfg.starting_wealth  # All cash
        
        print(f"Starting backtest...")
        print(f"Initial price: ${initial_price:.2f}")
        print(f"Initial wealth: ${self.cfg.starting_wealth:.2f}")
        print(f"BTC benchmark: {btc_amount:.6f} BTC")
        
        for i in range(len(self.data)):
            current_date = self.data['timestamp'].iloc[i]
            current_price = self.data['price'].iloc[i]
            current_log_price = np.log(current_price)
            
            # Update portfolio wealth based on current price (before any rebalancing)
            portfolio_wealth = portfolio_btc_amount * current_price + portfolio_cash_amount
            
            print(f"\nStep {i+1}/{len(self.data)}: {current_date.strftime('%Y-%m-%d')} - Price: ${current_price:.2f}")
            
            # Get data up to current point
            historical_data = self.data.iloc[:i+1]
            
            # Fit log trend to historical data
            X_hist = np.arange(len(historical_data))
            y_hist = np.log(historical_data['price'].values)
            
            if len(historical_data) >= 3:  # Need minimum data points for curve fitting
                try:
                    log_trend_params = fit_log_trend_to_data(X_hist, y_hist)
                    residuals = y_hist - log_func(X_hist, *log_trend_params)
                    
                    # Train/update GP model
                    if i == 0:
                        # First point - need more data for initial fit
                        pass
                    elif i == 1:
                        # Second point - still need more data 
                        pass
                    elif i == 2:
                        # Third point - can start GP training
                        self.gp_model.fit_initial(X_hist, residuals)
                    else:
                        # Add new datapoint to GP
                        new_x = len(historical_data) - 1
                        new_residual = residuals[-1]
                        self.gp_model.add_datapoint(new_x, new_residual)
                    
                    # Make predictions for optimization (if we have a trained GP)
                    if i >= 2:
                        # Predict future points
                        future_x = np.arange(len(historical_data), len(historical_data) + self.cfg.horizon_weeks)
                        
                        # Get GP predictions for residuals
                        gp_residual_pred, gp_residual_std = self.gp_model.predict(future_x)
                        
                        # Add back log trend to get full price predictions
                        log_trend_pred = log_func(future_x, *log_trend_params)
                        log_price_pred = gp_residual_pred + log_trend_pred
                        log_price_std = gp_residual_std  # Assume trend is deterministic
                        
                        # Create config for portfolio optimizer
                        opt_cfg = PortfolioOptimizerConfig(
                            initial_wealth=portfolio_wealth,
                            preference_curve=self.cfg.preference_curve,
                            gamma=self.cfg.gamma,
                            step_threshold=self.cfg.step_threshold,
                            step_steepness=self.cfg.step_steepness,
                            horizon_weeks=self.cfg.horizon_weeks,
                            rebalance_every=self.cfg.rebalance_every,
                            optimisation_method=self.cfg.optimisation_method,
                            n_calls_optimiser=self.cfg.n_calls_optimiser
                        )
                        
                        # Optimize portfolio
                        mu_seq = log_price_pred[:self.cfg.horizon_weeks//self.cfg.rebalance_every]
                        sigma_seq = log_price_std[:self.cfg.horizon_weeks//self.cfg.rebalance_every]
                        
                        optimal_allocation = optimize_portfolio(mu_seq, sigma_seq, current_log_price, opt_cfg)
                        btc_allocation = optimal_allocation[0]  # First period allocation
                        
                        print(f"Optimal BTC allocation: {btc_allocation:.3f}")
                        
                        # Rebalance portfolio
                        target_btc_value = portfolio_wealth * btc_allocation
                        target_cash_value = portfolio_wealth * (1 - btc_allocation)
                        
                        portfolio_btc_amount = target_btc_value / current_price
                        portfolio_cash_amount = target_cash_value
                        
                        self.allocations.append(btc_allocation)
                    else:
                        # Not enough data for optimization - stay in cash
                        btc_allocation = 0.0
                        portfolio_btc_amount = 0.0
                        portfolio_cash_amount = portfolio_wealth
                        self.allocations.append(btc_allocation)
                        
                except Exception as e:
                    print(f"Error in optimization: {e}")
                    # Stay in cash if optimization fails
                    btc_allocation = 0.0
                    portfolio_btc_amount = 0.0
                    portfolio_cash_amount = portfolio_wealth
                    self.allocations.append(btc_allocation)
            else:
                # Not enough data - stay in cash
                btc_allocation = 0.0
                portfolio_btc_amount = 0.0
                portfolio_cash_amount = portfolio_wealth
                self.allocations.append(btc_allocation)
            
            # Portfolio wealth was already updated at the beginning of the loop
            # Update benchmark portfolios
            btc_wealth = btc_amount * current_price
            # cash_wealth stays the same
            
            # Record values
            self.portfolio_values.append(portfolio_wealth)
            self.btc_values.append(btc_wealth)
            self.cash_values.append(cash_wealth)
            self.dates.append(current_date)
            
            print(f"Portfolio: ${portfolio_wealth:.2f} | BTC: ${btc_wealth:.2f} | Cash: ${cash_wealth:.2f}")
        
        self.print_results()
    
    def print_results(self):
        """Print backtest results"""
        initial_wealth = self.cfg.starting_wealth
        final_portfolio = self.portfolio_values[-1]
        final_btc = self.btc_values[-1]
        final_cash = self.cash_values[-1]
        
        portfolio_return = (final_portfolio / initial_wealth - 1) * 100
        btc_return = (final_btc / initial_wealth - 1) * 100
        cash_return = (final_cash / initial_wealth - 1) * 100
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {self.cfg.start_datetime} to {self.cfg.end_datetime}")
        print(f"Initial wealth: ${initial_wealth:.2f}")
        print()
        print(f"Final Portfolio Value: ${final_portfolio:.2f} ({portfolio_return:+.2f}%)")
        print(f"Final BTC Value:      ${final_btc:.2f} ({btc_return:+.2f}%)")
        print(f"Final Cash Value:     ${final_cash:.2f} ({cash_return:+.2f}%)")
        print()
        print(f"Portfolio vs BTC:  {final_portfolio/final_btc:.3f}x")
        print(f"Portfolio vs Cash: {final_portfolio/final_cash:.3f}x")
        print()
        
        # Calculate some basic stats
        portfolio_returns = np.array(self.portfolio_values)
        portfolio_daily_returns = np.diff(portfolio_returns) / portfolio_returns[:-1]
        
        btc_returns = np.array(self.btc_values)
        btc_daily_returns = np.diff(btc_returns) / btc_returns[:-1]
        
        print(f"Portfolio Volatility: {np.std(portfolio_daily_returns)*100:.2f}%")
        print(f"BTC Volatility:       {np.std(btc_daily_returns)*100:.2f}%")
        print()
        print(f"Max Portfolio Value: ${np.max(self.portfolio_values):.2f}")
        print(f"Min Portfolio Value: ${np.min(self.portfolio_values):.2f}")
        print()
        print(f"Average BTC Allocation: {np.mean(self.allocations):.3f}")
        print(f"Max BTC Allocation:     {np.max(self.allocations):.3f}")
        print(f"Min BTC Allocation:     {np.min(self.allocations):.3f}")

def main():
    # Configure backtest
    cfg = BacktesterConfig(
        starting_wealth=1000,
        start_datetime='2021-01-01',
        end_datetime='2022-12-31',
        horizon_weeks=1,
        rebalance_every=1,
        optimisation_method='bayesian_with_refinement',
        n_calls_optimiser=15
    )
    
    # Run backtest
    backtester = Backtester(cfg)
    backtester.run_backtest()

if __name__ == "__main__":
    main()
