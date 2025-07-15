"""
Main runner for the complete GP-based portfolio optimization pipeline.
"""
from pathlib import Path
import argparse

from config import GPModelConfig
from data.loader import load_btc_data
from log_fit.fit import fit_log_trend
from gp_fit.fit import run_gp_fit
from portfolio.optimize import run_portfolio_optimization

def run_full_pipeline(config: GPModelConfig) -> dict:
    """
    Run the complete analysis pipeline:
    1. Load data
    2. Fit log trend
    3. Fit GP to residuals  
    4. Optimize portfolio
    
    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("GP-BASED PORTFOLIO OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading BTC price data...")
    X, y = load_btc_data(config)
    print(f"Loaded {len(X)} data points")
    
    # Step 2: Fit log trend
    print("\n2. Fitting log trend...")
    log_trend = fit_log_trend(X, y)
    print("Log trend fitting complete")
    
    # Step 3: Fit GP to residuals
    print("\n3. Fitting GP to residuals and making predictions...")
    gp_results = run_gp_fit(config)
    print("GP fitting and prediction complete")
    
    # Step 4: Optimize portfolio
    print("\n4. Optimizing portfolio allocation...")
    portfolio_results = run_portfolio_optimization(config)
    print("Portfolio optimization complete")
    
    # Combine all results
    results = {
        'data_points': len(X),
        'log_trend_params': log_trend.tolist(),
        'gp_results': gp_results,
        'portfolio_results': portfolio_results,
        'config': {
            'utility_function': config.utility_function,
            'initial_wealth': config.initial_wealth,
            'predict_index_offset': config.predict_index_offset
        }
    }
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Optimal BTC allocation: {portfolio_results['optimal_btc_weight']:.1%}")
    print(f"Expected return: {portfolio_results['expected_percent_return']:.1f}%")
    print(f"Volatility (log scale): {portfolio_results['volatility']:.4f}")
    print(f"Utility function: {config.utility_function}")
    
    return results

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run GP-based portfolio optimization')
    parser.add_argument('--utility', type=str, default='log',
                       choices=['identity', 'log', 'sqrt', 'step', 'smooth_step', 
                               'sigmoid', 'tanh', 'tanh_custom', 'crra'],
                       help='Utility function to use')
    parser.add_argument('--wealth', type=float, default=1000,
                       help='Initial wealth amount')
    parser.add_argument('--offset', type=int, default=10,
                       help='Prediction index offset')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Risk aversion parameter for CRRA utility')
    
    args = parser.parse_args()
    
    # Create config with command line arguments  
    config = GPModelConfig(
        utility_function=args.utility,
        initial_wealth=args.wealth,
        predict_index_offset=args.offset,
        crra_gamma=args.gamma
    )
    
    # Run the pipeline
    results = run_full_pipeline(config)
    
    return results

if __name__ == "__main__":
    main()
