#!/usr/bin/env python3
"""
Test the utility distribution function with dummy data.
"""
import sys
import os
sys.path.append('/home/harvey/Git/quant-gp/gp-model')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import GPModelConfig
from step4_portfolio.optimize import plot_utility_distribution, plot_utility_function, plot_expected_utility_curve, plot_wealth_distribution

def test_utility_plots():
    """Test all utility plotting functions with dummy data."""
    print("Testing utility plotting functions...")
    
    config = GPModelConfig()
    
    # Dummy data that would come from GP predictions
    mu = 10.5  # Future log price prediction
    sigma = 0.1  # Volatility
    current_log_price = 10.4  # Current log price
    optimal_weight = 0.6  # Optimal BTC allocation
    
    print("1. Testing utility function plot...")
    plot_utility_function(config)
    print("   ✅ utility_func.png generated")
    
    print("2. Testing utility curve plot...")
    plot_expected_utility_curve(mu, sigma, current_log_price, optimal_weight, config)
    print("   ✅ utility_curve.png generated")
    
    print("3. Testing wealth distribution plot...")
    plot_wealth_distribution(mu, sigma, current_log_price, optimal_weight, config)
    print("   ✅ wealth_distribution.png generated")
    
    print("4. Testing utility distribution plot...")
    plot_utility_distribution(mu, sigma, current_log_price, optimal_weight, config)
    print("   ✅ utility_distribution.png generated")
    
    # Copy to web directory
    import shutil
    web_dir = "/home/harvey/Git/quant-gp/web/static/images"
    os.makedirs(web_dir, exist_ok=True)
    
    outputs_dir = config.outputs_dir
    for file in outputs_dir.glob("*.png"):
        shutil.copy(file, web_dir)
        print(f"   📂 Copied {file.name} to web/static/images/")
    
    print("\n🎉 All utility plots generated successfully!")
    print(f"📁 Check outputs in: {outputs_dir}")
    print(f"🌐 Web images in: {web_dir}")

if __name__ == "__main__":
    test_utility_plots()
