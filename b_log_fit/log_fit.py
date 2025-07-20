# log-fit.py

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import joblib
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    data_path: str = '../a_data/bitcoin_combined_weekly_data.csv'
    output_pkl: str = 'log_trend_params.pkl'
    cycle_length: int = 208

def load_data(cfg: Config):
    df = pd.read_csv(cfg.data_path, sep=',')
    print("Columns:", df.columns.tolist())
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    y_all = np.log(df['price'].astype(float).values)
    
    # Use all data instead of trimming to cycles
    y = y_all
    X = np.arange(len(y))
    
    return X, y

def log_func(x, a, b, c):
    z = b * x + 1
    z = np.clip(z, 1e-8, np.inf)  # Avoid log(0) or negative
    return a * np.log(z) + c

def fit_log_trend(X, y):
    params, _ = curve_fit(
        log_func,
        X,
        y,
        p0=[1, 0.01, 1],
        bounds=([0, 1e-6, -np.inf], [np.inf, 1.0, np.inf]),
        maxfev=5000
    )
    return params

def save_params(params, output_path):
    joblib.dump(params, output_path)
    print(f"Saved trend params to {output_path}")

def main():
    cfg = Config()
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(f"Data file not found: {cfg.data_path}")
    
    X, y = load_data(cfg)
    params = fit_log_trend(X, y)
    print(f"Fitted log curve params: a={params[0]:.4f}, b={params[1]:.6f}, c={params[2]:.4f}")
    save_params(params, cfg.output_pkl)

if __name__ == '__main__':
    main()
