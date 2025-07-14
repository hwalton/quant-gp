# log-fit.py

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import joblib
import os

DATA_PATH = '../0-data/btc_weekly_prices.csv'
OUTPUT_PKL = 'log_trend_params.pkl'

def load_data(path):
    df = pd.read_csv(path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    y = np.log(df['close'].astype(float).values)
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
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    X, y = load_data(DATA_PATH)
    params = fit_log_trend(X, y)
    print(f"Fitted log curve params: a={params[0]:.4f}, b={params[1]:.6f}, c={params[2]:.4f}")
    save_params(params, OUTPUT_PKL)

if __name__ == '__main__':
    main()
