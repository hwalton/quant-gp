# log-fit.py

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import joblib

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

INPUT_CSV = '../0-data/btc_monthly_prices.csv'
OUTPUT_PKL = 'log_trend_params.pkl'

# ─── LOAD AND PREPARE DATA ──────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV, sep=';')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')
y = np.log(df['close'].astype(float).values)
X = np.arange(len(y))

# ─── FIT LOG TREND ──────────────────────────────────────────────────────────────

def log_func(x, a, b, c):
    return a * np.log(b * x + 1) + c

params, _ = curve_fit(log_func, X, y, p0=[1, 0.01, 1])
print(f"Fitted log curve params: a={params[0]:.4f}, b={params[1]:.6f}, c={params[2]:.4f}")

joblib.dump(params, OUTPUT_PKL)
print(f"Saved trend params to {OUTPUT_PKL}")
