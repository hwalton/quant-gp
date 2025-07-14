# log_trend_fit.py

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import joblib

# Load and prepare data
df = pd.read_csv('data/btc_monthly_prices.csv', sep=';')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')
y = np.log(df['close'].astype(float).values)
X = np.arange(len(y))

# Define log function
def log_func(x, a, b, c):
    return a * np.log(b * x + 1) + c

# Fit the log curve
params, _ = curve_fit(log_func, X, y, p0=[1, 0.01, 1])
print(f"Fitted log curve params: a={params[0]:.4f}, b={params[1]:.6f}, c={params[2]:.4f}")

# Save params
joblib.dump(params, "log_trend_params.pkl")
