import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time

# --- Load Data ---
df = pd.read_csv('data/btc_weekly_prices.csv')
X = np.arange(len(df)).reshape(-1, 1)  # weeks as integer indices
y = df['price_usd'].values

# --- Define Kernel with Fixed Parameters ---
# Optional: wrap RBF in ConstantKernel if you want fixed amplitude
kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=5.0, length_scale_bounds="fixed")

# --- Define GP Regressor ---
gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)

# --- Fit and Predict ---
start = time.time()

gp.fit(X, y)
X_pred = np.linspace(0, len(X) + 10, 500).reshape(-1, 1)
y_pred, y_std = gp.predict(X_pred, return_std=True)

end = time.time()
print(f"GP fit + predict completed in {end - start:.3f} seconds")

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'kx', label='Observed BTC prices')
plt.plot(X_pred, y_pred, 'b-', label='GP mean prediction')
plt.fill_between(X_pred.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.2, label='1σ confidence')
plt.xlabel('Weeks since start')
plt.ylabel('BTC Price (USD)')
plt.title('Gaussian Process Regression on BTC Weekly Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gp_output.png")

print("Plot saved to gp_output.png")
