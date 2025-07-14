import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared, DotProduct
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import seaborn as sns
import time

# ─── CONTROL PANEL ──────────────────────────────────────────────────────────────

DATA_PATH = 'data/btc_monthly_prices.csv'
INITIAL_WEALTH = 1.0              # starting capital
CONFIDENCE_BAND = True            # plot GP std dev band
NORMALISE_RETURNS = True          # convert price difference to percentage return

UTILITY_FUNCTION = 'sigmoid'      # choices: 'log', 'sqrt', 'sigmoid'
SIGMOID_K = 25.0                   # steepness of sigmoid
SIGMOID_W0 = 0.98             # inflection point of sigmoid (target wealth)

Y_LIMIT = (4, 18)             # y-axis plot limits
MONTHS_INTO_FUTURE = 48

# ─── Kernel Hyperparameters
NOISE_LEVEL = 5e-1                # GP WhiteKernel noise
LENGTH_SCALE_SE = 50.0               # GP RBF kernel length scale

LENGTH_SCALE_SIN = 1.0
PERIOD_MONTHS = 48


# ─── LOAD DATA ──────────────────────────────────────────────────────────────────

# Read the semicolon-separated file
df = pd.read_csv(DATA_PATH, sep=';')

# Parse timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort just in case
df = df.sort_values(by='timestamp')

# Extract the close price and convert to float
y = np.log(df['close'].astype(float).values)

# Time index for GP
X = np.arange(len(y)).reshape(-1, 1)

print("First few rows of CSV data:")
print(df[['timestamp', 'close']].head())
print(f"\nX shape: {X.shape}, y shape: {y.shape}")
print(f"y min: {np.min(y):.2f}, y max: {np.max(y):.2f}")

print("First few rows of CSV data:")
print(df.head())
print(f"\nX shape: {X.shape}, y shape: {y.shape}")
print(f"y min: {np.min(y):.2f}, y max: {np.max(y):.2f}")

# ─── GP DEFINITION ──────────────────────────────────────────────────────────────

# Load fitted log trend
log_params = joblib.load("log_trend_params.pkl")

def log_trend(x):
    return log_params[0] * np.log(log_params[1] * x + 1) + log_params[2]

# Compute residuals from log trend
X_flat = X.ravel()
trend_vals = log_trend(X_flat)
residuals = y - trend_vals

# Define kernel without DotProduct
kernel = (
    C(1.0, constant_value_bounds="fixed") *
    (
        RBF(length_scale=LENGTH_SCALE_SE, length_scale_bounds="fixed") +
        ExpSineSquared(length_scale=LENGTH_SCALE_SIN, periodicity=PERIOD_MONTHS, length_scale_bounds="fixed", periodicity_bounds="fixed")
    ) +
    WhiteKernel(noise_level=NOISE_LEVEL, noise_level_bounds="fixed")
)

# GP on residuals
gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
gp.fit(X, residuals)

# Predict residuals
X_pred = np.linspace(0, len(X) + MONTHS_INTO_FUTURE, 700).reshape(-1, 1)
y_resid_pred, y_std = gp.predict(X_pred, return_std=True)

# Add trend back
start = time.time()
y_resid_pred, y_std = gp.predict(X_pred, return_std=True)
y_pred = y_resid_pred + log_trend(X_pred.ravel())
end = time.time()

print(f"\nGP fit + predict completed in {end - start:.3f} seconds")

# ─── UTILITY FUNCTION ───────────────────────────────────────────────────────────

def utility(w):
    if UTILITY_FUNCTION == 'log':
        return np.log(w) if w > 0 else -np.inf
    elif UTILITY_FUNCTION == 'sqrt':
        return np.sqrt(w) if w >= 0 else 0
    elif UTILITY_FUNCTION == 'sigmoid':
        return 1 / (1 + np.exp(-SIGMOID_K * (w - SIGMOID_W0)))
    else:
        raise ValueError("Unsupported utility function")

# ─── EXPECTED UTILITY FUNCTION ──────────────────────────────────────────────────

def expected_utility(weight, mu, sigma):
    def integrand(r):
        wealth = INITIAL_WEALTH + weight * r
        return utility(wealth) * norm.pdf(r, loc=mu, scale=sigma)
    result, _ = quad(integrand, mu - 6 * sigma, mu + 6 * sigma, limit=100)
    return -result  # for minimisation

# ─── TARGET PREDICTION POINT ────────────────────────────────────────────────────

MONTHS_INTO_FUTURE_index = X[-1][0] + MONTHS_INTO_FUTURE
target_index = np.searchsorted(X_pred.ravel(), MONTHS_INTO_FUTURE_index)
target_index = min(target_index, len(y_pred) - 1)

price_now = y[-1]
price_pred = y_pred[target_index]
mu = (price_pred - price_now) / price_now if NORMALISE_RETURNS else price_pred - price_now
sigma = y_std[target_index] / price_now if NORMALISE_RETURNS else y_std[target_index]

print(f"\nExpected BTC return over {MONTHS_INTO_FUTURE} months: {mu:.6f}")

print(f"Predicted standard deviation over {MONTHS_INTO_FUTURE} weeks: {sigma:.6f}")

# ─── OPTIMISE ALLOCATION ────────────────────────────────────────────────────────

opt_result = minimize_scalar(expected_utility, bounds=(0, 1), args=(mu, sigma), method='bounded')
optimal_weight = opt_result.x

print(f"Optimal BTC allocation based on expected utility: {optimal_weight:.3f}")
print(f"Optimal Cash allocation: {1 - optimal_weight:.3f}")

# ─── PLOT GP OUTPUT ─────────────────────────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.plot(X, y, 'kx', label='Observed BTC prices')
plt.plot(X_pred, y_pred, 'b-', label='GP mean prediction')
if CONFIDENCE_BAND:
    plt.fill_between(X_pred.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.2, label='1σ confidence')
plt.xlabel('Weeks since start')
plt.ylabel('BTC Price (USD)')
plt.title('Gaussian Process Regression on BTC Weekly Prices')
plt.legend()
plt.grid(True)
plt.ylim(*Y_LIMIT)
plt.tight_layout()
plt.savefig("gp_output.png")
print("Plot saved to gp_output.png")

# ─── PLOT UTILITY CURVE ─────────────────────────────────────────────────────────

weights = np.linspace(0, 1, 100)
utilities = [-expected_utility(w, mu, sigma) for w in weights]

plt.figure(figsize=(8, 4))
plt.plot(weights, utilities, label='Expected Utility')
plt.axvline(optimal_weight, color='r', linestyle='--', label='Optimal weight')
plt.xlabel('BTC Allocation')
plt.ylabel('Expected Utility')
plt.title('Expected Utility vs BTC Allocation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("utility_curve.png")
print("Utility curve saved to utility_curve.png")

# ─── PLOT WEALTH DISTRIBUTION ───────────────────────────────────────────────────

r_samples = np.random.normal(mu, sigma, 1000)
wealth_samples = INITIAL_WEALTH + optimal_weight * r_samples

plt.figure(figsize=(8, 4))
sns.histplot(wealth_samples, bins=50, kde=True)
plt.axvline(INITIAL_WEALTH, color='r', linestyle='--', label='Initial Wealth')
plt.xlabel('Simulated Future Wealth')
plt.title('Wealth Distribution (Optimal BTC Allocation)')
plt.legend()
plt.tight_layout()
plt.savefig("wealth_distribution.png")
print("Wealth distribution saved to wealth_distribution.png")

# ─── PLOT SIGMOID FUNCTION ──────────────────────────────────────────────────────

wealth_vals = np.linspace(SIGMOID_W0 - 0.5, SIGMOID_W0 + 0.5, 500)
sigmoid_vals = [utility(w) for w in wealth_vals]

plt.figure(figsize=(8, 4))
plt.plot(wealth_vals, sigmoid_vals, label=f'Sigmoid (k={SIGMOID_K}, w0={SIGMOID_W0})', color='blue')
plt.axvline(SIGMOID_W0, color='grey', linestyle='--', label='Inflection point (w0)')
plt.xlabel('Wealth')
plt.ylabel('Utility')
plt.title('Sigmoid Utility Function')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sigmoid_utility.png")
print("Saved plot to sigmoid_utility.png")
