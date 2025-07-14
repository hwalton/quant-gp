# gp_predict.py

import numpy as np
import joblib
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

X_PRED_PKL = '../2-gp_fit/X_pred.npy'
Y_PRED_PKL = '../2-gp_fit/y_pred.npy'
YSTD_PKL = '../2-gp_fit/y_std.npy'
ORIGINAL_LOG_CSV = '../0-data/btc_monthly_prices.csv'

INITIAL_WEALTH = 1.0
NORMALISE_RETURNS = True
UTILITY_FUNCTION = 'tanh'
SIGMOID_K = 25.0
W0 = 0.98
PREDICT_INDEX_OFFSET = 18  # months
Y_LIMIT = (4, 18)

# ─── LOAD ───────────────────────────────────────────────────────────────────────

X_pred = np.load(X_PRED_PKL)
y_pred = np.load(Y_PRED_PKL)
y_std = np.load(YSTD_PKL)

import pandas as pd
df = pd.read_csv(ORIGINAL_LOG_CSV, sep=';')
df = df.sort_values(by='timestamp')
y = np.log(df['close'].astype(float).values)

# ─── UTILITY FUNCTION ───────────────────────────────────────────────────────────

def utility(w):
    if UTILITY_FUNCTION == 'log':
        return np.log(w) if w > 0 else -np.inf
    elif UTILITY_FUNCTION == 'sqrt':
        return np.sqrt(w) if w >= 0 else 0
    elif UTILITY_FUNCTION == 'sigmoid':
        return 1 / (1 + np.exp(-SIGMOID_K * (w - W0)))
    elif UTILITY_FUNCTION == 'tanh':
        return np.tanh(SIGMOID_K * (w - W0))
    else:
        raise ValueError(f"Unsupported utility function: {UTILITY_FUNCTION}")

def expected_utility(weight, mu, sigma):
    def integrand(r):
        wealth = INITIAL_WEALTH + weight * r
        return utility(wealth) * norm.pdf(r, loc=mu, scale=sigma)
    result, _ = quad(integrand, mu - 6 * sigma, mu + 6 * sigma, limit=100)
    return -result

# ─── COMPUTE RETURN AND VOLATILITY ──────────────────────────────────────────────

target_index = np.searchsorted(X_pred.ravel(), len(y) + PREDICT_INDEX_OFFSET)
price_now = y[-1]
price_pred = y_pred[target_index]
mu = (price_pred - price_now) / price_now if NORMALISE_RETURNS else price_pred - price_now
sigma = y_std[target_index] / price_now if NORMALISE_RETURNS else y_std[target_index]

print(f"\nExpected BTC return over {PREDICT_INDEX_OFFSET} months: {mu:.6f}")
print(f"Predicted standard deviation: {sigma:.6f}")

# ─── OPTIMISE ALLOCATION ────────────────────────────────────────────────────────

opt_result = minimize_scalar(expected_utility, bounds=(0, 1), args=(mu, sigma), method='bounded')
optimal_weight = opt_result.x
print(f"Optimal BTC allocation: {optimal_weight:.3f}")
print(f"Optimal cash allocation: {1 - optimal_weight:.3f}")

# ─── PLOTS ──────────────────────────────────────────────────────────────────────

# Expected utility vs allocation
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
print("Saved utility_curve.png")

# Wealth distribution
r_vals = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
pdf_vals = norm.pdf(r_vals, loc=mu, scale=sigma)
wealth_vals = INITIAL_WEALTH + optimal_weight * r_vals

plt.figure(figsize=(8, 4))
plt.plot(wealth_vals, pdf_vals, label='Wealth PDF')
plt.axvline(INITIAL_WEALTH, color='r', linestyle='--', label='Initial Wealth')
plt.xlabel('Simulated Future Wealth')
plt.ylabel('Probability Density')
plt.title('Wealth Distribution')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wealth_distribution.png")

# Utility function plot
if UTILITY_FUNCTION == 'log':
    wealth_vals = np.linspace(0.01, 3, 500)
elif UTILITY_FUNCTION == 'sqrt':
    wealth_vals = np.linspace(0, 3, 500)
elif UTILITY_FUNCTION in ['sigmoid', 'tanh']:
    wealth_vals = np.linspace(W0 - 1.0, W0 + 1.0, 500)
else:
    raise ValueError(f"Unsupported utility function: {UTILITY_FUNCTION}")

utility_vals = [utility(w) for w in wealth_vals]

plt.figure(figsize=(8, 4))
plt.plot(wealth_vals, utility_vals, label=f'{UTILITY_FUNCTION.capitalize()} utility', color='blue')

if UTILITY_FUNCTION in ['sigmoid', 'tanh']:
    plt.axvline(W0, color='grey', linestyle='--', label='Inflection point (w0)')

plt.xlabel('Wealth')
plt.ylabel('Utility')
plt.title('Utility Function')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("utility_func.png")
print("Saved utility_func.png")

